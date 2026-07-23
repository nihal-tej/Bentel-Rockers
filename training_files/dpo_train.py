import os
import sys


import transformers.models.auto.modeling_auto as modeling_auto
if not hasattr(modeling_auto, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"):
    modeling_auto.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {}


os.environ["UNSLOTH_DISABLE_FAST_TOKENIZER_FIX"] = "1"

import trl
from trl import DPOConfig
OriginalDPOTrainer = trl.DPOTrainer

from unsloth import FastLanguageModel

trl.DPOTrainer = OriginalDPOTrainer
DPOTrainer = OriginalDPOTrainer 

import yaml
import torch
from datasets import load_dataset


with open("dpo_config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("=" * 80)
print("DPO CONFIG (EVAL DISABLED)")
print("=" * 80)

model_name = config["model"]["name"]
max_seq_length = config["training"].get("max_seq_length", 2048)
output_directory = config["output"].get("output_dir", "./outputs_dpo")

p_col = config["dataset"].get("prompt_column", "prompt")
c_col = config["dataset"].get("chosen_column", "chosen")
r_col = config["dataset"].get("rejected_column", "rejected")


print("\nLoading model...\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16 if config["optimization"].get("bf16", True) else None,
    load_in_4bit=config["quantization"].get("load_in_4bit", True),
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

print("PAD:", tokenizer.pad_token)
print("EOS:", tokenizer.eos_token)

model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=config["lora"].get("dropout", 0.0),
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing=(
        "unsloth" if config["optimization"].get("gradient_checkpointing", True) else False
    ),
    random_state=3407,
)

print("\nLoRA adapters integrated.")
model.print_trainable_parameters()


print("\nLoading DPO CSV dataset...\n")


dataset = load_dataset("csv", data_files={"train": config["dataset"]["train_path"]})
train_dataset = dataset["train"]


clean_filter = lambda x: (
    x[p_col] is not None and len(str(x[p_col]).strip()) > 0 and
    x[c_col] is not None and len(str(x[c_col]).strip()) > 0 and
    x[r_col] is not None and len(str(x[r_col]).strip()) > 0
)

train_dataset = train_dataset.filter(clean_filter)


def format_dpo_example(example):
    prompt = str(example[p_col])
    chosen = str(example[c_col])
    rejected = str(example[r_col])

    if not chosen.endswith(tokenizer.eos_token):
        chosen += tokenizer.eos_token
    if not rejected.endswith(tokenizer.eos_token):
        rejected += tokenizer.eos_token

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

train_dataset = train_dataset.map(format_dpo_example)

print(f"Verified DPO train samples: {len(train_dataset)}")


warmup_kwargs = {}
if "warmup_steps" in config["training"]:
    warmup_kwargs["warmup_steps"] = config["training"]["warmup_steps"]
elif "warmup_ratio" in config["training"]:
    warmup_kwargs["warmup_ratio"] = config["training"]["warmup_ratio"]

training_args = DPOConfig(
    output_dir=output_directory,
    num_train_epochs=config["training"]["epochs"],
    learning_rate=float(config["training"]["learning_rate"]),
    per_device_train_batch_size=config["training"]["batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    bf16=config["optimization"].get("bf16", True),
    logging_steps=config["training"]["logging_steps"],
    eval_strategy="no", # 👈 Evaluation explicitly disabled
    save_strategy=config["checkpoint"]["save_strategy"],
    save_steps=config["checkpoint"]["save_steps"],
    save_total_limit=config["checkpoint"]["save_total_limit"],
    lr_scheduler_type=config["scheduler"].get("type", "cosine"),
    weight_decay=config["regularization"].get("weight_decay", 0.01),
    report_to="tensorboard",
    max_length=max_seq_length,          
    # max_prompt_length=max_seq_length // 2, 
    beta=config["dpo"].get("beta", 0.1),   
    **warmup_kwargs,
)


tokenizer.tokenizer = tokenizer
trainer = DPOTrainer(
    model=model,
    ref_model=None, 
    tokenizer=tokenizer, 
    train_dataset=train_dataset,
    args=training_args,
)

print("\nDPOTrainer successfully initialized.")

resume_checkpoint = config["checkpoint"].get("resume_from_checkpoint", None)

print("\nCommencing DPO training sequence...\n")
if resume_checkpoint and os.path.exists(resume_checkpoint):
    trainer.train(resume_from_checkpoint=resume_checkpoint)
else:
    trainer.train()


final_path = os.path.join(output_directory, "final_dpo_adapter")

model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
trainer.save_state()

merged_path = os.path.join(output_directory, "dpo_merged")

try:
    model.save_pretrained_merged(
        merged_path,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged DPO model saved to: {merged_path}")
except Exception as e:
    print(f"Merge export skipped: {e}")

print(f"\nDPO Training execution complete. LoRA adapter saved to:\n{final_path}\n")