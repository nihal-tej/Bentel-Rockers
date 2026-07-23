
import os

os.environ["UNSLOTH_DISABLE_FAST_TOKENIZER_FIX"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import yaml
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from peft import PeftModel


with open("config_warmup.yaml", "r") as f:
    config = yaml.safe_load(f)

print("=" * 80)
print("CONFIG")
print("=" * 80)
print(config)

model_name = config["model"]["name"]
max_seq_length = config["training"].get("max_seq_length", 4096)
text_column = config["dataset"].get("text_column", "text")
output_directory = config["output"].get("output_dir", "./outputs")
do_eval = config["training"].get("do_eval", True)


print("\nLoading model...\n")

merge_adapter = config["checkpoint"].get(
    "merge_adapter_checkpoint",
    None,
)

need_merge = (
    merge_adapter is not None
    and os.path.exists(merge_adapter)
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16 if config["optimization"].get("bf16", True) else None,
    load_in_4bit=False if need_merge else config["quantization"].get("load_in_4bit", True),
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id

print("PAD:", tokenizer.pad_token)
print("PAD ID:", tokenizer.pad_token_id)

print("EOS:", tokenizer.eos_token)
print("EOS ID:", tokenizer.eos_token_id)
if need_merge:
    peft_model = PeftModel.from_pretrained(
        model,
        merge_adapter,
    )

    model = peft_model.merge_and_unload()

    print("Adapter merged into base model.")

model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=config["lora"].get("dropout", 0.0),
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    use_gradient_checkpointing=(
        "unsloth"
        if config["optimization"].get(
            "gradient_checkpointing",
            True,
        )
        else False
    ),
    random_state=3407,
)

print("\nLoRA adapters integrated.")
model.print_trainable_parameters()

print(repr(tokenizer.eos_token))
print(tokenizer.special_tokens_map)

print("\nLoading CSV datasets...\n")

data_files = {
    "train": config["dataset"]["train_path"]
}

if do_eval:
    data_files["validation"] = config["dataset"]["val_path"]

dataset = load_dataset(
    "csv",
    data_files=data_files,
)

train_dataset = dataset["train"]

if do_eval:
    eval_dataset = dataset["validation"]
else:
    eval_dataset = None


clean_filter = lambda x: (
    x[text_column] is not None 
    and isinstance(x[text_column], str) 
    and len(x[text_column].strip()) > 0
)

train_dataset = dataset["train"].filter(clean_filter)

if eval_dataset is not None:
    eval_dataset = eval_dataset.filter(clean_filter)

def append_eos_marker(example):
    if not example[text_column].endswith(tokenizer.eos_token):
        example[text_column] = example[text_column] + tokenizer.eos_token
    return example

train_dataset = train_dataset.map(append_eos_marker)

if eval_dataset is not None:
    eval_dataset = eval_dataset.map(append_eos_marker)

print(f"Verified packed train samples: {len(train_dataset)}")

if eval_dataset is not None:
    print(f"Verified packed validation samples: {len(eval_dataset)}")


warmup_kwargs = {}
if "warmup_steps" in config["training"]:
    warmup_kwargs["warmup_steps"] = config["training"]["warmup_steps"]
elif "warmup_ratio" in config["training"]:
    warmup_kwargs["warmup_ratio"] = config["training"]["warmup_ratio"]

training_args = SFTConfig(
    output_dir=output_directory,
    num_train_epochs=config["training"]["epochs"],
    learning_rate=float(config["training"]["learning_rate"]),
    per_device_train_batch_size=config["training"]["batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    bf16=config["optimization"].get("bf16", True),
    logging_steps=config["training"]["logging_steps"],
    eval_strategy="steps" if do_eval else "no",
    eval_steps=config["training"].get("eval_steps", 100) if do_eval else None, 
    save_strategy=config["checkpoint"]["save_strategy"],
    save_steps=config["checkpoint"]["save_steps"],
    save_total_limit=config["checkpoint"]["save_total_limit"],
    load_best_model_at_end=(
    do_eval and
    config["checkpoint"].get(
        "load_best_model_at_end",
        True
    )
),metric_for_best_model="eval_loss" if do_eval else None,
greater_is_better=False if do_eval else None,lr_scheduler_type=config["scheduler"].get("type", "linear"),
    weight_decay=config["regularization"].get("weight_decay", 0.01),
    report_to="tensorboard",
    max_length=max_seq_length,
    dataset_text_field=text_column,
    packing=config["optimization"].get("packing", True), 
    **warmup_kwargs,
)



# trainer = SFTTrainer(
#     model=model,
#     processing_class=tokenizer, 
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     args=training_args,
# )



print("=" * 80)
print("TOKENIZER / MODEL CHECK")
print("=" * 80)

print("Model:", model_name)

print("Tokenizer length:", len(tokenizer))
print("Tokenizer vocab_size:", tokenizer.vocab_size)

print("Model vocab_size:", model.config.vocab_size)

lm_head = model.get_output_embeddings()
print("LM Head shape:", lm_head.weight.shape)

max_token = -1

for i in range(min(1000, len(train_dataset))):
    ids = tokenizer(train_dataset[i][text_column])["input_ids"]
    max_token = max(max_token, max(ids))

print("Maximum token seen:", max_token)
assert max_token < model.config.vocab_size, (
    f"Dataset token {max_token} exceeds "
    f"model vocab {model.config.vocab_size}"
)
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if do_eval else None,
    args=training_args,
)
print("\nTrainer successfully initialized with sample packing.")

resume_checkpoint = config["checkpoint"].get("resume_from_checkpoint", None)

print("\nCommencing training sequence...\n")
if resume_checkpoint and os.path.exists(resume_checkpoint):
    trainer.train(resume_from_checkpoint=resume_checkpoint)
else:
    trainer.train()

final_path = os.path.join(output_directory, "final_adapter")

model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
trainer.save_state()
merged_path = os.path.join(
    output_directory,
    "warmup_merged"
)

try:
    model.save_pretrained_merged(
        merged_path,
        tokenizer,
        save_method="merged_16bit",
    )

    print(
        f"Merged model saved to: {merged_path}"
    )

except Exception as e:
    print(
        f"Merge export skipped: {e}"
    )
print(f"\nTraining execution complete. Packed adapter saved to:\n{final_path}\n")
