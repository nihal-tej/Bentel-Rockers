import os
os.environ["HF_HUB_DISABLE_XET"] = "1" 
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import re
import json
import os
import re
import json
import ast
import yaml
import torch
from datasets import Dataset, Features, Value
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# dataset = Dataset.from_list(processed_records, features=grpo_features)
# return dataset

import datasets
datasets.BuilderConfig.use_cache = False  

import pyarrow as pa
from datasets import Dataset



def clean_chat_data(chat_val):
    if chat_val is None or chat_val == "" or chat_val == []:
        return None
    if isinstance(chat_val, list):
        return chat_val
    if isinstance(chat_val, str):
        chat_val = chat_val.strip()
        try:
            return json.loads(chat_val)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(chat_val)
        except (SyntaxError, ValueError):
            return None
    return None

def process_jsonl_for_grpo(file_path: str) -> Dataset:
    """
    Parses conversational history blocks from the 'prompt' key, extracts the 
    user message, and structures it safely for the GRPO trainer.
    """
    processed_records = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f" Target training file not found at: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            cleaned_line = line.strip()
            if not cleaned_line:
                continue
                
            try:
                row = json.loads(cleaned_line)
            except json.JSONDecodeError as e:
                print(f" Warning: Skipping malformed JSON on line {line_num}: {e}")
                continue
            
           
            prompt_history = row.get("prompt")
            if not prompt_history or not isinstance(prompt_history, list):
                continue
                
          
            expected_output = row.get("ground_truth", "")
            
            def stringify_field(val):
                if val is None: return ""
                return json.dumps(val) if isinstance(val, (dict, list)) else str(val)

        
            processed_records.append({
                "prompt": prompt_history, 
                "ground_truth": stringify_field(expected_output),
                "target_grid": stringify_field(row.get("target_grid", "")),
                "game_id": stringify_field(row.get("game_id", "")),
                "episode": stringify_field(row.get("episode", ""))
            })
            
  
    grpo_features = Features({
        "prompt": [
            {
                "role": Value("string"),
                "content": Value("string")
            }
        ],
        "ground_truth": Value("string"),
        "target_grid": Value("string"),
        "game_id": Value("string"),
        "episode": Value("string")
    })
            
   
    dict_data = {key: [record[key] for record in processed_records] for key in grpo_features.keys()}
    dataset = Dataset.from_dict(dict_data, features=grpo_features)
    
    return dataset



def format_reward_func(completions, **kwargs) -> list[float]:
    """Rewards the model for outputting valid <think>...</think> and <answer>...</answer> blocks."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    return [1.0 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

def correctness_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    """Extracts text out of the generated <answer> tags and matches it against ground_truth."""
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for response, target in zip(responses, ground_truth):
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            extracted_action = match.group(1).strip()
            if extracted_action == target.strip():
                rewards.append(2.0)  # Correct answer match
            else:
                rewards.append(0.0)  # Correct format structural tags, wrong calculation
        else:
            rewards.append(0.0)      # Failed to provide completion answers wrappers
            
    return rewards



def main():
    # Load parameters from configuration mapping file
    with open("grpo_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    print(" Parsing dataset targets from storage paths...")
    dataset = process_jsonl_for_grpo(cfg["dataset"]["train_path"])
    print(len(dataset))
    print(f"ngestion complete. Dataset rows ready for training: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError(" Execution halted: Zero valid records remain after dataset evaluation.")

    print(" Downloading base architecture models via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["training"]["max_seq_length"],
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        fast_inference=False
        ,device_map={"": torch.cuda.current_device()}
    )

    # Wrap base weights inside LoRA fine-tuning parameters adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=cfg["optimization"]["use_gradient_checkpointing"]
    )

    print(" Assembling training configurations...")
    training_args = GRPOConfig(
        output_dir=cfg["output"]["output_dir"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        max_steps=cfg["training"]["max_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        save_strategy=cfg["checkpoint"]["save_strategy"],
        save_steps=cfg["checkpoint"]["save_steps"],
        save_total_limit=cfg["checkpoint"]["save_total_limit"],
        lr_scheduler_type=cfg["scheduler"]["type"],
        optim=cfg["optimization"]["optim"],
        weight_decay=cfg["regularization"]["weight_decay"],
        bf16=cfg["optimization"]["bf16"],
        gradient_checkpointing=cfg["optimization"]["gradient_checkpointing"],
        
 
        num_generations=cfg["grpo"]["num_generations"],
        
        temperature=cfg["grpo"]["temperature"],
        report_to="none",
        
     
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False 
    )


    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, correctness_reward_func], # Ensure your reward functions are here
        args=training_args,
        train_dataset=dataset,
        max_prompt_length=cfg["training"]["max_prompt_length"],
        max_completion_length=cfg["training"]["max_completion_length"]
    )


    compiled_class = trainer.__class__
    unsloth_compiled_get_logps = compiled_class._get_per_token_logps_and_entropies

    def safe_unpack_get_logps(*args, **kwargs):
        res = unsloth_compiled_get_logps(*args, **kwargs)
        # If the underlying compiled function returns 2 elements, pad it to 3
        if isinstance(res, tuple) and len(res) == 2:
            return (res[0], res[1], None)
        return res

    compiled_class._get_per_token_logps_and_entropies = safe_unpack_get_logps


    print("Starting GRPO Training Loop...")
    trainer.train()
    
    print(" Merging and saving final adapter outputs...")
    model.save_pretrained_merged(cfg["output"]["output_dir"], tokenizer, save_method="lora")
    print(" GRPO Execution Run Complete!")

if __name__ == "__main__":
    main()