import json
import ast
import os
from datasets import Dataset, Features, Value

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
    processed_records = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            cleaned_line = line.strip()
            if not cleaned_line:
                continue
                
            try:
                row = json.loads(cleaned_line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                continue
            
            chat_history = clean_chat_data(row.get("chat"))
            if not chat_history or len(chat_history) < 2:
                continue
                
            last_assistant_idx = None
            for i in range(len(chat_history) - 1, -1, -1):
                if chat_history[i].get("role") == "assistant":
                    last_assistant_idx = i
                    break
                    
            if last_assistant_idx is None:
                continue
                
            grpo_prompt = chat_history[:last_assistant_idx]
            expected_output = chat_history[last_assistant_idx].get("content", "")
            
            formatted_prompt = [
                {
                    "role": str(msg.get("role", "")).strip(),
                    "content": str(msg.get("content", ""))
                } 
                for msg in grpo_prompt if msg.get("role") and msg.get("content")
            ]
            
            if not formatted_prompt:
                continue

            if formatted_prompt[0]["role"] != "system":
                formatted_messages = [{
                    "role": "system",
                    "content": "You are playing a grid manipulation game. Format your internal thoughts inside <think>...</think> tags, and your final step inside <answer>...</answer> tags."
                }] + formatted_prompt
            else:
                formatted_messages = formatted_prompt
                
            def stringify_field(val):
                if val is None: return ""
                return json.dumps(val) if isinstance(val, (dict, list)) else str(val)

            
            processed_records.append({
                "prompt": formatted_messages, 
                "ground_truth": stringify_field(expected_output),
                "target_grid": stringify_field(row.get("target", "")),
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
            
    return Dataset.from_list(processed_records, features=grpo_features)

if __name__ == "__main__":
    input_file = r"C:\nihal\rough\playpen-paper-2025\SFT\Data\final_data\merge_old_new.jsonl" 
    output_file = r"C:\nihal\rough\playpen-paper-2025\SFT\Data\final_data\grpo.jsonl"
    
    if os.path.exists(input_file):
        grpo_dataset = process_jsonl_for_grpo(input_file)
        print(f"Successfully converted dataset! Total rows for GRPO training: {len(grpo_dataset)}")
        

        grpo_dataset.to_json(output_file, orient="records", lines=True)

        first_row = grpo_dataset[0]
        print("\n--- PROMPT STRUCTURE GIVEN TO MODEL ---")
        print(json.dumps(first_row["prompt"], indent=2))
        print("\n--- GROUND TRUTH VERIFICATION KEY ---")
        print(first_row["ground_truth"])
    else:
        print(f" Input file not found at: {input_file}")