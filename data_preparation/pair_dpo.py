import json
import pandas as pd
from collections import defaultdict

def create_dpo_pairs(input_filepath, output_filepath, verbose=True):
    """
    Parses a JSONL file, groups by game and episode, and pairs
    'Success' responses (chosen) with 'Lose'/'Aborted' responses (rejected).
    """
    print(f"Reading data from: {input_filepath}\n")
    print("-" * 50)
    
    grouped_episodes = defaultdict(lambda: {"prompt": None, "success": [], "fail": []})
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # --- PRINTING INPUT ROWS ---
                if verbose:
                    print(f"[Row {line_num} Input] Game: {data.get('game')} | Episode: {data.get('episode')} | Success: {data.get('Success')} | Aborted: {data.get('Aborted')} | Lose: {data.get('Lose')}")
                
                game = data.get("game", "unknown_game")
                experiment = data.get("experiment", "unknown_exp")
                episode = data.get("episode", "unknown_ep")
                unique_id = f"{game}_{experiment}_{episode}"
                
                chat = data.get("chat", [])
                
                if len(chat) >= 2 and chat[0]["role"] == "user" and chat[1]["role"] == "assistant":
                    prompt = chat[0]["content"]
                    response = chat[1]["content"]
                    
                    grouped_episodes[unique_id]["prompt"] = prompt
                    
                    if data.get("Success", 0) == 1:
                        grouped_episodes[unique_id]["success"].append(response)
                    elif data.get("Lose", 0) == 1 or data.get("Aborted", 0) == 1:
                        grouped_episodes[unique_id]["fail"].append(response)
                        
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON on line {line_num}")

    print("\n" + "=" * 50)
    print("FINISHED READING. NOW PAIRING DATA...")
    print("=" * 50 + "\n")
    
    dpo_records = []
    
    for unique_id, episode_data in grouped_episodes.items():
        prompt = episode_data["prompt"]
        success_responses = episode_data["success"]
        fail_responses = episode_data["fail"]
        
        if prompt and success_responses and fail_responses:
            chosen = success_responses[0]
            rejected = fail_responses[0]
            
            # --- PRINTING MATCHED PAIRS ---
            if verbose:
                print(f"[MATCH FOUND] ID: {unique_id}")
                print(f"  -> CHOSEN: {chosen[:60]}...") # Truncated for readability
                print(f"  -> REJECTED: {rejected[:60]}...\n")
            
            dpo_records.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })

    df = pd.DataFrame(dpo_records)
    
    if len(df) > 0:
        df.to_csv(output_filepath, index=False, encoding='utf-8')
        print(f"Successfully generated {len(df)} DPO pairs!")
        print(f"Saved DPO dataset to: {output_filepath}")
    else:
        print("Warning: No matching success/fail pairs were found for the same episodes.")

if __name__ == "__main__":
    INPUT_FILE = "merge_old_new.jsonl"
    OUTPUT_FILE = "DPO_dataset.csv"
    
    # Set verbose=True to see the input rows being processed
    create_dpo_pairs(INPUT_FILE, OUTPUT_FILE, verbose=True)