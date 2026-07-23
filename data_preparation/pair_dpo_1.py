import json
import pandas as pd
from collections import defaultdict

def create_dpo_pairs(input_filepath, output_filepath, verbose=True):
    """
    Parses a JSONL file, groups by game and episode, and pairs
    EVERY 'Success' response with EVERY 'Lose'/'Aborted' response.
    """
    print(f"Reading data from: {input_filepath}\n")
    print("-" * 50)
    
    # Dictionary to group attempts by a unique episode ID
    grouped_episodes = defaultdict(lambda: {"prompt": None, "success": [], "fail": []})
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # --- PRINTING INPUT ROWS ---
                # if verbose:
                #     print(f"[Row {line_num} Input] Game: {data.get('game')} | Episode: {data.get('episode')} | Success: {data.get('Success')} | Aborted: {data.get('Aborted')} | Lose: {data.get('Lose')}")
                
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
    print("FINISHED READING. NOW PAIRING DATA (ALL COMBINATIONS)...")
    print("=" * 50 + "\n")
    
    dpo_records = []
    
    for unique_id, episode_data in grouped_episodes.items():
        prompt = episode_data["prompt"]
        success_responses = episode_data["success"]
        fail_responses = episode_data["fail"]
        
        # If we have at least one success and at least one failure, map ALL of them
        if prompt and success_responses and fail_responses:
            for chosen in success_responses:
                # i=0
                for rejected in fail_responses:
                    
                    # --- PRINTING MATCHED PAIRS ---
                    # if verbose:
                    #     print(f"[MATCH FOUND] ID: {unique_id}")
                    #     print(f"  -> CHOSEN: {chosen[:60]}...")
                    #     print(f"  -> REJECTED: {rejected[:60]}...\n")
                    # if i==2:
                    #     break
                    dpo_records.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected
                    })
                    # i+=1

    df = pd.DataFrame(dpo_records)
    
    if len(df) > 0:
        df.to_csv(output_filepath, index=False, encoding='utf-8')
        print(f"Successfully generated {len(df)} combinatorial DPO pairs!")
        print(f"Saved DPO dataset to: {output_filepath}")
    else:
        print("Warning: No matching success/fail pairs were found for the same episodes.")

if __name__ == "__main__":
    INPUT_FILE = r"C:\nihal\rough\playpen-paper-2025\SFT\Data\final_data\merge_old_new.jsonl"
    OUTPUT_FILE = r"C:\nihal\rough\playpen-paper-2025\SFT\Data\final_data\DPO_dataset_2.csv"
    
    create_dpo_pairs(INPUT_FILE, OUTPUT_FILE, verbose=True)