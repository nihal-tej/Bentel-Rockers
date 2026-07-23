import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import ast
import pandas as pd
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

INPUT_CSV_PATH = "/home/shreya/Nihal/data/D10001.csv"
OUTPUT_CSV_PATH = "/home/shreya/Nihal/data/SPIN_D10001.csv"
MODEL_PATH = "/home/shreya/Nihal/outputs/D10001_nih_7/final_adapter"  



MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
BATCH_SIZE = 32  

print(f"Loading SFT model from: {MODEL_PATH}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    trust_remote_code=True,
)
FastLanguageModel.for_inference(model)

text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token
text_tokenizer.padding_side = "left" 
print(f"Reading benchmark dataset: {INPUT_CSV_PATH}")
df = pd.read_csv(INPUT_CSV_PATH)

raw_prompts_pool = []
gold_chosen_pool = []

print("\nExtracting conversation history tracks...")
for idx, row in df.iterrows():
    if pd.isna(row['chat']):
        continue
    try:
        chat_history = ast.literal_eval(row['chat'])
    except:
        continue

    current_turn_history = []
    for message in chat_history:
        role = message['role']
        content = message['content']
        
        if role == 'user':
            current_turn_history.append({"role": "user", "content": content})
        elif role == 'assistant':
            # Run the chat template on the history preceding this turn
            formatted_prompt = text_tokenizer.apply_chat_template(
                current_turn_history,
                tokenize=False,
                add_generation_prompt=True
            )
            raw_prompts_pool.append(formatted_prompt)
            
            gold_chosen = content.strip()
            if not gold_chosen.endswith(text_tokenizer.eos_token):
                gold_chosen += text_tokenizer.eos_token
            gold_chosen_pool.append(gold_chosen)
            
            # Keep history context grounded in gold track
            current_turn_history.append({"role": "assistant", "content": content})

total_samples = len(raw_prompts_pool)
print(f"Total parsed training turns ready for generation: {total_samples}")

spin_records = []

print(f"\nRunning batch inference (Batch Size: {BATCH_SIZE})...")
for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Processing Batches"):
    batch_prompts = raw_prompts_pool[i : i + BATCH_SIZE]
    batch_chosen = gold_chosen_pool[i : i + BATCH_SIZE]
    
   
    inputs = text_tokenizer(
        batch_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=MAX_SEQ_LENGTH
    ).to("cuda")
    
    input_lengths = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            eos_token_id=text_tokenizer.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
        )
    
  
    generated_tokens = outputs[:, input_lengths:]
    batch_rejected = text_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    for prompt, chosen, rejected in zip(batch_prompts, batch_chosen, batch_rejected):
        rejected_str = rejected.strip()
        if not rejected_str.endswith(text_tokenizer.eos_token):
            rejected_str += text_tokenizer.eos_token
            
        spin_records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected_str
        })

spin_df = pd.DataFrame(spin_records)
spin_df.to_csv(OUTPUT_CSV_PATH, index=False)

print("\n" + "="*50)
print(f"SPIN Dataset Generation Complete! Saved to: {OUTPUT_CSV_PATH}")
print("="*50)