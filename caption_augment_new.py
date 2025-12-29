import os
import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
INPUT_FILE = "./data/train_ready.json"
OUTPUT_FILE = "./data/train_augmented_stage2.json"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f"Loading model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", trust_remote_code=True, load_in_4bit=True
        )
        print("Model loaded in 4-bit mode.")
    except:
        print("Falling back to float16.")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
        )
    return model, tokenizer

def clean_text(text):
    """Remove numbering and common generation artifacts from text."""
    # Remove leading numbering
    text = re.sub(r'^\d+[\.\)]\s*', '', text)
    # Remove common prefixes
    text = re.sub(r'^(Here is|Sure|Rewritten|Summary|Output).*?:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

def run_generation(model, tokenizer, prompt, original_text, count, max_tokens):
    """Generate multiple text variations using the model."""
    user_prompt = f"Original content:\n{original_text}\n\nOutput:"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_return_sequences=count,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        results = []
        input_len = inputs.input_ids.shape[1]
        for gen_id in generated_ids:
            output_ids = gen_id[input_len:]
            response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            cleaned = clean_text(response)
            if len(cleaned) > 10:
                results.append(cleaned)
        return results
    except Exception as e:
        print(f"Generation Error: {e}")
        return []

def main():
    model, tokenizer = load_model()
    
    print(f"Reading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    augmented_data = []
    first_sample_displayed = False
    
    # System prompts for different augmentation strategies
    PROMPT_DETAIL = (
        "You are an AI assistant. Rewrite the driving video description. "
        "Keep all visual facts (objects, colors, actions) exactly the same. "
        "Maintain the detailed structure and information."
        "Vary sentence structure. Do NOT use numbering."
    )
    
    PROMPT_SUMMARY = (
        "You are an AI assistant. Summarize the driving video description into a SINGLE CONCISE paragraph (30-60 words). "
        "Focus ONLY on the main traffic events, ego-vehicle action, and weather. "
        "Be direct and brief. Do NOT use numbering."
    )

    print("Starting augmentation (Hybrid Strategy: Detail + Summary)...")
    
    for item in tqdm(data, desc="Processing"):
        original_caption = item['a']
        
        # Generate detailed captions (original + 4 rewrites)
        detail_gen = run_generation(model, tokenizer, PROMPT_DETAIL, original_caption, count=4, max_tokens=512)
        captions_detail = [original_caption] + detail_gen
        
        # Generate concise summaries
        summary_gen = run_generation(model, tokenizer, PROMPT_SUMMARY, original_caption, count=5, max_tokens=150)
        captions_summary = summary_gen
        if not captions_summary:
            captions_summary = [original_caption[:200]]

        # Display first sample for verification
        if not first_sample_displayed:
            print("\n" + "="*30 + " Sample Output " + "="*30)
            print(f"Video: {item['video']}")
            print(f"[Type 1: Detail] (Count: {len(captions_detail)})")
            orig_chars = len(captions_detail[0])
            orig_tokens = len(tokenizer(captions_detail[0]).input_ids)
            print(f"  [Original] (chars={orig_chars}, tokens={orig_tokens}):\n{captions_detail[0]}")
            if len(captions_detail) > 1:
                rw1_chars = len(captions_detail[1])
                rw1_tokens = len(tokenizer(captions_detail[1]).input_ids)
                print(f"  [Rewrite 1] (chars={rw1_chars}, tokens={rw1_tokens}):\n{captions_detail[1]}")

            print(f"\n[Type 2: Summary] (Count: {len(captions_summary)})")
            if len(captions_summary) > 0:
                sum1_chars = len(captions_summary[0])
                sum1_tokens = len(tokenizer(captions_summary[0]).input_ids)
                print(f"  [Sum 1] (chars={sum1_chars}, tokens={sum1_tokens}):\n{captions_summary[0]}")
            if len(captions_summary) > 1:
                sum2_chars = len(captions_summary[1])
                sum2_tokens = len(tokenizer(captions_summary[1]).input_ids)
                print(f"  [Sum 2] (chars={sum2_chars}, tokens={sum2_tokens}):\n{captions_summary[1]}")
            print("="*80 + "\n")
            first_sample_displayed = True

        # Store both detailed and summary versions
        augmented_data.append({
            "video": item['video'],
            "q": "Describe the video in detail.",
            "captions": captions_detail
        })
        
        augmented_data.append({
            "video": item['video'],
            "q": "Summarize the driving scene concisely.",
            "captions": captions_summary
        })
        
        # Periodic checkpoint
        if len(augmented_data) % 100 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(augmented_data, f, indent=2)

    # Final save
    print(f"Saving {len(augmented_data)} entries to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()