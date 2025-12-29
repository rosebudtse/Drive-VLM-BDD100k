import torch
from transformers import AutoTokenizer, AutoProcessor
from dataset import VideoInstructDataset, collate_fn

# Configuration
MODEL_LLM_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_VISION_ID = "google/siglip2-so400m-patch14-384"
DATA_JSON = "./data/train_augmented_stage2.json"
VIDEO_DIR = "./data/zip_folder/bdd100k"

def debug_dataset():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LLM_ID, trust_remote_code=True)
    # Set pad_token to prevent decoding errors
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    image_processor = AutoProcessor.from_pretrained(MODEL_VISION_ID, trust_remote_code=True)

    dataset = VideoInstructDataset(
        meta_file=DATA_JSON,
        video_dir=VIDEO_DIR,
        tokenizer=tokenizer,
        image_processor=image_processor,
        stage='2'  # Test Stage 2 masking logic
    )

    # Get first sample
    sample = dataset[0]
    
    input_ids = sample['input_ids']
    labels = sample['labels']
    attention_mask = sample['attention_mask']

    print("\n" + "="*50)
    print("DATASET TENSOR CHECK")
    print("="*50)

    # Check chat template closing tag
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"Full Tokenized Text:\n{full_text}\n")
    
    if "<|im_end|>" in full_text:
        print("[OK] <|im_end|> found in input_ids.")
    else:
        print("[ERROR] <|im_end|> MISSING in input_ids!")

    # Verify label masking for loss computation
    loss_tokens = input_ids[labels != -100]
    loss_text = tokenizer.decode(loss_tokens, skip_special_tokens=False)
    
    print("-" * 30)
    print(f"Tokens marked for LOSS (Labels != -100):\n{loss_text}")
    print("-" * 30)

    # Validate masking correctness
    if "user" in loss_text.lower():
        print("[ERROR] User prompt is NOT masked! (Found 'user' in loss tokens)")
    elif len(loss_tokens) == 0:
        print("[ERROR] Labels are ALL masked! Model learns nothing.")
    else:
        print("[OK] Only Assistant answer is being learned.")

    # Check padding masking
    pad_id = tokenizer.pad_token_id
    num_pads = (input_ids == pad_id).sum().item()
    num_masked_pads = (labels[input_ids == pad_id] == -100).sum().item()
    print(f"\nPadding Check:")
    print(f"   - Total Padding Tokens: {num_pads}")
    print(f"   - Properly Masked Paddings: {num_masked_pads}")
    
    if num_pads == num_masked_pads:
        print("[OK] Padding is correctly ignored in Loss.")
    else:
        print("[ERROR] Padding tokens are influencing Loss!")

if __name__ == "__main__":
    debug_dataset()