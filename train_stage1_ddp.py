import os
import json
import torch
import random
import numpy as np
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoProcessor

from model_new import VideoLMM
from dataset import VideoInstructDataset, collate_fn

# Configuration
DATA_META_FILE = "./data/train_ready.json"
VIDEO_DIR = "./data/zip_folder/bdd100k"
OUTPUT_DIR = "./ckpt/stage1_pretrain"

MODEL_VISION_ID = "google/siglip2-so400m-patch14-384"
MODEL_LLM_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Hyperparameters
MAX_FRAMES = 16
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40

VAL_SIZE = 40
VAL_SPLIT_TXT = os.path.join(OUTPUT_DIR, "val_split.txt")
TRAIN_SPLIT_JSON = os.path.join(OUTPUT_DIR, "train_split.json")
VAL_SPLIT_JSON = os.path.join(OUTPUT_DIR, "val_split.json")



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_video_id(item):
    """Extract video ID from common metadata fields."""
    for key in ["video", "video_path", "video_id", "path"]:
        if key in item:
            return item[key]
    raise KeyError("Video path field not found in sample (video/video_path/video_id/path)")

def prepare_splits(seed=42):
    """Create train/val splits ensuring reproducibility with Stage 2."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DATA_META_FILE, "r") as f:
        data = json.load(f)

    all_videos = [extract_video_id(it) for it in data]
    # Sorted list ensures consistent ordering for reproducible splits
    uniq_videos = sorted(list(set(all_videos))) 

    rng = random.Random(seed)
    val_videos = sorted(rng.sample(uniq_videos, VAL_SIZE))
    val_video_set = set(val_videos)

    # Distribute data by video ID
    train_data, val_data = [], []
    for it in data:
        vid = extract_video_id(it)
        if vid in val_video_set:
            val_data.append(it)
        else:
            train_data.append(it)

    with open(TRAIN_SPLIT_JSON, "w") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(VAL_SPLIT_JSON, "w") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    # Save validation video list for verification
    with open(VAL_SPLIT_TXT, "w") as f:
        for v in val_videos:
            f.write(v + "\n")

    return TRAIN_SPLIT_JSON, VAL_SPLIT_JSON

def main():
    set_seed(42)
    print("[Stage 1] Initializing Feature Alignment Pretraining...")
    
    # Create train/val splits
    train_meta, val_meta = prepare_splits(seed=42)
    
    print("Loading Tokenizer & Processor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LLM_ID, trust_remote_code=True)
    # 🔥 必须添加：确保 padding 逻辑正常工作
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id # 显式同步 ID 确保万无一失
        print(f"   >>> Set tokenizer.pad_token = eos_token (ID: {tokenizer.eos_token_id})")

    image_processor = AutoProcessor.from_pretrained(MODEL_VISION_ID, trust_remote_code=True)

    # Construct datasets
    train_dataset = VideoInstructDataset(
        meta_file=train_meta,
        video_dir=VIDEO_DIR,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_frames=MAX_FRAMES,
        stage='1'
    )
    val_dataset = VideoInstructDataset(
        meta_file=val_meta, 
        video_dir=VIDEO_DIR,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_frames=MAX_FRAMES,
        stage='eval'
    )

    # Initialize model
    model = VideoLMM(
        vision_path=MODEL_VISION_ID, 
        llm_path=MODEL_LLM_ID,
        max_frames=MAX_FRAMES
    )

    # Configure trainable parameters
    print("Configuring Trainable Parameters...")
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze projector
    for param in model.projector.parameters():
        param.requires_grad = True
        
    # Unfreeze temporal embeddings
    model.temporal_embed.requires_grad = True

    # 打印统计
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable: {trainable_params} / {all_params} ({trainable_params/all_params:.2%})")
    print("   -> Projector + Temporal Embeddings are UNLOCKED.")

    # Trainer configuration
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Start training
    print("Start Training Stage 1...")
    trainer.train()

    # Save weights
    if trainer.is_world_process_zero:
        print(f"Saving Stage 1 weights to {OUTPUT_DIR}...")
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        stage1_checkpoint = {
            "projector": model.projector.state_dict(),
            "temporal_embed": model.temporal_embed.data
        }
        torch.save(stage1_checkpoint, os.path.join(OUTPUT_DIR, "stage1_weights.pt"))
        print("[OK] Stage 1 Completed! Weights saved as 'stage1_weights.pt'")

if __name__ == "__main__":
    main()