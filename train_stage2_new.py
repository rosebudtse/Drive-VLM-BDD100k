import os
import json
import torch
import random
import numpy as np
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

from model_new import VideoLMM
from dataset import VideoInstructDataset, collate_fn

# Configuration
DATA_META_FILE = "./data/train_augmented_stage2.json"
VIDEO_DIR = "./data/zip_folder/bdd100k"
OUTPUT_DIR = "./ckpt/stage2_lora"

# Split configuration
VAL_SIZE = 40
VAL_SPLIT_TXT = os.path.join(OUTPUT_DIR, "val_split.txt")
TRAIN_SPLIT_JSON = os.path.join(OUTPUT_DIR, "train_split.json")
VAL_SPLIT_JSON = os.path.join(OUTPUT_DIR, "val_split.json")

# Stage 1 checkpoint path
STAGE1_WEIGHTS_PATH = "./ckpt/stage1_pretrain/stage1_weights.pt"

MODEL_VISION_ID = "google/siglip2-so400m-patch14-384"
MODEL_LLM_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Hyperparameters
MAX_FRAMES = 16
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 20

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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading data from {DATA_META_FILE}...")
    with open(DATA_META_FILE, "r") as f:
        data = json.load(f)

    # Extract unique video IDs
    all_videos = []
    for it in data:
        vid = extract_video_id(it)
        all_videos.append(vid)
    uniq_videos = sorted(list(set(all_videos)))

    print(f"Found {len(data)} total entries, corresponding to {len(uniq_videos)} unique videos.")

    if len(uniq_videos) < VAL_SIZE:
        raise ValueError(f"视频数量不足 {VAL_SIZE} 个，当前仅 {len(uniq_videos)} 个")

    # 2. 随机抽取 40 个视频作为验证集
    rng = random.Random(seed)
    val_videos = sorted(rng.sample(uniq_videos, VAL_SIZE))
    val_video_set = set(val_videos)

    # 3. 若已有缓存，校验一致性 (防止复现错误)
    if os.path.exists(VAL_SPLIT_TXT):
        with open(VAL_SPLIT_TXT, "r") as f:
            prev = [line.strip() for line in f if line.strip()]
        if sorted(prev) != val_videos:
            print("Warning: Found old val_split.txt inconsistent with current seed!")
            print("   This usually means the dataset changed. Overwriting old split file...")
            with open(VAL_SPLIT_TXT, "w") as f:
                for v in val_videos:
                    f.write(v + "\n")
    else:
        with open(VAL_SPLIT_TXT, "w") as f:
            for v in val_videos:
                f.write(v + "\n")

    # Distribute data by video ID (keeps detail/summary entries together)
    train_data, val_data = [], []
    
    for it in data:
        vid = extract_video_id(it)
        if vid in val_video_set:
            val_data.append(it)
        else:
            train_data.append(it)

    # 写出 json
    with open(TRAIN_SPLIT_JSON, "w") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(VAL_SPLIT_JSON, "w") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Split completed:")
    print(f"   - Train Data: {len(train_data)} samples (approx {len(uniq_videos) - VAL_SIZE} videos)")
    print(f"   - Val Data:   {len(val_data)} samples (approx {VAL_SIZE} videos)")
    
    return TRAIN_SPLIT_JSON, VAL_SPLIT_JSON

def main():
    set_seed(42)
    print("[Stage 2] Initializing Instruction Tuning (LoRA)...")

    # Create reproducible train/val splits
    train_meta, val_meta = prepare_splits(seed=42)

    # Load tokenizer and image processor
    print("Loading Tokenizer & Processor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LLM_ID, trust_remote_code=True)
    image_processor = AutoProcessor.from_pretrained(MODEL_VISION_ID, trust_remote_code=True)

    # Construct train and validation datasets
    train_dataset = VideoInstructDataset(
        meta_file=train_meta,
        video_dir=VIDEO_DIR,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_frames=MAX_FRAMES,
        stage='2'
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

    # Load Stage 1 pretrained weights
    print(f"Loading Stage 1 weights from {STAGE1_WEIGHTS_PATH}...")
    if os.path.exists(STAGE1_WEIGHTS_PATH):
        checkpoint = torch.load(STAGE1_WEIGHTS_PATH, map_location="cpu")
        model.projector.load_state_dict(checkpoint["projector"])
        model.temporal_embed.data = checkpoint["temporal_embed"].to(model.temporal_embed.device)
        print("[OK] Stage 1 weights (Projector + Temporal) loaded successfully.")
    else:
        raise FileNotFoundError("[ERROR] Please run Stage 1 first to generate weights!")

    # Configure LoRA and freeze strategy
    print("Configuring LoRA & Freeze Strategy...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = True
    model.temporal_embed.requires_grad = False  # Stage 2 不训练 Temporal Embeddings

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model.llm = get_peft_model(model.llm, peft_config)
    model.llm.config.use_cache = False
    model.llm.enable_input_require_grads()

    for name, param in model.llm.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    model.llm.print_trainable_parameters()

    # Trainer configuration with evaluation
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
        save_safetensors=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=8,
        save_total_limit=2,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Start training
    print("Start Training Stage 2...")
    trainer.train()

    # Save final model components
    print(f"Saving Stage 2 components to {OUTPUT_DIR}...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save LoRA adapter
    model.llm.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))

    # Save projector and temporal embeddings
    extra_weights = {
        "projector": model.projector.state_dict(),
        "temporal_embed": model.temporal_embed.data
    }
    torch.save(extra_weights, os.path.join(OUTPUT_DIR, "non_lora_weights.pt"))

    print("[OK] Stage 2 Completed! LoRA adapters + Non-LoRA weights saved.")

if __name__ == "__main__":
    main()