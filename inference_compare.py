import os
import json
import torch
import random
import argparse
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel
from decord import VideoReader, cpu
import numpy as np

from model_new import VideoLMM

# Configuration
MODEL_VISION_ID = "google/siglip2-so400m-patch14-384"
MODEL_LLM_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Checkpoint paths
STAGE1_PROJECTOR = "./ckpt/stage1_pretrain/stage1_weights.pt"

STAGE2_PROJECTOR = "./ckpt/stage2_lora/non_lora_weights.pt"
STAGE2_LORA = "./ckpt/stage2_lora/lora_adapter"

# Data paths
DATA_JSON = "/root/autodl-tmp/Project/ckpt/stage2_lora/train_split.json"
VIDEO_DIR = "./data/zip_folder/bdd100k"

# Inference parameters
NUM_FRAMES = 32
MAX_NEW_TOKENS = 512

def load_video(video_path, image_processor, num_frames=32):
    """Load and preprocess video frames."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
        indices = np.tile(indices, (num_frames // total_frames + 1))[:num_frames]
    
    pixel_values = vr.get_batch(indices).asnumpy()
    inputs = image_processor(images=list(pixel_values), return_tensors="pt")
    # Ensure batch dimension: [1, T, C, H, W]
    return inputs.pixel_values.unsqueeze(0)

def generate_text(model, tokenizer, pixel_values, prompt, device):
    """Generate text response from video and prompt."""
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    input_ids = model_inputs.input_ids
    
    pixel_values = pixel_values.to(device, dtype=model.llm.dtype)
    
    # Extract visual features and prepare embeddings
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(0)
    b, t, c, h, w = pixel_values.shape
    images = pixel_values.view(b * t, c, h, w)
    
    with torch.no_grad():
        if hasattr(model.vision_encoder, "vision_model"):
            vision_out = model.vision_encoder.vision_model(images)
        else:
            vision_out = model.vision_encoder(pixel_values=images)
        image_features = vision_out.last_hidden_state
        image_features = image_features.view(b, -1, image_features.shape[-1])
        # Align dtype for projector
        proj_dtype = model.projector.kv_proj.weight.dtype
        video_tokens = model.projector(image_features.to(dtype=proj_dtype))
        
        # Concatenate visual and text embeddings
        text_embeds = model.llm.get_input_embeddings()(input_ids)
        if video_tokens.dtype != text_embeds.dtype:
            video_tokens = video_tokens.to(text_embeds.dtype)
        inputs_embeds = torch.cat([video_tokens, text_embeds], dim=1)
        
        # Generate response
        generated_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    # Decode and clean response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Extract assistant response
    if "assistant\n" in response:
        response = response.split("assistant\n")[-1]
        
    return response.strip()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Inference Comparison on {device}...")

    # Load base model
    print("Loading Base Model (Vision + LLM)...")
    model = VideoLMM(vision_path=MODEL_VISION_ID, llm_path=MODEL_LLM_ID)
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LLM_ID, trust_remote_code=True)
    image_processor = AutoProcessor.from_pretrained(MODEL_VISION_ID, trust_remote_code=True)

    # Select test video
    with open(DATA_JSON, 'r') as f:
        data = json.load(f)
    
    item = random.choice(data)
    
    video_name = item['video']
    video_path = os.path.join(VIDEO_DIR, video_name)
    gt_caption = item['captions'][0]
    prompt = item['q']
    
    print(f"\nSelected Video: {video_name}")
    if not os.path.exists(video_path):
        print("[ERROR] Video file not found!")
        return

    # Load video data
    pixel_values = load_video(video_path, image_processor, num_frames=NUM_FRAMES)

    print("-" * 50)

    # Stage 1 inference
    print("Running Stage 1 Inference (Feature Alignment)...")
    
    if os.path.exists(STAGE1_PROJECTOR):
        ckpt = torch.load(STAGE1_PROJECTOR, map_location=device)
        if "projector" in ckpt:
            model.projector.load_state_dict(ckpt["projector"])
        else:
            model.projector.load_state_dict(ckpt)
        if "temporal_embed" in ckpt:
            model.temporal_embed.data = ckpt["temporal_embed"].to(model.temporal_embed.device)
        print("   [OK] Stage 1 projector/temporal loaded.")
        stage1_out = generate_text(model, tokenizer, pixel_values, prompt, device)
    else:
        stage1_out = "[ERROR] Stage 1 weights not found."

    # Stage 2 inference
    print("\nRunning Stage 2 Inference (Instruction Tuning)...")
    
    # 加载 Stage 2 Projector + Temporal
    if os.path.exists(STAGE2_PROJECTOR):
        ckpt2 = torch.load(STAGE2_PROJECTOR, map_location=device)
        if "projector" in ckpt2:
            model.projector.load_state_dict(ckpt2["projector"])
        else:
            model.projector.load_state_dict(ckpt2)
        if "temporal_embed" in ckpt2:
            model.temporal_embed.data = ckpt2["temporal_embed"].to(model.temporal_embed.device)
        print("   ✅ Stage 2 projector/temporal loaded.")
    else:
        print("   ❌ Stage 2 Projector not found!")

    # 加载 LoRA
    if os.path.exists(STAGE2_LORA):
        # 这里的 model.llm 是一个 AutoModelForCausalLM
        # 我们用 PeftModel 包装它
        model.llm = PeftModel.from_pretrained(model.llm, STAGE2_LORA)
        print("   ✅ Stage 2 LoRA Adapter loaded.")
        
        # 运行生成
        stage2_out = generate_text(model, tokenizer, pixel_values, prompt, device)
    else:
        stage2_out = "[ERROR] Stage 2 LoRA not found."

    # Print comparison results
    print("\n" + "=" * 20 + " RESULT COMPARISON " + "=" * 20)
    
    print(f"Video: {video_name}")
    print(f"Prompt: {prompt}\n")
    
    print("Ground Truth (Label):")
    print(f"{gt_caption}\n")
    
    print("Stage 1 Output (Pretrain):")
    print(f"{stage1_out}\n")
    
    print("Stage 2 Output (SFT + LoRA):")
    print(f"{stage2_out}\n")
    
    print("=" * 60)

if __name__ == "__main__":
    main()