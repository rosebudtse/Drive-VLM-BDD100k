import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer, AutoProcessor
from tqdm import tqdm
from copy import deepcopy
from decord import VideoReader, cpu

from model import VideoLMM
import sys
from stage3.grpo_reward_func import BDD100KRewardFunctionV2, SimplifiedRewardFunctionV2, BinaryRewardWrapper

# Configuration
DATA_META_FILE = "./data/train_ready.json" 
VIDEO_DIR = "./data/zip_folder/bdd100k"
MODEL_VISION_ID = "google/siglip2-so400m-patch14-384"
MODEL_LLM_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Checkpoint paths
STAGE2_CHECKPOINT_DIR = "./chpt/stage2_lora"
STAGE3_OUTPUT_DIR = "./chpt/stage3_grpo"
YOLO_CACHE_PATH = "./stage3/yolo_cache.pkl"

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random Seed set to: {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Video-LMM Stage 3: GRPO Training")
    parser.add_argument("--stage2_checkpoint", type=str, default=STAGE2_CHECKPOINT_DIR)
    parser.add_argument("--output_dir", type=str, default=STAGE3_OUTPUT_DIR)
    parser.add_argument("--reward_mode", type=str, default="binary", 
                       choices=["cached", "simplified", "binary"])
    parser.add_argument("--yolo_cache", type=str, default=YOLO_CACHE_PATH)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--kl_coef", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    return parser.parse_args()

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
    return inputs.pixel_values.squeeze(0)  # [T, C, H, W]

def load_stage2_model(stage2_checkpoint_dir, device='cuda'):
    """加载Stage 2的checkpoint"""
    print(f"📦 Loading Stage 2 checkpoint from {stage2_checkpoint_dir}...")
    model = VideoLMM(vision_path=MODEL_VISION_ID, llm_path=MODEL_LLM_ID)
    
    # 🔥 修复2: 统一Projector文件名
    projector_path = os.path.join(stage2_checkpoint_dir, "projector_final.pt")
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"❌ Projector not found at {projector_path}")
    
    state_dict = torch.load(projector_path, map_location="cpu")
    model.projector.load_state_dict(state_dict)
    print(f"   ✅ Loaded Projector from {projector_path}")
    
    # LoRA
    from peft import PeftModel
    lora_path = os.path.join(stage2_checkpoint_dir, "lora_adapter")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"❌ LoRA adapters not found at {lora_path}")
    
    model.llm = PeftModel.from_pretrained(model.llm, lora_path, is_trainable=True)
    print(f"   ✅ Loaded LoRA adapters from {lora_path}")
    
    # 确保dtype一致
    model.projector = model.projector.to(model.llm.dtype)
    model = model.to(device)
    
    return model

def create_reference_model(policy_model, device='cuda'):
    """Create reference model for KL divergence computation."""
    print("Creating reference model...")
    
    ref_model = deepcopy(policy_model)
    
    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False
    
    ref_model.eval()
    
    ref_model = ref_model.to(device)
    print("   [OK] Reference model ready")
    
    return ref_model

def generate_responses(model, pixel_values, prompt_ids, tokenizer, args, device):
    """Generate diverse responses using varied sampling parameters."""
    model.eval()
    responses = []
    
    # Compute video embeddings once
    b, t, c, h, w = pixel_values.shape
    images = pixel_values.view(b * t, c, h, w)
    
    with torch.no_grad():
        if hasattr(model.vision_encoder, "vision_model"):
            vision_out = model.vision_encoder.vision_model(images)
        else:
            vision_out = model.vision_encoder(pixel_values=images)
        
        image_features = vision_out.last_hidden_state
        image_features = image_features.view(b, -1, image_features.shape[-1])
        video_tokens = model.projector(image_features.to(model.llm.dtype))
        text_embeds = model.llm.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([video_tokens, text_embeds], dim=1)
        
        # Generate multiple responses with varying temperature
        for i in range(args.group_size):
            # Increase temperature slightly for diversity
            temp = args.temperature + (i * 0.05)
            temp = min(temp, 1.5)
            
            generated_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Extract response from generated text
            full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Parse Qwen chat template format
            if "<|im_start|>assistant" in full_text:
                response = full_text.split("<|im_start|>assistant")[-1]
                response = response.replace("<|im_end|>", "").strip()
            elif "assistant\n" in full_text:
                response = full_text.split("assistant\n")[-1].strip()
            else:
                response = full_text.strip()
            
            # Ensure non-empty response
            if len(response) < 10:
                response = "A video showing driving scenes."
            
            responses.append(response)
    
    return responses

def get_batch_logps(model, pixel_values, input_ids, label_ids):
    """Compute log probabilities for generated tokens."""
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
    )
    
    logits = outputs.logits[:, :-1, :]
    labels = label_ids[:, 1:]
    
    # Use float32 for numerical stability
    logits = logits.to(torch.float32)
    
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Prepare gather index (replace -100 to avoid errors)
    loss_mask = (labels != -100)
    safe_labels = labels.clone()
    safe_labels[~loss_mask] = 0
    
    per_token_logps = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
    
    # Mask padding tokens
    per_token_logps = per_token_logps * loss_mask.float()
    
    return per_token_logps.sum(dim=1)

def compute_grpo_loss_step(policy_model, ref_model, pixel_values, prompt_ids, 
                           responses, rewards, tokenizer, kl_coef, device):
    """Compute GRPO loss with KL divergence regularization.
    
    Returns:
        loss: Computed loss value
        metrics: Dictionary of training metrics
    """
    # Normalize rewards
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    
    if len(rewards) > 1 and rewards.std() > 1e-6:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        advantages = rewards - rewards.mean()
    
    total_loss = 0.0
    total_kl = 0.0
    
    for i, response in enumerate(responses):
        # 编码response
        resp_ids = tokenizer.encode(response, add_special_tokens=False, return_tensors='pt').to(device)
        
        # 拼接: [Prompt, Response]
        full_ids = torch.cat([prompt_ids, resp_ids], dim=1)
        
        # 构造labels: Prompt部分设为-100
        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100
        
        # 计算Policy log prob
        logp_policy = get_batch_logps(policy_model, pixel_values, full_ids, labels)
        
        # 计算Reference log prob
        with torch.no_grad():
            logp_ref = get_batch_logps(ref_model, pixel_values, full_ids, labels)
        
        # KL散度
        kl_div = logp_policy - logp_ref
        
        # GRPO损失: -advantage * log_p + kl_coef * KL
        loss = -(advantages[i] * logp_policy - kl_coef * kl_div)
        total_loss += loss
        total_kl += kl_div.item()
    
    avg_loss = total_loss / len(responses)
    avg_kl = total_kl / len(responses)
    
    metrics = {
        'loss': avg_loss.item(),
        'kl_div': avg_kl,
        'mean_reward': rewards.mean().item(),
        'std_reward': rewards.std().item() if len(rewards) > 1 else 0.0,
        'mean_advantage': advantages.mean().item(),
    }
    
    return avg_loss, metrics

def main():
    SEED = 42
    set_seed(SEED)
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("Video-LMM Stage 3: GRPO Training")
    print("="*60)
    print(f"   Stage 2: {args.stage2_checkpoint}")
    print(f"   Output: {args.output_dir}")
    print(f"   Reward: {args.reward_mode}")
    print(f"   Group Size: {args.group_size}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   LR: {args.learning_rate}")
    print(f"   KL Coef: {args.kl_coef}")
    
    # Load dataset
    print("\nLoading dataset...")
    with open(DATA_META_FILE, 'r') as f:
        dataset = json.load(f)
    print(f"   [OK] {len(dataset)} samples")
    
    # Configure reward function
    print(f"\nSetting up reward function ({args.reward_mode})...")
    
    if args.reward_mode == "cached":
        if not os.path.exists(args.yolo_cache):
            raise FileNotFoundError(f"[ERROR] YOLO cache not found: {args.yolo_cache}")
        reward_fn = BDD100KRewardFunctionV2(
            yolo_cache_path=args.yolo_cache,
            use_detector=False,
            use_nlp=True
        )
    elif args.reward_mode == "simplified":
        reward_fn = SimplifiedRewardFunctionV2()
    else:
        base = SimplifiedRewardFunctionV2()
        reward_fn = BinaryRewardWrapper(base, threshold=0.65)
    
    # Load policy model
    policy_model = load_stage2_model(args.stage2_checkpoint, device)
    
    # Enable trainable parameters
    for p in policy_model.projector.parameters():
        p.requires_grad = True
    for n, p in policy_model.llm.named_parameters():
        if "lora" in n:
            p.requires_grad = True
    
    # Print parameter statistics
    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy_model.parameters())
    print(f"\nTrainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    # Reference model
    ref_model = create_reference_model(policy_model, device)
    
    # 4. Tokenizer & Processor
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LLM_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    image_processor = AutoProcessor.from_pretrained(MODEL_VISION_ID, trust_remote_code=True)
    
    # 5. 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy_model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # 6. 训练循环
    print(f"\n🔥 Starting GRPO Training...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    
    # Training history
    history = {
        'loss': [],
        'reward': [],
        'kl_div': []
    }
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*60}")
        
        epoch_metrics = {'loss': 0.0, 'reward': 0.0, 'kl_div': 0.0, 'valid_samples': 0}
        random.shuffle(dataset)
        policy_model.train()
        
        progress_bar = tqdm(dataset, desc=f"Epoch {epoch+1}")
        
        for idx, item in enumerate(progress_bar):
            video_name = item['video']
            video_path = os.path.join(VIDEO_DIR, video_name)
            
            if not os.path.exists(video_path):
                # progress_bar.write(f"⚠️  Video not found: {video_name}") 
                continue
            
            try:
                # Load video
                pixel_values = load_video(video_path, image_processor).unsqueeze(0).to(device)
                
                # Construct prompt
                messages = [
                    {"role": "system", "content": "You are a helpful assistant capable of understanding video content."},
                    {"role": "user", "content": item['q']}
                ]
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(device)
                
                # Generate responses
                responses = generate_responses(
                    policy_model, pixel_values, prompt_ids, tokenizer, args, device
                )
                
                # Compute rewards
                rewards = []
                for resp in responses:
                    r = reward_fn.compute_reward(
                        video_name=video_name, 
                        generated_caption=resp, 
                        ground_truth_caption=item['a']
                    )
                    rewards.append(r)
                
                # Compute GRPO loss
                loss, metrics = compute_grpo_loss_step(
                    policy_model, ref_model, pixel_values, prompt_ids,
                    responses, rewards, tokenizer, args.kl_coef, device
                )
                
                # Backward pass
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                # Gradient accumulation
                if (idx + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    torch.cuda.empty_cache()
                    
                    # Save checkpoint periodically
                    if global_step % args.save_steps == 0:
                        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        
                        torch.save(
                            policy_model.projector.state_dict(),
                            os.path.join(ckpt_dir, "projector.pt")
                        )
                        policy_model.llm.save_pretrained(
                            os.path.join(ckpt_dir, "lora_adapter")
                        )
                        
                        progress_bar.write(f"   Checkpoint-{global_step} saved")
                del responses, rewards, pixel_values, prompt_ids, loss
                
                # Track metrics
                epoch_metrics['loss'] += metrics['loss']
                epoch_metrics['reward'] += metrics['mean_reward']
                epoch_metrics['kl_div'] += metrics['kl_div']
                epoch_metrics['valid_samples'] += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{metrics['loss']:.4f}",
                    'Reward': f"{metrics['mean_reward']:.3f}",
                    'KL': f"{metrics['kl_div']:.3f}"
                })
            
            except Exception as e:
                progress_bar.write(f"[ERROR] Error processing {item['video']}: {str(e)[:100]}")
                continue
        
        # Epoch summary
        if epoch_metrics['valid_samples'] > 0:
            avg_loss = epoch_metrics['loss'] / epoch_metrics['valid_samples']
            avg_reward = epoch_metrics['reward'] / epoch_metrics['valid_samples']
            avg_kl = epoch_metrics['kl_div'] / epoch_metrics['valid_samples']
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   Reward: {avg_reward:.4f}")
            print(f"   KL Div: {avg_kl:.4f}")
            print(f"   Valid: {epoch_metrics['valid_samples']}/{len(dataset)}")
            
            history['loss'].append(avg_loss)
            history['reward'].append(avg_reward)
            history['kl_div'].append(avg_kl)
    
    # Save final model
    print(f"\nSaving final model to {args.output_dir}...")
    torch.save(
        policy_model.projector.state_dict(),
        os.path.join(args.output_dir, "projector_final.pt")
    )
    policy_model.llm.save_pretrained(
        os.path.join(args.output_dir, "lora_adapter")
    )
    
    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("[OK] Stage 3 GRPO Training Completed!")
    print(f"   Final Reward: {history['reward'][-1]:.4f}")
    print(f"   Final KL: {history['kl_div'][-1]:.4f}")

if __name__ == "__main__":
    main()