
import os
import argparse
import torch
from safetensors.torch import load_file as safe_load_file
from peft import LoraConfig, get_peft_model
from model_new import VideoLMM

MODEL_VISION_ID = "google/siglip2-so400m-patch14-384"
MODEL_LLM_ID = "Qwen/Qwen3-4B-Instruct-2507"
MAX_FRAMES = 16

# LoRA configuration matching training setup
PEFT_CFG = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

def load_state_dict_from_checkpoint(ckpt_dir: str):
    safe_path = os.path.join(ckpt_dir, "model.safetensors")
    bin_path  = os.path.join(ckpt_dir, "pytorch_model.bin")

    if os.path.exists(safe_path):
        print(f"loading safetensors: {safe_path}")
        return safe_load_file(safe_path)
    elif os.path.exists(bin_path):
        # Note: requires torch>=2.6 for transformers compatibility
        print(f"loading torch bin: {bin_path}")
        return torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"no model file found in {ckpt_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint-XXX directory")
    ap.add_argument("--out",  required=True, help="Output directory for export")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Build model and inject LoRA
    model = VideoLMM(
        vision_path=MODEL_VISION_ID,
        llm_path=MODEL_LLM_ID,
        max_frames=MAX_FRAMES
    )
    model.llm = get_peft_model(model.llm, PEFT_CFG)

    # Load checkpoint weights (projector/temporal + LoRA)
    sd = load_state_dict_from_checkpoint(args.ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing keys:", len(missing), "unexpected:", len(unexpected))

    # Export LoRA adapter
    lora_out = os.path.join(args.out, "lora_adapter")
    model.llm.save_pretrained(lora_out)
    print("saved:", lora_out)

    # Export projector and temporal embeddings
    extra = {
        "projector": model.projector.state_dict(),
        "temporal_embed": model.temporal_embed.data
    }
    torch.save(extra, os.path.join(args.out, "non_lora_weights.pt"))
    print("saved:", os.path.join(args.out, "non_lora_weights.pt"))

if __name__ == "__main__":
    main()