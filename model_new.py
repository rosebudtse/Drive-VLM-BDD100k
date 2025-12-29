import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig

# Perceiver Resampler for vision-language feature compression
class PerceiverResampler(nn.Module):
    def __init__(self, input_dim, output_dim, num_queries=64):
        super().__init__()
        self.num_queries = num_queries
        self.latents = nn.Parameter(torch.randn(num_queries, output_dim))
        
        self.kv_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True)
        self.ln_q = nn.LayerNorm(output_dim)
        self.ln_kv = nn.LayerNorm(output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(self, visual_features):
        B = visual_features.shape[0]
        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        q = self.ln_q(q)
        kv = self.kv_proj(visual_features)
        kv = self.ln_kv(kv)
        attn_output, _ = self.attn(query=q, key=kv, value=kv)
        return self.out_proj(attn_output)

class VideoLMM(nn.Module):
    def __init__(self, 
                 vision_path="google/siglip2-so400m-patch14-384", 
                 llm_path="Qwen/Qwen3-4B-Instruct-2507",
                 max_frames=16):
        super().__init__()
        
        # Load vision encoder
        print(f"Loading Vision: {vision_path}")
        self.vision_encoder = AutoModel.from_pretrained(vision_path, trust_remote_code=True, dtype=torch.bfloat16)
        self.vision_encoder.requires_grad_(False)
        
        # Detect vision encoder dimension
        if hasattr(self.vision_encoder.config, "hidden_size"):
            vision_dim = self.vision_encoder.config.hidden_size
        elif hasattr(self.vision_encoder.config, "vision_config"):
            vision_dim = self.vision_encoder.config.vision_config.hidden_size
        else:
            vision_dim = 1152
        print(f"   >>> Vision Dimension detected: {vision_dim}")

        # Load LLM with 4-bit quantization
        print(f"Loading LLM: {llm_path}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            quantization_config=bnb_config,
            # DDP requires device_map=None for manual device placement
            device_map=None,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        llm_dim = self.llm.config.hidden_size

        
        # Initialize Perceiver Resampler projector
        print(f"Initializing Resampler: {vision_dim} -> {llm_dim} (64 Queries)")
        self.projector = PerceiverResampler(input_dim=vision_dim, output_dim=llm_dim, num_queries=64)
        self.projector.to(dtype=torch.bfloat16)
        
        # Temporal position embeddings for video frames
        self.temporal_embed = nn.Parameter(torch.zeros(1, max_frames, 1, vision_dim))
        nn.init.normal_(self.temporal_embed, std=0.02)
        
        # Enable gradient checkpointing for memory efficiency
        self.llm.gradient_checkpointing_enable()
        
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def _encode_video(self, pixel_values):
        """Encode video frames with temporal information.
        
        Args:
            pixel_values: (B, T, C, H, W) video tensor
        Returns:
            video_tokens: (B, Num_Queries, LLM_Dim) compressed video features
        """
        b, t, c, h, w = pixel_values.shape
        device = pixel_values.device
        
        # Ensure all components are on the same device
        if self.vision_encoder.device != device:
            self.vision_encoder = self.vision_encoder.to(device)
        if self.projector.latents.device != device:
            self.projector = self.projector.to(device)
        if self.temporal_embed.device != device:
            self.temporal_embed.data = self.temporal_embed.to(device)

        # Flatten frames for vision encoder
        images = pixel_values.view(b * t, c, h, w)
        
        with torch.no_grad():
            if hasattr(self.vision_encoder, "vision_model"):
                vision_out = self.vision_encoder.vision_model(images)
            else:
                vision_out = self.vision_encoder(pixel_values=images)
            
            image_features = vision_out.last_hidden_state

        # Add temporal embeddings
        image_features = image_features.view(b, t, -1, image_features.shape[-1])
        
        # Inject temporal position information
        if t <= self.temporal_embed.shape[1]:
            image_features = image_features + self.temporal_embed[:, :t, :, :]
        else:
            image_features = image_features + self.temporal_embed[:, :self.temporal_embed.shape[1], :, :]

        # Flatten and project to LLM dimension
        image_features = image_features.view(b, -1, image_features.shape[-1])
        
        # Apply Perceiver Resampler projection
        projector_dtype = self.projector.kv_proj.weight.dtype
        video_tokens = self.projector(image_features.to(projector_dtype))
        
        return video_tokens

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None, use_cache=None, **kwargs):
        device = input_ids.device
        
        # Encode video frames
        video_tokens = self._encode_video(pixel_values)
        
        # Concatenate video and text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([video_tokens, text_embeds], dim=1)
        
        b = video_tokens.shape[0]
        
        # Extend attention mask for video tokens
        if attention_mask is not None:
            video_mask = torch.ones(b, video_tokens.shape[1], device=device)
            attention_mask = torch.cat([video_mask, attention_mask], dim=1)
            
        # Mask video tokens in labels (not used for loss)
        if labels is not None:
            ignore_labels = torch.full((b, video_tokens.shape[1]), -100, device=device, dtype=labels.dtype)
            labels = torch.cat([ignore_labels, labels], dim=1)
        
        # Forward pass through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            **kwargs
        )
        
        return outputs
        
    def generate(self, pixel_values, input_ids, attention_mask=None, **kwargs):
        device = input_ids.device
        
        # Encode video frames
        video_tokens = self._encode_video(pixel_values)
        
        # Concatenate video and text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([video_tokens, text_embeds], dim=1)
        
        b = video_tokens.shape[0]

        # Extend attention mask for video tokens
        if attention_mask is not None:
            video_mask = torch.ones(b, video_tokens.shape[1], device=device)
            attention_mask = torch.cat([video_mask, attention_mask], dim=1)
            
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )