import json
import os
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import numpy as np
import random
from torchvision import transforms as T
import torchvision.transforms.functional as F

class VideoInstructDataset(Dataset):
    def __init__(self, meta_file, video_dir, tokenizer, image_processor, num_frames=16, max_len=512, stage='1'):
        self.stage = str(stage).lower()
        print(f"Loading data from {meta_file} (Stage: {self.stage})...")
        
        with open(meta_file, 'r') as f:
            self.data = json.load(f)
            
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.max_len = max_len

        self.do_video_aug = (self.stage != 'eval')
        self.do_caption_aug = (self.stage == '2')

        self.jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)
        
        # Verify video directory path
        print(f"Checking video files in {video_dir}...")
        found_count = 0
        for i in range(min(5, len(self.data))):
            vid_name = self.data[i]['video']
            if os.path.exists(os.path.join(video_dir, vid_name)):
                found_count += 1
        
        if found_count == 0 and len(self.data) > 0:
            print(f"Warning: Check your VIDEO_DIR! Could not find the first few videos.")
        else:
            print(f"Path check passed. Samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def _add_gaussian_noise(self, tensor, mean=0., std=0.05):
        noise = torch.randn_like(tensor) * std + mean
        return tensor + noise
        
    def _apply_visual_augmentation(self, frames_tensor):
        fn_idx, b, c, s, h = T.ColorJitter.get_params(
            self.jitter.brightness, self.jitter.contrast,
            self.jitter.saturation, self.jitter.hue
        )
        do_noise = random.random() < 0.5
        
        augmented_frames = []
        for frame in frames_tensor:
            frame = F.adjust_brightness(frame, b)
            frame = F.adjust_contrast(frame, c)
            frame = F.adjust_saturation(frame, s)
            frame = F.adjust_hue(frame, h)
            augmented_frames.append(frame)
            
        stack = torch.stack(augmented_frames)
        if do_noise:
            stack = self._add_gaussian_noise(stack)
        return torch.clamp(stack, 0.0, 1.0) 

    def __getitem__(self, idx):
        item = self.data[idx]
        
        video_filename = item['video']
        video_path = os.path.join(self.video_dir, video_filename)
        
        captions_list = item.get('captions', [item.get('a', '')])
        
        if self.do_caption_aug:
            answer = random.choice(captions_list)
        else:
            answer = captions_list[0]

        question = item.get('q', 'Describe this video.')

        # Video processing
        try:
            if not os.path.exists(video_path):
                # Retry with random sample to handle missing files
                return self.__getitem__(random.randint(0, len(self.data)-1))

            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames >= self.num_frames:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            else:
                indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
                indices = np.tile(indices, (self.num_frames // total_frames + 1))[:self.num_frames]
            
            pixel_values_np = vr.get_batch(indices).asnumpy() 
            video_tensor = torch.from_numpy(pixel_values_np).permute(0, 3, 1, 2).float() / 255.0
            
            if self.do_video_aug:
                video_tensor = self._apply_visual_augmentation(video_tensor)
            
            inputs = self.image_processor(images=list(video_tensor), return_tensors="pt", do_rescale=False)
            pixel_values_tensor = inputs.pixel_values 
            
        except Exception as e:
            print(f"Error loading {video_filename}: {e}. Skipping...")
            return self.__getitem__(random.randint(0, len(self.data)-1))

        # Text processing and tokenization
        
        # Construct user prompt without system message
        user_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # Construct full text with answer
        full_text = user_prompt + f"{answer}<|im_end|>"
        
        # Tokenize full text (manual chat template formatting)
        tokenized_full = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=False
        )
        
        input_ids = tokenized_full.input_ids.squeeze(0)
        attention_mask = tokenized_full.attention_mask.squeeze(0)
        
        # Create labels with masking strategy
        labels = input_ids.clone()
        
        # Calculate prompt length for masking
        tokenized_user = self.tokenizer(
            user_prompt,
            return_tensors="pt",
            add_special_tokens=False
        )
        prompt_len = tokenized_user.input_ids.shape[1]
        
        # Mask user prompt (not used in loss)
        if prompt_len < self.max_len:
            labels[:prompt_len] = -100
        else:
            labels[:] = -100
            
        # Mask padding tokens
        labels[attention_mask == 0] = -100
        
        return {
            "pixel_values": pixel_values_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
        
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    input_ids = torch.stack([b['input_ids'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    
    # Handle attention mask
    if 'attention_mask' in batch[0]:
        attention_mask = torch.stack([b['attention_mask'] for b in batch])
    else:
        attention_mask = (input_ids != 0).long()
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }