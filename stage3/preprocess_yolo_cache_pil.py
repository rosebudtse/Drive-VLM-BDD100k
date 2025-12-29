# preprocess_yolo_cache_pil.py
"""
YOLO Cache Preprocessing using PIL
Uses PIL instead of OpenCV to avoid resize bugs in OpenCV 4.12.0
"""

import os
import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
import pickle
from typing import Dict
from decord import VideoReader, cpu

class YOLOCacheGeneratorPIL:
    """
    YOLO cache generator using PIL for image processing.
    """
    
    def __init__(self, detector_model='yolov8x.pt', device='cuda'):
        print(f"Loading YOLO model: {detector_model}")
        self.detector = YOLO(detector_model)
        self.device = device
        self.coco_classes = self.detector.names
        print(f"Loaded {len(self.coco_classes)} COCO classes")
    
    
    def detect_video(self, video_path: str, num_frames: int = 32, sample_rate: int = 4) -> Dict:
        """
        Detect objects in a single video.
        """
        video_name = os.path.basename(video_path)
        
        try:
            # Read video
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames == 0:
                raise ValueError(f"Video has 0 frames")
            
            # Sample frame indices
            if total_frames >= num_frames:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            else:
                indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
                indices = np.tile(indices, (num_frames // total_frames + 1))[:num_frames]
            
            # Batch read frames
            frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
            
            if frames.shape[0] == 0 or frames.shape[1] == 0 or frames.shape[2] == 0:
                raise ValueError(f"Invalid frame shape: {frames.shape}")
            
            # Sample detection frames
            T = frames.shape[0]
            sample_indices = list(range(0, T, sample_rate))
            sampled_frames = frames[sample_indices]
            
            # Convert to PIL Image list (avoids OpenCV)
            pil_images = []
            for i in range(sampled_frames.shape[0]):
                frame = sampled_frames[i]  # [H, W, C]
                pil_img = Image.fromarray(frame)
                pil_images.append(pil_img)
            
            # Debug info for first few videos
            if hasattr(self, 'processed_count') and self.processed_count < 3:
                print(f"\n[DEBUG] {video_name}")
                print(f"  Total frames: {total_frames}")
                print(f"  Detection frames: {len(pil_images)}")
                print(f"  First image size: {pil_images[0].size}")
                print(f"  First image mode: {pil_images[0].mode}")
            
            # YOLO detection on PIL images
            results = self.detector(
                pil_images,
                imgsz=640,
                verbose=False,
                conf=0.3
            )
            
            # Parse detection results
            detected_objects = {}
            frame_detections = {}
            confidence_scores = {}
            
            for i, result in enumerate(results):
                frame_idx = sample_indices[i]
                frame_objs = []
                
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.coco_classes[cls_id]
                        conf = float(box.conf[0])
                        
                        detected_objects[cls_name] = detected_objects.get(cls_name, 0) + 1
                        frame_objs.append(cls_name)
                        
                        if cls_name not in confidence_scores or conf > confidence_scores[cls_name]:
                            confidence_scores[cls_name] = conf
                
                frame_detections[frame_idx] = frame_objs
            
            # Average object counts
            for key in detected_objects:
                detected_objects[key] = max(1, detected_objects[key] // len(sample_indices))
            
            return {
                'detected_objects': detected_objects,
                'detected_classes': list(set(detected_objects.keys())),
                'frame_detections': frame_detections,
                'confidence_scores': confidence_scores,
                'num_frames_sampled': len(sample_indices),
                'status': 'success',
                'video_info': {
                    'total_frames': total_frames,
                    'frame_shape': list(frames.shape)
                }
            }
        
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            tqdm.write(f"[ERROR] {video_name}: {error_type}")
            if hasattr(self, 'processed_count') and self.processed_count < 5:
                tqdm.write(f"Message: {error_msg[:200]}")
            
            return {
                'detected_objects': {},
                'detected_classes': [],
                'frame_detections': {},
                'confidence_scores': {},
                'status': 'error',
                'error': error_msg,
                'error_type': error_type
            }
    
    
    def process_dataset(self, video_dir: str, meta_file: str, output_cache: str,
                       num_frames: int = 32, sample_rate: int = 4) -> Dict:
        """
        Process entire dataset and generate YOLO cache.
        """
        
        print(f"Reading metadata from {meta_file}")
        with open(meta_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        video_names = [item['video'] for item in data]
        print(f"Found {len(video_names)} videos\n")
        
        os.makedirs(os.path.dirname(output_cache) if os.path.dirname(output_cache) else '.', exist_ok=True)
        
        cache = {}
        success_count = 0
        error_count = 0
        error_types = {}
        self.processed_count = 0
        
        print(f"Processing videos with YOLO (PIL backend)...")
        print(f"Frames per video: {num_frames}")
        print(f"Sample rate: 1/{sample_rate} (detecting {num_frames // sample_rate} frames)")
        
        for video_name in tqdm(video_names, desc="YOLO Detection"):
            video_path = os.path.join(video_dir, video_name)
            
            if not os.path.exists(video_path):
                tqdm.write(f"[WARNING] Video not found: {video_path}")
                cache[video_name] = {
                    'status': 'error',
                    'error': 'file_not_found',
                    'error_type': 'missing_file'
                }
                error_count += 1
                continue
            
            result = self.detect_video(video_path, num_frames, sample_rate)
            cache[video_name] = result
            
            if result.get('status') == 'success':
                success_count += 1
            else:
                error_count += 1
                err_type = result.get('error_type', 'unknown')
                error_types[err_type] = error_types.get(err_type, 0) + 1
            
            self.processed_count += 1
        
        print(f"\nProcessing completed:")
        print(f"Total: {len(video_names)}")
        print(f"Success: {success_count}")
        print(f"Errors: {error_count}")
        
        if error_types:
            print(f"\nError types:")
            for err_type, count in error_types.items():
                print(f"- {err_type}: {count}")
        
        print(f"\nSaving cache...")
        with open(output_cache, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
        
        pickle_cache = output_cache.replace('.json', '.pkl')
        with open(pickle_cache, 'wb') as f:
            pickle.dump(cache, f)
        
        print(f"[OK] JSON: {output_cache}")
        print(f"[OK] Pickle: {pickle_cache}")
        
        # Generate statistics report
        self._generate_statistics_report(cache, output_cache.replace('.json', '_stats.txt'))
        
        return cache
    
    
    def _generate_statistics_report(self, cache: Dict, report_path: str):
        """
        Generate statistics report for detection results.
        """
        successful_videos = {k: v for k, v in cache.items() if v.get('status') == 'success'}
        
        if len(successful_videos) == 0:
            print(f"[WARNING] No successful detections")
            return
        
        all_objects = []
        for video_name, detection in successful_videos.items():
            all_objects.extend(detection.get('detected_classes', []))
        
        from collections import Counter
        object_counts = Counter(all_objects)
        
        objects_per_video = []
        for video_name, detection in successful_videos.items():
            objects_per_video.append(len(detection.get('detected_classes', [])))
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("BDD100K YOLO Detection Statistics\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total videos: {len(cache)}\n")
            f.write(f"Successfully processed: {len(successful_videos)}\n")
            f.write(f"Failed videos: {len(cache) - len(successful_videos)}\n\n")
            
            if len(objects_per_video) > 0:
                f.write(f"Average objects per video: {np.mean(objects_per_video):.2f}\n")
                f.write(f"Max objects in a video: {np.max(objects_per_video)}\n")
                f.write(f"Min objects in a video: {np.min(objects_per_video)}\n\n")
            
            f.write("Top 20 Most Common Objects:\n")
            f.write("-"*60 + "\n")
            for obj, count in object_counts.most_common(20):
                percentage = (count / len(successful_videos)) * 100
                f.write(f"{obj:20s} {count:5d} videos ({percentage:5.1f}%)\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"[OK] Statistics: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--meta_file", type=str, required=True)
    parser.add_argument("--output", type=str, default="./yolo_cache.json")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--sample_rate", type=int, default=4)
    parser.add_argument("--detector", type=str, default="yolov8x.pt")
    args = parser.parse_args()
    
    generator = YOLOCacheGeneratorPIL(detector_model=args.detector)
    cache = generator.process_dataset(
        video_dir=args.video_dir,
        meta_file=args.meta_file,
        output_cache=args.output,
        num_frames=args.num_frames,
        sample_rate=args.sample_rate
    )
    
    print(f"\n[OK] Done!")