# grpo_reward_function_v2.py
"""
Stage 3 GRPO Reward Function
Supports:
1. Custom vocabulary import from training data
2. Cached YOLO detection results
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set
import re
import spacy

# Load custom vocabulary if available
try:
    from vocabulary_config import (
        DRIVING_OBJECTS, 
        ACTION_VERBS,
        DESCRIPTIVE_ADJECTIVES,
        TEMPORAL_KEYWORDS
    )
    print("[OK] Loaded custom vocabulary from vocabulary_config.py")
except ImportError:
    print("[WARNING] vocabulary_config.py not found, using default vocabulary")
    DRIVING_OBJECTS = {
        'vehicles': ['car', 'truck', 'bus', 'suv', 'vehicle', 'motorcycle'],
        'traffic_control': ['traffic light', 'stop sign', 'traffic signal', 'signal'],
        'pedestrians': ['pedestrian', 'person', 'people'],
        'road_elements': ['road', 'street', 'intersection', 'crosswalk', 'lane'],
        'landmarks': ['building', 'storefront', 'sign', 'fence']
    }
    
    ACTION_VERBS = [
        'moving', 'stopped', 'turning', 'crossing', 'passing', 
        'approaching', 'waiting', 'driving', 'proceeding'
    ]
    
    DESCRIPTIVE_ADJECTIVES = [
        'illuminated', 'lit', 'glowing', 'visible', 'marked',
        'nighttime', 'dark', 'bright', 'dimly', 'sparse'
    ]
    
    TEMPORAL_KEYWORDS = [
        'initially', 'then', 'as', 'after', 'before', 'when',
        'during', 'subsequently', 'eventually', 'continues', 'throughout'
    ]

# Load YOLO cache loader if available
try:
    from preprocess_yolo_cache import YOLOCacheLoader
    YOLO_CACHE_AVAILABLE = True
except ImportError:
    YOLO_CACHE_AVAILABLE = False
    print("[WARNING] YOLOCacheLoader not available, will use live detection")


class BDD100KRewardFunctionV2:
    """
    Multi-dimensional reward function for BDD100K driving scenarios.
    
    Features:
    1. Support for cached YOLO detection results
    2. Custom vocabulary support
    3. Fast inference
    """
    
    def __init__(self, 
                 yolo_cache_path: str = None,
                 use_detector: bool = True,
                 use_nlp: bool = True,
                 device: str = 'cuda'):
        """
        Initialize reward function.
        
        Args:
            yolo_cache_path: Path to YOLO cache file (.pkl or .json)
            use_detector: Whether to load YOLO detector if cache unavailable
            use_nlp: Whether to load spaCy NLP model
            device: Compute device (cuda or cpu)
        """
        self.device = device
        self.yolo_cache = None
        self.detector = None
        
        # Try to load YOLO cache first
        if yolo_cache_path and YOLO_CACHE_AVAILABLE:
            try:
                print(f"Loading YOLO cache from {yolo_cache_path}...")
                self.yolo_cache = YOLOCacheLoader(yolo_cache_path)
                print("Using cached YOLO results (fast mode)")
            except Exception as e:
                print(f"Failed to load cache: {e}")
                print("Falling back to live detection...")
        
        # Load YOLO detector if cache unavailable
        if self.yolo_cache is None and use_detector:
            print("Loading YOLO detector...")
            from ultralytics import YOLO
            self.detector = YOLO('yolov8x.pt')
            print("YOLO detector loaded")
        
        # Load NLP model
        if use_nlp:
            print("Loading spaCy NLP model...")
            self.nlp = spacy.load("en_core_web_sm")
            print("NLP model loaded")
        else:
            self.nlp = None
        
        # Print vocabulary statistics
        total_vocab = sum(len(words) for words in DRIVING_OBJECTS.values())
        print(f"Vocabulary loaded:")
        print(f"Objects: {total_vocab} words in {len(DRIVING_OBJECTS)} categories")
        print(f"Action verbs: {len(ACTION_VERBS)}")
        print(f"Adjectives: {len(DESCRIPTIVE_ADJECTIVES)}")
        print(f"Temporal keywords: {len(TEMPORAL_KEYWORDS)}")
    
    
    def compute_reward(self, 
                      video_name: str,
                      generated_caption: str,
                      ground_truth_caption: str,
                      video_frames: torch.Tensor = None,
                      metadata: Dict = None) -> float:
        """
        Compute total reward for generated caption.
        
        Args:
            video_name: Video filename (e.g., '00721168-56efa5c2.mov')
            generated_caption: Model-generated description
            ground_truth_caption: Ground truth annotation
            video_frames: Video frames (optional, required for live detection)
            metadata: Additional metadata
        
        Returns:
            reward: Score between 0.0 and 1.0
        """
        rewards = {}
        
        # Object accuracy (40% weight)
        rewards['object_accuracy'] = self._compute_object_accuracy_cached(
            video_name, generated_caption, ground_truth_caption, video_frames
        )
        
        # Temporal coherence (25% weight)
        rewards['temporal_coherence'] = self._compute_temporal_coherence(
            generated_caption, ground_truth_caption
        )
        
        # Detail richness (20% weight)
        rewards['detail_richness'] = self._compute_detail_richness(
            generated_caption, ground_truth_caption
        )
        
        # Hallucination penalty (15% weight)
        rewards['hallucination_penalty'] = self._compute_hallucination_penalty_cached(
            video_name, generated_caption, video_frames
        )
        
        # Combine weighted rewards
        weights = {
            'object_accuracy': 0.40,
            'temporal_coherence': 0.25,
            'detail_richness': 0.20,
            'hallucination_penalty': 0.15
        }
        
        final_reward = sum(rewards[k] * weights[k] for k in rewards.keys())
        
        # Store reward breakdown in metadata
        if metadata is not None:
            metadata['reward_breakdown'] = rewards
        
        return final_reward
    
    
    
    def _compute_object_accuracy_cached(self, 
                                       video_name: str,
                                       generated_caption: str,
                                       ground_truth_caption: str,
                                       video_frames: torch.Tensor = None) -> float:
        """
        Verify object accuracy using cached or live detection.
        """
        # Get detected objects (prefer cache)
        if self.yolo_cache is not None:
            detected_objects = self.yolo_cache.get_detected_objects(video_name)
        elif self.detector is not None and video_frames is not None:
            detected_objects = self._detect_objects_in_video(video_frames)
        else:
            # Fallback: text-only comparison
            return self._compare_objects_text_only(generated_caption, ground_truth_caption)
        
        # Extract objects from generated caption
        mentioned_objects = self._extract_objects_from_text(generated_caption)
        
        if len(mentioned_objects) == 0:
            return 0.5
        
        # Compute precision
        correct_mentions = 0
        for obj in mentioned_objects:
            if self._object_exists_in_detections(obj, detected_objects):
                correct_mentions += 1
        
        precision = correct_mentions / len(mentioned_objects)
        
        # Compute recall
        gt_objects = self._extract_objects_from_text(ground_truth_caption)
        recall = self._compute_recall(mentioned_objects, gt_objects)
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return f1
    
    
    def _detect_objects_in_video(self, video_frames: torch.Tensor) -> Set[str]:
        """
        Detect objects in video frames using YOLO (live detection fallback).
        """
        # Sample key frames
        T = video_frames.shape[0]
        sample_indices = np.linspace(0, T-1, min(8, T), dtype=int)
        sampled_frames = video_frames[sample_indices]
        
        # Convert format
        frames_np = sampled_frames.permute(0, 2, 3, 1).cpu().numpy()
        frames_np = (frames_np * 255).astype(np.uint8)
        
        # YOLO detection
        results = self.detector(frames_np, verbose=False, conf=0.3)
        
        # Collect class names
        detected_classes = set()
        for result in results:
            if result.boxes is not None:
                for cls_id in result.boxes.cls:
                    cls_name = self.detector.names[int(cls_id)]
                    detected_classes.add(cls_name)
        
        return detected_classes
    
    
    def _extract_objects_from_text(self, text: str) -> List[str]:
        """
        Extract objects from text using custom vocabulary.
        """
        if self.nlp is None:
            return self._extract_objects_regex(text)
        
        doc = self.nlp(text.lower())
        objects = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            text_chunk = chunk.text.strip()
            # Check against custom vocabulary
            for category, keywords in DRIVING_OBJECTS.items():
                for keyword in keywords:
                    if keyword in text_chunk:
                        objects.append(text_chunk)
                        break
        
        return list(set(objects))
    
    
    def _extract_objects_regex(self, text: str) -> List[str]:
        """
        Extract objects using regex (fallback if NLP unavailable).
        """
        text_lower = text.lower()
        objects = []
        
        for category, keywords in DRIVING_OBJECTS.items():
            for keyword in keywords:
                if re.search(r'\b' + keyword + r'\b', text_lower):
                    objects.append(keyword)
        
        return objects
    
    
    def _object_exists_in_detections(self, 
                                    mentioned_obj: str, 
                                    detected_objects: Set[str]) -> bool:
        """
        Check if mentioned object exists in detection results.
        """
        mentioned_obj_lower = mentioned_obj.lower()
        
        # YOLO class mapping
        mapping = {
            'suv': 'car', 'sedan': 'car', 'vehicle': 'car',
            'traffic light': 'traffic light', 'traffic signal': 'traffic light',
            'pedestrian': 'person', 'people': 'person',
        }
        
        # Direct match
        for det_obj in detected_objects:
            if det_obj in mentioned_obj_lower:
                return True
        
        # Mapped match
        for text_key, yolo_key in mapping.items():
            if text_key in mentioned_obj_lower and yolo_key in detected_objects:
                return True
        
        return False
    
    
    def _compare_objects_text_only(self, generated: str, ground_truth: str) -> float:
        """
        Fallback: compare objects using text only.
        """
        gen_objects = set(self._extract_objects_regex(generated))
        gt_objects = set(self._extract_objects_regex(ground_truth))
        
        if len(gt_objects) == 0:
            return 1.0 if len(gen_objects) == 0 else 0.5
        
        # Jaccard similarity
        intersection = len(gen_objects & gt_objects)
        union = len(gen_objects | gt_objects)
        
        return intersection / union if union > 0 else 0.0
    
    
    def _compute_recall(self, mentioned: List[str], ground_truth: List[str]) -> float:
        """
        Compute recall score.
        """
        if len(ground_truth) == 0:
            return 1.0
        
        mentioned_set = set(mentioned)
        gt_set = set(ground_truth)
        
        recalled = 0
        for gt_obj in gt_set:
            for mention in mentioned_set:
                if gt_obj in mention or mention in gt_obj:
                    recalled += 1
                    break
        
        return recalled / len(gt_set)
    
    
    
    def _compute_temporal_coherence(self, 
                                    generated: str,
                                    ground_truth: str) -> float:
        """
        Evaluate temporal coherence using custom temporal keywords.
        """
        gen_lower = generated.lower()
        gt_lower = ground_truth.lower()
        
        # Temporal keyword score
        gen_temporal_count = sum(1 for kw in TEMPORAL_KEYWORDS if kw in gen_lower)
        gt_temporal_count = sum(1 for kw in TEMPORAL_KEYWORDS if kw in gt_lower)
        
        if gt_temporal_count > 0:
            temporal_keyword_score = min(gen_temporal_count / gt_temporal_count, 1.0)
        else:
            temporal_keyword_score = 1.0 if gen_temporal_count == 0 else 0.5
        
        # Action verb score using custom verb list
        gen_action_count = sum(1 for verb in ACTION_VERBS if verb in gen_lower)
        gt_action_count = sum(1 for verb in ACTION_VERBS if verb in gt_lower)
        
        if gt_action_count > 0:
            action_score = min(gen_action_count / gt_action_count, 1.0)
        else:
            action_score = 1.0 if gen_action_count == 0 else 0.5
        
        # Paragraph structure score
        gen_sentences = generated.split('.')
        gt_sentences = ground_truth.split('.')
        structure_score = min(len(gen_sentences) / max(len(gt_sentences), 1), 1.0)
        
        # Weighted average
        final_score = (
            0.4 * temporal_keyword_score +
            0.4 * action_score +
            0.2 * structure_score
        )
        
        return final_score
    
    
    
    def _compute_detail_richness(self, 
                                 generated: str,
                                 ground_truth: str) -> float:
        """
        Evaluate detail richness using custom adjective list.
        """
        gen_words = len(generated.split())
        gt_words = len(ground_truth.split())
        
        # Length ratio score
        if gt_words > 0:
            length_ratio = gen_words / gt_words
            length_score = 1.0 / (1.0 + np.exp(-5 * (length_ratio - 0.5)))
        else:
            length_score = 0.5
        
        # Descriptive adjective score using custom list
        gen_desc_count = sum(1 for adj in DESCRIPTIVE_ADJECTIVES if adj in generated.lower())
        gt_desc_count = sum(1 for adj in DESCRIPTIVE_ADJECTIVES if adj in ground_truth.lower())
        
        if gt_desc_count > 0:
            desc_score = min(gen_desc_count / gt_desc_count, 1.0)
        else:
            desc_score = 1.0 if gen_desc_count == 0 else 0.5
        
        # Spatial relationship words
        spatial_words = ['left', 'right', 'ahead', 'behind', 'front', 'corner', 
                        'intersection', 'roadside', 'across', 'along']
        
        gen_spatial_count = sum(1 for word in spatial_words if word in generated.lower())
        gt_spatial_count = sum(1 for word in spatial_words if word in ground_truth.lower())
        
        if gt_spatial_count > 0:
            spatial_score = min(gen_spatial_count / gt_spatial_count, 1.0)
        else:
            spatial_score = 1.0 if gen_spatial_count == 0 else 0.5
        
        # Weighted average
        final_score = (
            0.5 * length_score +
            0.3 * desc_score +
            0.2 * spatial_score
        )
        
        return final_score
    
    
    
    def _compute_hallucination_penalty_cached(self, 
                                             video_name: str,
                                             generated: str,
                                             video_frames: torch.Tensor = None) -> float:
        """
        Detect hallucinations using cached or live detection.
        """
        # Get detected objects (prefer cache)
        if self.yolo_cache is not None:
            detected_objects = self.yolo_cache.get_detected_objects(video_name)
        elif self.detector is not None and video_frames is not None:
            detected_objects = self._detect_objects_in_video(video_frames)
        else:
            # Cannot verify, return neutral score
            return 0.75
        
        # Extract mentioned objects
        mentioned_objects = self._extract_objects_from_text(generated)
        
        if len(mentioned_objects) == 0:
            return 1.0
        
        # Check for hallucinated objects
        hallucinated_count = 0
        for obj in mentioned_objects:
            if not self._object_exists_in_detections(obj, detected_objects):
                hallucinated_count += 1
        
        # Compute penalty
        hallucination_rate = hallucinated_count / len(mentioned_objects)
        penalty = 1.0 - hallucination_rate
        
        return max(penalty, 0.0)


# Simplified reward function (text-only, no YOLO)

class SimplifiedRewardFunctionV2:
    """
    Simplified version: text-only comparison without YOLO detection.
    Uses custom vocabulary for scoring.
    """
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
    
    
    def compute_reward(self, 
                      video_name: str,
                      generated_caption: str,
                      ground_truth_caption: str) -> float:
        """
        Text-based reward computation using custom vocabulary.
        """
        # Object matching (40%)
        gen_objects = self._extract_objects(generated_caption)
        gt_objects = self._extract_objects(ground_truth_caption)
        object_score = self._jaccard_similarity(gen_objects, gt_objects)
        
        # Temporal word matching (30%)
        gen_temporal = [kw for kw in TEMPORAL_KEYWORDS if kw in generated_caption.lower()]
        gt_temporal = [kw for kw in TEMPORAL_KEYWORDS if kw in ground_truth_caption.lower()]
        temporal_score = self._jaccard_similarity(gen_temporal, gt_temporal)
        
        # Length ratio (30%)
        gen_len = len(generated_caption.split())
        gt_len = len(ground_truth_caption.split())
        length_ratio = min(gen_len / max(gt_len, 1), 1.0)
        length_score = 1.0 / (1.0 + np.exp(-5 * (length_ratio - 0.5)))
        
        final_reward = (
            0.4 * object_score +
            0.3 * temporal_score +
            0.3 * length_score
        )
        
        return final_reward
    
    
    def _extract_objects(self, text: str) -> set:
        """
        Extract objects using custom vocabulary.
        """
        objects = set()
        text_lower = text.lower()
        
        for category, keywords in DRIVING_OBJECTS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    objects.add(keyword)
        
        return objects
    
    
    def _jaccard_similarity(self, set1, set2) -> float:
        """
        Compute Jaccard similarity between two sets.
        """
        set1 = set(set1)
        set2 = set(set2)
        
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


# Binary reward wrapper

class BinaryRewardWrapper:
    """
    Wrap continuous reward as binary (0 or 1) using threshold.
    """
    
    def __init__(self, base_reward_fn, threshold=0.7):
        self.base_fn = base_reward_fn
        self.threshold = threshold
    
    
    def compute_reward(self, *args, **kwargs) -> float:
        """
        Compute binary reward based on threshold.
        """
        continuous_reward = self.base_fn.compute_reward(*args, **kwargs)
        return 1.0 if continuous_reward >= self.threshold else 0.0


# Usage examples

if __name__ == "__main__":
    
    # Example 1: Using YOLO cache (recommended)
    print("=== Using YOLO cache ===")
    reward_fn = BDD100KRewardFunctionV2(
        yolo_cache_path='./yolo_cache.pkl',
        use_detector=False,
        use_nlp=True
    )
    
    video_name = "00721168-56efa5c2.mov"
    generated = "A red SUV is moving forward at an intersection."
    ground_truth = "The video captures a nighttime journey..."
    
    reward = reward_fn.compute_reward(video_name, generated, ground_truth)
    print(f"Reward: {reward:.3f}")
    
    
    # Example 2: Simplified version (no YOLO)
    print("\n=== Simplified version (no YOLO) ===")
    simple_fn = SimplifiedRewardFunctionV2()
    reward_simple = simple_fn.compute_reward(video_name, generated, ground_truth)
    print(f"Reward: {reward_simple:.3f}")