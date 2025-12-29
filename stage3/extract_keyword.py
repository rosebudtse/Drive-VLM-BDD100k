# extract_vocabulary.py
"""
Extract high-frequency verbs and nouns from BDD100K training data
for optimizing GRPO reward function vocabulary.
"""

import json
import spacy
from collections import Counter
from typing import Dict, List, Tuple
import re

class VocabularyExtractor:
    """Extract high-frequency vocabulary from caption data."""
    
    def __init__(self):
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
    
    
    def extract_from_jsonl(self, 
                          jsonl_path: str,
                          min_freq: int = 5) -> Dict[str, List[str]]:
        """
        从JSONL文件中提取高频词汇
        
        Args:
            jsonl_path: train_ready.json路径
            min_freq: 最小出现频率
        
        Returns:
            {
                'nouns': ['vehicle', 'intersection', ...],
                'verbs': ['moving', 'approaching', ...],
                'adjectives': ['illuminated', 'visible', ...]
            }
        """
        print(f"📖 Reading captions from {jsonl_path}...")
        
        # 读取所有captions
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_captions = [item['a'] for item in data]
        print(f"   Loaded {len(all_captions)} captions")
        
        # Count word frequencies
        noun_counter = Counter()
        verb_counter = Counter()
        adj_counter = Counter()
        
        print("Extracting vocabulary...")
        for i, caption in enumerate(all_captions):
            if (i + 1) % 50 == 0:
                print(f"   Processed {i+1}/{len(all_captions)} captions...")
            
            doc = self.nlp(caption.lower())
            
            for token in doc:
                # Filter stopwords and punctuation
                if token.is_stop or token.is_punct or len(token.text) < 3:
                    continue
                
                # Collect nouns (driving scene related)
                if token.pos_ in ['NOUN', 'PROPN']:
                    noun_counter[token.lemma_] += 1
                
                # Collect verbs (action related)
                elif token.pos_ == 'VERB':
                    verb_counter[token.lemma_] += 1
                
                # Collect adjectives (descriptive)
                elif token.pos_ == 'ADJ':
                    adj_counter[token.lemma_] += 1
        
        # Filter high-frequency words
        high_freq_nouns = [word for word, count in noun_counter.most_common() 
                          if count >= min_freq]
        high_freq_verbs = [word for word, count in verb_counter.most_common() 
                          if count >= min_freq]
        high_freq_adjs = [word for word, count in adj_counter.most_common() 
                         if count >= min_freq]
        
        print(f"\nExtraction completed:")
        print(f"   Nouns: {len(high_freq_nouns)} (freq >= {min_freq})")
        print(f"   Verbs: {len(high_freq_verbs)} (freq >= {min_freq})")
        print(f"   Adjectives: {len(high_freq_adjs)} (freq >= {min_freq})")
        
        return {
            'nouns': high_freq_nouns,
            'verbs': high_freq_verbs,
            'adjectives': high_freq_adjs,
            'noun_freq': dict(noun_counter.most_common(100)),
            'verb_freq': dict(verb_counter.most_common(100)),
            'adj_freq': dict(adj_counter.most_common(100))
        }
    
    
    def categorize_nouns(self, nouns: List[str]) -> Dict[str, List[str]]:
        """Categorize nouns into driving scene categories.
        
        Returns:
            Dictionary mapping categories to word lists
        """
        # Predefined category keywords
        category_keywords = {
            'vehicles': ['car', 'vehicle', 'truck', 'bus', 'suv', 'sedan', 'motorcycle', 'bicycle'],
            'traffic_control': ['light', 'signal', 'sign', 'traffic', 'stop', 'yield'],
            'pedestrians': ['person', 'pedestrian', 'people', 'cyclist', 'walker'],
            'road_elements': ['road', 'street', 'lane', 'intersection', 'crosswalk', 'sidewalk', 'highway', 'curb'],
            'landmarks': ['building', 'store', 'shop', 'fence', 'house', 'structure', 'storefront'],
            'environment': ['tree', 'sky', 'cloud', 'weather', 'rain', 'fog'],
            'lighting': ['light', 'lamp', 'illumination', 'glow', 'headlight', 'streetlight']
        }
        
        categorized = {cat: [] for cat in category_keywords.keys()}
        uncategorized = []
        
        for noun in nouns:
            assigned = False
            for category, keywords in category_keywords.items():
                if any(kw in noun for kw in keywords):
                    categorized[category].append(noun)
                    assigned = True
                    break
            
            if not assigned:
                uncategorized.append(noun)
        
        # Keep top 20 uncategorized words
        categorized['other'] = uncategorized[:20]
        
        return categorized
    
    
    def generate_config_file(self, 
                           vocabulary: Dict,
                           output_path: str = './vocabulary_config.py'):
        """Generate importable vocabulary configuration file."""
        categorized_nouns = self.categorize_nouns(vocabulary['nouns'])
        
        config_content = '''# vocabulary_config.py
"""
Automatically extracted high-frequency vocabulary from BDD100K training data
for GRPO reward function.
"""

# Driving scene object categories
DRIVING_OBJECTS = {
'''
        
        # 添加名词类别
        for category, words in categorized_nouns.items():
            if len(words) > 0:
                words_str = ', '.join([f"'{w}'" for w in words[:30]])  # 限制每类30个
                config_content += f"    '{category}': [{words_str}],\n"
        
        config_content += '''}

# Action verbs
ACTION_VERBS = [
'''
        
        # Filter common verbs
        filtered_verbs = [v for v in vocabulary['verbs'][:50]
                         if v not in ['be', 'have', 'do', 'say', 'get', 'make', 'go', 'know', 'take', 'see']]
        
        for i, verb in enumerate(filtered_verbs):
            if i % 5 == 0:
                config_content += '\n    '
            config_content += f"'{verb}', "
        
        config_content += '''\n]

# Descriptive adjectives
DESCRIPTIVE_ADJECTIVES = [
'''
        
        # Filter common adjectives
        filtered_adjs = [a for a in vocabulary['adjectives'][:40]
                        if a not in ['good', 'new', 'first', 'last', 'long', 'great', 'little', 'old', 'different']]
        
        for i, adj in enumerate(filtered_adjs):
            if i % 5 == 0:
                config_content += '\n    '
            config_content += f"'{adj}', "
        
        config_content += '''\n]

# Temporal keywords (extracted from training data)
TEMPORAL_KEYWORDS = [
    'initially', 'then', 'as', 'after', 'before', 'when',
    'during', 'subsequently', 'eventually', 'continues', 'throughout',
    'later', 'soon', 'meanwhile', 'progressively', 'gradually'
]

# Frequency statistics (Top 20 for debugging)
TOP_NOUNS = {
'''
        
        for noun, count in list(vocabulary['noun_freq'].items())[:20]:
            config_content += f"    '{noun}': {count},\n"
        
        config_content += '''}

TOP_VERBS = {
'''
        
        for verb, count in list(vocabulary['verb_freq'].items())[:20]:
            config_content += f"    '{verb}': {count},\n"
        
        config_content += '''}

TOP_ADJECTIVES = {
'''
        
        for adj, count in list(vocabulary['adj_freq'].items())[:20]:
            config_content += f"    '{adj}': {count},\n"
        
        config_content += '''}
'''
        
        # Save file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"\nVocabulary config saved to {output_path}")
        print(f"   Usage: from vocabulary_config import DRIVING_OBJECTS, ACTION_VERBS")
    
    
    def print_summary(self, vocabulary: Dict):
        """Print vocabulary statistics summary."""
        print("\n" + "="*60)
        print("High-frequency Vocabulary Summary")
        print("="*60)
        
        print("\nTop 30 Nouns (Driving Scene Objects):")
        for word, count in list(vocabulary['noun_freq'].items())[:30]:
            print(f"   {word:20s} {count:4d} occurrences")
        
        print("\nTop 30 Verbs (Action Descriptions):")
        for word, count in list(vocabulary['verb_freq'].items())[:30]:
            print(f"   {word:20s} {count:4d} occurrences")
        
        print("\nTop 20 Adjectives (Descriptive Words):")
        for word, count in list(vocabulary['adj_freq'].items())[:20]:
            print(f"   {word:20s} {count:4d} occurrences")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract vocabulary from BDD100K captions")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to train_ready.json")
    parser.add_argument("--output", type=str, default="./vocabulary_config.py",
                       help="Output vocabulary config file")
    parser.add_argument("--min_freq", type=int, default=5,
                       help="Minimum word frequency")
    parser.add_argument("--show_summary", action="store_true",
                       help="Print detailed summary")
    args = parser.parse_args()
    
    # Extract vocabulary
    extractor = VocabularyExtractor()
    vocabulary = extractor.extract_from_jsonl(args.input, min_freq=args.min_freq)
    
    # Generate configuration file
    extractor.generate_config_file(vocabulary, output_path=args.output)
    
    # Print summary
    if args.show_summary:
        extractor.print_summary(vocabulary)
    
    print("\nVocabulary extraction completed!")
    print(f"   Next steps:")
    print(f"   1. Review {args.output}")
    print(f"   2. Import in grpo_reward_function.py:")
    print(f"      from vocabulary_config import DRIVING_OBJECTS, ACTION_VERBS")