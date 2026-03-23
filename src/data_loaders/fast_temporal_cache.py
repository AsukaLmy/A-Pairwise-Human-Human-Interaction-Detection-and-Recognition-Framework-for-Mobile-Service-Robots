#!/usr/bin/env python3
"""
Fast temporal feature caching system
"""

import torch
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm
from collections import defaultdict
import time

# Add datasets path for cross-directory imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
datasets_path = os.path.join(project_root, 'datasets')
sys.path.append(datasets_path)

class FastTemporalCache:
    """
    Pre-compute and cache all temporal features to disk for fast loading
    """
    
    def __init__(self, cache_dir='temporal_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def build_cache_from_dataset(self, dataset, split_name):
        """
        Pre-compute all temporal features for a dataset
        """
        print(f"Building temporal cache for {split_name} split...")
        
        cache_file = os.path.join(self.cache_dir, f'{split_name}_temporal_cache.pkl')
        
        # Check if cache already exists
        if os.path.exists(cache_file):
            print(f"Cache already exists: {cache_file}")
            return cache_file
        
        # Group samples by (person_A_id, person_B_id) for efficiency
        pair_samples = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            pair_key = (sample['person_A_id'], sample['person_B_id'])
            pair_samples[pair_key].append((idx, sample))
        
        print(f"Found {len(pair_samples)} unique person pairs")
        
        # Build temporal cache
        temporal_cache = {}
        
        for pair_key, pair_sample_list in tqdm(pair_samples.items(), desc="Processing pairs"):
            person_A_id, person_B_id = pair_key
            
            # Sort by frame for temporal ordering
            pair_sample_list.sort(key=lambda x: self._extract_frame_number(x[1]['frame_id']))
            
            # Compute temporal features for each sample in this pair
            pair_temporal_data = self._compute_pair_temporal_features(pair_sample_list)
            
            # Store in cache
            for idx, temporal_features in pair_temporal_data.items():
                temporal_cache[idx] = temporal_features
        
        # Save cache to disk
        with open(cache_file, 'wb') as f:
            pickle.dump(temporal_cache, f)
        
        print(f"Temporal cache saved: {cache_file}")
        print(f"Cached features for {len(temporal_cache)} samples")
        return cache_file
    
    def _compute_pair_temporal_features(self, pair_sample_list):
        """
        Compute temporal features for all samples of a person pair
        """
        from ..features.geometric_features import extract_geometric_features, extract_causal_motion_features
        
        results = {}
        
        for i, (sample_idx, sample) in enumerate(pair_sample_list):
            # Extract geometric features for current sample
            person_A_box = torch.tensor(sample['person_A_box'], dtype=torch.float32)
            person_B_box = torch.tensor(sample['person_B_box'], dtype=torch.float32)
            
            current_geometric = extract_geometric_features(person_A_box, person_B_box, 3760, 480)
            
            # Build history from previous samples (causal)
            history_geometric = []
            for j in range(max(0, i-5), i):  # Last 5 frames
                prev_sample = pair_sample_list[j][1]
                prev_A_box = torch.tensor(prev_sample['person_A_box'], dtype=torch.float32)
                prev_B_box = torch.tensor(prev_sample['person_B_box'], dtype=torch.float32)
                prev_geometric = extract_geometric_features(prev_A_box, prev_B_box, 3760, 480)
                history_geometric.append(prev_geometric)
            
            # Pad history if needed
            while len(history_geometric) < 5:
                history_geometric.insert(0, torch.zeros(7))
            
            history_tensor = torch.stack(history_geometric)
            
            # Compute motion features
            if len(history_geometric) >= 2:
                motion_features = extract_causal_motion_features(history_tensor.unsqueeze(0)).squeeze(0)
            else:
                motion_features = torch.zeros(4)
            
            # Store features
            results[sample_idx] = {
                'history_geometric': history_tensor,
                'motion_features': motion_features,
                'has_sufficient_history': len(history_geometric) >= 2
            }
        
        return results
    
    def load_cache(self, split_name):
        """
        Load pre-computed temporal cache
        """
        cache_file = os.path.join(self.cache_dir, f'{split_name}_temporal_cache.pkl')
        
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Temporal cache not found: {cache_file}")
        
        with open(cache_file, 'rb') as f:
            temporal_cache = pickle.load(f)
        
        print(f"Loaded temporal cache: {len(temporal_cache)} samples from {cache_file}")
        return temporal_cache
    
    def _extract_frame_number(self, frame_id):
        """Extract numeric frame number"""
        try:
            parts = frame_id.split('_')
            return int(parts[-1])
        except:
            return 0


class CachedGeometricDataset:
    """
    Dataset wrapper that uses pre-computed temporal cache
    """
    
    def __init__(self, base_dataset, temporal_cache=None):
        self.base_dataset = base_dataset
        self.temporal_cache = temporal_cache or {}
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        # Get base sample
        result = self.base_dataset[idx]
        
        # Override temporal features with cached data
        if idx in self.temporal_cache:
            cached_features = self.temporal_cache[idx]
            result.update(cached_features)
        else:
            # Fallback to zero features
            result['history_geometric'] = torch.zeros(5, 7)
            result['motion_features'] = torch.zeros(4)
            result['has_sufficient_history'] = False
            
        return result


def create_fast_cached_data_loaders(data_path, batch_size=32, num_workers=4, 
                                  use_temporal=True, use_scene_context=True, history_length=5):
    """
    Create data loaders with pre-computed temporal cache
    """
    from datasets.geometric_dataset import GeometricDualPersonDataset
    from torch.utils.data import DataLoader
    
    print("Creating fast cached data loaders...")
    
    # Initialize cache system
    cache_system = FastTemporalCache()
    
    # Create base datasets (without temporal)
    train_dataset = GeometricDualPersonDataset(
        data_path, split='train', history_length=history_length,
        use_temporal=False, use_scene_context=use_scene_context
    )
    
    val_dataset = GeometricDualPersonDataset(
        data_path, split='val', history_length=history_length,
        use_temporal=False, use_scene_context=use_scene_context
    )
    
    test_dataset = GeometricDualPersonDataset(
        data_path, split='test', history_length=history_length,
        use_temporal=False, use_scene_context=use_scene_context
    )
    
    if use_temporal:
        print("Building/loading temporal caches...")
        
        # Build or load caches
        try:
            train_cache = cache_system.load_cache('train')
            val_cache = cache_system.load_cache('val')
            test_cache = cache_system.load_cache('test')
        except FileNotFoundError:
            print("Cache not found, building new cache...")
            cache_system.build_cache_from_dataset(train_dataset, 'train')
            cache_system.build_cache_from_dataset(val_dataset, 'val')
            cache_system.build_cache_from_dataset(test_dataset, 'test')
            
            train_cache = cache_system.load_cache('train')
            val_cache = cache_system.load_cache('val')
            test_cache = cache_system.load_cache('test')
        
        # Wrap datasets with cache
        train_dataset = CachedGeometricDataset(train_dataset, train_cache)
        val_dataset = CachedGeometricDataset(val_dataset, val_cache)
        test_dataset = CachedGeometricDataset(test_dataset, test_cache)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Fast cached data loaders created!")
    print(f"Temporal features enabled: {use_temporal}")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the cached system
    data_path = r'C:\assignment\master programme\final\baseline\classificationnet\dataset'
    
    print("Testing fast cached data loader...")
    start_time = time.time()
    
    train_loader, val_loader, test_loader = create_fast_cached_data_loaders(
        data_path=data_path,
        batch_size=32,
        num_workers=0,
        use_temporal=True,
        use_scene_context=True
    )
    
    print(f"Data loader creation: {time.time() - start_time:.1f}s")
    
    # Test loading a few batches
    print("Testing batch loading...")
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= 3:  # Test 3 batches
            break
        print(f"Batch {i}: {batch['geometric_features'].shape}")
    
    print(f"Batch loading test: {time.time() - start_time:.1f}s")