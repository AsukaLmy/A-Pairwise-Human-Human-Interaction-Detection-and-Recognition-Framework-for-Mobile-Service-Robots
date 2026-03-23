#!/usr/bin/env python3
"""
Lazy loading temporal features - only compute when actually needed
"""

import torch
import numpy as np
from collections import defaultdict
import time
import sys
import os

# Add datasets path for cross-directory imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
datasets_path = os.path.join(project_root, 'datasets')
sys.path.append(datasets_path)


class LazyTemporalManager:
    """
    Lazy temporal manager that computes features on-demand
    """
    
    def __init__(self, dataset, history_length=5):
        self.dataset = dataset
        self.history_length = history_length
        
        # Build frame index for fast lookup
        print("Building frame index for lazy temporal loading...")
        self.frame_index = self._build_frame_index()
        print(f"Frame index built: {len(self.frame_index)} frames indexed")
        
        # Cache for computed features
        self._feature_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _build_frame_index(self):
        """Build index: frame_id -> list of sample indices"""
        frame_index = defaultdict(list)
        
        for idx, sample in enumerate(self.dataset.samples):
            frame_id = sample['frame_id']
            frame_index[frame_id].append(idx)
        
        return dict(frame_index)
    
    def get_temporal_features_lazy(self, sample_idx):
        """
        Lazily compute temporal features for a sample
        """
        # Check cache first
        if sample_idx in self._feature_cache:
            self._cache_hits += 1
            return self._feature_cache[sample_idx]
        
        self._cache_misses += 1
        
        # Get current sample
        current_sample = self.dataset.samples[sample_idx]
        person_A_id = current_sample['person_A_id']
        person_B_id = current_sample['person_B_id']
        current_frame_id = current_sample['frame_id']
        
        # Find historical samples for this person pair
        historical_samples = self._find_historical_samples(
            person_A_id, person_B_id, current_frame_id
        )
        
        # Compute temporal features
        temporal_features = self._compute_temporal_features(
            historical_samples, current_sample
        )
        
        # Cache the result
        self._feature_cache[sample_idx] = temporal_features
        
        # Limit cache size to prevent memory issues
        if len(self._feature_cache) > 10000:  # Keep last 10k features
            oldest_keys = list(self._feature_cache.keys())[:-5000]
            for key in oldest_keys:
                del self._feature_cache[key]
        
        return temporal_features
    
    def _find_historical_samples(self, person_A_id, person_B_id, current_frame_id):
        """
        Find historical samples for the person pair
        """
        current_frame_num = self._extract_frame_number(current_frame_id)
        historical_samples = []
        
        # Search in nearby frames (efficient for temporal locality)
        for frame_offset in range(1, self.history_length + 5):  # Look back a bit more
            past_frame_num = current_frame_num - frame_offset
            past_frame_id = self._construct_frame_id(current_frame_id, past_frame_num)
            
            if past_frame_id in self.frame_index:
                # Check samples in this frame
                for sample_idx in self.frame_index[past_frame_id]:
                    sample = self.dataset.samples[sample_idx]
                    
                    if (sample['person_A_id'] == person_A_id and 
                        sample['person_B_id'] == person_B_id):
                        historical_samples.append(sample)
                        break
            
            if len(historical_samples) >= self.history_length:
                break
        
        # Sort by frame number (most recent first)
        historical_samples.sort(
            key=lambda x: self._extract_frame_number(x['frame_id']), 
            reverse=True
        )
        
        return historical_samples[:self.history_length]
    
    def _compute_temporal_features(self, historical_samples, current_sample):
        """
        Compute temporal features from historical samples
        """
        from ..features.geometric_features import extract_geometric_features, extract_causal_motion_features
        
        # Extract geometric features for historical samples
        history_geometric = []
        
        for hist_sample in reversed(historical_samples):  # Chronological order
            person_A_box = torch.tensor(hist_sample['person_A_box'], dtype=torch.float32)
            person_B_box = torch.tensor(hist_sample['person_B_box'], dtype=torch.float32)
            
            geometric_features = extract_geometric_features(person_A_box, person_B_box, 3760, 480)
            history_geometric.append(geometric_features)
        
        # Pad if not enough history
        while len(history_geometric) < self.history_length:
            history_geometric.insert(0, torch.zeros(7))
        
        history_tensor = torch.stack(history_geometric)
        
        # Compute motion features
        if len(historical_samples) >= 2:
            motion_features = extract_causal_motion_features(history_tensor.unsqueeze(0)).squeeze(0)
            has_sufficient_history = True
        else:
            motion_features = torch.zeros(4)
            has_sufficient_history = False
        
        return {
            'history_geometric': history_tensor,
            'motion_features': motion_features,
            'has_sufficient_history': has_sufficient_history
        }
    
    def _extract_frame_number(self, frame_id):
        """Extract numeric frame number"""
        try:
            parts = frame_id.split('_')
            return int(parts[-1])
        except:
            return 0
    
    def _construct_frame_id(self, reference_frame_id, frame_number):
        """Construct frame ID from reference and number"""
        try:
            parts = reference_frame_id.split('_')
            parts[-1] = f"{frame_number:06d}"
            return '_'.join(parts)
        except:
            return f"frame_{frame_number:06d}"
    
    def get_cache_stats(self):
        """Get cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        
        return {
            'cache_size': len(self._feature_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }


class LazyGeometricDataset:
    """
    Dataset with lazy temporal feature computation
    """
    
    def __init__(self, base_dataset, use_temporal=True):
        self.base_dataset = base_dataset
        self.use_temporal = use_temporal
        
        if use_temporal:
            self.temporal_manager = LazyTemporalManager(base_dataset)
        else:
            self.temporal_manager = None
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base sample (without temporal processing)
        result = self.base_dataset.__getitem__(idx)
        
        # Override temporal features with lazy computation
        if self.use_temporal and self.temporal_manager:
            temporal_features = self.temporal_manager.get_temporal_features_lazy(idx)
            result.update(temporal_features)
        else:
            result['history_geometric'] = torch.zeros(5, 7)
            result['motion_features'] = torch.zeros(4)
            result['has_sufficient_history'] = False
        
        return result
    
    def get_class_distribution(self):
        """Delegate to base dataset"""
        return self.base_dataset.get_class_distribution()


def create_lazy_data_loaders(data_path, batch_size=32, num_workers=2, 
                           use_temporal=True, use_scene_context=True, history_length=5):
    """
    Create data loaders with lazy temporal computation
    """
    from datasets.geometric_dataset import GeometricDualPersonDataset
    from torch.utils.data import DataLoader
    
    print("Creating lazy temporal data loaders...")
    
    # Create base datasets (disable their temporal processing)
    train_base = GeometricDualPersonDataset(
        data_path, split='train', history_length=history_length, use_temporal=False, use_scene_context=use_scene_context
    )
    
    val_base = GeometricDualPersonDataset(
        data_path, split='val', history_length=history_length, use_temporal=False, use_scene_context=use_scene_context
    )
    
    test_base = GeometricDualPersonDataset(
        data_path, split='test', history_length=history_length, use_temporal=False, use_scene_context=use_scene_context
    )
    
    # Wrap with lazy datasets
    train_dataset = LazyGeometricDataset(train_base, use_temporal)
    val_dataset = LazyGeometricDataset(val_base, use_temporal)
    test_dataset = LazyGeometricDataset(test_base, use_temporal)
    
    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    print(f"Lazy data loaders created!")
    print(f"Temporal features: {use_temporal}")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test lazy loading
    data_path = r'C:\assignment\master programme\final\baseline\classificationnet\dataset'
    
    print("Testing lazy temporal data loaders...")
    start_time = time.time()
    
    train_loader, val_loader, test_loader = create_lazy_data_loaders(
        data_path=data_path,
        batch_size=16,
        num_workers=0,  # Single process for testing
        use_temporal=True
    )
    
    print(f"Lazy loader creation: {time.time() - start_time:.1f}s")
    
    # Test loading batches
    print("Testing batch loading with lazy computation...")
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        print(f"Batch {i}: {batch['geometric_features'].shape}")
    
    print(f"3 batch loading time: {time.time() - start_time:.1f}s")
    
    # Check cache statistics
    if hasattr(train_loader.dataset, 'temporal_manager'):
        stats = train_loader.dataset.temporal_manager.get_cache_stats()
        print(f"Cache stats: {stats}")