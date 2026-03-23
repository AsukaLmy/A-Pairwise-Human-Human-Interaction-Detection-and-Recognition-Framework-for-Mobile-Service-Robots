#!/usr/bin/env python3
"""
Optimized data loader with efficient temporal feature handling
"""

import torch
from torch.utils.data import DataLoader
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pickle
import os
import sys

# Add datasets path for cross-directory imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
datasets_path = os.path.join(project_root, 'datasets')
sys.path.append(datasets_path)


class MultiprocessTemporalBuffer:
    """
    Process-safe temporal buffer that can be shared across workers
    """
    
    def __init__(self, cache_file=None):
        self.cache_file = cache_file
        self._cache = None
        
    def _load_cache(self):
        """Lazy load cache when needed"""
        if self._cache is None and self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self._cache = pickle.load(f)
        return self._cache or {}
    
    def get_temporal_features(self, sample_idx):
        """Get temporal features for sample"""
        cache = self._load_cache()
        return cache.get(sample_idx, {
            'history_geometric': torch.zeros(5, 7),
            'motion_features': torch.zeros(4),
            'has_sufficient_history': False
        })


class OptimizedGeometricDataset:
    """
    Optimized dataset with minimal temporal computation during training
    """
    
    def __init__(self, base_dataset, temporal_cache_file=None):
        self.base_dataset = base_dataset
        self.temporal_buffer = MultiprocessTemporalBuffer(temporal_cache_file)
        
        # Pre-load essential data into memory for fast access
        print("Pre-loading essential dataset information...")
        self.samples = base_dataset.samples
        self.scene_data = getattr(base_dataset, 'scene_data', {})
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """Optimized sample retrieval"""
        sample = self.samples[idx]
        
        # Basic data extraction (fast)
        person_A_box = torch.tensor(sample['person_A_box'], dtype=torch.float32)
        person_B_box = torch.tensor(sample['person_B_box'], dtype=torch.float32)
        has_interaction = torch.tensor(sample['has_interaction'], dtype=torch.long)
        frame_id = sample['frame_id']
        
        # Fast geometric feature extraction
        from ..features.geometric_features import extract_geometric_features
        geometric_features = extract_geometric_features(person_A_box, person_B_box, 3760, 480)
        
        # Prepare result
        result = {
            'geometric_features': geometric_features,
            'stage1_label': has_interaction,
            'person_A_id': sample['person_A_id'],
            'person_B_id': sample['person_B_id'],
            'frame_id': frame_id,
            'person_A_box': person_A_box,
            'person_B_box': person_B_box
        }
        
        # Scene context (fast lookup)
        if frame_id in self.scene_data:
            result['scene_context'] = self.scene_data[frame_id].get('scene_context', torch.tensor([1.0]))
        else:
            result['scene_context'] = torch.tensor([1.0])
        
        # Temporal features (from cache)
        temporal_features = self.temporal_buffer.get_temporal_features(idx)
        result.update(temporal_features)
        
        return result


def custom_collate_fn(batch):
    """
    Custom collate function for faster batching
    """
    # Pre-allocate tensors for better performance
    batch_size = len(batch)
    
    result = {}
    
    # Stack geometric features efficiently
    geometric_features = torch.stack([item['geometric_features'] for item in batch])
    result['geometric_features'] = geometric_features
    
    # Stack labels
    stage1_labels = torch.stack([item['stage1_label'] for item in batch])
    result['stage1_label'] = stage1_labels
    
    # Stack temporal features
    history_geometric = torch.stack([item['history_geometric'] for item in batch])
    result['history_geometric'] = history_geometric
    
    motion_features = torch.stack([item['motion_features'] for item in batch])
    result['motion_features'] = motion_features
    
    # Scene context
    scene_contexts = torch.stack([item['scene_context'] for item in batch])
    result['scene_context'] = scene_contexts
    
    # Metadata (lists)
    result['person_A_id'] = [item['person_A_id'] for item in batch]
    result['person_B_id'] = [item['person_B_id'] for item in batch]
    result['frame_id'] = [item['frame_id'] for item in batch]
    
    # Boxes
    person_A_boxes = torch.stack([item['person_A_box'] for item in batch])
    person_B_boxes = torch.stack([item['person_B_box'] for item in batch])
    result['person_A_box'] = person_A_boxes
    result['person_B_box'] = person_B_boxes
    
    return result


def create_optimized_data_loaders(data_path, batch_size=32, num_workers=None, 
                                use_temporal=True, use_scene_context=True, history_length=5):
    """
    Create highly optimized data loaders
    """
    from datasets.geometric_dataset import GeometricDualPersonDataset
    from .fast_temporal_cache import FastTemporalCache
    
    print("Creating optimized data loaders...")
    
    # Auto-detect optimal number of workers
    if num_workers is None:
        num_workers = min(4, mp.cpu_count())
    
    # Create base datasets
    train_dataset = GeometricDualPersonDataset(
        data_path, split='train', history_length=history_length, use_temporal=False, use_scene_context=use_scene_context
    )
    
    val_dataset = GeometricDualPersonDataset(
        data_path, split='val', history_length=history_length, use_temporal=False, use_scene_context=use_scene_context
    )
    
    test_dataset = GeometricDualPersonDataset(
        data_path, split='test', history_length=history_length, use_temporal=False, use_scene_context=use_scene_context
    )
    
    # Setup temporal caching if needed
    train_cache_file = None
    val_cache_file = None 
    test_cache_file = None
    
    if use_temporal:
        print("Setting up temporal feature caching...")
        cache_system = FastTemporalCache()
        
        try:
            # Try to load existing caches
            train_cache_file = os.path.join(cache_system.cache_dir, 'train_temporal_cache.pkl')
            val_cache_file = os.path.join(cache_system.cache_dir, 'val_temporal_cache.pkl')
            test_cache_file = os.path.join(cache_system.cache_dir, 'test_temporal_cache.pkl')
            
            if not all(os.path.exists(f) for f in [train_cache_file, val_cache_file, test_cache_file]):
                print("Building temporal caches (this will take time but only happens once)...")
                cache_system.build_cache_from_dataset(train_dataset, 'train')
                cache_system.build_cache_from_dataset(val_dataset, 'val')
                cache_system.build_cache_from_dataset(test_dataset, 'test')
        except Exception as e:
            print(f"Temporal cache setup failed: {e}")
            use_temporal = False
    
    # Create optimized dataset wrappers
    opt_train_dataset = OptimizedGeometricDataset(train_dataset, train_cache_file)
    opt_val_dataset = OptimizedGeometricDataset(val_dataset, val_cache_file)
    opt_test_dataset = OptimizedGeometricDataset(test_dataset, test_cache_file)
    
    # Create optimized data loaders
    train_loader = DataLoader(
        opt_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=num_workers > 0,  # Keep workers alive
        prefetch_factor=2 if num_workers > 0 else 2  # Prefetch batches
    )
    
    val_loader = DataLoader(
        opt_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    test_loader = DataLoader(
        opt_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    print(f"Optimized data loaders created!")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")
    print(f"Temporal features: {use_temporal}")
    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


class ProgressiveTemporalLoader:
    """
    Progressive temporal feature loader for very large datasets
    """
    
    def __init__(self, base_loader, temporal_cache_files):
        self.base_loader = base_loader
        self.temporal_cache_files = temporal_cache_files
        self._cache = {}
        self._loaded_cache_portions = set()
    
    def _load_cache_portion(self, portion_id):
        """Load a portion of the temporal cache"""
        if portion_id not in self._loaded_cache_portions:
            # Implement partial cache loading logic here
            pass
    
    def __iter__(self):
        """Iterate with progressive cache loading"""
        for batch_idx, batch in enumerate(self.base_loader):
            # Load necessary cache portions for this batch
            self._ensure_cache_loaded_for_batch(batch)
            
            # Update batch with temporal features
            batch = self._add_temporal_features(batch)
            
            yield batch
    
    def _ensure_cache_loaded_for_batch(self, batch):
        """Ensure temporal cache is loaded for current batch"""
        # Implementation for progressive loading
        pass
    
    def _add_temporal_features(self, batch):
        """Add temporal features to batch"""
        # Implementation for adding temporal features
        return batch


if __name__ == '__main__':
    # Test optimized data loaders
    data_path = r'C:\assignment\master programme\final\baseline\classificationnet\dataset'
    
    print("Testing optimized data loaders...")
    import time
    
    start_time = time.time()
    
    train_loader, val_loader, test_loader = create_optimized_data_loaders(
        data_path=data_path,
        batch_size=32,
        num_workers=2,
        use_temporal=True,
        use_scene_context=True
    )
    
    print(f"Loader creation time: {time.time() - start_time:.1f}s")
    
    # Test batch loading speed
    print("Testing batch loading speed...")
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= 5:  # Test 5 batches
            break
        print(f"Batch {i}: geometric={batch['geometric_features'].shape}, "
              f"temporal={batch['history_geometric'].shape}")
    
    print(f"5 batch loading time: {time.time() - start_time:.1f}s")