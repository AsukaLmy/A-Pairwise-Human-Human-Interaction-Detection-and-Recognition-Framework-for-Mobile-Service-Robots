#!/usr/bin/env python3
"""
Optimized temporal buffer initialization for large datasets
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

def _frame_sort_key(frame_id):
    """Helper function for sorting frame IDs (pickle-friendly)"""
    parts = frame_id.split('_')
    if len(parts) >= 2:
        try:
            # Handle format like "scene_name_000001"
            return (parts[0], int(parts[-1]))  # Use last part as frame number
        except ValueError:
            # Fallback to string sorting
            return frame_id
    return frame_id

def update_temporal_buffer_optimized(dataset):
    """
    Optimized version of temporal buffer update
    
    Key optimizations:
    1. Avoid sorting entire sample list
    2. Cache geometric features computation
    3. Process in batches to save memory
    4. Skip temporal for initial testing
    """
    
    if not dataset.temporal_manager:
        print("ERROR: Temporal manager not initialized - cannot update buffer")
        print("Make sure dataset was created with use_temporal=True")
        return
    
    print("Starting optimized temporal buffer update...")
    start_time = time.time()
    
    # Pre-compute all geometric features in batches
    print("Pre-computing geometric features...")
    feature_cache = {}
    
    batch_size = 1000  # Process in batches to save memory
    total_samples = len(dataset.samples)
    
    for i in range(0, total_samples, batch_size):
        batch_samples = dataset.samples[i:i+batch_size]
        
        for sample in batch_samples:
            # Create cache key
            key = (
                tuple(sample['person_A_box']),
                tuple(sample['person_B_box']),
                sample['frame_id']
            )
            
            if key not in feature_cache:
                from ..features.geometric_features import extract_geometric_features
                
                geometric_features = extract_geometric_features(
                    torch.tensor(sample['person_A_box']),
                    torch.tensor(sample['person_B_box']),
                    3760, 480  # Use JRDB standard dimensions
                )
                feature_cache[key] = geometric_features
        
        if (i // batch_size + 1) % 10 == 0:
            processed = min(i + batch_size, total_samples)
            print(f"  Cached features for {processed}/{total_samples} samples ({100*processed/total_samples:.1f}%)")
    
    print(f"Feature caching completed in {time.time() - start_time:.1f}s")
    
    # Group samples by frame (using cached data)
    print("Grouping samples by frame...")
    frames = {}
    
    for sample in dataset.samples:
        frame_id = sample['frame_id']
        if frame_id not in frames:
            frames[frame_id] = []
        frames[frame_id].append(sample)
    
    print(f"Grouped into {len(frames)} frames")
    
    # Process frames in chronological order
    print("Processing temporal sequences...")
    frame_ids = sorted(frames.keys(), key=_frame_sort_key)
    
    processed_frames = 0
    for frame_id in frame_ids:
        frame_samples = frames[frame_id]
        frame_data = []
        
        for sample in frame_samples:
            # Use cached geometric features
            cache_key = (
                tuple(sample['person_A_box']),
                tuple(sample['person_B_box']),
                sample['frame_id']
            )
            
            geometric_features = feature_cache[cache_key]
            
            frame_data.append({
                'person_A_id': sample['person_A_id'],
                'person_B_id': sample['person_B_id'],
                'geometric_features': geometric_features
            })
        
        # Update temporal manager
        if frame_data:
            dataset.temporal_manager.update_frame(frame_data, frame_id)
        
        processed_frames += 1
        if processed_frames % 1000 == 0:
            print(f"  Processed {processed_frames}/{len(frame_ids)} frames ({100*processed_frames/len(frame_ids):.1f}%)")
    
    total_time = time.time() - start_time
    print(f"Temporal buffer update completed in {total_time:.1f}s")
    print(f"  Feature cache size: {len(feature_cache):,} entries")
    print(f"  Processed {len(frame_ids):,} frames")


def create_fast_geometric_data_loaders(data_path, batch_size=32, num_workers=4,
                                     use_temporal=False, use_scene_context=True, history_length=5,
                                     feature_mode='legacy'):
    """
    Fast version of data loader creation with optimizations

    Args:
        feature_mode: Feature extraction mode ('legacy', 'both', 'opticalflow_only', 'bboxposition_only')
    """
    from datasets.geometric_dataset import GeometricDualPersonDataset
    from torch.utils.data import DataLoader

    print("Creating optimized geometric data loaders...")

    # Create datasets with correct temporal setting from the start
    print(f"Creating datasets with use_temporal={use_temporal}, feature_mode={feature_mode}")

    train_dataset = GeometricDualPersonDataset(
        data_path, split='train', history_length=history_length,
        use_temporal=use_temporal,  # Use the correct setting
        use_scene_context=use_scene_context,
        feature_mode=feature_mode
    )

    val_dataset = GeometricDualPersonDataset(
        data_path, split='val', history_length=history_length,
        use_temporal=use_temporal,
        use_scene_context=use_scene_context,
        feature_mode=feature_mode
    )

    test_dataset = GeometricDualPersonDataset(
        data_path, split='test', history_length=history_length,
        use_temporal=use_temporal,
        use_scene_context=use_scene_context,
        feature_mode=feature_mode
    )
    
    # Update temporal buffers if temporal modeling is enabled
    if use_temporal:
        print("Enabling temporal modeling (this may take a while)...")
        
        # Update buffers with optimized method
        update_temporal_buffer_optimized(train_dataset)
        update_temporal_buffer_optimized(val_dataset)
        update_temporal_buffer_optimized(test_dataset)
    else:
        print("Temporal modeling disabled - using static geometric features only")
    
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
    
    # Print dataset statistics
    train_dist = train_dataset.get_class_distribution()
    val_dist = val_dataset.get_class_distribution()
    test_dist = test_dataset.get_class_distribution()
    
    print(f"\nOptimized Dataset Statistics:")
    print(f"Train: {train_dist}")
    print(f"Val: {val_dist}")
    print(f"Test: {test_dist}")
    print(f"Temporal features enabled: {use_temporal}")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the optimized version
    data_path = r'C:\assignment\master programme\final\baseline\classificationnet\dataset'
    
    print("Testing optimized data loader creation...")
    
    # Fast version without temporal
    start_time = time.time()
    train_loader, val_loader, test_loader = create_fast_geometric_data_loaders(
        data_path=data_path,
        batch_size=32,
        num_workers=0,
        use_temporal=False,  # Start with False for speed
        use_scene_context=True
    )
    
    print(f"Fast data loader creation completed in {time.time() - start_time:.1f}s")
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")