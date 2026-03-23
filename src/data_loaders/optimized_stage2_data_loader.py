#!/usr/bin/env python3
"""
Optimized data loader for Stage2 behavior classification
Fast loading with temporal feature caching and class balancing
"""

import torch
from torch.utils.data import DataLoader
import time
import sys
import os

# Add datasets path for cross-directory imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
datasets_path = os.path.join(project_root, 'datasets')
sys.path.append(datasets_path)

from datasets.geometric_stage2_dataset import GeometricStage2Dataset


def create_fast_stage2_data_loaders(data_path, batch_size=64, num_workers=2, 
                                   history_length=5, use_temporal=True, use_scene_context=True,
                                   use_hog_features=True):
    """
    创建优化的Stage2数据加载器
    
    Args:
        data_path: 数据集路径
        batch_size: 批次大小
        num_workers: 工作进程数
        history_length: 时序历史长度
        use_temporal: 是否使用时序特征
        use_scene_context: 是否使用场景上下文
        use_hog_features: 是否使用HoG特征
        
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Creating optimized Stage2 data loaders...")
    start_time = time.time()
    
    # 创建数据集（禁用时序以加速初始化）
    print("Loading train dataset...")
    train_dataset = GeometricStage2Dataset(
        data_path, split='train', history_length=history_length,
        use_temporal=False,  # 初始化时禁用时序
        use_scene_context=use_scene_context,
        use_oversampling=True,
        use_hog_features=use_hog_features
    )
    
    print("Loading validation dataset...")
    val_dataset = GeometricStage2Dataset(
        data_path, split='val', history_length=history_length,
        use_temporal=False,
        use_scene_context=use_scene_context,
        use_oversampling=False,
        use_hog_features=use_hog_features
    )
    
    print("Loading test dataset...")
    test_dataset = GeometricStage2Dataset(
        data_path, split='test', history_length=history_length,
        use_temporal=False,
        use_scene_context=use_scene_context,
        use_oversampling=False,
        use_hog_features=use_hog_features
    )
    
    # 如果需要时序特征，再启用
    if use_temporal:
        print("Enabling temporal features...")
        
        # 使用优化的时序缓冲区初始化
        from .optimized_temporal_buffer import update_temporal_buffer_optimized
        from .temporal_buffer import TemporalPairManager
        
        # 为训练集启用时序
        train_dataset.use_temporal = True
        train_dataset.temporal_manager = TemporalPairManager(history_length=history_length)
        update_temporal_buffer_optimized(train_dataset)
        
        # 为验证集启用时序
        val_dataset.use_temporal = True
        val_dataset.temporal_manager = TemporalPairManager(history_length=history_length)
        update_temporal_buffer_optimized(val_dataset)
        
        # 为测试集启用时序
        test_dataset.use_temporal = True
        test_dataset.temporal_manager = TemporalPairManager(history_length=history_length)
        update_temporal_buffer_optimized(test_dataset)
    else:
        print("Temporal features disabled - using static geometric features only")
    
    # 创建数据加载器
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
    
    # 打印数据集统计
    print(f"\nOptimized Stage2 Dataset Statistics:")
    print(f"Temporal features enabled: {use_temporal}")
    print(f"Scene context enabled: {use_scene_context}")
    
    train_dist = train_dataset.get_stage2_class_distribution()
    val_dist = val_dataset.get_stage2_class_distribution()
    test_dist = test_dataset.get_stage2_class_distribution()
    
    print(f"\nTrain distribution:")
    for class_id, count in train_dist['class_counts'].items():
        class_name = train_dist['class_names'][class_id]
        percentage = 100 * count / train_dist['total']
        print(f"  Class {class_id} ({class_name}): {count:,} ({percentage:.1f}%)")
    
    print(f"\nVal distribution:")
    for class_id, count in val_dist['class_counts'].items():
        class_name = val_dist['class_names'][class_id]
        percentage = 100 * count / val_dist['total']
        print(f"  Class {class_id} ({class_name}): {count:,} ({percentage:.1f}%)")
    
    print(f"\nTest distribution:")
    for class_id, count in test_dist['class_counts'].items():
        class_name = test_dist['class_names'][class_id]
        percentage = 100 * count / test_dist['total']
        print(f"  Class {class_id} ({class_name}): {count:,} ({percentage:.1f}%)")
    
    total_time = time.time() - start_time
    print(f"\nStage2 data loader creation completed in {total_time:.1f}s")
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches") 
    print(f"Test loader: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def verify_stage2_data_integrity(data_loader, num_samples=5):
    """
    验证Stage2数据加载的完整性
    
    Args:
        data_loader: Stage2数据加载器
        num_samples: 验证样本数量
    """
    print(f"Verifying Stage2 data integrity with {num_samples} samples...")
    
    sample_count = 0
    for batch_idx, batch in enumerate(data_loader):
        if sample_count >= num_samples:
            break
            
        # 检查数据格式
        features = batch['features']
        labels = batch['stage2_label']
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Features range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  Label values: {torch.unique(labels).tolist()}")
        
        # 检查单个样本
        if 'original_interaction' in batch:
            print(f"  Sample interactions: {batch['original_interaction'][:3]}")
        
        # 验证特征维度
        assert features.shape[1] == 16, f"Expected 16D features, got {features.shape[1]}D"
        
        # 验证标签范围
        assert torch.all(labels >= 0) and torch.all(labels < 5), f"Labels out of range [0, 4]: {torch.unique(labels)}"
        
        sample_count += 1
    
    print("✓ Stage2 data integrity verification passed!")


if __name__ == '__main__':
    # 测试优化的Stage2数据加载器
    print("Testing optimized Stage2 data loader...")
    
    data_path = r'C:\assignment\master programme\final\baseline\classificationnet\dataset'
    
    # 快速版本（无时序）
    print("\n=== Testing fast version (no temporal) ===")
    start_time = time.time()
    train_loader, val_loader, test_loader = create_fast_stage2_data_loaders(
        data_path=data_path,
        batch_size=32,
        num_workers=0,  # 使用0避免多进程问题
        use_temporal=False,
        use_scene_context=True
    )
    fast_time = time.time() - start_time
    
    # 验证数据完整性
    verify_stage2_data_integrity(train_loader, num_samples=2)
    
    print(f"\nFast version completed in {fast_time:.1f}s")
    
    # 完整版本（有时序）
    print("\n=== Testing full version (with temporal) ===")
    start_time = time.time()
    train_loader_full, val_loader_full, test_loader_full = create_fast_stage2_data_loaders(
        data_path=data_path,
        batch_size=32,
        num_workers=0,
        use_temporal=True,
        use_scene_context=True
    )
    full_time = time.time() - start_time
    
    # 验证时序数据完整性
    verify_stage2_data_integrity(train_loader_full, num_samples=2)
    
    print(f"\nFull version completed in {full_time:.1f}s")
    print(f"Speed improvement: {full_time/fast_time:.1f}x faster without temporal features")
    
    print("\nStage2 data loader test completed!")