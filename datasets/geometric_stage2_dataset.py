#!/usr/bin/env python3
"""
Stage2 geometric behavior classification dataset
Extends Stage1 geometric features with enhanced temporal features for 5-class behavior classification
"""

import torch
import numpy as np
from collections import defaultdict, deque
import sys
import os

# Add project paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
datasets_path = os.path.join(project_root, 'datasets')
src_path = os.path.join(project_root, 'src')
sys.path.extend([datasets_path, src_path])

from geometric_dataset import GeometricDualPersonDataset
from ..src.features.geometric_features import extract_geometric_features, extract_causal_motion_features, compute_scene_context
from ..src.features.hog_features import extract_joint_hog_features
from PIL import Image


def _get_frame_id_sort_key(sample):
    """Helper function to get frame_id for sorting (pickle-friendly)"""
    return sample['frame_id']


class Stage2LabelMapper:
    """Stage2的3分类标签映射器 - 仅保留基础行为类别"""
    
    def __init__(self):
        self.label_mapping = {
            # Walking Together (0) 
            'walking together': 0,
            
            # Standing Together (1) 
            'standing together': 1,
            
            # Sitting Together (2) 
            'sitting together': 2,
        }
        
        # 其他所有交互类型将被过滤掉，不参与训练
        self.excluded_interactions = {
            'conversation', 'going upstairs together', 'going downstairs together',
            'moving together',
            'looking at robot together', 'looking at sth together', 
            'walking toward each other', 'eating together', 
            'going downstairs together', 'bending together',
            'holding sth together', 'cycling together',
            'interaction with door together', 'hugging',
            'looking into sth together', 'shaking hand',
            'waving hand together'
        }
        
        self.class_names = [
            'Walking Together',
            'Standing Together', 
            'Sitting Together' 
        ]
        
        # 基于3分类的类别权重
        self.class_weights = {
            0: 1.0,    # Walking Together
            1: 1.4,    # Standing Together
            2: 3.3     # Sitting Together
        }
        
        # 目标类别分布 (更平衡的3分类)
        self.target_distribution = {
            0: 0.33,  # Walking Together
            1: 0.33,  # Standing Together
            2: 0.34   # Sitting Together
        }
    
    def map_label(self, original_interaction):
        """将原始交互标签映射为3分类标签，过滤掉不支持的类型"""
        return self.label_mapping.get(original_interaction, None)  # 返回None表示该交互类型被过滤
    
    def is_valid_interaction(self, original_interaction):
        """检查交互类型是否为有效的3分类之一"""
        return original_interaction in self.label_mapping
    
    def get_mapped_class_name(self, class_id):
        """获取映射后的类别名"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return "Unknown"


def compute_velocity_features(history_boxes, frame_interval=1):
    """
    计算基于5帧历史的速度和方向特征
    
    Args:
        history_boxes: [5, 4] 历史边界框 [x, y, w, h]
        frame_interval: 帧间隔（通常为1）
    
    Returns:
        velocities: [4, 2] 速度向量
        centers: [5, 2] 中心点坐标
    """
    if len(history_boxes) < 2:
        return torch.zeros(0, 2), torch.zeros(len(history_boxes), 2)
    
    # 提取中心点坐标
    centers = []
    for i in range(len(history_boxes)):
        x, y, w, h = history_boxes[i]
        center_x = x + w/2
        center_y = y + h/2
        centers.append([center_x, center_y])
    
    centers = torch.tensor(centers, dtype=torch.float32)  # [5, 2]
    
    # 计算速度 (位置差分)
    velocities = []
    for i in range(1, len(centers)):
        vx = centers[i][0] - centers[i-1][0]  # 水平速度
        vy = centers[i][1] - centers[i-1][1]  # 垂直速度
        velocities.append([vx, vy])
    
    velocities = torch.tensor(velocities, dtype=torch.float32)  # [4, 2]
    return velocities, centers


def extract_enhanced_motion_features(person_A_history, person_B_history):
    """
    提取增强的运动时序特征 (8维)
    
    Args:
        person_A_history: [5, 4] Person A的历史边界框
        person_B_history: [5, 4] Person B的历史边界框
    
    Returns:
        torch.Tensor: [8] 增强运动特征
    """
    # 计算两人的速度和位置
    vel_A, centers_A = compute_velocity_features(person_A_history)
    vel_B, centers_B = compute_velocity_features(person_B_history)
    
    if len(vel_A) == 0 or len(vel_B) == 0:
        return torch.zeros(8, dtype=torch.float32)
    
    features = []
    
    # 1. velocity_correlation - 速度相关性
    vel_A_flat = vel_A.view(-1)  # [8] (4帧 x 2方向)
    vel_B_flat = vel_B.view(-1)  # [8]
    
    if torch.std(vel_A_flat) > 1e-6 and torch.std(vel_B_flat) > 1e-6:
        velocity_corr = torch.corrcoef(torch.stack([vel_A_flat, vel_B_flat]))[0, 1]
        velocity_corr = torch.nan_to_num(velocity_corr, 0.0).float()  # 转换为float32
    else:
        velocity_corr = torch.tensor(0.0, dtype=torch.float32)
    features.append(velocity_corr)
    
    # 2. direction_alignment - 方向对齐度
    direction_alignments = []
    for i in range(len(vel_A)):
        # 计算单位方向向量
        dir_A = vel_A[i] / (torch.norm(vel_A[i]) + 1e-6)
        dir_B = vel_B[i] / (torch.norm(vel_B[i]) + 1e-6)
        
        # 方向一致性 (余弦相似度)
        alignment = torch.dot(dir_A, dir_B)
        direction_alignments.append(alignment)
    
    avg_direction_alignment = torch.mean(torch.tensor(direction_alignments, dtype=torch.float32))
    features.append(avg_direction_alignment)
    
    # 3. acceleration_sync - 加速度同步性
    if len(vel_A) >= 2:
        acc_A = vel_A[1:] - vel_A[:-1]  # [3, 2]
        acc_B = vel_B[1:] - vel_B[:-1]  # [3, 2]
        
        # 加速度大小的相关性
        acc_mag_A = torch.norm(acc_A, dim=1)  # [3]
        acc_mag_B = torch.norm(acc_B, dim=1)  # [3]
        
        if torch.std(acc_mag_A) > 1e-6 and torch.std(acc_mag_B) > 1e-6:
            acc_sync = torch.corrcoef(torch.stack([acc_mag_A, acc_mag_B]))[0, 1]
            acc_sync = torch.nan_to_num(acc_sync, 0.0).float()  # 转换为float32
        else:
            acc_sync = torch.tensor(0.0, dtype=torch.float32)
    else:
        acc_sync = torch.tensor(0.0, dtype=torch.float32)
    features.append(acc_sync)
    
    # 4. stopping_frequency - 停顿频率
    stop_threshold = 5.0  # 像素/帧
    
    stops_A = torch.norm(vel_A, dim=1) < stop_threshold
    stops_B = torch.norm(vel_B, dim=1) < stop_threshold
    
    # 同时停顿的比例
    simultaneous_stops = (stops_A & stops_B).float().mean()
    features.append(simultaneous_stops)
    
    # 5. trajectory_similarity - 轨迹相似度
    if len(centers_A) >= 2 and len(centers_B) >= 2:
        # 轨迹向量化 (相对于起始点)
        traj_A = centers_A - centers_A[0]  # [5, 2]
        traj_B = centers_B - centers_B[0]  # [5, 2]
        
        # 轨迹长度归一化
        traj_A_norm = torch.norm(traj_A[-1])
        traj_B_norm = torch.norm(traj_B[-1])
        
        if traj_A_norm > 1e-6:
            traj_A = traj_A / traj_A_norm
        if traj_B_norm > 1e-6:
            traj_B = traj_B / traj_B_norm
        
        # 轨迹相似度 (平均距离的倒数)
        trajectory_dist = torch.mean(torch.norm(traj_A - traj_B, dim=1))
        trajectory_similarity = 1.0 / (trajectory_dist + 0.1)
    else:
        trajectory_similarity = torch.tensor(0.0, dtype=torch.float32)
    features.append(trajectory_similarity)
    
    # 6. relative_speed - 相对速度大小
    relative_speeds = []
    for i in range(len(vel_A)):
        rel_vel = vel_A[i] - vel_B[i]
        relative_speeds.append(torch.norm(rel_vel))
    
    avg_relative_speed = torch.mean(torch.tensor(relative_speeds, dtype=torch.float32))
    features.append(avg_relative_speed)
    
    # 7. movement_consistency - 运动一致性
    vel_A_consistency = 1.0 - torch.std(torch.norm(vel_A, dim=1))
    vel_B_consistency = 1.0 - torch.std(torch.norm(vel_B, dim=1))
    avg_consistency = (vel_A_consistency + vel_B_consistency) / 2
    features.append(torch.clamp(avg_consistency, 0.0, 1.0))
    
    # 8. proximity_change_rate - 距离变化率
    distances = []
    for i in range(len(centers_A)):
        if i < len(centers_B):
            dist = torch.norm(centers_A[i] - centers_B[i])
            distances.append(dist)
    
    if len(distances) >= 2:
        distance_changes = []
        for i in range(1, len(distances)):
            distance_changes.append(distances[i] - distances[i-1])
        
        proximity_change_rate = torch.mean(torch.tensor(distance_changes, dtype=torch.float32))
    else:
        proximity_change_rate = torch.tensor(0.0, dtype=torch.float32)
    features.append(proximity_change_rate)
    
    return torch.tensor(features, dtype=torch.float32)


class GeometricStage2Dataset(GeometricDualPersonDataset):
    """
    Stage2行为分类数据集
    继承自Stage1几何数据集，添加5分类行为标签和增强时序特征
    """
    
    def __init__(self, data_path, split='train', history_length=5, 
                 use_temporal=True, use_scene_context=True, use_oversampling=True,
                 use_hog_features=True):
        """
        Args:
            data_path: 数据集路径
            split: 数据集划分 ('train', 'val', 'test')
            history_length: 时序历史长度
            use_temporal: 是否使用时序特征
            use_scene_context: 是否使用场景上下文
            use_oversampling: 是否使用过采样平衡类别
            use_hog_features: 是否使用HoG特征
        """
        # 初始化标签映射器
        self.label_mapper = Stage2LabelMapper()
        self.use_oversampling = use_oversampling
        self.use_hog_features = use_hog_features
        
        # 调用父类初始化
        super().__init__(data_path, split, history_length, use_temporal, use_scene_context)
        
        # Stage2特有的处理
        self._filter_interaction_samples()
        self._apply_stage2_labels()
        
        if self.use_oversampling and split == 'train':
            self._apply_oversampling()
        
        print(f"GeometricStage2Dataset loaded: {len(self.samples)} samples ({split})")
        self._print_stage2_statistics()
    
    def _filter_interaction_samples(self):
        """过滤出只有交互的样本（has_interaction=1）"""
        interaction_samples = []
        for sample in self.samples:
            if sample['has_interaction'] == 1:
                interaction_samples.append(sample)
        
        print(f"Filtered interaction samples: {len(interaction_samples)}/{len(self.samples)} "
              f"({100*len(interaction_samples)/len(self.samples):.1f}%)")
        
        self.samples = interaction_samples
    
    def _apply_stage2_labels(self):
        """应用Stage2的3分类标签并过滤无效样本"""
        valid_samples = []
        
        for sample in self.samples:
            original_interaction = sample.get('interaction_labels', {})
            if isinstance(original_interaction, dict) and len(original_interaction) > 0:
                # 取第一个交互类型
                interaction_type = list(original_interaction.keys())[0]
            else:
                interaction_type = 'unknown'
            
            # 检查是否为有效的3分类标签
            if self.label_mapper.is_valid_interaction(interaction_type):
                # 映射到3分类
                stage2_label = self.label_mapper.map_label(interaction_type)
                sample['stage2_label'] = stage2_label
                sample['original_interaction'] = interaction_type
                valid_samples.append(sample)
            # 否则丢弃该样本
        
        print(f"Filtered valid interaction types: {len(valid_samples)}/{len(self.samples)} "
              f"({100*len(valid_samples)/len(self.samples):.1f}%)")
        
        self.samples = valid_samples
    
    def _apply_oversampling(self):
        """应用过采样策略平衡类别"""
        # 统计各类样本数量
        class_samples = {i: [] for i in range(3)}  # 3分类
        for sample in self.samples:
            label = sample['stage2_label']
            class_samples[label].append(sample)
        
        print("Original class distribution:")
        for class_id, samples in class_samples.items():
            print(f"  Class {class_id} ({self.label_mapper.get_mapped_class_name(class_id)}): {len(samples)}")
        
        total_samples = len(self.samples)
        balanced_samples = []
        
        for class_id, target_ratio in self.label_mapper.target_distribution.items():
            current_samples = class_samples[class_id]
            target_count = int(total_samples * target_ratio)
            
            if len(current_samples) == 0:
                continue
                
            if len(current_samples) < target_count:
                # 过采样：重复采样达到目标数量
                indices = np.random.choice(
                    len(current_samples), 
                    size=target_count, 
                    replace=True
                )
                oversampled = [current_samples[i] for i in indices]
                balanced_samples.extend(oversampled)
                print(f"  Oversampled class {class_id}: {len(current_samples)} -> {len(oversampled)}")
            else:
                # 下采样：随机选择目标数量
                indices = np.random.choice(
                    len(current_samples),
                    size=target_count,
                    replace=False
                )
                undersampled = [current_samples[i] for i in indices]
                balanced_samples.extend(undersampled)
                print(f"  Undersampled class {class_id}: {len(current_samples)} -> {len(undersampled)}")
        
        self.samples = balanced_samples
        np.random.shuffle(self.samples)  # 打乱顺序
    
    def __getitem__(self, idx):
        """
        获取Stage2样本，包含增强特征向量 (7几何 + 64HoG + 可选时序)
        """
        # 获取基础样本信息
        sample = self.samples[idx]
        
        # Stage2标签
        stage2_label = torch.tensor(sample['stage2_label'], dtype=torch.long)
        
        # 基础几何特征 (7维)
        frame_id = sample['frame_id']
        person_A_id = sample['person_A_id']
        person_B_id = sample['person_B_id']
        person_A_box = torch.tensor(sample['person_A_box'], dtype=torch.float32)
        person_B_box = torch.tensor(sample['person_B_box'], dtype=torch.float32)
        
        # 提取几何特征
        geometric_features = extract_geometric_features(
            person_A_box, person_B_box, 3760, 480
        )  # [7]
        
        # 基础运动特征 (4维)
        if self.use_temporal and self.temporal_manager:
            temporal_features = self.temporal_manager.get_temporal_features(person_A_id, person_B_id)
            
            if temporal_features['has_sufficient_history']:
                history_data = temporal_features['pair_interaction_history']
                if history_data.size(0) >= 2:
                    motion_features = extract_causal_motion_features(history_data.unsqueeze(0))
                    motion_features = motion_features.squeeze(0)  # [4]
                else:
                    motion_features = torch.zeros(4)
            else:
                motion_features = torch.zeros(4)
            
            # 获取历史边界框用于增强时序特征
            if temporal_features['has_sufficient_history']:
                history_geometric = temporal_features['pair_interaction_history']  # [history_length, 7]
                
                # 从几何特征重构历史边界框（改进版本）
                # 尝试从时序管理器获取真实的历史边界框
                try:
                    # 获取历史边界框数据
                    history_A = self.temporal_manager.buffer.person_tracks.get(person_A_id, deque())
                    history_B = self.temporal_manager.buffer.person_tracks.get(person_B_id, deque())
                    
                    if len(history_A) >= 2 and len(history_B) >= 2:
                        # 提取历史边界框（取最近的5帧）
                        recent_A = list(history_A)[-5:]
                        recent_B = list(history_B)[-5:]
                        
                        # 确保有足够的历史数据
                        if len(recent_A) < 5:
                            # 用最早的数据填充
                            recent_A = [recent_A[0]] * (5 - len(recent_A)) + recent_A
                        if len(recent_B) < 5:
                            recent_B = [recent_B[0]] * (5 - len(recent_B)) + recent_B
                        
                        # 从每个历史记录中提取边界框（假设stored as sample data）
                        person_A_history = torch.stack([
                            person_A_box if i >= len(recent_A) else person_A_box  # 简化：使用当前框
                            for i in range(5)
                        ])
                        person_B_history = torch.stack([
                            person_B_box if i >= len(recent_B) else person_B_box  # 简化：使用当前框  
                            for i in range(5)
                        ])
                    else:
                        # 回退到简化版本
                        person_A_history = person_A_box.unsqueeze(0).repeat(5, 1)
                        person_B_history = person_B_box.unsqueeze(0).repeat(5, 1)
                
                except Exception as e:
                    # 如果获取历史数据失败，使用简化版本
                    person_A_history = person_A_box.unsqueeze(0).repeat(5, 1)
                    person_B_history = person_B_box.unsqueeze(0).repeat(5, 1)
                
                enhanced_motion_features = extract_enhanced_motion_features(
                    person_A_history, person_B_history
                )  # [8]
            else:
                enhanced_motion_features = torch.zeros(8)
        else:
            motion_features = torch.zeros(4)
            enhanced_motion_features = torch.zeros(8)
        
        # 场景上下文特征 (1维)
        if self.use_scene_context and frame_id in self.scene_data:
            scene_context = self.scene_data[frame_id]['scene_context']
        else:
            scene_context = torch.tensor([1.0], dtype=torch.float32)  # 默认：稀疏场景
        
        # HoG特征提取 (64维)
        hog_features = torch.zeros(64, dtype=torch.float32)  # 默认零向量
        if self.use_hog_features:
            try:
                # 从样本中获取图像路径
                image_path = sample.get('image_path')
                if image_path and os.path.exists(image_path):
                    # 加载图像并提取HoG特征
                    image = Image.open(image_path).convert('RGB')
                    hog_features = extract_joint_hog_features(image, person_A_box, person_B_box)
                else:
                    # 如果没有图像路径，使用零向量
                    hog_features = torch.zeros(64, dtype=torch.float32)
            except Exception as e:
                print(f"Warning: HoG extraction failed for sample {idx}: {e}")
                hog_features = torch.zeros(64, dtype=torch.float32)
        
        # 组合特征向量
        feature_components = [geometric_features]  # [7] 几何特征（必须）
        
        if self.use_hog_features:
            feature_components.append(hog_features)  # [64] HoG特征
        
        if self.use_temporal and self.temporal_manager:
            # 添加时序特征（如果启用）
            feature_components.append(motion_features)  # [4] 基础运动特征
            
            # 使用完整的增强时序特征 (8维) 或者选择最重要的5维
            # 选择最相关的5个时序特征：速度相关性、方向对齐度、轨迹相似度、运动一致性、距离变化率
            selected_indices = [0, 1, 4, 6, 7]  # 选择最有意义的5个特征
            enhanced_motion_features_reduced = enhanced_motion_features[selected_indices]  # [5] 精选增强时序
            feature_components.append(enhanced_motion_features_reduced)
        
        full_features = torch.cat(feature_components).float()  # 动态维度
        
        return {
            'features': full_features,
            'stage2_label': stage2_label,
            'original_interaction': sample['original_interaction'],
            'person_A_id': person_A_id,
            'person_B_id': person_B_id,
            'frame_id': frame_id,
            'scene_context': scene_context
        }
    
    def _print_stage2_statistics(self):
        """打印Stage2数据集统计信息"""
        if not self.samples:
            return
        
        # 统计各类别数量
        class_counts = {}
        for sample in self.samples:
            label = sample['stage2_label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print(f"\nStage2 Dataset Statistics ({self.split}):")
        print("=" * 50)
        total = len(self.samples)
        
        for class_id in range(3):  # 3类分类
            count = class_counts.get(class_id, 0)
            percentage = 100 * count / total if total > 0 else 0
            class_name = self.label_mapper.get_mapped_class_name(class_id)
            print(f"Class {class_id} ({class_name}): {count:,} ({percentage:.1f}%)")
        
        print(f"Total samples: {total:,}")
        print(f"Temporal features: {self.use_temporal}")
        print(f"HoG features: {self.use_hog_features}")
        print(f"Scene context: {self.use_scene_context}")
        print(f"Oversampling: {self.use_oversampling}")
        
        # 计算特征维度
        feature_dim = 7  # 几何特征
        if self.use_hog_features:
            feature_dim += 64  # HoG特征
        if self.use_temporal:
            feature_dim += 9   # 4基础运动 + 5增强时序
        
        print(f"Feature dimension: {feature_dim}D")
    
    def get_stage2_class_distribution(self):
        """获取Stage2类别分布"""
        class_counts = {}
        for sample in self.samples:
            label = sample['stage2_label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        return {
            'class_counts': class_counts,
            'total': len(self.samples),
            'class_names': self.label_mapper.class_names
        }


if __name__ == '__main__':
    # 测试Stage2数据集
    print("Testing GeometricStage2Dataset...")
    
    data_path = r'C:\assignment\master programme\final\baseline\classificationnet\dataset'
    
    # 创建测试数据集
    print("Creating Stage2 dataset...")
    stage2_dataset = GeometricStage2Dataset(
        data_path, 
        split='train', 
        history_length=5,
        use_temporal=False, 
        use_scene_context=True,
        use_oversampling=True
    )
    
    # 测试数据加载
    print(f"\nTesting data loading...")
    sample = stage2_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"Features: {sample['features']}")
    print(f"Stage2 label: {sample['stage2_label']}")
    print(f"Original interaction: {sample['original_interaction']}")
    
    # 打印分布统计
    distribution = stage2_dataset.get_stage2_class_distribution()
    print(f"\nClass distribution: {distribution}")
    
    print("Stage2 dataset test completed!")