#!/usr/bin/env python3
"""
Improved Stage2 Dataset for Basic Mode
Clean, efficient, and modular design without complex temporal dependencies
"""

import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
from PIL import Image
import sys

# Add project paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_path = os.path.join(project_root, 'models')
sys.path.append(models_path)

from models.feature_extractors import BasicFeatureFusion


def get_fixed_scene_splits():
    """
    固定的数据集场景划分 - 确保所有模式使用相同的划分
    """
    trainset_split = [
        'bytes-cafe-2019-02-07_0', 
        # 'clark-center-2019-02-28_0', 
        'clark-center-2019-02-28_1',
        # 'clark-center-intersection-2019-02-28_0', 
        'cubberly-auditorium-2019-04-22_0',
        # 'cubberly-auditorium-2019-04-22_1',
        # 'discovery-walk-2019-02-28_0',
        'discovery-walk-2019-02-28_1',
        'food-trucks-2019-02-12_0',
        'forbes-cafe-2019-01-22_0', 
        'gates-159-group-meeting-2019-04-03_0',
        'gates-to-clark-2019-02-28_1',
        # 'gates-to-clark-2019-02-28_0',
        # 'gates-ai-lab-2019-02-08_0',
        'gates-ai-lab-2019-04-17_0',
        # 'gates-basement-elevators-2019-01-17_0',
        'gates-basement-elevators-2019-01-17_1', 
        # 'gates-foyer-2019-01-17_0',
        'hewlett-class-2019-01-23_0',
        # 'hewlett-class-2019-01-23_1',
        'hewlett-packard-intersection-2019-01-24_0', 
        'huang-2-2019-01-25_0', 
        'huang-2-2019-01-25_1',
        'huang-basement-2019-01-25_0',
        # 'huang-lane-2019-02-12_0', 
        'huang-intersection-2019-01-22_0',
        'indoor-coupa-cafe-2019-02-06_0',
        'lomita-serra-intersection-2019-01-30_0',
        'memorial-court-2019-03-16_0', 
        # 'meyer-green-2019-03-16_0',
        'meyer-green-2019-03-16_1',
        'nvidia-aud-2019-04-18_0',
        # 'nvidia-aud-2019-04-18_1', 
        # 'nvidia-aud-2019-04-18_2',
        'nvidia-aud-2019-01-25_0',
        'outdoor-coupa-cafe-2019-02-06_0',
        'quarry-road-2019-02-28_0',
        'serra-street-2019-01-30_0',
        'stlc-111-2019-04-19_0', 
        # 'stlc-111-2019-04-19_1', 
        # 'stlc-111-2019-04-19_2',
        'packard-poster-session-2019-03-20_2', 
        # 'packard-poster-session-2019-03-20_0',
        # 'packard-poster-session-2019-03-20_1',
        # 'svl-meeting-gates-2-2019-04-08_0',
        'svl-meeting-gates-2-2019-04-08_1',
        # 'tressider-2019-03-16_0',
        'tressider-2019-03-16_2',
        # 'tressider-2019-03-16_1',
        # 'tressider-2019-04-26_0',
        'tressider-2019-04-26_1',
        'tressider-2019-04-26_2',
        # 'tressider-2019-04-26_3',
        # 'jordan-hall-2019-04-22_0', 
    ]
    
    valset_split = [
        'clark-center-2019-02-28_0',
        'discovery-walk-2019-02-28_0',
        'gates-ai-lab-2019-02-08_0',
        'gates-foyer-2019-01-17_0',
        'hewlett-class-2019-01-23_1',
        'jordan-hall-2019-04-22_0', 
        'nvidia-aud-2019-04-18_1', 
        'packard-poster-session-2019-03-20_1',
        'stlc-111-2019-04-19_1', 
        'svl-meeting-gates-2-2019-04-08_0',
        'tressider-2019-03-16_1',
        'tressider-2019-04-26_3',

    ]
    
    testset_split = [
        'clark-center-intersection-2019-02-28_0', 
        'cubberly-auditorium-2019-04-22_1',
        'gates-basement-elevators-2019-01-17_0',
        'gates-to-clark-2019-02-28_0',
        'meyer-green-2019-03-16_0',
        'nvidia-aud-2019-04-18_2',
        'packard-poster-session-2019-03-20_0',
        'stlc-111-2019-04-19_2',
        'tressider-2019-03-16_0',
        'tressider-2019-04-26_0',

    ]
    
    return {
        'train': trainset_split,
        'val': valset_split,
        'test': testset_split
    }


class Stage2LabelMapper:
    """Stage2标签映射器 - 3分类"""
    
    def __init__(self):
        self.label_mapping = {
            'walking together': 0,    # Walking Together
            'standing together': 1,   # Standing Together  
            'sitting together': 2,    # Sitting Together
        }
        
        self.class_names = [
            'Walking Together',   # 移动行为
            'Standing Together',  # 站立行为  
            'Sitting Together'    # 坐着行为
        ]
        
        # 类别权重 (3分类) - ResNet训练中未使用，动态计算
        # self.class_weights = {0: 1.0, 1: 1.4, 2: 6.1}

        # 目标类别分布 (用于过采样) - ResNet训练中未使用，使用WeightedRandomSampler
        # self.target_distribution = {0: 0.50, 1: 0.35, 2: 0.15}
    
    def map_label(self, original_interaction: str) -> Optional[int]:
        """将原始交互标签映射为3分类标签"""
        return self.label_mapping.get(original_interaction, None)
    
    def is_valid_interaction(self, original_interaction: str) -> bool:
        """检查交互类型是否为有效的3分类之一"""
        return original_interaction in self.label_mapping


class BasicStage2Dataset(Dataset):
    """
    Basic模式的Stage2数据集
    简化设计，专注于几何特征和HoG特征，不包含复杂的时序依赖
    """
    
    def __init__(self, data_path: str, split: str = 'train',
                 use_geometric: bool = True, use_hog: bool = True, 
                 use_scene_context: bool = True, frame_interval: int = 1,
                 use_oversampling: bool = False):
        """
        Args:
            data_path: 数据集路径
            split: 数据集划分 ('train', 'val', 'test')
            use_geometric: 是否使用几何特征
            use_hog: 是否使用HoG特征
            use_scene_context: 是否使用场景上下文
            frame_interval: 帧采样间隔 (1=每帧, 10=每10帧)
            use_oversampling: 是否使用过采样平衡类别
        """
        self.data_path = data_path
        self.split = split
        self.use_geometric = use_geometric
        self.use_hog = use_hog
        self.use_scene_context = use_scene_context
        self.frame_interval = frame_interval
        self.use_oversampling = use_oversampling
        
        # 初始化标签映射器和特征融合器
        self.label_mapper = Stage2LabelMapper()
        self.feature_fusion = BasicFeatureFusion(
            use_geometric=use_geometric,
            use_hog=use_hog, 
            use_scene_context=use_scene_context
        )
        
        # 数据存储
        self.samples = []
        self.scene_data = {}
        
        # 加载和处理数据
        self._load_data()
        self._filter_and_label_samples()
        
        if self.use_oversampling and split == 'train':
            self._apply_oversampling()
        
        print(f"✅ BasicStage2Dataset loaded: {len(self.samples)} samples ({split})")
        self._print_statistics()
    
    def _load_data(self):
        """加载JRDB格式的社交标签数据"""
        social_labels_dir = os.path.join(self.data_path, 'labels', 'labels_2d_activity_social_stitched')
        images_dir = os.path.join(self.data_path, 'images', 'image_stitched')
        
        if not os.path.exists(social_labels_dir):
            raise FileNotFoundError(f"Social labels directory not found: {social_labels_dir}")
        
        # 获取场景文件
        scene_files = [f for f in os.listdir(social_labels_dir) if f.endswith('.json')]
        scene_files.sort()  # 确保一致的顺序
        
        # 使用固定场景划分
        scene_splits = get_fixed_scene_splits()
        target_scenes = scene_splits[self.split]
        
        # 筛选存在的场景文件
        selected_files = []
        for scene_name in target_scenes:
            scene_file = f"{scene_name}.json"
            if scene_file in scene_files:
                selected_files.append(scene_file)
            else:
                print(f"Warning: Scene file {scene_file} not found in dataset")
        
        print(f"Loading {len(selected_files)}/{len(target_scenes)} scenes for {self.split} split")
        
        # 加载数据
        sample_count = 0
        for scene_file in selected_files:
            scene_path = os.path.join(social_labels_dir, scene_file)
            scene_name = os.path.splitext(scene_file)[0]
            
            try:
                with open(scene_path, 'r') as f:
                    scene_data = json.load(f)
                
                # 处理场景中的每一帧
                frame_names = list(scene_data.get('labels', {}).keys())
                frame_names.sort()  # 确保按顺序处理
                
                # 应用帧间隔采样
                selected_frames = frame_names[::self.frame_interval]
                
                for image_name in selected_frames:
                    annotations = scene_data['labels'][image_name]
                    frame_id = f"{scene_name}_{self._extract_frame_id(image_name)}"
                    
                    # 构建图像路径
                    image_path = os.path.join(images_dir, scene_name, image_name)
                    
                    # 收集该帧的所有人员信息
                    person_dict = {}
                    all_boxes = []
                    
                    for ann in annotations:
                        person_id = ann.get('label_id', '')
                        if person_id.startswith('pedestrian:'):
                            pid = int(person_id.split(':')[1])
                            box = ann.get('box', [0, 0, 100, 100])
                            
                            # 数据验证：检查边界框有效性
                            if self._is_valid_box(box):
                                all_boxes.append(box)
                                person_dict[pid] = {
                                    'box': box,
                                    'interactions': ann.get('H-interaction', []) or ann.get('HHI', [])
                                }
                    
                    # 存储场景信息
                    self.scene_data[frame_id] = {
                        'scene_name': scene_name,
                        'image_name': image_name,
                        'image_path': image_path if os.path.exists(image_path) else None,
                        'all_boxes': all_boxes,
                        'persons': person_dict
                    }
                    
                    # 提取正样本（有交互的人员对）
                    for ann in annotations:
                        person_id = ann.get('label_id', '')
                        if not person_id.startswith('pedestrian:'):
                            continue
                        
                        person_A_id = int(person_id.split(':')[1])
                        if person_A_id not in person_dict:
                            continue
                        
                        person_A_box = person_dict[person_A_id]['box']
                        
                        # 处理该人员的所有交互
                        for interaction in (ann.get('H-interaction', []) or ann.get('HHI', [])):
                            pair_id = interaction.get('pair', '')
                            if pair_id.startswith('pedestrian:'):
                                person_B_id = int(pair_id.split(':')[1])
                                
                                if person_B_id in person_dict:
                                    # 避免重复交互对：只保留ID较小者作为person_A
                                    if person_A_id < person_B_id:
                                        person_B_box = person_dict[person_B_id]['box']
                                        interaction_labels = interaction.get('inter_labels', {})
                                        
                                        # 创建正样本
                                        sample = {
                                            'frame_id': frame_id,
                                            'scene_name': scene_name,
                                            'image_name': image_name,
                                            'person_A_id': person_A_id,
                                            'person_B_id': person_B_id,
                                            'person_A_box': person_A_box,
                                            'person_B_box': person_B_box,
                                            'interaction_labels': interaction_labels,
                                            'sample_type': 'positive'
                                        }
                                        self.samples.append(sample)
                                        sample_count += 1
                    
                    # 定期打印进度
                    if sample_count % 1000 == 0 and sample_count > 0:
                        print(f"  Processed {sample_count} samples...")
                        
            except Exception as e:
                print(f"Warning: Error loading scene {scene_file}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} raw samples from {len(selected_files)} scenes")
        if self.frame_interval > 1:
            print(f"Applied frame interval {self.frame_interval} (reduced samples by ~{((self.frame_interval-1)/self.frame_interval)*100:.0f}%)")
    
    def _extract_frame_id(self, image_name: str) -> str:
        """从图像名提取帧ID"""
        # JRDB format: "000000.jpg" -> "000000"
        return os.path.splitext(image_name)[0]
    
    def _is_valid_box(self, box: List[float]) -> bool:
        """验证边界框的有效性"""
        if len(box) != 4:
            return False
        x, y, w, h = box
        # 检查边界框的合理性
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return False
        if w > 5000 or h > 5000:  # 异常大的边界框
            return False
        return True
    
    def _filter_and_label_samples(self):
        """过滤样本并应用Stage2标签"""
        valid_samples = []
        
        for sample in self.samples:
            interaction_labels = sample.get('interaction_labels', {})
            
            if isinstance(interaction_labels, dict) and len(interaction_labels) > 0:
                # 取第一个交互类型
                interaction_type = list(interaction_labels.keys())[0]
                
                # 检查是否为有效的3分类标签
                if self.label_mapper.is_valid_interaction(interaction_type):
                    # 映射到3分类
                    stage2_label = self.label_mapper.map_label(interaction_type)
                    sample['stage2_label'] = stage2_label
                    sample['original_interaction'] = interaction_type
                    valid_samples.append(sample)
        
        print(f"Filtered valid Stage2 samples: {len(valid_samples)}/{len(self.samples)} "
              f"({100*len(valid_samples)/len(self.samples) if self.samples else 0:.1f}%)")
        
        self.samples = valid_samples
    
    def _apply_oversampling(self):
        """应用过采样策略平衡类别"""
        # 统计各类样本数量
        class_samples = {i: [] for i in range(3)}
        for sample in self.samples:
            label = sample['stage2_label']
            class_samples[label].append(sample)
        
        print("Original class distribution:")
        for class_id, samples in class_samples.items():
            class_name = self.label_mapper.class_names[class_id]
            print(f"  Class {class_id} ({class_name}): {len(samples)}")
        
        total_samples = len(self.samples)
        balanced_samples = []
        
        for class_id, target_ratio in self.label_mapper.target_distribution.items():
            current_samples = class_samples[class_id]
            target_count = int(total_samples * target_ratio)
            
            if len(current_samples) == 0:
                continue
                
            if len(current_samples) < target_count:
                # 过采样
                indices = np.random.choice(
                    len(current_samples), size=target_count, replace=True
                )
                oversampled = [current_samples[i] for i in indices]
                balanced_samples.extend(oversampled)
                print(f"  Oversampled class {class_id}: {len(current_samples)} -> {len(oversampled)}")
            else:
                # 下采样
                indices = np.random.choice(
                    len(current_samples), size=target_count, replace=False
                )
                undersampled = [current_samples[i] for i in indices]
                balanced_samples.extend(undersampled)
                print(f"  Undersampled class {class_id}: {len(current_samples)} -> {len(undersampled)}")
        
        self.samples = balanced_samples
        np.random.shuffle(self.samples)  # 打乱顺序
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取样本
        
        Returns:
            dict: 包含'features'和'stage2_label'等字段的样本
        """
        sample = self.samples[idx]
        
        # 获取基础信息
        frame_id = sample['frame_id']
        person_A_box = torch.tensor(sample['person_A_box'], dtype=torch.float32)
        person_B_box = torch.tensor(sample['person_B_box'], dtype=torch.float32)
        stage2_label = torch.tensor(sample['stage2_label'], dtype=torch.long)
        
        # 获取图像和场景信息
        scene_info = self.scene_data.get(frame_id, {})
        image_path = scene_info.get('image_path')
        all_boxes = scene_info.get('all_boxes', [])
        
        # 加载图像 (用于HoG特征)
        image = None
        if self.use_hog and image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")
                image = None
        
        # 使用特征融合器提取特征
        try:
            features = self.feature_fusion(
                person_A_box=person_A_box,
                person_B_box=person_B_box,
                image=image,
                all_boxes=all_boxes,
                image_width=3760,  # JRDB标准分辨率
                image_height=480
            )
        except Exception as e:
            print(f"Warning: Feature extraction failed for sample {idx}: {e}")
            # 使用零向量作为fallback
            features = torch.zeros(self.feature_fusion.get_output_dim(), dtype=torch.float32)
        
        return {
            'features': features,
            'stage2_label': stage2_label,
            'original_interaction': sample['original_interaction'],
            'person_A_id': sample['person_A_id'],
            'person_B_id': sample['person_B_id'],
            'frame_id': frame_id
        }
    
    def get_class_distribution(self) -> Dict:
        """获取类别分布信息"""
        class_counts = {}
        for sample in self.samples:
            label = sample['stage2_label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        return {
            'class_counts': class_counts,
            'total': len(self.samples),
            'class_names': self.label_mapper.class_names
        }
    
    def _print_statistics(self):
        """打印数据集统计信息"""
        if not self.samples:
            return
        
        print(f"\nBasicStage2Dataset Statistics ({self.split}):")
        print("=" * 50)
        
        # 基本信息
        print(f"Total samples: {len(self.samples):,}")
        print(f"Frame interval: {self.frame_interval}")
        print(f"Features: {self.feature_fusion.get_feature_info()}")
        
        # 类别分布
        distribution = self.get_class_distribution()
        print(f"\nClass Distribution:")
        total = distribution['total']
        for class_id, count in distribution['class_counts'].items():
            class_name = distribution['class_names'][class_id]
            percentage = 100 * count / total if total > 0 else 0
            print(f"  Class {class_id} ({class_name}): {count:,} ({percentage:.1f}%)")
        
        # 采样统计
        if self.use_oversampling and self.split == 'train':
            print(f"Oversampling: Enabled")
        else:
            print(f"Oversampling: Disabled")


class LSTMStage2Dataset(Dataset):
    """
    LSTM模式的Stage2数据集
    提供时序特征序列用于LSTM训练
    """
    
    def __init__(self, data_path: str, split: str = 'train',
                 use_geometric: bool = True, use_hog: bool = True, 
                 use_scene_context: bool = True, sequence_length: int = 5,
                 frame_interval: int = 1, use_oversampling: bool = False):
        """
        Args:
            data_path: 数据集路径
            split: 数据集划分 ('train', 'val', 'test')
            use_geometric: 是否使用几何特征
            use_hog: 是否使用HoG特征
            use_scene_context: 是否使用场景上下文
            sequence_length: 时序长度
            frame_interval: 帧采样间隔
            use_oversampling: 是否使用过采样平衡类别
        """
        self.data_path = data_path
        self.split = split
        self.use_geometric = use_geometric
        self.use_hog = use_hog
        self.use_scene_context = use_scene_context
        self.sequence_length = sequence_length
        self.frame_interval = frame_interval
        self.use_oversampling = use_oversampling
        
        # 初始化标签映射器和特征融合器
        self.label_mapper = Stage2LabelMapper()
        self.feature_fusion = BasicFeatureFusion(
            use_geometric=use_geometric,
            use_hog=use_hog, 
            use_scene_context=use_scene_context
        )
        
        # 数据存储
        self.sequences = []
        self.scene_data = {}
        
        # 加载和处理数据
        self._load_temporal_data()
        self._create_sequences()
        
        if self.use_oversampling and split == 'train':
            self._apply_temporal_oversampling()
        
        print(f"✅ LSTMStage2Dataset loaded: {len(self.sequences)} sequences ({split})")
        self._print_statistics()
    
    def _load_temporal_data(self):
        """加载时序数据，基于BasicStage2Dataset的逻辑"""
        # 复用BasicStage2Dataset的数据加载逻辑
        basic_dataset = BasicStage2Dataset(
            data_path=self.data_path,
            split=self.split,
            use_geometric=self.use_geometric,
            use_hog=self.use_hog,
            use_scene_context=self.use_scene_context,
            frame_interval=self.frame_interval,
            use_oversampling=False  # 在时序级别进行采样
        )
        
        # 获取基础数据
        self.basic_samples = basic_dataset.samples
        self.scene_data = basic_dataset.scene_data
        self.feature_fusion = basic_dataset.feature_fusion
    
    def _create_sequences(self):
        """将单帧样本组织成时序序列"""
        print(f"Creating temporal sequences (length={self.sequence_length})...")
        
        # 按场景和交互对分组
        interaction_groups = {}
        
        for sample in self.basic_samples:
            scene_name = sample['scene_name']
            person_A_id = sample['person_A_id']
            person_B_id = sample['person_B_id']
            frame_id = sample['frame_id']
            
            # 提取帧号
            frame_num = int(frame_id.split('_')[-1])
            
            # 分组key：场景+交互对
            group_key = f"{scene_name}_{person_A_id}_{person_B_id}"
            
            if group_key not in interaction_groups:
                interaction_groups[group_key] = []
            
            interaction_groups[group_key].append({
                'frame_num': frame_num,
                'sample': sample
            })
        
        # 为每个交互对创建时序序列
        sequence_count = 0
        for group_key, frame_samples in interaction_groups.items():
            # 按帧号排序
            frame_samples.sort(key=lambda x: x['frame_num'])
            
            # 滑动窗口创建序列
            for i in range(len(frame_samples) - self.sequence_length + 1):
                sequence_frames = frame_samples[i:i + self.sequence_length]
                
                # 检查时序连续性（允许一定间隙）
                if self._is_valid_sequence(sequence_frames):
                    sequence = {
                        'group_key': group_key,
                        'frame_samples': [fs['sample'] for fs in sequence_frames],
                        'start_frame': sequence_frames[0]['frame_num'],
                        'end_frame': sequence_frames[-1]['frame_num'],
                        'stage2_label': sequence_frames[-1]['sample']['stage2_label']  # 使用最后一帧的标签
                    }
                    self.sequences.append(sequence)
                    sequence_count += 1
        
        print(f"Created {sequence_count} temporal sequences from {len(self.basic_samples)} frames")
    
    def _is_valid_sequence(self, sequence_frames: List[Dict]) -> bool:
        """检查序列的有效性"""
        if len(sequence_frames) != self.sequence_length:
            return False
        
        # 检查帧号连续性（允许一定的跳跃）
        frame_nums = [fs['frame_num'] for fs in sequence_frames]
        max_gap = self.frame_interval * 3  # 允许最大3倍间隔的跳跃
        
        for i in range(1, len(frame_nums)):
            gap = frame_nums[i] - frame_nums[i-1]
            if gap > max_gap:
                return False
        
        return True
    
    def _apply_temporal_oversampling(self):
        """时序级别的过采样"""
        # 统计各类别序列数量
        class_counts = {}
        for seq in self.sequences:
            label = seq['stage2_label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # 找到多数类数量
        max_count = max(class_counts.values()) if class_counts else 0
        
        # 对少数类进行过采样
        oversampled_sequences = list(self.sequences)
        for label, count in class_counts.items():
            if count < max_count:
                # 随机重复少数类序列
                class_sequences = [seq for seq in self.sequences if seq['stage2_label'] == label]
                oversample_count = max_count - count
                
                import random
                random.shuffle(class_sequences)
                
                for i in range(oversample_count):
                    seq_to_duplicate = class_sequences[i % len(class_sequences)]
                    oversampled_sequences.append(seq_to_duplicate)
        
        self.sequences = oversampled_sequences
        print(f"Applied temporal oversampling: {len(self.sequences)} sequences")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取时序样本
        
        Returns:
            dict: 包含'sequences'和'stage2_label'等字段的样本
        """
        sequence = self.sequences[idx]
        frame_samples = sequence['frame_samples']
        stage2_label = torch.tensor(sequence['stage2_label'], dtype=torch.long)
        
        # 提取每帧的特征
        sequence_features = []
        for sample in frame_samples:
            # 获取基础信息
            frame_id = sample['frame_id']
            person_A_box = torch.tensor(sample['person_A_box'], dtype=torch.float32)
            person_B_box = torch.tensor(sample['person_B_box'], dtype=torch.float32)
            
            # 获取图像和场景信息
            scene_info = self.scene_data.get(frame_id, {})
            image_path = scene_info.get('image_path')
            all_boxes = scene_info.get('all_boxes', [])
            
            # 加载图像 (用于HoG特征)
            image = None
            if self.use_hog and image_path and os.path.exists(image_path):
                try:
                    from PIL import Image
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"Warning: Failed to load image {image_path}: {e}")
                    image = None
            
            # 使用特征融合器提取特征
            try:
                features = self.feature_fusion(
                    person_A_box=person_A_box,
                    person_B_box=person_B_box,
                    image=image,
                    all_boxes=all_boxes,
                    image_width=3760,  # JRDB标准分辨率
                    image_height=480
                )
                sequence_features.append(features)
            except Exception as e:
                print(f"Warning: Feature extraction failed for sequence {idx}, frame {frame_id}: {e}")
                # 使用零向量作为fallback
                features = torch.zeros(self.feature_fusion.get_output_dim(), dtype=torch.float32)
                sequence_features.append(features)
        
        # 堆叠成时序张量 [sequence_length, feature_dim]
        sequences_tensor = torch.stack(sequence_features)
        
        return {
            'sequences': sequences_tensor,
            'stage2_label': stage2_label,
            'group_key': sequence['group_key'],
            'start_frame': sequence['start_frame'],
            'end_frame': sequence['end_frame']
        }
    
    def get_class_distribution(self) -> Dict:
        """获取类别分布信息"""
        class_counts = {}
        for seq in self.sequences:
            label = seq['stage2_label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        return {
            'class_counts': class_counts,
            'total': len(self.sequences),
            'class_names': self.label_mapper.class_names
        }
    
    def _print_statistics(self):
        """打印数据集统计信息"""
        if not self.sequences:
            return
        
        print(f"\nLSTMStage2Dataset Statistics ({self.split}):")
        print("=" * 50)
        
        # 基本信息
        print(f"Total sequences: {len(self.sequences):,}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Frame interval: {self.frame_interval}")
        print(f"Features: {self.feature_fusion.get_feature_info()}")
        
        # 类别分布
        distribution = self.get_class_distribution()
        print(f"\nClass Distribution:")
        total = distribution['total']
        for class_id, count in distribution['class_counts'].items():
            class_name = distribution['class_names'][class_id]
            percentage = 100 * count / total if total > 0 else 0
            print(f"  Class {class_id} ({class_name}): {count:,} ({percentage:.1f}%)")
        
        # 时序统计
        if self.use_oversampling and self.split == 'train':
            print(f"\nApplied temporal oversampling for balanced training")


class RelationStage2Dataset(Dataset):
    """
    Relation Network模式的Stage2数据集
    将两人特征分离，支持RelationStage2Classifier的输入格式
    """
    
    def __init__(self, data_path: str, split: str = 'train',
                 use_geometric: bool = True, use_hog: bool = True, 
                 use_scene_context: bool = True, frame_interval: int = 1,
                 use_oversampling: bool = False):
        """
        Args:
            data_path: 数据集路径
            split: 数据集划分 ('train', 'val', 'test')
            use_geometric: 是否使用几何特征
            use_hog: 是否使用HoG特征
            use_scene_context: 是否使用场景上下文
            frame_interval: 帧采样间隔
            use_oversampling: 是否使用过采样平衡类别
        """
        self.data_path = data_path
        self.split = split
        self.use_geometric = use_geometric
        self.use_hog = use_hog
        self.use_scene_context = use_scene_context
        self.frame_interval = frame_interval
        self.use_oversampling = use_oversampling
        
        # 初始化标签映射器和特征融合器
        self.label_mapper = Stage2LabelMapper()
        
        # 为每个人单独创建特征融合器
        from ..models.feature_extractors import RelationFeatureFusion
        self.feature_fusion = RelationFeatureFusion(
            use_geometric=use_geometric,
            use_hog=use_hog, 
            use_scene_context=use_scene_context
        )
        
        # 数据存储
        self.samples = []
        self.scene_data = {}
        
        # 加载和处理数据
        self._load_data()
        self._filter_and_label_samples()
        
        if self.use_oversampling and split == 'train':
            self._apply_oversampling()
        
        print(f"✅ RelationStage2Dataset loaded: {len(self.samples)} samples ({split})")
        self._print_statistics()
    
    def _load_data(self):
        """加载JRDB格式的社交标签数据 - 复用BasicStage2Dataset的逻辑"""
        social_labels_dir = os.path.join(self.data_path, 'labels', 'labels_2d_activity_social_stitched')
        images_dir = os.path.join(self.data_path, 'images', 'image_stitched')
        
        if not os.path.exists(social_labels_dir):
            raise FileNotFoundError(f"Social labels directory not found: {social_labels_dir}")
        
        # 获取场景文件
        scene_files = [f for f in os.listdir(social_labels_dir) if f.endswith('.json')]
        scene_files.sort()  # 确保一致的顺序
        
        # 使用固定场景划分
        scene_splits = get_fixed_scene_splits()
        target_scenes = scene_splits[self.split]
        
        # 筛选存在的场景文件
        selected_files = []
        for scene_name in target_scenes:
            scene_file = f"{scene_name}.json"
            if scene_file in scene_files:
                selected_files.append(scene_file)
            else:
                print(f"Warning: Scene file {scene_file} not found in dataset")
        
        print(f"Loading {len(selected_files)}/{len(target_scenes)} scenes for {self.split} split")
        
        # 加载数据
        sample_count = 0
        for scene_file in selected_files:
            scene_path = os.path.join(social_labels_dir, scene_file)
            scene_name = os.path.splitext(scene_file)[0]
            
            try:
                with open(scene_path, 'r') as f:
                    scene_data = json.load(f)
                
                # 处理场景中的每一帧
                frame_names = list(scene_data.get('labels', {}).keys())
                frame_names.sort()  # 确保按顺序处理
                
                # 应用帧间隔采样
                selected_frames = frame_names[::self.frame_interval]
                
                for image_name in selected_frames:
                    annotations = scene_data['labels'][image_name]
                    frame_id = f"{scene_name}_{self._extract_frame_id(image_name)}"
                    
                    # 构建图像路径
                    image_path = os.path.join(images_dir, scene_name, image_name)
                    
                    # 收集该帧的所有人员信息
                    person_dict = {}
                    all_boxes = []
                    
                    for ann in annotations:
                        person_id = ann.get('label_id', '')
                        if person_id.startswith('pedestrian:'):
                            pid = int(person_id.split(':')[1])
                            box = ann.get('box', [0, 0, 100, 100])
                            
                            # 数据验证：检查边界框有效性
                            if self._is_valid_box(box):
                                all_boxes.append(box)
                                person_dict[pid] = {
                                    'box': box,
                                    'interactions': ann.get('H-interaction', []) or ann.get('HHI', [])
                                }
                    
                    # 存储场景信息
                    self.scene_data[frame_id] = {
                        'scene_name': scene_name,
                        'image_name': image_name,
                        'image_path': image_path if os.path.exists(image_path) else None,
                        'all_boxes': all_boxes,
                        'persons': person_dict
                    }
                    
                    # 提取正样本（有交互的人员对）
                    for ann in annotations:
                        person_id = ann.get('label_id', '')
                        if not person_id.startswith('pedestrian:'):
                            continue
                        
                        person_A_id = int(person_id.split(':')[1])
                        if person_A_id not in person_dict:
                            continue
                        
                        person_A_box = person_dict[person_A_id]['box']
                        
                        # 处理该人员的所有交互
                        for interaction in (ann.get('H-interaction', []) or ann.get('HHI', [])):
                            pair_id = interaction.get('pair', '')
                            if pair_id.startswith('pedestrian:'):
                                person_B_id = int(pair_id.split(':')[1])
                                
                                if person_B_id in person_dict:
                                    # 避免重复交互对：只保留ID较小者作为person_A
                                    if person_A_id < person_B_id:
                                        person_B_box = person_dict[person_B_id]['box']
                                        interaction_labels = interaction.get('inter_labels', {})
                                        
                                        # 创建正样本
                                        sample = {
                                            'frame_id': frame_id,
                                            'scene_name': scene_name,
                                            'image_name': image_name,
                                            'person_A_id': person_A_id,
                                            'person_B_id': person_B_id,
                                            'person_A_box': person_A_box,
                                            'person_B_box': person_B_box,
                                            'interaction_labels': interaction_labels,
                                            'sample_type': 'positive'
                                        }
                                        self.samples.append(sample)
                                        sample_count += 1
                    
                    # 定期打印进度
                    if sample_count % 1000 == 0 and sample_count > 0:
                        print(f"  Processed {sample_count} samples...")
                        
            except Exception as e:
                print(f"Warning: Error loading scene {scene_file}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} raw samples from {len(selected_files)} scenes")
        if self.frame_interval > 1:
            print(f"Applied frame interval {self.frame_interval} (reduced samples by ~{((self.frame_interval-1)/self.frame_interval)*100:.0f}%)")
    
    def _extract_frame_id(self, image_name: str) -> str:
        """从图像名提取帧ID"""
        return os.path.splitext(image_name)[0]
    
    def _is_valid_box(self, box: List[float]) -> bool:
        """验证边界框的有效性"""
        if len(box) != 4:
            return False
        x, y, w, h = box
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return False
        if w > 5000 or h > 5000:  # 异常大的边界框
            return False
        return True
    
    def _filter_and_label_samples(self):
        """过滤样本并应用Stage2标签"""
        valid_samples = []
        
        for sample in self.samples:
            interaction_labels = sample.get('interaction_labels', {})
            
            if isinstance(interaction_labels, dict) and len(interaction_labels) > 0:
                # 取第一个交互类型
                interaction_type = list(interaction_labels.keys())[0]
                
                # 检查是否为有效的3分类标签
                if self.label_mapper.is_valid_interaction(interaction_type):
                    # 映射到3分类
                    stage2_label = self.label_mapper.map_label(interaction_type)
                    sample['stage2_label'] = stage2_label
                    sample['original_interaction'] = interaction_type
                    valid_samples.append(sample)
        
        print(f"Filtered valid Stage2 samples: {len(valid_samples)}/{len(self.samples)} "
              f"({100*len(valid_samples)/len(self.samples) if self.samples else 0:.1f}%)")
        
        self.samples = valid_samples
    
    def _apply_oversampling(self):
        """应用过采样策略平衡类别 - 复用BasicStage2Dataset逻辑"""
        # 统计各类样本数量
        class_samples = {i: [] for i in range(3)}
        for sample in self.samples:
            label = sample['stage2_label']
            class_samples[label].append(sample)
        
        print("Original class distribution:")
        for class_id, samples in class_samples.items():
            class_name = self.label_mapper.class_names[class_id]
            print(f"  Class {class_id} ({class_name}): {len(samples)}")
        
        total_samples = len(self.samples)
        balanced_samples = []
        
        for class_id, target_ratio in self.label_mapper.target_distribution.items():
            current_samples = class_samples[class_id]
            target_count = int(total_samples * target_ratio)
            
            if len(current_samples) == 0:
                continue
                
            if len(current_samples) < target_count:
                # 过采样
                indices = np.random.choice(
                    len(current_samples), size=target_count, replace=True
                )
                oversampled = [current_samples[i] for i in indices]
                balanced_samples.extend(oversampled)
                print(f"  Oversampled class {class_id}: {len(current_samples)} -> {len(oversampled)}")
            else:
                # 下采样
                indices = np.random.choice(
                    len(current_samples), size=target_count, replace=False
                )
                undersampled = [current_samples[i] for i in indices]
                balanced_samples.extend(undersampled)
                print(f"  Undersampled class {class_id}: {len(current_samples)} -> {len(undersampled)}")
        
        self.samples = balanced_samples
        np.random.shuffle(self.samples)  # 打乱顺序
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取Relation Network样本
        
        Returns:
            dict: 包含'person_A_features', 'person_B_features', 'spatial_features', 'stage2_label'的样本
        """
        sample = self.samples[idx]
        
        # 获取基础信息
        frame_id = sample['frame_id']
        person_A_box = torch.tensor(sample['person_A_box'], dtype=torch.float32)
        person_B_box = torch.tensor(sample['person_B_box'], dtype=torch.float32)
        stage2_label = torch.tensor(sample['stage2_label'], dtype=torch.long)
        
        # 获取图像和场景信息
        scene_info = self.scene_data.get(frame_id, {})
        image_path = scene_info.get('image_path')
        all_boxes = scene_info.get('all_boxes', [])
        
        # 加载图像 (用于HoG特征)
        image = None
        if self.use_hog and image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")
                image = None
        
        # 使用RelationFeatureFusion提取分离的特征
        try:
            relation_features = self.feature_fusion(
                person_A_box=person_A_box,
                person_B_box=person_B_box,
                image=image,
                all_boxes=all_boxes,
                image_width=3760,  # JRDB标准分辨率
                image_height=480
            )
            
            person_A_features = relation_features['person_A_features']
            person_B_features = relation_features['person_B_features'] 
            spatial_features = relation_features.get('spatial_features', torch.tensor([], dtype=torch.float32))
            
        except Exception as e:
            print(f"Warning: Relation feature extraction failed for sample {idx}: {e}")
            # 使用零向量作为fallback
            feature_dim = self.feature_fusion.get_person_feature_dim()
            person_A_features = torch.zeros(feature_dim, dtype=torch.float32)
            person_B_features = torch.zeros(feature_dim, dtype=torch.float32)
            spatial_features = torch.tensor([], dtype=torch.float32)
        
        return {
            'person_A_features': person_A_features,
            'person_B_features': person_B_features,
            'spatial_features': spatial_features,
            'stage2_label': stage2_label,
            'original_interaction': sample['original_interaction'],
            'person_A_id': sample['person_A_id'],
            'person_B_id': sample['person_B_id'],
            'frame_id': frame_id
        }
    
    def get_class_distribution(self) -> Dict:
        """获取类别分布信息"""
        class_counts = {}
        for sample in self.samples:
            label = sample['stage2_label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        return {
            'class_counts': class_counts,
            'total': len(self.samples),
            'class_names': self.label_mapper.class_names
        }
    
    def _print_statistics(self):
        """打印数据集统计信息"""
        if not self.samples:
            return
        
        print(f"\nRelationStage2Dataset Statistics ({self.split}):")
        print("=" * 50)
        
        # 基本信息
        print(f"Total samples: {len(self.samples):,}")
        print(f"Frame interval: {self.frame_interval}")
        print(f"Features: {self.feature_fusion.get_feature_info()}")
        
        # 类别分布
        distribution = self.get_class_distribution()
        print(f"\nClass Distribution:")
        total = distribution['total']
        for class_id, count in distribution['class_counts'].items():
            class_name = distribution['class_names'][class_id]
            percentage = 100 * count / total if total > 0 else 0
            print(f"  Class {class_id} ({class_name}): {count:,} ({percentage:.1f}%)")
        
        # 采样统计
        if self.use_oversampling and self.split == 'train':
            print(f"Oversampling: Enabled")
        else:
            print(f"Oversampling: Disabled")


# 向后兼容的别名
Stage2Dataset = BasicStage2Dataset


if __name__ == '__main__':
    # 测试数据集
    print("Testing BasicStage2Dataset...")
    
    # 测试配置
    data_path = r'C:\assignment\master programme\final\baseline\classificationnet\dataset'
    
    try:
        print("\n1. Testing Basic mode dataset...")
        dataset = BasicStage2Dataset(
            data_path=data_path,
            split='train',
            use_geometric=True,
            use_hog=True,
            use_scene_context=True,
            frame_interval=1,
            use_oversampling=False  # 测试时不使用过采样
        )
        
        if len(dataset) > 0:
            print(f"\n2. Testing sample loading...")
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Features shape: {sample['features'].shape}")
            print(f"Label: {sample['stage2_label']}")
            print(f"Original interaction: {sample['original_interaction']}")
            
            # 测试多个样本
            print(f"\n3. Testing batch loading...")
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"Sample {i}: features={sample['features'].shape}, label={sample['stage2_label'].item()}")
        
        print(f"\n4. Testing class distribution...")
        distribution = dataset.get_class_distribution()
        print(f"Distribution: {distribution}")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        print("This is expected if the dataset path doesn't exist")
    
    print("\n✅ BasicStage2Dataset test completed!")