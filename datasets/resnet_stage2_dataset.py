#!/usr/bin/env python3
"""
ResNet-based Stage2 Dataset
Dataset implementation for ResNet backbone with Relation Network
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import Counter

# Add project paths for imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_path = os.path.join(project_root, 'models')
datasets_path = os.path.join(project_root, 'datasets')
src_path = os.path.join(project_root, 'src')
sys.path.extend([models_path, datasets_path, src_path])

from models.resnet_feature_extractors import ResNetRelationFeatureFusion
from stage2_dataset import Stage2LabelMapper  # 复用标签映射器
from src.features.geometric_features import extract_geometric_features
from src.features.optical_flow_features import OpticalFlowExtractor


class ResNetStage2Dataset(Dataset):
    """
    ResNet-based Stage2 dataset for behavior classification
    Compatible with Relation Network architecture using ResNet backbone
    """
    
    def __init__(self, data_path: str, split: str = 'train',
                 backbone_name: str = 'resnet18', visual_feature_dim: int = 256,
                 use_geometric: bool = True, use_scene_context: bool = True,
                 use_optical_flow: bool = False, optical_flow_method: str = 'farneback',
                 pretrained: bool = True, freeze_backbone: bool = False,
                 frame_interval: int = 1, use_oversampling: bool = False,
                 crop_size: int = 112,
                 filter_occlusion: bool = True, filter_edge_cases: bool = True,
                 edge_threshold: int = 200):
        """
        Args:
            data_path: 数据集路径
            split: 数据集划分 ('train', 'val', 'test')
            backbone_name: ResNet架构名称
            visual_feature_dim: 视觉特征维度
            use_geometric: 是否使用几何特征
            use_scene_context: 是否使用场景上下文
            use_optical_flow: 是否使用光流特征 (NEW)
            optical_flow_method: 光流计算方法 ('farneback' or 'lucas_kanade') (NEW)
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone
            frame_interval: 帧采样间隔
            use_oversampling: 是否使用过采样
            crop_size: 裁剪尺寸
            filter_occlusion: 是否过滤遮挡样本（保留Fully_visible和Mostly_visible）
            filter_edge_cases: 是否过滤边缘样本（过滤左右边界附近的box）
            edge_threshold: 边缘阈值（像素）
        """
        self.data_path = data_path
        self.split = split
        self.backbone_name = backbone_name
        self.visual_feature_dim = visual_feature_dim
        self.use_geometric = use_geometric
        self.use_scene_context = use_scene_context
        self.use_optical_flow = use_optical_flow
        self.optical_flow_method = optical_flow_method
        self.frame_interval = frame_interval
        self.use_oversampling = use_oversampling
        self.filter_occlusion = filter_occlusion
        self.filter_edge_cases = filter_edge_cases
        self.edge_threshold = edge_threshold

        # Initialize optical flow extractor if needed
        if self.use_optical_flow:
            self.optical_flow_extractor = OpticalFlowExtractor(
                method=optical_flow_method,
                cache_enabled=True,
                compensate_ego_motion=True  # Enable ego-motion compensation for moving robot
            )
            print(f"[INFO] Optical flow enabled: method={optical_flow_method}, ego-motion compensation=ON")
        else:
            self.optical_flow_extractor = None
        
        # 创建标签映射器
        self.label_mapper = Stage2LabelMapper()
        
        # 创建特征融合器
        self.feature_fusion = ResNetRelationFeatureFusion(
            backbone_name=backbone_name,
            visual_feature_dim=visual_feature_dim,
            use_geometric=use_geometric,
            use_scene_context=use_scene_context,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            crop_size=crop_size
        )

        # store crop size
        self.crop_size = crop_size

        self.trainset_split = [
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
        #'gates-to-clark-2019-02-28_0',
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
        #'nvidia-aud-2019-04-18_2',
        'nvidia-aud-2019-01-25_0',
        'outdoor-coupa-cafe-2019-02-06_0',
        'quarry-road-2019-02-28_0',
        'serra-street-2019-01-30_0',
        'stlc-111-2019-04-19_0', 
        # 'stlc-111-2019-04-19_1', 
        #'stlc-111-2019-04-19_2',
        'packard-poster-session-2019-03-20_2', 
        # 'packard-poster-session-2019-03-20_0',
        # 'packard-poster-session-2019-03-20_1',
        # 'svl-meeting-gates-2-2019-04-08_0',
        'svl-meeting-gates-2-2019-04-08_1',
        # 'tressider-2019-03-16_0',
        'tressider-2019-03-16_2',
        # 'tressider-2019-03-16_1',
        #'tressider-2019-04-26_0',
        'tressider-2019-04-26_1',
        'tressider-2019-04-26_2',
        # 'tressider-2019-04-26_3',
        # 'jordan-hall-2019-04-22_0', 
        ]
        
        self.valset_split = [
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
        
        self.testset_split = [
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
        
        # 加载数据
        self._load_data()

        # Dataset returns cropped image tensors; backbone is inside model for finetuning
        # Debug prints (use instance attributes so they run during construction, not import)
        print(f"ResNet Stage2 Dataset created:")
        print(f"  Split: {self.split}")
        print(f"  Backbone: {self.backbone_name}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Visual features: {self.visual_feature_dim}D")
        print(f"  Frame interval: {self.frame_interval}")

    def _load_data(self):
        """加载JRDB格式的社交标签数据"""
        # 使用JRDB数据结构
        social_labels_dir = os.path.join(self.data_path, 'labels', 'labels_2d_activity_social_stitched')
        images_dir = os.path.join(self.data_path, 'images', 'image_stitched')
        
        if not os.path.exists(social_labels_dir):
            raise FileNotFoundError(f"Social labels directory not found: {social_labels_dir}")
        
        # 根据split选择场景
        if self.split == 'train':
            scene_splits = self.trainset_split
        elif self.split == 'val':
            scene_splits = self.valset_split
        elif self.split == 'test':
            scene_splits = self.testset_split
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        self.samples = []
        self.stage2_labels = []
        
        # 获取场景文件
        scene_files = [f for f in os.listdir(social_labels_dir) if f.endswith('.json')]
        scene_files.sort()
        
        # 筛选存在的场景文件
        selected_files = []
        for scene_name in scene_splits:
            scene_file = f"{scene_name}.json"
            if scene_file in scene_files:
                selected_files.append(scene_file)
            else:
                print(f"Warning: Scene file {scene_file} not found in dataset")
        
        print(f"Loading {len(selected_files)}/{len(scene_splits)} scenes for {self.split} split")
        
        # 加载数据
        sample_count = 0
        filtered_occlusion = 0  # 因遮挡过滤的样本数
        filtered_edge = 0  # 因边缘过滤的样本数
        for scene_file in selected_files:
            scene_path = os.path.join(social_labels_dir, scene_file)
            scene_name = os.path.splitext(scene_file)[0]
            
            try:
                with open(scene_path, 'r') as f:
                    scene_data = json.load(f)
                
                # 处理场景中的每一帧
                frame_names = list(scene_data.get('labels', {}).keys())
                frame_names.sort()

                # 应用帧间隔采样：从索引1开始（而不是0），这样所有目标帧都有前一帧
                # 例如: frame_interval=10 → indices=[1, 11, 21, 31, ...]
                # selected_frames只包含目标帧，前一帧从完整frame_names中查找
                sampled_indices = list(range(1, len(frame_names), self.frame_interval))
                selected_frames = [frame_names[idx] for idx in sampled_indices]
                
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
                                # 提取occlusion信息用于数据清洗
                                occlusion = ann.get('attributes', {}).get('occlusion', 'unknown')
                                person_dict[pid] = {
                                    'box': box,
                                    'occlusion': occlusion,
                                    'interactions': ann.get('H-interaction', []) or ann.get('HHI', [])
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
                                        
                                        # 检查是否为有效的Stage2交互
                                        if isinstance(interaction_labels, dict) and len(interaction_labels) > 0:
                                            interaction_type = list(interaction_labels.keys())[0]
                                            
                                            # 映射到Stage2标签
                                            stage2_label = self.label_mapper.map_label(interaction_type)

                                            if stage2_label is not None:
                                                # ========== 数据清洗：过滤不合格样本 ==========
                                                # 获取person A和B的occlusion信息
                                                person_A_occlusion = person_dict[person_A_id].get('occlusion', 'unknown')
                                                person_B_occlusion = person_dict[person_B_id].get('occlusion', 'unknown')

                                                # 过滤条件1：occlusion检查
                                                if self.filter_occlusion:
                                                    if not (self._is_valid_occlusion(person_A_occlusion) and
                                                            self._is_valid_occlusion(person_B_occlusion)):
                                                        filtered_occlusion += 1
                                                        continue  # 跳过此交互对

                                                # 过滤条件2：边缘box检查
                                                if self.filter_edge_cases:
                                                    if (self._is_edge_box(person_A_box, threshold=self.edge_threshold) or
                                                        self._is_edge_box(person_B_box, threshold=self.edge_threshold)):
                                                        filtered_edge += 1
                                                        continue  # 跳过此交互对
                                                # ============================================

                                                # Find previous frame for optical flow
                                                # Look for the previous frame in the complete frame_names sequence
                                                prev_image_path = None
                                                current_idx = frame_names.index(image_name)
                                                if current_idx > 0:
                                                    prev_frame_name = frame_names[current_idx - 1]
                                                    prev_image_path = os.path.join(images_dir, scene_name, prev_frame_name)
                                                    # Verify file exists
                                                    if not os.path.exists(prev_image_path):
                                                        prev_image_path = None

                                                sample = {
                                                    'image_path': image_path if os.path.exists(image_path) else None,
                                                    'prev_image_path': prev_image_path,
                                                    'person_A_box': torch.tensor(person_A_box, dtype=torch.float32),
                                                    'person_B_box': torch.tensor(person_B_box, dtype=torch.float32),
                                                    'stage2_label': stage2_label,
                                                    'scene_name': scene_name,
                                                    'frame_id': frame_id,
                                                    'all_boxes': all_boxes,
                                                    'original_interaction': interaction_type
                                                }

                                                self.samples.append(sample)
                                                self.stage2_labels.append(stage2_label)
                                                sample_count += 1
                    
                    # 定期打印进度
                    if sample_count % 1000 == 0 and sample_count > 0:
                        print(f"  Processed {sample_count} samples...")
                        
            except Exception as e:
                print(f"Warning: Error loading scene {scene_file}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} samples from {len(selected_files)} scenes")
        if self.frame_interval > 1:
            print(f"Applied frame interval {self.frame_interval} (reduced samples by ~{((self.frame_interval-1)/self.frame_interval)*100:.0f}%)")
        
        # 打印类别分布
        if self.stage2_labels:
            label_counts = Counter(self.stage2_labels)
            print(f"Label distribution: {dict(label_counts)}")

        # 打印数据清洗统计
        if self.filter_occlusion or self.filter_edge_cases:
            print(f"\n{'='*60}")
            print("DATA FILTERING STATISTICS")
            print(f"{'='*60}")
            total_filtered = filtered_occlusion + filtered_edge
            total_before_filter = len(self.samples) + total_filtered
            print(f"Total samples before filtering: {total_before_filter}")
            print(f"Total samples after filtering: {len(self.samples)}")
            print(f"Total filtered: {total_filtered} ({total_filtered/total_before_filter*100:.2f}%)")
            if self.filter_occlusion:
                print(f"  - Filtered by occlusion (not Fully/Mostly_visible): {filtered_occlusion}")
            if self.filter_edge_cases:
                print(f"  - Filtered by edge cases (within {self.edge_threshold}px of left/right border): {filtered_edge}")
            print(f"{'='*60}\n")

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

    def _is_valid_occlusion(self, occlusion: str) -> bool:
        """
        检查遮挡状态是否有效
        仅保留Fully_visible和Mostly_visible
        """
        return occlusion in ['Fully_visible', 'Mostly_visible']

    def _is_edge_box(self, box: List[float], image_width: int = 3760, threshold: int = 200) -> bool:
        """
        检查边界框是否在图像左右边缘附近
        Args:
            box: [x, y, w, h]
            image_width: 图像宽度（默认3760）
            threshold: 边界阈值（像素）
        Returns:
            True表示box在边缘区域，应该被过滤
        """
        x, y, w, h = box
        # 检查左边界和右边界
        return x < threshold or (x + w) > (image_width - threshold)

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据样本
        
        Returns:
            Dict包含:
            - person_A_features: [visual_feature_dim] A的视觉特征
            - person_B_features: [visual_feature_dim] B的视觉特征  
            - spatial_features: [spatial_feature_dim] 空间关系特征
            - stage2_label: int 行为标签
        """
        sample = self.samples[idx]

        # 获取图像
        image = None
        image_path = sample['image_path']
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")

        # Return cropped image tensors for person A and B (so model performs backbone forward)
        # PersonCropExtractor has an internal preprocess that resizes to crop_size
        if image is not None:
            person_A_img_tensor = self.feature_fusion.person_extractor.crop_person_region(image, sample['person_A_box'], image_width=3760, image_height=480)
            person_B_img_tensor = self.feature_fusion.person_extractor.crop_person_region(image, sample['person_B_box'], image_width=3760, image_height=480)

            # preprocess to tensor
            person_A_tensor = self.feature_fusion.person_extractor.crop_preprocess(person_A_img_tensor)  # [3,H,W]
            person_B_tensor = self.feature_fusion.person_extractor.crop_preprocess(person_B_img_tensor)
            # Return per-sample tensors with shape [3,H,W]. DataLoader will stack them into [B,3,H,W].
        else:
            # fallback zero images
            person_A_tensor = torch.zeros(3, self.crop_size, self.crop_size, dtype=torch.float32)
            person_B_tensor = torch.zeros(3, self.crop_size, self.crop_size, dtype=torch.float32)

        # spatial features: compute using feature_fusion helpers (without computing visual features)
        spatial_feats = []
        if self.use_geometric:
            try:
                geom = extract_geometric_features(sample['person_A_box'], sample['person_B_box'], 3760, 480)
                if isinstance(geom, torch.Tensor):
                    geom = geom.cpu().numpy()
                spatial_feats.append(torch.tensor(geom, dtype=torch.float32))
            except Exception:
                spatial_feats.append(torch.zeros(7, dtype=torch.float32))
        if self.use_scene_context:
            try:
                scene = self.feature_fusion.scene_extractor(sample.get('all_boxes', []))
                spatial_feats.append(scene)
            except Exception:
                spatial_feats.append(torch.zeros(1, dtype=torch.float32))

        # Optical flow features (NEW)
        if self.use_optical_flow:
            try:
                prev_image_path = sample.get('prev_image_path')
                if prev_image_path is not None and image is not None and os.path.exists(prev_image_path):
                    # Load previous frame
                    prev_image = Image.open(prev_image_path).convert('RGB')

                    # Extract optical flow features for this person pair
                    flow_result = self.optical_flow_extractor.extract_person_pair_optical_flow(
                        prev_image, image,
                        sample['person_A_box'], sample['person_B_box'],
                        image_width=3760, image_height=480
                    )

                    # Use combined flow features (8D)
                    optical_flow_feats = flow_result['combined_flow_features']
                    spatial_feats.append(optical_flow_feats)
                else:
                    # No previous frame available, use zero features
                    spatial_feats.append(torch.zeros(8, dtype=torch.float32))
            except Exception as e:
                # Optical flow computation failed, use zero features
                # print(f"Warning: Optical flow computation failed: {e}")
                spatial_feats.append(torch.zeros(8, dtype=torch.float32))

        if spatial_feats:
            spatial_tensor = torch.cat(spatial_feats, dim=0)
        else:
            spatial_tensor = torch.zeros(self.feature_fusion.get_spatial_feature_dim(), dtype=torch.float32)

        return {
            'person_A_features': person_A_tensor,  # [3,H,W]
            'person_B_features': person_B_tensor,
            'spatial_features': spatial_tensor,    # [spatial_dim]
            'stage2_label': torch.tensor(sample['stage2_label'], dtype=torch.long)
        }

    def _get_cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"{self.split}_resnet_features.pth")

    def _ensure_feature_cache(self):
        """If cache exists, load cached feature tensors into samples; otherwise compute and save cache."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                data = torch.load(cache_path, map_location='cpu')
                if not isinstance(data, dict):
                    raise ValueError("Invalid cache format")
                # load cached features into samples
                for i, feat in enumerate(data.get('features', [])):
                    if i < len(self.samples):
                        self.samples[i]['cached_person_A'] = feat['person_A']
                        self.samples[i]['cached_person_B'] = feat['person_B']
                        self.samples[i]['cached_spatial'] = feat['spatial']
                print(f"Loaded feature cache: {cache_path}")
                return
            except Exception as e:
                print(f"Warning: Failed to load feature cache: {e}, will recompute")

        # compute and cache
        print(f"Precomputing visual+spatial features for {len(self.samples)} samples (this may take a while)...")
        features_list = []
        for i, sample in enumerate(self.samples):
            image = None
            image_path = sample['image_path']
            if image_path and os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception:
                    image = None

            feats = self.feature_fusion(
                person_A_box=sample['person_A_box'],
                person_B_box=sample['person_B_box'],
                image=image,
                all_boxes=sample.get('all_boxes', [])
            )

            person_A_feat = feats['person_A_features'].detach().cpu()
            person_B_feat = feats['person_B_features'].detach().cpu()
            spatial_feat = feats['spatial_features'].detach().cpu()

            # ensure fixed spatial dim
            if spatial_feat.numel() != self.feature_fusion.get_spatial_feature_dim() and self.feature_fusion.get_spatial_feature_dim() > 0:
                desired = self.feature_fusion.get_spatial_feature_dim()
                cur = spatial_feat.numel()
                if cur < desired:
                    pad = torch.zeros(desired - cur, dtype=torch.float32)
                    spatial_feat = torch.cat([spatial_feat, pad], dim=0)
                else:
                    spatial_feat = spatial_feat[:desired]

            self.samples[i]['cached_person_A'] = person_A_feat
            self.samples[i]['cached_person_B'] = person_B_feat
            self.samples[i]['cached_spatial'] = spatial_feat

            features_list.append({
                'person_A': person_A_feat,
                'person_B': person_B_feat,
                'spatial': spatial_feat
            })

            if (i + 1) % 500 == 0:
                print(f"  Precomputed {i+1}/{len(self.samples)} samples")

        # save cache
        try:
            torch.save({'features': features_list}, cache_path)
            print(f"Saved feature cache to: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save feature cache: {e}")
    
    def get_labels(self) -> List[int]:
        """获取所有样本的标签"""
        return self.stage2_labels.copy()
    
    def get_class_distribution(self) -> Dict:
        """获取类别分布信息"""
        if not self.stage2_labels:
            return {"message": "No labels available"}
        
        label_counts = Counter(self.stage2_labels)
        total = len(self.stage2_labels)
        
        class_names = self.label_mapper.class_names
        
        return {
            'total': total,
            'class_counts': dict(label_counts),
            'class_names': class_names,
            'class_weights': {k: total / (len(label_counts) * v) for k, v in label_counts.items()}
        }
    
    def get_feature_info(self) -> Dict:
        """获取特征信息"""
        return self.feature_fusion.get_feature_info()


# 数据加载器创建函数
def create_resnet_stage2_data_loaders(config) -> Tuple:
    """
    创建ResNet Stage2数据加载器
    
    Args:
        config: ResNetStage2Config配置对象
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from collections import Counter
    
    print(f"Creating ResNet Stage2 data loaders...")
    print(f"  Backbone: {config.backbone_name}")
    print(f"  Visual features: {config.visual_feature_dim}D")
    print(f"  Frame interval: {config.frame_interval}")
    
    # 创建数据集
    train_dataset = ResNetStage2Dataset(
        data_path=config.data_path,
        split='train',
        backbone_name=config.backbone_name,
        visual_feature_dim=config.visual_feature_dim,
        use_geometric=config.use_geometric,
        use_scene_context=config.use_scene_context,
        use_optical_flow=getattr(config, 'use_optical_flow', False),
        optical_flow_method=getattr(config, 'optical_flow_method', 'farneback'),
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        frame_interval=config.frame_interval,
        use_oversampling=True
    )

    val_dataset = ResNetStage2Dataset(
        data_path=config.data_path,
        split='val',
        backbone_name=config.backbone_name,
        visual_feature_dim=config.visual_feature_dim,
        use_geometric=config.use_geometric,
        use_scene_context=config.use_scene_context,
        use_optical_flow=getattr(config, 'use_optical_flow', False),
        optical_flow_method=getattr(config, 'optical_flow_method', 'farneback'),
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        frame_interval=config.frame_interval,
        use_oversampling=False
    )

    test_dataset = ResNetStage2Dataset(
        data_path=config.data_path,
        split='test',
        backbone_name=config.backbone_name,
        visual_feature_dim=config.visual_feature_dim,
        use_geometric=config.use_geometric,
        use_scene_context=config.use_scene_context,
        use_optical_flow=getattr(config, 'use_optical_flow', False),
        optical_flow_method=getattr(config, 'optical_flow_method', 'farneback'),
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        frame_interval=config.frame_interval,
        use_oversampling=False
    )
    
    # 创建训练集采样器（用于类别平衡）
    train_sampler = None
    if config.use_oversampling:
        try:
            labels = train_dataset.get_labels()
            if labels and len(labels) > 0:
                counts = Counter(labels)
                total = len(labels)
                
                # 计算类别权重并更新config
                class_weights = {int(c): total / (len(counts) * counts[c]) for c in counts.keys()}
                config.class_weights = class_weights
                
                # 创建样本权重
                sample_weights = [1.0 / counts[int(l)] for l in labels]
                train_sampler = WeightedRandomSampler(
                    sample_weights, num_samples=len(sample_weights), replacement=True
                )
                print(f"[SUCCESS] Created WeightedRandomSampler: {dict(counts)}")
                print(f"   Class weights: {class_weights}")
        except Exception as e:
            print(f"[WARNING] Failed to create weighted sampler: {e}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # 打印数据集统计信息
    print(f"[SUCCESS] ResNet Stage2 data loaders created:")
    print(f"   Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset):,} samples, {len(val_loader)} batches") 
    print(f"   Test:  {len(test_dataset):,} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试ResNet数据集
    print("Testing ResNet Stage2 Dataset...")
    
    # 由于需要实际数据集，这里只测试配置
    from configs.resnet_stage2_config import get_resnet18_config
    
    config = get_resnet18_config(
        data_path="../dataset",  # 假设路径
        batch_size=4
    )
    
    print(f"Config validation passed")
    print(f"Model info: {config.get_model_info()}")
    
    # 如果有实际数据路径，可以测试数据加载
    test_data_path = "../dataset"
    if os.path.exists(test_data_path):
        print(f"\nTesting with real data at {test_data_path}...")
        try:
            train_loader, val_loader, test_loader = create_resnet_stage2_data_loaders(config)
            print("[SUCCESS] Data loaders created successfully!")
        except Exception as e:
            print(f"[FAILED] Data loading failed: {e}")
    else:
        print(f"[WARNING] Test data path {test_data_path} not found, skipping data loading test")

    print("\n[SUCCESS] ResNet Stage2 dataset test completed!")