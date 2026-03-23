#!/usr/bin/env python3
"""
Feature Extractors for Stage2 Behavior Classification
Modular design supporting geometric features and HoG features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import os
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

# Add src path for imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

try:
    from src.features.geometric_features import extract_geometric_features
    from src.features.hog_features import extract_joint_hog_features
except ImportError as e:
    print(f"Warning: Could not import feature extraction functions: {e}")
    print("Please ensure geometric_features.py and hog_features.py are available")


class GeometricFeatureExtractor(nn.Module):
    """
    几何特征提取器 - 继承Stage1的7维几何特征
    提取两人之间的空间关系特征
    """
    
    def __init__(self):
        super().__init__()
        self.feature_dim = 7
        self.feature_names = [
            'relative_distance',     # 相对距离
            'relative_position_x',   # 水平相对位置  
            'relative_position_y',   # 垂直相对位置
            'box_size_ratio',       # 尺寸比例
            'overlap_ratio',        # 重叠度
            'angle_between',        # 夹角
            'distance_normalized'   # 归一化距离
        ]
    
    def forward(self, person_A_box: torch.Tensor, person_B_box: torch.Tensor, 
                image_width: int = 3760, image_height: int = 480) -> torch.Tensor:
        """
        提取几何特征
        
        Args:
            person_A_box: [batch_size, 4] or [4] 边界框 [x, y, w, h]
            person_B_box: [batch_size, 4] or [4] 边界框 [x, y, w, h]
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            torch.Tensor: [batch_size, 7] or [7] 几何特征向量
        """
        # 处理批次维度
        single_sample = False
        if person_A_box.dim() == 1:
            person_A_box = person_A_box.unsqueeze(0)
            person_B_box = person_B_box.unsqueeze(0)
            single_sample = True
        
        batch_size = person_A_box.size(0)
        geometric_features = []
        
        # 逐样本提取特征（使用现有函数）
        for i in range(batch_size):
            try:
                features = extract_geometric_features(
                    person_A_box[i], person_B_box[i], image_width, image_height
                )
                geometric_features.append(features)
            except Exception as e:
                print(f"Warning: Geometric feature extraction failed for sample {i}: {e}")
                # 使用零向量作为fallback
                geometric_features.append(torch.zeros(7))
        
        result = torch.stack(geometric_features)
        
        if single_sample:
            result = result.squeeze(0)
            
        return result
    
    def get_feature_names(self):
        """获取特征名称"""
        return self.feature_names


class HoGFeatureExtractor(nn.Module):
    """
    HoG视觉特征提取器 - 提取64维HoG特征
    基于两人的边界框区域提取视觉外观特征
    """
    
    def __init__(self):
        super().__init__()
        self.feature_dim = 64
        
    def forward(self, image: Optional[Image.Image], 
                person_A_box: torch.Tensor, person_B_box: torch.Tensor) -> torch.Tensor:
        """
        提取HoG特征
        
        Args:
            image: PIL Image对象，如果为None则返回零向量
            person_A_box: [4] 边界框 [x, y, w, h]
            person_B_box: [4] 边界框 [x, y, w, h]
            
        Returns:
            torch.Tensor: [64] HoG特征向量
        """
        if image is None:
            # print(f"Warning: No image provided for HoG extraction, using zeros")
            return torch.zeros(self.feature_dim, dtype=torch.float32)
        
        try:
            # 使用现有的HoG提取函数
            hog_features = extract_joint_hog_features(image, person_A_box, person_B_box)
            
            # # 检查提取的特征是否有效
            # if hog_features is None:
            #     print(f"Warning: HoG extraction returned None")
            #     return torch.zeros(self.feature_dim, dtype=torch.float32)
                
            # if torch.all(hog_features == 0):
            #     print(f"Warning: HoG extraction returned all zeros")
                
            return hog_features
        except Exception as e:
            print(f"Warning: HoG feature extraction failed: {e}")
            return torch.zeros(self.feature_dim, dtype=torch.float32)


class SceneContextExtractor(nn.Module):
    """
    场景上下文特征提取器 - 提取场景密度等上下文信息
    """
    
    def __init__(self):
        super().__init__()
        self.feature_dim = 1
        
    def forward(self, all_boxes: list) -> torch.Tensor:
        """
        提取场景上下文特征
        
        Args:
            all_boxes: 当前帧中所有人的边界框列表
            
        Returns:
            torch.Tensor: [1] 场景上下文特征 (场景密度)
        """
        if not all_boxes:
            return torch.tensor([1.0], dtype=torch.float32)  # 默认稀疏场景

        
        # 增强的场景上下文计算
        num_people = len(all_boxes)
        scene_density = min(num_people / 10.0, 1.0)  # 归一化到[0,1]
        return torch.tensor([scene_density], dtype=torch.float32)



class BasicFeatureFusion(nn.Module):
    """
    Basic模式特征融合器
    将几何特征和HoG特征融合为统一的特征向量
    """
    
    def __init__(self, use_geometric: bool = True, use_hog: bool = True, 
                 use_scene_context: bool = True):
        super().__init__()
        self.use_geometric = use_geometric
        self.use_hog = use_hog
        self.use_scene_context = use_scene_context
        
        # 初始化特征提取器
        if self.use_geometric:
            self.geometric_extractor = GeometricFeatureExtractor()
            
        if self.use_hog:
            self.hog_extractor = HoGFeatureExtractor()
            
        if self.use_scene_context:
            self.scene_extractor = SceneContextExtractor()
        
        # 计算输出维度
        self.output_dim = self._calculate_output_dim()
        
    def _calculate_output_dim(self) -> int:
        """计算融合后的特征维度"""
        dim = 0
        if self.use_geometric:
            dim += 7      # 几何特征
        if self.use_hog:
            dim += 64     # HoG特征
        if self.use_scene_context:
            dim += 1      # 场景上下文
        return dim
    
    def forward(self, person_A_box: torch.Tensor, person_B_box: torch.Tensor,
                image: Optional[Image.Image] = None, 
                all_boxes: Optional[list] = None,
                image_width: int = 3760, image_height: int = 480) -> torch.Tensor:
        """
        融合多种特征
        
        Args:
            person_A_box: [4] 边界框
            person_B_box: [4] 边界框  
            image: PIL Image对象 (HoG特征需要)
            all_boxes: 所有人的边界框列表 (场景上下文需要)
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            torch.Tensor: [output_dim] 融合特征向量
        """
        feature_components = []

        # Helper to ensure batch dimension: return tensor with shape [B, D]
        def _ensure_batch(t: torch.Tensor):
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.float32)
            if t.dim() == 1:
                return t.unsqueeze(0)
            return t

        single_sample = False
        # If input boxes are single-sample (1D), treat outputs as single-sample and squeeze later
        if person_A_box.dim() == 1:
            single_sample = True

        # 几何特征
        if self.use_geometric:
            geometric_features = self.geometric_extractor(
                person_A_box, person_B_box, image_width, image_height
            )
            geometric_features = _ensure_batch(geometric_features)
            feature_components.append(geometric_features)

        # HoG特征
        if self.use_hog:
            hog_features = self.hog_extractor(image, person_A_box, person_B_box)
            hog_features = _ensure_batch(hog_features)
            feature_components.append(hog_features)

        # 场景上下文特征
        if self.use_scene_context and all_boxes is not None:
            scene_features = self.scene_extractor(all_boxes)
            scene_features = _ensure_batch(scene_features)
            feature_components.append(scene_features)

        # 融合特征：在最后一个维度拼接
        if feature_components:
            # all components now have shape [B, D_i]
            fused_features = torch.cat(feature_components, dim=1)
            
            # # 添加特征标准化 - 提高特征可分性
            # # L2标准化，避免不同特征类型的量级差异
            # feature_norm = torch.norm(fused_features, dim=1, keepdim=True)
            # # 避免除零，添加小的epsilon
            # feature_norm = torch.clamp(feature_norm, min=1e-8)
            # fused_features = fused_features / feature_norm
        else:
            fused_features = torch.zeros((1, 0), dtype=torch.float32)

        if single_sample:
            return fused_features.squeeze(0)
        return fused_features
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.output_dim
    
    def get_feature_info(self) -> dict:
        """获取特征信息"""
        info = {
            'geometric': {'enabled': self.use_geometric, 'dim': 7 if self.use_geometric else 0},
            'hog': {'enabled': self.use_hog, 'dim': 64 if self.use_hog else 0},
            'scene_context': {'enabled': self.use_scene_context, 'dim': 1 if self.use_scene_context else 0},
            'total_dim': self.output_dim
        }
        return info


class RelationFeatureFusion(nn.Module):
    """
    Relation Network模式特征融合器
    分离提取两人的个体特征，支持Relation Network的输入格式
    """
    
    def __init__(self, use_geometric: bool = True, use_hog: bool = True, 
                 use_scene_context: bool = True):
        super().__init__()
        self.use_geometric = use_geometric
        self.use_hog = use_hog
        self.use_scene_context = use_scene_context
        
        # 初始化特征提取器
        if self.use_hog:
            self.hog_extractor = HoGFeatureExtractor()
            
        if self.use_scene_context:
            self.scene_extractor = SceneContextExtractor()
        
        # 计算个人特征维度 (不包含几何特征，因为几何特征是关系特征)
        self.person_feature_dim = self._calculate_person_feature_dim()
        self.spatial_feature_dim = self._calculate_spatial_feature_dim()
        
    def _calculate_person_feature_dim(self) -> int:
        """计算每个人的个体特征维度"""
        dim = 0
        if self.use_hog:
            dim += 32  # 每个人HoG特征的一半
        return dim
    
    def _calculate_spatial_feature_dim(self) -> int:
        """计算空间关系特征维度"""
        dim = 0
        if self.use_geometric:
            dim += 7      # 几何关系特征
        if self.use_scene_context:
            dim += 1      # 场景上下文
        return dim
    
    def forward(self, person_A_box: torch.Tensor, person_B_box: torch.Tensor,
                image: Optional[Image.Image] = None, 
                all_boxes: Optional[list] = None,
                image_width: int = 3760, image_height: int = 480) -> dict:
        """
        分离提取两人特征和空间关系特征
        
        Args:
            person_A_box: [4] 边界框
            person_B_box: [4] 边界框  
            image: PIL Image对象 (HoG特征需要)
            all_boxes: 所有人的边界框列表 (场景上下文需要)
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            dict: {
                'person_A_features': [person_feature_dim] A的个体特征
                'person_B_features': [person_feature_dim] B的个体特征
                'spatial_features': [spatial_feature_dim] 空间关系特征
            }
        """
        # 提取个体特征
        person_A_features = []
        person_B_features = []
        spatial_features = []

        # Helper to ensure batch shape [B, D]
        def _ensure_batch(t: torch.Tensor):
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.float32)
            if t.dim() == 1:
                return t.unsqueeze(0)
            return t

        single_sample = False
        if person_A_box.dim() == 1:
            single_sample = True

        # HoG特征分离 (假设joint_hog为 [64] 或 [B,64])
        if self.use_hog:
            try:
                joint_hog = self.hog_extractor(image, person_A_box, person_B_box)
                joint_hog = _ensure_batch(joint_hog)  # [B, 64]
                # split into two 32-dim vectors along last dim
                if joint_hog.size(-1) >= 64:
                    person_A_hog = joint_hog[:, :32]
                    person_B_hog = joint_hog[:, 32:64]
                else:
                    # fallback zeros
                    B = joint_hog.size(0)
                    person_A_hog = torch.zeros((B, 32), dtype=torch.float32)
                    person_B_hog = torch.zeros((B, 32), dtype=torch.float32)

                person_A_features.append(person_A_hog)
                person_B_features.append(person_B_hog)
            except Exception as e:
                print(f"Warning: HoG feature extraction failed in RelationFeatureFusion: {e}")
                if single_sample:
                    person_A_features.append(torch.zeros(32, dtype=torch.float32))
                    person_B_features.append(torch.zeros(32, dtype=torch.float32))
                else:
                    # assume batch size from boxes
                    B = person_A_box.size(0)
                    person_A_features.append(torch.zeros((B, 32), dtype=torch.float32))
                    person_B_features.append(torch.zeros((B, 32), dtype=torch.float32))

        # 空间关系特征 (几何特征)
        if self.use_geometric:
            try:
                from src.features.geometric_features import extract_geometric_features
                geometric_features = extract_geometric_features(
                    person_A_box, person_B_box, image_width, image_height
                )
                geometric_features = _ensure_batch(geometric_features)
                spatial_features.append(geometric_features)
            except Exception as e:
                print(f"Warning: Geometric feature extraction failed: {e}")
                if single_sample:
                    spatial_features.append(torch.zeros(7, dtype=torch.float32))
                else:
                    B = person_A_box.size(0)
                    spatial_features.append(torch.zeros((B, 7), dtype=torch.float32))

        # 场景上下文特征
        if self.use_scene_context and all_boxes is not None:
            try:
                scene_features = self.scene_extractor(all_boxes)
                scene_features = _ensure_batch(scene_features)
                spatial_features.append(scene_features)
            except Exception as e:
                print(f"Warning: Scene context extraction failed: {e}")
                if single_sample:
                    spatial_features.append(torch.zeros(1, dtype=torch.float32))
                else:
                    B = person_A_box.size(0)
                    spatial_features.append(torch.zeros((B, 1), dtype=torch.float32))

        # 组装特征，保证返回单样本时为1D，批量时为2D
        if person_A_features:
            person_A_tensor = torch.cat(person_A_features, dim=1)  # [B, D]
        else:
            person_A_tensor = torch.zeros((1, 0), dtype=torch.float32) if single_sample else torch.zeros((person_A_box.size(0), 0), dtype=torch.float32)

        if person_B_features:
            person_B_tensor = torch.cat(person_B_features, dim=1)
        else:
            person_B_tensor = torch.zeros((1, 0), dtype=torch.float32) if single_sample else torch.zeros((person_A_box.size(0), 0), dtype=torch.float32)

        if spatial_features:
            spatial_tensor = torch.cat(spatial_features, dim=1)
        else:
            spatial_tensor = torch.zeros((1, 0), dtype=torch.float32) if single_sample else torch.zeros((person_A_box.size(0), 0), dtype=torch.float32)

        if single_sample:
            person_A_tensor = person_A_tensor.squeeze(0)
            person_B_tensor = person_B_tensor.squeeze(0)
            spatial_tensor = spatial_tensor.squeeze(0)

        return {
            'person_A_features': person_A_tensor,
            'person_B_features': person_B_tensor,
            'spatial_features': spatial_tensor
        }
    
    def get_person_feature_dim(self) -> int:
        """获取单个人的特征维度"""
        return self.person_feature_dim
    
    def get_spatial_feature_dim(self) -> int:
        """获取空间关系特征维度"""
        return self.spatial_feature_dim
    
    def get_feature_info(self) -> dict:
        """获取特征信息"""
        info = {
            'person_hog': {'enabled': self.use_hog, 'dim': 32 if self.use_hog else 0},
            'spatial_geometric': {'enabled': self.use_geometric, 'dim': 7 if self.use_geometric else 0},
            'spatial_scene_context': {'enabled': self.use_scene_context, 'dim': 1 if self.use_scene_context else 0},
            'person_feature_dim': self.person_feature_dim,
            'spatial_feature_dim': self.spatial_feature_dim
        }
        return info


if __name__ == '__main__':
    # 测试特征提取器
    print("Testing Feature Extractors...")
    
    # 创建测试数据
    person_A_box = torch.tensor([100, 100, 50, 150], dtype=torch.float32)
    person_B_box = torch.tensor([200, 120, 60, 140], dtype=torch.float32)
    all_boxes = [[100, 100, 50, 150], [200, 120, 60, 140], [300, 110, 55, 145]]
    
    # 测试几何特征提取器
    print("\n1. Testing GeometricFeatureExtractor...")
    geo_extractor = GeometricFeatureExtractor()
    geo_features = geo_extractor(person_A_box, person_B_box)
    print(f"Geometric features shape: {geo_features.shape}")
    print(f"Geometric features: {geo_features}")
    
    # 测试HoG特征提取器 (没有图像)
    print("\n2. Testing HoGFeatureExtractor...")
    hog_extractor = HoGFeatureExtractor()
    hog_features = hog_extractor(None, person_A_box, person_B_box)
    print(f"HoG features shape: {hog_features.shape}")
    print(f"HoG features (first 5): {hog_features[:5]}")
    
    # 测试场景上下文提取器
    print("\n3. Testing SceneContextExtractor...")
    scene_extractor = SceneContextExtractor()
    scene_features = scene_extractor(all_boxes)
    print(f"Scene context features: {scene_features}")
    
    # 测试特征融合器
    print("\n4. Testing BasicFeatureFusion...")
    fusion = BasicFeatureFusion(use_geometric=True, use_hog=True, use_scene_context=True)
    fused_features = fusion(person_A_box, person_B_box, None, all_boxes)
    print(f"Fused features shape: {fused_features.shape}")
    print(f"Expected output dim: {fusion.get_output_dim()}")
    print(f"Feature info: {fusion.get_feature_info()}")
    
    # 测试只使用几何特征
    print("\n5. Testing Geometric-only fusion...")
    geo_only_fusion = BasicFeatureFusion(use_geometric=True, use_hog=False, use_scene_context=False)
    geo_only_features = geo_only_fusion(person_A_box, person_B_box)
    print(f"Geometric-only features shape: {geo_only_features.shape}")
    print(f"Feature info: {geo_only_fusion.get_feature_info()}")
    
    print("\n✅ Feature extractors test completed!")