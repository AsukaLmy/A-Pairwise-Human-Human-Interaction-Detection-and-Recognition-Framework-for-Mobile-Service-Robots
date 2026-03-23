#!/usr/bin/env python3
"""
HoG features extraction for human interaction detection
Optimized for JRDB panoramic images (3760×480)
"""

import torch
import numpy as np
from PIL import Image
import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# JRDB image dimensions
JRDB_WIDTH = 3760
JRDB_HEIGHT = 480


class HoGFeatureExtractor:
    """
    HoG特征提取器，专门针对JRDB 3760×480全景图像优化
    """
    
    def __init__(self, target_dim=64, orientations=8, pixels_per_cell=(12, 12), 
                 cells_per_block=(2, 2), block_norm='L2-Hys', use_simple_truncation=True):
        """
        Args:
            target_dim: 目标特征维度
            orientations: HoG方向bin数量
            pixels_per_cell: 每个cell的像素大小
            cells_per_block: 每个block的cell数量
            block_norm: 块归一化方法
            use_simple_truncation: 是否使用简单截断（True）或PCA降维（False）
        """
        self.target_dim = target_dim
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.use_simple_truncation = use_simple_truncation
        
        # PCA降维器（延迟初始化，仅在use_simple_truncation=False时使用）
        self.pca_reducer = None
        self.is_fitted = False
        
        if use_simple_truncation:
            print(f"HoGFeatureExtractor: Using simple truncation to {target_dim}D (avoiding PCA)")
        else:
            print(f"HoGFeatureExtractor: Using PCA reduction to {target_dim}D")
    
    def compute_smart_combined_box(self, person_A_box, person_B_box, padding_ratio=0.3):
        """
        计算智能的联合边界框
        
        Args:
            person_A_box: [x, y, width, height]
            person_B_box: [x, y, width, height]
            padding_ratio: padding比例
            
        Returns:
            tuple: (x1, y1, x2, y2) 联合边界框
        """
        if isinstance(person_A_box, torch.Tensor):
            person_A_box = person_A_box.cpu().numpy()
        if isinstance(person_B_box, torch.Tensor):
            person_B_box = person_B_box.cpu().numpy()
        
        # 计算中心点
        center_A = (person_A_box[0] + person_A_box[2]/2, person_A_box[1] + person_A_box[3]/2)
        center_B = (person_B_box[0] + person_B_box[2]/2, person_B_box[1] + person_B_box[3]/2)
        
        # 计算距离
        distance = ((center_A[0] - center_B[0])**2 + (center_A[1] - center_B[1])**2)**0.5
        
        # 基于距离和人体尺寸的自适应padding
        avg_person_size = (person_A_box[2] + person_A_box[3] + person_B_box[2] + person_B_box[3]) / 4
        padding = max(20, min(100, int(avg_person_size * padding_ratio)))
        
        # 计算联合区域
        x1 = min(person_A_box[0], person_B_box[0]) - padding
        y1 = min(person_A_box[1], person_B_box[1]) - padding
        x2 = max(person_A_box[0] + person_A_box[2], person_B_box[0] + person_B_box[2]) + padding
        y2 = max(person_A_box[1] + person_A_box[3], person_B_box[1] + person_B_box[3]) + padding
        
        # 边界检查（使用正确的JRDB尺寸）
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(JRDB_WIDTH, int(x2))
        y2 = min(JRDB_HEIGHT, int(y2))
        
        # 确保区域有效：检查是否为空区域
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            print(f"Warning: Invalid bounding box computed: ({x1}, {y1}, {x2}, {y2}), size=({width}x{height})")
            # 使用最小有效区域
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            min_size = 48  # 确保至少48x48用于HoG
            x1 = max(0, center_x - min_size//2)
            y1 = max(0, center_y - min_size//2)
            x2 = min(JRDB_WIDTH, x1 + min_size)
            y2 = min(JRDB_HEIGHT, y1 + min_size)
        
        return (x1, y1, x2, y2)
    
    def resize_with_aspect_ratio(self, image, max_height=240, max_width=480):
        """
        保持长宽比的智能resize
        
        Args:
            image: PIL Image
            max_height: 最大高度
            max_width: 最大宽度
            
        Returns:
            PIL Image: resize后的图像
        """
        original_width, original_height = image.size
        
        # 计算缩放比例
        height_ratio = max_height / original_height
        width_ratio = max_width / original_width
        
        # 选择较小的比例以保证不超出限制
        scale_ratio = min(height_ratio, width_ratio)
        
        # 计算新尺寸
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)
        
        # 确保尺寸至少为16×16（HoG最小要求）
        new_width = max(16, new_width)
        new_height = max(16, new_height)
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def extract_raw_hog(self, image):
        """
        提取原始HoG特征
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            numpy.ndarray: HoG特征向量
        """
        # 转换为numpy数组和灰度图
        if isinstance(image, Image.Image):
            image_array = np.array(image.convert('L'))
        else:
            # 如果是彩色numpy数组，转为灰度
            if len(image.shape) == 3:
                image_array = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                image_array = image
        
        # 检查图像尺寸
        original_shape = image_array.shape
        if image_array.shape[0] < 16 or image_array.shape[1] < 16:
            print(f"Warning: Image too small ({original_shape}), resizing to (32, 32)")
            image_array = cv2.resize(image_array, (32, 32))
        
        try:
            # 计算期望的HoG特征维度（用于验证）
            height, width = image_array.shape
            cells_y = height // self.pixels_per_cell[0]
            cells_x = width // self.pixels_per_cell[1] 
            blocks_y = max(0, cells_y - self.cells_per_block[0] + 1)
            blocks_x = max(0, cells_x - self.cells_per_block[1] + 1)
            expected_dim = blocks_y * blocks_x * self.cells_per_block[0] * self.cells_per_block[1] * self.orientations
            
            if blocks_y <= 0 or blocks_x <= 0:
                print(f"Warning: Image size ({height}x{width}) too small for HoG parameters, expected blocks=({blocks_x}x{blocks_y})")
                return None
                
            # 提取HoG特征
            hog_features = hog(
                image_array,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                feature_vector=True
            )
            
            # 验证返回的特征维度
            if len(hog_features) != expected_dim:
                print(f"Warning: HoG dimension mismatch - got {len(hog_features)}, expected {expected_dim}")
            
            if len(hog_features) == 0:
                print(f"Warning: HoG extraction returned empty features for image shape {image_array.shape}")
                return None
                
            return hog_features
        except Exception as e:
            print(f"HoG extraction error: {e}")
            print(f"Image shape: {image_array.shape}, HoG params: orientations={self.orientations}, pixels_per_cell={self.pixels_per_cell}")
            # 返回 None，让上层在降维阶段统一处理
            return None
    
    def fit_pca_reducer(self, sample_features_list):
        """
        使用样本特征训练PCA降维器
        
        Args:
            sample_features_list: list of numpy arrays，样本HoG特征列表
        """
        if len(sample_features_list) == 0:
            print("Warning: No sample features provided for PCA fitting")
            return
        
        # 堆叠所有特征
        features_matrix = np.vstack(sample_features_list)
        
        # 训练PCA
        n_components = min(self.target_dim, features_matrix.shape[1], features_matrix.shape[0])
        self.pca_reducer = PCA(n_components=n_components)
        self.pca_reducer.fit(features_matrix)
        self.is_fitted = True
        
        print(f"PCA reducer fitted: {features_matrix.shape[1]} -> {n_components} dimensions")
        print(f"Explained variance ratio: {self.pca_reducer.explained_variance_ratio_.sum():.3f}")
    
    def reduce_hog_dimensions(self, hog_features):
        """
        根据配置使用简单截断或PCA来处理HoG特征维度
        
        Args:
            hog_features: numpy array, 原始HoG特征
            
        Returns:
            numpy array: 处理后的特征
        """
        # 处理 extract_raw_hog 返回的异常情况
        if hog_features is None:
            print("Warning: HoG features is None, returning zeros")
            return np.zeros(self.target_dim, dtype=np.float32)
        
        # 确保输入是numpy数组
        if not isinstance(hog_features, np.ndarray):
            hog_features = np.array(hog_features, dtype=np.float32)
        
        if self.use_simple_truncation:
            # 使用智能采样策略（避免PCA的复杂性和零值问题）
            if len(hog_features) >= self.target_dim:
                # 如果原始特征维度大于等于目标维度，使用均匀采样而不是简单截断
                # 这确保我们从整个特征空间中获取信息，而不只是前N维
                indices = np.linspace(0, len(hog_features)-1, self.target_dim, dtype=int)
                sampled_features = hog_features[indices].astype(np.float32)
                
                # 检查采样结果的有效性
                zero_ratio = (sampled_features == 0).sum() / len(sampled_features)
                if zero_ratio > 0.9:  # 如果超过90%是零
                    print(f"Warning: Sampled HoG features have {zero_ratio*100:.1f}% zeros, trying alternative sampling")
                    # 尝试选择非零特征较多的区域
                    non_zero_indices = np.nonzero(hog_features)[0]
                    if len(non_zero_indices) >= self.target_dim:
                        selected_indices = np.linspace(0, len(non_zero_indices)-1, self.target_dim, dtype=int)
                        sampled_features = hog_features[non_zero_indices[selected_indices]].astype(np.float32)
                        print(f"Used non-zero feature sampling, new zero ratio: {(sampled_features == 0).sum() / len(sampled_features)*100:.1f}%")
                
                return sampled_features
            else:
                # 如果原始特征维度小于目标维度，优先保留原始特征，然后补零
                padded = np.zeros(self.target_dim, dtype=np.float32)
                padded[:len(hog_features)] = hog_features
                print(f"Warning: HoG features ({len(hog_features)}D) smaller than target ({self.target_dim}D), padding with zeros")
                return padded
        else:
            # 使用原来的PCA降维逻辑
            if not self.is_fitted:
                # 如果PCA未训练，直接截断或补零到目标维度
                if len(hog_features) > self.target_dim:
                    return hog_features[:self.target_dim]
                else:
                    padded = np.zeros(self.target_dim, dtype=np.float32)
                    padded[:len(hog_features)] = hog_features
                    return padded

            # 使用训练好的PCA降维
            expected = getattr(self.pca_reducer, 'n_features_', None)
            if expected is None:
                features_reshaped = hog_features.reshape(1, -1)
                reduced_features = self.pca_reducer.transform(features_reshaped)[0]
            else:
                if len(hog_features) != expected:
                    adjusted = np.zeros(expected, dtype=np.float32)
                    copy_len = min(len(hog_features), expected)
                    adjusted[:copy_len] = hog_features[:copy_len]
                    hog_features = adjusted

                features_reshaped = hog_features.reshape(1, -1)
                reduced_features = self.pca_reducer.transform(features_reshaped)[0]

            # 确保降维结果符合 target_dim
            if len(reduced_features) < self.target_dim:
                padded = np.zeros(self.target_dim, dtype=np.float32)
                padded[:len(reduced_features)] = reduced_features
                return padded

            return reduced_features[:self.target_dim]
    
    def extract_joint_hog_features(self, image, person_A_box, person_B_box):
        """
        提取联合区域的HoG特征（主要方法）
        
        Args:
            image: PIL Image, 原始图像
            person_A_box: [x, y, width, height]
            person_B_box: [x, y, width, height]
            
        Returns:
            torch.Tensor: [target_dim] HoG特征向量
        """
        try:
            # 1. 计算智能联合边界框
            combined_box = self.compute_smart_combined_box(person_A_box, person_B_box)
            
            # 2. 裁剪联合区域
            joint_region = image.crop(combined_box)
            
            # 3. 保持长宽比的resize
            joint_region = self.resize_with_aspect_ratio(joint_region)
            
            # 4. 提取原始HoG特征
            raw_hog = self.extract_raw_hog(joint_region)
            
            # 5. 降维到目标维度
            reduced_hog = self.reduce_hog_dimensions(raw_hog)
            
            return torch.tensor(reduced_hog, dtype=torch.float32)
            
        except Exception as e:
            print(f"Joint HoG extraction failed: {e}")
            # 返回零向量
            return torch.zeros(self.target_dim, dtype=torch.float32)


# 全局HoG提取器实例
_global_hog_extractor = None

def get_hog_extractor():
    """获取全局HoG提取器实例"""
    global _global_hog_extractor
    if _global_hog_extractor is None:
        _global_hog_extractor = HoGFeatureExtractor(target_dim=64, use_simple_truncation=True)
    return _global_hog_extractor


def extract_joint_hog_features(image, person_A_box, person_B_box):
    """
    便捷函数：提取联合HoG特征
    
    Args:
        image: PIL Image or image path
        person_A_box: [x, y, width, height]
        person_B_box: [x, y, width, height]
        
    Returns:
        torch.Tensor: [64] HoG特征向量
    """
    # 处理图像路径
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    extractor = get_hog_extractor()
    return extractor.extract_joint_hog_features(image, person_A_box, person_B_box)


def batch_extract_hog_features(image_paths, person_A_boxes, person_B_boxes):
    """
    批量提取HoG特征（用于PCA训练）
    
    Args:
        image_paths: list of image paths
        person_A_boxes: list of [x, y, width, height]
        person_B_boxes: list of [x, y, width, height]
        
    Returns:
        list of numpy arrays: HoG特征列表
    """
    extractor = get_hog_extractor()
    hog_features_list = []
    
    for i, image_path in enumerate(image_paths):
        try:
            image = Image.open(image_path).convert('RGB')
            hog_features = extractor.extract_joint_hog_features(
                image, person_A_boxes[i], person_B_boxes[i]
            )
            hog_features_list.append(hog_features.numpy())
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            hog_features_list.append(np.zeros(64, dtype=np.float32))
    
    return hog_features_list


if __name__ == "__main__":
    """测试HoG特征提取"""
    # 创建测试数据
    test_image = Image.new('RGB', (3760, 480), (128, 128, 128))
    person_A_box = [1000, 200, 100, 200]  # [x, y, w, h]
    person_B_box = [1200, 180, 120, 220]
    
    # 测试特征提取
    hog_features = extract_joint_hog_features(test_image, person_A_box, person_B_box)
    
    print(f"HoG特征形状: {hog_features.shape}")
    print(f"HoG特征统计: min={hog_features.min():.3f}, max={hog_features.max():.3f}, mean={hog_features.mean():.3f}")
    
    # 测试批量提取
    extractor = get_hog_extractor()
    print(f"HoG提取器配置: orientations={extractor.orientations}, "
          f"pixels_per_cell={extractor.pixels_per_cell}, target_dim={extractor.target_dim}")