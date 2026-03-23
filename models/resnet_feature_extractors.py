#!/usr/bin/env python3
"""
ResNet-based Feature Extractors for Stage2 Behavior Classification
Uses pretrained ResNet as backbone for visual feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict
import os
from PIL import Image

# Add src path for imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

try:
    from src.features.geometric_features import extract_geometric_features
except ImportError as e:
    print(f"Warning: Could not import geometric_features: {e}")


class ResNetBackbone(nn.Module):
    """
    预训练ResNet backbone for visual feature extraction
    """
    
    def __init__(self, backbone_name: str = 'resnet18', feature_dim: int = 256, 
                 pretrained: bool = True, freeze_backbone: bool = False, input_size: int = 224):
        """
        Args:
            backbone_name: ResNet架构名称 ('resnet18', 'resnet34', 'resnet50')
            feature_dim: 输出特征维度
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone参数
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # 创建backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # 去除fc层
            backbone_dim = 512

        elif backbone_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            backbone_dim = 512

        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            backbone_dim = 2048

        # VGG系列
        elif backbone_name == 'vgg11':
            vgg = models.vgg11(pretrained=pretrained)
            self.backbone = nn.Sequential(
                vgg.features,
                nn.AdaptiveAvgPool2d((1, 1))  # 统一输出到1x1，保持与ResNet一致
            )
            backbone_dim = 512

        elif backbone_name == 'vgg13':
            vgg = models.vgg13(pretrained=pretrained)
            self.backbone = nn.Sequential(
                vgg.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )
            backbone_dim = 512

        elif backbone_name == 'vgg16':
            vgg = models.vgg16(pretrained=pretrained)
            self.backbone = nn.Sequential(
                vgg.features,
                nn.AdaptiveAvgPool2d((1, 1))  # 关键：统一输出维度
            )
            backbone_dim = 512

        elif backbone_name == 'vgg19':
            vgg = models.vgg19(pretrained=pretrained)
            self.backbone = nn.Sequential(
                vgg.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )
            backbone_dim = 512

        # AlexNet
        elif backbone_name == 'alexnet':
            alexnet = models.alexnet(pretrained=pretrained)
            self.backbone = nn.Sequential(
                alexnet.features,
                nn.AdaptiveAvgPool2d((1, 1))  # 统一输出到1x1
            )
            backbone_dim = 256

        # MobileNetV3 系列
        elif backbone_name == 'mobilenet_v3_small':
            mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
            self.backbone = nn.Sequential(
                mobilenet.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )
            backbone_dim = 576  # MobileNetV3-Small最后一层输出通道数

        elif backbone_name == 'mobilenet_v3_large':
            mobilenet = models.mobilenet_v3_large(pretrained=pretrained)
            self.backbone = nn.Sequential(
                mobilenet.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )
            backbone_dim = 960  # MobileNetV3-Large最后一层输出通道数

        # EfficientNetV2-Small
        elif backbone_name == 'efficientnet_v2_s':
            efficientnet = models.efficientnet_v2_s(pretrained=pretrained)
            self.backbone = nn.Sequential(
                efficientnet.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )
            backbone_dim = 1280  # EfficientNetV2-S最后一层输出通道数

        # Lite-HRNet系列（使用MMPose）
        elif backbone_name == 'litehrnet_18':
            try:
                from mmpose.models.backbones import LiteHRNet

                # 定义Naive Lite-HRNet-18配置（匹配checkpoint）
                litehrnet = LiteHRNet(
                    extra=dict(
                        stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
                        num_stages=3,
                        stages_spec=dict(
                            num_modules=(2, 4, 2),
                            num_branches=(2, 3, 4),
                            num_blocks=(2, 2, 2),
                            module_type=('LITE', 'LITE', 'LITE'),
                            with_fuse=(True, True, True),
                            reduce_ratios=(8, 8, 8),
                            num_channels=(
                                (30, 60),        # 匹配checkpoint: 30, 60
                                (30, 60, 120),   # 匹配checkpoint: 30, 60, 120
                                (30, 60, 120, 240),  # 匹配checkpoint: 30, 60, 120, 240
                            )),
                        with_head=False,  # 不使用姿态估计head，只提取backbone特征
                    )
                )

                # 加载预训练权重
                checkpoint_path = os.path.join(
                    os.path.dirname(__file__),
                    '../naive_litehrnet_18_coco_256x192.pth'
                )

                if pretrained and os.path.exists(checkpoint_path):
                    print(f"Loading Lite-HRNet-18 from {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')

                    # 提取backbone权重
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint

                    # 过滤backbone参数
                    backbone_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('backbone.'):
                            new_key = k.replace('backbone.', '')
                            backbone_dict[new_key] = v

                    litehrnet.load_state_dict(backbone_dict, strict=False)
                    print(f"Loaded {len(backbone_dict)} parameters from checkpoint")
                elif pretrained:
                    print(f"Warning: Checkpoint not found at {checkpoint_path}, using random initialization")

                self.backbone = litehrnet
                backbone_dim = 30  # Lite-HRNet with_head=False只返回最高分辨率特征

            except ImportError as e:
                raise ImportError(f"Lite-HRNet requires mmpose. Install: pip install mmcv-full mmpose\nError: {e}")

        # HRNet系列（使用MMPose预训练权重）
        elif backbone_name in ['hrnet_w18', 'hrnet_w32', 'hrnet_w48']:
            try:
                from mmpose.models.backbones import HRNet

                # HRNet配置映射
                hrnet_configs = {
                    'hrnet_w18': {
                        'channels': (18, 36, 72, 144),
                        'checkpoint': 'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w18-c9e9eb23_20200130.pth'
                    },
                    'hrnet_w32': {
                        'channels': (32, 64, 128, 256),
                        'checkpoint': 'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth'
                    },
                    'hrnet_w48': {
                        'channels': (48, 96, 192, 384),
                        'checkpoint': 'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth'
                    }
                }

                config = hrnet_configs[backbone_name]
                channels = config['channels']

                # 创建HRNet backbone
                hrnet = HRNet(
                    extra=dict(
                        stage1=dict(
                            num_modules=1,
                            num_branches=1,
                            block='BOTTLENECK',
                            num_blocks=(4,),
                            num_channels=(64,)
                        ),
                        stage2=dict(
                            num_modules=1,
                            num_branches=2,
                            block='BASIC',
                            num_blocks=(4, 4),
                            num_channels=channels[:2]
                        ),
                        stage3=dict(
                            num_modules=4,
                            num_branches=3,
                            block='BASIC',
                            num_blocks=(4, 4, 4),
                            num_channels=channels[:3]
                        ),
                        stage4=dict(
                            num_modules=3,
                            num_branches=4,
                            block='BASIC',
                            num_blocks=(4, 4, 4, 4),
                            num_channels=channels
                        )
                    )
                )

                # 加载预训练权重
                if pretrained:
                    import torch.utils.model_zoo as model_zoo
                    print(f"Loading {backbone_name.upper()} pretrained weights from OpenMMLab...")
                    try:
                        state_dict = model_zoo.load_url(config['checkpoint'], map_location='cpu')
                        # 过滤backbone参数
                        if 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                        backbone_dict = {k: v for k, v in state_dict.items() if not k.startswith('keypoint_head')}
                        hrnet.load_state_dict(backbone_dict, strict=False)
                        print(f"Loaded pretrained {backbone_name.upper()} successfully")
                    except Exception as e:
                        print(f"Warning: Failed to load pretrained weights: {e}")
                        print(f"Using random initialization for {backbone_name.upper()}")

                self.backbone = hrnet
                # HRNet返回多尺度特征，取最高分辨率（第一个通道）
                backbone_dim = channels[0]

            except ImportError as e:
                raise ImportError(f"HRNet requires mmpose. Install: pip install mmcv-full mmpose\nError: {e}")

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. "
                           f"Supported: resnet18/34/50, vgg11/13/16/19, alexnet, "
                           f"mobilenet_v3_small/large, efficientnet_v2_s, litehrnet_18, hrnet_w18/w32/w48")

        # 添加自适应池化确保固定输出尺寸（对于ResNet）
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征映射层
        self.feature_mapper = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # 冻结backbone参数
        if freeze_backbone:
            # 冻结features
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 冻结feature_mapper (这样才能完全冻结backbone)
            for param in self.feature_mapper.parameters():
                param.requires_grad = False

        # 图像预处理
        self.input_size = input_size

        # Lite-HRNet需要特殊的矩形输入尺寸 (256x192)
        if backbone_name == 'litehrnet_18':
            resize_size = (192, 256)  # (height, width) for 256x192
        else:
            resize_size = (self.input_size, self.input_size)  # 正方形

        self.preprocess = transforms.Compose([
            transforms.Resize(resize_size),  # 根据backbone调整尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])

        print(f"Created {backbone_name} backbone: {backbone_dim}D -> {feature_dim}D, "
              f"pretrained={pretrained}, frozen={freeze_backbone}")
    
    def freeze_early_layers(self, freeze_layers: int = 3):
        """
        冻结ResNet前几个层（部分冻结策略）
        
        Args:
            freeze_layers: 冻结前几个残差块 (1-4)
        """
        if freeze_layers <= 0:
            return
            
        children = list(self.backbone.children())
        # ResNet structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        # For freeze_layers=1: freeze conv1, bn1, relu, maxpool, layer1
        # For freeze_layers=2: freeze conv1, bn1, relu, maxpool, layer1, layer2, etc.
        
        layers_to_freeze = []
        # Always freeze initial conv, bn, relu, maxpool when freezing any layers
        if len(children) >= 4:
            layers_to_freeze.extend(children[:4])  # conv1, bn1, relu, maxpool
        
        # Add the specified number of residual layers
        start_idx = 4  # Start after initial layers
        end_idx = min(start_idx + freeze_layers, len(children))
        layers_to_freeze.extend(children[start_idx:end_idx])
        
        frozen_count = 0
        for layer in layers_to_freeze:
            for param in layer.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
        
        print(f"Frozen first {freeze_layers} residual blocks of {self.backbone_name} ({frozen_count} parameters)")
    
    def unfreeze_last_layers(self, unfreeze_layers: int = 1):
        """
        解冻ResNet后几个层（渐进式解冻）

        Args:
            unfreeze_layers: 解冻后几个残差块
        """
        children = list(self.backbone.children())
        layers_to_unfreeze = children[-unfreeze_layers:]

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        print(f"Unfrozen last {unfreeze_layers} residual blocks of {self.backbone_name}")

    def freeze_all_features(self):
        """
        冻结所有特征提取层（包括backbone features和feature_mapper）
        只保留分类头（relation network）可训练

        适用场景:
          - 迁移学习初期，只训练分类头
          - 小数据集，防止过拟合
          - 快速原型验证

        等价于 freeze_backbone=True 或 freeze_blocks=99
        """
        frozen_count = 0

        # 冻结backbone features
        for param in self.backbone.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1

        # 冻结feature_mapper
        for param in self.feature_mapper.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1

        total_params = sum(p.numel() for p in self.backbone.parameters()) + \
                      sum(p.numel() for p in self.feature_mapper.parameters())
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad) + \
                          sum(p.numel() for p in self.feature_mapper.parameters() if p.requires_grad)

        print(f"Frozen all feature extraction layers of {self.backbone_name}:")
        print(f"  Frozen parameters: {frozen_count}")
        print(f"  Total backbone params: {total_params:,}")
        print(f"  Remaining trainable in backbone: {trainable_params}")
        print(f"  [OK] Only classification head is trainable now")

    def unfreeze_all_features(self):
        """
        解冻所有特征提取层

        用于渐进式训练的后期阶段
        """
        unfrozen_count = 0

        # 解冻backbone features
        for param in self.backbone.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1

        # 解冻feature_mapper
        for param in self.feature_mapper.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1

        print(f"Unfrozen all feature extraction layers of {self.backbone_name}:")
        print(f"  Unfrozen parameters: {unfrozen_count}")

    def freeze_mobilenet_blocks(self, freeze_blocks: int = 2):
        """
        冻结MobileNetV3前几个blocks (支持渐进式训练)

        MobileNetV3结构:
          - Small: [0]初始Conv + [1-11]InvertedResidual + [12]最后Conv = 13层
          - Large: [0]初始Conv + [1-15]InvertedResidual + [16]最后Conv = 17层

        冻结策略 (每block约3个InvertedResidual):
          - freeze_blocks=0: 不冻结
          - freeze_blocks=1: 冻结[0] + [1-3]  (初始conv + 前3个IR)
          - freeze_blocks=2: 冻结[0] + [1-6]  (初始conv + 前6个IR)
          - freeze_blocks=3: 冻结[0] + [1-9]  (初始conv + 前9个IR)
          - freeze_blocks=4: 冻结[0] + [1-12] (初始conv + 前12个IR, 几乎全部)

        Args:
            freeze_blocks: 冻结块数量 (0-4)
        """
        if freeze_blocks <= 0:
            print(f"MobileNet: No layers frozen (freeze_blocks={freeze_blocks})")
            return

        if 'mobilenet' not in self.backbone_name.lower():
            print(f"Warning: freeze_mobilenet_blocks called on non-MobileNet backbone: {self.backbone_name}")
            return

        # 获取features模块
        # self.backbone是Sequential: [0]=features (Sequential), [1]=AdaptiveAvgPool2d
        features = None
        backbone_children = list(self.backbone.children())
        if len(backbone_children) > 0 and isinstance(backbone_children[0], nn.Sequential):
            features = backbone_children[0]  # 第一个child就是features

        if features is None:
            print("Warning: Could not locate MobileNet features module")
            return

        total_layers = len(features)
        print(f"  MobileNet features has {total_layers} layers")

        # 自适应计算要冻结的层索引 (适配Small和Large)
        # Small: 13层 = [0]Conv + [1-11]IR(11个) + [12]Conv
        # Large: 17层 = [0]Conv + [1-15]IR(15个) + [16]Conv

        # 计算InvertedResidual的数量
        num_ir = sum(1 for layer in features if 'InvertedResidual' in layer.__class__.__name__)
        print(f"  Found {num_ir} InvertedResidual blocks")

        # 每个freeze_block冻结约25%的IR
        # freeze_blocks=1: 冻结前25% IR
        # freeze_blocks=2: 冻结前50% IR
        # freeze_blocks=3: 冻结前75% IR
        # freeze_blocks=4: 冻结前95% IR (几乎全部)

        freeze_ratios = {1: 0.25, 2: 0.50, 3: 0.75, 4: 0.95}
        freeze_ratio = freeze_ratios.get(freeze_blocks, 0.50)

        # 计算要冻结的IR数量
        num_freeze_ir = int(num_ir * freeze_ratio)

        # 构建冻结索引: 初始Conv + 前N个IR
        freeze_indices = [0]  # 总是冻结初始Conv

        ir_count = 0
        for idx, layer in enumerate(features):
            if 'InvertedResidual' in layer.__class__.__name__:
                ir_count += 1
                if ir_count <= num_freeze_ir:
                    freeze_indices.append(idx)

        print(f"  Freezing {len(freeze_indices)} layers (ratio={freeze_ratio:.0%})")

        # 执行冻结
        frozen_count = 0
        frozen_layers = []

        for idx in freeze_indices:
            if idx < len(features):
                layer = features[idx]
                for param in layer.parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                        frozen_count += 1
                frozen_layers.append(f"[{idx}]{layer.__class__.__name__}")

        print(f"Frozen MobileNet blocks (freeze_blocks={freeze_blocks}):")
        print(f"  Frozen layers: {', '.join(frozen_layers)}")
        print(f"  Frozen parameters: {frozen_count}")

        # 验证冻结效果
        total_params = sum(1 for p in self.backbone.parameters() if p.requires_grad)
        print(f"  Remaining trainable params in backbone: {total_params}")

    def unfreeze_mobilenet_blocks(self, unfreeze_blocks: int = 1):
        """
        解冻MobileNetV3后几个blocks (渐进式解冻)

        用于逐步微调策略:
          1. 先冻结全部features, 只训练relation network
          2. 解冻最后1个block (unfreeze_blocks=1)
          3. 解冻最后2个blocks (unfreeze_blocks=2)
          4. 依此类推

        Args:
            unfreeze_blocks: 从后往前解冻的块数 (1-4)
        """
        if unfreeze_blocks <= 0:
            print("MobileNet: No layers unfrozen")
            return

        if 'mobilenet' not in self.backbone_name.lower():
            print(f"Warning: unfreeze_mobilenet_blocks called on non-MobileNet: {self.backbone_name}")
            return

        # 获取features模块
        features = None
        backbone_children = list(self.backbone.children())
        if len(backbone_children) > 0 and isinstance(backbone_children[0], nn.Sequential):
            features = backbone_children[0]

        if features is None:
            print("Warning: Could not locate MobileNet features module")
            return

        total_layers = len(features)
        num_ir = sum(1 for layer in features if 'InvertedResidual' in layer.__class__.__name__)
        print(f"  MobileNet features has {total_layers} layers, {num_ir} InvertedResidual blocks")

        # 自适应计算要解冻的层索引 (从后往前)
        # unfreeze_blocks=1: 解冻最后25% IR
        # unfreeze_blocks=2: 解冻最后50% IR
        # unfreeze_blocks=3: 解冻最后75% IR
        # unfreeze_blocks=4: 解冻最后95% IR (几乎全部)

        unfreeze_ratios = {1: 0.25, 2: 0.50, 3: 0.75, 4: 0.95}
        unfreeze_ratio = unfreeze_ratios.get(unfreeze_blocks, 0.50)

        # 计算要解冻的IR数量 (从后往前)
        num_unfreeze_ir = int(num_ir * unfreeze_ratio)

        # 构建解冻索引: 从后往前数N个IR + 最后的Conv
        unfreeze_indices = []

        # 先找到所有IR的索引
        ir_indices = [idx for idx, layer in enumerate(features)
                     if 'InvertedResidual' in layer.__class__.__name__]

        # 取最后N个IR
        if num_unfreeze_ir > 0:
            unfreeze_indices.extend(ir_indices[-num_unfreeze_ir:])

        # 添加最后的Conv
        if total_layers > 0:
            unfreeze_indices.append(total_layers - 1)

        print(f"  Unfreezing {len(unfreeze_indices)} layers (ratio={unfreeze_ratio:.0%})")

        # 执行解冻
        unfrozen_count = 0
        unfrozen_layers = []

        for idx in unfreeze_indices:
            if idx < len(features):
                layer = features[idx]
                for param in layer.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen_count += 1
                unfrozen_layers.append(f"[{idx}]{layer.__class__.__name__}")

        print(f"Unfrozen MobileNet blocks (unfreeze_blocks={unfreeze_blocks}):")
        print(f"  Unfrozen layers: {', '.join(unfrozen_layers)}")
        print(f"  Unfrozen parameters: {unfrozen_count}")

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        提取视觉特征

        Args:
            image_tensor: [B, C, H, W] 预处理后的图像tensor

        Returns:
            torch.Tensor: [B, feature_dim] 特征向量
        """
        # Lite-HRNet需要特定输入尺寸 (192x256 HxW)
        if self.backbone_name == 'litehrnet_18':
            if image_tensor.shape[2:] != (192, 256):
                image_tensor = F.interpolate(image_tensor, size=(192, 256), mode='bilinear', align_corners=False)

        # HRNet需要特定输入尺寸 (256x192 HxW for pose estimation pretrained)
        if self.backbone_name in ['hrnet_w18', 'hrnet_w32', 'hrnet_w48']:
            if image_tensor.shape[2:] != (256, 192):
                image_tensor = F.interpolate(image_tensor, size=(256, 192), mode='bilinear', align_corners=False)

        # 通过backbone提取特征
        features = self.backbone(image_tensor)  # [B, backbone_dim, H', W']

        # 处理HRNet和Lite-HRNet的多尺度输出
        if self.backbone_name in ['litehrnet_18', 'hrnet_w18', 'hrnet_w32', 'hrnet_w48']:
            # HRNet/Lite-HRNet返回多个尺度的特征 (tuple/list)
            if isinstance(features, (tuple, list)):
                # 只取最高分辨率特征（第一个）
                features = features[0]
                features = self.adaptive_pool(features)
                features = features.view(features.size(0), -1)
            else:
                # 如果只返回单个特征，正常处理
                features = self.adaptive_pool(features)
                features = features.view(features.size(0), -1)
        else:
            # 其他backbone的正常处理
            features = self.adaptive_pool(features)  # [B, backbone_dim, 1, 1]
            features = features.view(features.size(0), -1)  # [B, backbone_dim]
        
        # 特征映射
        features = self.feature_mapper(features)  # [B, feature_dim]
        
        return features
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        预处理单张图像
        
        Args:
            image: PIL Image对象
            
        Returns:
            torch.Tensor: [1, 3, 224, 224] 预处理后的图像tensor
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        tensor = self.preprocess(image).unsqueeze(0)  # 添加batch维度
        return tensor


class PersonCropExtractor(nn.Module):
    """
    Person region cropping and feature extraction
    """
    
    def __init__(self, backbone: ResNetBackbone, crop_size: int = 112, 
                 padding_ratio: float = 0.2):
        """
        Args:
            backbone: ResNet backbone
            crop_size: 裁剪区域目标尺寸
            padding_ratio: 边界框padding比例
        """
        super().__init__()
        self.backbone = backbone
        self.crop_size = crop_size
        self.padding_ratio = padding_ratio
        
        # 调整预处理以适应较小的裁剪区域
        self.crop_preprocess = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),  # 适应较小区域
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def crop_person_region(self, image: Image.Image, bbox: torch.Tensor, 
                          image_width: int = 3760, image_height: int = 480) -> Image.Image:
        """
        裁剪人体区域
        
        Args:
            image: PIL Image对象
            bbox: [4] 边界框 [x, y, w, h]
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            Image.Image: 裁剪的人体区域
        """
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()
        
        x, y, w, h = bbox
        
        # 添加padding
        padding_w = w * self.padding_ratio
        padding_h = h * self.padding_ratio
        
        # 计算裁剪区域
        x1 = max(0, int(x - padding_w))
        y1 = max(0, int(y - padding_h))
        x2 = min(image_width, int(x + w + padding_w))
        y2 = min(image_height, int(y + h + padding_h))
        
        # 确保区域有效
        if x2 <= x1 or y2 <= y1:
            # 使用最小有效区域
            center_x, center_y = int(x + w/2), int(y + h/2)
            min_size = max(32, int(max(w, h)))
            x1 = max(0, center_x - min_size//2)
            y1 = max(0, center_y - min_size//2)
            x2 = min(image_width, x1 + min_size)
            y2 = min(image_height, y1 + min_size)
        
        # 裁剪区域
        person_region = image.crop((x1, y1, x2, y2))
        
        return person_region
    
    def forward(self, image: Image.Image, bbox: torch.Tensor, 
                image_width: int = 3760, image_height: int = 480) -> torch.Tensor:
        """
        提取单个人的视觉特征
        
        Args:
            image: PIL Image对象
            bbox: [4] 边界框
            image_width: 图像宽度  
            image_height: 图像高度
            
        Returns:
            torch.Tensor: [feature_dim] 人体特征向量
        """
        try:
            # 裁剪人体区域
            person_region = self.crop_person_region(image, bbox, image_width, image_height)
            
            # 预处理
            if person_region.mode != 'RGB':
                person_region = person_region.convert('RGB')
            
            image_tensor = self.crop_preprocess(person_region).unsqueeze(0)  # [1, 3, H, W]

            # Ensure backbone and input are on same device
            try:
                device = next(self.backbone.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
            image_tensor = image_tensor.to(device)

            # 提取特征
            with torch.no_grad():
                features = self.backbone(image_tensor)  # [1, feature_dim]
            
            return features.squeeze(0)  # [feature_dim]
            
        except Exception as e:
            print(f"Warning: Person feature extraction failed: {e}")
            # 返回零向量作为fallback
            return torch.zeros(self.backbone.feature_dim, dtype=torch.float32)


class ResNetRelationFeatureFusion(nn.Module):
    """
    ResNet-based Relation Network特征融合器
    使用预训练ResNet提取人体视觉特征，结合几何和场景特征
    """
    
    def __init__(self, backbone_name: str = 'resnet18', visual_feature_dim: int = 256,
                 use_geometric: bool = True, use_scene_context: bool = True,
                 pretrained: bool = True, freeze_backbone: bool = False, crop_size: int = 224):
        """
        Args:
            backbone_name: ResNet架构名称
            visual_feature_dim: 视觉特征维度
            use_geometric: 是否使用几何特征
            use_scene_context: 是否使用场景上下文
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.visual_feature_dim = visual_feature_dim
        self.use_geometric = use_geometric
        self.use_scene_context = use_scene_context
        
        # 创建ResNet backbone (input size aligned with crop_size)
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            feature_dim=visual_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            input_size=crop_size
        )

        # 创建人体特征提取器
        self.person_extractor = PersonCropExtractor(self.backbone, crop_size=crop_size)
        
        # 场景上下文提取器
        if self.use_scene_context:
            self.scene_extractor = SceneContextExtractor()
        
        # 计算特征维度
        self.person_feature_dim = self._calculate_person_feature_dim()
        self.spatial_feature_dim = self._calculate_spatial_feature_dim()
        
        print(f"ResNet Relation Feature Fusion created:")
        print(f"  Person features: {self.person_feature_dim}D (ResNet {backbone_name})")
        print(f"  Spatial features: {self.spatial_feature_dim}D")
    
    def _calculate_person_feature_dim(self) -> int:
        """计算每个人的特征维度"""
        return self.visual_feature_dim  # ResNet输出维度
    
    def _calculate_spatial_feature_dim(self) -> int:
        """计算空间关系特征维度"""
        dim = 0
        if self.use_geometric:
            dim += 7  # 几何关系特征
        if self.use_scene_context:
            dim += 1  # 场景上下文
        return dim
    
    def forward(self, person_A_box: torch.Tensor, person_B_box: torch.Tensor,
                image: Optional[Image.Image] = None, 
                all_boxes: Optional[list] = None,
                image_width: int = 3760, image_height: int = 480) -> Dict[str, torch.Tensor]:
        """
        提取relation network所需的分离特征
        
        Args:
            person_A_box: [4] 人A边界框
            person_B_box: [4] 人B边界框
            image: PIL Image对象
            all_boxes: 所有人的边界框列表
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            Dict: {
                'person_A_features': [person_feature_dim] 人A特征
                'person_B_features': [person_feature_dim] 人B特征  
                'spatial_features': [spatial_feature_dim] 空间关系特征
            }
        """
        # 提取个体视觉特征
        if image is not None:
            # 提取人A特征
            person_A_features = self.person_extractor(image, person_A_box, image_width, image_height)
            
            # 提取人B特征
            person_B_features = self.person_extractor(image, person_B_box, image_width, image_height)
        else:
            # 如果没有图像，使用零向量
            print("Warning: No image provided, using zero visual features")
            person_A_features = torch.zeros(self.visual_feature_dim, dtype=torch.float32)
            person_B_features = torch.zeros(self.visual_feature_dim, dtype=torch.float32)
        
        # 提取空间关系特征
        spatial_features = []
        
        # 几何特征
        if self.use_geometric:
            try:
                geometric_features = extract_geometric_features(
                    person_A_box, person_B_box, image_width, image_height
                )
                if isinstance(geometric_features, torch.Tensor):
                    geometric_features = geometric_features.cpu().numpy()
                spatial_features.append(torch.tensor(geometric_features, dtype=torch.float32))
            except Exception as e:
                print(f"Warning: Geometric feature extraction failed: {e}")
                spatial_features.append(torch.zeros(7, dtype=torch.float32))
        
        # 场景上下文特征
        if self.use_scene_context and all_boxes is not None:
            try:
                scene_features = self.scene_extractor(all_boxes)
                spatial_features.append(scene_features)
            except Exception as e:
                print(f"Warning: Scene context extraction failed: {e}")
                spatial_features.append(torch.zeros(1, dtype=torch.float32))
        
        # 组合空间特征并确保固定维度
        if self.spatial_feature_dim > 0:
            if spatial_features:
                spatial_tensor = torch.cat(spatial_features, dim=0)
                # pad or trim to exact spatial_feature_dim
                if spatial_tensor.numel() < self.spatial_feature_dim:
                    pad = torch.zeros(self.spatial_feature_dim - spatial_tensor.numel(), dtype=torch.float32)
                    spatial_tensor = torch.cat([spatial_tensor, pad], dim=0)
                elif spatial_tensor.numel() > self.spatial_feature_dim:
                    spatial_tensor = spatial_tensor[:self.spatial_feature_dim]
            else:
                spatial_tensor = torch.zeros(self.spatial_feature_dim, dtype=torch.float32)
        else:
            spatial_tensor = torch.zeros(0, dtype=torch.float32)
        
        return {
            'person_A_features': person_A_features,
            'person_B_features': person_B_features,
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
        return {
            'backbone': self.backbone_name,
            'visual_feature_dim': self.visual_feature_dim,
            'person_feature_dim': self.person_feature_dim,
            'spatial_feature_dim': self.spatial_feature_dim,
            'geometric': {'enabled': self.use_geometric, 'dim': 7 if self.use_geometric else 0},
            'scene_context': {'enabled': self.use_scene_context, 'dim': 1 if self.use_scene_context else 0}
        }


class SceneContextExtractor(nn.Module):
    """
    场景上下文特征提取器 - 复用原有实现
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
        
        # 场景密度计算
        num_people = len(all_boxes)
        scene_density = min(num_people / 10.0, 1.0)  # 归一化到[0,1]
        return torch.tensor([scene_density], dtype=torch.float32)


if __name__ == '__main__':
    # 测试ResNet特征提取器
    print("Testing ResNet Feature Extractors...")
    
    # 测试参数
    test_image = Image.new('RGB', (3760, 480), (128, 128, 128))
    person_A_box = torch.tensor([1000, 200, 100, 200], dtype=torch.float32)
    person_B_box = torch.tensor([1200, 180, 120, 220], dtype=torch.float32)
    all_boxes = [[1000, 200, 100, 200], [1200, 180, 120, 220], [800, 150, 80, 180]]
    
    print("\n1. Testing ResNet18 backbone...")
    feature_fusion = ResNetRelationFeatureFusion(
        backbone_name='resnet18',
        visual_feature_dim=256,
        pretrained=True,
        freeze_backbone=False
    )
    
    # 提取特征
    features = feature_fusion(person_A_box, person_B_box, test_image, all_boxes)
    
    print(f"Person A features shape: {features['person_A_features'].shape}")
    print(f"Person B features shape: {features['person_B_features'].shape}")
    print(f"Spatial features shape: {features['spatial_features'].shape}")
    
    # 特征统计
    person_A_stats = features['person_A_features']
    print(f"Person A stats: min={person_A_stats.min():.4f}, max={person_A_stats.max():.4f}, mean={person_A_stats.mean():.4f}")
    
    print(f"\nFeature info: {feature_fusion.get_feature_info()}")
    
    print("\n✅ ResNet feature extractors test completed!")