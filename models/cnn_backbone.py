#!/usr/bin/env python3
"""
Universal CNN Backbone for Stage2 Behavior Classification
Supports ResNet, VGG, AlexNet for SOTA comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict


class UniversalCNNBackbone(nn.Module):
    """
    通用CNN backbone，支持ResNet、VGG、AlexNet等架构
    """

    def __init__(self, backbone_name: str = 'resnet18', feature_dim: int = 256,
                 pretrained: bool = True, freeze_backbone: bool = False,
                 input_size: int = 224):
        """
        Args:
            backbone_name: 架构名称 ('resnet18/34/50', 'vgg11/13/16/19', 'alexnet')
            feature_dim: 输出特征维度
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone参数
            input_size: 输入图像尺寸
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.input_size = input_size

        # 创建backbone和获取特征维度
        self.backbone, self.backbone_dim = self._create_backbone()

        # 特征映射层
        self.feature_mapper = nn.Sequential(
            nn.Linear(self.backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # 冻结backbone参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 图像预处理
        self.preprocess = self._get_preprocess_transform()

        print(f"Created {backbone_name} backbone: {self.backbone_dim}D -> {feature_dim}D, "
              f"pretrained={pretrained}, frozen={freeze_backbone}")

    def _create_backbone(self) -> Tuple[nn.Module, int]:
        """根据backbone名称创建对应的网络结构"""

        # ResNet系列
        if self.backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=self.pretrained)
            backbone = nn.Sequential(*list(backbone.children())[:-1])  # 移除fc层
            backbone_dim = 512

        elif self.backbone_name == 'resnet34':
            backbone = models.resnet34(pretrained=self.pretrained)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_dim = 512

        elif self.backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=self.pretrained)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_dim = 2048

        # VGG系列
        elif self.backbone_name == 'vgg11':
            backbone = models.vgg11(pretrained=self.pretrained)
            backbone = nn.Sequential(
                backbone.features,  # 只保留特征层
                nn.AdaptiveAvgPool2d((7, 7))  # 固定输出尺寸
            )
            backbone_dim = 512 * 7 * 7

        elif self.backbone_name == 'vgg13':
            backbone = models.vgg13(pretrained=self.pretrained)
            backbone = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d((7, 7))
            )
            backbone_dim = 512 * 7 * 7

        elif self.backbone_name == 'vgg16':
            backbone = models.vgg16(pretrained=self.pretrained)
            backbone = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d((7, 7))
            )
            backbone_dim = 512 * 7 * 7

        elif self.backbone_name == 'vgg19':
            backbone = models.vgg19(pretrained=self.pretrained)
            backbone = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d((7, 7))
            )
            backbone_dim = 512 * 7 * 7

        # AlexNet
        elif self.backbone_name == 'alexnet':
            backbone = models.alexnet(pretrained=self.pretrained)
            backbone = nn.Sequential(
                backbone.features,  # 只保留特征层
                nn.AdaptiveAvgPool2d((6, 6))  # AlexNet默认输出尺寸
            )
            backbone_dim = 256 * 6 * 6

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        return backbone, backbone_dim

    def _get_preprocess_transform(self) -> transforms.Compose:
        """获取对应backbone的预处理变换"""
        if 'resnet' in self.backbone_name or 'vgg' in self.backbone_name:
            # ResNet和VGG使用ImageNet标准化
            return transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

        elif 'alexnet' in self.backbone_name:
            # AlexNet使用相同的ImageNet标准化
            return transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

        else:
            # 默认标准化
            return transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        提取视觉特征

        Args:
            image_tensor: [B, C, H, W] 预处理后的图像tensor

        Returns:
            torch.Tensor: [B, feature_dim] 特征向量
        """
        # 通过backbone提取特征
        features = self.backbone(image_tensor)  # [B, backbone_dim, H', W']

        # 展平特征（对于ResNet会先通过AdaptiveAvgPool，VGG和AlexNet已经在backbone中处理）
        if len(features.shape) == 4:  # [B, C, H, W]
            if 'resnet' in self.backbone_name:
                # ResNet需要额外的AdaptiveAvgPool
                features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)  # [B, backbone_dim]

        # 特征映射
        features = self.feature_mapper(features)  # [B, feature_dim]

        return features

    def freeze_early_layers(self, freeze_ratio: float = 0.5):
        """
        冻结backbone前部分层

        Args:
            freeze_ratio: 冻结层的比例 (0.0-1.0)
        """
        if freeze_ratio <= 0:
            return

        # 获取所有参数
        all_params = list(self.backbone.parameters())
        freeze_count = int(len(all_params) * freeze_ratio)

        frozen_params = 0
        for param in all_params[:freeze_count]:
            if param.requires_grad:
                param.requires_grad = False
                frozen_params += 1

        print(f"Frozen {frozen_params}/{len(all_params)} parameters "
              f"({freeze_ratio:.1%}) of {self.backbone_name}")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'backbone_name': self.backbone_name,
            'backbone_dim': self.backbone_dim,
            'feature_dim': self.feature_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'pretrained': self.pretrained,
            'frozen': self.freeze_backbone,
            'input_size': self.input_size
        }


def get_backbone_configs() -> Dict:
    """获取不同backbone的推荐配置"""
    return {
        # ResNet系列
        'resnet18': {
            'feature_dim': 256,
            'input_size': 224,
            'batch_size': 32,
            'learning_rate': 1e-4
        },
        'resnet34': {
            'feature_dim': 256,
            'input_size': 224,
            'batch_size': 32,
            'learning_rate': 1e-4
        },
        'resnet50': {
            'feature_dim': 512,
            'input_size': 224,
            'batch_size': 16,
            'learning_rate': 5e-5
        },

        # VGG系列
        'vgg11': {
            'feature_dim': 256,
            'input_size': 224,
            'batch_size': 16,
            'learning_rate': 1e-4
        },
        'vgg13': {
            'feature_dim': 256,
            'input_size': 224,
            'batch_size': 16,
            'learning_rate': 1e-4
        },
        'vgg16': {
            'feature_dim': 512,
            'input_size': 224,
            'batch_size': 8,
            'learning_rate': 5e-5
        },
        'vgg19': {
            'feature_dim': 512,
            'input_size': 224,
            'batch_size': 8,
            'learning_rate': 5e-5
        },

        # AlexNet
        'alexnet': {
            'feature_dim': 256,
            'input_size': 224,
            'batch_size': 32,
            'learning_rate': 1e-4
        }
    }


if __name__ == '__main__':
    # 测试不同backbone
    backbones = ['resnet18', 'vgg16', 'alexnet']

    for backbone_name in backbones:
        print(f"\nTesting {backbone_name}:")

        # 创建backbone
        backbone = UniversalCNNBackbone(
            backbone_name=backbone_name,
            feature_dim=256,
            pretrained=False  # 测试时不下载权重
        )

        # 测试前向传播
        batch_size = 4
        test_input = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            features = backbone(test_input)

        print(f"  Input: {test_input.shape}")
        print(f"  Output: {features.shape}")
        print(f"  Info: {backbone.get_model_info()}")

    print("\n✅ Universal CNN backbone test completed!")