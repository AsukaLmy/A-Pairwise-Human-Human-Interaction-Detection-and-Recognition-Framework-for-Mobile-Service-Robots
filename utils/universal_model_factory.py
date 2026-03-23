#!/usr/bin/env python3
"""
Universal Model Factory for Stage2 Behavior Classification
Supports ResNet, VGG, AlexNet backbones for SOTA comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import os
import sys

# Add project paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
configs_path = os.path.join(project_root, 'configs')
models_path = os.path.join(project_root, 'models')
sys.path.extend([configs_path, models_path])

# 导入通用组件
from configs.universal_stage2_config import UniversalStage2Config
from models.resnet_stage2_classifier import ResNetStage2Loss  # 复用损失函数
from models.cnn_backbone import UniversalCNNBackbone


class UniversalStage2Classifier(nn.Module):
    """
    通用Stage2分类器，支持多种CNN backbone
    """

    def __init__(self, person_feature_dim: int, spatial_feature_dim: int,
                 hidden_dims: list = [256, 128, 64], dropout: float = 0.3,
                 fusion_strategy: str = "concat", backbone_name: str = "resnet18",
                 pretrained: bool = True, freeze_backbone: bool = False,
                 crop_size: int = 112):
        """
        Args:
            person_feature_dim: 单人特征维度
            spatial_feature_dim: 空间特征维度
            hidden_dims: 关系网络隐层维度
            dropout: Dropout比率
            fusion_strategy: 特征融合策略
            backbone_name: Backbone名称
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone
            crop_size: 裁剪尺寸
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.person_feature_dim = person_feature_dim
        self.spatial_feature_dim = spatial_feature_dim
        self.fusion_strategy = fusion_strategy

        # 创建CNN backbone用于人员特征提取
        self.backbone = UniversalCNNBackbone(
            backbone_name=backbone_name,
            feature_dim=person_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            input_size=224  # 标准输入尺寸
        )

        # 空间特征编码器（如果有空间特征）
        if spatial_feature_dim > 0:
            self.spatial_encoder = nn.Sequential(
                nn.Linear(spatial_feature_dim, spatial_feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            self.spatial_encoder = None

        # 计算关系网络输入维度
        relation_input_dim = self._get_relation_input_dim()

        # 关系网络
        relation_layers = []
        input_dim = relation_input_dim

        for hidden_dim in hidden_dims:
            relation_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        # 最终分类层
        relation_layers.append(nn.Linear(input_dim, 3))  # 3个类别

        self.relation_network = nn.Sequential(*relation_layers)

        # Bilinear融合层（如果使用bilinear策略）
        if fusion_strategy == "bilinear":
            combined_dim = 2 * person_feature_dim + spatial_feature_dim if spatial_feature_dim > 0 else 2 * person_feature_dim
            self.bilinear = nn.Bilinear(combined_dim, combined_dim, hidden_dims[0])

        print(f"Created Universal Stage2 Classifier:")
        print(f"  Backbone: {backbone_name}")
        print(f"  Person feature dim: {person_feature_dim}")
        print(f"  Spatial feature dim: {spatial_feature_dim}")
        print(f"  Relation input dim: {relation_input_dim}")
        print(f"  Fusion strategy: {fusion_strategy}")

    def _get_relation_input_dim(self) -> int:
        """计算关系网络输入维度"""
        if self.fusion_strategy == "concat":
            return 2 * self.person_feature_dim + self.spatial_feature_dim
        elif self.fusion_strategy == "add":
            return self.person_feature_dim + self.spatial_feature_dim
        elif self.fusion_strategy == "bilinear":
            return self.hidden_dims[0]  # bilinear输出到第一个隐层
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def forward(self, person_A_images: torch.Tensor, person_B_images: torch.Tensor,
                spatial_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            person_A_images: [B, C, H, W] 人员A图像
            person_B_images: [B, C, H, W] 人员B图像
            spatial_features: [B, spatial_dim] 空间特征

        Returns:
            torch.Tensor: [B, 3] 分类logits
        """
        batch_size = person_A_images.size(0)

        # 提取人员特征
        person_A_features = self.backbone(person_A_images)  # [B, person_feature_dim]
        person_B_features = self.backbone(person_B_images)  # [B, person_feature_dim]

        # 处理空间特征
        if self.spatial_encoder is not None and spatial_features.size(-1) > 0:
            spatial_features = self.spatial_encoder(spatial_features)  # [B, spatial_dim]
        else:
            spatial_features = torch.zeros(batch_size, 0).to(person_A_images.device)

        # 特征融合
        if self.fusion_strategy == "concat":
            # 拼接所有特征
            if spatial_features.size(-1) > 0:
                combined_features = torch.cat([
                    person_A_features, person_B_features, spatial_features
                ], dim=1)
            else:
                combined_features = torch.cat([
                    person_A_features, person_B_features
                ], dim=1)

        elif self.fusion_strategy == "add":
            # 加法融合（要求特征维度相同）
            combined_features = person_A_features + person_B_features
            if spatial_features.size(-1) > 0:
                combined_features = torch.cat([combined_features, spatial_features], dim=1)

        elif self.fusion_strategy == "bilinear":
            # 双线性融合
            if spatial_features.size(-1) > 0:
                concat_features = torch.cat([
                    person_A_features, person_B_features, spatial_features
                ], dim=1)
            else:
                concat_features = torch.cat([
                    person_A_features, person_B_features
                ], dim=1)
            combined_features = self.bilinear(concat_features, concat_features)

        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # 关系网络推理
        logits = self.relation_network(combined_features)  # [B, 3]

        return logits

    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        backbone_info = self.backbone.get_model_info()

        return {
            'model_type': 'universal_stage2_classifier',
            'backbone': self.backbone_name,
            'backbone_info': backbone_info,
            'person_feature_dim': self.person_feature_dim,
            'spatial_feature_dim': self.spatial_feature_dim,
            'fusion_strategy': self.fusion_strategy,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }


def create_universal_stage2_model(config: UniversalStage2Config) -> nn.Module:
    """
    根据配置创建通用Stage2模型

    Args:
        config: 通用Stage2配置对象

    Returns:
        nn.Module: 创建的通用Stage2模型
    """
    # 获取特征维度
    person_feature_dim = config.get_person_feature_dim()
    spatial_feature_dim = config.get_spatial_feature_dim()

    # 创建模型
    model = UniversalStage2Classifier(
        person_feature_dim=person_feature_dim,
        spatial_feature_dim=spatial_feature_dim,
        hidden_dims=config.relation_hidden_dims,
        dropout=config.dropout,
        fusion_strategy=config.fusion_strategy,
        backbone_name=config.backbone_name,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        crop_size=config.crop_size
    )

    print(f"✅ Created Universal Stage2 Model:")
    print(f"   Backbone: {config.backbone_name}")
    print(f"   Person features: {person_feature_dim}D")
    print(f"   Spatial features: {spatial_feature_dim}D")
    print(f"   Fusion: {config.fusion_strategy}")

    # 打印模型信息
    model_info = model.get_model_info()
    print(f"   Total parameters: {model_info['total_params']:,}")
    print(f"   Trainable parameters: {model_info['trainable_params']:,}")
    print(f"   Model size: {model_info['model_size_mb']:.1f} MB")

    return model


def create_universal_stage2_loss(config: UniversalStage2Config) -> ResNetStage2Loss:
    """
    创建通用Stage2损失函数（复用ResNet的损失函数）

    Args:
        config: 通用Stage2配置对象

    Returns:
        ResNetStage2Loss: 损失函数
    """
    if not hasattr(config, 'class_weights') or config.class_weights is None:
        config.class_weights = {0: 1.0, 1: 1.0, 2: 1.0}

    criterion = ResNetStage2Loss(class_weights=config.class_weights)
    print(f"✅ Created Universal Stage2 Loss: weights={config.class_weights}")
    return criterion


def create_universal_optimizer(model: nn.Module, config: UniversalStage2Config) -> optim.Optimizer:
    """
    创建通用模型的优化器

    Args:
        model: 通用模型
        config: 配置对象

    Returns:
        torch.optim.Optimizer: 优化器
    """
    # 分离backbone参数和其他参数，使用不同学习率
    backbone_params = []
    other_params = []

    model_for_iter = model.module if hasattr(model, 'module') else model
    if hasattr(model_for_iter, 'backbone'):
        for param in model_for_iter.backbone.parameters():
            if param.requires_grad:
                backbone_params.append(param)

        # 其他参数
        backbone_param_ids = {id(p) for p in backbone_params}
        for param in model_for_iter.parameters():
            if id(param) not in backbone_param_ids and param.requires_grad:
                other_params.append(param)
    else:
        # 如果没有backbone属性，所有参数使用相同学习率
        other_params = list(model.parameters())

    # 设置参数组
    param_groups = []

    if backbone_params:
        backbone_lr = config.learning_rate * 0.1  # Backbone使用1/10学习率
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'name': 'backbone'
        })
        print(f"   Backbone params: {sum(p.numel() for p in backbone_params):,}, lr={backbone_lr}")

    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': config.learning_rate,
            'name': 'classifier'
        })
        print(f"   Classifier params: {sum(p.numel() for p in other_params):,}, lr={config.learning_rate}")

    # 创建优化器
    if config.optimizer == 'adam':
        optimizer = optim.Adam(
            param_groups if param_groups else model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(
            param_groups if param_groups else model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(
            param_groups if param_groups else model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    print(f"✅ Created {config.optimizer} optimizer with differential learning rates")
    return optimizer


def create_universal_training_setup(config: UniversalStage2Config, device: torch.device) -> Tuple:
    """
    创建完整的通用训练设置

    Args:
        config: 配置对象
        device: 设备

    Returns:
        Tuple: (model, criterion, optimizer, scheduler)
    """
    print(f"\n🏗️ Creating Universal training setup for {config.backbone_name}...")

    # 创建模型并移动到设备
    model = create_universal_stage2_model(config).to(device)

    # 如果有多个GPU，使用DataParallel
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        model = nn.DataParallel(model)
        print(f"   Using DataParallel on {torch.cuda.device_count()} GPUs")

    # 创建损失函数
    criterion = create_universal_stage2_loss(config).to(device)

    # 创建优化器
    optimizer = create_universal_optimizer(model, config)

    # 创建调度器（复用ResNet的调度器创建逻辑）
    from utils.resnet_model_factory import create_resnet_scheduler
    scheduler = create_resnet_scheduler(optimizer, config)

    print(f"✅ Universal training setup completed on {device}")

    return model, criterion, optimizer, scheduler


if __name__ == '__main__':
    # 测试通用模型工厂
    print("Testing Universal Model Factory...")

    from configs.universal_stage2_config import create_backbone_config

    # 测试不同backbone
    backbones = ['resnet18', 'vgg16', 'alexnet']

    for backbone in backbones:
        print(f"\n{'='*50}")
        print(f"Testing {backbone}:")

        # 创建配置
        config = create_backbone_config(
            backbone,
            visual_feature_dim=256,
            relation_hidden_dims=[256, 128, 64],
            dropout=0.3,
            batch_size=4
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建训练设置
        model, criterion, optimizer, scheduler = create_universal_training_setup(config, device)

        # 测试前向传播
        batch_size = 2
        person_A_images = torch.randn(batch_size, 3, 224, 224).to(device)
        person_B_images = torch.randn(batch_size, 3, 224, 224).to(device)
        spatial_features = torch.randn(batch_size, config.get_spatial_feature_dim()).to(device)
        targets = torch.randint(0, 3, (batch_size,)).to(device)

        # 前向传播
        with torch.no_grad():
            logits = model(person_A_images, person_B_images, spatial_features)
            loss, loss_dict = criterion(logits, targets)

        print(f"Input shapes:")
        print(f"  Person A/B: {person_A_images.shape}")
        print(f"  Spatial: {spatial_features.shape}")
        print(f"Output:")
        print(f"  Logits: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")

        # 获取模型信息
        model_info = model.get_model_info() if hasattr(model, 'get_model_info') else \
                    model.module.get_model_info()
        print(f"Model info: {model_info}")

    print("\n✅ Universal model factory test completed!")