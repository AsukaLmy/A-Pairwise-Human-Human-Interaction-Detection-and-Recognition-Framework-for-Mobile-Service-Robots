#!/usr/bin/env python3
"""
Universal Stage2 Configuration for SOTA Comparison
Supports ResNet, VGG, AlexNet backbones for behavior classification
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import sys
import os

# Add models path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_path = os.path.join(project_root, 'models')
sys.path.append(models_path)


@dataclass
class UniversalStage2Config:
    """通用Stage2行为分类配置，支持多种backbone"""

    # === Model Architecture ===
    model_type: str = "universal_relation"  # Model type identifier
    backbone_name: str = "resnet18"  # 支持: resnet18/34/50, vgg11/13/16/19, alexnet
    visual_feature_dim: int = 256    # Visual feature dimension from backbone

    # === Feature Configuration ===
    use_geometric: bool = True       # Use geometric spatial features
    use_scene_context: bool = True   # Use scene context features

    # === Backbone Settings ===
    pretrained: bool = True          # Use ImageNet pretrained weights
    freeze_backbone: bool = False    # Whether to freeze backbone parameters
    freeze_ratio: float = 0.0       # Ratio of early layers to freeze (0.0-1.0)
    crop_size: int = 112            # Person crop size for backbone input
    padding_ratio: float = 0.2      # Padding ratio for person cropping

    # === Relation Network Settings ===
    fusion_strategy: str = "concat"  # Feature fusion: concat, bilinear, add
    relation_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3

    # === Training Configuration ===
    epochs: int = 50
    batch_size: int = 16            # Will be adjusted based on backbone
    learning_rate: float = 1e-4     # Will be adjusted based on backbone
    weight_decay: float = 1e-5

    # === Optimization ===
    optimizer: str = "adam"
    scheduler: str = "cosine"
    step_size: int = 15
    warmup_epochs: int = 3

    # === Loss Configuration ===
    class_weights: Optional[Dict] = None
    mpca_weight: float = 0.1
    acc_weight: float = 0.05

    # === Data Configuration ===
    data_path: str = "../dataset"
    frame_interval: int = 1
    num_workers: int = 4
    use_oversampling: bool = True

    # === Early Stopping & Checkpointing ===
    early_stopping_patience: int = 15
    early_stopping_metric: str = "mpca"
    save_best_only: bool = True
    checkpoint_dir: str = "./checkpoints/universal_stage2"

    # === Logging ===
    log_interval: int = 10
    eval_interval: int = 1

    def __post_init__(self):
        """Post-initialization: 根据backbone调整参数"""
        self.validate()
        self._adjust_backbone_params()
        self._setup_derived_configs()

    def validate(self):
        """验证配置参数"""
        # Backbone validation
        supported_backbones = [
            'resnet18', 'resnet34', 'resnet50',
            'vgg11', 'vgg13', 'vgg16', 'vgg19',
            'alexnet'
        ]
        if self.backbone_name not in supported_backbones:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}. "
                           f"Supported: {supported_backbones}")

        # Fusion strategy validation
        supported_fusion = ['concat', 'bilinear', 'add']
        if self.fusion_strategy not in supported_fusion:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")

        print(f"[SUCCESS] Universal Stage2 config validation passed for {self.backbone_name}")

    def _adjust_backbone_params(self):
        """根据backbone自动调整参数"""
        from ..models.cnn_backbone import get_backbone_configs

        configs = get_backbone_configs()
        if self.backbone_name in configs:
            backbone_config = configs[self.backbone_name]

            # 只在用户未明确设置时调整参数
            if self.visual_feature_dim == 256:  # 默认值
                self.visual_feature_dim = backbone_config['feature_dim']

            if self.batch_size == 16:  # 默认值
                self.batch_size = backbone_config['batch_size']

            if self.learning_rate == 1e-4:  # 默认值
                self.learning_rate = backbone_config['learning_rate']

            print(f"[INFO] Adjusted params for {self.backbone_name}:")
            print(f"   feature_dim: {self.visual_feature_dim}")
            print(f"   batch_size: {self.batch_size}")
            print(f"   learning_rate: {self.learning_rate}")

    def _setup_derived_configs(self):
        """设置派生配置"""
        import os

        # 根据backbone创建专门的checkpoint目录
        self.checkpoint_dir = f"./checkpoints/{self.backbone_name}_stage2"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Set default class weights if not provided
        if self.class_weights is None:
            self.class_weights = {0: 1.0, 1: 1.0, 2: 1.0}

        print(f"[INFO] Checkpoint directory: {self.checkpoint_dir}")

    def get_person_feature_dim(self) -> int:
        """Get dimension of individual person features"""
        return self.visual_feature_dim

    def get_spatial_feature_dim(self) -> int:
        """Get dimension of spatial relation features"""
        dim = 0
        if self.use_geometric:
            dim += 7  # Geometric features
        if self.use_scene_context:
            dim += 1  # Scene context
        return dim

    def get_model_info(self) -> Dict:
        """Get model architecture information"""
        return {
            'model_type': self.model_type,
            'backbone': self.backbone_name,
            'visual_feature_dim': self.visual_feature_dim,
            'person_feature_dim': self.get_person_feature_dim(),
            'spatial_feature_dim': self.get_spatial_feature_dim(),
            'fusion_strategy': self.fusion_strategy,
            'hidden_dims': self.relation_hidden_dims,
            'pretrained': self.pretrained,
            'freeze_backbone': self.freeze_backbone,
            'freeze_ratio': self.freeze_ratio,
        }

    def print_config(self):
        """Print configuration summary"""
        print("=" * 60)
        print("UNIVERSAL STAGE2 CONFIGURATION")
        print("=" * 60)

        print(f"Model Architecture:")
        print(f"  Type: {self.model_type}")
        print(f"  Backbone: {self.backbone_name}")
        print(f"  Visual features: {self.visual_feature_dim}D")
        print(f"  Spatial features: {self.get_spatial_feature_dim()}D")
        print(f"  Fusion strategy: {self.fusion_strategy}")

        print(f"\nBackbone Settings:")
        print(f"  Pretrained: {self.pretrained}")
        print(f"  Freeze backbone: {self.freeze_backbone}")
        print(f"  Freeze ratio: {self.freeze_ratio}")
        print(f"  Crop size: {self.crop_size}x{self.crop_size}")

        print(f"\nTraining:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Optimizer: {self.optimizer}")

        print(f"\nFeatures:")
        print(f"  Geometric: {self.use_geometric}")
        print(f"  Scene context: {self.use_scene_context}")

        print("=" * 60)


# 预定义配置函数
def get_resnet18_config(**kwargs) -> UniversalStage2Config:
    """ResNet18 configuration"""
    config = UniversalStage2Config(
        backbone_name="resnet18",
        visual_feature_dim=256,
        relation_hidden_dims=[256, 128, 64],
        batch_size=32,
        learning_rate=1e-4
    )
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_resnet50_config(**kwargs) -> UniversalStage2Config:
    """ResNet50 configuration"""
    config = UniversalStage2Config(
        backbone_name="resnet50",
        visual_feature_dim=512,
        relation_hidden_dims=[1024, 512, 256],
        batch_size=16,
        learning_rate=5e-5
    )
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_vgg16_config(**kwargs) -> UniversalStage2Config:
    """VGG16 configuration"""
    config = UniversalStage2Config(
        backbone_name="vgg16",
        visual_feature_dim=512,
        relation_hidden_dims=[1024, 512, 256],
        batch_size=8,
        learning_rate=5e-5
    )
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_vgg19_config(**kwargs) -> UniversalStage2Config:
    """VGG19 configuration"""
    config = UniversalStage2Config(
        backbone_name="vgg19",
        visual_feature_dim=512,
        relation_hidden_dims=[1024, 512, 256],
        batch_size=8,
        learning_rate=5e-5
    )
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_alexnet_config(**kwargs) -> UniversalStage2Config:
    """AlexNet configuration"""
    config = UniversalStage2Config(
        backbone_name="alexnet",
        visual_feature_dim=256,
        relation_hidden_dims=[256, 128, 64],
        batch_size=32,
        learning_rate=1e-4
    )
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def create_backbone_config(backbone_name: str, **kwargs) -> UniversalStage2Config:
    """根据backbone名称创建对应配置"""
    backbone_configs = {
        'resnet18': get_resnet18_config,
        'resnet34': lambda **kw: get_resnet18_config(backbone_name='resnet34', **kw),
        'resnet50': get_resnet50_config,
        'vgg11': lambda **kw: get_alexnet_config(backbone_name='vgg11', **kw),
        'vgg13': lambda **kw: get_alexnet_config(backbone_name='vgg13', **kw),
        'vgg16': get_vgg16_config,
        'vgg19': get_vgg19_config,
        'alexnet': get_alexnet_config
    }

    if backbone_name in backbone_configs:
        return backbone_configs[backbone_name](**kwargs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


if __name__ == '__main__':
    # 测试不同backbone配置
    backbones = ['resnet18', 'vgg16', 'alexnet']

    for backbone in backbones:
        print(f"\nTesting {backbone} config:")
        config = create_backbone_config(backbone)
        config.print_config()
        print(f"Model info: {config.get_model_info()}")

    print("\n✅ Universal Stage2 configurations test completed!")