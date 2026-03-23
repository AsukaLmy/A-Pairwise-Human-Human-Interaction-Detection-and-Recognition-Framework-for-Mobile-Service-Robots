#!/usr/bin/env python3
"""
ResNet-based Stage2 Configuration
Configuration for ResNet backbone with Relation Network
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ResNetStage2Config:
    """ResNet-based Stage2 behavior classification configuration"""
    
    # === Model Architecture ===
    model_type: str = "resnet_relation"  # Model type identifier
    backbone_name: str = "resnet18"  # ResNet architecture: resnet18, resnet34, resnet50
    visual_feature_dim: int = 128    # Visual feature dimension from ResNet
    
    # === Feature Configuration ===
    use_geometric: bool = True       # Use geometric spatial features
    use_scene_context: bool = True   # Use scene context features
    use_optical_flow: bool = False   # Use optical flow motion features (NEW)
    optical_flow_method: str = 'farneback'  # Optical flow method: 'farneback' or 'lucas_kanade'
    spatial_feature_dim: Optional[int] = None  # Manual override for spatial feature dimension (if set, overrides automatic calculation)
    feature_mode: str = 'both'  # Feature selection mode: 'both' (10D), 'opticalflow_only' (5D), 'bboxposition_only' (5D)

    # === ResNet Backbone Settings ===
    pretrained: bool = True          # Use ImageNet pretrained weights
    freeze_backbone: bool = False    # Whether to freeze backbone parameters
    freeze_blocks: int = 0           # Number of early residual blocks to freeze (0-4)
    crop_size: int = 112            # Person crop size for ResNet input
    padding_ratio: float = 0.2      # Padding ratio for person cropping
    
    # === Relation Network Settings ===
    fusion_strategy: str = "concat"  # Feature fusion: concat, bilinear, add
    relation_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    # Relation module selection: 'mlp', 'deep_mlp', 'transformer', 'transformer_cross', 'mfb'
    relation_type: str = "mlp"

    # Transformer params (used when relation_type startswith 'transformer')
    token_dim: int = 256
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff: int = 1024
    transformer_dropout: float = 0.1

    # MFB params
    mfb_out_dim: int = 512
    mfb_k: int = 5

    # Deep MLP params
    deep_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
    deep_dropout: float = 0.2
    
    # === Training Configuration ===
    epochs: int = 30
    batch_size: int = 16            # Smaller batch size for ResNet memory usage
    learning_rate: float = 1e-3     # Lower LR for pretrained features
    weight_decay: float = 1e-5      # L2 regularization
    
    # === Optimization ===
    optimizer: str = "adam"         # Optimizer type
    scheduler: str = "cosine"       # LR scheduler: step, cosine, plateau, none
    step_size: int = 15            # For StepLR
    warmup_epochs: int = 3         # Warmup epochs for pretrained backbone
    
    # === Loss Configuration ===
    class_weights: Optional[Dict] = None  # Will be set by data loader
    mpca_weight: float = 0.1       # MPCA loss weight  
    acc_weight: float = 0.05       # Accuracy regularization weight
    
    # === Data Configuration ===
    data_path: str = "../../dataset"
    frame_interval: int = 1         # Frame sampling interval
    num_workers: int = 4           # DataLoader workers
    use_oversampling: bool = True  # Use weighted sampling for class balance
    
    # === Early Stopping & Checkpointing ===
    early_stopping_patience: int = 20
    early_stopping_metric: str = "mpca"  # Metric for early stopping: mpca, accuracy
    save_best_only: bool = True
    checkpoint_dir: str = "./checkpoints/resnet_stage2"
    
    # === Logging ===
    log_interval: int = 10         # Print frequency
    eval_interval: int = 1         # Validation frequency (epochs)
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self.validate()
        self._setup_derived_configs()
    
    def validate(self):
        """Validate configuration parameters"""
        # Backbone validation
        supported_backbones = ['resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                              'alexnet', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_v2_s',
                              'litehrnet_18', 'hrnet_w18', 'hrnet_w32', 'hrnet_w48']
        if self.backbone_name not in supported_backbones:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}. Supported: {supported_backbones}"
            )
        
        # Fusion strategy validation
        supported_fusion = ['concat', 'bilinear', 'add']
        if self.fusion_strategy not in supported_fusion:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}. "
                           f"Supported: {supported_fusion}")
        
        # Feature configuration validation
        if not (self.use_geometric or self.use_scene_context or self.use_optical_flow):
            print("Warning: No spatial features enabled. Only using visual features.")

        # Optical flow method validation
        if self.use_optical_flow:
            supported_methods = ['farneback', 'lucas_kanade']
            if self.optical_flow_method not in supported_methods:
                raise ValueError(f"Unsupported optical flow method: {self.optical_flow_method}. "
                               f"Supported: {supported_methods}")
        
        # Training parameters validation
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not (0 <= self.dropout <= 1):
            raise ValueError("dropout must be in [0, 1]")
        
        # Dimension validation
        if self.visual_feature_dim < 1:
            raise ValueError("visual_feature_dim must be >= 1")
        if len(self.relation_hidden_dims) == 0:
            raise ValueError("relation_hidden_dims cannot be empty")

        # freeze_blocks validation
        if not isinstance(self.freeze_blocks, int) or not (0 <= self.freeze_blocks <= 4):
            raise ValueError("freeze_blocks must be an int in [0,4]")

        print("[SUCCESS] ResNet Stage2 config validation passed")
    
    def _setup_derived_configs(self):
        """Setup derived configuration values"""
        import os
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set default class weights if not provided
        if self.class_weights is None:
            self.class_weights = {0: 1.0, 1: 1.0, 2: 1.0}  # Will be updated by data loader
        
        print(f"[INFO] Checkpoint directory: {self.checkpoint_dir}")
    
    def get_person_feature_dim(self) -> int:
        """Get dimension of individual person features"""
        return self.visual_feature_dim

    def get_spatial_feature_dim(self) -> int:
        """Get dimension of spatial relation features"""
        # Priority 1: Use manual override if set
        if self.spatial_feature_dim is not None:
            return self.spatial_feature_dim

        # Priority 2: Calculate based on feature flags
        dim = 0
        if self.use_geometric:
            dim += 7  # Geometric features
        if self.use_scene_context:
            dim += 1  # Scene context
        if self.use_optical_flow:
            dim += 8  # Optical flow features (NEW)
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
            'freeze_blocks': self.freeze_blocks,
        }
    
    def print_config(self):
        """Print configuration summary"""
        print("=" * 60)
        print("RESNET STAGE2 CONFIGURATION")
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
        print(f"  Crop size: {self.crop_size}x{self.crop_size}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Optimizer: {self.optimizer}")
        print(f"  Scheduler: {self.scheduler}")
        
        print(f"\nFeatures:")
        print(f"  Geometric: {self.use_geometric}")
        print(f"  Scene context: {self.use_scene_context}")
        print(f"  Optical flow: {self.use_optical_flow}")
        if self.use_optical_flow:
            print(f"    Method: {self.optical_flow_method}")
        
        print(f"\nData:")
        print(f"  Data path: {self.data_path}")
        print(f"  Frame interval: {self.frame_interval}")
        print(f"  Use oversampling: {self.use_oversampling}")
        
        print("=" * 60)


# 预定义配置
def get_resnet18_config(**kwargs) -> ResNetStage2Config:
    """Get ResNet18 configuration"""
    config = ResNetStage2Config(
        backbone_name="resnet18",
        visual_feature_dim=128,
        relation_hidden_dims=[128, 64, 32],
        batch_size=16,
        learning_rate=1e-4
    )
    
    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")
    
    return config


def get_resnet50_config(**kwargs) -> ResNetStage2Config:
    """Get ResNet50 configuration (more powerful but slower)"""
    config = ResNetStage2Config(
        backbone_name="resnet50",
        visual_feature_dim=512,
        relation_hidden_dims=[1024, 512, 256],
        batch_size=8,  # Smaller batch size for larger model
        learning_rate=5e-5  # Lower LR for larger model
    )

    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def get_vgg16_config(**kwargs) -> ResNetStage2Config:
    """Get VGG16 configuration"""
    config = ResNetStage2Config(
        backbone_name="vgg16",
        visual_feature_dim=512,  # VGG16输出512维特征
        relation_hidden_dims=[512, 256, 128],
        batch_size=8,   # VGG需要更多显存
        learning_rate=5e-5  # 较低学习率
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def get_vgg19_config(**kwargs) -> ResNetStage2Config:
    """Get VGG19 configuration"""
    config = ResNetStage2Config(
        backbone_name="vgg19",
        visual_feature_dim=512,
        relation_hidden_dims=[512, 256, 128],
        batch_size=6,   # 更小批次
        learning_rate=5e-5
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def get_alexnet_config(**kwargs) -> ResNetStage2Config:
    """Get AlexNet configuration"""
    config = ResNetStage2Config(
        backbone_name="alexnet",
        visual_feature_dim=256,  # AlexNet输出256维特征
        relation_hidden_dims=[256, 128, 64],
        batch_size=32,  # AlexNet较轻量，可以用大批次
        learning_rate=1e-4
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def create_backbone_config(backbone_name: str, **kwargs) -> ResNetStage2Config:
    """根据backbone名称创建对应配置"""
    config_functions = {
        'resnet18': get_resnet18_config,
        'resnet34': lambda **kw: get_resnet18_config(backbone_name='resnet34', **kw),
        'resnet50': get_resnet50_config,
        'vgg11': lambda **kw: get_alexnet_config(backbone_name='vgg11', **kw),
        'vgg13': lambda **kw: get_alexnet_config(backbone_name='vgg13', **kw),
        'vgg16': get_vgg16_config,
        'vgg19': get_vgg19_config,
        'alexnet': get_alexnet_config,
        'mobilenet_v3_small': get_mobilenet_v3_small_config,
        'mobilenet_v3_large': get_mobilenet_v3_large_config,
        'efficientnet_v2_s': get_efficientnet_v2_s_config,
        'litehrnet_18': get_litehrnet_18_config,
        'hrnet_w18': get_hrnet_w18_config,
        'hrnet_w32': get_hrnet_w32_config,
        'hrnet_w48': get_hrnet_w48_config
    }

    if backbone_name in config_functions:
        return config_functions[backbone_name](**kwargs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


def get_backbone_config(backbone_name: str, **kwargs) -> ResNetStage2Config:
    """获取backbone配置的便捷函数"""
    return create_backbone_config(backbone_name, **kwargs)


def get_mobilenet_v3_small_config(**kwargs) -> ResNetStage2Config:
    """Get MobileNetV3-Small configuration (轻量级, 参数量2.5M)"""
    config = ResNetStage2Config(
        backbone_name="mobilenet_v3_small",
        visual_feature_dim=256,  # 映射到统一维度
        relation_hidden_dims=[256, 128, 64],  # 轻量级MLP
        batch_size=32,  # 更大批次 (显存占用少)
        learning_rate=1e-3,  # 轻量模型使用较高学习率
        crop_size=224,  # MobileNetV3推荐224输入
        dropout=0.3,
        scheduler='cosine',  # 推荐cosine scheduler
        warmup_epochs=3,
        early_stopping_patience=25
    )

    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def get_mobilenet_v3_large_config(**kwargs) -> ResNetStage2Config:
    """Get MobileNetV3-Large configuration (平衡, 参数量5.5M)"""
    config = ResNetStage2Config(
        backbone_name="mobilenet_v3_large",
        visual_feature_dim=256,  # 映射到统一维度
        relation_hidden_dims=[256, 128, 64],  # 适中MLP
        batch_size=24,  # 中等批次
        learning_rate=8e-4,  # 稍低学习率
        crop_size=224,  # MobileNetV3推荐224输入
        dropout=0.3,
        scheduler='cosine',
        warmup_epochs=3,
        early_stopping_patience=25
    )

    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def get_litehrnet_18_config(**kwargs) -> ResNetStage2Config:
    """Get Lite-HRNet-18 configuration (轻量级高分辨率网络)"""
    config = ResNetStage2Config(
        backbone_name="litehrnet_18",
        visual_feature_dim=256,
        relation_hidden_dims=[512, 256, 128],
        batch_size=16,  # 轻量级，显存占用适中
        learning_rate=1e-4,
        crop_size=192,  # Lite-HRNet训练时的输入高度 (256x192)
        dropout=0.3,
        scheduler='cosine',
        warmup_epochs=3,
        early_stopping_patience=25
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def get_hrnet_w18_config(**kwargs) -> ResNetStage2Config:
    """Get HRNet-W18 configuration (高分辨率网络-宽度18)"""
    config = ResNetStage2Config(
        backbone_name="hrnet_w18",
        visual_feature_dim=256,
        relation_hidden_dims=[512, 256, 128],
        batch_size=16,
        learning_rate=1e-4,
        crop_size=256,  # HRNet训练时的输入高度 (256x192)
        dropout=0.3,
        scheduler='cosine',
        warmup_epochs=3,
        early_stopping_patience=25
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def get_hrnet_w32_config(**kwargs) -> ResNetStage2Config:
    """Get HRNet-W32 configuration (高分辨率网络-宽度32)"""
    config = ResNetStage2Config(
        backbone_name="hrnet_w32",
        visual_feature_dim=256,
        relation_hidden_dims=[512, 256, 128],
        batch_size=12,  # 稍大的模型，减小batch size
        learning_rate=8e-5,
        crop_size=256,  # HRNet训练时的输入高度 (256x192)
        dropout=0.3,
        scheduler='cosine',
        warmup_epochs=3,
        early_stopping_patience=25
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def get_hrnet_w48_config(**kwargs) -> ResNetStage2Config:
    """Get HRNet-W48 configuration (高分辨率网络-宽度48)"""
    config = ResNetStage2Config(
        backbone_name="hrnet_w48",
        visual_feature_dim=256,
        relation_hidden_dims=[512, 256, 128],
        batch_size=8,  # 最大的模型，最小batch size
        learning_rate=5e-5,
        crop_size=256,  # HRNet训练时的输入高度 (256x192)
        dropout=0.3,
        scheduler='cosine',
        warmup_epochs=3,
        early_stopping_patience=25
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def get_efficientnet_v2_s_config(**kwargs) -> ResNetStage2Config:
    """Get EfficientNetV2-Small configuration (高效轻量级, 参数量~21M)"""
    config = ResNetStage2Config(
        backbone_name="efficientnet_v2_s",
        visual_feature_dim=256,  # 映射到统一维度
        relation_hidden_dims=[256, 128, 64],
        batch_size=16,  # 适中批次
        learning_rate=1e-4,  # 标准学习率
        crop_size=224,  # EfficientNetV2推荐224输入
        dropout=0.3,
        scheduler='cosine',
        warmup_epochs=3,
        early_stopping_patience=25
    )

    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


if __name__ == '__main__':
    # Test configurations
    print("Testing ResNet Stage2 Configurations...")
    
    print("\n1. Testing ResNet18 config:")
    config18 = get_resnet18_config()
    config18.print_config()
    
    print(f"\nModel info: {config18.get_model_info()}")
    
    print("\n2. Testing ResNet50 config:")
    config50 = get_resnet50_config(epochs=30)
    print(f"ResNet50 model info: {config50.get_model_info()}")
    
    print("\n3. Testing custom config:")
    custom_config = ResNetStage2Config(
        backbone_name="resnet34",
        visual_feature_dim=512,
        fusion_strategy="bilinear",
        freeze_backbone=True
    )
    print(f"Custom model info: {custom_config.get_model_info()}")
    
    print("\n✅ ResNet Stage2 configurations test completed!")