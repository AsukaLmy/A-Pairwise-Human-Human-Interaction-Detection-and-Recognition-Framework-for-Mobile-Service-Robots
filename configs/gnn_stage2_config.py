#!/usr/bin/env python3
"""
GNN-based Stage2 Configuration
Configuration for Graph Attention Network (GAT) with ResNet backbone
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class GNNStage2Config:
    """GNN-based Stage2 behavior classification configuration"""

    # === Model Architecture ===
    model_type: str = "gnn_gat"

    # === Backbone ===
    backbone_name: str = "efficientnet_v2_s"   # default; also supports resnet*, vgg*, etc.
    visual_feature_dim: int = 256              # 256 for efficientnet_v2_s / resnet50+
    pretrained: bool = True
    freeze_backbone: bool = True        # Freeze backbone; only train GAT + classifier
    crop_size: int = 112                # Person crop size for ResNet input

    # === Graph Node Features ===
    # node_feat_dim = visual_feature_dim + 4 (position encoding)
    # position encoding: [cx_norm, cy_norm, w_norm, h_norm]
    node_position_dim: int = 4

    # === Graph Edge Features ===
    edge_feat_dim: int = 7              # Directly use extract_geometric_features() output

    # === GAT Hyperparameters ===
    gnn_hidden_dim: int = 256           # GAT hidden dimension (total, across all heads)
    gnn_num_layers: int = 2             # Number of GAT layers (2-3 recommended)
    gnn_num_heads: int = 4              # Number of attention heads
    gnn_dropout: float = 0.1           # Dropout inside GAT layers

    # === Edge Classification Head ===
    # Input dim = 4 * gnn_hidden_dim (concat of [A, B, |A-B|, A*B])
    edge_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    classifier_dropout: float = 0.3

    # === Training ===
    epochs: int = 40
    batch_size: int = 4                 # Unit: frames (each frame contains multiple pairs)
    learning_rate: float = 1e-3
    backbone_lr_multiplier: float = 0.1 # Backbone uses smaller LR
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # === Loss ===
    class_weights: Optional[Dict] = None   # Set by data loader or manually
    focal_gamma: float = 2.0

    # === Scheduler ===
    scheduler: str = "cosine"           # cosine, step, none
    warmup_epochs: int = 3

    # === Data ===
    data_path: str = "../../dataset"
    frame_interval: int = 1
    num_workers: int = 4
    image_width: int = 3760
    image_height: int = 480
    filter_occlusion: bool = True
    filter_edge_cases: bool = True
    edge_threshold: int = 200

    # === Early Stopping & Checkpointing ===
    early_stopping_patience: int = 20
    early_stopping_metric: str = "mpca"
    checkpoint_dir: str = "./checkpoints/gnn_stage2"

    # === Logging ===
    log_interval: int = 10

    def __post_init__(self):
        if self.class_weights is None:
            self.class_weights = {0: 1.0, 1: 1.4, 2: 6.1}

    def get_node_feat_dim(self) -> int:
        """Total node feature dimension = visual + position encoding"""
        return self.visual_feature_dim + self.node_position_dim

    def get_edge_cls_input_dim(self) -> int:
        """Edge classifier input dim = 4 * gnn_hidden_dim ([A, B, |A-B|, A*B])"""
        return 4 * self.gnn_hidden_dim

    def get_model_info(self) -> Dict:
        return {
            'model_type': self.model_type,
            'backbone': self.backbone_name,
            'visual_dim': self.visual_feature_dim,
            'node_dim': self.get_node_feat_dim(),
            'edge_dim': self.edge_feat_dim,
            'gnn_hidden': self.gnn_hidden_dim,
            'gnn_layers': self.gnn_num_layers,
            'gnn_heads': self.gnn_num_heads,
            'edge_cls_hidden': self.edge_hidden_dims,
        }


def get_gnn_resnet18_config(data_path: str = "../../dataset", **kwargs) -> GNNStage2Config:
    """Default config with ResNet18 backbone"""
    config = GNNStage2Config(
        backbone_name="resnet18",
        visual_feature_dim=128,
        freeze_backbone=True,
        gnn_hidden_dim=256,
        gnn_num_layers=2,
        gnn_num_heads=4,
        data_path=data_path,
    )
    for k, v in kwargs.items():
        setattr(config, k, v)
    return config


def get_gnn_efficientnet_config(data_path: str = "../../dataset", **kwargs) -> GNNStage2Config:
    """Default config with EfficientNetV2-S backbone (matches withbackbone script default)"""
    config = GNNStage2Config(
        backbone_name="efficientnet_v2_s",
        visual_feature_dim=256,
        freeze_backbone=True,
        gnn_hidden_dim=256,
        gnn_num_layers=2,
        gnn_num_heads=4,
        data_path=data_path,
    )
    for k, v in kwargs.items():
        setattr(config, k, v)
    return config


def get_gnn_resnet50_config(data_path: str = "../../dataset", **kwargs) -> GNNStage2Config:
    """Larger config with ResNet50 backbone"""
    config = GNNStage2Config(
        backbone_name="resnet50",
        visual_feature_dim=256,
        freeze_backbone=True,
        gnn_hidden_dim=512,
        gnn_num_layers=2,
        gnn_num_heads=8,
        edge_hidden_dims=[1024, 512, 256],
        data_path=data_path,
    )
    for k, v in kwargs.items():
        setattr(config, k, v)
    return config


if __name__ == '__main__':
    config = get_gnn_resnet18_config()
    print("GNN Stage2 Config:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")
    print(f"\nDerived dims:")
    print(f"  node_feat_dim: {config.get_node_feat_dim()}")
    print(f"  edge_cls_input_dim: {config.get_edge_cls_input_dim()}")
