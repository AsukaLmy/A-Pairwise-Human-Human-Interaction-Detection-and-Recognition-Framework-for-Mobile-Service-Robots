#!/usr/bin/env python3
"""
GNN Geometric-Only Stage2 Configuration
No visual backbone. Node features derived purely from bounding-box geometry.
Analogous to the existing MoE/MLP nobackbone pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class GNNGeometricConfig:
    """
    GNN Stage2 config – no visual backbone.

    Node features per person (5D, purely from bounding box):
        [cx_norm, cy_norm, w_norm, h_norm, aspect_ratio]

    GAT edge features per directed pair (7D):
        extract_geometric_features() – spatial relationship for scene context

    Pair flow features per labeled pair (10D, same as original nobackbone MLP):
        9D from GeometricFlowExtractor (geometric distances + Farneback optical
        flow stats, symmetrically averaged over A→B and B→A perspectives)
        + 1D interaction synchrony from compute_interaction_synchrony()
        These are concatenated to the GAT node embeddings before the edge
        classifier, matching the original 10D nobackbone feature pipeline.
    """

    model_type: str = "gnn_geometric"

    # === Node / Edge dims ===
    node_feat_dim: int = 5      # cx_norm, cy_norm, w_norm, h_norm, aspect_ratio
    # edge_feat_dim is set in __post_init__ based on inject_flow_to_edges:
    #   True  → 17D (7D geometric + 10D flow injected for target pair edges)
    #   False → 7D  (geometric only, original behaviour)
    edge_feat_dim: int = 17     # overwritten by __post_init__

    # === Pair-level flow features (appended to edge classifier input) ===
    # 9D from GeometricFlowExtractor + 1D sync = 10D total.
    # Same feature set as the original nobackbone MLP pipeline.
    pair_feat_dim: int = 10

    # === Flow injection into edge initialization (Improvement 1) ===
    # True: 对 target pair 对应的边注入 10D 流特征，与 7D 几何特征拼接 → 17D
    #       让消息传递阶段也能感知光流动态信息
    # False: 纯 7D 几何边特征（消融对照）
    inject_flow_to_edges: bool = True

    # === Per-node optical flow features (Improvement A) ===
    # True: 每人节点追加 8D 运动统计特征，node_feat_dim 5→13；
    #       pair 追加 1D 接近速率，pair_feat_dim 10→11
    #       使 GNN 消息传递阶段能区分行走/站立/坐着的人
    # False: 仅 5D 静态 bbox 节点特征（消融对照）
    flow_node_feats: bool = True

    # === Graph Transformer (replaces static-edge GAT) ===
    # If use_graph_transformer=True, uses GraphTransformerLayer (edge updates enabled).
    # If False, falls back to GATLayer (original, for ablation).
    use_graph_transformer: bool = True
    edge_hidden_dim: int = 64           # evolved edge feature dimension (all layers)
    use_edge_in_cls: bool = True        # append evolved edge to behavior classifier input

    # === Improvement II: DropEdge ===
    # 训练时随机丢弃一部分非关键边（target pair 边强制保留），正则化 GNN
    # 0.0 = 关闭（消融），0.3 = 丢弃 30% 非关键边（推荐）
    drop_edge_rate: float = 0.0

    # === Improvement III: Virtual Global Node ===
    # 为每帧场景图添加一个与所有人节点双向连接的虚拟全局节点
    # 经 GNN 传播后，该节点聚合整个场景上下文，追加到行为分类器输入
    use_virtual_node: bool = False

    # === Improvement IV: Cross-Pair Attention ===
    # GNN 传播后，对同一场景内所有 target pair 的表示运行 1 层自注意力
    # 让每个 pair 的分类感知同场景内其他 pair 的状态（场景内 pair 高度相关）
    # cross_pair_dim: 注意力投影维度（残差结构，不改变分类器输入维度）
    use_cross_pair_attn: bool = False
    cross_pair_dim: int = 256

    # Intentionally same as GNNStage2Config – Stage2 GNN is shared architecture.
    # The only difference vs. backbone version: node features are 5D geometric
    # instead of (visual_dim + 4)D. The GAT and edge classifier are identical.
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 2
    gnn_num_heads: int = 4
    gnn_dropout: float = 0.1

    # === Edge classifier ===
    # 默认从 [512,256,128] 缩减为 [256,128,64]，与 MLP 基线参数量相近，减少过拟合
    edge_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    classifier_dropout: float = 0.3

    # === Classification ===
    # Number of behavior classes. JRDB: 3 (Walking/Standing/Sitting Together).
    # CAD Stage2: 6 (NA/Crossing/Waiting/Queuing/Walking/Talking) or 3 (merged).
    num_classes: int = 3

    # === Multi-task: grouping detection ===
    lambda_behavior: float = 0.8       # weight for behavior loss
    lambda_group:    float = 0.2       # weight for binary grouping loss
    group_neg_ratio: float = 2.0       # max negatives = neg_ratio × n_positives

    # === Training ===
    epochs: int = 50
    batch_size: int = 8         # larger batch OK – no backbone memory cost
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # === Loss ===
    class_weights: Optional[Dict] = None
    focal_gamma: float = 2.0
    # 标签平滑：0.0=关闭（默认），0.1=推荐。与 focal loss 联合使用：
    # focal 权重由硬标签决定，CE 损失使用平滑分布（对少数类 Sitting Together 尤其有效）
    label_smoothing: float = 0.0

    # === Scheduler ===
    scheduler: str = "cosine"
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

    # === Early stopping ===
    early_stopping_patience: int = 20
    early_stopping_metric: str = "mpca"
    checkpoint_dir: str = "./checkpoints/gnn_geometric"

    # === Graph sparsity ===
    # 0 = 全连接（原行为），>0 = K-NN 稀疏图（target pair 边强制保留）
    graph_knn: int = 0

    # === E2E warmup ===
    # 前 e2e_warmup_epochs 个 epoch 不启用软门控（等同于 GNNMultiTaskLoss）
    # 之后线性升至完全软门控，避免初期 Stage1 噪声干扰 Stage2 学习
    e2e_warmup_epochs: int = 10

    # === Reproducibility & caching ===
    random_seed: int = 42
    # 预计算特征缓存目录。None 表示不缓存。
    # 缓存按数据参数自动命名子文件夹，不同 seed 复用同一缓存，只重采样负样本对。
    cache_dir: Optional[str] = None

    # === CAD-specific extra features (no effect on JRDB) ===
    # individual_action_id + group_size_norm per node (+2D nodes)
    use_individual_action_feat: bool = False
    # pair axis angle (cos/sin) + lateral relative velocity per pair (+2D or +3D pairs)
    use_extra_pair_feats: bool = False

    def __post_init__(self):
        if self.class_weights is None and self.num_classes == 3:
            self.class_weights = {0: 1.0, 1: 1.4, 2: 6.1}
        # For num_classes != 3, leave class_weights=None; training scripts
        # auto-compute from data distribution.
        # Derive feature dims from flags
        self.edge_feat_dim = 17 if self.inject_flow_to_edges else 7
        self.node_feat_dim = 13 if self.flow_node_feats else 5
        self.pair_feat_dim = 11 if self.flow_node_feats else 10

    def get_edge_cls_input_dim(self) -> int:
        """Behavior classifier input: [A, B, |A-B|, A*B] + flow_feats + (evolved_e if enabled)"""
        evolved = self.edge_hidden_dim if (self.use_graph_transformer and self.use_edge_in_cls) else 0
        vn_dim  = self.gnn_hidden_dim if self.use_virtual_node else 0
        return 4 * self.gnn_hidden_dim + self.pair_feat_dim + evolved + vn_dim

    def get_group_head_input_dim(self) -> int:
        """Group head input: [A, B, |A-B|, A*B] + evolved_e (no flow_feats for negative pairs)"""
        evolved = self.edge_hidden_dim if self.use_graph_transformer else 0
        return 4 * self.gnn_hidden_dim + evolved

    def get_model_info(self) -> Dict:
        return {
            'model_type':          self.model_type,
            'node_feat_dim':       self.node_feat_dim,
            'edge_feat_dim':       self.edge_feat_dim,
            'inject_flow_to_edges': self.inject_flow_to_edges,
            'pair_feat_dim':       self.pair_feat_dim,
            'use_graph_transformer': self.use_graph_transformer,
            'edge_hidden_dim':     self.edge_hidden_dim,
            'gnn_hidden':          self.gnn_hidden_dim,
            'gnn_layers':          self.gnn_num_layers,
            'gnn_heads':           self.gnn_num_heads,
            'edge_cls_hidden':     self.edge_hidden_dims,
            'edge_cls_input':      self.get_edge_cls_input_dim(),
            'group_head_input':    self.get_group_head_input_dim(),
            'lambda_behavior':     self.lambda_behavior,
            'lambda_group':        self.lambda_group,
        }


def get_gnn_geometric_default(data_path: str = "../../dataset",
                               **kwargs) -> GNNGeometricConfig:
    config = GNNGeometricConfig(data_path=data_path)
    for k, v in kwargs.items():
        setattr(config, k, v)
    return config


if __name__ == '__main__':
    cfg = get_gnn_geometric_default()
    print("GNNGeometricConfig:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
    print(f"  edge_cls_input_dim: {cfg.get_edge_cls_input_dim()}")
