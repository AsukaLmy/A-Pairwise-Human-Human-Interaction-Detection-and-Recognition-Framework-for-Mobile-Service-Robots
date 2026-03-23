#!/usr/bin/env python3
"""
Stage2 Configuration System
Supports flexible feature combinations and temporal modes
"""

from dataclasses import dataclass
from typing import List, Optional
import argparse


@dataclass
class Stage2Config:
    """Stage2训练配置类 - 支持Basic和LSTM模式"""
    
    # === 模型架构配置 ===
    temporal_mode: str = 'none'           # 'none' (Basic模式), 'lstm', or 'relation'
    use_geometric: bool = True            # 使用Stage1几何特征 (7维)
    use_hog: bool = True                  # 使用HoG视觉特征 (64维)
    use_scene_context: bool = True        # 使用场景上下文
    
    # === Relation Network配置 ===
    fusion_strategy: str = 'concat'       # 特征融合策略: 'concat' or 'elementwise'
    relation_hidden_dims: List[int] = None  # Relation网络隐藏层 [64, 64]
    spatial_feature_dim: int = 0          # 额外空间特征维度
    
    # === Basic模式配置 ===
    hidden_dims: List[int] = None         # 隐藏层维度 [128, 64, 32]
    dropout: float = 0.2                  # Dropout率
    use_attention: bool = False           # Basic模式不使用注意力
    
    # === LSTM模式配置 (暂时保留，后续实现) ===
    sequence_length: int = 5              # 时序长度
    lstm_hidden_dim: int = 64             # LSTM隐藏维度
    lstm_layers: int = 2                  # LSTM层数
    bidirectional: bool = True            # 双向LSTM
    
    # === 数据配置 ===
    data_path: str = "../dataset"         # 数据集路径
    batch_size: int = 64                  # 批次大小
    num_workers: int = 2                  # 数据加载进程数
    frame_interval: int = 1               # 帧采样间隔 (1=每帧, 10=每10帧)
    
    # === 训练配置 ===
    epochs: int = 100                     # 训练轮数
    learning_rate: float = 1e-3           # 学习率
    weight_decay: float = 1e-4            # 权重衰减
    optimizer: str = 'adam'               # 优化器类型
    scheduler: str = 'step'               # 学习率调度器
    step_size: int = 30                   # StepLR步长
    
    # === 损失函数配置 ===
    mpca_weight: float = 0.03             # MPCA正则化权重
    acc_weight: float = 0.01              # 准确率正则化权重
    max_grad_norm: float = 1.0            # 梯度裁剪
    
    # === 训练控制 ===
    early_stopping_patience: int = 15     # 早停耐心
    early_stopping_metric: str = 'mpca'   # 早停指标
    log_interval: int = 10                # 日志间隔
    
    # === 类别权重 (3分类) ===
    class_weights: dict = None            # 类别权重 {0: 1.0, 1: 1.4, 2: 6.1}
    
    # === 模型标识 ===
    model_type: str = 'stage2_basic'      # 模型类型标识
    
    def __post_init__(self):
        """初始化后处理"""
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]
            
        if self.relation_hidden_dims is None:
            self.relation_hidden_dims = [64, 64]
            
        if self.class_weights is None:
            self.class_weights = {0: 1.0, 1: 1.4, 2: 6.1}
            
        # 根据temporal_mode调整模型类型
        if self.temporal_mode == 'lstm':
            self.model_type = 'stage2_lstm'
        elif self.temporal_mode == 'relation':
            self.model_type = 'stage2_relation'
        else:
            self.model_type = 'stage2_basic'
    
    def get_input_dim(self) -> int:
        """计算模型输入维度"""
        input_dim = 0
        
        if self.use_geometric:
            input_dim += 7      # Stage1几何特征
            
        if self.use_hog:
            input_dim += 64     # HoG特征
            
        if self.use_scene_context:
            input_dim += 1      # 场景上下文特征
            
        # Basic模式不包含时序特征
        if self.temporal_mode == 'none':
            pass  # 没有额外的时序维度
        
        return input_dim
    
    def get_num_classes(self) -> int:
        """获取类别数量"""
        return 3  # Stage2固定3分类
    
    def validate(self):
        """验证配置有效性"""
        # 检查temporal_mode
        if self.temporal_mode not in ['none', 'lstm', 'relation']:
            raise ValueError(f"temporal_mode must be 'none', 'lstm', or 'relation', got {self.temporal_mode}")
        
        # 检查relation模式特定配置
        if self.temporal_mode == 'relation':
            if self.fusion_strategy not in ['concat', 'elementwise']:
                raise ValueError(f"fusion_strategy must be 'concat' or 'elementwise', got {self.fusion_strategy}")
        
        # 检查必要特征
        if not (self.use_geometric or self.use_hog):
            raise ValueError("At least one of use_geometric or use_hog must be True")
        
        # 检查输入维度
        input_dim = self.get_input_dim()
        if input_dim <= 0:
            raise ValueError(f"Input dimension must be positive, got {input_dim}")
        
        print(f"✅ Configuration validated: {self.model_type}, input_dim={input_dim}")


def create_config_from_args(args) -> Stage2Config:
    """从命令行参数创建配置"""
    config = Stage2Config()
    
    # 数据配置
    config.data_path = args.data_path
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.frame_interval = getattr(args, 'frame_interval', 1)
    
    # 模型配置
    config.temporal_mode = getattr(args, 'temporal_mode', 'none')
    config.use_geometric = getattr(args, 'use_geometric', True)
    config.use_hog = getattr(args, 'use_hog_features', True)
    config.use_scene_context = getattr(args, 'use_scene_context', True)
    config.hidden_dims = getattr(args, 'hidden_dims', [128, 64, 32])
    config.dropout = args.dropout
    
    # Relation Network配置
    config.fusion_strategy = getattr(args, 'fusion_strategy', 'concat')
    config.relation_hidden_dims = getattr(args, 'relation_hidden_dims', [64, 64])
    config.spatial_feature_dim = getattr(args, 'spatial_feature_dim', 0)
    
    # 训练配置
    config.epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.optimizer = args.optimizer
    config.scheduler = args.scheduler
    config.step_size = getattr(args, 'step_size', 30)
    
    # 损失函数配置
    config.mpca_weight = args.mpca_weight
    config.acc_weight = args.acc_weight
    config.max_grad_norm = args.max_grad_norm
    
    # 训练控制
    config.early_stopping_patience = args.early_stopping_patience
    config.early_stopping_metric = args.early_stopping_metric
    config.log_interval = args.log_interval
    
    # 验证配置
    config.validate()
    return config


def add_config_args(parser: argparse.ArgumentParser):
    """添加配置相关的命令行参数"""
    
    # === 核心配置 ===
    parser.add_argument('--temporal_mode', type=str, default='none',
                        choices=['none', 'lstm', 'relation'],
                        help='Temporal processing mode (none=Basic, lstm=LSTM, relation=Relation Network)')
    
    # === 数据参数 ===
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    parser.add_argument('--frame_interval', type=int, default=1,
                        help='Frame sampling interval (1=every frame, 10=every 10th frame)')
    
    # === 特征配置 ===
    parser.add_argument('--use_geometric', action='store_true', default=True,
                        help='Use Stage1 geometric features')
    parser.add_argument('--no_geometric', dest='use_geometric', action='store_false',
                        help='Disable geometric features')
    parser.add_argument('--use_hog_features', action='store_true', default=True,
                        help='Use HoG visual features')
    parser.add_argument('--no_hog', dest='use_hog_features', action='store_false',
                        help='Disable HoG features')
    parser.add_argument('--use_scene_context', action='store_true', default=True,
                        help='Use scene context features')
    
    # === 模型参数 ===
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # === Relation Network参数 ===
    parser.add_argument('--fusion_strategy', type=str, default='concat',
                        choices=['concat', 'elementwise'],
                        help='Feature fusion strategy for Relation Network')
    parser.add_argument('--relation_hidden_dims', type=int, nargs='+', default=[64, 64],
                        help='Hidden layer dimensions for Relation Network MLP')
    parser.add_argument('--spatial_feature_dim', type=int, default=0,
                        help='Additional spatial feature dimension')
    
    # === 训练参数 ===
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30,
                        help='Step size for step scheduler')
    
    # === 损失函数参数 ===
    parser.add_argument('--mpca_weight', type=float, default=0.03,
                        help='MPCA regularization weight')
    parser.add_argument('--acc_weight', type=float, default=0.01,
                        help='Accuracy regularization weight')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # === 训练控制 ===
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--early_stopping_metric', type=str, default='mpca',
                        choices=['loss', 'accuracy', 'mpca'],
                        help='Early stopping metric')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval')


if __name__ == '__main__':
    # 测试配置系统
    print("Testing Stage2 Configuration System...")
    
    # 测试默认配置
    config = Stage2Config()
    config.validate()
    print(f"Default config: {config.model_type}, input_dim={config.get_input_dim()}")
    
    # 测试Basic模式
    config_basic = Stage2Config(temporal_mode='none')
    config_basic.validate()
    print(f"Basic mode: {config_basic.model_type}, input_dim={config_basic.get_input_dim()}")
    
    # 测试只使用几何特征
    config_geo = Stage2Config(use_hog=False)
    config_geo.validate()
    print(f"Geometric only: {config_geo.model_type}, input_dim={config_geo.get_input_dim()}")
    
    print("✅ Configuration system test completed!")