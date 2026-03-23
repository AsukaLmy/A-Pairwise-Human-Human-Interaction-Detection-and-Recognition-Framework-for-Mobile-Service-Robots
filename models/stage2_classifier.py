#!/usr/bin/env python3
"""
Unified Stage2 Behavior Classifier
Supports both Basic and LSTM modes with modular architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from .feature_extractors import BasicFeatureFusion


class LSTMStage2Classifier(nn.Module):
    """
    LSTM模式的Stage2行为分类器
    输入: 时序特征序列 [batch_size, sequence_length, feature_dim]
    输出: 3类行为分类
    """
    
    def __init__(self, feature_dim: int, sequence_length: int = 5, 
                 lstm_hidden_dim: int = 64, lstm_layers: int = 2,
                 bidirectional: bool = True, hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.2):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.num_classes = 3
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 计算LSTM输出维度
        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        
        # 全连接分类器
        layers = []
        in_dim = lstm_output_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(in_dim, self.num_classes))
        self.classifier = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            sequences: [batch_size, sequence_length, feature_dim] 时序特征序列
            
        Returns:
            torch.Tensor: [batch_size, 3] 分类logits
        """
        # 输入验证
        if sequences.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {sequences.dim()}D")
        
        batch_size, seq_len, feat_dim = sequences.shape
        
        if seq_len != self.sequence_length:
            raise ValueError(f"Expected sequence_length={self.sequence_length}, got {seq_len}")
        
        if feat_dim != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {feat_dim}")
        
        # 检查NaN/Inf
        if torch.isnan(sequences).any() or torch.isinf(sequences).any():
            raise ValueError("Input sequences contain NaN or Inf values")
        
        # LSTM处理
        lstm_out, (hidden, cell) = self.lstm(sequences)  # [batch, seq, hidden_dim * directions]
        
        # 使用最后时刻的输出进行分类
        if self.bidirectional:
            # 对于双向LSTM，取最后时刻的前向和后向输出
            final_output = lstm_out[:, -1, :]  # [batch, hidden_dim * 2]
        else:
            final_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # 分类
        logits = self.classifier(final_output)  # [batch, 3]
        
        return logits
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'LSTMStage2Classifier',
            'feature_dim': self.feature_dim,
            'sequence_length': self.sequence_length,
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'lstm_layers': self.lstm_layers,
            'bidirectional': self.bidirectional,
            'hidden_dims': self.hidden_dims,
            'num_classes': self.num_classes,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'dropout': self.dropout
        }


class RelationStage2Classifier(nn.Module):
    """
    Relation Network模式的Stage2行为分类器
    分别提取两人特征，然后进行关系建模
    输入: person_A特征 + person_B特征 + 空间关系特征
    输出: 3类行为分类
    """
    
    def __init__(self, person_feature_dim: int, spatial_feature_dim: int = 0,
                 hidden_dims: List[int] = [64, 64], dropout: float = 0.2,
                 fusion_strategy: str = 'concat'):
        super().__init__()
        
        self.person_feature_dim = person_feature_dim
        self.spatial_feature_dim = spatial_feature_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.fusion_strategy = fusion_strategy
        self.num_classes = 3
        
        # 计算关系网络输入维度
        if fusion_strategy == 'concat':
            # Simple Concatenation: person_A + person_B + spatial
            relation_input_dim = person_feature_dim * 2 + spatial_feature_dim
        elif fusion_strategy == 'elementwise':
            # Element-wise operations: multiply + subtract + add + spatial
            relation_input_dim = person_feature_dim * 3 + spatial_feature_dim
        else:
            raise ValueError(f"Unknown fusion_strategy: {fusion_strategy}")
        
        # 构建关系网络MLP
        layers = []
        prev_dim = relation_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, self.num_classes))
        self.relation_mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.relation_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, person_A_features: torch.Tensor, person_B_features: torch.Tensor,
                spatial_features: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            person_A_features: [batch_size, person_feature_dim] 人员A特征
            person_B_features: [batch_size, person_feature_dim] 人员B特征  
            spatial_features: [batch_size, spatial_feature_dim] 空间关系特征(可选)
            
        Returns:
            torch.Tensor: [batch_size, 3] 分类logits
        """
        # 输入验证：支持单样本(1D)和batch(2D)
        # 如果输入为1D，尝试unsqueeze以形成batch
        if person_A_features.dim() == 1:
            person_A_features = person_A_features.unsqueeze(0)
        if person_B_features.dim() == 1:
            person_B_features = person_B_features.unsqueeze(0)
        if spatial_features is not None and spatial_features.dim() == 1:
            spatial_features = spatial_features.unsqueeze(0)

        batch_size = person_A_features.size(0)
        if person_B_features.size(0) != batch_size:
            raise ValueError("person_A and person_B must have same batch size")

        if person_A_features.size(1) != self.person_feature_dim:
            raise ValueError(f"Expected person_feature_dim={self.person_feature_dim}, got {person_A_features.size(1)}")
        
        # 特征融合
        if self.fusion_strategy == 'concat':
            # Simple Concatenation
            relation_input = torch.cat([person_A_features, person_B_features], dim=1)
            
        elif self.fusion_strategy == 'elementwise':
            # Element-wise operations
            multiply = person_A_features * person_B_features    # 乘性交互
            subtract = torch.abs(person_A_features - person_B_features)  # 差异信息
            add = person_A_features + person_B_features         # 加性组合
            relation_input = torch.cat([multiply, subtract, add], dim=1)
        
        # 添加空间特征(如果有)
        if spatial_features is not None and self.spatial_feature_dim > 0:
            if spatial_features.size(1) != self.spatial_feature_dim:
                raise ValueError(f"Expected spatial_feature_dim={self.spatial_feature_dim}, got {spatial_features.size(1)}")
            relation_input = torch.cat([relation_input, spatial_features], dim=1)
        
        # 关系推理
        logits = self.relation_mlp(relation_input)
        
        return logits
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        relation_input_dim = self.person_feature_dim * 2 + self.spatial_feature_dim
        if self.fusion_strategy == 'elementwise':
            relation_input_dim = self.person_feature_dim * 3 + self.spatial_feature_dim
        
        return {
            'model_type': 'RelationStage2Classifier',
            'person_feature_dim': self.person_feature_dim,
            'spatial_feature_dim': self.spatial_feature_dim,
            'fusion_strategy': self.fusion_strategy,
            'hidden_dims': self.hidden_dims,
            'relation_input_dim': relation_input_dim,
            'num_classes': self.num_classes,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'dropout': self.dropout
        }


class BasicStage2Classifier(nn.Module):
    """
    Basic模式的Stage2行为分类器
    输入: 几何特征(7) + HoG特征(64) + 场景上下文(1) = 72维
    输出: 3类行为分类
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.2, use_attention: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_attention = use_attention
        self.num_classes = 3
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 可选的注意力机制 (Basic模式通常不使用)
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[0], num_heads=8, batch_first=True
            )
        
        # 隐藏层
        layers = []
        in_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(in_dim, self.num_classes))
        self.classifier = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: [batch_size, input_dim] 输入特征向量
            
        Returns:
            torch.Tensor: [batch_size, 3] 分类logits
        """
        # 输入验证
        if features.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {features.dim()}D")
        
        if features.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {features.size(1)}")
        
        # 检查NaN/Inf
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("Input features contain NaN or Inf values")
        
        # 输入层处理
        processed_features = self.input_layer(features)  # [batch, hidden_dims[0]]
        
        # 可选的注意力机制
        if self.use_attention:
            # 重塑为序列格式进行注意力计算
            feature_seq = processed_features.unsqueeze(1)  # [batch, 1, hidden_dims[0]]
            attended_features, _ = self.attention(feature_seq, feature_seq, feature_seq)
            processed_features = attended_features.squeeze(1)  # [batch, hidden_dims[0]]
        
        # 分类
        logits = self.classifier(processed_features)  # [batch, 3]
        
        return logits
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'BasicStage2Classifier',
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'num_classes': self.num_classes,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'use_attention': self.use_attention,
            'dropout': self.dropout
        }


class Stage2Loss(nn.Module):
    """Stage2专用损失函数：Focal Loss + MPCA + 准确率正则化"""

    def __init__(self, class_weights: dict, gamma: float = 2.0, mpca_weight: float = 0.03, acc_weight: float = 0.01):
        super().__init__()

        # 类别权重处理
        if isinstance(class_weights, dict):
            class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float32)
        else:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        # 注册为buffer，PyTorch自动管理设备转移
        self.register_buffer('alpha_weights', class_weights_tensor)
        self.gamma = gamma
        self.mpca_weight = mpca_weight
        self.acc_weight = acc_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        计算损失
        
        Args:
            predictions: [batch_size, 3] 预测logits
            targets: [batch_size] 真实标签
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细信息
        """
        # 输入验证
        if predictions.dim() != 2 or predictions.size(1) != 3:
            raise ValueError(f"Expected predictions shape [batch_size, 3], got {predictions.shape}")
        
        if targets.dim() != 1:
            raise ValueError(f"Expected 1D targets tensor, got {targets.dim()}D")
        
        if predictions.size(0) != targets.size(0):
            raise ValueError(f"Batch size mismatch: predictions {predictions.size(0)}, targets {targets.size(0)}")
        
        # 标签范围验证
        if (targets < 0).any() or (targets >= 3).any():
            raise ValueError(f"Target labels must be in range [0, 2], got min={targets.min()}, max={targets.max()}")
        
        # 主损失：Focal Loss
        ce_loss = self._focal_loss(predictions, targets)
        
        # # MPCA正则化：最小化类别间准确率方差（可微分版本）
        # probs = F.softmax(predictions, dim=1)  # [batch_size, 3]
        # per_class_acc = []
        
        # for class_id in range(3):
        #     mask = targets == class_id
        #     if mask.sum() > 0:
        #         # 使用软准确率：预测该类的概率
        #         class_probs = probs[mask, class_id]  # 该类样本预测为该类的概率
        #         class_acc = class_probs.mean()  # 软准确率，可微分
        #         per_class_acc.append(class_acc)
        
        # # 数值稳定性处理
        # if len(per_class_acc) > 1:
        #     acc_tensor = torch.stack(per_class_acc)
        #     # 使用方差而非标准差，避免开方运算的数值不稳定
        #     mpca_loss = torch.var(acc_tensor) + 1e-8
        # else:
        #     mpca_loss = torch.tensor(1e-8, device=predictions.device, requires_grad=True)
        
        # # 准确率正则化：鼓励整体准确率提升（使用软准确率）
        # with torch.no_grad():
        #     pred_classes = torch.argmax(predictions, dim=1)
        #     overall_acc = (pred_classes == targets).float().mean()
        
        # # 软准确率正则化（可微分）
        # soft_acc = torch.mean(torch.sum(probs * F.one_hot(targets, num_classes=3).float(), dim=1))
        # acc_regularization = -soft_acc  # 负号表示最大化准确率
        
        # 计算总损失 (仅使用交叉熵)
        total_loss = ce_loss
        
        # 计算整体准确率用于监控
        with torch.no_grad():
            pred_classes = torch.argmax(predictions, dim=1)
            overall_acc = (pred_classes == targets).float().mean()
        
        # 返回详细信息
        loss_dict = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            # 'mpca_loss': mpca_loss.item(),  # 已注释
            'overall_acc': overall_acc.item()
        }
        
        return total_loss, loss_dict

    def _focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss

        Args:
            predictions: [batch_size, num_classes] logits
            targets: [batch_size] 目标标签

        Returns:
            torch.Tensor: focal loss值
        """
        # 计算交叉熵（不进行reduction）
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')

        # 计算概率
        probs = F.softmax(predictions, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # p_t

        # 计算focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # 应用focal weight
        focal_loss = focal_weight * ce_loss

        # 应用alpha权重
        alpha_t = self.alpha_weights[targets]
        focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class Stage2Evaluator:
    """Stage2专用评估器"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()
    
    def reset(self):
        """重置评估状态"""
        self.predictions = []
        self.targets = []
        self.correct_per_class = {i: 0 for i in range(self.num_classes)}
        self.total_per_class = {i: 0 for i in range(self.num_classes)}
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        更新评估结果
        
        Args:
            predictions: [batch_size, 3] logits或[batch_size] 预测类别
            targets: [batch_size] 真实标签
        """
        # 输入验证
        if predictions.size(0) != targets.size(0):
            raise ValueError(f"Batch size mismatch: predictions {predictions.size(0)}, targets {targets.size(0)}")
        
        if targets.dim() != 1:
            raise ValueError(f"Expected 1D targets tensor, got {targets.dim()}D")
        
        # 获取预测类别
        if predictions.dim() > 1:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions
        
        # 转换为numpy
        pred_classes = pred_classes.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # 存储预测结果
        self.predictions.extend(pred_classes)
        self.targets.extend(targets)
        
        # 更新各类别统计
        for pred, target in zip(pred_classes, targets):
            if 0 <= target < self.num_classes:  # 验证标签范围
                self.total_per_class[target] += 1
                if pred == target:
                    self.correct_per_class[target] += 1
    
    def compute_metrics(self) -> dict:
        """计算评估指标"""
        if not self.predictions:
            # 返回默认值字典，避免KeyError
            return {
                'overall_accuracy': 0.0,
                'mpca': 0.0,
                'per_class_accuracy': {i: 0.0 for i in range(self.num_classes)},
                'precision': np.zeros(self.num_classes),
                'recall': np.zeros(self.num_classes),
                'f1_score': np.zeros(self.num_classes),
                'macro_f1': 0.0,
                'weighted_f1': 0.0,
                'support': np.zeros(self.num_classes),
                'confusion_matrix': np.zeros((self.num_classes, self.num_classes)),
                'class_names': self.class_names
            }
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # 整体准确率
        overall_acc = np.mean(predictions == targets)
        
        # 各类别准确率
        per_class_acc = {}
        valid_classes = []
        
        for class_id in range(self.num_classes):
            if self.total_per_class[class_id] > 0:
                acc = self.correct_per_class[class_id] / self.total_per_class[class_id]
                per_class_acc[class_id] = acc
                valid_classes.append(acc)
            else:
                per_class_acc[class_id] = 0.0
        
        # MPCA (Mean Per-Class Accuracy)
        mpca = np.mean(valid_classes) if valid_classes else 0.0
        
        # 计算混淆矩阵相关指标
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0, labels=range(self.num_classes)
        )
        
        # 加权平均
        macro_f1 = np.mean(f1)
        weighted_f1 = np.average(f1, weights=support) if support.sum() > 0 else 0.0
        
        # 混淆矩阵
        cm = confusion_matrix(targets, predictions, labels=range(self.num_classes))
        
        return {
            'overall_accuracy': overall_acc,
            'mpca': mpca,
            'per_class_accuracy': per_class_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'support': support,
            'confusion_matrix': cm,
            'class_names': self.class_names
        }
    
    def print_evaluation_report(self):
        """打印详细的评估报告"""
        metrics = self.compute_metrics()
        
        if not metrics or metrics['overall_accuracy'] == 0:
            print("No evaluation data available.")
            return
        
        print("\n" + "="*60)
        print("STAGE2 BEHAVIOR CLASSIFICATION EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"MPCA (Mean Per-Class Accuracy): {metrics['mpca']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print("-" * 80)
        print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 80)
        
        for i, class_name in enumerate(metrics['class_names']):
            if i < len(metrics['precision']):
                print(f"{class_name:<25} {metrics['precision'][i]:<10.4f} "
                      f"{metrics['recall'][i]:<10.4f} {metrics['f1_score'][i]:<10.4f} "
                      f"{int(metrics['support'][i]):<10}")
        
        print("\nPer-Class Accuracy:")
        print("-" * 40)
        for class_id, acc in metrics['per_class_accuracy'].items():
            class_name = metrics['class_names'][class_id] if class_id < len(metrics['class_names']) else f"Class_{class_id}"
            print(f"{class_name:<25}: {acc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])


if __name__ == '__main__':
    # 测试Stage2分类器
    print("Testing Stage2 Classifier...")
    
    # 测试Basic分类器
    print("\n1. Testing BasicStage2Classifier...")
    input_dim = 72  # 7几何 + 64HoG + 1场景上下文
    model = BasicStage2Classifier(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        use_attention=False
    )
    
    print(f"Model info: {model.get_model_info()}")
    
    # 测试前向传播
    batch_size = 8
    features = torch.randn(batch_size, input_dim)
    targets = torch.randint(0, 3, (batch_size,))
    
    print(f"\nInput shape: {features.shape}")
    logits = model(features)
    print(f"Output shape: {logits.shape}")
    
    # 测试损失函数
    print("\n2. Testing Stage2Loss...")
    class_weights = {0: 1.0, 1: 1.4, 2: 6.1}
    criterion = Stage2Loss(class_weights, mpca_weight=0.03, acc_weight=0.01)
    
    loss, loss_dict = criterion(logits, targets)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss details: {loss_dict}")
    
    # 测试评估器
    print("\n3. Testing Stage2Evaluator...")
    class_names = ['Walking Together', 'Standing Together', 'Sitting Together']
    evaluator = Stage2Evaluator(class_names)
    
    # 添加一些测试数据
    for _ in range(3):
        test_logits = torch.randn(batch_size, 3)
        test_targets = torch.randint(0, 3, (batch_size,))
        evaluator.update(test_logits, test_targets)
    
    evaluator.print_evaluation_report()
    
    print("\n✅ Stage2 classifier test completed!")