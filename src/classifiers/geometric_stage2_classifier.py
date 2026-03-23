#!/usr/bin/env python3
"""
Stage2 geometric behavior classifier
Specialized architecture for 4-class behavior classification using 16D geometric+temporal features (方案A)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class GeometricStage2Classifier(nn.Module):
    """
    专门为3类基础行为分类设计的几何+HoG+时序特征分类器
    输入：动态维度特征 (7几何 + 64HoG + 可选9时序)
    输出：3类行为分类 (Walking Together, Standing Together, Sitting Together)
    """
    
    def __init__(self, input_dim=80, hidden_dims=[64, 32, 16], dropout=0.2, use_attention=True):
        super().__init__()
        
        # 保存输入维度（动态）
        self.input_dim = input_dim
        
        # 特征维度分组 (用于特征重要性分析)
        self.geometric_dim = 7      # 基础几何特征
        self.hog_dim = 64          # HoG特征
        self.temporal_dim = 9       # 时序特征 (4基础 + 5增强)
        
        # 输入特征处理层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 特征注意力机制 (可选)
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[0], num_heads=8, batch_first=True
            )
        
        # 主干分类器
        layers = []
        in_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 3))  # 3类基础行为分类
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
    
    def forward(self, features):
        """
        Args:
            features: [batch_size, input_dim] 动态维度特征向量
        Returns:
            [batch_size, 3] 分类logits
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
        
        # 最终分类
        logits = self.classifier(processed_features)  # [batch, 3]
        
        return logits
    

    def get_feature_importance(self, features):
        """
        分析特征重要性（通过梯度计算）

        Args:
            features: [batch_size, input_dim] 输入特征
        Returns:
            Dict: 各组特征的重要性分数
        """
        self.eval()
        features.requires_grad_(True)

        # 前向传播
        logits = self.forward(features)

        # 计算各类的梯度
        importance_scores = {}
        for class_id in range(3):  # 3类分析
            # 计算对该类的梯度
            class_score = logits[:, class_id].sum()
            grads = torch.autograd.grad(class_score, features, retain_graph=True)[0]

            # 分析不同特征组的重要性（基于已知的特征分组）
            geo_importance = torch.mean(torch.abs(grads[:, :self.geometric_dim]))

            # HoG特征重要性 - 修复设备和dtype一致性
            if features.size(1) > self.geometric_dim:
                hog_start = self.geometric_dim
                hog_end = min(hog_start + self.hog_dim, features.size(1))
                hog_importance = torch.mean(torch.abs(grads[:, hog_start:hog_end]))
            else:
                # 使用与grads相同的设备和dtype
                hog_importance = torch.zeros_like(geo_importance)  # ✅ 保持一致性

            # 时序特征重要性（如果存在）- 修复设备和dtype一致性
            if features.size(1) > self.geometric_dim + self.hog_dim:
                temporal_start = self.geometric_dim + self.hog_dim
                temporal_importance = torch.mean(torch.abs(grads[:, temporal_start:]))
            else:
                # 使用与grads相同的设备和dtype
                temporal_importance = torch.zeros_like(geo_importance)  # ✅ 保持一致性

            importance_scores[f'class_{class_id}'] = {
                'geometric': geo_importance.item(),
                'hog': hog_importance.item(),
                'temporal': temporal_importance.item()
            }

        return importance_scores

class Stage2Loss(nn.Module):
    """Stage2专用损失函数：Focal Loss + MPCA + 准确率正则"""

    def __init__(self, class_weights, gamma=2.0, mpca_weight=0.1, acc_weight=0.05):
        super().__init__()
        if isinstance(class_weights, dict):
            class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float32)
        else:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        # 注册为buffer，PyTorch自动管理设备转移
        self.register_buffer('alpha_weights', class_weights_tensor)
        self.gamma = gamma
        self.mpca_weight = mpca_weight
        self.acc_weight = acc_weight
    
    def forward(self, predictions, targets):
        """
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
            raise ValueError(f"Batch size mismatch: predictions {predictions.size(0)}, targets{targets.size(0)}")

        # 标签范围验证
        if (targets < 0).any() or (targets >= 3).any():
            raise ValueError(f"Target labels must be in range [0, 2], got min={targets.min()},max={targets.max()}")



        # 主损失：Focal Loss
        ce_loss = self._focal_loss(predictions, targets)
        
        # MPCA正则化：最小化类别间准确率方差（可微分版本）
        probs = F.softmax(predictions, dim=1)  # [batch_size, 3]
        per_class_acc = []

        for class_id in range(3):
            mask = targets == class_id
            if mask.sum() > 0:
                # 使用软准确率：预测该类的概率 * 真实标签匹配度
                class_probs = probs[mask, class_id]  # 该类样本预测为该类的概率
                class_acc = class_probs.mean()  # 软准确率，可微分
                per_class_acc.append(class_acc)

        # Line 181-184 - 更全面的稳定性处理
        if len(per_class_acc) > 1:
            acc_tensor = torch.stack(per_class_acc)
            # 使用方差而非标准差，避免开方运算的数值不稳定
            mpca_loss = torch.var(acc_tensor) + 1e-8
            # 或者使用clamp确保最小值
            # mpca_loss = torch.clamp(torch.std(acc_tensor), min=1e-8)
        else:
            mpca_loss = torch.tensor(1e-8, device=predictions.device, requires_grad=True)  # 非零值保持梯度流 

        # 准确率正则化：鼓励整体准确率提升（使用软准确率）
        with torch.no_grad():
            pred_classes = torch.argmax(predictions, dim=1)
            overall_acc = (pred_classes == targets).float().mean()
        
        # 软准确率正则化（可微分）
        soft_acc = torch.mean(torch.sum(probs * F.one_hot(targets, num_classes=3).float(), dim=1))
        acc_regularization = -soft_acc  # 负号表示最大化准确率

        # 计算总损失
        total_loss = ce_loss + self.mpca_weight * mpca_loss + self.acc_weight * acc_regularization
        
        # 返回详细信息
        loss_dict = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'mpca_loss': mpca_loss.item(),
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
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """重置评估状态"""
        self.predictions = []
        self.targets = []
        self.correct_per_class = {i: 0 for i in range(len(self.class_names))}
        self.total_per_class = {i: 0 for i in range(len(self.class_names))}
    
    def update(self, predictions, targets):
        """
        更新评估结果
        
        Args:
            predictions: [batch_size, 3] logits或[batch_size] 预测类别
            targets: [batch_size] 真实标签
        """
        # 确保输入是numpy数组
        if hasattr(predictions, 'cpu'):  # PyTorch tensor
            predictions = predictions.cpu().numpy()
        if hasattr(targets, 'cpu'):  # PyTorch tensor
            targets = targets.cpu().numpy()
        
        # 输入验证
        if predictions.shape[0] != targets.shape[0]:
            raise ValueError(f"Batch size mismatch: predictions {predictions.shape[0]}, targets {targets.shape[0]}")

        if len(targets.shape) != 1:
            raise ValueError(f"Expected 1D targets array, got {len(targets.shape)}D")

        if len(predictions.shape) > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions
        
        # 确保是numpy数组
        pred_classes = np.asarray(pred_classes)
        targets = np.asarray(targets)
        
        # 存储预测结果
        self.predictions.extend(pred_classes)
        self.targets.extend(targets)
        
        # 更新各类别统计
        for pred, target in zip(pred_classes, targets):
            self.total_per_class[target] += 1
            if pred == target:
                self.correct_per_class[target] += 1
    
    def compute_metrics(self):
        """计算评估指标"""
        if not self.predictions:
            # 返回包含默认值的完整指标字典，避免KeyError
            return {
                'overall_accuracy': 0.0,
                'mpca': 0.0,
                'per_class_accuracy': {i: 0.0 for i in range(len(self.class_names))},
                'precision': np.zeros(len(self.class_names)),
                'recall': np.zeros(len(self.class_names)),
                'f1_score': np.zeros(len(self.class_names)),
                'macro_f1': 0.0,
                'weighted_f1': 0.0,
                'support': np.zeros(len(self.class_names)),
                'confusion_matrix': np.zeros((len(self.class_names), len(self.class_names))),
                'class_names': self.class_names
            }

        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # 整体准确率
        overall_acc = np.mean(predictions == targets)
        
        # 各类别准确率
        per_class_acc = {}
        valid_classes = []
        
        for class_id in range(len(self.class_names)):
            if self.total_per_class[class_id] > 0:
                acc = self.correct_per_class[class_id] / self.total_per_class[class_id]
                per_class_acc[class_id] = acc
                valid_classes.append(acc)
        
        # MPCA (Mean Per-Class Accuracy)
        mpca = np.mean(valid_classes) if valid_classes else 0.0
        
        # 计算混淆矩阵相关指标
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # 加权平均
        macro_f1 = np.mean(f1)
        weighted_f1 = np.average(f1, weights=support)
        
        # 混淆矩阵
        cm = confusion_matrix(targets, predictions, labels=range(len(self.class_names)))
        
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
        
        if not metrics:
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
                      f"{metrics['support'][i]:<10}")
        
        print("\nPer-Class Accuracy:")
        print("-" * 40)
        for class_id, acc in metrics['per_class_accuracy'].items():
            class_name = metrics['class_names'][class_id]
            print(f"{class_name:<25}: {acc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])


if __name__ == '__main__':
    # 测试Stage2分类器
    print("Testing GeometricStage2Classifier...")
    
    # 创建模型
    model = GeometricStage2Classifier(
        input_dim=80,
        hidden_dims=[64, 32, 16],
        dropout=0.2,
        use_attention=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 8
    features = torch.randn(batch_size, 80)
    targets = torch.randint(0, 3, (batch_size,))
    
    print(f"\nInput shape: {features.shape}")
    
    # 前向传播
    logits = model(features)
    print(f"Output shape: {logits.shape}")
    
    # 测试损失函数
    class_weights ={0: 1.0, 1: 1.4, 2: 6.1}
    criterion = Stage2Loss(class_weights, mpca_weight=0.1, acc_weight=0.05)
    
    loss, loss_dict = criterion(logits, targets)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss details: {loss_dict}")
    
    # 测试特征重要性
    print(f"\nTesting feature importance...")
    importance = model.get_feature_importance(features)
    print(f"Feature importance (sample): {importance['class_0']}")
    
    # 测试评估器
    print(f"\nTesting evaluator...")
    class_names = ['Walking Together', 'Standing Together', 'Sitting Together']
    
    evaluator = Stage2Evaluator(class_names)
    evaluator.update(logits, targets)
    evaluator.print_evaluation_report()
    
    print("\nStage2 classifier test completed!")