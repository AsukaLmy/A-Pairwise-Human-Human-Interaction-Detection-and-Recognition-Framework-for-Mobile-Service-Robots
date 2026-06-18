#!/usr/bin/env python3
"""
GNN End-to-End Loss (Stage 1 + Stage 2 joint training)

与 GNNMultiTaskLoss 的核心区别：
  行为分类损失通过 Stage 1（group_head）的预测概率进行软门控（soft-gating），
  使 Stage 1 的梯度与行为分类任务直接耦合，实现端到端联合训练。

损失结构：
    p_s1[i]  = sigmoid(group_logits[i])   # GT正对的 Stage1 预测概率，i ∈ [0, P)
    L_e2e    = mean(p_s1[i] * per_sample_focal(behavior_logits[i], label[i]))
    L_group  = BinaryFocal(group_logits, group_labels)   # 与原版相同
    L_total  = lambda_behavior * L_e2e + lambda_group * L_group

端到端评估指标：
    predicted_pos[i] = (p_s1[i] > threshold)
    correct_behav[i] = (argmax(behavior_logits[i]) == label[i])
    e2e_acc = mean(predicted_pos[i] AND correct_behav[i])
    含义：GT正对中 Stage1正确检测 且 Stage2正确分类行为的比例
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.gnn_multitask_loss import BinaryFocalLoss


# ============================================================================
# End-to-End Loss
# ============================================================================

class GNNEndToEndLoss(nn.Module):
    """
    端到端软门控多任务损失。

    Args:
        lambda_behavior: 行为分类损失权重（默认 0.8）
        lambda_group:    分组检测损失权重（默认 0.2）
        class_weights:   3类行为的 per-class focal 权重字典 {0:..., 1:..., 2:...}
        focal_gamma:     focal loss 的 gamma 参数
        group_alpha:     分组 binary focal 的正类权重 alpha
        threshold:       Stage1 二值化阈值（用于评估指标，不影响梯度）
    """

    def __init__(
        self,
        lambda_behavior: float = 0.8,
        lambda_group:    float = 0.2,
        class_weights:   Optional[Dict] = None,
        focal_gamma:     float = 2.0,
        group_alpha:     float = 0.25,
        threshold:       float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_behavior = lambda_behavior
        self.lambda_group    = lambda_group
        self.focal_gamma     = focal_gamma
        self.threshold       = threshold
        self.label_smoothing = label_smoothing

        # ---- 3类行为 Focal 权重 ----
        if class_weights is None:
            class_weights = {0: 1.0, 1: 1.4, 2: 6.1}
        w = torch.tensor(
            [class_weights[i] for i in range(len(class_weights))],
            dtype=torch.float32
        )
        self.register_buffer('alpha_weights', w)

        # ---- Binary Focal Loss（分组检测，与原版相同） ----
        self.group_loss_fn = BinaryFocalLoss(alpha=group_alpha, gamma=focal_gamma)

    def forward(
        self,
        behavior_logits: torch.Tensor,   # [P, 3]
        behavior_labels: torch.Tensor,   # [P]   int64
        group_logits:    torch.Tensor,   # [P+Q, 1]
        group_labels:    torch.Tensor,   # [P+Q]  float32 {0,1}
        warmup_gate:     float = 1.0,    # 0=无软门控(等同MultiTask), 1=完全软门控
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            total_loss : scalar tensor（带梯度）
            loss_dict  : 各损失分量与评估指标
        """
        P = behavior_labels.size(0)

        # ================================================================
        # Stage 1 软门控概率（GT 正对，前 P 个 group_logit）
        # ================================================================
        # group_logits 结构：前 P 个为正对（label=1），后 Q 个为负对（label=0）
        p_stage1 = torch.sigmoid(group_logits[:P].squeeze(-1))   # [P]，有梯度

        # ================================================================
        # Per-sample 3类行为 Focal Loss（不直接 mean）
        # ================================================================
        log_pt    = F.log_softmax(behavior_logits, dim=1)          # [P, 3]
        batch_idx = torch.arange(P, device=behavior_logits.device)
        log_pt_y  = log_pt[batch_idx, behavior_labels]            # [P]  hard-label log-prob
        pt        = log_pt_y.exp()                                 # [P]  for focal weight
        alpha_t   = self.alpha_weights[behavior_labels]           # [P]
        focal_w   = alpha_t * (1 - pt) ** self.focal_gamma        # [P]
        if self.label_smoothing > 0.0:
            C = behavior_logits.size(1)
            smooth_eps = self.label_smoothing
            smoothed = (1.0 - smooth_eps) * F.one_hot(
                behavior_labels, C).float() + smooth_eps / C      # [P, 3]
            ce_per_sample = -(smoothed * log_pt).sum(dim=1)       # [P]
        else:
            ce_per_sample = -log_pt_y                             # [P]
        per_sample_loss = focal_w * ce_per_sample                  # [P]

        # ================================================================
        # 软门控行为损失：warmup_gate 控制软门控强度
        #   warmup_gate=0: effective_p=1（等权，等同 GNNMultiTaskLoss）
        #   warmup_gate=1: effective_p=p_stage1（完全软门控）
        # 梯度路径：L_e2e → p_stage1 → group_head（当 warmup_gate > 0 时有效）
        # ================================================================
        effective_p   = warmup_gate * p_stage1 + (1.0 - warmup_gate)   # [P]
        behavior_loss = (effective_p * per_sample_loss).mean()

        # ================================================================
        # 分组检测损失（Binary Focal，覆盖 P+Q 对，与原版相同）
        # ================================================================
        group_loss = self.group_loss_fn(group_logits, group_labels)

        # ================================================================
        # 总损失
        # ================================================================
        total_loss = self.lambda_behavior * behavior_loss + self.lambda_group * group_loss

        # ================================================================
        # 评估指标（无梯度）
        # ================================================================
        with torch.no_grad():
            behavior_preds = behavior_logits.argmax(dim=1)        # [P]
            behavior_acc   = (behavior_preds == behavior_labels).float().mean()

            # ---------- Stage 1 指标 ----------
            predicted_pos  = p_stage1 > self.threshold            # [P] bool
            stage1_recall  = predicted_pos.float().mean()         # GT正对中被检测到的比例

            # group_acc 覆盖全部 P+Q 对
            group_preds_all = (group_logits.squeeze(-1) > 0).float()
            group_acc       = (group_preds_all == group_labels).float().mean()

            # group_logits 后 Q 个为负对
            Q = group_logits.size(0) - P
            if Q > 0:
                neg_preds = group_preds_all[P:]                   # [Q]，预测为1表示误检
                # 正预测数 / (正预测数 + 误检数)
                tp = predicted_pos.float().sum()
                fp = neg_preds.sum()
                stage1_precision = tp / (tp + fp + 1e-8)
            else:
                stage1_precision = torch.tensor(float('nan'))

            # ---------- 端到端准确率 ----------
            # GT正对中：Stage1正确检测 且 Stage2正确分类 的比例
            correct_behav = (behavior_preds == behavior_labels)   # [P]
            e2e_correct   = predicted_pos & correct_behav          # [P]
            e2e_acc       = e2e_correct.float().mean()

            # ---------- 每类行为准确率（MPCA） ----------
            per_class_acc = {}
            for c in range(3):
                mask = behavior_labels == c
                if mask.sum() > 0:
                    per_class_acc[c] = (behavior_preds[mask] == c).float().mean().item()
                else:
                    per_class_acc[c] = float('nan')

            mpca = sum(v for v in per_class_acc.values()
                       if v == v) / max(1, sum(
                           1 for v in per_class_acc.values() if v == v))

            # 用于外部累计的绝对计数（方便跨 batch 聚合）
            e2e_correct_count = int(e2e_correct.sum().item())
            e2e_total_count   = P

        loss_dict = {
            'total_loss':        total_loss.item(),
            'behavior_loss':     behavior_loss.item(),
            'group_loss':        group_loss.item(),
            # 传统指标（假设分组完全正确）
            'behavior_acc':      behavior_acc.item(),
            'group_acc':         group_acc.item(),
            'mpca':              mpca,
            'per_class_acc':     per_class_acc,
            # 端到端指标（Stage1误差传导至行为准确率）
            'e2e_acc':           e2e_acc.item(),
            'stage1_recall':     stage1_recall.item(),
            'stage1_precision':  stage1_precision.item() if not stage1_precision.isnan() else float('nan'),
            # 用于跨 batch 累计的计数
            'e2e_correct_count': e2e_correct_count,
            'e2e_total_count':   e2e_total_count,
        }

        return total_loss, loss_dict


# ============================================================================
# Factory
# ============================================================================

def create_e2e_loss(config) -> GNNEndToEndLoss:
    """从 GNNGeometricConfig 创建端到端损失实例。"""
    return GNNEndToEndLoss(
        lambda_behavior=getattr(config, 'lambda_behavior', 0.8),
        lambda_group=getattr(config, 'lambda_group', 0.2),
        class_weights=getattr(config, 'class_weights', None),
        focal_gamma=getattr(config, 'focal_gamma', 2.0),
        threshold=getattr(config, 'e2e_threshold', 0.5),
        label_smoothing=getattr(config, 'label_smoothing', 0.0),
    )


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == '__main__':
    print("Testing GNNEndToEndLoss...")
    criterion = GNNEndToEndLoss(lambda_behavior=0.8, lambda_group=0.2, threshold=0.5)

    P, Q = 20, 40
    behavior_logits = torch.randn(P, 3, requires_grad=True)
    behavior_labels = torch.randint(0, 3, (P,))
    group_logits    = torch.randn(P + Q, 1, requires_grad=True)
    group_labels    = torch.cat([torch.ones(P), torch.zeros(Q)])

    loss, d = criterion(behavior_logits, behavior_labels, group_logits, group_labels)

    print(f"total_loss:        {d['total_loss']:.4f}")
    print(f"behavior_loss:     {d['behavior_loss']:.4f}  (soft-gated)")
    print(f"group_loss:        {d['group_loss']:.4f}")
    print(f"behavior_acc:      {d['behavior_acc']:.3f}  (traditional, GT pairs)")
    print(f"e2e_acc:           {d['e2e_acc']:.3f}  (end-to-end, incl. Stage1 errors)")
    print(f"stage1_recall:     {d['stage1_recall']:.3f}")
    print(f"stage1_precision:  {d['stage1_precision']:.3f}")
    print(f"group_acc:         {d['group_acc']:.3f}")
    print(f"MPCA:              {d['mpca']:.3f}")
    print(f"e2e_correct/total: {d['e2e_correct_count']}/{d['e2e_total_count']}")

    loss.backward()
    print("\nBackward pass OK.")

    # 验证 group_logits[:P] 确实接收到来自行为损失的梯度
    assert group_logits.grad is not None, "group_logits.grad should not be None"
    grad_from_pos = group_logits.grad[:P].abs().mean().item()
    grad_from_neg = group_logits.grad[P:].abs().mean().item()
    print(f"group_logits grad (pos pairs): {grad_from_pos:.6f}  ← from soft-gating + group loss")
    print(f"group_logits grad (neg pairs): {grad_from_neg:.6f}  ← from group loss only")
    assert grad_from_pos > 0, "GT positive pairs should receive gradient"
    print("\nAll tests passed!")
