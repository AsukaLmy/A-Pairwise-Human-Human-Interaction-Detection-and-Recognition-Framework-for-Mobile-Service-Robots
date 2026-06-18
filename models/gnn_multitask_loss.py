#!/usr/bin/env python3
"""
GNN Multi-Task Loss
Combines behavior classification loss (3-class Focal) with
grouping detection loss (binary Focal) for end-to-end multi-task training.

Loss structure:
    L_total = lambda_behavior * L_behavior + lambda_group * L_group

    L_behavior: 3-class Focal Loss on labeled interaction pairs
                (Walking/Standing/Sitting Together)
    L_group:    Binary Focal Loss on all pairs (positive = labeled = 1,
                negative = sampled unlabeled = 0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ============================================================================
# Binary Focal Loss (for grouping task)
# ============================================================================

class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification (sigmoid output).

    L = -alpha * (1 - p_t)^gamma * log(p_t)

    For positives: p_t = sigmoid(logit)
    For negatives: p_t = 1 - sigmoid(logit)
    """

    def __init__(
        self,
        alpha: float = 0.25,    # weight for positive class
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,   # [N] or [N, 1]
        targets: torch.Tensor,  # [N]  float {0.0, 1.0}
    ) -> torch.Tensor:
        logits  = logits.view(-1)
        targets = targets.view(-1).float()

        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal_weight = (1 - p_t) ** self.gamma
        loss = alpha_t * focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# Multi-Task Loss
# ============================================================================

class GNNMultiTaskLoss(nn.Module):
    """
    End-to-end multi-task training loss for GNN Stage2 classifier.

    Tasks:
        1. Behavior classification (3-class Focal Loss)
           Input: behavior_logits [P, 3], behavior_labels [P]
           Pairs: all labeled interaction pairs

        2. Group detection (Binary Focal Loss)
           Input: group_logits [P+Q, 1], group_labels [P+Q]
           Pairs: all labeled (positive=1) + sampled unlabeled (negative=0)

    Args:
        lambda_behavior: weight for behavior loss
        lambda_group:    weight for grouping loss
        class_weights:   per-class weights for 3-class behavior loss
        focal_gamma:     gamma for both focal losses
        group_alpha:     alpha (pos weight) for binary focal loss
    """

    def __init__(
        self,
        lambda_behavior: float = 0.8,
        lambda_group:    float = 0.2,
        class_weights:   Optional[Dict] = None,
        focal_gamma:     float = 2.0,
        group_alpha:     float = 0.25,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_behavior = lambda_behavior
        self.lambda_group    = lambda_group
        self.label_smoothing = label_smoothing

        # ---- N-class Focal Loss (behavior) ----
        if class_weights is None:
            class_weights = {0: 1.0, 1: 1.4, 2: 6.1}
        n_cls = max(class_weights.keys()) + 1
        self.n_classes = n_cls
        w = torch.tensor(
            [class_weights.get(i, 1.0) for i in range(n_cls)],
            dtype=torch.float32
        )
        self.register_buffer('alpha_weights', w)
        self.focal_gamma = focal_gamma

        # ---- Binary Focal Loss (grouping) ----
        self.group_loss_fn = BinaryFocalLoss(
            alpha=group_alpha, gamma=focal_gamma)

    def forward(
        self,
        behavior_logits: torch.Tensor,   # [P, 3]
        behavior_labels: torch.Tensor,   # [P]   int64
        group_logits:    torch.Tensor,   # [P+Q, 1]
        group_labels:    torch.Tensor,   # [P+Q]  float32 {0,1}
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            total_loss: scalar
            loss_dict:  dict with individual loss values and metrics
        """
        # ---- Behavior loss (3-class focal + optional label smoothing) ----
        log_pt    = F.log_softmax(behavior_logits, dim=1)  # [P, 3]
        batch_idx = torch.arange(behavior_logits.size(0),
                                 device=behavior_logits.device)
        log_pt_y  = log_pt[batch_idx, behavior_labels]     # [P]  hard-label log-prob
        pt        = log_pt_y.exp()                          # [P]  for focal weight
        alpha_t   = self.alpha_weights[behavior_labels]    # [P]
        focal_w   = alpha_t * (1 - pt) ** self.focal_gamma  # [P]
        if self.label_smoothing > 0.0:
            # Soft targets: (1-ε)·one_hot + ε/C
            C = behavior_logits.size(1)
            smooth_eps = self.label_smoothing
            smoothed = (1.0 - smooth_eps) * F.one_hot(
                behavior_labels, C).float() + smooth_eps / C        # [P, 3]
            ce_per_sample = -(smoothed * log_pt).sum(dim=1)         # [P]
        else:
            ce_per_sample = -log_pt_y                               # [P]
        behavior_loss = (focal_w * ce_per_sample).mean()

        # ---- Grouping loss (binary focal) ----
        group_loss = self.group_loss_fn(group_logits, group_labels)

        # ---- Combined ----
        total_loss = (self.lambda_behavior * behavior_loss
                      + self.lambda_group    * group_loss)

        # ---- Metrics ----
        with torch.no_grad():
            behavior_preds = behavior_logits.argmax(dim=1)
            behavior_acc   = (behavior_preds == behavior_labels).float().mean()

            group_preds = (group_logits.squeeze() > 0).float()
            group_acc   = (group_preds == group_labels).float().mean()

            # Per-class behavior accuracy
            per_class_acc = {}
            for c in range(self.n_classes):
                mask = behavior_labels == c
                if mask.sum() > 0:
                    per_class_acc[c] = (
                        behavior_preds[mask] == c).float().mean().item()
                else:
                    per_class_acc[c] = float('nan')

            mpca = sum(v for v in per_class_acc.values()
                       if v == v) / max(1, sum(
                           1 for v in per_class_acc.values() if v == v))

        loss_dict = {
            'total_loss':    total_loss.item(),
            'behavior_loss': behavior_loss.item(),
            'group_loss':    group_loss.item(),
            'behavior_acc':  behavior_acc.item(),
            'group_acc':     group_acc.item(),
            'mpca':          mpca,
            'per_class_acc': per_class_acc,
        }

        return total_loss, loss_dict


# ============================================================================
# Factory
# ============================================================================

def create_multitask_loss(config) -> GNNMultiTaskLoss:
    return GNNMultiTaskLoss(
        lambda_behavior=config.lambda_behavior,
        lambda_group=config.lambda_group,
        class_weights=config.class_weights,
        focal_gamma=config.focal_gamma,
        label_smoothing=getattr(config, 'label_smoothing', 0.0),
    )


if __name__ == '__main__':
    print("Testing GNNMultiTaskLoss...")
    criterion = GNNMultiTaskLoss(lambda_behavior=0.8, lambda_group=0.2)

    P, Q = 20, 40
    behavior_logits = torch.randn(P, 3)
    behavior_labels = torch.randint(0, 3, (P,))
    group_logits    = torch.randn(P + Q, 1)
    group_labels    = torch.cat([torch.ones(P), torch.zeros(Q)])

    loss, d = criterion(behavior_logits, behavior_labels,
                        group_logits, group_labels)
    print(f"total_loss:    {d['total_loss']:.4f}")
    print(f"behavior_loss: {d['behavior_loss']:.4f}")
    print(f"group_loss:    {d['group_loss']:.4f}")
    print(f"behavior_acc:  {d['behavior_acc']:.3f}")
    print(f"group_acc:     {d['group_acc']:.3f}")
    print(f"MPCA:          {d['mpca']:.3f}")
    loss.backward()
    print("Backward pass OK.")
    print("All tests passed!")
