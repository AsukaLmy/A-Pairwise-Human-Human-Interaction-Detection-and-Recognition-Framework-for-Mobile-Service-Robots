#!/usr/bin/env python3
"""
Loss Functions for MoE Geometric Classifier
Combined loss for behavior classification and scene routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    https://arxiv.org/abs/1708.02002

    Args:
        gamma: Focusing parameter for modulating loss (default: 2.0)
               Higher gamma increases focus on hard examples
        alpha: Optional class weights [num_classes] (default: None)
        reduction: Specifies the reduction to apply to the output (default: 'mean')
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss

        Args:
            inputs: Predicted logits [B, num_classes]
            targets: Ground truth labels [B]

        Returns:
            loss: Scalar focal loss
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)  # [B, num_classes]

        # Get probabilities of correct classes
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [B]
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]

        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Focal loss
        focal_loss = focal_weight * ce_loss

        # Apply alpha if specified
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def moe_loss(behavior_logits: torch.Tensor,
             behavior_labels: torch.Tensor,
             gate_logits: torch.Tensor,
             scene_labels: torch.Tensor,
             alpha: float = 0.7,
             beta: float = 0.3,
             focal_gamma: float = 2.0,
             use_focal_loss: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Combined MoE loss: behavior classification + gate classification

    Args:
        behavior_logits: Predicted behavior logits [B, 3]
        behavior_labels: True behavior labels [B] (0=Walking, 1=Standing, 2=Sitting)
        gate_logits: Gate network logits [B, num_experts]
        scene_labels: True scene labels [B] (0 to num_experts-1)
        alpha: Weight for behavior loss (default: 0.7)
        beta: Weight for gate loss (default: 0.3)
        focal_gamma: Gamma parameter for focal loss (default: 2.0)
        use_focal_loss: Whether to use focal loss for behavior classification (default: True)

    Returns:
        total_loss: Combined loss
        behavior_loss: Behavior classification loss
        gate_loss: Gate classification loss
        metrics: Dictionary with accuracy metrics
    """
    device = behavior_logits.device

    # ========================================================================
    # Behavior Classification Loss (Main Task)
    # ========================================================================
    if use_focal_loss:
        focal_loss_fn = FocalLoss(gamma=focal_gamma, alpha=None, reduction='mean')
        behavior_loss = focal_loss_fn(behavior_logits, behavior_labels)
    else:
        behavior_loss = F.cross_entropy(behavior_logits, behavior_labels)

    # ========================================================================
    # Gate Classification Loss (Auxiliary Task)
    # ========================================================================
    gate_loss = F.cross_entropy(gate_logits, scene_labels)

    # ========================================================================
    # Combined Loss
    # ========================================================================
    total_loss = alpha * behavior_loss + beta * gate_loss

    # ========================================================================
    # Compute Metrics
    # ========================================================================
    with torch.no_grad():
        # Behavior accuracy
        behavior_preds = torch.argmax(behavior_logits, dim=1)
        behavior_acc = (behavior_preds == behavior_labels).float().mean()

        # Gate accuracy
        gate_preds = torch.argmax(gate_logits, dim=1)
        gate_acc = (gate_preds == scene_labels).float().mean()

        # Per-class behavior accuracy
        walking_mask = (behavior_labels == 0)
        standing_mask = (behavior_labels == 1)
        sitting_mask = (behavior_labels == 2)

        walking_acc = (behavior_preds[walking_mask] == 0).float().mean() if walking_mask.sum() > 0 else torch.tensor(0.0)
        standing_acc = (behavior_preds[standing_mask] == 1).float().mean() if standing_mask.sum() > 0 else torch.tensor(0.0)
        sitting_acc = (behavior_preds[sitting_mask] == 2).float().mean() if sitting_mask.sum() > 0 else torch.tensor(0.0)

        metrics = {
            'behavior_acc': behavior_acc.item(),
            'gate_acc': gate_acc.item(),
            'walking_acc': walking_acc.item(),
            'standing_acc': standing_acc.item(),
            'sitting_acc': sitting_acc.item(),
            'n_walking': walking_mask.sum().item(),
            'n_standing': standing_mask.sum().item(),
            'n_sitting': sitting_mask.sum().item()
        }

    return total_loss, behavior_loss, gate_loss, metrics


def behavior_only_loss(behavior_logits: torch.Tensor,
                       behavior_labels: torch.Tensor,
                       focal_gamma: float = 2.0,
                       use_focal_loss: bool = True) -> Tuple[torch.Tensor, Dict]:
    """
    Behavior classification loss only (for Phase 1 training or inference)

    Args:
        behavior_logits: Predicted behavior logits [B, 3]
        behavior_labels: True behavior labels [B]
        focal_gamma: Gamma parameter for focal loss (default: 2.0)
        use_focal_loss: Whether to use focal loss (default: True)

    Returns:
        loss: Behavior classification loss
        metrics: Dictionary with accuracy metrics
    """
    device = behavior_logits.device

    if use_focal_loss:
        focal_loss_fn = FocalLoss(gamma=focal_gamma, alpha=None, reduction='mean')
        loss = focal_loss_fn(behavior_logits, behavior_labels)
    else:
        loss = F.cross_entropy(behavior_logits, behavior_labels)

    with torch.no_grad():
        behavior_preds = torch.argmax(behavior_logits, dim=1)
        behavior_acc = (behavior_preds == behavior_labels).float().mean()

        # Per-class accuracy
        walking_mask = (behavior_labels == 0)
        standing_mask = (behavior_labels == 1)
        sitting_mask = (behavior_labels == 2)

        walking_acc = (behavior_preds[walking_mask] == 0).float().mean() if walking_mask.sum() > 0 else torch.tensor(0.0)
        standing_acc = (behavior_preds[standing_mask] == 1).float().mean() if standing_mask.sum() > 0 else torch.tensor(0.0)
        sitting_acc = (behavior_preds[sitting_mask] == 2).float().mean() if sitting_mask.sum() > 0 else torch.tensor(0.0)

        metrics = {
            'behavior_acc': behavior_acc.item(),
            'walking_acc': walking_acc.item(),
            'standing_acc': standing_acc.item(),
            'sitting_acc': sitting_acc.item(),
            'n_walking': walking_mask.sum().item(),
            'n_standing': standing_mask.sum().item(),
            'n_sitting': sitting_mask.sum().item()
        }

    return loss, metrics


def compute_load_balancing_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute load balancing loss to encourage uniform expert usage

    Args:
        gate_weights: Gate softmax weights [B, num_experts]

    Returns:
        load_loss: Load balancing loss (coefficient of variation)
    """
    # Average gate weight per expert across batch
    mean_weights = gate_weights.mean(dim=0)  # [num_experts]

    # Coefficient of variation (std / mean)
    std_weights = gate_weights.std(dim=0)
    cv = (std_weights / (mean_weights + 1e-8)).mean()

    return cv


def compute_gate_confidence(gate_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute confidence of gate predictions (max weight)

    Args:
        gate_weights: Gate softmax weights [B, num_experts]

    Returns:
        confidence: Max gate weight per sample [B]
    """
    confidence, _ = torch.max(gate_weights, dim=1)
    return confidence


def compute_gate_entropy(gate_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of gate distribution (uncertainty measure)

    Args:
        gate_weights: Gate softmax weights [B, num_experts]

    Returns:
        entropy: Gate entropy per sample [B]
    """
    entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=1)
    return entropy


def moe_loss_with_load_balancing(behavior_logits: torch.Tensor,
                                  behavior_labels: torch.Tensor,
                                  gate_logits: torch.Tensor,
                                  scene_labels: torch.Tensor,
                                  gate_weights: torch.Tensor,
                                  alpha: float = 0.7,
                                  beta: float = 0.3,
                                  gamma: float = 0.01,
                                  class_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    MoE loss with load balancing term

    Args:
        behavior_logits: Predicted behavior logits [B, 3]
        behavior_labels: True behavior labels [B]
        gate_logits: Gate network logits [B, num_experts]
        scene_labels: True scene labels [B]
        gate_weights: Gate softmax weights [B, num_experts]
        alpha: Weight for behavior loss (default: 0.7)
        beta: Weight for gate loss (default: 0.3)
        gamma: Weight for load balancing loss (default: 0.01)
        class_weights: Optional class weights for behavior loss [3]

    Returns:
        total_loss: Combined loss
        behavior_loss: Behavior classification loss
        gate_loss: Gate classification loss
        load_loss: Load balancing loss
        metrics: Dictionary with metrics
    """
    # Standard MoE loss
    total_loss_base, behavior_loss, gate_loss, metrics = moe_loss(
        behavior_logits, behavior_labels,
        gate_logits, scene_labels,
        alpha, beta, class_weights
    )

    # Load balancing loss
    load_loss = compute_load_balancing_loss(gate_weights)

    # Combined loss
    total_loss = total_loss_base + gamma * load_loss

    # Add load balancing metrics
    with torch.no_grad():
        mean_weights = gate_weights.mean(dim=0)
        max_weight = mean_weights.max().item()
        min_weight = mean_weights.min().item()

        metrics['load_loss'] = load_loss.item()
        metrics['max_expert_weight'] = max_weight
        metrics['min_expert_weight'] = min_weight
        metrics['weight_ratio'] = max_weight / (min_weight + 1e-8)

    return total_loss, behavior_loss, gate_loss, load_loss, metrics


if __name__ == '__main__':
    print("Testing MoE Loss Functions...\n")

    # Create dummy data
    batch_size = 16
    num_experts = 31
    num_classes = 3

    behavior_logits = torch.randn(batch_size, num_classes)
    behavior_labels = torch.randint(0, num_classes, (batch_size,))
    gate_logits = torch.randn(batch_size, num_experts)
    scene_labels = torch.randint(0, num_experts, (batch_size,))
    gate_weights = F.softmax(gate_logits, dim=1)

    print("--- Test 1: Standard MoE Loss ---")
    total_loss, behavior_loss, gate_loss, metrics = moe_loss(
        behavior_logits, behavior_labels,
        gate_logits, scene_labels,
        alpha=0.7, beta=0.3
    )

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Behavior loss: {behavior_loss.item():.4f}")
    print(f"Gate loss: {gate_loss.item():.4f}")
    print(f"Metrics: {metrics}")

    print("\n--- Test 2: Behavior Only Loss ---")
    loss, metrics = behavior_only_loss(behavior_logits, behavior_labels)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    print("\n--- Test 3: Load Balancing Loss ---")
    load_loss = compute_load_balancing_loss(gate_weights)
    print(f"Load balancing loss: {load_loss.item():.4f}")

    print("\n--- Test 4: Gate Confidence ---")
    confidence = compute_gate_confidence(gate_weights)
    print(f"Confidence shape: {confidence.shape}")
    print(f"Mean confidence: {confidence.mean().item():.4f}")

    print("\n--- Test 5: Gate Entropy ---")
    entropy = compute_gate_entropy(gate_weights)
    print(f"Entropy shape: {entropy.shape}")
    print(f"Mean entropy: {entropy.mean().item():.4f}")

    print("\n--- Test 6: MoE Loss with Load Balancing ---")
    total_loss, behavior_loss, gate_loss, load_loss, metrics = moe_loss_with_load_balancing(
        behavior_logits, behavior_labels,
        gate_logits, scene_labels,
        gate_weights,
        alpha=0.7, beta=0.3, gamma=0.01
    )

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Behavior loss: {behavior_loss.item():.4f}")
    print(f"Gate loss: {gate_loss.item():.4f}")
    print(f"Load balancing loss: {load_loss.item():.4f}")
    print(f"Metrics: {metrics}")

    print("\n--- Test 7: With Class Weights ---")
    class_weights = torch.tensor([1.0, 1.5, 2.0])  # Higher weight for sitting
    total_loss, behavior_loss, gate_loss, metrics = moe_loss(
        behavior_logits, behavior_labels,
        gate_logits, scene_labels,
        alpha=0.7, beta=0.3,
        class_weights=class_weights
    )

    print(f"Total loss (with weights): {total_loss.item():.4f}")
    print(f"Behavior loss (with weights): {behavior_loss.item():.4f}")

    print("\n✅ MoE Loss Functions test completed!")
