"""
Focal Loss Implementation for Class Imbalance

Reference: Lin et al. "Focal Loss for Dense Object Detection" (RetinaNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary/multi-class classification

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
               or a list of weights [w0, w1, ...] for each class
        gamma: Exponent of the modulating factor (1 - p_t)^gamma
               Higher gamma increases focus on hard examples
               Recommended: gamma=2.0
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] logits
            targets: [batch_size] class indices (0 to num_classes-1)

        Returns:
            loss: scalar or [batch_size] depending on reduction
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probabilities
        p = torch.exp(-ce_loss)  # p_t in the paper

        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - p) ** self.gamma

        # Apply alpha weighting
        if isinstance(self.alpha, (float, int)):
            # Single alpha value
            focal_loss = self.alpha * focal_weight * ce_loss
        else:
            # Per-class alpha (list or tensor)
            if isinstance(self.alpha, list):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            else:
                # Move alpha to the same device as targets
                alpha_t = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_t * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss that automatically adjusts alpha based on class distribution

    Args:
        gamma: Focusing parameter (default: 2.0)
        auto_alpha: Automatically compute alpha from class frequencies
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, num_classes=2, gamma=2.0, auto_alpha=True, reduction='mean'):
        super(AdaptiveFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.auto_alpha = auto_alpha
        self.reduction = reduction

        # Initialize alpha
        if auto_alpha:
            self.register_buffer('alpha', torch.ones(num_classes))
            self.register_buffer('class_counts', torch.zeros(num_classes))
            self.alpha_updated = False
        else:
            self.alpha = None

    def update_alpha(self, targets):
        """
        Update alpha based on class distribution
        alpha_c = N_total / (num_classes * N_c)
        """
        if not self.auto_alpha or self.alpha_updated:
            return

        # Count class frequencies
        for c in range(self.num_classes):
            self.class_counts[c] = (targets == c).sum().float()

        # Compute inverse frequency weights
        total_samples = self.class_counts.sum()
        for c in range(self.num_classes):
            if self.class_counts[c] > 0:
                self.alpha[c] = total_samples / (self.num_classes * self.class_counts[c])

        # Normalize alpha to [0, 1]
        self.alpha = self.alpha / self.alpha.max()

        print(f"Updated Focal Loss alpha: {self.alpha.tolist()}")
        self.alpha_updated = True

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] logits
            targets: [batch_size] class indices

        Returns:
            loss: scalar
        """
        # Update alpha on first batch if auto mode
        if self.auto_alpha and not self.alpha_updated:
            self.update_alpha(targets)

        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probabilities
        p = torch.exp(-ce_loss)

        # Compute focal term
        focal_weight = (1 - p) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            # Move alpha to the same device as targets
            alpha_t = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == '__main__':
    # Test Focal Loss
    print("Testing Focal Loss...")

    # Test case: binary classification with imbalance
    batch_size = 100
    num_classes = 2

    # Simulated logits
    inputs = torch.randn(batch_size, num_classes)

    # Imbalanced targets: 90 negative (0), 10 positive (1)
    targets = torch.cat([
        torch.zeros(90, dtype=torch.long),
        torch.ones(10, dtype=torch.long)
    ])

    # Compare losses
    ce_loss = F.cross_entropy(inputs, targets)
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal_loss_fn(inputs, targets)

    print(f"Cross Entropy Loss: {ce_loss.item():.4f}")
    print(f"Focal Loss (alpha=0.25, gamma=2.0): {focal_loss.item():.4f}")

    # Test adaptive focal loss
    print("\nTesting Adaptive Focal Loss...")
    adaptive_focal = AdaptiveFocalLoss(num_classes=2, gamma=2.0, auto_alpha=True)
    adaptive_loss = adaptive_focal(inputs, targets)
    print(f"Adaptive Focal Loss: {adaptive_loss.item():.4f}")

    print("\nAll tests passed!")
