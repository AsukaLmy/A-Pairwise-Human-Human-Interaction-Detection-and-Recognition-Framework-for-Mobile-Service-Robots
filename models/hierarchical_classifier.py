#!/usr/bin/env python3
"""
Hierarchical Stage2 Classifier
Two-layer classification: geometric features -> sitting detection, backbone -> standing/walking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class GeometricBinaryClassifier(nn.Module):
    """
    Geometric feature MLP for binary classification: sitting vs not-sitting

    Architecture:
        Input (6D geometric features) -> MLP -> Output (2 classes)
    """

    def __init__(self, input_dim=6, hidden_dim=64, dropout=0.3, use_layernorm=True):
        """
        Args:
            input_dim: Dimension of geometric features (default: 6)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            use_layernorm: Whether to use LayerNorm (default: True)
        """
        super(GeometricBinaryClassifier, self).__init__()

        if use_layernorm:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Linear(hidden_dim, 2)  # Binary: sitting (0) vs not-sitting (1)
            )
        else:
            # Original architecture without LayerNorm (for backward compatibility)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Linear(hidden_dim, 2)  # Binary: sitting (0) vs not-sitting (1)
            )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: [B, 6] geometric features

        Returns:
            logits: [B, 2] class logits
        """
        return self.mlp(x)


class HierarchicalStage2Classifier(nn.Module):
    """
    Hierarchical classifier for Stage2 behavior classification

    Architecture:
        Layer 1: Geometric classifier (sitting vs not-sitting)
        Layer 2: Backbone classifier (standing vs walking, only for non-sitting samples)

    Decision flow:
        if P(sitting) > threshold:
            return "sitting"
        else:
            return backbone.predict(standing vs walking)
    """

    def __init__(self, backbone_classifier, geometric_mlp, threshold=0.5):
        """
        Args:
            backbone_classifier: Existing ResNetRelationStage2Classifier
            geometric_mlp: GeometricBinaryClassifier
            threshold: Probability threshold for sitting classification
        """
        super(HierarchicalStage2Classifier, self).__init__()

        self.geometric_mlp = geometric_mlp
        self.backbone_classifier = backbone_classifier
        self.threshold = threshold

        print(f"HierarchicalStage2Classifier initialized:")
        print(f"  Geometric MLP: {sum(p.numel() for p in geometric_mlp.parameters())} parameters")
        print(f"  Backbone: {sum(p.numel() for p in backbone_classifier.parameters())} parameters")
        print(f"  Sitting threshold: {threshold}")

    def forward(self, person_A_img, person_B_img, person_A_box, person_B_box,
                geometric_features, return_both=True):
        """
        Forward pass through hierarchical classifier

        Args:
            person_A_img: [B, C, H, W] Person A images
            person_B_img: [B, C, H, W] Person B images
            person_A_box: [B, 4] Person A bounding boxes [x, y, w, h]
            person_B_box: [B, 4] Person B bounding boxes [x, y, w, h]
            geometric_features: [B, 6] Geometric features
            return_both: If True, return both geometric and backbone logits

        Returns:
            If return_both=True:
                geometric_logits: [B, 2] Binary classification logits (sitting vs not-sitting)
                backbone_logits: [B, 3] Three-way classification logits (walking, standing, sitting)
                sitting_prob: [B] Probability of sitting (from geometric classifier)
            If return_both=False:
                final_logits: [B, 3] Final three-way classification logits
        """
        batch_size = person_A_img.size(0)

        # Layer 1: Geometric classifier (binary)
        geometric_logits = self.geometric_mlp(geometric_features)  # [B, 2]
        geometric_probs = F.softmax(geometric_logits, dim=1)
        sitting_prob = geometric_probs[:, 0]  # P(sitting)

        # Layer 2: Backbone classifier (three-way)
        # Backbone expects: (person_A_features, person_B_features, spatial_features)
        # Since we pass images [B, 3, H, W], it will extract features internally
        # For spatial_features, we pass empty tensor (not used in hierarchical mode)
        device = person_A_img.device
        spatial_features = torch.zeros(batch_size, 1, device=device)  # Dummy spatial features

        backbone_logits = self.backbone_classifier(
            person_A_img, person_B_img, spatial_features
        )  # [B, 3]

        if return_both:
            return geometric_logits, backbone_logits, sitting_prob
        else:
            # Hierarchical inference: combine predictions
            final_logits = self.hierarchical_inference(
                geometric_logits, backbone_logits, sitting_prob
            )
            return final_logits

    def hierarchical_inference(self, geometric_logits, backbone_logits, sitting_prob):
        """
        Combine geometric and backbone predictions using hierarchical decision

        Args:
            geometric_logits: [B, 2]
            backbone_logits: [B, 3]
            sitting_prob: [B]

        Returns:
            final_logits: [B, 3] Combined logits for walking, standing, sitting
        """
        batch_size = sitting_prob.size(0)
        device = sitting_prob.device

        # Create final logits [B, 3]
        final_logits = torch.zeros(batch_size, 3, device=device)

        # For each sample, decide hierarchically
        is_sitting = (sitting_prob > self.threshold)

        # For sitting samples: set sitting logit to high value
        final_logits[is_sitting, 2] = 10.0  # High confidence for sitting
        final_logits[is_sitting, 0] = -10.0  # Low confidence for walking
        final_logits[is_sitting, 1] = -10.0  # Low confidence for standing

        # For non-sitting samples: use backbone predictions
        final_logits[~is_sitting] = backbone_logits[~is_sitting]

        return final_logits

    def predict(self, person_A_img, person_B_img, person_A_box, person_B_box, geometric_features):
        """
        Predict class labels using hierarchical decision

        Args:
            person_A_img: [B, C, H, W]
            person_B_img: [B, C, H, W]
            person_A_box: [B, 4]
            person_B_box: [B, 4]
            geometric_features: [B, 6]

        Returns:
            predictions: [B] Predicted class labels (0=walking, 1=standing, 2=sitting)
        """
        with torch.no_grad():
            geometric_logits, backbone_logits, sitting_prob = self.forward(
                person_A_img, person_B_img, person_A_box, person_B_box,
                geometric_features, return_both=True
            )

            batch_size = sitting_prob.size(0)
            predictions = torch.zeros(batch_size, dtype=torch.long, device=sitting_prob.device)

            # Hierarchical decision
            is_sitting = (sitting_prob > self.threshold)

            # Sitting samples: label = 2
            predictions[is_sitting] = 2

            # Non-sitting samples: use backbone prediction
            if (~is_sitting).sum() > 0:
                backbone_preds = torch.argmax(backbone_logits[~is_sitting], dim=1)
                predictions[~is_sitting] = backbone_preds

            return predictions

    def set_threshold(self, threshold):
        """Update sitting classification threshold"""
        self.threshold = threshold
        print(f"Updated sitting threshold to: {threshold}")

    def load_sitting_module(self, checkpoint_path, device='cuda'):
        """
        Load pretrained sitting module (geometric MLP) from checkpoint

        Args:
            checkpoint_path: Path to sitting module checkpoint
            device: Device to load model to

        Returns:
            dict: Checkpoint information (epoch, accuracy, etc.)
        """
        import torch
        import os

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Sitting module checkpoint not found: {checkpoint_path}")

        print(f"Loading sitting module from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load geometric MLP state dict
        if 'geometric_mlp_state_dict' in checkpoint:
            self.geometric_mlp.load_state_dict(checkpoint['geometric_mlp_state_dict'])
            print(f"✅ Loaded sitting module weights")
        else:
            raise KeyError("'geometric_mlp_state_dict' not found in checkpoint")

        # Print checkpoint info
        info = {}
        if 'binary_acc' in checkpoint:
            info['binary_acc'] = checkpoint['binary_acc']
            print(f"   Sitting module Binary Accuracy: {checkpoint['binary_acc']:.4f}")
        if 'sitting_acc' in checkpoint:
            info['sitting_acc'] = checkpoint['sitting_acc']
            print(f"   Sitting class Accuracy: {checkpoint['sitting_acc']:.4f}")
        if 'non_sitting_acc' in checkpoint:
            info['non_sitting_acc'] = checkpoint['non_sitting_acc']
            print(f"   Non-sitting class Accuracy: {checkpoint['non_sitting_acc']:.4f}")
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
            print(f"   Trained for {checkpoint['epoch']} epochs")

        return info

    def freeze_sitting_module(self):
        """Freeze all parameters in the sitting module (geometric MLP)"""
        for param in self.geometric_mlp.parameters():
            param.requires_grad = False

        frozen_params = sum(p.numel() for p in self.geometric_mlp.parameters())
        print(f"✅ Frozen sitting module: {frozen_params:,} parameters")

    def unfreeze_sitting_module(self):
        """Unfreeze all parameters in the sitting module (geometric MLP)"""
        for param in self.geometric_mlp.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.geometric_mlp.parameters())
        print(f"✅ Unfrozen sitting module: {trainable_params:,} parameters")


def hierarchical_loss(geometric_logits, backbone_logits, labels,
                     alpha=0.5, beta=0.5, class_weights=None):
    """
    Hierarchical joint loss function

    Args:
        geometric_logits: [B, 2] Geometric classifier output
        backbone_logits: [B, 3] Backbone classifier output
        labels: [B] True labels (0=walking, 1=standing, 2=sitting)
        alpha: Weight for geometric loss
        beta: Weight for backbone loss
        class_weights: Optional class weights for imbalanced data

    Returns:
        total_loss: Combined loss
        loss_geometric: Geometric classifier loss
        loss_backbone: Backbone classifier loss
        metrics: Dict with additional metrics
    """
    device = labels.device

    # Convert 3-class labels to binary for geometric classifier
    # sitting (label=2) -> class 0, not-sitting (label=0,1) -> class 1
    geometric_labels = (labels != 2).long()

    # Geometric loss (binary classification)
    if class_weights is not None:
        # Use class weights for imbalanced data
        geometric_weight = class_weights[:2]  # Only first 2 weights for binary
        loss_geometric = F.cross_entropy(geometric_logits, geometric_labels, weight=geometric_weight)
    else:
        loss_geometric = F.cross_entropy(geometric_logits, geometric_labels)

    # Backbone loss (three-way classification, only for non-sitting samples)
    non_sitting_mask = (labels != 2)

    if non_sitting_mask.sum() > 0:
        # Only compute backbone loss on non-sitting samples
        if class_weights is not None:
            # Use full class weights
            loss_backbone = F.cross_entropy(
                backbone_logits[non_sitting_mask],
                labels[non_sitting_mask],
                weight=class_weights
            )
        else:
            loss_backbone = F.cross_entropy(
                backbone_logits[non_sitting_mask],
                labels[non_sitting_mask]
            )
    else:
        # All samples are sitting, no backbone loss
        loss_backbone = torch.tensor(0.0, device=device)

    # Total loss
    total_loss = alpha * loss_geometric + beta * loss_backbone

    # Additional metrics
    with torch.no_grad():
        geometric_acc = (torch.argmax(geometric_logits, dim=1) == geometric_labels).float().mean()

        if non_sitting_mask.sum() > 0:
            backbone_acc = (torch.argmax(backbone_logits[non_sitting_mask], dim=1) ==
                           labels[non_sitting_mask]).float().mean()
        else:
            backbone_acc = torch.tensor(0.0, device=device)

    metrics = {
        'geometric_acc': geometric_acc.item(),
        'backbone_acc': backbone_acc.item(),
        'n_sitting': (labels == 2).sum().item(),
        'n_non_sitting': non_sitting_mask.sum().item()
    }

    return total_loss, loss_geometric, loss_backbone, metrics


if __name__ == '__main__':
    print("Testing Hierarchical Classifier...")

    # Create dummy geometric classifier
    geometric_mlp = GeometricBinaryClassifier(input_dim=6, hidden_dim=64)

    # Create dummy backbone (simplified for testing)
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(256, 3)  # Dummy feature dim

        def forward(self, img_a, img_b, box_a, box_b):
            batch_size = img_a.size(0)
            # Return random logits
            return torch.randn(batch_size, 3)

    backbone = DummyBackbone()

    # Create hierarchical classifier
    hierarchical_model = HierarchicalStage2Classifier(
        backbone_classifier=backbone,
        geometric_mlp=geometric_mlp,
        threshold=0.5
    )

    # Test forward pass
    batch_size = 4
    dummy_img = torch.randn(batch_size, 3, 224, 224)
    dummy_box = torch.randn(batch_size, 4)
    dummy_geometric = torch.randn(batch_size, 6)

    geometric_logits, backbone_logits, sitting_prob = hierarchical_model(
        dummy_img, dummy_img, dummy_box, dummy_box, dummy_geometric
    )

    print(f"\nForward pass results:")
    print(f"  Geometric logits: {geometric_logits.shape}")
    print(f"  Backbone logits: {backbone_logits.shape}")
    print(f"  Sitting prob: {sitting_prob.shape}")

    # Test loss
    dummy_labels = torch.randint(0, 3, (batch_size,))
    total_loss, loss_geo, loss_back, metrics = hierarchical_loss(
        geometric_logits, backbone_logits, dummy_labels
    )

    print(f"\nLoss computation:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Geometric loss: {loss_geo.item():.4f}")
    print(f"  Backbone loss: {loss_back.item():.4f}")
    print(f"  Metrics: {metrics}")

    print("\n✅ Hierarchical Classifier test completed!")
