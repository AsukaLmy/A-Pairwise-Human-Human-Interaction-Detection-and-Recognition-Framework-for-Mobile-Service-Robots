#!/usr/bin/env python3
"""
Mixture of Experts (MoE) Geometric Classifier
Scene-aware three-class behavior classification using 10D geometric features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GeometricMoEClassifier(nn.Module):
    """
    Mixture of Experts for Geometric Feature-based Behavior Classification

    Architecture:
        - Gating Network: 10D geometric features -> scene prediction (num_experts classes)
        - Expert Networks: num_experts independent classifiers for behavior (3 classes)
        - Output: Weighted combination of expert predictions

    Args:
        input_dim: Input feature dimension (default: 10 for geometric features)
        num_experts: Number of expert networks (default: 31 for training scenes)
        num_classes: Number of behavior classes (default: 3 for Walking/Standing/Sitting)
        hidden_dim: Hidden dimension for expert networks (default: 256)
        gate_hidden: Hidden dimension for gating network (default: 128)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(self,
                 input_dim: int = 10,
                 num_experts: int = 31,
                 num_classes: int = 3,
                 hidden_dim: int = 256,
                 gate_hidden: int = 128,
                 dropout: float = 0.3,
                 use_layer_norm: bool = True,
                 layer_norm_eps: float = 1e-5):
        super(GeometricMoEClassifier, self).__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

        # ====================================================================
        # Gating Network: Predicts scene/expert from geometric features
        # ====================================================================
        if use_layer_norm:
            # Pre-LN: Linear → LayerNorm → ReLU → Dropout (reduced)
            self.gate = nn.Sequential(
                nn.Linear(input_dim, gate_hidden),
                nn.LayerNorm(gate_hidden, eps=layer_norm_eps),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),  # Reduced from 0.3

                nn.Linear(gate_hidden, 64),
                nn.LayerNorm(64, eps=layer_norm_eps),
                nn.ReLU(inplace=True),
                nn.Dropout(0.10),  # Reduced from 0.15

                nn.Linear(64, num_experts)
                # Note: No softmax here - will apply in forward for flexibility
            )
        else:
            # Original architecture without LayerNorm
            self.gate = nn.Sequential(
                nn.Linear(input_dim, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Linear(gate_hidden, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),

                nn.Linear(64, num_experts)
            )

        # ====================================================================
        # Expert Networks: Scene-specific behavior classifiers
        # ====================================================================
        if use_layer_norm:
            # Pre-LN: Linear → LayerNorm → ReLU → Dropout (reduced)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.15),  # Reduced from 0.21 (0.3 * 0.7)

                    nn.Linear(hidden_dim, 128),
                    nn.LayerNorm(128, eps=layer_norm_eps),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.15),  # Reduced from 0.21

                    nn.Linear(128, num_classes)
                ) for _ in range(num_experts)
            ])
        else:
            # Original architecture without LayerNorm
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout * 0.7),

                    nn.Linear(hidden_dim, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout * 0.7),

                    nn.Linear(128, num_classes)
                ) for _ in range(num_experts)
            ])

        # Initialize weights
        self._initialize_weights()

        print(f"GeometricMoEClassifier initialized:")
        print(f"  Input dim: {input_dim}D geometric features")
        print(f"  Num experts: {num_experts} (scene-specific)")
        print(f"  Num classes: {num_classes} (Walking/Standing/Sitting)")
        print(f"  Expert hidden dim: {hidden_dim}")
        print(f"  Gate hidden dim: {gate_hidden}")
        print(f"  LayerNorm: {'Enabled' if use_layer_norm else 'Disabled'}")

        # Count parameters
        gate_params = sum(p.numel() for p in self.gate.parameters())
        expert_params = sum(p.numel() for p in self.experts.parameters())
        total_params = gate_params + expert_params

        print(f"  Parameters:")
        print(f"    Gate network: {gate_params:,}")
        print(f"    Expert networks: {expert_params:,} ({expert_params // num_experts:,} per expert)")
        print(f"    Total: {total_params:,}")

    def _initialize_weights(self):
        """Initialize network weights with Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,
                x: torch.Tensor,
                return_gate_weights: bool = False,
                return_expert_outputs: bool = False,
                use_hard_routing: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through MoE network

        Args:
            x: Input geometric features [B, 6]
            return_gate_weights: If True, return gating network softmax weights
            return_expert_outputs: If True, return all expert predictions
            use_hard_routing: If True, use hard routing (select single expert)

        Returns:
            output: Weighted behavior predictions [B, 3]
            gate_weights: Gating weights [B, num_experts] (if return_gate_weights=True)
            expert_outputs: All expert predictions [B, num_experts, 3] (if return_expert_outputs=True)
        """
        batch_size = x.size(0)

        # ====================================================================
        # Gating Network: Compute expert weights
        # ====================================================================
        gate_logits = self.gate(x)  # [B, num_experts]
        gate_weights = F.softmax(gate_logits, dim=1)  # [B, num_experts]

        # Hard routing: select single best expert (for inference)
        if use_hard_routing:
            gate_weights_hard = torch.zeros_like(gate_weights)
            best_expert_idx = torch.argmax(gate_weights, dim=1)
            gate_weights_hard.scatter_(1, best_expert_idx.unsqueeze(1), 1.0)
            gate_weights = gate_weights_hard

        # ====================================================================
        # Expert Networks: Get predictions from all experts
        # ====================================================================
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # [B, num_classes]
            expert_outputs.append(expert_output)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, num_classes]

        # ====================================================================
        # Weighted Combination: Mix expert predictions
        # ====================================================================
        # Compute weighted average: [B, 1, num_experts] x [B, num_experts, num_classes] -> [B, 1, num_classes]
        output = torch.bmm(
            gate_weights.unsqueeze(1),  # [B, 1, num_experts]
            expert_outputs               # [B, num_experts, num_classes]
        ).squeeze(1)                    # [B, num_classes]

        # ====================================================================
        # Return outputs
        # ====================================================================
        returns = [output]

        if return_gate_weights:
            returns.append(gate_weights)

        if return_expert_outputs:
            returns.append(expert_outputs)

        if len(returns) == 1:
            return output
        else:
            return tuple(returns)

    def forward_with_gate_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both behavior predictions and gate logits
        Useful for computing dual loss (behavior + gate)

        Args:
            x: Input geometric features [B, 6]

        Returns:
            behavior_logits: Behavior predictions [B, 3]
            gate_logits: Gate network logits (before softmax) [B, num_experts]
        """
        gate_logits = self.gate(x)  # [B, num_experts]
        gate_weights = F.softmax(gate_logits, dim=1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        behavior_logits = torch.bmm(
            gate_weights.unsqueeze(1),
            expert_outputs
        ).squeeze(1)

        return behavior_logits, gate_logits

    def get_expert_predictions(self, x: torch.Tensor, expert_id: int) -> torch.Tensor:
        """
        Get predictions from a specific expert

        Args:
            x: Input geometric features [B, 6]
            expert_id: Expert index (0 to num_experts-1)

        Returns:
            predictions: Expert predictions [B, 3]
        """
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError(f"Invalid expert_id {expert_id}. Must be in [0, {self.num_experts-1}]")

        return self.experts[expert_id](x)

    def freeze_gate_network(self):
        """Freeze gating network parameters (for Phase 1 training)"""
        for param in self.gate.parameters():
            param.requires_grad = False
        print("✓ Froze gating network")

    def unfreeze_gate_network(self):
        """Unfreeze gating network parameters (for Phase 2 training)"""
        for param in self.gate.parameters():
            param.requires_grad = True
        print("✓ Unfroze gating network")

    def freeze_expert_networks(self):
        """Freeze all expert networks"""
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
        print("✓ Froze all expert networks")

    def unfreeze_expert_networks(self):
        """Unfreeze all expert networks"""
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = True
        print("✓ Unfroze all expert networks")

    def get_gate_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of gate distribution (measure of uncertainty)

        Args:
            x: Input geometric features [B, 6]

        Returns:
            entropy: Gate entropy [B]
        """
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=1)

        # Entropy: -sum(p * log(p))
        entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=1)

        return entropy


if __name__ == '__main__':
    print("Testing GeometricMoEClassifier...\n")

    # Create model
    model = GeometricMoEClassifier(
        input_dim=6,
        num_experts=31,
        num_classes=3,
        hidden_dim=256,
        gate_hidden=128,
        dropout=0.3
    )

    # Test forward pass
    batch_size = 8
    dummy_features = torch.randn(batch_size, 6)

    print(f"\n--- Test 1: Basic forward pass ---")
    output = model(dummy_features)
    print(f"Input shape: {dummy_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (first sample): {output[0]}")

    print(f"\n--- Test 2: Forward with gate weights ---")
    output, gate_weights = model(dummy_features, return_gate_weights=True)
    print(f"Gate weights shape: {gate_weights.shape}")
    print(f"Gate weights (first sample): {gate_weights[0]}")
    print(f"Gate weights sum: {gate_weights[0].sum().item():.4f} (should be ~1.0)")

    print(f"\n--- Test 3: Forward with expert outputs ---")
    output, gate_weights, expert_outputs = model(
        dummy_features,
        return_gate_weights=True,
        return_expert_outputs=True
    )
    print(f"Expert outputs shape: {expert_outputs.shape}")

    print(f"\n--- Test 4: Hard routing ---")
    output_hard = model(dummy_features, use_hard_routing=True, return_gate_weights=True)
    print(f"Hard routing output shape: {output_hard[0].shape}")
    print(f"Selected experts (first 4 samples): {output_hard[1][:4].argmax(dim=1)}")

    print(f"\n--- Test 5: Specific expert prediction ---")
    expert_id = 0
    expert_pred = model.get_expert_predictions(dummy_features, expert_id)
    print(f"Expert {expert_id} prediction shape: {expert_pred.shape}")

    print(f"\n--- Test 6: Gate entropy ---")
    entropy = model.get_gate_entropy(dummy_features)
    print(f"Gate entropy shape: {entropy.shape}")
    print(f"Gate entropy (first 4 samples): {entropy[:4]}")
    print(f"Mean entropy: {entropy.mean().item():.4f}")

    print(f"\n--- Test 7: Freeze/unfreeze ---")
    model.freeze_gate_network()
    model.unfreeze_gate_network()
    model.freeze_expert_networks()
    model.unfreeze_expert_networks()

    print("\n✅ GeometricMoEClassifier test completed!")
