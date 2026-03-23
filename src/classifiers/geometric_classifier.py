import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

def _importance_sort_key(item):
    """Helper function for sorting importance items (pickle-friendly)"""
    return item[1]


class AdaptiveGeometricClassifier(nn.Module):
    """
    Adaptive geometric classifier with learnable feature weights
    Supports variable input dimensions (7D, 10D, or 5D features)
    """

    def __init__(self, num_geometric_features=None, input_dim=None, hidden_dims=[32, 16], dropout=0.1):
        super().__init__()

        # Auto-detect input dimension (prioritize input_dim, then num_geometric_features, default to 7)
        if input_dim is not None:
            self.num_features = input_dim
        elif num_geometric_features is not None:
            self.num_features = num_geometric_features
        else:
            self.num_features = 7

        # Feature names based on dimension
        if self.num_features == 7:
            self.feature_names = [
                'horizontal_gap_norm', 'height_ratio', 'ground_distance',
                'v_overlap', 'area_ratio', 'center_dist_norm', 'vertical_gap'
            ]
        elif self.num_features == 10:
            self.feature_names = [
                'distance/height', 'distance/width', 'flow_mean/area',
                'flow_std/area', 'vertical_dominance', 'aspect_ratio',
                'relative_height', 'relative_bottom', 'direction_consistency',
                'interaction_synchrony'
            ]
        elif self.num_features == 5:
            self.feature_names = [f'f{i}' for i in range(5)]
        else:
            self.feature_names = [f'f{i}' for i in range(self.num_features)]
        
        # Learnable feature weights and transforms
        self.feature_weights = nn.Parameter(torch.ones(self.num_features))
        self.feature_scales = nn.Parameter(torch.ones(self.num_features))
        self.feature_biases = nn.Parameter(torch.zeros(self.num_features))

        # Classification network
        layers = []
        in_dim = self.num_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 2))  # Binary classification
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, geometric_features):
        """
        Forward pass with adaptive feature weighting

        Args:
            geometric_features: [batch_size, num_features] or [num_features]
                where num_features can be 7 (legacy), 10 (full), or 5 (subset)

        Returns:
            [batch_size, 2] or [2] classification logits
        """
        if geometric_features.dim() == 1:
            geometric_features = geometric_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply learnable feature transformations
        normalized_features = (geometric_features + self.feature_biases) * self.feature_scales
        weighted_features = normalized_features * self.feature_weights
        
        # Classification
        output = self.classifier(weighted_features)
        
        return output.squeeze(0) if squeeze_output else output
    
    def get_feature_importance(self):
        """Get learned feature importance ranking"""
        importance = torch.abs(self.feature_weights).detach().cpu().numpy()
        importance_dict = dict(zip(self.feature_names, importance))
        return sorted(importance_dict.items(), key=_importance_sort_key, reverse=True)


class CausalTemporalStage1(nn.Module):
    """
    Causal temporal geometric classifier for Stage 1 interaction detection
    Uses historical information in a causal manner (past frames only)
    Supports variable input dimensions (7D, 10D, or 5D features)
    """

    def __init__(self, history_length=5, geometric_dim=None, input_dim=None,
                 hidden_size=16, num_layers=1, dropout=0.1):
        super().__init__()

        # Auto-detect input dimension (prioritize input_dim, then geometric_dim, default to 7)
        if input_dim is not None:
            self.geometric_dim = input_dim
        elif geometric_dim is not None:
            self.geometric_dim = geometric_dim
        else:
            self.geometric_dim = 7

        self.history_length = history_length
        self.hidden_size = hidden_size

        # Temporal encoder for historical features
        self.temporal_encoder = nn.LSTM(
            input_size=self.geometric_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Current frame processor
        self.current_processor = nn.Sequential(
            nn.Linear(self.geometric_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Motion pattern analyzer
        self.motion_analyzer = nn.Sequential(
            nn.Linear(4, 8),  # 4 motion features -> 8
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Scene context processor (simplified)
        self.context_processor = nn.Sequential(
            nn.Linear(1, 4),  # 1 crowd level -> 4
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final fusion classifier
        fusion_input_dim = hidden_size + hidden_size + 8 + 4  # temporal + current + motion + context
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Binary classification
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, current_geometric, history_geometric=None,
                motion_features=None, scene_context=None):
        """
        Forward pass with temporal, motion, and context information

        Args:
            current_geometric: [batch_size, geometric_dim] current frame geometric features
                where geometric_dim can be 7 (legacy), 10 (full), or 5 (subset)
            history_geometric: [batch_size, history_length, geometric_dim] historical features
            motion_features: [batch_size, 4] motion pattern features
            scene_context: [batch_size, 3] scene context features

        Returns:
            [batch_size, 2] classification logits
        """
        batch_size = current_geometric.size(0)
        
        # Process current frame
        current_features = self.current_processor(current_geometric)
        
        # Process historical information
        if history_geometric is not None and history_geometric.size(1) > 0:
            # Check if we have actual historical data (not just zeros)
            has_real_history = torch.sum(torch.abs(history_geometric)) > 1e-6
            
            if has_real_history:
                lstm_out, (hidden, _) = self.temporal_encoder(history_geometric)
                temporal_features = lstm_out[:, -1, :]  # Last time step output
            else:
                temporal_features = torch.zeros(batch_size, self.hidden_size).to(current_geometric.device)
        else:
            temporal_features = torch.zeros(batch_size, self.hidden_size).to(current_geometric.device)
        
        # Process motion features
        if motion_features is not None:
            motion_processed = self.motion_analyzer(motion_features)
        else:
            motion_processed = torch.zeros(batch_size, 8).to(current_geometric.device)
        
        # Process scene context
        if scene_context is not None:
            context_processed = self.context_processor(scene_context)
        else:
            context_processed = torch.zeros(batch_size, 4).to(current_geometric.device)  # Match output dim
        
        # Fuse all information
        fused_features = torch.cat([
            temporal_features,    # Historical patterns
            current_features,     # Current state
            motion_processed,     # Motion trends
            context_processed     # Scene context
        ], dim=1)
        
        # Final classification
        output = self.fusion_classifier(fused_features)
        return output


class ContextAwareGeometricClassifier(nn.Module):
    """
    Context-aware geometric classifier with dynamic feature weighting
    Supports variable input dimensions (7D, 10D, or 5D features)
    """

    def __init__(self, num_geometric_features=None, input_dim=None, hidden_dim=32):
        super().__init__()

        # Auto-detect input dimension (prioritize input_dim, then num_geometric_features, default to 7)
        if input_dim is not None:
            self.num_features = input_dim
        elif num_geometric_features is not None:
            self.num_features = num_geometric_features
        else:
            self.num_features = 7

        # Context encoder for dynamic weights (simplified)
        self.context_encoder = nn.Sequential(
            nn.Linear(1, 8),   # Crowd level: 0-3
            nn.ReLU(),
            nn.Linear(8, self.num_features),
            nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )

        # Feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(self.num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, geometric_features, scene_context=None):
        """
        Forward pass with context-aware feature weighting

        Args:
            geometric_features: [batch_size, num_features]
                where num_features can be 7 (legacy), 10 (full), or 5 (subset)
            scene_context: [batch_size, 3] optional scene context

        Returns:
            [batch_size, 2] classification logits
        """
        if scene_context is not None:
            # Dynamic feature weighting based on scene context
            feature_weights = self.context_encoder(scene_context)
            weighted_features = geometric_features * feature_weights
        else:
            weighted_features = geometric_features
        
        # Process and classify
        processed = self.feature_processor(weighted_features)
        output = self.classifier(processed)
        
        return output


class GeometricStage1Ensemble(nn.Module):
    """
    Ensemble of geometric classifiers for robust Stage 1 detection
    Supports variable input dimensions (7D, 10D, or 5D features)
    """

    def __init__(self, num_models=3, input_dim=7):
        super().__init__()

        # Create ensemble of different architectures with consistent input dimension
        self.models = nn.ModuleList([
            AdaptiveGeometricClassifier(input_dim=input_dim, hidden_dims=[32, 16]),
            AdaptiveGeometricClassifier(input_dim=input_dim, hidden_dims=[64, 32, 16]),
            ContextAwareGeometricClassifier(input_dim=input_dim)
        ][:num_models])

        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, geometric_features, scene_context=None):
        """
        Ensemble forward pass
        """
        outputs = []
        
        for i, model in enumerate(self.models):
            if isinstance(model, ContextAwareGeometricClassifier) and scene_context is not None:
                output = model(geometric_features, scene_context)
            else:
                output = model(geometric_features)
            outputs.append(output)
        
        # Weighted ensemble
        ensemble_output = sum(w * out for w, out in zip(self.ensemble_weights, outputs))
        return ensemble_output


def compute_adaptive_loss(predictions, targets, feature_weights=None, 
                         weight_regularization=0.01, sparsity_regularization=0.01):
    """
    Compute loss with feature weight regularization
    
    Args:
        predictions: [batch_size, 2] logits
        targets: [batch_size] labels
        feature_weights: Optional feature weights for regularization
        weight_regularization: Variance regularization coefficient
        sparsity_regularization: Sparsity regularization coefficient
        
    Returns:
        Total loss
    """
    # Main classification loss
    classification_loss = F.cross_entropy(predictions, targets)
    
    total_loss = classification_loss
    
    if feature_weights is not None:
        # Feature weight variance regularization (prevents one feature dominating)
        weight_var_loss = weight_regularization * torch.var(feature_weights)
        
        # Sparsity regularization (encourages using fewer features)
        sparsity_loss = sparsity_regularization * torch.sum(torch.abs(feature_weights))
        
        total_loss = total_loss + weight_var_loss + sparsity_loss
    
    return total_loss


if __name__ == '__main__':
    # Test adaptive geometric classifier
    print("Testing AdaptiveGeometricClassifier...")
    
    model = AdaptiveGeometricClassifier()
    
    # Test data
    batch_size = 4
    geometric_features = torch.randn(batch_size, 7)
    
    # Forward pass
    output = model(geometric_features)
    print(f"Output shape: {output.shape}")
    
    # Test feature importance
    importance = model.get_feature_importance()
    print("Feature importance:", importance[:3])
    
    # Test temporal classifier
    print("\nTesting CausalTemporalStage1...")
    
    temporal_model = CausalTemporalStage1(history_length=5)
    
    current_features = torch.randn(batch_size, 7)
    history_features = torch.randn(batch_size, 5, 7)
    motion_features = torch.randn(batch_size, 4)
    scene_context = torch.randn(batch_size, 3)
    
    temporal_output = temporal_model(
        current_features, history_features, motion_features, scene_context
    )
    print(f"Temporal output shape: {temporal_output.shape}")
    
    # Test ensemble
    print("\nTesting GeometricStage1Ensemble...")
    
    ensemble = GeometricStage1Ensemble(num_models=3)
    ensemble_output = ensemble(geometric_features, scene_context)
    print(f"Ensemble output shape: {ensemble_output.shape}")
    
    # Test loss computation
    targets = torch.randint(0, 2, (batch_size,))
    loss = compute_adaptive_loss(output, targets, model.feature_weights)
    print(f"Loss: {loss.item()}")
    
    print("All tests passed!")