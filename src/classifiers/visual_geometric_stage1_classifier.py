#!/usr/bin/env python3
"""
Visual-Enhanced Geometric Stage1 Classifier
Uses CNN features from Stage2 architecture for Stage1 interaction detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
# Add models path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
models_path = os.path.join(project_root, 'models')
sys.path.append(models_path)

from models.resnet_feature_extractors import ResNetBackbone


class VisualGeometricStage1Classifier(nn.Module):
    """
    Stage1 classifier using both visual features (from Stage2) and geometric features
    """
    
    def __init__(self, backbone_name='resnet18', visual_feature_dim=256, 
                 geometric_feature_dim=7, fusion_strategy='concat',
                 hidden_dims=[128, 64], dropout=0.1, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        self.fusion_strategy = fusion_strategy
        self.visual_feature_dim = visual_feature_dim
        self.geometric_feature_dim = geometric_feature_dim
        
        # Visual feature extractor (reuse Stage2 backbone)
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            feature_dim=visual_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        # Geometric feature processor
        self.geometric_processor = nn.Sequential(
            nn.Linear(geometric_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16)
        )
        
        # Feature fusion
        if fusion_strategy == 'concat':
            # Person A + Person B visual features + geometric features
            fusion_input_dim = visual_feature_dim * 2 + 16
        elif fusion_strategy == 'add':
            # Visual features projected to same dim as geometric, then added
            assert visual_feature_dim == 16, "For add fusion, visual_feature_dim must equal geometric processed dim"
            fusion_input_dim = visual_feature_dim + 16
        elif fusion_strategy == 'attention':
            # Cross-attention between visual and geometric
            self.cross_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
            fusion_input_dim = 64
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Classification head
        layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        
        self.classifier = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in [self.geometric_processor, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, person_A_image, person_B_image, geometric_features):
        """
        Forward pass with visual and geometric features
        
        Args:
            person_A_image: [batch_size, 3, H, W] Person A cropped image
            person_B_image: [batch_size, 3, H, W] Person B cropped image  
            geometric_features: [batch_size, 7] Geometric features
            
        Returns:
            [batch_size, 2] Classification logits
        """
        batch_size = person_A_image.size(0)
        
        # Extract visual features
        person_A_visual = self.backbone(person_A_image)  # [batch_size, visual_dim]
        person_B_visual = self.backbone(person_B_image)  # [batch_size, visual_dim]
        
        # Process geometric features
        geometric_processed = self.geometric_processor(geometric_features)  # [batch_size, 16]
        
        # Feature fusion
        if self.fusion_strategy == 'concat':
            # Concatenate all features
            fused_features = torch.cat([person_A_visual, person_B_visual, geometric_processed], dim=1)
            
        elif self.fusion_strategy == 'add':
            # Project visual features and add with geometric
            visual_combined = (person_A_visual + person_B_visual) / 2  # Average of two persons
            fused_features = torch.cat([visual_combined + geometric_processed, geometric_processed], dim=1)
            
        elif self.fusion_strategy == 'attention':
            # Cross-attention fusion
            visual_combined = torch.stack([person_A_visual, person_B_visual], dim=1)  # [batch, 2, visual_dim]
            geometric_expanded = geometric_processed.unsqueeze(1)  # [batch, 1, 16]
            
            # Project to same dimension for attention
            visual_proj = F.linear(visual_combined, torch.randn(64, self.visual_feature_dim).to(visual_combined.device))
            geometric_proj = F.linear(geometric_expanded, torch.randn(64, 16).to(geometric_expanded.device))
            
            # Cross attention: query=visual, key=value=geometric
            attended_features, _ = self.cross_attention(
                visual_proj.transpose(0, 1),  # [2, batch, 64]
                geometric_proj.transpose(0, 1),  # [1, batch, 64] 
                geometric_proj.transpose(0, 1)
            )
            
            fused_features = attended_features.transpose(0, 1).mean(dim=1)  # [batch, 64]
        
        # Classification
        output = self.classifier(fused_features)
        return output
    
    def get_feature_info(self):
        """Get feature information"""
        return {
            'backbone': self.backbone.backbone_name,
            'visual_feature_dim': self.visual_feature_dim,
            'geometric_feature_dim': self.geometric_feature_dim,
            'fusion_strategy': self.fusion_strategy,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class VisualOnlyStage1Classifier(nn.Module):
    """
    Stage1 classifier using only visual features (no geometric features)
    For comparison baseline
    """
    
    def __init__(self, backbone_name='resnet18', visual_feature_dim=256,
                 hidden_dims=[128, 64], dropout=0.1, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        self.visual_feature_dim = visual_feature_dim
        
        # Visual feature extractor
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            feature_dim=visual_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        # Classification head (Person A + Person B features)
        fusion_input_dim = visual_feature_dim * 2
        
        layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        
        self.classifier = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, person_A_image, person_B_image, geometric_features=None):
        """
        Forward pass with only visual features
        
        Args:
            person_A_image: [batch_size, 3, H, W] Person A cropped image
            person_B_image: [batch_size, 3, H, W] Person B cropped image
            geometric_features: Ignored (for compatibility)
            
        Returns:
            [batch_size, 2] Classification logits
        """
        # Extract visual features
        person_A_visual = self.backbone(person_A_image)  # [batch_size, visual_dim]
        person_B_visual = self.backbone(person_B_image)  # [batch_size, visual_dim]
        
        # Concatenate person features
        fused_features = torch.cat([person_A_visual, person_B_visual], dim=1)
        
        # Classification
        output = self.classifier(fused_features)
        return output


class HybridStage1Ensemble(nn.Module):
    """
    Ensemble combining geometric-only, visual-only, and visual+geometric models
    """
    
    def __init__(self, backbone_name='resnet18', visual_feature_dim=256):
        super().__init__()
        
        # Import geometric-only model
        from .geometric_classifier import AdaptiveGeometricClassifier
        
        self.geometric_only = AdaptiveGeometricClassifier()
        self.visual_only = VisualOnlyStage1Classifier(
            backbone_name=backbone_name,
            visual_feature_dim=visual_feature_dim
        )
        self.visual_geometric = VisualGeometricStage1Classifier(
            backbone_name=backbone_name,
            visual_feature_dim=visual_feature_dim,
            fusion_strategy='concat'
        )
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, person_A_image, person_B_image, geometric_features):
        """Ensemble forward pass"""
        # Get outputs from all models
        geo_output = self.geometric_only(geometric_features)
        vis_output = self.visual_only(person_A_image, person_B_image)
        hybrid_output = self.visual_geometric(person_A_image, person_B_image, geometric_features)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_output = (weights[0] * geo_output + 
                          weights[1] * vis_output + 
                          weights[2] * hybrid_output)
        
        return ensemble_output


if __name__ == '__main__':
    # Test visual-geometric classifier
    print("Testing VisualGeometricStage1Classifier...")
    
    batch_size = 4
    model = VisualGeometricStage1Classifier(
        backbone_name='resnet18',
        visual_feature_dim=256,
        fusion_strategy='concat'
    )
    
    # Test data
    person_A_img = torch.randn(batch_size, 3, 224, 224)
    person_B_img = torch.randn(batch_size, 3, 224, 224)
    geometric_feats = torch.randn(batch_size, 7)
    
    output = model(person_A_img, person_B_img, geometric_feats)
    print(f"Output shape: {output.shape}")
    
    # Test visual-only classifier
    print("\nTesting VisualOnlyStage1Classifier...")
    visual_model = VisualOnlyStage1Classifier()
    visual_output = visual_model(person_A_img, person_B_img)
    print(f"Visual-only output shape: {visual_output.shape}")
    
    # Test ensemble
    print("\nTesting HybridStage1Ensemble...")
    ensemble = HybridStage1Ensemble()
    ensemble_output = ensemble(person_A_img, person_B_img, geometric_feats)
    print(f"Ensemble output shape: {ensemble_output.shape}")
    
    print("All tests passed!")