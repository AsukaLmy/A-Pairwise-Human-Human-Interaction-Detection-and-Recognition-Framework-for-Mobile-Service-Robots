#!/usr/bin/env python3
"""
Geometric Stage1 Configuration
Configuration for geometric feature-based binary interaction detection
Supports both legacy 7D features and new 10D/5D features
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class GeometricStage1Config:
    """
    Configuration for Geometric Stage1 Binary Interaction Detection

    Feature Modes:
        - 'both' (10D): Full geometric + optical flow + synchrony
        - 'opticalflow_only' (5D): Optical flow subset + synchrony
        - 'bboxposition_only' (5D): Static position features only
    """

    # ========================================================================
    # Feature Configuration
    # ========================================================================
    feature_mode: str = 'both'  # 'both', 'opticalflow_only', 'bboxposition_only'
    use_temporal: bool = False
    use_scene_context: bool = True
    history_length: int = 5

    # ========================================================================
    # Model Architecture
    # ========================================================================
    model_type: str = 'adaptive'  # 'adaptive', 'temporal', 'context_aware', 'ensemble'
    hidden_dims: List[int] = field(default_factory=lambda: [32, 16])
    hidden_size: int = 16
    num_ensemble_models: int = 3
    dropout: float = 0.1

    # ========================================================================
    # Training Configuration
    # ========================================================================
    epochs: int = 17
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adam'  # 'adam', 'sgd'
    scheduler: str = 'step'  # 'step', 'cosine', 'none'
    step_size: int = 20
    gamma: float = 0.5

    # ========================================================================
    # Data Configuration
    # ========================================================================
    data_path: str = ''
    frame_interval: int = 1
    num_workers: int = 0  # Set to 0 for Windows, 4+ for Linux

    # ========================================================================
    # Regularization
    # ========================================================================
    weight_regularization: float = 0.01
    sparsity_regularization: float = 0.01
    max_grad_norm: float = 1.0

    # ========================================================================
    # Early Stopping & Logging
    # ========================================================================
    early_stopping_patience: int = 5
    log_interval: int = 10
    checkpoint_dir: str = './checkpoints'
    save_best_only: bool = True

    # ========================================================================
    # Hardware
    # ========================================================================
    device: str = 'cuda'  # 'cuda' or 'cpu'

    def get_feature_dim(self) -> int:
        """Get input feature dimension based on feature_mode"""
        feature_dim_map = {
            'both': 10,
            'opticalflow_only': 5,
            'bboxposition_only': 5
        }
        if self.feature_mode not in feature_dim_map:
            raise ValueError(
                f"Invalid feature_mode: {self.feature_mode}. "
                f"Must be one of {list(feature_dim_map.keys())}"
            )
        return feature_dim_map[self.feature_mode]

    def validate(self):
        """Validate configuration parameters"""
        # Validate feature mode
        valid_modes = ['both', 'opticalflow_only', 'bboxposition_only']
        if self.feature_mode not in valid_modes:
            raise ValueError(f"feature_mode must be one of {valid_modes}")

        # Validate model type
        valid_models = ['adaptive', 'temporal', 'context_aware', 'ensemble']
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}")

        # Validate optimizer
        valid_optimizers = ['adam', 'sgd']
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}")

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print(f"[CONFIG] GeometricStage1Config validation passed")
        print(f"  Feature mode: {self.feature_mode} ({self.get_feature_dim()}D)")
        print(f"  Model type: {self.model_type}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")

    def print_config(self):
        """Print configuration in formatted way"""
        print(f"\n{'='*80}")
        print("Geometric Stage1 Configuration")
        print(f"{'='*80}\n")

        print("Feature Configuration:")
        print(f"  Mode: {self.feature_mode} ({self.get_feature_dim()}D)")
        print(f"  Use temporal: {self.use_temporal}")
        print(f"  Use scene context: {self.use_scene_context}")
        if self.use_temporal:
            print(f"  History length: {self.history_length}")

        print("\nModel Architecture:")
        print(f"  Type: {self.model_type}")
        if self.model_type == 'adaptive':
            print(f"  Hidden dims: {self.hidden_dims}")
        elif self.model_type in ['temporal', 'context_aware']:
            print(f"  Hidden size: {self.hidden_size}")
        elif self.model_type == 'ensemble':
            print(f"  Num models: {self.num_ensemble_models}")
        print(f"  Dropout: {self.dropout}")

        print("\nTraining Configuration:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Weight decay: {self.weight_decay}")
        print(f"  Optimizer: {self.optimizer}")
        print(f"  Scheduler: {self.scheduler}")

        print("\nData Configuration:")
        print(f"  Data path: {self.data_path}")
        print(f"  Frame interval: {self.frame_interval}")
        print(f"  Num workers: {self.num_workers}")

        print("\nRegularization:")
        print(f"  Weight regularization: {self.weight_regularization}")
        print(f"  Sparsity regularization: {self.sparsity_regularization}")
        print(f"  Max grad norm: {self.max_grad_norm}")

        print(f"\n{'='*80}\n")

    def to_dict(self):
        """Convert config to dictionary"""
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__.keys()
        }


def get_default_config() -> GeometricStage1Config:
    """Get default configuration"""
    return GeometricStage1Config()


def get_quick_test_config() -> GeometricStage1Config:
    """Get configuration for quick testing"""
    return GeometricStage1Config(
        feature_mode='bboxposition_only',  # Fastest mode (no optical flow)
        epochs=5,
        batch_size=16,
        early_stopping_patience=3,
        log_interval=5
    )


if __name__ == '__main__':
    print("Testing GeometricStage1Config...\n")

    # Test default config
    print("--- Default Configuration ---")
    config = get_default_config()
    config.validate()
    config.print_config()

    # Test quick test config
    print("\n--- Quick Test Configuration ---")
    config_test = get_quick_test_config()
    config_test.validate()
    config_test.print_config()

    # Test to_dict
    print("\n--- Config as Dictionary ---")
    config_dict = config.to_dict()
    print(f"Keys: {list(config_dict.keys())[:10]}...")

    print("\n✅ GeometricStage1Config test completed!")
