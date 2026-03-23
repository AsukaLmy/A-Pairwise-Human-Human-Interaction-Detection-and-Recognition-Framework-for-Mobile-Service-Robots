#!/usr/bin/env python3
"""
Configuration for MoE Geometric Classifier Training
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MoEGeometricConfig:
    """
    Configuration for MoE-based geometric behavior classification

    Model Architecture:
        - Input: 10D geometric features
        - Gate: Scene prediction network
        - Experts: 31 scene-specific behavior classifiers
        - Output: 3-class behavior prediction (Walking/Standing/Sitting)
    """

    # ========================================================================
    # Model Architecture
    # ========================================================================
    input_dim: int = 10  # Will be dynamically adjusted based on feature_mode
    num_classes: int = 3  # Walking, Standing, Sitting
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])  # MLP hidden dimensions
    dropout: float = 0.3  # Dropout rate

    # Feature selection mode
    feature_mode: str = 'both'  # 'both' (10D), 'opticalflow_only' (5D), 'bboxposition_only' (5D)

    # ========================================================================
    # Data Augmentation
    # ========================================================================
    use_augmentation: bool = False  # Enable data augmentation on geometric features
    augmentation_noise_std: float = 0.05  # Gaussian noise std for feature augmentation
    augmentation_mixup_alpha: float = 0.2  # Mixup alpha parameter (0=disabled)
    augmentation_prob: float = 0.5  # Probability of applying augmentation

    # ========================================================================
    # Reproducibility
    # ========================================================================
    seed: int = 59  # Random seed for reproducibility
    deterministic: bool = False  # Enable deterministic mode (slower but fully reproducible)

    # ========================================================================
    # Training Parameters
    # ========================================================================
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # Focal Loss (for class imbalance)
    use_focal_loss: bool = True  # Use focal loss for behavior classification
    focal_gamma: float = 2.0  # Focal loss gamma parameter (higher = more focus on hard examples)

    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    scheduler_patience: int = 5  # For ReduceLROnPlateau
    scheduler_factor: float = 0.5  # For ReduceLROnPlateau
    scheduler_step_size: int = 10  # For StepLR
    scheduler_gamma: float = 0.5  # For StepLR
    min_lr: float = 1e-6  # Minimum learning rate

    # ========================================================================
    # Data Parameters
    # ========================================================================
    data_path: str = '../dataset'
    frame_interval: int = 1

    # Feature normalization
    normalize_features: bool = True
    normalization_stats_path: Optional[str] = None  # If None, compute from training data

    # Class balancing
    use_class_weights: bool = True  # Use weighted loss for imbalanced classes
    use_oversampling: bool = False  # Oversample minority classes

    # ========================================================================
    # Checkpoint and Logging
    # ========================================================================
    checkpoint_dir: str = './checkpoints/moe_geometric'
    log_dir: str = './logs/moe_geometric'
    save_frequency: int = 5  # Save checkpoint every N epochs
    save_best_only: bool = True  # Save only best model by val accuracy

    # ========================================================================
    # Evaluation
    # ========================================================================
    eval_frequency: int = 1  # Evaluate on val set every N epochs
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_metric: str = 'val_behavior_acc'  # Metric to monitor

    # ========================================================================
    # Hardware
    # ========================================================================
    device: str = 'cuda'  # 'cuda' or 'cpu'
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    num_workers: int = 4  # Set to 0 for Windows compatibility (use 4 on Linux)
    pin_memory: bool = True   # Set to true in sarah's device!!!!!!!!!!!! Disabled due to Windows CUDA compatibility issues

    # ========================================================================
    # Experiment Tracking
    # ========================================================================
    experiment_name: str = 'moe_geometric_baseline'
    notes: str = 'MoE with 6D geometric features, 31 experts, 3-class'

    def __post_init__(self):
        """Validate and adjust configuration"""
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def print_config(self):
        """Print configuration in formatted way"""
        print(f"\n{'='*80}")
        print("Geometric MLP Classifier Configuration")
        print(f"{'='*80}\n")

        print("Model Architecture:")
        print(f"  Input dimension: {self.input_dim}D geometric features")
        print(f"  Number of classes: {self.num_classes} (Walking/Standing/Sitting)")
        print(f"  Hidden dims: {self.hidden_dims}")
        print(f"  Dropout: {self.dropout}")

        print("\nTraining Parameters:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Total epochs: {self.num_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Weight decay: {self.weight_decay}")
        print(f"  Focal Loss: {'Enabled' if self.use_focal_loss else 'Disabled'} (gamma={self.focal_gamma})")

        print("\nData Parameters:")
        print(f"  Data path: {self.data_path}")
        print(f"  Frame interval: {self.frame_interval}")
        print(f"  Normalize features: {self.normalize_features}")
        print(f"  Use class weights: {self.use_class_weights}")
        print(f"  Use oversampling: {self.use_oversampling}")

        print("\nData Augmentation:")
        print(f"  Enabled: {self.use_augmentation}")
        if self.use_augmentation:
            print(f"  - Gaussian noise std: {self.augmentation_noise_std}")
            print(f"  - Mixup alpha: {self.augmentation_mixup_alpha}")
            print(f"  - Augmentation probability: {self.augmentation_prob}")

        print("\nCheckpoint and Logging:")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Log dir: {self.log_dir}")
        print(f"  Save frequency: every {self.save_frequency} epochs")
        print(f"  Early stopping: {self.early_stopping} (patience={self.early_stopping_patience})")

        print("\nExperiment:")
        print(f"  Name: {self.experiment_name}")
        print(f"  Notes: {self.notes}")

        print(f"\n{'='*80}\n")

    def to_dict(self):
        """Convert config to dictionary"""
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__.keys()
        }


def get_default_config() -> MoEGeometricConfig:
    """Get default MoE geometric configuration"""
    return MoEGeometricConfig()


def get_quick_test_config() -> MoEGeometricConfig:
    """Get configuration for quick testing (small model, few epochs)"""
    config = MoEGeometricConfig(
        hidden_dims=[128, 64],
        batch_size=32,
        num_epochs=10,
        save_frequency=2,
        early_stopping_patience=5,
        experiment_name='geometric_mlp_quick_test',
        notes='Quick test configuration with reduced model size'
    )
    return config


def get_full_training_config() -> MoEGeometricConfig:
    """Get configuration for full training (large model, many epochs)"""
    config = MoEGeometricConfig(
        hidden_dims=[512, 256, 128],
        dropout=0.4,
        batch_size=128,
        num_epochs=100,
        learning_rate=0.0005,
        save_frequency=10,
        early_stopping_patience=25,
        use_focal_loss=True,
        focal_gamma=2.5,
        experiment_name='geometric_mlp_full_training',
        notes='Full training with large MLP model, extended epochs, and focal loss'
    )
    return config


if __name__ == '__main__':
    print("Testing MoEGeometricConfig...\n")

    # Test default config
    print("--- Default Configuration ---")
    config = get_default_config()
    config.print_config()

    # Test quick test config
    print("\n--- Quick Test Configuration ---")
    config_test = get_quick_test_config()
    config_test.print_config()

    # Test full training config
    print("\n--- Full Training Configuration ---")
    config_full = get_full_training_config()
    config_full.print_config()

    # Test to_dict
    print("\n--- Config as Dictionary ---")
    config_dict = config.to_dict()
    print(f"Keys: {list(config_dict.keys())[:10]}...")  # Print first 10 keys

    print("\n✅ MoEGeometricConfig test completed!")
