"""
Geometric MLP Classifier Training Script
Train ResidualBlock MLP for three-class behavior classification
using geometric features
"""

# Fix OpenMP errors and encoding
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# FLOPs calculation
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
import random
import time
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import json
from datetime import datetime

# Import project components
from configs.moe_config import MoEGeometricConfig, get_default_config, get_quick_test_config
from models.geometric_mlp_classifier import GeometricMLPClassifier
from datasets.resnet_stage2_dataset import ResNetStage2Dataset
from src.features.geometric_flow_extractor import GeometricFlowExtractor
from src.classifiers.geometric_stage2_classifier import Stage2Evaluator
from utils.profiling import MoEProfiler


def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic mode (slower but fully reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Clear CUDA cache to avoid memory fragmentation issues
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if deterministic:
        # Enable deterministic algorithms (slower but fully reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[SEED] Deterministic mode enabled (seed={seed})")
    else:
        # Allow CuDNN to find optimal algorithms (faster but less reproducible)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print(f"[SEED] Random seed set to {seed}")


def augment_features_noise(features, noise_std=0.05):
    """
    Add Gaussian noise to geometric features for augmentation

    Args:
        features: [B, D] tensor of geometric features
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Augmented features with same shape
    """
    noise = torch.randn_like(features) * noise_std
    return features + noise


def augment_features_mixup(features1, labels1, features2, labels2, alpha=0.2):
    """
    Mixup augmentation: mix two samples

    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)

    Args:
        features1, features2: [B, D] feature tensors
        labels1, labels2: [B] label tensors
        alpha: Beta distribution parameter

    Returns:
        mixed_features: [B, D] mixed features
        mixed_labels: [B] mixed labels (hard label from dominant sample)
        lam: mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # Mix features
    mixed_features = lam * features1 + (1 - lam) * features2

    # Hard label: use label from dominant sample
    mixed_labels = labels1 if lam > 0.5 else labels2

    return mixed_features, mixed_labels, lam


def apply_augmentation(features, behavior_labels, scene_labels, config):
    """
    Apply data augmentation to geometric features

    Args:
        features: [B, D] feature tensor
        behavior_labels: [B] behavior labels
        scene_labels: [B] scene labels
        config: MoEGeometricConfig

    Returns:
        augmented_features, augmented_behavior_labels, augmented_scene_labels
    """
    if not config.use_augmentation:
        return features, behavior_labels, scene_labels

    # Apply augmentation with probability
    if np.random.rand() < config.augmentation_prob:
        # Gaussian noise augmentation
        features = augment_features_noise(features, config.augmentation_noise_std)

    # Mixup augmentation (if alpha > 0)
    if config.augmentation_mixup_alpha > 0 and np.random.rand() < 0.5:
        # Create random permutation for mixup pairs
        batch_size = features.size(0)
        indices = torch.randperm(batch_size, device=features.device)

        features, behavior_labels, _ = augment_features_mixup(
            features, behavior_labels,
            features[indices], behavior_labels[indices],
            alpha=config.augmentation_mixup_alpha
        )
        # Note: scene_labels not mixed, kept from original sample

    return features, behavior_labels, scene_labels


# ============================================================================
# FLOPS CALCULATION
# ============================================================================

def format_number(num):
    """Format large numbers with appropriate suffix (K, M, G, T)"""
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


def calculate_flops(config, device):
    """
    Calculate inference FLOPs for MoE Geometric Classifier.

    Args:
        config: MoEGeometricConfig
        device: torch device
    """
    if not THOP_AVAILABLE:
        print("Error: thop library not available. Install with 'pip install thop'")
        return

    print(f"\n{'='*80}")
    print("INFERENCE FLOPs ANALYSIS (for deployment device evaluation)")
    print(f"{'='*80}")

    # Create model
    model = GeometricMLPClassifier(
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    model.eval()

    # ========================================================================
    # 1. Model Information
    # ========================================================================
    print(f"\n[1] MODEL: Geometric MLP Classifier")
    print("-" * 40)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Input dim: {config.input_dim}D")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Total parameters: {format_number(total_params)}")

    # ========================================================================
    # 2. FLOPs Calculation
    # ========================================================================
    print(f"\n[2] FLOPs ANALYSIS")
    print("-" * 40)

    # Create dummy input
    dummy_input = torch.randn(1, config.input_dim).to(device)

    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

        print(f"  FLOPs per sample: {format_number(flops)}FLOPs")
        print(f"  Parameters: {format_number(params)}")
    except Exception as e:
        print(f"  Error calculating FLOPs: {e}")
        flops = 0
        params = 0

    # ========================================================================
    # 3. Throughput Estimation
    # ========================================================================
    print(f"\n[3] THROUGHPUT ESTIMATION (reference)")
    print("-" * 40)

    if flops > 0:
        # Common GPU theoretical peak FLOPs (FP32)
        gpu_references = [
            ("RTX 4090", 82.6e12),
            ("RTX 4060 Laptop", 15.0e12),
            ("RTX 3060", 12.7e12),
            ("GTX 1080 Ti", 11.3e12),
            ("T4 (Cloud)", 8.1e12),
        ]

        print(f"  Theoretical max throughput (assuming 100% utilization):")
        for gpu_name, gpu_tflops in gpu_references:
            theoretical_fps = gpu_tflops / flops
            print(f"    {gpu_name:20s}: ~{theoretical_fps:,.0f} samples/sec")

        print(f"\n  Note: MLP model is lightweight - throughput will be memory/data bound")
        print(f"        Real throughput depends on feature extraction pipeline")

    # ========================================================================
    # 4. Configuration Summary
    # ========================================================================
    print(f"\n[4] CONFIGURATION SUMMARY")
    print("-" * 40)
    print(f"  Feature mode:  {config.feature_mode}")
    print(f"  Input dim:     {config.input_dim}D")
    print(f"  Hidden dims:   {config.hidden_dims}")
    print(f"  Num classes:   {config.num_classes}")

    print(f"\n{'='*80}")

    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# FEATURE SELECTION UTILITIES
# ============================================================================

def select_features_by_mode(full_features_9d: np.ndarray, feature_mode: str) -> np.ndarray:
    """
    Select feature subset based on feature_mode

    Args:
        full_features_9d: Complete 9D features from GeometricFlowExtractor
            [f0, f1, f2, f3, f4, f5, f7, f8, f9]
        feature_mode: 'both', 'opticalflow_only', or 'bboxposition_only'

    Returns:
        Selected features subset
    """
    if feature_mode == 'both':
        # Return all 9 features (will add sync as 10th later)
        return full_features_9d

    elif feature_mode == 'opticalflow_only':
        # Optical flow features: f2, f3, f4, f9 (indices 2, 3, 4, 8)
        # Note: sync will be added later as 5th feature
        return np.array([
            full_features_9d[2],  # f2: flow_mean/area
            full_features_9d[3],  # f3: flow_std/area
            full_features_9d[4],  # f4: vertical_dominance
            full_features_9d[8],  # f9: direction_consistency
        ], dtype=np.float32)

    elif feature_mode == 'bboxposition_only':
        # Bbox position features: f0, f1, f5, f7, f8 (indices 0, 1, 5, 6, 7)
        return np.array([
            full_features_9d[0],  # f0: distance/height
            full_features_9d[1],  # f1: distance/width
            full_features_9d[5],  # f5: aspect_ratio
            full_features_9d[6],  # f7: relative_height
            full_features_9d[7],  # f8: relative_bottom
        ], dtype=np.float32)

    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")


class MoEGeometricTrainer:
    """Trainer for MoE Geometric Classifier"""

    def __init__(self, config: MoEGeometricConfig, device: str):
        self.config = config
        self.device = device

        # Create model
        self.model = GeometricMLPClassifier(
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(device)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Create learning rate scheduler
        if config.use_scheduler:
            if config.scheduler_type == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.num_epochs,
                    eta_min=config.min_lr
                )
            elif config.scheduler_type == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config.scheduler_step_size,
                    gamma=config.scheduler_gamma
                )
            elif config.scheduler_type == 'plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=config.scheduler_factor,
                    patience=config.scheduler_patience,
                    min_lr=config.min_lr
                )
        else:
            self.scheduler = None

        # Create evaluator
        self.evaluator = Stage2Evaluator(['Walking Together', 'Standing Together', 'Sitting Together'])

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        # Normalization stats
        self.normalization_stats = None

        print(f"\nGeometricMLPTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})")
        if self.scheduler:
            print(f"  Scheduler: {config.scheduler_type}")

    def extract_features_and_labels(self, dataset, split_name='train'):
        """
        Extract geometric features and labels from dataset

        Returns:
            features: [N, 10] geometric features (9D + interaction synchrony)
            behavior_labels: [N] behavior labels (0/1/2)
        """
        print(f"\n{'='*60}")
        print(f"Extracting features from {split_name} split")
        print(f"{'='*60}")

        geometric_extractor = GeometricFlowExtractor(flow_bound=20.0, cache_enabled=True)

        # Import synchrony feature calculator
        from src.features.interaction_synchrony import compute_interaction_synchrony

        features_list = []
        behavior_labels_list = []

        pbar = tqdm(range(len(dataset.samples)), desc=f"Extracting {split_name}")

        for idx in pbar:
            try:
                sample = dataset.samples[idx]

                # Load images
                prev_image_path = sample.get('prev_image_path')
                image_path = sample.get('image_path')

                if not prev_image_path or not os.path.exists(prev_image_path):
                    continue
                if not image_path or not os.path.exists(image_path):
                    continue

                prev_image = Image.open(prev_image_path).convert('RGB')
                curr_image = Image.open(image_path).convert('RGB')

                # Extract 9D geometric features (A -> B perspective)
                features_A_to_B = geometric_extractor.extract_geometric_features(
                    prev_image, curr_image,
                    sample['person_A_box'],
                    sample['person_B_box']
                )  # [9]

                # Extract 9D geometric features (B -> A perspective)
                features_B_to_A = geometric_extractor.extract_geometric_features(
                    prev_image, curr_image,
                    sample['person_B_box'],
                    sample['person_A_box']
                )  # [9]

                # Convert to numpy
                if isinstance(features_A_to_B, torch.Tensor):
                    feat_A = features_A_to_B.cpu().numpy()
                else:
                    feat_A = features_A_to_B

                if isinstance(features_B_to_A, torch.Tensor):
                    feat_B = features_B_to_A.cpu().numpy()
                else:
                    feat_B = features_B_to_A

                # ========== STEP 1: Select features based on mode ==========
                feat_A_selected = select_features_by_mode(feat_A, self.config.feature_mode)
                feat_B_selected = select_features_by_mode(feat_B, self.config.feature_mode)

                # ========== STEP 2: Symmetric Averaging ==========
                if self.config.feature_mode == 'both':
                    # Original 9D symmetric averaging
                    # For asymmetric features (f2, f3, f4, f9), take average
                    # For symmetric features (f0, f1, f5, f7, f8), keep as is
                    symmetric_features = np.array([
                        feat_A_selected[0],                              # f0: distance/height (symmetric)
                        feat_A_selected[1],                              # f1: distance/width (symmetric)
                        (feat_A_selected[2] + feat_B_selected[2]) / 2.0, # f2: avg flow intensity (AVERAGED)
                        (feat_A_selected[3] + feat_B_selected[3]) / 2.0, # f3: avg flow variability (AVERAGED)
                        (feat_A_selected[4] + feat_B_selected[4]) / 2.0, # f4: avg vertical dominance (AVERAGED)
                        feat_A_selected[5],                              # f5: aspect ratio (symmetric)
                        feat_A_selected[6],                              # f7: relative height (symmetric)
                        feat_A_selected[7],                              # f8: relative bottom position (symmetric)
                        (feat_A_selected[8] + feat_B_selected[8]) / 2.0, # f9: avg motion direction consistency (AVERAGED)
                    ], dtype=np.float32)

                elif self.config.feature_mode == 'opticalflow_only':
                    # 4D optical flow averaging (all asymmetric)
                    symmetric_features = np.array([
                        (feat_A_selected[0] + feat_B_selected[0]) / 2.0, # f2: avg flow intensity
                        (feat_A_selected[1] + feat_B_selected[1]) / 2.0, # f3: avg flow variability
                        (feat_A_selected[2] + feat_B_selected[2]) / 2.0, # f4: avg vertical dominance
                        (feat_A_selected[3] + feat_B_selected[3]) / 2.0, # f9: avg motion direction consistency
                    ], dtype=np.float32)

                elif self.config.feature_mode == 'bboxposition_only':
                    # 5D bbox features (all symmetric, no averaging needed)
                    symmetric_features = feat_A_selected

                else:
                    raise ValueError(f"Unknown feature_mode: {self.config.feature_mode}")

                # ========== STEP 3: Add interaction synchrony ==========
                # Only compute sync if using optical flow features
                if self.config.feature_mode in ['both', 'opticalflow_only']:
                    sync_score = compute_interaction_synchrony(
                        torch.from_numpy(feat_A).unsqueeze(0),
                        torch.from_numpy(feat_B).unsqueeze(0)
                    )  # scalar or [1] array

                    # Ensure sync_score is a scalar value
                    if isinstance(sync_score, torch.Tensor):
                        sync_score_np = sync_score.item() if sync_score.numel() == 1 else sync_score.cpu().numpy().flatten()[0]
                    elif isinstance(sync_score, np.ndarray):
                        sync_score_np = sync_score.item() if sync_score.size == 1 else sync_score.flatten()[0]
                    else:
                        sync_score_np = float(sync_score)

                    features_final = np.concatenate([symmetric_features, [sync_score_np]])
                else:
                    # bbox-only mode: no sync feature
                    features_final = symmetric_features

                # Get labels
                behavior_label = sample['stage2_label']  # 0/1/2

                features_list.append(features_final)
                behavior_labels_list.append(behavior_label)

            except Exception as e:
                continue

        pbar.close()

        # Convert to arrays
        features = np.vstack(features_list)
        behavior_labels = np.array(behavior_labels_list)

        print(f"\nExtracted {len(features)} samples:")
        print(f"  Features: {features.shape} ({self.config.feature_mode} mode)")
        print(f"  Behavior distribution: ", dict(zip(*np.unique(behavior_labels, return_counts=True))))

        return features, behavior_labels

    def compute_normalization_stats(self, train_features):
        """Compute mean and std from training features"""
        mean = np.mean(train_features, axis=0)
        std = np.std(train_features, axis=0)

        self.normalization_stats = {
            'mean': mean,
            'std': std,
            'n_samples': len(train_features)
        }

        # Save stats
        stats_path = os.path.join(self.config.checkpoint_dir, 'normalization_stats.pth')
        torch.save(self.normalization_stats, stats_path)
        print(f"\nNormalization stats computed ({len(train_features)} samples), saved to: {stats_path}")

    def normalize_features(self, features):
        """Apply Z-score normalization to features"""
        if self.normalization_stats is None:
            return features

        mean = self.normalization_stats['mean']
        std = self.normalization_stats['std']

        normalized = (features - mean) / (std + 1e-8)
        return normalized

    def create_dataloader(self, features, behavior_labels, shuffle=True):
        """Create DataLoader from features and labels"""
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(features).float(),
            torch.from_numpy(behavior_labels).long()
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with geometric extraction
            pin_memory=self.config.pin_memory
        )

        return loader

    def train_epoch(self, train_loader, epoch, profiler=None):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for features, behavior_labels in pbar:
            # Track data loading time
            if profiler is not None:
                data_start_time = time.perf_counter()

            features = features.to(self.device)
            behavior_labels = behavior_labels.to(self.device)

            # Apply data augmentation (training only)
            features, behavior_labels, _ = apply_augmentation(
                features, behavior_labels, behavior_labels, self.config
            )

            if profiler is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                data_time = time.perf_counter() - data_start_time
                compute_start_time = time.perf_counter()

            # Forward pass
            logits = self.model(features)

            # Compute loss
            if self.config.use_focal_loss:
                probs = torch.softmax(logits, dim=1)
                ce_loss = nn.functional.cross_entropy(logits, behavior_labels, reduction='none')
                pt = probs.gather(1, behavior_labels.unsqueeze(1)).squeeze(1)
                focal_weight = (1 - pt) ** self.config.focal_gamma
                loss = (focal_weight * ce_loss).mean()
            else:
                loss = nn.functional.cross_entropy(logits, behavior_labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track compute time
            if profiler is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                compute_time = time.perf_counter() - compute_start_time
                profiler.record_batch_timing(data_time, compute_time)

            # Accumulate metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == behavior_labels).sum().item()
            total += behavior_labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct / total:.3f}",
            })

        # Compute epoch averages
        n_batches = len(train_loader)
        avg_metrics = {
            'loss': total_loss / n_batches,
            'behavior_acc': correct / total,
        }

        return avg_metrics

    def evaluate(self, val_loader, split_name='val'):
        """Evaluate on validation/test set"""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, behavior_labels in val_loader:
                features = features.to(self.device)

                logits = self.model(features)
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(behavior_labels)

        # Concatenate all predictions
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Compute behavior metrics using evaluator
        self.evaluator.reset()
        self.evaluator.update(all_preds, all_labels)
        eval_metrics = self.evaluator.compute_metrics()

        # Extract metrics
        behavior_acc = eval_metrics['overall_accuracy']
        confusion = eval_metrics['confusion_matrix']
        per_class_acc = eval_metrics['per_class_accuracy']

        # Compute macro F1 score
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        metrics = {
            'behavior_acc': behavior_acc,
            'macro_f1': macro_f1,
            'walking_acc': per_class_acc.get(0, 0.0),
            'standing_acc': per_class_acc.get(1, 0.0),
            'sitting_acc': per_class_acc.get(2, 0.0),
        }

        return metrics, confusion

    def train(self, train_loader, val_loader, test_loader=None, profiler=None):
        """Main training loop"""
        print(f"\n{'='*80}")
        print("STARTING TRAINING")
        print(f"{'='*80}\n")

        best_checkpoint_path = None

        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch

            # Start epoch timing
            if profiler is not None:
                profiler.start_epoch()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch, profiler=profiler)

            # Evaluate
            if epoch % self.config.eval_frequency == 0:
                val_metrics, val_confusion = self.evaluate(val_loader, 'val')

                print(f"\nEpoch {epoch}/{self.config.num_epochs}:")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['behavior_acc']:.3f}")
                print(f"  Val   - Acc: {val_metrics['behavior_acc']:.3f}, Macro F1: {val_metrics['macro_f1']:.3f}")
                print(f"          Walking: {val_metrics['walking_acc']:.3f}, Standing: {val_metrics['standing_acc']:.3f}, Sitting: {val_metrics['sitting_acc']:.3f}")

                # Save best model
                if val_metrics['behavior_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['behavior_acc']
                    self.epochs_without_improvement = 0

                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_acc': self.best_val_acc,
                        'config': self.config.to_dict(),
                        'normalization_stats': self.normalization_stats
                    }

                    best_checkpoint_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
                    torch.save(checkpoint, best_checkpoint_path)
                    print(f"  ✓ Saved best model (val_acc={self.best_val_acc:.3f})")
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if self.config.early_stopping and self.epochs_without_improvement >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered ({self.config.early_stopping_patience} epochs without improvement)")
                    break

            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['behavior_acc'])
                else:
                    self.scheduler.step()

            # End epoch timing
            if profiler is not None:
                num_samples = len(train_loader.dataset)
                timing_info = profiler.end_epoch(num_samples)
                if epoch % self.config.eval_frequency == 0:
                    print(f"  Timing - Epoch: {timing_info['epoch_time']:.2f}s, "
                          f"Data: {timing_info['data_time']:.2f}s ({timing_info['data_fraction']*100:.1f}%), "
                          f"Compute: {timing_info['compute_time']:.2f}s ({timing_info['compute_fraction']*100:.1f}%), "
                          f"Throughput: {timing_info['samples_per_second']:.2f} samples/s")

        # Final evaluation on test set
        if test_loader is not None:
            print(f"\n{'='*80}")
            print("FINAL TEST EVALUATION")
            print(f"{'='*80}\n")

            # Load best model
            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                checkpoint = torch.load(best_checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint['epoch']}")

            test_metrics, test_confusion = self.evaluate(test_loader, 'test')

            print(f"\nTest Results:")
            print(f"  Accuracy:     {test_metrics['behavior_acc']:.3f}")
            print(f"  Macro F1:     {test_metrics['macro_f1']:.3f}")
            print(f"  Walking Acc:  {test_metrics['walking_acc']:.3f}")
            print(f"  Standing Acc: {test_metrics['standing_acc']:.3f}")
            print(f"  Sitting Acc:  {test_metrics['sitting_acc']:.3f}")
            print(f"\nConfusion Matrix:")
            print(test_confusion)

        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Geometric MLP Classifier')

    # Basic settings
    parser.add_argument('--config', type=str, default='default', choices=['default', 'quick', 'full'],
                       help='Configuration preset')
    parser.add_argument('--data_path', type=str, default='../dataset',
                       help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    # Data parameters
    parser.add_argument('--frame_interval', type=int, default=None,
                       help='Frame sampling interval (default: use config value)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (default: use config value)')

    # Model parameters
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=None,
                       help='MLP hidden dimensions (default: 256 128)')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout rate (default: use config value)')

    # Data augmentation
    parser.add_argument('--use_augmentation', action='store_true',
                       help='Enable data augmentation on geometric features')
    parser.add_argument('--augmentation_noise_std', type=float, default=None,
                       help='Gaussian noise std for augmentation (default: 0.05)')
    parser.add_argument('--augmentation_mixup_alpha', type=float, default=None,
                       help='Mixup alpha parameter (default: 0.2)')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Total training epochs (default: use config value)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (default: use config value)')

    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Checkpoint directory (default: use config value)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: use config value)')

    # Profiling
    parser.add_argument('--profile', action='store_true',
                       help='Enable comprehensive profiling (model size, FLOPs, memory, timing)')
    parser.add_argument('--profile_flops', action='store_true',
                       help='Calculate FLOPs (one-time at start)')
    parser.add_argument('--profile_memory', action='store_true',
                       help='Profile GPU memory usage')
    parser.add_argument('--profile_save_dir', type=str, default=None,
                       help='Directory to save profiling results (default: checkpoint_dir/profiling)')

    # Feature selection
    parser.add_argument('--feature_mode', type=str, default='both',
                       choices=['both', 'opticalflow_only', 'bboxposition_only'],
                       help='Feature mode: both (10D all), opticalflow_only (5D motion), bboxposition_only (5D geometry)')

    # FLOPs calculation only
    parser.add_argument('--flops_only', action='store_true',
                       help='Only calculate inference FLOPs without training')

    args = parser.parse_args()

    # Load configuration
    if args.config == 'quick':
        config = get_quick_test_config()
    elif args.config == 'full':
        from configs.moe_config import get_full_training_config
        config = get_full_training_config()
    else:
        config = get_default_config()

    # Override config with command-line arguments
    config.data_path = args.data_path
    config.device = args.device if torch.cuda.is_available() else 'cpu'

    # Override data parameters
    if args.frame_interval is not None:
        config.frame_interval = args.frame_interval
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # Override model parameters
    if args.hidden_dims is not None:
        config.hidden_dims = args.hidden_dims
    if args.dropout is not None:
        config.dropout = args.dropout

    # Override data augmentation
    if args.use_augmentation:
        config.use_augmentation = True
        print("[CLI] Enabling data augmentation")
    if args.augmentation_noise_std is not None:
        config.augmentation_noise_std = args.augmentation_noise_std
    if args.augmentation_mixup_alpha is not None:
        config.augmentation_mixup_alpha = args.augmentation_mixup_alpha

    # Override training parameters
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    # Override checkpoint and logging
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name

    # Override feature mode and update input_dim accordingly
    if args.feature_mode is not None:
        config.feature_mode = args.feature_mode
        if config.feature_mode == 'both':
            config.input_dim = 10  # 9D geometric + 1D sync
        elif config.feature_mode == 'opticalflow_only':
            config.input_dim = 5   # 4D flow + 1D sync
        elif config.feature_mode == 'bboxposition_only':
            config.input_dim = 5   # 5D bbox (no sync)
        print(f"[CLI] Using feature mode: {config.feature_mode}")
        print(f"[Config] Input dimension set to {config.input_dim}D based on feature_mode={config.feature_mode}")

    config.print_config()

    # ========================================================================
    # FLOPs ONLY MODE
    # ========================================================================
    if args.flops_only:
        calculate_flops(config, config.device)
        print("\n✓ FLOPs calculation complete. Exiting without training.")
        return

    # Set random seed for reproducibility
    set_seed(config.seed, config.deterministic)

    # Create trainer
    trainer = MoEGeometricTrainer(config, config.device)

    # Initialize profiler if requested
    profiler = None
    if args.profile or args.profile_flops or args.profile_memory:
        profiler_save_dir = args.profile_save_dir or os.path.join(config.checkpoint_dir, 'profiling')
        profiler = MoEProfiler(trainer.model, device=config.device, save_dir=profiler_save_dir)

        print(f"\n{'='*80}")
        print("PROFILING MODEL")
        print(f"{'='*80}\n")

        # Profile model size
        print("Profiling model size...")
        profiler.profile_model_size()

        # Profile FLOPs
        if args.profile_flops or args.profile:
            print("Profiling FLOPs...")
            profiler.profile_flops(input_shape=(config.input_dim,), batch_size=config.batch_size)

        # Profile memory (will be done later with actual data)
        if args.profile_memory or args.profile:
            print("Memory profiling will be performed with actual training data...")

    # Load datasets
    print(f"\n{'='*80}")
    print("LOADING DATASETS")
    print(f"{'='*80}\n")

    train_dataset = ResNetStage2Dataset(
        data_path=config.data_path,
        split='train',
        use_optical_flow=True,
        use_geometric=False,
        use_scene_context=False,
        frame_interval=config.frame_interval
    )

    val_dataset = ResNetStage2Dataset(
        data_path=config.data_path,
        split='val',
        use_optical_flow=True,
        use_geometric=False,
        use_scene_context=False,
        frame_interval=config.frame_interval
    )

    test_dataset = ResNetStage2Dataset(
        data_path=config.data_path,
        split='test',
        use_optical_flow=True,
        use_geometric=False,
        use_scene_context=False,
        frame_interval=config.frame_interval
    )

    # Extract features
    train_features, train_behavior_labels = trainer.extract_features_and_labels(train_dataset, 'train')
    val_features, val_behavior_labels = trainer.extract_features_and_labels(val_dataset, 'val')
    test_features, test_behavior_labels = trainer.extract_features_and_labels(test_dataset, 'test')

    # Compute normalization stats
    if config.normalize_features:
        trainer.compute_normalization_stats(train_features)

        # Normalize all splits
        train_features = trainer.normalize_features(train_features)
        val_features = trainer.normalize_features(val_features)
        test_features = trainer.normalize_features(test_features)

    # Create dataloaders
    train_loader = trainer.create_dataloader(train_features, train_behavior_labels, shuffle=True)
    val_loader = trainer.create_dataloader(val_features, val_behavior_labels, shuffle=False)
    test_loader = trainer.create_dataloader(test_features, test_behavior_labels, shuffle=False)

    # Profile memory with actual training data
    if profiler is not None and (args.profile_memory or args.profile):
        print("\nProfiling GPU memory usage with training data...")
        # Get a batch of training data
        sample_features, _, _ = next(iter(train_loader))
        sample_features = sample_features.to(config.device)
        profiler.profile_memory(sample_features)

    # Train
    trainer.train(train_loader, val_loader, test_loader, profiler=profiler)

    # Print final profiling summary
    if profiler is not None:
        profiler.print_summary()
        profiler.save_summary()

    print("\n✅ Training completed successfully!")


if __name__ == '__main__':
    main()
