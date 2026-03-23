"""
Offline Feature Extraction + Geometric Features Training Script
Combines frozen EfficientNet backbone visual features with 10D geometric features
Only trains the Relation Network classification head
"""

# Fix OpenMP errors and encoding
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '1'

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from datetime import datetime
import json
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# FLOPs calculation
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

# Import project components
from configs.resnet_stage2_config import create_backbone_config
from datasets.resnet_stage2_dataset import ResNetStage2Dataset
from models.resnet_stage2_classifier import ResNetRelationStage2Classifier
from src.features.geometric_flow_extractor import GeometricFlowExtractor
from src.features.interaction_synchrony import compute_interaction_synchrony
from src.classifiers.geometric_stage2_classifier import Stage2Evaluator
from utils.resnet_model_factory import create_resnet_stage2_model


# ============================================================================
# FEATURE SELECTION UTILITIES
# ============================================================================

def select_features_by_mode(full_features_9d: np.ndarray, feature_mode: str) -> np.ndarray:
    """
    Select feature subset based on feature_mode

    Args:
        full_features_9d: Complete 9D features from GeometricFlowExtractor
            [f0, f1, f2, f3, f4, f5, f7, f8, f9]
        feature_mode: 'both', 'opticalflow_only', 'bboxposition_only', or 'none'

    Returns:
        Selected features subset (empty array for 'none' mode)
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

    elif feature_mode == 'none':
        # No geometric features - visual only mode
        return np.array([], dtype=np.float32)

    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")


# ============================================================================
# PHASE 1: OFFLINE FEATURE EXTRACTION
# ============================================================================

class OfflineFeatureExtractor:
    """Extract visual and geometric features offline before training"""

    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Create and freeze backbone
        print("Creating backbone model for feature extraction...")
        print(f"  Backbone: {config.backbone_name}")
        print(f"  Visual feature dim: {config.visual_feature_dim}")

        full_model = create_resnet_stage2_model(config).to(device)
        self.backbone = full_model.backbone
        self.backbone.eval()

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        print(f"Backbone ready for offline feature extraction")

        # Create geometric feature extractor
        self.geometric_extractor = GeometricFlowExtractor(flow_bound=20.0, cache_enabled=True)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((config.crop_size, config.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_and_crop_person(self, image_path, bbox):
        """Load image and crop person region"""
        try:
            # Load full image
            img = Image.open(image_path).convert('RGB')

            # Extract bbox coordinates
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.cpu().numpy()
            x, y, w, h = bbox

            # Crop person region
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))

            if x2 <= x1 or y2 <= y1:
                # Invalid crop, return blank image
                person_img = Image.new('RGB', (64, 64), (128, 128, 128))
            else:
                person_img = img.crop((x1, y1, x2, y2))

            # Apply transforms
            person_tensor = self.transform(person_img)

            return person_tensor

        except Exception as e:
            # Return blank tensor on error
            return torch.zeros(3, self.config.crop_size, self.config.crop_size)

    def extract_visual_features(self, image_path, person_A_box, person_B_box):
        """Extract visual features for both persons using frozen backbone"""
        with torch.no_grad():
            # Load and crop both persons
            person_A_img = self.load_and_crop_person(image_path, person_A_box)
            person_B_img = self.load_and_crop_person(image_path, person_B_box)

            # Add batch dimension and move to device
            person_A_batch = person_A_img.unsqueeze(0).to(self.device)
            person_B_batch = person_B_img.unsqueeze(0).to(self.device)

            # Extract features
            visual_A = self.backbone(person_A_batch).squeeze(0)  # [visual_dim]
            visual_B = self.backbone(person_B_batch).squeeze(0)  # [visual_dim]

            return visual_A.cpu().numpy(), visual_B.cpu().numpy()

    def extract_10d_geometric_features(self, sample, debug=False):
        """
        Extract 10D geometric features (9D + 1D synchrony) with symmetric averaging

        Uses symmetric averaging for asymmetric features to eliminate dependency on
        person A/B labeling order, ensuring the same feature vector regardless of
        which person is labeled as A or B.
        """
        try:
            # Get image paths
            prev_image_path = sample.get('prev_image_path')
            image_path = sample.get('image_path')

            # Check if paths exist
            if not prev_image_path:
                if debug:
                    print(f"[DEBUG] Missing prev_image_path in sample")
                return None

            if not os.path.exists(prev_image_path):
                if debug:
                    print(f"[DEBUG] prev_image_path does not exist: {prev_image_path}")
                return None

            if not image_path:
                if debug:
                    print(f"[DEBUG] Missing image_path in sample")
                return None

            if not os.path.exists(image_path):
                if debug:
                    print(f"[DEBUG] image_path does not exist: {image_path}")
                return None

            # Load images
            prev_image = Image.open(prev_image_path).convert('RGB')
            curr_image = Image.open(image_path).convert('RGB')

            # Extract 9D geometric features (A -> B perspective)
            features_A_to_B = self.geometric_extractor.extract_geometric_features(
                prev_image, curr_image,
                sample['person_A_box'],
                sample['person_B_box']
            )  # [9]

            # Extract 9D geometric features (B -> A perspective)
            features_B_to_A = self.geometric_extractor.extract_geometric_features(
                prev_image, curr_image,
                sample['person_B_box'],
                sample['person_A_box']
            )  # [9]

            # Convert to numpy for easier manipulation
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
                # For asymmetric features (f2, f3, f4, f9), take average of A->B and B->A
                # For symmetric features (f0, f1, f5, f7, f8), they're already the same
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

            elif self.config.feature_mode == 'none':
                # No geometric features - return empty array
                return np.array([], dtype=np.float32)

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

            return features_final  # [10] for both, [5] for opticalflow_only/bboxposition_only, [0] for none

        except Exception as e:
            if debug:
                print(f"[DEBUG] Exception in extract_10d_geometric_features: {e}")
                import traceback
                traceback.print_exc()
            return None

    def extract_all_features(self, dataset, split_name='train'):
        """Extract all features from dataset"""
        print(f"\n{'='*80}")
        print(f"EXTRACTING FEATURES: {split_name}")
        print(f"{'='*80}")
        print(f"Total samples in dataset: {len(dataset.samples)}")

        visual_A_list = []
        visual_B_list = []
        geometric_list = []
        label_list = []

        # Track extraction statistics
        total_samples = 0
        failed_geometric = 0
        failed_visual = 0
        success_count = 0

        pbar = tqdm(dataset.samples, desc=f"Extracting {split_name}")

        for sample in pbar:
            total_samples += 1
            # Enable debug for first 3 samples if no success yet
            debug_mode = (success_count == 0 and total_samples <= 3)

            try:
                # Extract visual features
                visual_A, visual_B = self.extract_visual_features(
                    sample['image_path'],
                    sample['person_A_box'],
                    sample['person_B_box']
                )

                # Extract geometric features (skip for 'none' mode)
                if self.config.feature_mode == 'none':
                    geometric_10d = np.array([], dtype=np.float32)
                else:
                    geometric_10d = self.extract_10d_geometric_features(sample, debug=debug_mode)

                    if geometric_10d is None:
                        failed_geometric += 1
                        if debug_mode:
                            print(f"\n[DEBUG] Sample {total_samples}: geometric_10d is None")
                            print(f"  image_path: {sample.get('image_path')}")
                            print(f"  prev_image_path: {sample.get('prev_image_path')}")
                        continue

                # Get label
                label = sample['stage2_label']

                # Append to lists
                visual_A_list.append(visual_A)
                visual_B_list.append(visual_B)
                geometric_list.append(geometric_10d)
                label_list.append(label)
                success_count += 1

            except Exception as e:
                failed_visual += 1
                if debug_mode:
                    # Print first few errors for debugging
                    print(f"\n[ERROR] Sample {total_samples}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue

        pbar.close()

        print(f"\nExtraction Statistics:")
        print(f"  Total samples: {total_samples}")
        print(f"  Successful: {success_count}")
        print(f"  Failed (geometric): {failed_geometric}")
        print(f"  Failed (visual/other): {failed_visual}")

        # Check if we have any valid samples
        if len(visual_A_list) == 0:
            raise ValueError(
                f"No valid samples extracted for {split_name} split!\n"
                f"  Total samples attempted: {total_samples}\n"
                f"  Failed geometric extraction: {failed_geometric}\n"
                f"  Failed visual extraction: {failed_visual}\n"
                f"Possible reasons:\n"
                f"  1. Missing 'prev_image_path' in dataset samples\n"
                f"  2. Image files not found at specified paths\n"
                f"  3. Invalid bounding boxes\n"
                f"Please check:\n"
                f"  - Dataset has optical flow enabled (use_optical_flow=True)\n"
                f"  - frame_interval allows for previous frames\n"
                f"  - Image paths are correct"
            )

        # Convert to numpy arrays
        visual_A = np.vstack(visual_A_list)  # [N, visual_dim]
        visual_B = np.vstack(visual_B_list)  # [N, visual_dim]
        labels = np.array(label_list)  # [N]

        # Handle geometric features (empty for 'none' mode)
        if self.config.feature_mode == 'none':
            geometric = np.zeros((len(labels), 0), dtype=np.float32)  # [N, 0]
        else:
            geometric = np.vstack(geometric_list)  # [N, spatial_dim]

        print(f"\n{split_name.upper()} Feature Extraction Complete:")
        print(f"  Samples: {len(labels)}")
        print(f"  Visual A shape: {visual_A.shape}")
        print(f"  Visual B shape: {visual_B.shape}")
        if self.config.feature_mode == 'none':
            print(f"  Geometric shape: {geometric.shape} (none - visual only)")
        else:
            print(f"  Geometric shape: {geometric.shape} ({self.config.spatial_feature_dim}D features)")
        print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

        return visual_A, visual_B, geometric, labels


# ============================================================================
# PHASE 2: TRAINING WITH PRE-EXTRACTED FEATURES
# ============================================================================

class PreExtractedDataset(torch.utils.data.Dataset):
    """Dataset for pre-extracted features"""

    def __init__(self, visual_A, visual_B, geometric, labels):
        self.visual_A = torch.from_numpy(visual_A).float()
        self.visual_B = torch.from_numpy(visual_B).float()
        self.geometric = torch.from_numpy(geometric).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'visual_A': self.visual_A[idx],
            'visual_B': self.visual_B[idx],
            'geometric': self.geometric[idx],
            'label': self.labels[idx]
        }


class RelationNetworkTrainer:
    """Trainer for Relation Network with pre-extracted features"""

    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Create full model
        # Note: Backbone is already frozen via config.freeze_blocks=99 in create_resnet_stage2_model
        print("\nCreating model...")
        self.model = create_resnet_stage2_model(config).to(device)

        # Verify backbone is frozen
        if hasattr(self.model, 'backbone'):
            backbone_trainable = sum(p.numel() for p in self.model.backbone.parameters() if p.requires_grad)
            backbone_total = sum(p.numel() for p in self.model.backbone.parameters())
            print(f"Backbone: {backbone_total:,} params, {backbone_trainable:,} trainable")

        print(f"\nModel initialized")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # Create loss
        from models.resnet_stage2_classifier import ResNetStage2Loss
        class_weights = getattr(config, 'class_weights', None)
        gamma = getattr(config, 'focal_gamma', 2.0)  # Default gamma=2.0 for Focal Loss
        self.criterion = ResNetStage2Loss(
            class_weights=class_weights,
            gamma=gamma
        ).to(device)

        # Create optimizer (only for relation network)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6
        )

        # Training state
        self.best_val_acc = 0.0
        self.best_val_mpca = 0.0
        self.epochs_without_improvement = 0

        # Checkpoint dir
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        class_names = ['Walking Together', 'Standing Together', 'Sitting Together']
        evaluator = Stage2Evaluator(class_names)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:
            visual_A = batch['visual_A'].to(self.device)
            visual_B = batch['visual_B'].to(self.device)
            geometric = batch['geometric'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(visual_A, visual_B, geometric)

            # Compute loss
            loss, loss_dict = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            evaluator.update(predictions.cpu().numpy(), labels.cpu().numpy())

        pbar.close()

        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total
        train_metrics = evaluator.compute_metrics()
        train_mpca = train_metrics.get('mpca', 0.0)

        print(f"Train Epoch {epoch}: Loss={avg_loss:.6f}, Acc={avg_acc:.4f}, MPCA={train_mpca:.4f}")

        return avg_loss, avg_acc, train_mpca, train_metrics

    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch"""
        self.model.eval()

        total_loss = 0.0

        class_names = ['Walking Together', 'Standing Together', 'Sitting Together']
        evaluator = Stage2Evaluator(class_names)

        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ")

        with torch.no_grad():
            for batch in pbar:
                visual_A = batch['visual_A'].to(self.device)
                visual_B = batch['visual_B'].to(self.device)
                geometric = batch['geometric'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                logits = self.model(visual_A, visual_B, geometric)
                loss, _ = self.criterion(logits, labels)

                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)
                evaluator.update(predictions.cpu().numpy(), labels.cpu().numpy())

        pbar.close()

        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        val_metrics = evaluator.compute_metrics()
        val_acc = val_metrics.get('overall_accuracy', 0.0)
        val_mpca = val_metrics.get('mpca', 0.0)

        print(f"Val Epoch {epoch}: Loss={avg_loss:.6f}, Acc={val_acc:.4f}, MPCA={val_mpca:.4f}")

        return avg_loss, val_acc, val_mpca, val_metrics

    def test_epoch(self, test_loader):
        """Test the model"""
        self.model.eval()

        class_names = ['Walking Together', 'Standing Together', 'Sitting Together']
        evaluator = Stage2Evaluator(class_names)

        print(f"\n{'='*80}")
        print("TESTING MODEL")
        print(f"{'='*80}")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                visual_A = batch['visual_A'].to(self.device)
                visual_B = batch['visual_B'].to(self.device)
                geometric = batch['geometric'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(visual_A, visual_B, geometric)
                predictions = torch.argmax(logits, dim=1)

                evaluator.update(predictions.cpu().numpy(), labels.cpu().numpy())

        # Print detailed results
        evaluator.print_evaluation_report()

        test_metrics = evaluator.compute_metrics()
        test_acc = test_metrics.get('overall_accuracy', 0.0)
        test_mpca = test_metrics.get('mpca', 0.0)

        return test_acc, test_mpca, test_metrics

    def train(self, train_loader, val_loader, test_loader=None):
        """Main training loop"""
        print(f"\n{'='*80}")
        print("STARTING TRAINING")
        print(f"{'='*80}\n")

        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss, train_acc, train_mpca, train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc, val_mpca, val_metrics = self.validate_epoch(val_loader, epoch)

            # Step scheduler
            self.scheduler.step()

            # Check for improvement
            improved = False
            if val_mpca > self.best_val_mpca:
                self.best_val_mpca = val_mpca
                self.best_val_acc = val_acc
                improved = True
                self.epochs_without_improvement = 0

                # Save best model
                checkpoint_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'best_val_mpca': self.best_val_mpca,
                    'config': self.config.__dict__
                }, checkpoint_path)
                print(f"✓ Saved best model (MPCA={self.best_val_mpca:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\nEarly stopping after {self.config.early_stopping_patience} epochs without improvement")
                break

            print(f"Best Val MPCA: {self.best_val_mpca:.4f}, Epochs w/o improvement: {self.epochs_without_improvement}")

        # Test on best model
        if test_loader is not None:
            print(f"\n{'='*80}")
            print("FINAL TEST EVALUATION")
            print(f"{'='*80}")

            # Load best model
            checkpoint_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint['epoch']}")

            test_acc, test_mpca, test_metrics = self.test_epoch(test_loader)

            print(f"\nTest Results: Acc={test_acc:.4f}, MPCA={test_mpca:.4f}")

        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Best Val MPCA: {self.best_val_mpca:.4f}")
        print(f"Best Val Acc: {self.best_val_acc:.4f}")


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
    Calculate inference FLOPs for deployment device evaluation.

    Args:
        config: Model configuration
        device: torch device
    """
    if not THOP_AVAILABLE:
        print("Error: thop library not available. Install with 'pip install thop'")
        return

    print(f"\n{'='*80}")
    print("INFERENCE FLOPs ANALYSIS (for deployment device evaluation)")
    print(f"{'='*80}")

    # ========================================================================
    # 1. Backbone FLOPs (Feature Extraction)
    # ========================================================================
    print(f"\n[1] BACKBONE: {config.backbone_name}")
    print("-" * 40)

    # Create backbone model
    full_model = create_resnet_stage2_model(config).to(device)
    backbone = full_model.backbone
    backbone.eval()

    # Create dummy input for backbone
    crop_size = getattr(config, 'crop_size', 224)
    dummy_image = torch.randn(1, 3, crop_size, crop_size).to(device)

    try:
        backbone_flops, backbone_params = profile(backbone, inputs=(dummy_image,), verbose=False)

        print(f"  Input size: {crop_size}x{crop_size}")
        print(f"  FLOPs: {format_number(backbone_flops)}FLOPs / image")
        print(f"  Parameters: {format_number(backbone_params)}")
    except Exception as e:
        print(f"  Error calculating backbone FLOPs: {e}")
        backbone_flops = 0
        backbone_params = 0

    # ========================================================================
    # 2. Relation Network FLOPs (Classification)
    # ========================================================================
    print(f"\n[2] RELATION NETWORK")
    print("-" * 40)

    # Get dimensions
    visual_dim = config.visual_feature_dim
    spatial_dim = config.spatial_feature_dim

    print(f"  Feature mode: {config.feature_mode}")
    if config.feature_mode == 'none':
        print(f"  Spatial features: None (visual only)")
    else:
        print(f"  Spatial features: {spatial_dim}D")

    # Create a wrapper for relation network forward pass
    class RelationNetworkWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, visual_A, visual_B, geometric):
            return self.model(visual_A, visual_B, geometric)

    relation_wrapper = RelationNetworkWrapper(full_model).to(device)
    relation_wrapper.eval()

    # Create dummy inputs for relation network
    dummy_visual_A = torch.randn(1, visual_dim).to(device)
    dummy_visual_B = torch.randn(1, visual_dim).to(device)
    # Handle 'none' mode with empty geometric tensor
    if spatial_dim > 0:
        dummy_geometric = torch.randn(1, spatial_dim).to(device)
    else:
        dummy_geometric = torch.zeros(1, 0).to(device)

    try:
        relation_flops, relation_params = profile(
            relation_wrapper,
            inputs=(dummy_visual_A, dummy_visual_B, dummy_geometric),
            verbose=False
        )

        print(f"  FLOPs: {format_number(relation_flops)}FLOPs / sample")
        print(f"  Parameters: {format_number(relation_params)}")
    except Exception as e:
        print(f"  Error calculating relation network FLOPs: {e}")
        relation_flops = 0
        relation_params = 0

    # ========================================================================
    # 3. Total Inference FLOPs per Sample
    # ========================================================================
    print(f"\n[3] TOTAL INFERENCE (per sample)")
    print("-" * 40)

    if backbone_flops > 0 and relation_flops > 0:
        # Total = Backbone × 2 (Person A + B) + Relation Network
        backbone_per_sample = backbone_flops * 2
        total_inference_flops = backbone_per_sample + relation_flops
        total_params = backbone_params + relation_params

        print(f"  Backbone x2 (Person A+B): {format_number(backbone_per_sample)}FLOPs")
        print(f"  Relation Network:         {format_number(relation_flops)}FLOPs")
        print(f"  ─────────────────────────────────────")
        print(f"  Total per sample:         {format_number(total_inference_flops)}FLOPs")
        print(f"  Total parameters:         {format_number(total_params)}")
    else:
        total_inference_flops = 0
        total_params = 0

    # ========================================================================
    # 4. Throughput Estimation (Reference)
    # ========================================================================
    print(f"\n[4] THROUGHPUT ESTIMATION (reference)")
    print("-" * 40)

    if total_inference_flops > 0:
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
            theoretical_fps = gpu_tflops / total_inference_flops
            print(f"    {gpu_name:20s}: ~{theoretical_fps:,.0f} samples/sec")

        print(f"\n  Note: Real throughput is typically 10-30% of theoretical max")
        print(f"        due to memory bandwidth, data loading, etc.")

    # ========================================================================
    # 5. Summary
    # ========================================================================
    print(f"\n[5] CONFIGURATION SUMMARY")
    print("-" * 40)
    print(f"  Backbone:      {config.backbone_name}")
    print(f"  Feature mode:  {config.feature_mode}")
    print(f"  Visual dim:    {visual_dim}")
    print(f"  Spatial dim:   {spatial_dim}")
    print(f"  Input size:    {crop_size}x{crop_size}")

    print(f"\n{'='*80}")

    # Clean up
    del full_model, backbone, relation_wrapper
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Relation Network with Frozen Backbone + 10D Geometric Features')

    # Model arguments
    parser.add_argument('--backbone', type=str, default='efficientnet_v2_s',
                       choices=['resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                               'alexnet', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_v2_s',
                               'litehrnet_18', 'hrnet_w18', 'hrnet_w32', 'hrnet_w48'],
                       help='CNN backbone architecture (ResNet/VGG/AlexNet/MobileNet/EfficientNetV2/Lite-HRNet/HRNet, will be frozen)')
    parser.add_argument('--visual_dim', type=int, default=256,
                       help='Visual feature dimension')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')

    # Data arguments
    parser.add_argument('--data_path', type=str, default='../dataset',
                       help='Path to dataset')
    parser.add_argument('--frame_interval', type=int, default=1,
                       help='Frame sampling interval')

    # Other arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/opfw_geo_stage2',
                       help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling')

    # Feature extraction
    parser.add_argument('--skip_extraction', action='store_true',
                       help='Skip feature extraction (load from saved files)')
    parser.add_argument('--features_dir', type=str, default=None,
                       help='Directory to save/load extracted features')

    # Feature selection
    parser.add_argument('--feature_mode', type=str, default='both',
                       choices=['both', 'opticalflow_only', 'bboxposition_only', 'none'],
                       help='Feature mode: both (10D all), opticalflow_only (5D motion), bboxposition_only (5D geometry), none (visual only)')

    # FLOPs calculation
    parser.add_argument('--flops_only', action='store_true',
                       help='Only calculate FLOPs without training')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Create config and set ALL parameters BEFORE creating any models
    print(f"\nConfiguring model for: {args.backbone}")
    config = create_backbone_config(args.backbone)

    # Model architecture
    config.backbone_name = args.backbone
    config.visual_feature_dim = args.visual_dim

    # Feature selection mode
    config.feature_mode = args.feature_mode

    # Set spatial feature dimension based on feature_mode
    if config.feature_mode == 'both':
        config.spatial_feature_dim = 10  # 9D geometric + 1D sync
    elif config.feature_mode == 'opticalflow_only':
        config.spatial_feature_dim = 5   # 4D flow + 1D sync
    elif config.feature_mode == 'bboxposition_only':
        config.spatial_feature_dim = 5   # 5D bbox (no sync)
    elif config.feature_mode == 'none':
        config.spatial_feature_dim = 0   # No geometric features, visual only

    # Training parameters
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay

    # Data parameters
    config.data_path = args.data_path
    config.frame_interval = args.frame_interval
    config.checkpoint_dir = args.checkpoint_dir

    # CRITICAL: Freeze backbone completely (only train classification head)
    config.freeze_backbone = True
    config.freeze_blocks = 99  # Full freeze

    # Feature settings
    config.use_geometric = False  # We use 10D geometric instead
    config.use_scene_context = False
    config.use_optical_flow = True  # Required for geometric extraction

    # Add num_classes if not present
    if not hasattr(config, 'num_classes'):
        config.num_classes = 3  # Walking Together, Standing Together, Sitting Together

    print(f"\nConfig: {config.backbone_name} | {config.feature_mode} ({config.spatial_feature_dim}D) | "
          f"bs={config.batch_size} lr={config.learning_rate} epochs={config.epochs}")

    # ========================================================================
    # FLOPs ONLY MODE
    # ========================================================================
    if args.flops_only:
        calculate_flops(config, device)
        print("\n✓ FLOPs calculation complete. Exiting without training.")
        return

    # Feature directory
    features_dir = args.features_dir or os.path.join(config.checkpoint_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)

    # ========================================================================
    # PHASE 1: FEATURE EXTRACTION (or load from disk)
    # ========================================================================

    if args.skip_extraction:
        print(f"\n{'='*80}")
        print("LOADING PRE-EXTRACTED FEATURES")
        print(f"{'='*80}")

        # Load features from disk
        train_visual_A = np.load(os.path.join(features_dir, 'train_visual_A.npy'))
        train_visual_B = np.load(os.path.join(features_dir, 'train_visual_B.npy'))
        train_geometric = np.load(os.path.join(features_dir, 'train_geometric.npy'))
        train_labels = np.load(os.path.join(features_dir, 'train_labels.npy'))

        val_visual_A = np.load(os.path.join(features_dir, 'val_visual_A.npy'))
        val_visual_B = np.load(os.path.join(features_dir, 'val_visual_B.npy'))
        val_geometric = np.load(os.path.join(features_dir, 'val_geometric.npy'))
        val_labels = np.load(os.path.join(features_dir, 'val_labels.npy'))

        test_visual_A = np.load(os.path.join(features_dir, 'test_visual_A.npy'))
        test_visual_B = np.load(os.path.join(features_dir, 'test_visual_B.npy'))
        test_geometric = np.load(os.path.join(features_dir, 'test_geometric.npy'))
        test_labels = np.load(os.path.join(features_dir, 'test_labels.npy'))

        print("✓ Features loaded from disk")
    else:
        # Create datasets
        print(f"\n{'='*80}")
        print("LOADING DATASETS")
        print(f"{'='*80}")

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
        extractor = OfflineFeatureExtractor(config, device)

        train_visual_A, train_visual_B, train_geometric, train_labels = extractor.extract_all_features(train_dataset, 'train')
        val_visual_A, val_visual_B, val_geometric, val_labels = extractor.extract_all_features(val_dataset, 'val')
        test_visual_A, test_visual_B, test_geometric, test_labels = extractor.extract_all_features(test_dataset, 'test')

        # Save features to disk
        print(f"\nSaving features to {features_dir}...")
        np.save(os.path.join(features_dir, 'train_visual_A.npy'), train_visual_A)
        np.save(os.path.join(features_dir, 'train_visual_B.npy'), train_visual_B)
        np.save(os.path.join(features_dir, 'train_geometric.npy'), train_geometric)
        np.save(os.path.join(features_dir, 'train_labels.npy'), train_labels)

        np.save(os.path.join(features_dir, 'val_visual_A.npy'), val_visual_A)
        np.save(os.path.join(features_dir, 'val_visual_B.npy'), val_visual_B)
        np.save(os.path.join(features_dir, 'val_geometric.npy'), val_geometric)
        np.save(os.path.join(features_dir, 'val_labels.npy'), val_labels)

        np.save(os.path.join(features_dir, 'test_visual_A.npy'), test_visual_A)
        np.save(os.path.join(features_dir, 'test_visual_B.npy'), test_visual_B)
        np.save(os.path.join(features_dir, 'test_geometric.npy'), test_geometric)
        np.save(os.path.join(features_dir, 'test_labels.npy'), test_labels)

        print("✓ Features saved to disk")

    # ========================================================================
    # PHASE 2: TRAINING
    # ========================================================================

    # Create datasets
    train_dataset = PreExtractedDataset(train_visual_A, train_visual_B, train_geometric, train_labels)
    val_dataset = PreExtractedDataset(val_visual_A, val_visual_B, val_geometric, val_labels)
    test_dataset = PreExtractedDataset(test_visual_A, test_visual_B, test_geometric, test_labels)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    # Create trainer
    trainer = RelationNetworkTrainer(config, device)

    # Train
    start_time = time.time()
    trainer.train(train_loader, val_loader, test_loader)

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s")

    print("\n✓ All done!")


if __name__ == '__main__':
    main()
