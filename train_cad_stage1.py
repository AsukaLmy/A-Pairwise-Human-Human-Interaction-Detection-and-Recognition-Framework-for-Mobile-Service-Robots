"""
Training script for CAD Stage1: Binary Interaction Detection

Uses 7D geometric features with AdaptiveGeometricClassifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
import json
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

from datasets.cad_geometric_stage1 import CADGeometricStage1Dataset, cad_stage1_collate_fn
from src.classifiers.geometric_classifier import AdaptiveGeometricClassifier
from src.losses.focal_loss import FocalLoss, AdaptiveFocalLoss


# ============================================================================
# Social Group Clustering and Evaluation Functions
# ============================================================================

class UnionFind:
    """Union-Find data structure for connected components"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union by rank"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_groups(self):
        """Get all connected components as groups"""
        groups_dict = defaultdict(set)
        for i in range(len(self.parent)):
            groups_dict[self.find(i)].add(i)
        return list(groups_dict.values())


def cluster_social_groups(interactions, threshold=0.5):
    """
    Cluster people into social groups based on pairwise interaction probabilities

    Args:
        interactions: dict {(person_i, person_j): probability}
        threshold: probability threshold for connecting people

    Returns:
        groups: List[Set] - list of social groups
    """
    # Get all unique persons
    all_persons = set()
    for (person_i, person_j) in interactions.keys():
        all_persons.add(person_i)
        all_persons.add(person_j)

    if len(all_persons) == 0:
        return []

    # Create person_id to index mapping
    person_list = sorted(list(all_persons))
    person_to_idx = {p: i for i, p in enumerate(person_list)}

    # Initialize Union-Find
    uf = UnionFind(len(person_list))

    # Connect people with interaction probability > threshold
    for (person_i, person_j), prob in interactions.items():
        if prob > threshold:
            idx_i = person_to_idx[person_i]
            idx_j = person_to_idx[person_j]
            uf.union(idx_i, idx_j)

    # Get groups (in index space)
    groups_idx = uf.get_groups()

    # Convert back to person IDs
    groups = []
    for group_idx in groups_idx:
        group = {person_list[idx] for idx in group_idx}
        groups.append(group)

    return groups


def compute_membership_accuracy(pred_groups, gt_groups):
    """
    Compute membership accuracy using Hungarian algorithm

    Args:
        pred_groups: List[Set] - predicted social groups
        gt_groups: List[Set] - ground truth social groups

    Returns:
        accuracy: float - membership accuracy [0, 1]
        correct: int - number of correctly assigned persons
        total: int - total number of persons
    """
    # Get all persons
    all_persons = set()
    for group in pred_groups + gt_groups:
        all_persons.update(group)

    if len(all_persons) == 0:
        return 1.0, 0, 0

    n_pred = len(pred_groups)
    n_gt = len(gt_groups)

    if n_pred == 0 or n_gt == 0:
        return 0.0, 0, len(all_persons)

    n_max = max(n_pred, n_gt)

    # Build cost matrix (number of mismatched persons)
    cost_matrix = np.zeros((n_max, n_max))

    for i in range(n_pred):
        for j in range(n_gt):
            intersection = len(pred_groups[i] & gt_groups[j])
            union = len(pred_groups[i] | gt_groups[j])
            cost_matrix[i, j] = union - intersection

    # Fill dummy groups
    if n_pred < n_max:
        for i in range(n_pred, n_max):
            for j in range(n_gt):
                cost_matrix[i, j] = len(gt_groups[j])

    if n_gt < n_max:
        for i in range(n_pred):
            for j in range(n_gt, n_max):
                cost_matrix[i, j] = len(pred_groups[i])

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Count correct assignments
    correct_assignments = 0
    for i, j in zip(row_ind, col_ind):
        if i < n_pred and j < n_gt:
            correct_assignments += len(pred_groups[i] & gt_groups[j])

    total_persons = len(all_persons)
    accuracy = correct_assignments / total_persons if total_persons > 0 else 0

    return accuracy, correct_assignments, total_persons


def compute_group_map(pred_groups, gt_groups, iou_threshold=0.5):
    """
    Compute mean Average Precision for group detection

    Args:
        pred_groups: List[Set] - predicted groups
        gt_groups: List[Set] - ground truth groups
        iou_threshold: IoU threshold for matching groups

    Returns:
        map_score: float - mean Average Precision
        precision: float - group-level precision
        recall: float - group-level recall
    """
    def compute_iou(group1, group2):
        if len(group1) == 0 and len(group2) == 0:
            return 1.0
        intersection = len(group1 & group2)
        union = len(group1 | group2)
        return intersection / union if union > 0 else 0

    if len(pred_groups) == 0 or len(gt_groups) == 0:
        return 0.0, 0.0, 0.0

    # Match predicted groups to GT groups
    matched_pred = set()
    matched_gt = set()

    for i, pred_group in enumerate(pred_groups):
        best_iou = 0
        best_j = -1

        for j, gt_group in enumerate(gt_groups):
            iou = compute_iou(pred_group, gt_group)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(best_j)

    # Compute metrics
    num_correct = len(matched_pred)
    precision = num_correct / len(pred_groups)
    recall = num_correct / len(gt_groups)

    # mAP (simplified as F1 for group detection)
    if precision + recall > 0:
        map_score = 2 * precision * recall / (precision + recall)
    else:
        map_score = 0.0

    return map_score, precision, recall


# ============================================================================
# Trainer Class
# ============================================================================

class CADStage1Trainer:
    """
    Trainer for CAD Stage1 binary interaction detection
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and not config.cpu else 'cpu')

        # Create save directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join('checkpoints', f'cad_stage1_{timestamp}')
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize model
        self._initialize_model()

        # Setup training components
        self._setup_training()

        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0

        # Save configuration
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)

        print(f"CADStage1Trainer initialized on {self.device}")
        print(f"Save directory: {self.save_dir}")

    def _initialize_model(self):
        """Initialize the AdaptiveGeometricClassifier (7D features)"""
        self.model = AdaptiveGeometricClassifier(
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout
        )

        self.model = self.model.to(self.device)

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: AdaptiveGeometricClassifier (7D features)")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def _setup_training(self):
        """Setup optimizer, criterion, scheduler"""
        # Loss function
        if self.config.loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma
            )
            print(f"Using Focal Loss (alpha={self.config.focal_alpha}, gamma={self.config.focal_gamma})")
        elif self.config.loss_type == 'adaptive_focal':
            self.criterion = AdaptiveFocalLoss(
                num_classes=2,
                gamma=self.config.focal_gamma,
                auto_alpha=True
            )
            print(f"Using Adaptive Focal Loss (gamma={self.config.focal_gamma}, auto_alpha=True)")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("Using Cross Entropy Loss")

        # Optimizer
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Scheduler
        if self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.step_size, gamma=0.5
            )
        elif self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            self.scheduler = None

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        all_predictions = []
        all_targets = []

        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            geometric_features = batch['geometric_features'].to(self.device)
            targets = batch['stage1_label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(geometric_features)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Print progress
            if (batch_idx + 1) % self.config.print_freq == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = accuracy_score(all_targets, all_predictions)
                print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] '
                      f'Loss: {avg_loss:.4f} Acc: {accuracy:.4f}')

        # Epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = accuracy_score(all_targets, all_predictions)
        epoch_f1 = f1_score(all_targets, all_predictions, average='binary')

        return epoch_loss, epoch_accuracy, epoch_f1

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()

        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                geometric_features = batch['geometric_features'].to(self.device)
                targets = batch['stage1_label'].to(self.device)

                # Forward pass
                outputs = self.model(geometric_features)
                loss = self.criterion(outputs, targets)

                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Compute metrics
        val_loss = total_loss / len(val_loader)
        val_accuracy = accuracy_score(all_targets, all_predictions)
        val_f1 = f1_score(all_targets, all_predictions, average='binary')

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)

        return val_loss, val_accuracy, val_f1, cm

    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("\nStarting training...")

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validate
            val_loss, val_acc, val_f1, val_cm = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f'\nEpoch [{epoch + 1}/{self.config.epochs}] ({epoch_time:.1f}s)')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
            print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}')
            print(f'  Confusion Matrix:\n{val_cm}')

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'  Learning Rate: {current_lr:.6f}')

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.epochs_without_improvement = 0

                save_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'train_acc': train_acc,
                    'train_f1': train_f1,
                    'config': vars(self.config)
                }, save_path)
                print(f'  *** Best model saved (Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}) ***')
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.config.early_stopping > 0 and self.epochs_without_improvement >= self.config.early_stopping:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break

        # Save final model
        final_path = os.path.join(self.save_dir, 'final_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
            'config': vars(self.config)
        }, final_path)

        print(f'\nTraining completed!')
        print(f'Best validation accuracy: {self.best_val_acc:.4f}')
        print(f'Best validation F1: {self.best_val_f1:.4f}')

        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    def train_no_val(self, train_loader):
        """Training loop without validation (fixed epochs, no early stopping)"""
        print("\nStarting training (no validation mode)...")
        print(f"Training for {self.config.epochs} epochs without validation")

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f'\nEpoch [{epoch + 1}/{self.config.epochs}] ({epoch_time:.1f}s)')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')

            # Learning rate scheduling (use train loss for plateau scheduler)
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'  Learning Rate: {current_lr:.6f}')

        # Save final model as best model (since no validation to select best)
        final_path = os.path.join(self.save_dir, 'best_model.pth')
        torch.save({
            'epoch': self.config.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_acc': train_acc,
            'train_f1': train_f1,
            'config': vars(self.config)
        }, final_path)
        print(f'\nFinal model saved as best_model.pth')

        print(f'\nTraining completed! (no validation mode)')
        print(f'Final training accuracy: {train_acc:.4f}')
        print(f'Final training F1: {train_f1:.4f}')

        # Save training history
        history = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'mode': 'no_val'
        }
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    def test(self, test_loader):
        """Test the model on test set with social group clustering evaluation"""
        print(f'\n{"=" * 80}')
        print('Testing on Test Set with Social Group Clustering')
        print(f'{"=" * 80}')

        # Load best model
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print(f'Loading best model from: {best_model_path}')
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # Handle both val and no_val modes
            if 'val_acc' in checkpoint:
                print(f'Best model from epoch {checkpoint["epoch"]} (Val Acc: {checkpoint["val_acc"]:.4f})')
            else:
                print(f'Final model from epoch {checkpoint["epoch"]} (Train Acc: {checkpoint["train_acc"]:.4f})')
        else:
            print('Warning: Best model not found, using current model')

        self.model.eval()

        # Collect frame-level data for clustering
        frame_data = defaultdict(lambda: {
            'track_ids': [],
            'features': [],
            'gt_groups': []
        })

        print('\nCollecting frame-level predictions...')
        with torch.no_grad():
            for batch in test_loader:
                seq_nums = batch['sequence']
                frame_ids = batch['frame']
                geometric_features = batch['geometric_features'].to(self.device)
                track_id_A = batch['track_id_A']
                track_id_B = batch['track_id_B']
                social_group_A = batch['social_group_id_A']
                social_group_B = batch['social_group_id_B']

                # Forward pass to get logits
                logits = self.model(geometric_features)

                # Convert logits to probabilities
                probs = F.softmax(logits, dim=1)
                interaction_probs = probs[:, 1].cpu().numpy()

                # Group by frame
                for i in range(len(seq_nums)):
                    frame_key = (seq_nums[i].item(), frame_ids[i].item())

                    # Store interaction probability for this pair
                    pair_key = tuple(sorted([track_id_A[i].item(), track_id_B[i].item()]))

                    if 'interactions' not in frame_data[frame_key]:
                        frame_data[frame_key]['interactions'] = {}

                    frame_data[frame_key]['interactions'][pair_key] = interaction_probs[i]

                    # Store track IDs and GT groups
                    track_a = track_id_A[i].item()
                    track_b = track_id_B[i].item()
                    group_a = social_group_A[i].item()
                    group_b = social_group_B[i].item()

                    if 'track_to_group' not in frame_data[frame_key]:
                        frame_data[frame_key]['track_to_group'] = {}
                    frame_data[frame_key]['track_to_group'][track_a] = group_a
                    frame_data[frame_key]['track_to_group'][track_b] = group_b

        print(f'Collected data for {len(frame_data)} frames')

        # Test different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        print(f'\n{"=" * 80}')
        print('Evaluating Social Group Clustering with Different Thresholds')
        print(f'{"=" * 80}')

        threshold_results = []

        for threshold in thresholds:
            print(f'\nEvaluating threshold = {threshold:.1f}...')

            frame_accuracies = []
            frame_maps = []
            total_correct = 0
            total_persons = 0

            for frame_key, data in frame_data.items():
                if 'interactions' not in data or 'track_to_group' not in data:
                    continue

                # Cluster predicted groups
                pred_groups = cluster_social_groups(data['interactions'], threshold)

                # Get GT groups
                track_to_group = data['track_to_group']
                gt_groups_dict = defaultdict(set)
                for track_id, group_id in track_to_group.items():
                    gt_groups_dict[group_id].add(track_id)
                gt_groups = list(gt_groups_dict.values())

                # Compute metrics for this frame
                if len(pred_groups) > 0 and len(gt_groups) > 0:
                    acc, correct, total = compute_membership_accuracy(pred_groups, gt_groups)
                    frame_accuracies.append(acc)
                    total_correct += correct
                    total_persons += total

                    map_score, precision, recall = compute_group_map(pred_groups, gt_groups)
                    frame_maps.append(map_score)

            # Compute overall metrics
            avg_acc = np.mean(frame_accuracies) if frame_accuracies else 0.0
            overall_acc = total_correct / total_persons if total_persons > 0 else 0.0
            avg_map = np.mean(frame_maps) if frame_maps else 0.0

            threshold_results.append({
                'threshold': threshold,
                'membership_acc': overall_acc,
                'avg_frame_acc': avg_acc,
                'map': avg_map,
                'num_frames': len(frame_accuracies)
            })

            print(f'  Membership Acc: {overall_acc:.4f} | Avg Frame Acc: {avg_acc:.4f} | mAP: {avg_map:.4f}')

        # Print results table
        print(f'\n{"=" * 80}')
        print('Social Group Clustering Results (Different Thresholds)')
        print(f'{"=" * 80}')
        print(f'{"Threshold":<12} {"Membership Acc":<18} {"Avg Frame Acc":<18} {"mAP":<12}')
        print(f'{"-" * 80}')

        for result in threshold_results:
            print(f'{result["threshold"]:<12.1f} '
                  f'{result["membership_acc"]:<18.4f} '
                  f'{result["avg_frame_acc"]:<18.4f} '
                  f'{result["map"]:<12.4f}')

        print(f'{"=" * 80}')

        # Find best threshold
        best_result = max(threshold_results, key=lambda x: x['membership_acc'])
        print(f'\nBest Threshold: {best_result["threshold"]:.1f}')
        print(f'  Membership Accuracy: {best_result["membership_acc"]:.4f}')
        print(f'  mAP: {best_result["map"]:.4f}')

        # Save results
        results_file = os.path.join(self.save_dir, 'clustering_results.json')
        with open(results_file, 'w') as f:
            json.dump(threshold_results, f, indent=2)
        print(f'\nClustering results saved to: {results_file}')

        return best_result['membership_acc'], best_result['map']


def main():
    parser = argparse.ArgumentParser(description='CAD Stage1 Training')

    # Dataset parameters
    parser.add_argument('--cad_root', type=str, default='../dataset/cad/ActivityDataset',
                        help='Path to CAD ActivityDataset directory')
    parser.add_argument('--train_sequences', type=str, default='30,31,32,33,34,35,36,37,38,39,40,41,42,43,44',
                        help='Training sequences (e.g., "1-30" or "1,2,3")')
    parser.add_argument('--val_sequences', type=str, default='1,2,3,4,12,13,14,17,18,19,20,21,22,23,24,26',
                        help='Validation sequences')
    parser.add_argument('--test_sequences', type=str, default='5,6,7,8,9,10,11,15,16,25,28,29',
                        help='Test sequences')
    parser.add_argument('--image_width', type=int, default=720,
                        help='CAD image width')
    parser.add_argument('--image_height', type=int, default=480,
                        help='CAD image height')

    # Model parameters
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 16],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=20,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='Early stopping patience (0=disabled)')

    # Loss function parameters
    parser.add_argument('--loss_type', type=str, default='adaptive_focal',
                        choices=['ce', 'focal', 'adaptive_focal'],
                        help='Loss function type (default: adaptive_focal for auto class balancing)')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter (only for --loss_type focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter (focusing parameter, recommended: 2.0)')

    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency during training')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--run_test', action='store_true',
                        help='Run test evaluation after training')
    parser.add_argument('--no_val', action='store_true',
                        help='No validation split: merge train and val into training set, use fixed epochs')

    args = parser.parse_args()

    # Parse sequence ranges
    def parse_sequences(seq_str):
        if '-' in seq_str:
            start, end = map(int, seq_str.split('-'))
            return list(range(start, end + 1))
        else:
            return [int(s) for s in seq_str.split(',')]

    train_seqs = parse_sequences(args.train_sequences)
    val_seqs = parse_sequences(args.val_sequences)
    test_seqs = parse_sequences(args.test_sequences)

    # Handle --no_val mode: merge train and val sequences
    if args.no_val:
        train_seqs = sorted(set(train_seqs + val_seqs))
        val_seqs = []  # No validation set

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("CAD Stage1 Training: Binary Interaction Detection")
    print("=" * 80)
    if args.no_val:
        print(f"Mode: NO VALIDATION (fixed {args.epochs} epochs)")
        print(f"Train sequences (merged): {train_seqs}")
    else:
        print(f"Train sequences: {train_seqs}")
        print(f"Val sequences: {val_seqs}")
    print(f"Test sequences: {test_seqs}")
    print(f"Image dimensions: {args.image_width}x{args.image_height}")
    print()

    # Create datasets
    print("Loading datasets...")
    # Note: Using negative_ratio=0 to keep all samples (Focal Loss handles imbalance)
    train_dataset = CADGeometricStage1Dataset(
        cad_root=args.cad_root,
        split='train',
        sequences=train_seqs,
        image_width=args.image_width,
        image_height=args.image_height,
        negative_ratio=0  # Keep all negative samples, Focal Loss handles class imbalance
    )

    # Create val dataset only if not in no_val mode
    val_dataset = None
    val_loader = None
    if not args.no_val:
        val_dataset = CADGeometricStage1Dataset(
            cad_root=args.cad_root,
            split='val',
            sequences=val_seqs,
            image_width=args.image_width,
            image_height=args.image_height,
            negative_ratio=0  # Keep all negative samples
        )

    # Load test dataset if needed (always load in no_val mode for final evaluation)
    test_dataset = None
    test_loader = None
    if args.run_test or args.no_val:
        test_dataset = CADGeometricStage1Dataset(
            cad_root=args.cad_root,
            split='test',
            sequences=test_seqs,
            image_width=args.image_width,
            image_height=args.image_height,
            negative_ratio=0
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=cad_stage1_collate_fn,
        pin_memory=True
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=cad_stage1_collate_fn,
            pin_memory=True
        )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=cad_stage1_collate_fn,
            pin_memory=True
        )

    print(f"Train batches: {len(train_loader)}")
    if val_loader is not None:
        print(f"Val batches:   {len(val_loader)}")
    if test_loader is not None:
        print(f"Test batches:  {len(test_loader)}")

    # Initialize trainer
    trainer = CADStage1Trainer(args)

    # Train
    if args.no_val:
        # No validation mode: train for fixed epochs, then test
        trainer.train_no_val(train_loader)
        if test_loader is not None:
            print(f"\n{'=' * 80}")
            print("Final Evaluation on Test Set")
            print(f"{'=' * 80}")
            trainer.test(test_loader)
    else:
        # Normal mode with validation
        trainer.train(train_loader, val_loader)
        if args.run_test and test_loader is not None:
            trainer.test(test_loader)

    print("\nTraining complete!")
    print(f"Model saved to: {trainer.save_dir}")


if __name__ == '__main__':
    main()
