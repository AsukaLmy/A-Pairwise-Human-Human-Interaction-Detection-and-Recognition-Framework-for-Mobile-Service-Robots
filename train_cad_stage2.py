"""
Training script for CAD Stage2: Interaction Type Classification

Uses visual features (EfficientNet) + 10D OpGeo features for 6-class classification.
"""

import torch
import torch.nn as nn
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
from collections import Counter, defaultdict

from datasets.cad_resnet_stage2 import CADResNetStage2Dataset, cad_stage2_collate_fn
from models.resnet_stage2_classifier import ResNetRelationStage2Classifier
from src.losses.focal_loss import FocalLoss, AdaptiveFocalLoss


# ============================================================================
# Helper Functions
# ============================================================================

def print_confusion_matrix(cm, class_names, indent=''):
    """
    Print formatted confusion matrix with row/column labels

    Args:
        cm: confusion matrix array
        class_names: list of class names
        indent: string to prepend to each line
    """
    print(f'{indent}Rows = Ground Truth (Actual), Columns = Predicted')
    print(f'{indent}{"" :<12}', end='')
    for name in class_names:
        print(f'{name :<12}', end='')
    print()
    for i, row_name in enumerate(class_names):
        print(f'{indent}{row_name :<12}', end='')
        for j in range(len(class_names)):
            if i < cm.shape[0] and j < cm.shape[1]:
                print(f'{cm[i][j] :<12}', end='')
            else:
                print(f'{0 :<12}', end='')
        print()


# ============================================================================
# Group Activity Voting and Evaluation Functions
# ============================================================================

def merge_moving_activities(activities):
    """
    Merge Crossing(1) and Walking(4) into Moving(1)

    Original 6 classes:
        0: NA, 1: Crossing, 2: Waiting, 3: Queuing, 4: Walking, 5: Talking

    Merged 5 classes:
        0: NA, 1: Moving, 2: Waiting, 3: Queuing, 4: Talking

    Args:
        activities: list of activity labels (0-5)

    Returns:
        merged_activities: list of merged activity labels (0-4)
    """
    merged = []
    for act in activities:
        if act == 1 or act == 4:  # Crossing or Walking → Moving
            merged.append(1)
        elif act == 5:  # Talking → 4 (renumber)
            merged.append(4)
        else:  # 0(NA), 2(Waiting), 3(Queuing) stay the same
            merged.append(act)
    return merged


def compute_and_print_metrics(pred_activities, gt_activities, class_names,
                              group_type_name="All Groups", num_classes=6):
    """
    Compute and print evaluation metrics for a specific group type

    Args:
        pred_activities: list of predicted activities
        gt_activities: list of ground truth activities
        class_names: list of class names
        group_type_name: name of the group type (e.g., "All Groups", "Single-Person", "Multi-Person")
        num_classes: number of classes

    Returns:
        dict with accuracy, map, per_class_ap, per_class_metrics, confusion_matrix
    """
    if len(pred_activities) == 0:
        print(f'\nNo {group_type_name} to evaluate')
        return None

    # Compute overall accuracy
    accuracy = accuracy_score(gt_activities, pred_activities)

    # Compute per-class mAP
    map_score, per_class_ap, per_class_metrics = compute_per_class_map(
        pred_activities, gt_activities, num_classes=num_classes
    )

    # Confusion matrix
    cm = confusion_matrix(gt_activities, pred_activities)

    # Print results
    print(f'\n{"=" * 80}')
    print(f'{group_type_name} - Metrics')
    print(f'{"=" * 80}')
    print(f'  Overall Accuracy: {accuracy:.4f}')
    print(f'  mAP (F1):        {map_score:.4f}')
    print(f'  Total Groups:     {len(pred_activities)}')

    print(f'\n{"Class":<12} {"Precision":<12} {"Recall":<12} {"F1 (AP)":<12} {"Support":<10}')
    print(f'{"-" * 80}')

    for class_id in range(num_classes):
        metrics = per_class_metrics[class_id]
        class_name = class_names[class_id] if class_id < len(class_names) else f'Class{class_id}'
        print(f'{class_name:<12} '
              f'{metrics["precision"]:<12.4f} '
              f'{metrics["recall"]:<12.4f} '
              f'{metrics["f1"]:<12.4f} '
              f'{metrics["support"]:<10}')

    print(f'\nConfusion Matrix:')
    print_confusion_matrix(cm, class_names)

    return {
        'overall_accuracy': accuracy,
        'map': map_score,
        'per_class_ap': {class_names[i]: per_class_ap[i] for i in range(num_classes)},
        'per_class_metrics': {
            class_names[i]: {
                'precision': per_class_metrics[i]['precision'],
                'recall': per_class_metrics[i]['recall'],
                'f1': per_class_metrics[i]['f1'],
                'support': per_class_metrics[i]['support']
            } for i in range(num_classes)
        },
        'confusion_matrix': cm.tolist(),
        'num_groups': len(pred_activities)
    }


def compute_bbox_center(bbox):
    """Compute center of bounding box [x1, y1, x2, y2]"""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def compute_distance(center1, center2):
    """Compute Euclidean distance between two centers"""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def compute_group_center(group_bboxes):
    """Compute center of a group (average of all person centers)"""
    centers = [compute_bbox_center(bbox) for bbox in group_bboxes.values()]
    avg_x = sum(c[0] for c in centers) / len(centers)
    avg_y = sum(c[1] for c in centers) / len(centers)
    return (avg_x, avg_y)


def find_nearest_group(person_bbox, other_groups, group_bboxes):
    """
    Find the nearest group to a single person

    Args:
        person_bbox: bbox of the single person [x1, y1, x2, y2]
        other_groups: list of group IDs
        group_bboxes: dict {group_id: {person_id: bbox}}

    Returns:
        nearest_group_id: ID of the nearest group
    """
    person_center = compute_bbox_center(person_bbox)

    min_distance = float('inf')
    nearest_group = None

    for group_id in other_groups:
        group_center = compute_group_center(group_bboxes[group_id])
        distance = compute_distance(person_center, group_center)

        if distance < min_distance:
            min_distance = distance
            nearest_group = group_id

    return nearest_group


def vote_group_activity(pair_predictions, group_members, person_bboxes, all_groups, group_members_dict):
    """
    Vote for group activity based on pairwise predictions

    Args:
        pair_predictions: dict {(person_i, person_j): activity_class}
        group_members: set of person IDs in this group
        person_bboxes: dict {person_id: bbox} for all persons in frame
        all_groups: list of all group IDs in frame
        group_members_dict: dict {group_id: set of person_ids}

    Returns:
        group_activity: int (0-5)
        confidence: float
        vote_counts: dict for debugging
    """
    # Single person group - Strategy C: find nearest group
    if len(group_members) == 1:
        person_id = list(group_members)[0]
        person_bbox = person_bboxes[person_id]

        # Find other groups (multi-person groups)
        other_groups = [gid for gid in all_groups
                       if len(group_members_dict[gid]) > 1]

        if len(other_groups) == 0:
            # No other groups, default to NA (0)
            return 0, 0.0, {'default': 1}

        # Build group bboxes
        group_bboxes = {}
        for gid in other_groups:
            group_bboxes[gid] = {pid: person_bboxes[pid]
                                for pid in group_members_dict[gid]}

        # Find nearest group
        nearest_group_id = find_nearest_group(person_bbox, other_groups, group_bboxes)

        # Get activity from nearest group
        nearest_members = group_members_dict[nearest_group_id]
        nearest_activity, nearest_conf, _ = vote_group_activity(
            pair_predictions, nearest_members, person_bboxes,
            all_groups, group_members_dict
        )

        return nearest_activity, nearest_conf * 0.5, {'inherited_from_nearest': 1}

    # Multi-person group: collect votes from all pairs within group
    votes = []
    for (p_i, p_j), activity in pair_predictions.items():
        if p_i in group_members and p_j in group_members:
            votes.append(activity)

    if len(votes) == 0:
        # No pairs found (shouldn't happen for multi-person groups)
        return 0, 0.0, {'no_pairs': 1}

    # Majority voting
    vote_counts = Counter(votes)
    group_activity = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[group_activity] / len(votes)

    return group_activity, confidence, dict(vote_counts)


def compute_per_class_map(pred_activities, gt_activities, num_classes=6):
    """
    Compute per-class mAP (mean Average Precision)

    Args:
        pred_activities: list of predicted activities
        gt_activities: list of ground truth activities
        num_classes: number of activity classes

    Returns:
        map_score: overall mAP
        per_class_ap: dict {class_id: AP}
        per_class_metrics: dict with precision, recall, f1 for each class
    """
    per_class_ap = {}
    per_class_metrics = {}

    for class_id in range(num_classes):
        # Binary classification for this class
        pred_binary = [1 if p == class_id else 0 for p in pred_activities]
        gt_binary = [1 if g == class_id else 0 for g in gt_activities]

        tp = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 0 and g == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_ap[class_id] = f1  # Using F1 as AP
        per_class_metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'support': sum(gt_binary)
        }

    # Compute mAP (average of all class APs)
    map_score = np.mean(list(per_class_ap.values()))

    return map_score, per_class_ap, per_class_metrics


# ============================================================================
# Dataset Statistics
# ============================================================================

def print_dataset_statistics(dataset, name, num_classes=6, class_merge=False):
    """Print detailed statistics about the dataset"""
    print(f"\n{'=' * 80}")
    print(f"{name} Dataset Statistics")
    print(f"{'=' * 80}")

    # Count labels
    labels = [dataset[i]['label'].item() for i in range(len(dataset))]

    print(f"Total samples: {len(dataset)}")

    # Per-class distribution
    if class_merge:
        class_names = ['Moving', 'Standing', 'Talking']
    else:
        class_names = ['NA', 'Crossing', 'Waiting', 'Queuing', 'Walking', 'Talking']
    print(f"\nClass Distribution:")
    for c in range(num_classes):
        count = labels.count(c)
        percentage = count / len(dataset) * 100 if len(dataset) > 0 else 0
        class_name = class_names[c] if c < len(class_names) else f'Class{c}'
        print(f"  {c} ({class_name:10s}): {count:6d} ({percentage:5.2f}%)")

    # Get unique sequences and frames
    sequences = set([dataset[i]['sequence'] for i in range(len(dataset))])
    frames = set([dataset[i]['frame'] for i in range(len(dataset))])

    print(f"\nData coverage:")
    print(f"  Sequences: {len(sequences)} ({sorted(sequences)[0]} to {sorted(sequences)[-1]})")
    print(f"  Frames:    {len(frames)} unique frames")
    print(f"  Avg samples per frame: {len(dataset) / len(frames):.2f}" if len(frames) > 0 else "  Avg samples per frame: N/A")
    print(f"{'=' * 80}")


class CADStage2Trainer:
    """
    Trainer for CAD Stage2: 6-class interaction type classification
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and not config.cpu else 'cpu')
        self.class_merge = getattr(config, 'class_merge', False)

        # Set class names based on class_merge
        if self.class_merge:
            self.class_names = ['Moving', 'Standing', 'Talking']
        else:
            self.class_names = ['NA', 'Crossing', 'Waiting', 'Queuing', 'Walking', 'Talking']

        # Create save directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join('checkpoints', f'cad_stage2_{timestamp}')
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

        print(f"CADStage2Trainer initialized on {self.device}")
        print(f"Save directory: {self.save_dir}")

    def _initialize_model(self):
        """Initialize the ResNet Stage2 classifier"""
        self.model = ResNetRelationStage2Classifier(
            person_feature_dim=self.config.visual_dim,
            spatial_feature_dim=self.config.spatial_dim,
            hidden_dims=self.config.relation_hidden_dims,
            dropout=self.config.dropout,
            fusion_strategy='concat',
            backbone_name='resnet18',
            pretrained=True,
            freeze_backbone=True,  # Backbone already frozen in dataset
            num_classes=self.config.num_classes
        )

        self.model = self.model.to(self.device)

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: ResNetRelationStage2Classifier")
        print(f"  Visual dim: {self.config.visual_dim}")
        print(f"  Spatial dim: {self.config.spatial_dim}")
        print(f"  Num classes: {self.config.num_classes}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

    def _setup_training(self):
        """Setup optimizer, criterion, scheduler"""
        # Loss function
        if self.config.loss_type == 'focal':
            if self.config.class_weights:
                # Use provided class weights as alpha
                self.criterion = FocalLoss(
                    alpha=self.config.class_weights,
                    gamma=self.config.focal_gamma
                )
                print(f"Using Focal Loss with class weights: {self.config.class_weights}, gamma={self.config.focal_gamma}")
            else:
                self.criterion = FocalLoss(
                    alpha=self.config.focal_alpha,
                    gamma=self.config.focal_gamma
                )
                print(f"Using Focal Loss (alpha={self.config.focal_alpha}, gamma={self.config.focal_gamma})")
        elif self.config.loss_type == 'adaptive_focal':
            self.criterion = AdaptiveFocalLoss(
                num_classes=self.config.num_classes,
                gamma=self.config.focal_gamma,
                auto_alpha=True
            )
            print(f"Using Adaptive Focal Loss (num_classes={self.config.num_classes}, gamma={self.config.focal_gamma}, auto_alpha=True)")
        else:
            # Cross Entropy Loss
            if self.config.class_weights:
                class_weights = torch.FloatTensor(self.config.class_weights).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
                print(f"Using Cross Entropy Loss with class weights: {self.config.class_weights}")
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
            visual_A = batch['visual_A'].to(self.device)
            visual_B = batch['visual_B'].to(self.device)
            geometric = batch['geometric'].to(self.device)
            targets = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(visual_A, visual_B, geometric)

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
        epoch_f1 = f1_score(all_targets, all_predictions, average='weighted')

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
                visual_A = batch['visual_A'].to(self.device)
                visual_B = batch['visual_B'].to(self.device)
                geometric = batch['geometric'].to(self.device)
                targets = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(visual_A, visual_B, geometric)
                loss = self.criterion(outputs, targets)

                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Compute metrics
        val_loss = total_loss / len(val_loader)
        val_accuracy = accuracy_score(all_targets, all_predictions)
        val_f1 = f1_score(all_targets, all_predictions, average='weighted')

        # Confusion matrix and classification report
        cm = confusion_matrix(all_targets, all_predictions)

        # Get unique labels present in predictions and targets
        unique_labels = sorted(list(set(all_targets) | set(all_predictions)))
        target_names_present = [self.class_names[i] if i < len(self.class_names) else f'Class{i}'
                               for i in unique_labels]

        report = classification_report(all_targets, all_predictions,
                                      labels=unique_labels,
                                      target_names=target_names_present,
                                      digits=4,
                                      zero_division=0)

        return val_loss, val_accuracy, val_f1, cm, report

    def train(self, train_loader, val_loader):
        """Main training loop"""

        for epoch in range(self.config.epochs):
            print(f"\n{'=' * 80}")
            print(f"Epoch [{epoch + 1}/{self.config.epochs}]")
            print(f"{'=' * 80}")

            epoch_start = time.time()

            # Train
            print(f"\nTraining...")
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validate
            print(f"\nValidating...")
            val_loss, val_acc, val_f1, val_cm, val_report = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f'\n{"=" * 80}')
            print(f'Epoch [{epoch + 1}/{self.config.epochs}] Summary (Time: {epoch_time:.1f}s)')
            print(f'{"=" * 80}')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
            print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}')
            print(f'\n  Confusion Matrix:')
            print_confusion_matrix(val_cm, self.class_names, indent='  ')

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

                # Save classification report
                with open(os.path.join(self.save_dir, 'best_classification_report.txt'), 'w') as f:
                    f.write(val_report)
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

        print(f'\n{"=" * 80}')
        print(f'Training Completed!')
        print(f'{"=" * 80}')
        print(f'Best validation accuracy: {self.best_val_acc:.4f}')
        print(f'Best validation F1: {self.best_val_f1:.4f}')
        print(f'Models saved to: {self.save_dir}')

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
        print("Starting training (no validation mode)...")
        print(f"Training for {self.config.epochs} epochs without validation")

        for epoch in range(self.config.epochs):
            print(f"\n{'=' * 80}")
            print(f"Epoch [{epoch + 1}/{self.config.epochs}]")
            print(f"{'=' * 80}")

            epoch_start = time.time()

            # Train
            print(f"\nTraining...")
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f'\n{"=" * 80}')
            print(f'Epoch [{epoch + 1}/{self.config.epochs}] Summary (Time: {epoch_time:.1f}s)')
            print(f'{"=" * 80}')
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

        print(f'\n{"=" * 80}')
        print(f'Training Completed! (no validation mode)')
        print(f'{"=" * 80}')
        print(f'Final training accuracy: {train_acc:.4f}')
        print(f'Final training F1: {train_f1:.4f}')
        print(f'Model saved to: {self.save_dir}')

        # Save training history
        history = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'mode': 'no_val'
        }
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    def test(self, test_loader):
        """Test the model on test set with group activity evaluation"""
        print(f'\n{"=" * 80}')
        print('Testing on Test Set with Group Activity Recognition')
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

        # Collect frame-level data
        frame_data = defaultdict(lambda: {
            'pair_predictions': {},  # {(track_i, track_j): activity_class}
            'person_bboxes': {},     # {track_id: bbox}
            'gt_groups': {}          # {group_id: {activity, members}}
        })

        print('\nCollecting frame-level predictions...')
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                visual_A = batch['visual_A'].to(self.device)
                visual_B = batch['visual_B'].to(self.device)
                geometric = batch['geometric'].to(self.device)

                # Get metadata
                seq_nums = batch['sequence']
                frame_ids = batch['frame']
                track_id_A = batch['track_id_A']
                track_id_B = batch['track_id_B']
                bbox_A = batch['bbox_A']
                bbox_B = batch['bbox_B']
                social_group_A = batch['social_group_id_A']
                social_group_B = batch['social_group_id_B']
                social_activity = batch['social_activity_id']

                # Forward pass
                outputs = self.model(visual_A, visual_B, geometric)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                # Group by frame
                for i in range(len(seq_nums)):
                    frame_key = (seq_nums[i].item(), frame_ids[i].item())

                    track_a = track_id_A[i].item()
                    track_b = track_id_B[i].item()
                    pair_key = tuple(sorted([track_a, track_b]))

                    # Store pair prediction
                    frame_data[frame_key]['pair_predictions'][pair_key] = predictions[i]

                    # Store person bboxes
                    frame_data[frame_key]['person_bboxes'][track_a] = bbox_A[i].cpu().numpy()
                    frame_data[frame_key]['person_bboxes'][track_b] = bbox_B[i].cpu().numpy()

                    # Store GT group info
                    group_a = social_group_A[i].item()
                    group_b = social_group_B[i].item()
                    gt_activity = social_activity[i].item()

                    if group_a not in frame_data[frame_key]['gt_groups']:
                        frame_data[frame_key]['gt_groups'][group_a] = {
                            'activity': gt_activity,
                            'members': set()
                        }
                    frame_data[frame_key]['gt_groups'][group_a]['members'].add(track_a)

                    if group_b not in frame_data[frame_key]['gt_groups']:
                        frame_data[frame_key]['gt_groups'][group_b] = {
                            'activity': gt_activity,
                            'members': set()
                        }
                    frame_data[frame_key]['gt_groups'][group_b]['members'].add(track_b)

        print(f'Collected data for {len(frame_data)} frames')

        # Evaluate group activities
        print(f'\n{"=" * 80}')
        print('Evaluating Group Activities')
        print(f'{"=" * 80}')

        # Collect predictions for all groups, single-person groups, and multi-person groups
        all_pred_activities = []
        all_gt_activities = []
        single_pred_activities = []
        single_gt_activities = []
        multi_pred_activities = []
        multi_gt_activities = []
        num_single_person_groups = 0
        num_multi_person_groups = 0

        for frame_key, data in frame_data.items():
            pair_predictions = data['pair_predictions']
            person_bboxes = data['person_bboxes']
            gt_groups = data['gt_groups']

            # Build group_members_dict
            group_members_dict = {gid: info['members'] for gid, info in gt_groups.items()}
            all_groups = list(group_members_dict.keys())

            # Vote for each group
            for group_id, group_info in gt_groups.items():
                group_members = group_info['members']
                gt_activity = group_info['activity']

                # Vote for predicted activity
                pred_activity, confidence, vote_counts = vote_group_activity(
                    pair_predictions, group_members, person_bboxes,
                    all_groups, group_members_dict
                )

                # Store in all groups
                all_pred_activities.append(pred_activity)
                all_gt_activities.append(gt_activity)

                # Store separately for single/multi-person groups
                if len(group_members) == 1:
                    single_pred_activities.append(pred_activity)
                    single_gt_activities.append(gt_activity)
                    num_single_person_groups += 1
                else:
                    multi_pred_activities.append(pred_activity)
                    multi_gt_activities.append(gt_activity)
                    num_multi_person_groups += 1

        # Class names depend on class_merge mode
        if self.class_merge:
            # 3-class mode: Moving, Standing, Talking
            class_names = ['Moving', 'Standing', 'Talking']
            num_classes = 3
        else:
            # 6-class mode
            class_names_6 = ['NA', 'Crossing', 'Waiting', 'Queuing', 'Walking', 'Talking']
            class_names_5 = ['NA', 'Moving', 'Waiting', 'Queuing', 'Talking']

        print(f'\n{"=" * 80}')
        if self.class_merge:
            print(f'Group Activity Recognition Results (3 Classes - Merged)')
        else:
            print(f'Group Activity Recognition Results (6 Classes)')
        print(f'{"=" * 80}')
        print(f'Total Groups: {len(all_pred_activities)}')
        print(f'  Single-person:  {num_single_person_groups}')
        print(f'  Multi-person:   {num_multi_person_groups}')

        if self.class_merge:
            # ========================================================================
            # 3-Class Evaluation (class_merge mode)
            # ========================================================================

            # All groups (3 classes)
            all_3class_results = compute_and_print_metrics(
                all_pred_activities, all_gt_activities, class_names,
                group_type_name="All Groups (3 classes)", num_classes=3
            )

            # Single-person groups (3 classes)
            single_3class_results = compute_and_print_metrics(
                single_pred_activities, single_gt_activities, class_names,
                group_type_name="Single-Person Groups (3 classes)", num_classes=3
            )

            # Multi-person groups (3 classes)
            multi_3class_results = compute_and_print_metrics(
                multi_pred_activities, multi_gt_activities, class_names,
                group_type_name="Multi-Person Groups (3 classes)", num_classes=3
            )

            # Save results
            results = {
                '3_class_merged_results': {
                    'all_groups': all_3class_results,
                    'single_person_groups': single_3class_results,
                    'multi_person_groups': multi_3class_results
                },
                'statistics': {
                    'total_groups': len(all_pred_activities),
                    'num_single_person_groups': num_single_person_groups,
                    'num_multi_person_groups': num_multi_person_groups
                },
                'mode': 'class_merge_3_classes'
            }

            results_file = os.path.join(self.save_dir, 'group_activity_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f'\n{"=" * 80}')
            print(f'Results saved to: {results_file}')
            print(f'{"=" * 80}')

            # Return 3-class results as the main metric
            return all_3class_results['overall_accuracy'], all_3class_results['map']

        else:
            # ========================================================================
            # 6-Class Evaluation (Original)
            # ========================================================================

            # All groups (6 classes)
            all_6class_results = compute_and_print_metrics(
                all_pred_activities, all_gt_activities, class_names_6,
                group_type_name="All Groups (6 classes)", num_classes=6
            )

            # Single-person groups (6 classes)
            single_6class_results = compute_and_print_metrics(
                single_pred_activities, single_gt_activities, class_names_6,
                group_type_name="Single-Person Groups (6 classes)", num_classes=6
            )

            # Multi-person groups (6 classes)
            multi_6class_results = compute_and_print_metrics(
                multi_pred_activities, multi_gt_activities, class_names_6,
                group_type_name="Multi-Person Groups (6 classes)", num_classes=6
            )

            # ========================================================================
            # 5-Class Evaluation (Merged: Crossing + Walking → Moving)
            # ========================================================================
            print(f'\n{"=" * 80}')
            print(f'Merged Activity Recognition (Crossing + Walking → Moving, 5 Classes)')
            print(f'{"=" * 80}')

            # Merge activities for all groups
            all_merged_pred = merge_moving_activities(all_pred_activities)
            all_merged_gt = merge_moving_activities(all_gt_activities)

            # Merge activities for single-person groups
            single_merged_pred = merge_moving_activities(single_pred_activities)
            single_merged_gt = merge_moving_activities(single_gt_activities)

            # Merge activities for multi-person groups
            multi_merged_pred = merge_moving_activities(multi_pred_activities)
            multi_merged_gt = merge_moving_activities(multi_gt_activities)

            # All groups (5 classes)
            all_5class_results = compute_and_print_metrics(
                all_merged_pred, all_merged_gt, class_names_5,
                group_type_name="All Groups (5 classes merged)", num_classes=5
            )

            # Single-person groups (5 classes)
            single_5class_results = compute_and_print_metrics(
                single_merged_pred, single_merged_gt, class_names_5,
                group_type_name="Single-Person Groups (5 classes merged)", num_classes=5
            )

            # Multi-person groups (5 classes)
            multi_5class_results = compute_and_print_metrics(
                multi_merged_pred, multi_merged_gt, class_names_5,
                group_type_name="Multi-Person Groups (5 classes merged)", num_classes=5
            )

            # Save results
            results = {
                '6_class_results': {
                    'all_groups': all_6class_results,
                    'single_person_groups': single_6class_results,
                    'multi_person_groups': multi_6class_results
                },
                '5_class_merged_results': {
                    'all_groups': all_5class_results,
                    'single_person_groups': single_5class_results,
                    'multi_person_groups': multi_5class_results
                },
                'statistics': {
                    'total_groups': len(all_pred_activities),
                    'num_single_person_groups': num_single_person_groups,
                    'num_multi_person_groups': num_multi_person_groups
                },
                'mode': 'original_6_classes'
            }

            results_file = os.path.join(self.save_dir, 'group_activity_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f'\n{"=" * 80}')
            print(f'Results saved to: {results_file}')
            print(f'{"=" * 80}')

            # Return merged all groups results as the main metric
            return all_5class_results['overall_accuracy'], all_5class_results['map']


def main():
    parser = argparse.ArgumentParser(description='CAD Stage2 Training')

    # Dataset parameters
    parser.add_argument('--cad_root', type=str, default='../dataset/cad/ActivityDataset',
                        help='Path to CAD ActivityDataset directory')
    parser.add_argument('--train_sequences', type=str, default='30,31,32,33,34,35,36,37,38,39,40,41,42,43,44',
                        help='Training sequences (e.g., "1-30" or "1,2,3")')
    parser.add_argument('--val_sequences', type=str, default='1,2,3,4,12,13,14,17,18,19,20,21,22,23,24,26',
                        help='Validation sequences')
    parser.add_argument('--test_sequences', type=str,default='5,6,7,8,9,10,11,15,16,25,28,29', 
                        help='Test sequences')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of interaction classes (6 for CAD)')
    parser.add_argument('--image_width', type=int, default=720,
                        help='CAD image width')
    parser.add_argument('--image_height', type=int, default=480,
                        help='CAD image height')
    parser.add_argument('--backbone_name', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='ResNet backbone variant')
    parser.add_argument('--feature_mode', type=str, default='both',
                        choices=['both', 'opticalflow_only', 'bboxposition_only'],
                        help='Feature mode for geometric features')

    # Model parameters
    parser.add_argument('--visual_dim', type=int, default=512,
                        help='Visual feature dimension (512 for resnet18, 2048 for resnet50)')
    parser.add_argument('--spatial_dim', type=int, default=9,
                        help='Spatial (geometric) feature dimension')
    parser.add_argument('--relation_hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='Relation network hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--class_weights', type=float, nargs='+', default=None,
                        help='Class weights for loss (6 values for CAD, used as alpha in focal loss)')

    # Loss function parameters
    parser.add_argument('--loss_type', type=str, default='adaptive_focal',
                        choices=['ce', 'focal', 'adaptive_focal'],
                        help='Loss function type (default: adaptive_focal for auto class balancing)')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter (only for --loss_type focal, ignored if --class_weights provided)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter (focusing parameter, recommended: 2.0)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=15,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience (0=disabled)')

    # Other parameters
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers (use 0 if errors)')
    parser.add_argument('--print_freq', type=int, default=20,
                        help='Print frequency during training')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--run_test', action='store_true',
                        help='Run test evaluation after training')
    parser.add_argument('--no_val', action='store_true',
                        help='No validation split: merge train and val into training set, use fixed epochs')
    parser.add_argument('--class_merge', action='store_true',
                        help='Merge to 3 classes: NA->skip, Crossing+Walking->Moving, Waiting+Queuing->Standing, Talking->Talking')

    args = parser.parse_args()

    # Adjust visual_dim based on backbone
    if args.backbone_name == 'resnet18':
        args.visual_dim = 512
    elif args.backbone_name == 'resnet50':
        args.visual_dim = 2048

    # Adjust num_classes based on class_merge
    if args.class_merge:
        args.num_classes = 3
        print("Class merge enabled: NA->skip, Crossing+Walking->Moving, Waiting+Queuing->Standing, Talking->Talking")
        print(f"  Training with 3 classes: Moving(0), Standing(1), Talking(2)")

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

    print(f"\n{'=' * 80}")
    print("Sequence Split Configuration:")
    if args.no_val:
        print(f"  Mode: NO VALIDATION (fixed {args.epochs} epochs)")
        print(f"  Train sequences (merged): {train_seqs}")
    else:
        print(f"  Train sequences: {train_seqs}")
        print(f"  Val sequences: {val_seqs}")
    print(f"  Test sequences: {test_seqs}")
    print(f"{'=' * 80}\n")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("CAD Stage2 Training: Interaction Type Classification")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Model: {args.backbone_name} | Loss: {args.loss_type} | LR: {args.learning_rate}")
    print(f"  Batch: {args.batch_size} | Epochs: {args.epochs} | Scheduler: {args.scheduler}")
    print(f"  Classes: {args.num_classes} | Feature mode: {args.feature_mode}")
    print(f"  Class merge: {'Enabled' if args.class_merge else 'Disabled'}")
    print()

    # Create datasets
    print(f"\n{'=' * 80}")
    print("Loading Datasets")
    print(f"{'=' * 80}")

    train_dataset = CADResNetStage2Dataset(
        cad_root=args.cad_root,
        split='train',
        sequences=train_seqs,
        num_classes=args.num_classes,
        image_width=args.image_width,
        image_height=args.image_height,
        backbone_name=args.backbone_name,
        feature_mode=args.feature_mode,
        class_merge=args.class_merge
    )
    print_dataset_statistics(train_dataset, "Training", num_classes=args.num_classes, class_merge=args.class_merge)

    # Create val dataset only if not in no_val mode
    val_dataset = None
    val_loader = None
    if not args.no_val:
        val_dataset = CADResNetStage2Dataset(
            cad_root=args.cad_root,
            split='val',
            sequences=val_seqs,
            num_classes=args.num_classes,
            image_width=args.image_width,
            image_height=args.image_height,
            backbone_name=args.backbone_name,
            feature_mode=args.feature_mode,
            class_merge=args.class_merge
        )
        print_dataset_statistics(val_dataset, "Validation", num_classes=args.num_classes, class_merge=args.class_merge)

    # Load test dataset if needed (always load in no_val mode for final evaluation)
    test_dataset = None
    test_loader = None
    if args.run_test or args.no_val:
        test_dataset = CADResNetStage2Dataset(
            cad_root=args.cad_root,
            split='test',
            sequences=test_seqs,
            num_classes=args.num_classes,
            image_width=args.image_width,
            image_height=args.image_height,
            backbone_name=args.backbone_name,
            feature_mode=args.feature_mode,
            class_merge=args.class_merge
        )
        print_dataset_statistics(test_dataset, "Test", num_classes=args.num_classes, class_merge=args.class_merge)

    # Create dataloaders
    print(f"\n{'=' * 80}")
    print("Creating Data Loaders")
    print(f"{'=' * 80}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=cad_stage2_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"Train loader: {len(train_loader)} batches ({len(train_dataset)} samples)")

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=cad_stage2_collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f"Val loader:   {len(val_loader)} batches ({len(val_dataset)} samples)")

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=cad_stage2_collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f"Test loader:  {len(test_loader)} batches ({len(test_dataset)} samples)")

    # Initialize trainer
    trainer = CADStage2Trainer(args)

    # Train
    print(f"\n{'=' * 80}")
    print("Starting Training")
    print(f"{'=' * 80}\n")

    if args.no_val:
        # No validation mode: train for fixed epochs, then test
        trainer.train_no_val(train_loader)
        print(f"\n{'=' * 80}")
        print("Training Complete! (no validation mode)")
        print(f"{'=' * 80}")
        print(f"Model saved to: {trainer.save_dir}")
        # Test evaluation
        if test_loader is not None:
            print(f"\n{'=' * 80}")
            print("Final Evaluation on Test Set")
            print(f"{'=' * 80}")
            test_acc, test_map = trainer.test(test_loader)
            print(f"\nTest Results: Accuracy={test_acc:.4f}, mAP={test_map:.4f}")
    else:
        # Normal mode with validation
        trainer.train(train_loader, val_loader)
        print(f"\n{'=' * 80}")
        print("Training Complete!")
        print(f"{'=' * 80}")
        print(f"Model saved to: {trainer.save_dir}")
        print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
        print(f"Best validation F1: {trainer.best_val_f1:.4f}")
        # Test evaluation
        if args.run_test and test_loader is not None:
            print(f"\n{'=' * 80}")
            print("Running Test Evaluation")
            print(f"{'=' * 80}")
            test_acc, test_map = trainer.test(test_loader)
            print(f"\nTest Results: Accuracy={test_acc:.4f}, mAP={test_map:.4f}")

    print(f"\n{'=' * 80}")
    print("All Done!")
    print(f"{'=' * 80}")
    print(f"Results saved to: {trainer.save_dir}")


if __name__ == '__main__':
    main()
