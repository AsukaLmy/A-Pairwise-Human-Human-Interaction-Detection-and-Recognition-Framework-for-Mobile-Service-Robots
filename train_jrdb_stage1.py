import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import sys
import json
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

from src.data_loaders.optimized_temporal_buffer import create_fast_geometric_data_loaders
from src.classifiers.geometric_classifier import (
    AdaptiveGeometricClassifier,
    CausalTemporalStage1,
    ContextAwareGeometricClassifier,
    GeometricStage1Ensemble,
    compute_adaptive_loss
)


# ============================================================================
# Social Group Clustering Functions (same as train_opgeo_stage1.py)
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
        interactions: dict mapping (person_a, person_b) -> interaction_probability
        threshold: probability threshold for considering an interaction

    Returns:
        List[Set]: List of sets, each containing person IDs in a group
    """
    if len(interactions) == 0:
        return []

    # Get all unique person IDs
    persons = set()
    for (a, b) in interactions.keys():
        persons.add(a)
        persons.add(b)
    persons = sorted(list(persons))

    if len(persons) == 0:
        return []

    # Map person ID to index
    person_to_idx = {p: i for i, p in enumerate(persons)}
    idx_to_person = {i: p for p, i in person_to_idx.items()}

    # Build Union-Find structure
    uf = UnionFind(len(persons))

    # Union pairs with interaction probability above threshold
    for (a, b), prob in interactions.items():
        if prob >= threshold:
            uf.union(person_to_idx[a], person_to_idx[b])

    # Get groups and convert indices back to person IDs
    idx_groups = uf.get_groups()
    groups = []
    for idx_group in idx_groups:
        person_group = {idx_to_person[idx] for idx in idx_group}
        groups.append(person_group)

    return groups


def compute_group_size_ap(pred_groups, gt_groups):
    """
    Compute AP for different group sizes:
    - G1: 1 member
    - G2: 2 members
    - G3: 3 members
    - G4: 4 members
    - G5+: 5+ members

    Args:
        pred_groups: List[Set] - predicted groups
        gt_groups: List[Set] - ground truth groups

    Returns:
        dict with AP for each group size category and mAP
    """
    # Categorize GT groups by size
    gt_by_size = {
        'G1': [g for g in gt_groups if len(g) == 1],
        'G2': [g for g in gt_groups if len(g) == 2],
        'G3': [g for g in gt_groups if len(g) == 3],
        'G4': [g for g in gt_groups if len(g) == 4],
        'G5+': [g for g in gt_groups if len(g) >= 5]
    }

    results = {}
    aps = []

    for size_name, gt_subset in gt_by_size.items():
        if len(gt_subset) == 0:
            results[size_name] = {'ap': 0.0, 'num_gt': 0, 'num_pred': 0}
            continue

        # Find matching predicted groups (by IoU)
        matched_pred = []
        for pred_group in pred_groups:
            for gt_group in gt_subset:
                iou = len(pred_group & gt_group) / len(pred_group | gt_group) if len(pred_group | gt_group) > 0 else 0
                if iou > 0.5:  # IoU threshold
                    matched_pred.append(pred_group)
                    break

        # Compute precision and recall
        tp = len(matched_pred)
        fp = len(pred_groups) - tp
        fn = len(gt_subset) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[size_name] = {
            'ap': f1,  # Using F1 as AP
            'precision': precision,
            'recall': recall,
            'num_gt': len(gt_subset),
            'num_pred': len(matched_pred),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        aps.append(f1)

    # Compute mAP across all size categories (only count non-zero categories)
    non_zero_aps = aps  # Include all categories
    map_score = np.mean(non_zero_aps) if len(non_zero_aps) > 0 else 0

    results['mAP'] = map_score

    return results


class GeometricStage1Trainer:
    """
    Trainer for geometric Stage1 interaction detection
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join('checkpoints', f'geometric_stage1_{timestamp}')
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
        
        print(f"GeometricStage1Trainer initialized on {self.device}")
        print(f"Save directory: {self.save_dir}")
    
    def _initialize_model(self):
        """Initialize the geometric model based on config"""
        if self.config.model_type == 'adaptive':
            self.model = AdaptiveGeometricClassifier(
                hidden_dims=self.config.hidden_dims,
                dropout=self.config.dropout
            )
        elif self.config.model_type == 'temporal':
            self.model = CausalTemporalStage1(
                history_length=self.config.history_length,
                hidden_size=self.config.hidden_size,
                dropout=self.config.dropout
            )
        elif self.config.model_type == 'context_aware':
            self.model = ContextAwareGeometricClassifier(
                hidden_dim=self.config.hidden_size
            )
        elif self.config.model_type == 'ensemble':
            self.model = GeometricStage1Ensemble(
                num_models=self.config.num_ensemble_models
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {self.config.model_type}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_training(self):
        """Setup optimizer, criterion, scheduler"""
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
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
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for batch in pbar:
            # Move data to device
            geometric_features = batch['geometric_features'].to(self.device)
            targets = batch['stage1_label'].to(self.device)

            # Prepare model inputs based on model type
            if self.config.model_type == 'temporal':
                history_geometric = batch['history_geometric'].to(self.device)
                motion_features = batch['motion_features'].to(self.device)
                scene_context = batch['scene_context'].to(self.device)

                outputs = self.model(
                    geometric_features, history_geometric,
                    motion_features, scene_context
                )
            elif self.config.model_type == 'context_aware' or self.config.model_type == 'ensemble':
                scene_context = batch['scene_context'].to(self.device)
                outputs = self.model(geometric_features, scene_context)
            else:
                outputs = self.model(geometric_features)

            # Compute loss
            if self.config.model_type == 'adaptive' and hasattr(self.model, 'feature_weights'):
                loss = compute_adaptive_loss(
                    outputs, targets, self.model.feature_weights,
                    self.config.weight_regularization, self.config.sparsity_regularization
                )
            else:
                loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()

            # Record metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        f1_macro = f1_score(all_targets, all_predictions, average='macro')

        return avg_loss, accuracy, f1_weighted, f1_macro
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch"""
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
                if self.config.model_type == 'temporal':
                    history_geometric = batch['history_geometric'].to(self.device)
                    motion_features = batch['motion_features'].to(self.device)
                    scene_context = batch['scene_context'].to(self.device)
                    
                    outputs = self.model(
                        geometric_features, history_geometric,
                        motion_features, scene_context
                    )
                elif self.config.model_type == 'context_aware' or self.config.model_type == 'ensemble':
                    scene_context = batch['scene_context'].to(self.device)
                    outputs = self.model(geometric_features, scene_context)
                else:
                    outputs = self.model(geometric_features)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Record metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        f1_macro = f1_score(all_targets, all_predictions, average='macro')

        return avg_loss, accuracy, f1_weighted, f1_macro, all_targets, all_predictions
    
    def test_epoch(self, test_loader):
        """Test epoch evaluation (same logic as validation)"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                geometric_features = batch['geometric_features'].to(self.device)
                targets = batch['stage1_label'].to(self.device)
                
                # Forward pass
                if self.config.model_type == 'temporal':
                    history_geometric = batch['history_geometric'].to(self.device)
                    motion_features = batch['motion_features'].to(self.device)
                    scene_context = batch['scene_context'].to(self.device)
                    
                    outputs = self.model(
                        geometric_features, history_geometric,
                        motion_features, scene_context
                    )
                elif self.config.model_type == 'context_aware' or self.config.model_type == 'ensemble':
                    scene_context = batch['scene_context'].to(self.device)
                    outputs = self.model(geometric_features, scene_context)
                else:
                    outputs = self.model(geometric_features)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Record metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        f1_macro = f1_score(all_targets, all_predictions, average='macro')

        return avg_loss, accuracy, f1_weighted, f1_macro, all_targets, all_predictions

    def test_baseline_clustering(self, test_loader, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        """
        Test with baseline group clustering evaluation

        Clusters interactions into social groups and evaluates by group size:
        - G1: 1 member
        - G2: 2 members
        - G3: 3 members
        - G4: 4 members
        - G5+: 5+ members

        Args:
            test_loader: DataLoader for test set
            thresholds: List of probability thresholds to test
        """
        print(f'\n{"=" * 80}')
        print('Baseline Group Clustering Evaluation')
        print(f'{"=" * 80}')

        self.model.eval()

        # Collect frame-level predictions and GT interactions
        frame_data = defaultdict(lambda: {
            'pred_interactions': {},  # (person_a, person_b) -> predicted probability
            'gt_interactions': {}     # (person_a, person_b) -> 1.0 (only store positive GT)
        })

        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                geometric_features = batch['geometric_features'].to(self.device)

                # Forward pass
                if self.config.model_type == 'temporal':
                    history_geometric = batch['history_geometric'].to(self.device)
                    motion_features = batch['motion_features'].to(self.device)
                    scene_context = batch['scene_context'].to(self.device)
                    outputs = self.model(geometric_features, history_geometric, motion_features, scene_context)
                elif self.config.model_type == 'context_aware' or self.config.model_type == 'ensemble':
                    scene_context = batch['scene_context'].to(self.device)
                    outputs = self.model(geometric_features, scene_context)
                else:
                    outputs = self.model(geometric_features)

                # Convert logits to probabilities
                probs = F.softmax(outputs, dim=1)
                interaction_probs = probs[:, 1].cpu().numpy()  # P(has_interaction)

                # Get metadata from batch
                frame_ids = batch.get('frame_id', None)
                person_A_ids = batch.get('person_A_id', None)
                person_B_ids = batch.get('person_B_id', None)
                gt_labels = batch.get('stage1_label', None)  # GT interaction labels

                # Store pairwise predictions and GT interactions
                if frame_ids is not None and person_A_ids is not None and person_B_ids is not None:
                    for i in range(len(interaction_probs)):
                        frame_id = frame_ids[i]
                        person_a = person_A_ids[i].item() if torch.is_tensor(person_A_ids[i]) else person_A_ids[i]
                        person_b = person_B_ids[i].item() if torch.is_tensor(person_B_ids[i]) else person_B_ids[i]

                        # Store predicted interaction probability
                        frame_data[frame_id]['pred_interactions'][(person_a, person_b)] = interaction_probs[i]

                        # Store GT interaction (only positive pairs)
                        if gt_labels is not None:
                            gt_label = gt_labels[i].item() if torch.is_tensor(gt_labels[i]) else gt_labels[i]
                            if gt_label == 1:  # Has interaction
                                frame_data[frame_id]['gt_interactions'][(person_a, person_b)] = 1.0

        print(f'Collected data for {len(frame_data)} frames')

        # First, cluster GT groups (only once, not affected by threshold)
        all_gt_groups = []
        total_gt_interactions = 0
        for frame_id, data in frame_data.items():
            gt_interactions = data['gt_interactions']
            total_gt_interactions += len(gt_interactions)
            if len(gt_interactions) > 0:
                # Cluster GT groups using GT interactions (threshold=0.5 since values are 0 or 1)
                gt_groups = cluster_social_groups(gt_interactions, threshold=0.5)
                all_gt_groups.extend(gt_groups)

        # Print GT statistics
        print(f'Total GT interactions: {total_gt_interactions}')
        print(f'Total GT groups: {len(all_gt_groups)}')
        if len(all_gt_groups) > 0:
            gt_sizes = [len(g) for g in all_gt_groups]
            print(f'GT group size distribution:')
            print(f'  G1 (size=1): {sum(1 for s in gt_sizes if s == 1)}')
            print(f'  G2 (size=2): {sum(1 for s in gt_sizes if s == 2)}')
            print(f'  G3 (size=3): {sum(1 for s in gt_sizes if s == 3)}')
            print(f'  G4 (size=4): {sum(1 for s in gt_sizes if s == 4)}')
            print(f'  G5+ (size>=5): {sum(1 for s in gt_sizes if s >= 5)}')

        # Test multiple thresholds
        results_by_threshold = []

        for threshold in thresholds:
            print(f'\nEvaluating threshold = {threshold}...')

            all_pred_groups = []

            for frame_id, data in frame_data.items():
                pred_interactions = data['pred_interactions']

                # Cluster predicted groups using predicted probabilities
                pred_groups = cluster_social_groups(pred_interactions, threshold=threshold)
                all_pred_groups.extend(pred_groups)

            print(f'  Total predicted groups: {len(all_pred_groups)}')

            # Compute group size AP
            if len(all_gt_groups) > 0:
                group_size_results = compute_group_size_ap(all_pred_groups, all_gt_groups)

                print(f'  G1 AP: {group_size_results.get("G1", {}).get("ap", 0):.4f}')
                print(f'  G2 AP: {group_size_results.get("G2", {}).get("ap", 0):.4f}')
                print(f'  G3 AP: {group_size_results.get("G3", {}).get("ap", 0):.4f}')
                print(f'  G4 AP: {group_size_results.get("G4", {}).get("ap", 0):.4f}')
                print(f'  G5+ AP: {group_size_results.get("G5+", {}).get("ap", 0):.4f}')
                print(f'  mAP: {group_size_results.get("mAP", 0):.4f}')

                results_by_threshold.append({
                    'threshold': threshold,
                    'group_size_results': group_size_results
                })
            else:
                print('  Warning: No GT groups available for evaluation')
                results_by_threshold.append({
                    'threshold': threshold,
                    'group_size_results': None
                })

        # Print summary
        print(f'\n{"=" * 80}')
        print('Baseline Group Clustering Results (Different Thresholds)')
        print(f'{"=" * 80}')
        print(f'{"Threshold":<12} {"G1 AP":<10} {"G2 AP":<10} {"G3 AP":<10} {"G4 AP":<10} {"G5+ AP":<10} {"mAP":<10}')
        print(f'{"-" * 80}')

        for result in results_by_threshold:
            threshold = result['threshold']
            group_results = result['group_size_results']
            if group_results is not None:
                g1_ap = group_results.get('G1', {}).get('ap', 0)
                g2_ap = group_results.get('G2', {}).get('ap', 0)
                g3_ap = group_results.get('G3', {}).get('ap', 0)
                g4_ap = group_results.get('G4', {}).get('ap', 0)
                g5_ap = group_results.get('G5+', {}).get('ap', 0)
                map_score = group_results.get('mAP', 0)
                print(f'{threshold:<12.1f} {g1_ap:<10.4f} {g2_ap:<10.4f} {g3_ap:<10.4f} {g4_ap:<10.4f} {g5_ap:<10.4f} {map_score:<10.4f}')
            else:
                print(f'{threshold:<12.1f} {"N/A":<10} {"N/A":<10} {"N/A":<10} {"N/A":<10} {"N/A":<10} {"N/A":<10}')

        print(f'{"=" * 80}')

        # Save results
        results_file = os.path.join(self.save_dir, 'baseline_clustering_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_by_threshold, f, indent=2)

        print(f'\nBaseline clustering results saved to: {results_file}')

        # Find best threshold by mAP
        valid_results = [r for r in results_by_threshold if r['group_size_results'] is not None]
        if len(valid_results) > 0:
            best_result = max(valid_results, key=lambda x: x['group_size_results'].get('mAP', 0))
            best_threshold = best_result['threshold']
            best_map = best_result['group_size_results'].get('mAP', 0)
            print(f'\nBest Threshold: {best_threshold}')
            print(f'  mAP: {best_map:.4f}')

        return results_by_threshold

    def train(self, train_loader, val_loader, test_loader=None):
        """Main training loop"""
        print(f"Starting training for {self.config.epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc, train_f1_weighted, train_f1_macro = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc, val_f1_weighted, val_f1_macro, val_targets, val_predictions = self.validate_epoch(val_loader, epoch)
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler:
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check improvement
            improved = val_acc > self.best_val_acc
            if improved:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1_weighted
                self.epochs_without_improvement = 0

                # Save best model
                self.save_checkpoint('best_model', epoch, val_loss, val_acc, val_f1_weighted, val_f1_macro)
            else:
                self.epochs_without_improvement += 1
            
            epoch_time = time.time() - epoch_start
            
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'           F1 Weighted: {val_f1_weighted:.4f}, F1 Macro: {val_f1_macro:.4f}, Time: {epoch_time:.1f}s')
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}', epoch, val_loss, val_acc, val_f1_weighted, val_f1_macro)
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.1f} seconds')
        print(f'Best validation accuracy: {self.best_val_acc:.4f}')
        print(f'Best validation F1: {self.best_val_f1:.4f}')
        
        # Generate final report
        self.generate_final_report(val_targets, val_predictions)
        self.plot_training_curves()
        
        # Analyze feature importance if available
        self.analyze_feature_importance()

        # Load best model for evaluation
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print(f"\nLoading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Best model loaded (epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_accuracy']:.4f})")

        # Baseline clustering evaluation on validation set
        print(f"\n{'=' * 80}")
        print("Baseline Group Clustering Evaluation on Validation Set (Best Model)")
        print(f"{'=' * 80}")
        val_clustering_results = self.test_baseline_clustering(val_loader)

        # Test evaluation if test_loader is provided
        if test_loader is not None:
            print("\nEvaluating on test set...")
            test_loss, test_acc, test_f1_weighted, test_f1_macro, test_targets, test_predictions = self.test_epoch(test_loader)

            print(f'Test Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')
            print(f'Test F1 Weighted: {test_f1_weighted:.4f}, F1 Macro: {test_f1_macro:.4f}')

            # Generate test report
            self.generate_test_report(test_targets, test_predictions, test_loss, test_acc, test_f1_weighted, test_f1_macro)
    
    def save_checkpoint(self, name, epoch, val_loss, val_acc, val_f1_weighted, val_f1_macro=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1_weighted': val_f1_weighted,
            'val_f1_macro': val_f1_macro if val_f1_macro is not None else 0.0,
            'config': vars(self.config)
        }
        
        torch.save(checkpoint, os.path.join(self.save_dir, f'{name}.pth'))
        print(f'Checkpoint saved: {name}.pth')
    
    def generate_final_report(self, val_targets, val_predictions):
        """Generate evaluation report"""
        print("\nGenerating Geometric Stage1 evaluation report...")
        
        # Classification report
        class_names = ['No Interaction', 'Has Interaction']
        report = classification_report(val_targets, val_predictions, target_names=class_names)
        
        # Confusion matrix
        cm = confusion_matrix(val_targets, val_predictions)
        
        # Save report
        with open(os.path.join(self.save_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("Geometric Stage1 Binary Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {self.config.model_type}\n")
            f.write(f"Best Validation Accuracy: {self.best_val_acc:.4f}\n")
            f.write(f"Best Validation F1: {self.best_val_f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write(f"\n\nConfusion Matrix:\n{cm}\n")
        
        print(f"Evaluation report saved to {self.save_dir}")
    
    def generate_test_report(self, test_targets, test_predictions, test_loss, test_acc, test_f1_weighted, test_f1_macro):
        """Generate test evaluation report"""
        print("\nGenerating test evaluation report...")
        
        # Classification report
        class_names = ['No Interaction', 'Has Interaction']
        report = classification_report(test_targets, test_predictions, target_names=class_names)
        
        # Confusion matrix
        cm = confusion_matrix(test_targets, test_predictions)
        
        # Save test report
        with open(os.path.join(self.save_dir, 'test_evaluation_report.txt'), 'w') as f:
            f.write("Geometric Stage1 Test Set Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {self.config.model_type}\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test F1 Weighted: {test_f1_weighted:.4f}\n")
            f.write(f"Test F1 Macro: {test_f1_macro:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(cm))
        
        print(f"Test evaluation report saved to: {os.path.join(self.save_dir, 'test_evaluation_report.txt')}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        ax2.plot(self.val_accuracies, label='Val Acc', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        plt.close()
    
    def analyze_feature_importance(self):
        """Analyze and save feature importance if model supports it"""
        if hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()

            print("\nLearned Feature Importance:")
            for feature, weight in importance:
                print(f"  {feature}: {weight:.4f}")

            # Save to file (convert numpy types to Python native types)
            with open(os.path.join(self.save_dir, 'feature_importance.json'), 'w') as f:
                # Convert numpy.float32 to float for JSON serialization
                importance_dict = {k: float(v) for k, v in importance}
                json.dump(importance_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train Geometric Stage1 Classifier')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='../dataset',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='adaptive',
                        choices=['adaptive', 'temporal', 'context_aware', 'ensemble'],
                        help='Type of geometric model')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 16],
                        help='Hidden layer dimensions for adaptive model')
    parser.add_argument('--hidden_size', type=int, default=16,
                        help='Hidden size for temporal/context models')
    parser.add_argument('--history_length', type=int, default=5,
                        help='Length of temporal history')
    parser.add_argument('--num_ensemble_models', type=int, default=3,
                        help='Number of models in ensemble')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--frame_interval', type=int, default=1,
                        help='Frame sampling interval (1=every frame, 5=every 5th frame)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=17,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=20,
                        help='Step size for step scheduler')
    
    # Regularization parameters
    parser.add_argument('--weight_regularization', type=float, default=0.01,
                        help='Feature weight regularization')
    parser.add_argument('--sparsity_regularization', type=float, default=0.01,
                        help='Feature sparsity regularization')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # Training control
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval')
    
    # Feature options
    parser.add_argument('--use_temporal', action='store_true', default=False,
                        help='Use temporal features')
    parser.add_argument('--no_temporal', dest='use_temporal', action='store_false',
                        help='Disable temporal features (default)')
    parser.add_argument('--use_scene_context', action='store_true', default=True,
                        help='Use scene context features')
    
    # Loading optimization parameters
    parser.add_argument('--loading_strategy', type=str, default='lazy',
                        choices=['cached', 'optimized', 'lazy', 'original'],
                        help='Data loading strategy: cached (fastest), optimized (balanced), lazy (memory efficient), original (fallback)')

    # Dataset split parameters
    parser.add_argument('--use_custom_splits', action='store_true', default=True,
                        help='Use predefined scene splits instead of percentage-based splits')
    parser.add_argument('--trainset_scenes', type=str, nargs='*', default=None,
                        help='List of scene names for training set (only used with --use_custom_splits)')
    parser.add_argument('--valset_scenes', type=str, nargs='*', default=None,
                        help='List of scene names for validation set (only used with --use_custom_splits)')
    parser.add_argument('--testset_scenes', type=str, nargs='*', default=None,
                        help='List of scene names for test set (only used with --use_custom_splits)')

    # Baseline clustering evaluation
    parser.add_argument('--run_baseline_test', action='store_true',
                        help='Run baseline group clustering evaluation on test set')

    args = parser.parse_args()

    # Handle custom dataset splits
    if args.use_custom_splits:
        print("Using custom scene splits for dataset...")
        if args.trainset_scenes is None or args.valset_scenes is None or args.testset_scenes is None:
            print("Warning: --use_custom_splits specified but not all scene lists provided.")
            print("Using default scene splits from resnet_stage2_dataset.py")
            from datasets.geometric_dataset import create_geometric_data_loaders_with_custom_splits
            train_loader, val_loader, test_loader = create_geometric_data_loaders_with_custom_splits(
                data_path=args.data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                history_length=args.history_length,
                use_temporal=args.use_temporal,
                use_scene_context=args.use_scene_context,
                frame_interval=args.frame_interval
            )
        else:
            print(f"Custom splits: Train={len(args.trainset_scenes)}, Val={len(args.valset_scenes)}, Test={len(args.testset_scenes)} scenes")
            from datasets.geometric_dataset import create_geometric_data_loaders
            train_loader, val_loader, test_loader = create_geometric_data_loaders(
                data_path=args.data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                history_length=args.history_length,
                use_temporal=args.use_temporal,
                use_scene_context=args.use_scene_context,
                trainset_split=args.trainset_scenes,
                valset_split=args.valset_scenes,
                testset_split=args.testset_scenes,
                use_custom_splits=True,
                frame_interval=args.frame_interval
            )
    else:
        # Create data loaders with optimized loading (original logic)
        print("Loading geometric data with optimized loader...")

        # Choose loading strategy
        loading_strategy = args.loading_strategy

        if loading_strategy == 'cached':
            # Use pre-computed cache for maximum speed
            from src.data_loaders.fast_temporal_cache import create_fast_cached_data_loaders
            train_loader, val_loader, test_loader = create_fast_cached_data_loaders(
                data_path=args.data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_temporal=args.use_temporal,
                use_scene_context=args.use_scene_context,
                history_length=args.history_length
            )
        elif loading_strategy == 'optimized':
            # Use optimized multi-process loader
            from src.data_loaders.optimized_dataloader import create_optimized_data_loaders
            train_loader, val_loader, test_loader = create_optimized_data_loaders(
                data_path=args.data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_temporal=args.use_temporal,
                use_scene_context=args.use_scene_context,
                history_length=args.history_length
            )
        elif loading_strategy == 'lazy':
            # Use lazy loading for memory efficiency
            from src.data_loaders.lazy_temporal import create_lazy_data_loaders
            train_loader, val_loader, test_loader = create_lazy_data_loaders(
                data_path=args.data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_temporal=args.use_temporal,
                use_scene_context=args.use_scene_context,
                history_length=args.history_length
            )
        else:
            # Fallback to original loader
            from src.data_loaders.optimized_temporal_buffer import create_fast_geometric_data_loaders
            train_loader, val_loader, test_loader = create_fast_geometric_data_loaders(
                data_path=args.data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                history_length=args.history_length,
                use_temporal=args.use_temporal,
                use_scene_context=args.use_scene_context
            )

        print(f"Data loading strategy: {loading_strategy}")

    # Create trainer
    trainer = GeometricStage1Trainer(args)

    # Train model
    trainer.train(train_loader, val_loader, test_loader)

    print("Training completed!")

    # Run baseline group clustering evaluation on test set if requested
    if args.run_baseline_test:
        print(f"\n{'=' * 80}")
        print("Baseline Group Clustering Evaluation on Test Set")
        print(f"{'=' * 80}")
        trainer.test_baseline_clustering(test_loader)


if __name__ == '__main__':
    main()