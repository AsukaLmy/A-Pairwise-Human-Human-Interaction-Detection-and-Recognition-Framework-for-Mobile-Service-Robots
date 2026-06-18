"""
GNN-based Stage2 Behavior Classification Training Script
Trains a Graph Attention Network (GAT) for scene-level pedestrian interaction classification.

Key differences from train_jrdb_stage2_withbackbone.py:
  - Dataset unit: frame (scene) instead of pair
  - DataLoader uses gnn_collate_fn (returns List[Dict], not stacked tensors)
  - Model forward receives a List[Dict] per batch
  - Labels are gathered from all pairs across all scenes in the batch
  - Optimizer uses differential LR (backbone vs GNN+classifier)
"""

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '1'

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from datetime import datetime
from tqdm import tqdm

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

from configs.gnn_stage2_config import GNNStage2Config, get_gnn_resnet18_config, get_gnn_efficientnet_config
from datasets.gnn_stage2_dataset import GNNStage2Dataset, create_gnn_stage2_data_loaders, gnn_collate_fn
from models.gnn_stage2_classifier import GNNStage2Classifier, create_gnn_stage2_model
from models.resnet_stage2_classifier import ResNetStage2Loss
from src.classifiers.geometric_stage2_classifier import Stage2Evaluator


CLASS_NAMES = ['Walking Together', 'Standing Together', 'Sitting Together']


# ============================================================================
# Trainer
# ============================================================================

class GNNStage2Trainer:
    """
    Trainer for GNN-based Stage2 classifier.
    Handles scene-level batches where each sample is a full frame graph.
    """

    def __init__(self, config: GNNStage2Config, device: torch.device):
        self.config = config
        self.device = device

        # ---- Model ----
        print("\nCreating GNN model...")
        self.model = create_gnn_stage2_model(config).to(device)

        # ---- Loss ----
        self.criterion = ResNetStage2Loss(
            class_weights=config.class_weights,
            gamma=config.focal_gamma,
        ).to(device)

        # ---- Optimizer: differential LR (backbone much lower) ----
        backbone_params = list(self.model.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.model.parameters()
                        if id(p) not in backbone_ids]

        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params,
             'lr': config.learning_rate * config.backbone_lr_multiplier},
            {'params': other_params,
             'lr': config.learning_rate},
        ], weight_decay=config.weight_decay)

        # ---- Scheduler ----
        if config.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=1e-6
            )
        elif config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=15, gamma=0.5
            )
        else:
            self.scheduler = None

        # ---- State ----
        self.best_val_mpca = 0.0
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        print(f"Checkpoint dir: {config.checkpoint_dir}")

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def _prepare_batch(self, batch):
        """
        batch: List[Dict] from gnn_collate_fn

        Returns:
            scene_data: List[Dict] with tensors moved to device (without labels)
            all_labels: [P_total] label tensor on device
        """
        scene_data = []
        label_list = []

        for item in batch:
            scene_dict = {
                'person_crops': item['person_crops'].to(self.device),
                'person_boxes': item['person_boxes'].to(self.device),
                'target_pairs': item['target_pairs'].to(self.device),
            }
            # Pass precomputed edges when available (avoids slow per-forward edge loop)
            if 'pre_edge_index' in item:
                scene_dict['pre_edge_index'] = item['pre_edge_index'].to(self.device)
                scene_dict['pre_edge_feats'] = item['pre_edge_feats'].to(self.device)
            scene_data.append(scene_dict)
            label_list.append(item['pair_labels'])

        all_labels = torch.cat(label_list, dim=0).to(self.device)
        return scene_data, all_labels

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(self, train_loader, epoch: int):
        self.model.train()
        evaluator = Stage2Evaluator(CLASS_NAMES)
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            if not batch:   # empty after collate (all None items)
                continue

            scene_data, all_labels = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            output = self.model(scene_data)
            logits = output['logits']   # [P_total, 3]

            loss, loss_dict = self.criterion(logits, all_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1)
            evaluator.update(preds.cpu().numpy(), all_labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()

        avg_loss = total_loss / max(n_batches, 1)
        metrics = evaluator.compute_metrics()
        mpca = metrics.get('mpca', 0.0)
        acc  = metrics.get('overall_accuracy', 0.0)
        print(f"Train Epoch {epoch}: Loss={avg_loss:.6f}, Acc={acc:.4f}, MPCA={mpca:.4f}")
        return avg_loss, acc, mpca, metrics

    # ------------------------------------------------------------------
    # Validate epoch
    # ------------------------------------------------------------------

    def validate_epoch(self, val_loader, epoch: int):
        self.model.eval()
        evaluator = Stage2Evaluator(CLASS_NAMES)
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ")
        with torch.no_grad():
            for batch in pbar:
                if not batch:
                    continue
                scene_data, all_labels = self._prepare_batch(batch)
                output = self.model(scene_data)
                loss, _ = self.criterion(output['logits'], all_labels)

                total_loss += loss.item()
                n_batches += 1

                preds = output['logits'].argmax(dim=1)
                evaluator.update(preds.cpu().numpy(), all_labels.cpu().numpy())

        pbar.close()

        avg_loss = total_loss / max(n_batches, 1)
        metrics = evaluator.compute_metrics()
        acc  = metrics.get('overall_accuracy', 0.0)
        mpca = metrics.get('mpca', 0.0)
        print(f"Val   Epoch {epoch}: Loss={avg_loss:.6f}, Acc={acc:.4f}, MPCA={mpca:.4f}")
        return avg_loss, acc, mpca, metrics

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self, test_loader):
        self.model.eval()
        evaluator = Stage2Evaluator(CLASS_NAMES)

        print(f"\n{'='*80}")
        print("FINAL TEST EVALUATION")
        print(f"{'='*80}")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                if not batch:
                    continue
                scene_data, all_labels = self._prepare_batch(batch)
                output = self.model(scene_data)
                preds = output['logits'].argmax(dim=1)
                evaluator.update(preds.cpu().numpy(), all_labels.cpu().numpy())

        evaluator.print_evaluation_report()
        metrics = evaluator.compute_metrics()
        acc  = metrics.get('overall_accuracy', 0.0)
        mpca = metrics.get('mpca', 0.0)
        print(f"\nTest Results: Acc={acc:.4f}, MPCA={mpca:.4f}")
        return acc, mpca, metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader, test_loader=None):
        print(f"\n{'='*80}")
        print("STARTING GNN TRAINING")
        print(f"{'='*80}\n")

        for epoch in range(1, self.config.epochs + 1):

            train_loss, train_acc, train_mpca, _ = self.train_epoch(train_loader, epoch)
            val_loss,   val_acc,   val_mpca,   _ = self.validate_epoch(val_loader, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            # Check improvement
            if val_mpca > self.best_val_mpca:
                self.best_val_mpca = val_mpca
                self.best_val_acc  = val_acc
                self.epochs_no_improve = 0

                ckpt_path = os.path.join(self.config.checkpoint_dir, 'best_model_gnn.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_mpca': self.best_val_mpca,
                    'best_val_acc': self.best_val_acc,
                    'config': self.config.__dict__,
                }, ckpt_path)
                print(f"  [Saved best model] MPCA={self.best_val_mpca:.4f} → {ckpt_path}")
            else:
                self.epochs_no_improve += 1

            print(f"  Best Val MPCA: {self.best_val_mpca:.4f} "
                  f"| Epochs w/o improvement: {self.epochs_no_improve}")

            # Early stopping
            if self.epochs_no_improve >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {self.config.early_stopping_patience} "
                      f"epochs without improvement.")
                break

        # Final test
        if test_loader is not None:
            ckpt_path = os.path.join(self.config.checkpoint_dir, 'best_model_gnn.pth')
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                print(f"Loaded best model from epoch {ckpt['epoch']} for test evaluation")
            self.test(test_loader)

        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"  Best Val MPCA: {self.best_val_mpca:.4f}")
        print(f"  Best Val Acc:  {self.best_val_acc:.4f}")
        print(f"{'='*80}")


# ============================================================================
# FLOPs estimation
# ============================================================================

def estimate_gnn_flops(config, device):
    """Rough FLOPs estimate for backbone + GAT for a typical scene."""
    if not THOP_AVAILABLE:
        print("thop not available – skipping FLOPs analysis")
        return

    print(f"\n{'='*80}")
    print("GNN FLOPS ANALYSIS")
    print(f"{'='*80}")

    model = create_gnn_stage2_model(config).to(device)
    model.eval()

    # Backbone FLOPs (per person crop)
    crop = torch.randn(1, 3, config.crop_size, config.crop_size).to(device)
    bb_flops, bb_params = profile(model.backbone, inputs=(crop,), verbose=False)
    print(f"Backbone ({config.backbone_name}):")
    print(f"  FLOPs / person:  {bb_flops / 1e9:.3f} G")
    print(f"  Params:          {bb_params / 1e6:.2f} M")

    N_typical = 8
    print(f"\nEstimate for N={N_typical} persons / scene:")
    print(f"  Backbone total:  {N_typical * bb_flops / 1e9:.3f} G  "
          f"(baseline pairwise for 10 pairs would be ~{2*10*bb_flops/1e9:.3f} G)")
    print(f"  GAT computation: typically << backbone")
    print(f"  → GNN is ~{int(2*10/(N_typical))}x more efficient on backbone calls "
          f"when N persons participate in many pairs")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='GNN Stage2 Training')
    parser.add_argument('--data_path',    type=str,   default='../dataset')
    parser.add_argument('--backbone',     type=str,   default='efficientnet_v2_s',
                        choices=['resnet18', 'resnet34', 'resnet50',
                                 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'alexnet', 'mobilenet_v3_small', 'mobilenet_v3_large',
                                 'efficientnet_v2_s',
                                 'litehrnet_18', 'hrnet_w18', 'hrnet_w32', 'hrnet_w48'],
                        help='CNN backbone architecture (default: efficientnet_v2_s)')
    parser.add_argument('--visual_dim',   type=int,   default=256,
                        help='Visual feature dim (default 256 for efficientnet_v2_s)')
    parser.add_argument('--gnn_hidden',   type=int,   default=256)
    parser.add_argument('--gnn_layers',   type=int,   default=2)
    parser.add_argument('--gnn_heads',    type=int,   default=4)
    parser.add_argument('--batch_size',   type=int,   default=4)
    parser.add_argument('--epochs',       type=int,   default=40)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--frame_interval', type=int, default=1)
    parser.add_argument('--freeze_backbone', action='store_true', default=True)
    parser.add_argument('--no_freeze_backbone', dest='freeze_backbone',
                        action='store_false')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./checkpoints/gnn_stage2')
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--flops_only',   action='store_true',
                        help='Only calculate FLOPs, no training')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Config ----
    if args.backbone == 'efficientnet_v2_s':
        config = get_gnn_efficientnet_config(data_path=args.data_path)
    else:
        config = get_gnn_resnet18_config(data_path=args.data_path)
    config.backbone_name      = args.backbone
    config.visual_feature_dim = args.visual_dim
    config.gnn_hidden_dim     = args.gnn_hidden
    config.gnn_num_layers     = args.gnn_layers
    config.gnn_num_heads      = args.gnn_heads
    config.batch_size         = args.batch_size
    config.epochs             = args.epochs
    config.learning_rate      = args.lr
    config.weight_decay       = args.weight_decay
    config.frame_interval     = args.frame_interval
    config.freeze_backbone    = args.freeze_backbone
    config.checkpoint_dir     = args.checkpoint_dir
    config.num_workers        = args.num_workers

    print(f"\nGNN Stage2 Config:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    if args.flops_only:
        estimate_gnn_flops(config, device)
        return

    # ---- Data ----
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_gnn_stage2_data_loaders(config)

    # Auto-compute class weights from training data distribution
    train_dataset = train_loader.dataset
    label_dist = train_dataset.get_class_distribution()
    if label_dist.get('class_counts'):
        counts = label_dist['class_counts']
        total = label_dist['total_pairs']
        n_cls = len(counts)
        config.class_weights = {
            int(c): total / (n_cls * cnt) for c, cnt in counts.items()
        }
        print(f"Auto class weights: {config.class_weights}")

    # ---- Trainer ----
    trainer = GNNStage2Trainer(config, device)

    # ---- FLOPs estimate (informational) ----
    estimate_gnn_flops(config, device)

    # ---- Train ----
    start = time.time()
    trainer.train(train_loader, val_loader, test_loader)
    elapsed = time.time() - start
    print(f"\nTotal training time: {elapsed / 60:.1f} min")


if __name__ == '__main__':
    main()
