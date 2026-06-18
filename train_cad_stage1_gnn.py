#!/usr/bin/env python3
"""
CAD Stage1 GNN Training: Binary Interaction Detection (No Visual Backbone)

Graph Transformer on geometric / optical-flow features.
Optimises binary group-detection loss only (lambda_behavior=0, lambda_group=1).

Evaluation:
  - Validation: binary F1 on group head
  - Test:       social-group clustering (membership accuracy + mAP) at
                thresholds 0.3-0.9 (same protocol as train_cad_stage1.py)
"""

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '1'

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import random
import torch
import numpy as np
import argparse
import time
from tqdm import tqdm
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

from configs.gnn_geometric_config import GNNGeometricConfig, get_gnn_geometric_default
from datasets.cad_gnn_geometric_dataset import (
    CADGNNGeometricDataset, create_cad_gnn_data_loaders, cad_gnn_collate_fn
)
from models.gnn_geometric_classifier import create_gnn_geometric_model
from models.gnn_multitask_loss import create_multitask_loss


# ============================================================================
# Social Group Clustering Utilities
# (copied verbatim from train_cad_stage1.py)
# ============================================================================

class UnionFind:
    """Union-Find data structure for connected components"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_groups(self):
        groups_dict = defaultdict(set)
        for i in range(len(self.parent)):
            groups_dict[self.find(i)].add(i)
        return list(groups_dict.values())


def cluster_social_groups(interactions, threshold=0.5):
    """Cluster people into social groups based on pairwise interaction probabilities."""
    all_persons = set()
    for (pi, pj) in interactions.keys():
        all_persons.add(pi)
        all_persons.add(pj)

    if len(all_persons) == 0:
        return []

    person_list = sorted(list(all_persons))
    person_to_idx = {p: i for i, p in enumerate(person_list)}
    uf = UnionFind(len(person_list))

    for (pi, pj), prob in interactions.items():
        if prob > threshold:
            uf.union(person_to_idx[pi], person_to_idx[pj])

    return [{person_list[idx] for idx in g} for g in uf.get_groups()]


def compute_membership_accuracy(pred_groups, gt_groups):
    """Compute membership accuracy using Hungarian algorithm."""
    all_persons = set()
    for g in pred_groups + gt_groups:
        all_persons.update(g)

    if len(all_persons) == 0:
        return 1.0, 0, 0

    n_pred, n_gt = len(pred_groups), len(gt_groups)
    if n_pred == 0 or n_gt == 0:
        return 0.0, 0, len(all_persons)

    n_max = max(n_pred, n_gt)
    cost = np.zeros((n_max, n_max))

    for i in range(n_pred):
        for j in range(n_gt):
            inter = len(pred_groups[i] & gt_groups[j])
            union = len(pred_groups[i] | gt_groups[j])
            cost[i, j] = union - inter

    for i in range(n_pred, n_max):
        for j in range(n_gt):
            cost[i, j] = len(gt_groups[j])
    for i in range(n_pred):
        for j in range(n_gt, n_max):
            cost[i, j] = len(pred_groups[i])

    row_ind, col_ind = linear_sum_assignment(cost)
    correct = sum(
        len(pred_groups[i] & gt_groups[j])
        for i, j in zip(row_ind, col_ind)
        if i < n_pred and j < n_gt
    )
    total = len(all_persons)
    return correct / total if total > 0 else 0.0, correct, total


def compute_group_map(pred_groups, gt_groups, iou_threshold=0.5):
    """Compute mAP (as group-level F1) for group detection."""
    def iou(a, b):
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 0.0

    if not pred_groups or not gt_groups:
        return 0.0, 0.0, 0.0

    matched_pred, matched_gt = set(), set()
    for i, pg in enumerate(pred_groups):
        best_iou, best_j = 0.0, -1
        for j, gg in enumerate(gt_groups):
            v = iou(pg, gg)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_threshold and best_j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(best_j)

    nc = len(matched_pred)
    prec = nc / len(pred_groups)
    rec  = nc / len(gt_groups)
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


# ============================================================================
# Utilities
# ============================================================================

def parse_sequences(seq_str: str):
    """Parse '1,2,3' or '1-10' into a list of ints."""
    if seq_str is None or seq_str == '':
        return []
    if '-' in seq_str and ',' not in seq_str:
        lo, hi = map(int, seq_str.split('-'))
        return list(range(lo, hi + 1))
    return [int(s) for s in seq_str.split(',')]


# ============================================================================
# Trainer
# ============================================================================

class CADStage1GNNTrainer:

    def __init__(self, config: GNNGeometricConfig, device: torch.device):
        self.config = config
        self.device = device

        print("\nCreating GNN Geometric model (CAD Stage1)...")
        self.model = create_gnn_geometric_model(config).to(device)

        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {total:,}")
        print(f"  Node feat dim: {config.node_feat_dim}D  "
              f"Edge feat dim: {config.edge_feat_dim}D  "
              f"Pair feat dim: {config.pair_feat_dim}D")

        # Stage1: lambda_behavior=0, lambda_group=1
        self.criterion = create_multitask_loss(config).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        if config.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=1e-6)
        elif config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=15, gamma=0.5)
        else:
            self.scheduler = None

        self.best_val_f1      = 0.0
        self.epochs_no_improve = 0
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def _prepare_batch(self, batch):
        """batch: List[Dict] from cad_gnn_collate_fn"""
        scene_data, labels = [], []
        for item in batch:
            scene_data.append({
                'node_feats':      item['node_feats'].to(self.device),
                'pre_edge_index':  item['pre_edge_index'].to(self.device),
                'pre_edge_feats':  item['pre_edge_feats'].to(self.device),
                'target_pairs':    item['target_pairs'].to(self.device),
                'pair_flow_feats': item['pair_flow_feats'].to(self.device),
                'negative_pairs':  item['negative_pairs'].to(self.device),
            })
            labels.append(item['pair_labels'])
        all_labels = torch.cat(labels, dim=0).to(self.device)
        return scene_data, all_labels

    # ------------------------------------------------------------------

    def train_epoch(self, loader, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_g_loss = 0.0
        all_g_preds, all_g_labels = [], []
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            if not batch:
                continue
            scene_data, labels = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            out = self.model(scene_data)
            loss, loss_dict = self.criterion(
                out['behavior_logits'], labels,
                out['group_logits'],    out['group_labels'],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()

            total_loss   += loss_dict['total_loss']
            total_g_loss += loss_dict['group_loss']
            n_batches    += 1

            g_preds = (torch.sigmoid(out['group_logits'].squeeze(-1)) > 0.5).long()
            all_g_preds.extend(g_preds.cpu().tolist())
            all_g_labels.extend(out['group_labels'].cpu().tolist())

            pbar.set_postfix(
                loss=f"{loss_dict['total_loss']:.4f}",
                grp=f"{loss_dict['group_loss']:.4f}",
            )
        pbar.close()

        n = max(n_batches, 1)
        g_f1 = f1_score(all_g_labels, all_g_preds, average='binary', zero_division=0)
        print(f"Train Epoch {epoch}: Loss={total_loss/n:.4f}  "
              f"GrpLoss={total_g_loss/n:.4f}  GrpF1={g_f1:.4f}")
        return total_loss / n, g_f1

    # ------------------------------------------------------------------

    def validate_epoch(self, loader, epoch: int):
        self.model.eval()
        total_loss = 0.0
        total_g_loss = 0.0
        all_g_preds, all_g_labels = [], []
        n_batches = 0

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]  ")
            for batch in pbar:
                if not batch:
                    continue
                scene_data, labels = self._prepare_batch(batch)
                out = self.model(scene_data)
                _, loss_dict = self.criterion(
                    out['behavior_logits'], labels,
                    out['group_logits'],    out['group_labels'],
                )
                total_loss   += loss_dict['total_loss']
                total_g_loss += loss_dict['group_loss']
                n_batches    += 1

                g_preds = (torch.sigmoid(out['group_logits'].squeeze(-1)) > 0.5).long()
                all_g_preds.extend(g_preds.cpu().tolist())
                all_g_labels.extend(out['group_labels'].cpu().tolist())
            pbar.close()

        n = max(n_batches, 1)
        g_f1 = f1_score(all_g_labels, all_g_preds, average='binary', zero_division=0)
        print(f"Val   Epoch {epoch}: Loss={total_loss/n:.4f}  "
              f"GrpLoss={total_g_loss/n:.4f}  GrpF1={g_f1:.4f}")
        return total_loss / n, g_f1

    # ------------------------------------------------------------------

    def test(self, loader):
        """Test with social group clustering evaluation (same as train_cad_stage1.py)."""
        self.model.eval()
        print(f"\n{'='*80}\nFINAL TEST — Social Group Clustering\n{'='*80}")

        frame_data = {}   # {(seq_num, frame_id): {'interactions': {...}, 'track_to_group': {...}}}

        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting predictions"):
                if not batch:
                    continue
                # Process each frame individually to keep per-frame bookkeeping
                for item in batch:
                    seq_num  = item['seq_num']
                    frame_id = item['frame_id']
                    frame_key = (seq_num, frame_id)

                    scene = [{
                        'node_feats':      item['node_feats'].to(self.device),
                        'pre_edge_index':  item['pre_edge_index'].to(self.device),
                        'pre_edge_feats':  item['pre_edge_feats'].to(self.device),
                        'target_pairs':    item['target_pairs'].to(self.device),
                        'pair_flow_feats': item['pair_flow_feats'].to(self.device),
                        'negative_pairs':  item['negative_pairs'].to(self.device),
                    }]

                    out = self.model(scene)
                    g_logits = out['group_logits'].squeeze(-1)   # [P+Q]
                    g_probs  = torch.sigmoid(g_logits).cpu().numpy()

                    track_ids   = item['track_ids']          # list of ints
                    grp_ids     = item['social_group_ids']   # list of ints

                    # Reconstruct all pairs: target_pairs ++ negative_pairs
                    tp = item['target_pairs'].cpu()     # [P, 2]
                    np_ = item['negative_pairs'].cpu()  # [Q, 2]
                    all_pairs = torch.cat([tp, np_], dim=0)  # [P+Q, 2]

                    interactions = {}
                    for k, (ni, nj) in enumerate(all_pairs.tolist()):
                        ni, nj = int(ni), int(nj)
                        if ni < len(track_ids) and nj < len(track_ids):
                            ta = int(track_ids[ni])
                            tb = int(track_ids[nj])
                            pair_key = tuple(sorted([ta, tb]))
                            # Take max prob if same pair appears from both directions
                            interactions[pair_key] = max(
                                interactions.get(pair_key, 0.0),
                                float(g_probs[k])
                            )

                    track_to_group = {
                        int(track_ids[i]): int(grp_ids[i])
                        for i in range(len(track_ids))
                    }

                    frame_data[frame_key] = {
                        'interactions':   interactions,
                        'track_to_group': track_to_group,
                    }

        print(f"Collected {len(frame_data)} frames")

        # Evaluate at multiple thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []

        for thr in thresholds:
            total_correct, total_persons = 0, 0
            frame_accs, frame_maps = [], []

            for data in frame_data.values():
                pred_groups = cluster_social_groups(data['interactions'], thr)

                gt_dict = defaultdict(set)
                for tid, gid in data['track_to_group'].items():
                    gt_dict[gid].add(tid)
                gt_groups = list(gt_dict.values())

                if pred_groups and gt_groups:
                    acc, correct, total = compute_membership_accuracy(pred_groups, gt_groups)
                    frame_accs.append(acc)
                    total_correct += correct
                    total_persons += total
                    map_score, _, _ = compute_group_map(pred_groups, gt_groups)
                    frame_maps.append(map_score)

            overall_acc = total_correct / total_persons if total_persons > 0 else 0.0
            avg_acc     = float(np.mean(frame_accs)) if frame_accs else 0.0
            avg_map     = float(np.mean(frame_maps)) if frame_maps else 0.0
            results.append({'threshold': thr, 'membership_acc': overall_acc,
                            'avg_frame_acc': avg_acc, 'map': avg_map})

        # Print table
        print(f'\n{"="*80}')
        print('Social Group Clustering Results (Different Thresholds)')
        print(f'{"="*80}')
        print(f'{"Threshold":<12} {"Membership Acc":<18} {"Avg Frame Acc":<18} {"mAP":<12}')
        print(f'{"-"*80}')
        for r in results:
            print(f'{r["threshold"]:<12.1f} {r["membership_acc"]:<18.4f} '
                  f'{r["avg_frame_acc"]:<18.4f} {r["map"]:<12.4f}')
        print(f'{"="*80}')

        best = max(results, key=lambda x: x['membership_acc'])
        print(f'\nBest thr={best["threshold"]:.1f}  '
              f'MembershipAcc={best["membership_acc"]:.4f}  mAP={best["map"]:.4f}')
        return best['membership_acc'], best['map']

    # ------------------------------------------------------------------

    def _load_best_model(self):
        ckpt = os.path.join(self.config.checkpoint_dir, 'best_model_cad_s1_gnn.pth')
        if os.path.exists(ckpt):
            data = torch.load(ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(data['model_state_dict'])
            print(f"Loaded best model from epoch {data.get('epoch', '?')} "
                  f"(GrpF1={data.get('best_val_f1', 0):.4f})")
        else:
            print("Warning: best model checkpoint not found, using current weights")

    def _save_model(self, epoch, val_f1):
        ckpt = os.path.join(self.config.checkpoint_dir, 'best_model_cad_s1_gnn.pth')
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1':          val_f1,
            'config':               self.config.__dict__,
        }, ckpt)
        print(f"  [Saved] GrpF1={val_f1:.4f} → {ckpt}")

    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader, test_loader=None):
        print(f"\n{'='*80}\nSTARTING CAD STAGE1 GNN TRAINING\n{'='*80}\n")

        for epoch in range(1, self.config.epochs + 1):
            self.train_epoch(train_loader, epoch)
            _, val_f1 = self.validate_epoch(val_loader, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.epochs_no_improve = 0
                self._save_model(epoch, val_f1)
            else:
                self.epochs_no_improve += 1

            print(f"  Best GrpF1={self.best_val_f1:.4f} | No-improve={self.epochs_no_improve}")

            if self.epochs_no_improve >= self.config.early_stopping_patience:
                print("\nEarly stopping triggered.")
                break

        if test_loader is not None:
            self._load_best_model()
            self.test(test_loader)

        print(f"\n{'='*80}\nDONE  Best Val GrpF1={self.best_val_f1:.4f}\n{'='*80}")

    def train_no_val(self, train_loader, test_loader=None):
        """No-validation training (fixed epochs)."""
        print(f"\n{'='*80}\nSTARTING CAD STAGE1 GNN TRAINING (no val)\n{'='*80}\n")

        for epoch in range(1, self.config.epochs + 1):
            self.train_epoch(train_loader, epoch)
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final model as the best
        ckpt = os.path.join(self.config.checkpoint_dir, 'best_model_cad_s1_gnn.pth')
        torch.save({
            'epoch':                self.config.epochs,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config':               self.config.__dict__,
        }, ckpt)
        print(f"Final model saved → {ckpt}")

        if test_loader is not None:
            self.test(test_loader)

        print(f"\n{'='*80}\nDONE (no val)\n{'='*80}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='CAD Stage1 GNN Training: Binary Interaction Detection')

    # Dataset
    p.add_argument('--cad_root',        type=str,
                   default='../dataset/cad/ActivityDataset',
                   help='Path to CAD ActivityDataset directory')
    p.add_argument('--train_sequences', type=str,
                   default='30,31,32,33,34,35,36,37,38,39,40,41,42,43,44')
    p.add_argument('--val_sequences',   type=str,
                   default='1,2,3,4,12,13,14,17,18,19,20,21,22,23,24,26')
    p.add_argument('--test_sequences',  type=str,
                   default='5,6,7,8,9,10,11,15,16,25,28,29')
    p.add_argument('--image_width',     type=int, default=720)
    p.add_argument('--image_height',    type=int, default=480)

    # GNN architecture
    p.add_argument('--gnn_hidden',      type=int,   default=256)
    p.add_argument('--gnn_layers',      type=int,   default=2)
    p.add_argument('--gnn_heads',       type=int,   default=4)
    p.add_argument('--edge_hidden_dims', nargs='+', type=int, default=[256, 128, 64])
    p.add_argument('--no_graph_transformer', action='store_true')
    p.add_argument('--no_edge_in_cls',       action='store_true')
    p.add_argument('--no_inject_flow_to_edges', action='store_true')
    p.add_argument('--no_flow_node_feats',   action='store_true')
    p.add_argument('--graph_knn',       type=int,   default=0)

    # Structural improvements
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--drop_edge_rate',  type=float, default=0.0)
    p.add_argument('--use_virtual_node',    action='store_true')
    p.add_argument('--use_cross_pair_attn', action='store_true')
    p.add_argument('--cross_pair_dim',  type=int,   default=256)

    # Training
    p.add_argument('--batch_size',      type=int,   default=8)
    p.add_argument('--epochs',          type=int,   default=50)
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--weight_decay',    type=float, default=1e-4)
    p.add_argument('--num_workers',     type=int,   default=4)
    p.add_argument('--checkpoint_dir',  type=str,
                   default='./checkpoints/cad_stage1_gnn')
    p.add_argument('--group_neg_ratio', type=float, default=2.0)

    # Misc
    p.add_argument('--random_seed',     type=int,   default=42)
    p.add_argument('--cache_dir',       type=str,   default=None)
    p.add_argument('--run_test',        action='store_true')
    p.add_argument('--no_val',          action='store_true',
                   help='Merge train+val, train for fixed epochs, then test')

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    print(f"Random seed: {args.random_seed}")

    train_seqs = parse_sequences(args.train_sequences)
    val_seqs   = parse_sequences(args.val_sequences)
    test_seqs  = parse_sequences(args.test_sequences)

    if args.no_val:
        train_seqs = sorted(set(train_seqs + val_seqs))
        val_seqs   = []

    print(f"\nCAD Stage1 GNN Training: Binary Interaction Detection")
    print(f"  Train seqs: {train_seqs}")
    if not args.no_val:
        print(f"  Val seqs:   {val_seqs}")
    print(f"  Test seqs:  {test_seqs}")

    # --- Build config ---
    config = get_gnn_geometric_default()
    config.cad_root            = args.cad_root
    config.num_classes         = 2        # binary (behavior head unused in stage1)
    config.lambda_behavior     = 0.0      # no behavior loss for stage1
    config.lambda_group        = 1.0      # group detection only
    config.image_width         = args.image_width
    config.image_height        = args.image_height
    config.gnn_hidden_dim      = args.gnn_hidden
    config.gnn_num_layers      = args.gnn_layers
    config.gnn_num_heads       = args.gnn_heads
    config.edge_hidden_dims    = args.edge_hidden_dims
    config.batch_size          = args.batch_size
    config.epochs              = args.epochs
    config.learning_rate       = args.lr
    config.weight_decay        = args.weight_decay
    config.num_workers         = args.num_workers
    config.checkpoint_dir      = args.checkpoint_dir
    config.group_neg_ratio     = args.group_neg_ratio
    config.use_graph_transformer  = not args.no_graph_transformer
    config.use_edge_in_cls        = not args.no_edge_in_cls
    config.inject_flow_to_edges   = not args.no_inject_flow_to_edges
    config.flow_node_feats        = not args.no_flow_node_feats
    config.graph_knn           = args.graph_knn
    config.label_smoothing     = args.label_smoothing
    config.drop_edge_rate      = args.drop_edge_rate
    config.use_virtual_node    = args.use_virtual_node
    config.use_cross_pair_attn = args.use_cross_pair_attn
    config.cross_pair_dim      = args.cross_pair_dim
    config.random_seed         = args.random_seed
    config.cache_dir           = args.cache_dir
    # Re-derive feature dims after flags
    config.edge_feat_dim       = 17 if config.inject_flow_to_edges else 7
    config.node_feat_dim       = 13 if config.flow_node_feats else 5
    config.pair_feat_dim       = 11 if config.flow_node_feats else 10

    print(f"\nConfig summary:")
    for k in ('num_classes', 'lambda_behavior', 'lambda_group',
              'node_feat_dim', 'edge_feat_dim', 'pair_feat_dim',
              'gnn_hidden_dim', 'gnn_num_layers', 'gnn_num_heads',
              'use_graph_transformer', 'inject_flow_to_edges', 'flow_node_feats',
              'drop_edge_rate', 'use_virtual_node', 'use_cross_pair_attn'):
        print(f"  {k}: {getattr(config, k)}")

    # --- Data ---
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_cad_gnn_data_loaders(
        config, stage='stage1',
        seqs_train=train_seqs,
        seqs_val=val_seqs if not args.no_val else [],
        seqs_test=test_seqs,
        test_all_negatives=True,  # keep ALL pairs for clustering eval
    )
    print(f"  Train batches: {len(train_loader)}")
    if not args.no_val:
        print(f"  Val   batches: {len(val_loader)}")
    print(f"  Test  batches: {len(test_loader)}")

    # --- Train ---
    trainer = CADStage1GNNTrainer(config, device)
    start = time.time()

    if args.no_val:
        trainer.train_no_val(train_loader, test_loader)
    else:
        trainer.train(
            train_loader,
            val_loader,
            test_loader if args.run_test else None,
        )

    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")


if __name__ == '__main__':
    main()
