#!/usr/bin/env python3
"""
CAD Stage2 GNN Training: Group Activity Classification (No Visual Backbone)

Graph Transformer on geometric / optical-flow features.
Supports 6-class and 3-class (--class_merge) modes.

Evaluation:
  - Validation: MPCA (mean per-class accuracy) on pair-level behavior head
  - Test:       group-level majority-vote activity recognition
                (same protocol as train_cad_stage2.py)
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
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score

from configs.gnn_geometric_config import GNNGeometricConfig, get_gnn_geometric_default
from datasets.cad_gnn_geometric_dataset import (
    CADGNNGeometricDataset, create_cad_gnn_data_loaders, cad_gnn_collate_fn
)
from models.gnn_geometric_classifier import create_gnn_geometric_model
from models.gnn_multitask_loss import create_multitask_loss
from src.classifiers.geometric_stage2_classifier import Stage2Evaluator


# ============================================================================
# Group Activity Evaluation Helpers
# (adapted from train_cad_stage2.py)
# ============================================================================

def print_confusion_matrix(cm, class_names, indent=''):
    print(f'{indent}Rows = Ground Truth, Columns = Predicted')
    print(f'{indent}{"" :<12}', end='')
    for name in class_names:
        print(f'{name :<12}', end='')
    print()
    for i, row_name in enumerate(class_names):
        print(f'{indent}{row_name :<12}', end='')
        for j in range(len(class_names)):
            val = cm[i][j] if i < cm.shape[0] and j < cm.shape[1] else 0
            print(f'{val :<12}', end='')
        print()


def merge_moving_activities(activities):
    """
    Merge Crossing(1) and Walking(4) → Moving(1).
    Talking(5) → 4.  NA(0), Waiting(2), Queuing(3) unchanged.
    """
    merged = []
    for act in activities:
        if act in (1, 4):
            merged.append(1)
        elif act == 5:
            merged.append(4)
        else:
            merged.append(act)
    return merged


def compute_per_class_map(pred_activities, gt_activities, num_classes=6):
    """Per-class F1 as AP proxy; returns (mAP, per_class_ap, per_class_metrics)."""
    per_class_ap = {}
    per_class_metrics = {}

    for cls in range(num_classes):
        p_bin = [1 if p == cls else 0 for p in pred_activities]
        g_bin = [1 if g == cls else 0 for g in gt_activities]
        tp = sum(1 for a, b in zip(p_bin, g_bin) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(p_bin, g_bin) if a == 1 and b == 0)
        fn = sum(1 for a, b in zip(p_bin, g_bin) if a == 0 and b == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_ap[cls] = f1
        per_class_metrics[cls] = {
            'precision': prec, 'recall': rec, 'f1': f1,
            'support': sum(g_bin)
        }

    return float(np.mean(list(per_class_ap.values()))), per_class_ap, per_class_metrics


def compute_and_print_metrics(pred_activities, gt_activities, class_names,
                               group_type_name='All Groups', num_classes=6):
    if not pred_activities:
        print(f'\nNo {group_type_name} to evaluate')
        return None

    accuracy = accuracy_score(gt_activities, pred_activities)
    map_score, per_class_ap, per_class_metrics = compute_per_class_map(
        pred_activities, gt_activities, num_classes=num_classes)
    cm = confusion_matrix(gt_activities, pred_activities)

    print(f'\n{"=" * 80}')
    print(f'{group_type_name} — Metrics')
    print(f'{"=" * 80}')
    print(f'  Overall Accuracy: {accuracy:.4f}')
    print(f'  mAP (F1):        {map_score:.4f}')
    print(f'  Total Groups:     {len(pred_activities)}')
    print(f'\n{"Class":<12} {"Precision":<12} {"Recall":<12} {"F1 (AP)":<12} {"Support":<10}')
    print(f'{"-" * 80}')
    for cls in range(num_classes):
        m = per_class_metrics[cls]
        name = class_names[cls] if cls < len(class_names) else f'Class{cls}'
        print(f'{name:<12} {m["precision"]:<12.4f} {m["recall"]:<12.4f} '
              f'{m["f1"]:<12.4f} {m["support"]:<10}')
    print(f'\nConfusion Matrix:')
    print_confusion_matrix(cm, class_names)

    return {
        'overall_accuracy': accuracy,
        'map': map_score,
        'per_class_ap': {class_names[i]: per_class_ap[i] for i in range(num_classes)},
        'confusion_matrix': cm.tolist(),
        'num_groups': len(pred_activities),
    }


def find_nearest_group(person_bbox_xywh, other_groups, group_bboxes_xywh):
    """Find the nearest (by centre distance) group to a lone person."""
    def centre(bb):  # [x, y, w, h] → (cx, cy)
        return (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)

    def dist(c1, c2):
        return float(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))

    cx, cy = centre(person_bbox_xywh)
    best_d, best_gid = float('inf'), None

    for gid in other_groups:
        bboxes = group_bboxes_xywh[gid]
        ctrs = [centre(bb) for bb in bboxes.values()]
        gcx = sum(c[0] for c in ctrs) / len(ctrs)
        gcy = sum(c[1] for c in ctrs) / len(ctrs)
        d = dist((cx, cy), (gcx, gcy))
        if d < best_d:
            best_d, best_gid = d, gid

    return best_gid


def vote_group_activity(pair_predictions, group_members, person_bboxes_xywh,
                        all_groups, group_members_dict):
    """
    Majority-vote group activity from pairwise predictions.
    Single-person groups: inherit activity from nearest multi-person group.
    """
    if len(group_members) == 1:
        pid = next(iter(group_members))
        bbox = person_bboxes_xywh.get(pid)
        multi_groups = [g for g in all_groups if len(group_members_dict[g]) > 1]
        if not multi_groups or bbox is None:
            return 0, 0.0, {'default': 1}
        group_bboxes = {g: {p: person_bboxes_xywh[p]
                             for p in group_members_dict[g]
                             if p in person_bboxes_xywh}
                        for g in multi_groups}
        nearest = find_nearest_group(bbox, multi_groups, group_bboxes)
        act, conf, _ = vote_group_activity(
            pair_predictions, group_members_dict[nearest],
            person_bboxes_xywh, all_groups, group_members_dict)
        return act, conf * 0.5, {'inherited_from_nearest': 1}

    votes = [
        act for (pi, pj), act in pair_predictions.items()
        if pi in group_members and pj in group_members
    ]
    if not votes:
        return 0, 0.0, {'no_pairs': 1}
    cnt = Counter(votes)
    act = cnt.most_common(1)[0][0]
    return act, cnt[act] / len(votes), dict(cnt)


# ============================================================================
# Utilities
# ============================================================================

def parse_sequences(seq_str: str):
    if seq_str is None or seq_str == '':
        return []
    if '-' in seq_str and ',' not in seq_str:
        lo, hi = map(int, seq_str.split('-'))
        return list(range(lo, hi + 1))
    return [int(s) for s in seq_str.split(',')]


# ============================================================================
# Trainer
# ============================================================================

class CADStage2GNNTrainer:

    def __init__(self, config: GNNGeometricConfig, device: torch.device):
        self.config      = config
        self.device      = device
        self.class_merge = getattr(config, 'class_merge', False)

        if self.class_merge:
            self.class_names = ['Moving', 'Standing', 'Talking']
        else:
            self.class_names = ['NA', 'Crossing', 'Waiting', 'Queuing', 'Walking', 'Talking']

        print(f"\nCreating GNN Geometric model (CAD Stage2, {config.num_classes} classes)...")
        self.model = create_gnn_geometric_model(config).to(device)

        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {total:,}")
        print(f"  Node feat dim: {config.node_feat_dim}D  "
              f"Edge feat dim: {config.edge_feat_dim}D  "
              f"Pair feat dim: {config.pair_feat_dim}D")
        print(f"  Class names: {self.class_names}")

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

        self.best_val_mpca     = 0.0
        self.best_val_acc      = 0.0
        self.best_val_macro_f1 = 0.0
        self.epochs_no_improve = 0
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def _prepare_batch(self, batch):
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
        evaluator  = Stage2Evaluator(self.class_names)
        total_loss = total_b = total_g = 0.0
        n_batches  = 0

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

            total_loss += loss_dict['total_loss']
            total_b    += loss_dict['behavior_loss']
            total_g    += loss_dict['group_loss']
            n_batches  += 1

            preds = out['behavior_logits'].argmax(dim=1)
            evaluator.update(preds.cpu().numpy(), labels.cpu().numpy())

            pbar.set_postfix(
                loss=f"{loss_dict['total_loss']:.4f}",
                beh=f"{loss_dict['behavior_loss']:.4f}",
                grp=f"{loss_dict['group_loss']:.4f}",
            )
        pbar.close()

        n = max(n_batches, 1)
        m = evaluator.compute_metrics()
        print(f"Train Epoch {epoch}: Loss={total_loss/n:.4f} "
              f"(beh={total_b/n:.4f}, grp={total_g/n:.4f})  "
              f"Acc={m.get('overall_accuracy',0):.4f}  "
              f"MPCA={m.get('mpca',0):.4f}  "
              f"MacroF1={m.get('macro_f1',0):.4f}")
        return total_loss / n, m.get('overall_accuracy', 0), m.get('mpca', 0)

    # ------------------------------------------------------------------

    def validate_epoch(self, loader, epoch: int):
        self.model.eval()
        evaluator  = Stage2Evaluator(self.class_names)
        total_loss = total_b = total_g = 0.0
        n_batches  = 0

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
                total_loss += loss_dict['total_loss']
                total_b    += loss_dict['behavior_loss']
                total_g    += loss_dict['group_loss']
                n_batches  += 1
                preds = out['behavior_logits'].argmax(dim=1)
                evaluator.update(preds.cpu().numpy(), labels.cpu().numpy())
            pbar.close()

        n = max(n_batches, 1)
        m = evaluator.compute_metrics()
        acc      = m.get('overall_accuracy', 0)
        mpca     = m.get('mpca', 0)
        macro_f1 = m.get('macro_f1', 0)
        print(f"Val   Epoch {epoch}: Loss={total_loss/n:.4f} "
              f"(beh={total_b/n:.4f}, grp={total_g/n:.4f})  "
              f"Acc={acc:.4f}  MPCA={mpca:.4f}  MacroF1={macro_f1:.4f}")
        return total_loss / n, acc, mpca, macro_f1

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _print_pair_metrics(self, all_gt, all_pred, title='Pair-Level Classification'):
        """Print confusion matrix and per-class acc/precision/recall/F1."""
        n_cls = self.config.num_classes
        names = self.class_names

        gt   = np.array(all_gt,   dtype=int)
        pred = np.array(all_pred, dtype=int)
        total = len(gt)

        # Confusion matrix  (rows=GT, cols=Pred)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for g, p in zip(gt, pred):
            if 0 <= g < n_cls and 0 <= p < n_cls:
                cm[g, p] += 1

        print(f"\n{'='*80}")
        print(f"{title} — Prediction Distribution Matrix")
        print(f"  Total pairs: {total}  |  Overall Acc: {(gt==pred).mean():.4f}")
        print(f"{'='*80}")
        print(f"Rows = Ground Truth, Columns = Predicted")

        # Header
        col_w = 10
        print(f"{'':12}", end='')
        for name in names:
            print(f"{name[:col_w]:<{col_w}}", end='')
        print(f"{'Total':>{col_w}}")
        print('-' * (12 + col_w * (n_cls + 1)))

        for i, name in enumerate(names):
            print(f"{name[:12]:<12}", end='')
            for j in range(n_cls):
                print(f"{cm[i,j]:<{col_w}}", end='')
            print(f"{cm[i].sum():<{col_w}}")

        # Column totals
        print(f"{'Total':<12}", end='')
        for j in range(n_cls):
            print(f"{cm[:,j].sum():<{col_w}}", end='')
        print(f"{cm.sum():<{col_w}}")

        # Per-class metrics
        print(f"\n{'='*80}")
        print(f"{title} — Per-Class Metrics")
        print(f"{'='*80}")
        print(f"{'Class':<12} {'Acc':<10} {'Precision':<12} {'Recall':<10} "
              f"{'F1':<10} {'Support':<10} {'Predicted':<10}")
        print('-' * 80)

        per_class = {}
        for c in range(n_cls):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum()) - tp          # predicted c but not c
            fn = int(cm[c, :].sum()) - tp           # actual c but not predicted c
            sup = int(cm[c, :].sum())               # actual count
            pred_total = int(cm[:, c].sum())        # predicted count

            prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            acc_c  = tp / sup if sup > 0 else 0.0  # per-class recall = class acc

            name = names[c] if c < len(names) else f'Class{c}'
            print(f"{name:<12} {acc_c:<10.4f} {prec:<12.4f} {rec:<10.4f} "
                  f"{f1:<10.4f} {sup:<10} {pred_total:<10}")
            per_class[c] = {'acc': acc_c, 'precision': prec,
                             'recall': rec, 'f1': f1, 'support': sup}

        # Summary row
        mpca     = float(np.mean([v['acc']  for v in per_class.values() if v['support'] > 0]))
        macro_p  = float(np.mean([v['precision'] for v in per_class.values()]))
        macro_r  = float(np.mean([v['recall']    for v in per_class.values()]))
        macro_f1 = float(np.mean([v['f1']        for v in per_class.values()]))
        print('-' * 80)
        print(f"{'Macro Avg':<12} {mpca:<10.4f} {macro_p:<12.4f} {macro_r:<10.4f} "
              f"{macro_f1:<10.4f} {total:<10}")
        print(f"{'='*80}")
        return per_class, mpca, macro_f1

    # ------------------------------------------------------------------

    def test(self, loader):
        """
        Test with group-level majority-vote activity recognition.
        Also prints pair-level confusion matrix and per-class metrics.
        Same evaluation protocol as train_cad_stage2.py.
        """
        self.model.eval()
        print(f"\n{'='*80}\nFINAL TEST — Group Activity Recognition\n{'='*80}")
        print(f"Class mode: {'3-class merged' if self.class_merge else '6-class'}")

        # frame_data[key] = {
        #   'pair_predictions': {(ta,tb): pred_class},
        #   'person_bboxes': {track_id: [x,y,w,h]},
        #   'gt_groups': {group_id: {'activity': int, 'members': set}}
        # }
        frame_data = {}
        all_pair_preds = []   # pair-level: model output class
        all_pair_gts   = []   # pair-level: GT activity label

        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting predictions"):
                if not batch:
                    continue
                for item in batch:
                    seq_num  = item['seq_num']
                    frame_id = item['frame_id']
                    key      = (seq_num, frame_id)

                    scene = [{
                        'node_feats':      item['node_feats'].to(self.device),
                        'pre_edge_index':  item['pre_edge_index'].to(self.device),
                        'pre_edge_feats':  item['pre_edge_feats'].to(self.device),
                        'target_pairs':    item['target_pairs'].to(self.device),
                        'pair_flow_feats': item['pair_flow_feats'].to(self.device),
                        'negative_pairs':  item['negative_pairs'].to(self.device),
                    }]
                    out = self.model(scene)
                    preds = out['behavior_logits'].argmax(dim=1).cpu().tolist()  # [P]

                    track_ids     = item['track_ids']           # list[int], len=N
                    grp_ids       = item['social_group_ids']    # list[int], len=N
                    act_ids       = item['social_activity_ids'] # list[int], len=N
                    target_pairs  = item['target_pairs'].cpu()  # [P, 2]
                    pair_labels   = item['pair_labels'].tolist()  # [P] GT activity
                    person_boxes  = item['person_boxes']        # [N, 4] xywh

                    # Collect pair-level predictions for detailed metrics
                    all_pair_preds.extend(preds)
                    all_pair_gts.extend(pair_labels)

                    if key not in frame_data:
                        frame_data[key] = {
                            'pair_predictions': {},
                            'person_bboxes':    {},
                            'gt_groups':        {},
                        }

                    fd = frame_data[key]

                    # Build person bboxes {track_id: [x,y,w,h]}
                    for ni in range(len(track_ids)):
                        tid = int(track_ids[ni])
                        fd['person_bboxes'][tid] = person_boxes[ni].tolist()

                    # Build pair predictions and GT groups
                    for pi, (na, nb) in enumerate(target_pairs.tolist()):
                        na, nb = int(na), int(nb)
                        ta = int(track_ids[na])
                        tb = int(track_ids[nb])
                        pair_key = tuple(sorted([ta, tb]))
                        fd['pair_predictions'][pair_key] = preds[pi]

                        ga    = int(grp_ids[na])
                        act_a = int(act_ids[na])
                        if ga not in fd['gt_groups']:
                            fd['gt_groups'][ga] = {'activity': act_a, 'members': set()}
                        fd['gt_groups'][ga]['members'].add(ta)
                        fd['gt_groups'][ga]['members'].add(tb)

                    # Add lone persons from all nodes
                    for ni in range(len(track_ids)):
                        tid = int(track_ids[ni])
                        gid = int(grp_ids[ni])
                        act = int(act_ids[ni])
                        if gid not in fd['gt_groups']:
                            fd['gt_groups'][gid] = {'activity': act, 'members': set()}
                        fd['gt_groups'][gid]['members'].add(tid)

        print(f"Collected {len(frame_data)} frames, {len(all_pair_preds)} pairs")

        # ================================================================
        # Pair-level metrics (direct model output before group voting)
        # ================================================================
        self._print_pair_metrics(all_pair_gts, all_pair_preds,
                                  title='Pair-Level Classification')

        # --- Aggregate group-level predictions ---
        all_pred,    all_gt    = [], []
        single_pred, single_gt = [], []
        multi_pred,  multi_gt  = [], []
        n_single = n_multi = 0

        for fd in frame_data.values():
            pair_preds = fd['pair_predictions']
            bboxes     = fd['person_bboxes']
            gt_groups  = fd['gt_groups']

            group_members_dict = {g: info['members'] for g, info in gt_groups.items()}
            all_groups         = list(group_members_dict.keys())

            for gid, info in gt_groups.items():
                members    = info['members']
                gt_activity = info['activity']

                pred_act, _, _ = vote_group_activity(
                    pair_preds, members, bboxes, all_groups, group_members_dict)

                all_pred.append(pred_act)
                all_gt.append(gt_activity)

                if len(members) == 1:
                    single_pred.append(pred_act)
                    single_gt.append(gt_activity)
                    n_single += 1
                else:
                    multi_pred.append(pred_act)
                    multi_gt.append(gt_activity)
                    n_multi += 1

        print(f"\nTotal groups: {len(all_pred)}  "
              f"(single={n_single}, multi={n_multi})")

        n_cls = self.config.num_classes

        if self.class_merge:
            # 3-class evaluation
            all_res    = compute_and_print_metrics(all_pred, all_gt, self.class_names,
                                                    'All Groups (3-class)', n_cls)
            single_res = compute_and_print_metrics(single_pred, single_gt, self.class_names,
                                                    'Single-Person (3-class)', n_cls)
            multi_res  = compute_and_print_metrics(multi_pred, multi_gt, self.class_names,
                                                    'Multi-Person (3-class)', n_cls)
            return (all_res or {}).get('overall_accuracy', 0), \
                   (all_res or {}).get('map', 0)

        else:
            # 6-class evaluation
            cls6 = ['NA', 'Crossing', 'Waiting', 'Queuing', 'Walking', 'Talking']
            all_res    = compute_and_print_metrics(all_pred, all_gt, cls6,
                                                    'All Groups (6-class)', 6)
            single_res = compute_and_print_metrics(single_pred, single_gt, cls6,
                                                    'Single-Person (6-class)', 6)
            multi_res  = compute_and_print_metrics(multi_pred, multi_gt, cls6,
                                                    'Multi-Person (6-class)', 6)

            # Also 5-class merged evaluation
            cls5 = ['NA', 'Moving', 'Waiting', 'Queuing', 'Talking']
            all5_pred  = merge_moving_activities(all_pred)
            all5_gt    = merge_moving_activities(all_gt)
            s5_pred    = merge_moving_activities(single_pred)
            s5_gt      = merge_moving_activities(single_gt)
            m5_pred    = merge_moving_activities(multi_pred)
            m5_gt      = merge_moving_activities(multi_gt)

            print(f'\n{"=" * 80}')
            print('Merged (Crossing+Walking → Moving, 5 classes)')
            print(f'{"=" * 80}')
            compute_and_print_metrics(all5_pred, all5_gt, cls5,
                                       'All Groups (5-class merged)', 5)
            compute_and_print_metrics(s5_pred, s5_gt, cls5,
                                       'Single-Person (5-class merged)', 5)
            compute_and_print_metrics(m5_pred, m5_gt, cls5,
                                       'Multi-Person (5-class merged)', 5)

            return (all_res or {}).get('overall_accuracy', 0), \
                   (all_res or {}).get('map', 0)

    # ------------------------------------------------------------------

    def _load_best_model(self):
        ckpt = os.path.join(self.config.checkpoint_dir, 'best_model_cad_s2_gnn.pth')
        if os.path.exists(ckpt):
            data = torch.load(ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(data['model_state_dict'])
            print(f"Loaded best model from epoch {data.get('epoch', '?')} "
                  f"(MPCA={data.get('best_val_mpca', 0):.4f})")
        else:
            print("Warning: best model checkpoint not found, using current weights")

    def _save_model(self, epoch, val_mpca, val_acc, val_macro_f1):
        ckpt = os.path.join(self.config.checkpoint_dir, 'best_model_cad_s2_gnn.pth')
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_mpca':        val_mpca,
            'best_val_acc':         val_acc,
            'best_val_macro_f1':    val_macro_f1,
            'config':               self.config.__dict__,
        }, ckpt)
        print(f"  [Saved] MPCA={val_mpca:.4f} Acc={val_acc:.4f} → {ckpt}")

    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader, test_loader=None):
        print(f"\n{'='*80}\nSTARTING CAD STAGE2 GNN TRAINING\n{'='*80}\n")

        for epoch in range(1, self.config.epochs + 1):
            self.train_epoch(train_loader, epoch)
            _, val_acc, val_mpca, val_macro_f1 = self.validate_epoch(val_loader, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            if val_mpca > self.best_val_mpca:
                self.best_val_mpca     = val_mpca
                self.best_val_acc      = val_acc
                self.best_val_macro_f1 = val_macro_f1
                self.epochs_no_improve = 0
                self._save_model(epoch, val_mpca, val_acc, val_macro_f1)
            else:
                self.epochs_no_improve += 1

            print(f"  Best MPCA={self.best_val_mpca:.4f}  "
                  f"MacroF1={self.best_val_macro_f1:.4f} | "
                  f"No-improve={self.epochs_no_improve}")

            if self.epochs_no_improve >= self.config.early_stopping_patience:
                print("\nEarly stopping triggered.")
                break

        if test_loader is not None:
            self._load_best_model()
            self.test(test_loader)

        print(f"\n{'='*80}\nDONE  Best Val MPCA={self.best_val_mpca:.4f}  "
              f"MacroF1={self.best_val_macro_f1:.4f}\n{'='*80}")

    def train_no_val(self, train_loader, test_loader=None):
        """No-validation training (fixed epochs)."""
        print(f"\n{'='*80}\nSTARTING CAD STAGE2 GNN TRAINING (no val)\n{'='*80}\n")

        for epoch in range(1, self.config.epochs + 1):
            self.train_epoch(train_loader, epoch)
            if self.scheduler is not None:
                self.scheduler.step()

        ckpt = os.path.join(self.config.checkpoint_dir, 'best_model_cad_s2_gnn.pth')
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
        description='CAD Stage2 GNN Training: Group Activity Classification')

    # Dataset
    p.add_argument('--cad_root',        type=str,
                   default='../dataset/cad/ActivityDataset')
    p.add_argument('--train_sequences', type=str,
                   default='3,4,14,17,18,20,21,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44')
    p.add_argument('--val_sequences',   type=str,
                   default='1,2,12,13,19,22,23,24,26')
    p.add_argument('--test_sequences',  type=str,
                   default='5,6,7,8,9,10,11,15,16,25,28,29')
    p.add_argument('--image_width',     type=int, default=720)
    p.add_argument('--image_height',    type=int, default=480)
    p.add_argument('--class_merge',     action='store_true',
                   help='Merge 6 classes to 3 (Moving/Standing/Talking), skip NA')

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

    # Multi-task loss
    p.add_argument('--lambda_behavior', type=float, default=0.8)
    p.add_argument('--lambda_group',    type=float, default=0.2)
    p.add_argument('--group_neg_ratio', type=float, default=2.0)

    # Training
    p.add_argument('--batch_size',      type=int,   default=8)
    p.add_argument('--epochs',          type=int,   default=50)
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--weight_decay',    type=float, default=1e-4)
    p.add_argument('--num_workers',     type=int,   default=4)
    p.add_argument('--checkpoint_dir',  type=str,
                   default='./checkpoints/cad_stage2_gnn')

    # Feature improvements (CAD-specific)
    p.add_argument('--use_individual_action_feat', action='store_true',
                   help='CAD: append individual_action_id + group_size_norm to node feats (+2D)')
    p.add_argument('--use_extra_pair_feats', action='store_true',
                   help='CAD: append pair axis angle + lateral rate to pair feats (+2D or +3D)')

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

    n_cls = 3 if args.class_merge else 6

    print(f"\nCAD Stage2 GNN Training: {'3-class merged' if args.class_merge else '6-class'}")
    print(f"  Train seqs: {train_seqs}")
    if not args.no_val:
        print(f"  Val seqs:   {val_seqs}")
    print(f"  Test seqs:  {test_seqs}")

    # --- Build config ---
    config = get_gnn_geometric_default()
    config.cad_root            = args.cad_root
    config.num_classes         = n_cls
    config.class_merge         = args.class_merge
    config.class_weights       = None   # auto-computed below
    config.lambda_behavior     = args.lambda_behavior
    config.lambda_group        = args.lambda_group
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
    config.use_individual_action_feat = args.use_individual_action_feat
    config.use_extra_pair_feats       = args.use_extra_pair_feats
    # Re-derive feature dims after flags
    config.edge_feat_dim = 17 if config.inject_flow_to_edges else 7
    config.node_feat_dim = 13 if config.flow_node_feats else 5
    config.pair_feat_dim = 11 if config.flow_node_feats else 10
    if config.use_individual_action_feat:
        config.node_feat_dim += 2   # +individual_action_id + group_size_norm
    if config.use_extra_pair_feats:
        # +cos_angle + sin_angle [+ lateral_rate if flow enabled]
        config.pair_feat_dim += 3 if config.flow_node_feats else 2

    print(f"\nConfig summary:")
    for k in ('num_classes', 'class_merge', 'lambda_behavior', 'lambda_group',
              'node_feat_dim', 'edge_feat_dim', 'pair_feat_dim',
              'gnn_hidden_dim', 'gnn_num_layers', 'gnn_num_heads',
              'use_graph_transformer', 'inject_flow_to_edges', 'flow_node_feats',
              'drop_edge_rate', 'use_virtual_node', 'use_cross_pair_attn',
              'use_individual_action_feat', 'use_extra_pair_feats'):
        print(f"  {k}: {getattr(config, k)}")

    # --- Data ---
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_cad_gnn_data_loaders(
        config, stage='stage2',
        seqs_train=train_seqs,
        seqs_val=val_seqs if not args.no_val else [],
        seqs_test=test_seqs,
        test_all_negatives=False,  # stage2: test uses same neg ratio
    )
    print(f"  Train batches: {len(train_loader)}")
    if not args.no_val:
        print(f"  Val   batches: {len(val_loader)}")
    print(f"  Test  batches: {len(test_loader)}")

    # Auto class weights from training data (sqrt inverse-frequency, median-normalised).
    # Softer than plain inverse-frequency: gives minority classes higher weights
    # without shrinking the majority class weight to near zero.
    dist = train_loader.dataset.get_class_distribution()
    if dist.get('class_counts'):
        counts = dist['class_counts']
        total  = dist['total_pairs']
        config.class_weights = {
            c: (total / counts[c]) ** 0.5 if c in counts else 1.0
            for c in range(n_cls)
        }
        med = sorted(config.class_weights.values())[n_cls // 2]
        if med > 0:
            config.class_weights = {c: w / med for c, w in config.class_weights.items()}
        print(f"Auto class weights (sqrt-inv-freq, median-norm): {config.class_weights}")

    # --- Train ---
    trainer = CADStage2GNNTrainer(config, device)
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
