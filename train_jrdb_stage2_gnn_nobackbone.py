"""
GNN Geometric-Only Stage2 Training Script (No Backbone) – Multi-task
Trains a Graph Transformer that jointly optimises:
  1. Behavior classification (3-class Focal Loss): Walking/Standing/Sitting Together
  2. Group detection (Binary Focal Loss): interacting vs. non-interacting pair

Analogous to train_jrdb_stage2_nobackbone.py (MLP on 10D geometric+flow features),
but replaces the MLP with a scene-level Graph Transformer that captures inter-person
context and learns to update edge features across layers.

Key characteristics:
  - No ResNet backbone (no visual feature extraction)
  - Node features: [cx_norm, cy_norm, w_norm, h_norm, aspect_ratio] (5D bbox, per person)
  - GraphTransformerLayer: jointly updates node AND edge features each layer
  - GAT edge features: 7D geometric → edge_hidden_dim (evolved across layers)
  - Pair flow features: 10D from GeometricFlowExtractor + interaction synchrony (labeled pairs)
    → same 10D feature set as the original nobackbone MLP, appended to behavior classifier
  - Negative pairs: unlabeled pairs sampled per scene for grouping supervision
  - Multi-task loss: L = λ_behavior * L_behavior + λ_group * L_group
  - Batch unit: frame (scene), not pair
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
from collections import Counter

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

from configs.gnn_geometric_config import GNNGeometricConfig, get_gnn_geometric_default
from datasets.gnn_geometric_dataset import (
    GNNGeometricDataset, create_gnn_geometric_data_loaders, gnn_geometric_collate_fn
)
from models.gnn_geometric_classifier import GNNGeometricClassifier, create_gnn_geometric_model
from models.gnn_multitask_loss import GNNMultiTaskLoss, create_multitask_loss
from src.classifiers.geometric_stage2_classifier import Stage2Evaluator


CLASS_NAMES = ['Walking Together', 'Standing Together', 'Sitting Together']


# ============================================================================
# Trainer
# ============================================================================

class GNNGeometricTrainer:

    def __init__(self, config: GNNGeometricConfig, device: torch.device):
        self.config = config
        self.device = device

        print("\nCreating GNN Geometric model...")
        self.model = create_gnn_geometric_model(config).to(device)

        self.criterion = create_multitask_loss(config).to(device)

        # Single param group – no backbone, all params treated equally
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
        """batch: List[Dict] from gnn_geometric_collate_fn"""
        scene_data, labels = [], []
        for item in batch:
            scene_data.append({
                'node_feats':      item['node_feats'].to(self.device),
                'pre_edge_index':  item['pre_edge_index'].to(self.device),     # [2, E]
                'pre_edge_feats':  item['pre_edge_feats'].to(self.device),     # [E, 7]
                'target_pairs':    item['target_pairs'].to(self.device),
                'pair_flow_feats': item['pair_flow_feats'].to(self.device),    # [P, 10]
                'negative_pairs':  item['negative_pairs'].to(self.device),     # [Q, 2]
            })
            labels.append(item['pair_labels'])
        all_labels = torch.cat(labels, dim=0).to(self.device)
        return scene_data, all_labels

    # ------------------------------------------------------------------

    def train_epoch(self, loader, epoch: int):
        self.model.train()
        evaluator = Stage2Evaluator(CLASS_NAMES)
        total_loss = 0.0
        total_b_loss = 0.0
        total_g_loss = 0.0
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
                out['group_logits'],   out['group_labels'],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()

            total_loss   += loss_dict['total_loss']
            total_b_loss += loss_dict['behavior_loss']
            total_g_loss += loss_dict['group_loss']
            n_batches    += 1
            preds = out['behavior_logits'].argmax(dim=1)
            evaluator.update(preds.cpu().numpy(), labels.cpu().numpy())
            pbar.set_postfix(
                loss=f"{loss_dict['total_loss']:.4f}",
                b=f"{loss_dict['behavior_loss']:.4f}",
                g=f"{loss_dict['group_loss']:.4f}",
            )

        pbar.close()
        n = max(n_batches, 1)
        avg_loss   = total_loss   / n
        avg_b_loss = total_b_loss / n
        avg_g_loss = total_g_loss / n
        metrics  = evaluator.compute_metrics()
        mpca     = metrics.get('mpca', 0.0)
        acc      = metrics.get('overall_accuracy', 0.0)
        macro_f1 = metrics.get('macro_f1', 0.0)
        print(f"Train Epoch {epoch}: Loss={avg_loss:.4f} "
              f"(beh={avg_b_loss:.4f}, grp={avg_g_loss:.4f}), "
              f"Acc={acc:.4f}, MPCA={mpca:.4f}, MacroF1={macro_f1:.4f}")
        return avg_loss, acc, mpca, macro_f1

    # ------------------------------------------------------------------

    def validate_epoch(self, loader, epoch: int):
        self.model.eval()
        evaluator = Stage2Evaluator(CLASS_NAMES)
        total_loss = 0.0
        total_b_loss = 0.0
        total_g_loss = 0.0
        total_g_acc = 0.0
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
                    out['group_logits'],   out['group_labels'],
                )
                total_loss   += loss_dict['total_loss']
                total_b_loss += loss_dict['behavior_loss']
                total_g_loss += loss_dict['group_loss']
                total_g_acc  += loss_dict['group_acc']
                n_batches    += 1
                preds = out['behavior_logits'].argmax(dim=1)
                evaluator.update(preds.cpu().numpy(), labels.cpu().numpy())
            pbar.close()

        n = max(n_batches, 1)
        avg_loss   = total_loss   / n
        avg_b_loss = total_b_loss / n
        avg_g_loss = total_g_loss / n
        avg_g_acc  = total_g_acc  / n
        metrics  = evaluator.compute_metrics()
        acc      = metrics.get('overall_accuracy', 0.0)
        mpca     = metrics.get('mpca', 0.0)
        macro_f1 = metrics.get('macro_f1', 0.0)
        print(f"Val   Epoch {epoch}: Loss={avg_loss:.4f} "
              f"(beh={avg_b_loss:.4f}, grp={avg_g_loss:.4f}), "
              f"Acc={acc:.4f}, MPCA={mpca:.4f}, MacroF1={macro_f1:.4f}, GrpAcc={avg_g_acc:.4f}")
        return avg_loss, acc, mpca, macro_f1

    # ------------------------------------------------------------------

    def test(self, loader):
        self.model.eval()
        evaluator = Stage2Evaluator(CLASS_NAMES)
        print(f"\n{'='*80}\nFINAL TEST EVALUATION\n{'='*80}")
        with torch.no_grad():
            for batch in tqdm(loader, desc="Testing"):
                if not batch:
                    continue
                scene_data, labels = self._prepare_batch(batch)
                out = self.model(scene_data)
                preds = out['behavior_logits'].argmax(dim=1)
                evaluator.update(preds.cpu().numpy(), labels.cpu().numpy())
        evaluator.print_evaluation_report()
        metrics = evaluator.compute_metrics()
        print(f"\nTest Acc={metrics.get('overall_accuracy',0):.4f}  "
              f"MPCA={metrics.get('mpca',0):.4f}  "
              f"MacroF1={metrics.get('macro_f1',0):.4f}")
        return metrics

    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader, test_loader=None):
        print(f"\n{'='*80}\nSTARTING GNN GEOMETRIC TRAINING\n{'='*80}\n")

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

                ckpt = os.path.join(self.config.checkpoint_dir,
                                    'best_model_gnn_geometric.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_mpca': self.best_val_mpca,
                    'config': self.config.__dict__,
                }, ckpt)
                print(f"  [Saved] MPCA={self.best_val_mpca:.4f} → {ckpt}")
            else:
                self.epochs_no_improve += 1

            print(f"  Best MPCA={self.best_val_mpca:.4f} MacroF1={self.best_val_macro_f1:.4f} | "
                  f"No-improve={self.epochs_no_improve}")

            if self.epochs_no_improve >= self.config.early_stopping_patience:
                print("\nEarly stopping triggered.")
                break

        if test_loader is not None:
            ckpt = os.path.join(self.config.checkpoint_dir,
                                'best_model_gnn_geometric.pth')
            if os.path.exists(ckpt):
                data = torch.load(ckpt, map_location=self.device)
                self.model.load_state_dict(data['model_state_dict'])
                print(f"Loaded best model from epoch {data['epoch']}")
            self.test(test_loader)

        print(f"\n{'='*80}\nTRAINING DONE  "
              f"Best Val MPCA={self.best_val_mpca:.4f}  "
              f"MacroF1={self.best_val_macro_f1:.4f}  "
              f"Acc={self.best_val_acc:.4f}\n{'='*80}")


# ============================================================================
# FLOPs
# ============================================================================

def estimate_flops(config, device):
    if not THOP_AVAILABLE:
        print("thop not available – skip FLOPs")
        return
    model = create_gnn_geometric_model(config).to(device).eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"\nGNN Geometric model: {total:,} parameters (no backbone)")
    print(f"Node features: {config.node_feat_dim}D (bounding-box only)")
    print(f"Edge features: {config.edge_feat_dim}D (geometric)")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='GNN Geometric Stage2 Multi-task Training (no backbone)')
    p.add_argument('--data_path',    type=str,   default='../dataset')
    p.add_argument('--gnn_hidden',   type=int,   default=256)
    p.add_argument('--gnn_layers',   type=int,   default=2)
    p.add_argument('--gnn_heads',    type=int,   default=4)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--frame_interval', type=int, default=1)
    p.add_argument('--checkpoint_dir', type=str,
                   default='./checkpoints/gnn_geometric')
    p.add_argument('--num_workers',  type=int,   default=4)
    # Multi-task / Graph Transformer options
    p.add_argument('--lambda_behavior', type=float, default=0.8,
                   help='Weight for behavior classification loss')
    p.add_argument('--lambda_group',    type=float, default=0.2,
                   help='Weight for grouping detection loss')
    p.add_argument('--group_neg_ratio', type=float, default=2.0,
                   help='Negative pair sampling ratio per scene')
    p.add_argument('--no_graph_transformer', action='store_true',
                   help='Ablation: disable Graph Transformer (use static-edge GAT)')
    p.add_argument('--no_edge_in_cls', action='store_true',
                   help='Ablation: do not append evolved edge to behavior classifier')
    p.add_argument('--flops_only',   action='store_true')
    # 模型容量
    p.add_argument('--edge_hidden_dims', nargs='+', type=int, default=[256, 128, 64],
                   help='行为分类头隐藏层维度列表，如 256 128 64')
    # K-NN 稀疏图
    p.add_argument('--graph_knn', type=int, default=0,
                   help='K-NN 稀疏图的 K 值（0=全连接）')
    # 流特征注入边初始化
    p.add_argument('--no_inject_flow_to_edges', action='store_true',
                   help='消融：禁用 10D 流特征注入边初始化（退回纯 7D 几何边）')
    # 每节点光流特征
    p.add_argument('--no_flow_node_feats', action='store_true',
                   help='消融：禁用每节点 8D 光流特征（退回 5D 静态节点，pair 10D）')
    # === 结构性改进 ===
    p.add_argument('--label_smoothing', type=float, default=0.0,
                   help='标签平滑系数（0.0=关闭，0.1=推荐）；减少少数类过拟合')
    p.add_argument('--drop_edge_rate', type=float, default=0.0,
                   help='DropEdge 丢弃率（0.0=关闭，0.3=推荐）；训练时随机丢弃非关键边')
    p.add_argument('--use_virtual_node', action='store_true',
                   help='为每帧场景图添加虚拟全局节点（聚合全场景上下文）')
    p.add_argument('--use_cross_pair_attn', action='store_true',
                   help='在行为分类前对场景内所有 pair 做 1 层自注意力')
    p.add_argument('--cross_pair_dim', type=int, default=256,
                   help='Cross-Pair Attention 的投影维度（默认 256）')
    # 可复现性与缓存
    p.add_argument('--random_seed', type=int, default=42,
                   help='全局随机种子（影响负样本采样及模型初始化）')
    p.add_argument('--cache_dir',   type=str, default='./cache/gnn_features',
                   help='预计算特征缓存目录；不同 seed 运行可复用同一缓存，省去光流计算')
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 设置全局随机种子（必须在数据集创建之前） ----
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    print(f"Random seed: {args.random_seed}")

    config = get_gnn_geometric_default(data_path=args.data_path)
    config.gnn_hidden_dim         = args.gnn_hidden
    config.gnn_num_layers         = args.gnn_layers
    config.gnn_num_heads          = args.gnn_heads
    config.batch_size             = args.batch_size
    config.epochs                 = args.epochs
    config.learning_rate          = args.lr
    config.weight_decay           = args.weight_decay
    config.frame_interval         = args.frame_interval
    config.checkpoint_dir         = args.checkpoint_dir
    config.num_workers            = args.num_workers
    config.lambda_behavior        = args.lambda_behavior
    config.lambda_group           = args.lambda_group
    config.group_neg_ratio        = args.group_neg_ratio
    config.use_graph_transformer  = not args.no_graph_transformer
    config.use_edge_in_cls        = not args.no_edge_in_cls
    config.edge_hidden_dims       = args.edge_hidden_dims
    config.graph_knn              = args.graph_knn
    config.inject_flow_to_edges   = not args.no_inject_flow_to_edges
    config.flow_node_feats        = not args.no_flow_node_feats
    config.label_smoothing        = args.label_smoothing
    config.drop_edge_rate         = args.drop_edge_rate
    config.use_virtual_node       = args.use_virtual_node
    config.use_cross_pair_attn    = args.use_cross_pair_attn
    config.cross_pair_dim         = args.cross_pair_dim
    config.random_seed            = args.random_seed
    config.cache_dir              = args.cache_dir
    # Re-derive all feature dims after flags may have changed
    config.edge_feat_dim          = 17 if config.inject_flow_to_edges else 7
    config.node_feat_dim          = 13 if config.flow_node_feats else 5
    config.pair_feat_dim          = 11 if config.flow_node_feats else 10

    print(f"\nGNN Geometric Config:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    if args.flops_only:
        estimate_flops(config, device)
        return

    # ---- Data ----
    print("\nCreating data loaders (no image loading)...")
    train_loader, val_loader, test_loader = create_gnn_geometric_data_loaders(config)

    # Auto class weights
    dist = train_loader.dataset.get_class_distribution()
    if dist.get('class_counts'):
        counts = dist['class_counts']
        total  = dist['total_pairs']
        n_cls  = len(counts)
        config.class_weights = {
            int(c): total / (n_cls * cnt) for c, cnt in counts.items()
        }
        print(f"Auto class weights: {config.class_weights}")

    estimate_flops(config, device)

    # ---- Train ----
    trainer = GNNGeometricTrainer(config, device)
    start = time.time()
    trainer.train(train_loader, val_loader, test_loader)
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")


if __name__ == '__main__':
    main()
