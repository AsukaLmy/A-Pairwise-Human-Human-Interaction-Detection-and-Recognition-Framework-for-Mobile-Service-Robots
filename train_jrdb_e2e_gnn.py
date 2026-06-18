"""
GNN 端到端 Stage 1+2 联合训练脚本

在 train_jrdb_stage2_gnn_nobackbone.py 基础上，将 group_head 升级为真正的
Stage 1（pair 检测），与 Stage 2（行为分类）通过软门控损失端到端联合训练。

核心改动（相比 train_jrdb_stage2_gnn_nobackbone.py）：
  1. 损失函数：GNNEndToEndLoss（软门控行为损失）
       梯度路径：行为分类损失 → p_stage1 = sigmoid(group_logits[:P]) → group_head 参数
  2. 新增评估指标：
       e2e_acc      — GT正对中 Stage1正确检测 且 Stage2正确分类 的比例
       stage1_recall — Stage1对GT正对的召回率
       stage1_prec   — Stage1的检测精确率
  3. 模型选择：以 e2e_acc（端到端准确率）而非 behavior_acc 为最优模型标准
  4. 命令行新增：--e2e_threshold（Stage1二值化阈值，默认0.5）

模型与数据集无需修改：
  - GNNGeometricClassifier：架构不变
  - GNNGeometricDataset：不变（提供 target_pairs + negative_pairs）
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
from models.gnn_e2e_loss import GNNEndToEndLoss, create_e2e_loss
from src.classifiers.geometric_stage2_classifier import Stage2Evaluator


CLASS_NAMES = ['Walking Together', 'Standing Together', 'Sitting Together']


# ============================================================================
# Trainer
# ============================================================================

class GNNEndToEndTrainer:
    """
    端到端 Stage 1+2 联合训练器。

    与 GNNGeometricTrainer 的唯一结构差异：
      - 使用 GNNEndToEndLoss（软门控）
      - 追踪并报告 e2e_acc / stage1_recall / stage1_prec
      - 以 e2e_acc 为最优模型选择标准
    """

    def __init__(self, config: GNNGeometricConfig, device: torch.device):
        self.config = config
        self.device = device

        print("\nCreating GNN E2E model (Stage 1 + Stage 2 joint)...")
        self.model = create_gnn_geometric_model(config).to(device)

        self.criterion = create_e2e_loss(config).to(device)

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

        # 以 MPCA 为主指标（模型保存 + early stopping），与 nobackbone 脚本保持一致
        # E2E_Acc 作为副指标单独跟踪，不参与模型选择
        self.best_val_mpca     = 0.0
        self.best_val_acc      = 0.0
        self.best_val_macro_f1 = 0.0
        self.best_e2e_acc      = 0.0   # 仅报告，不触发保存
        self.epochs_no_improve = 0

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def _prepare_batch(self, batch):
        """batch: List[Dict] from gnn_geometric_collate_fn"""
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

    def train_epoch(self, loader, epoch: int, warmup_gate: float = 1.0):
        self.model.train()
        evaluator = Stage2Evaluator(CLASS_NAMES)

        total_loss   = 0.0
        total_b_loss = 0.0
        total_g_loss = 0.0
        # 端到端计数（绝对值，避免 batch 大小不同导致的均值偏差）
        e2e_correct_sum = 0
        e2e_total_sum   = 0
        s1_recall_sum   = 0.0
        n_batches       = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train] gate={warmup_gate:.2f}")
        for batch in pbar:
            if not batch:
                continue
            scene_data, labels = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            out = self.model(scene_data)
            loss, loss_dict = self.criterion(
                out['behavior_logits'], labels,
                out['group_logits'],   out['group_labels'],
                warmup_gate=warmup_gate,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()

            total_loss   += loss_dict['total_loss']
            total_b_loss += loss_dict['behavior_loss']
            total_g_loss += loss_dict['group_loss']
            e2e_correct_sum += loss_dict['e2e_correct_count']
            e2e_total_sum   += loss_dict['e2e_total_count']
            s1_recall_sum   += loss_dict['stage1_recall']
            n_batches       += 1

            preds = out['behavior_logits'].argmax(dim=1)
            evaluator.update(preds.cpu().numpy(), labels.cpu().numpy())

            pbar.set_postfix(
                loss=f"{loss_dict['total_loss']:.4f}",
                e2e=f"{loss_dict['e2e_acc']:.3f}",
                s1r=f"{loss_dict['stage1_recall']:.3f}",
            )

        pbar.close()
        n = max(n_batches, 1)
        avg_loss     = total_loss   / n
        avg_b_loss   = total_b_loss / n
        avg_g_loss   = total_g_loss / n
        e2e_acc      = e2e_correct_sum / max(e2e_total_sum, 1)
        avg_s1_recall = s1_recall_sum  / n

        metrics  = evaluator.compute_metrics()
        mpca     = metrics.get('mpca', 0.0)
        acc      = metrics.get('overall_accuracy', 0.0)
        macro_f1 = metrics.get('macro_f1', 0.0)

        print(f"Train Epoch {epoch}: Loss={avg_loss:.4f} "
              f"(beh={avg_b_loss:.4f}, grp={avg_g_loss:.4f}) | "
              f"BehAcc={acc:.4f} MPCA={mpca:.4f} MacroF1={macro_f1:.4f} | "
              f"E2E_Acc={e2e_acc:.4f} S1_Recall={avg_s1_recall:.4f}")
        return avg_loss, acc, mpca, e2e_acc, macro_f1

    # ------------------------------------------------------------------

    def validate_epoch(self, loader, epoch: int):
        self.model.eval()
        evaluator = Stage2Evaluator(CLASS_NAMES)

        total_loss   = 0.0
        total_b_loss = 0.0
        total_g_loss = 0.0
        e2e_correct_sum  = 0
        e2e_total_sum    = 0
        s1_recall_sum    = 0.0
        s1_prec_sum      = 0.0
        s1_prec_n        = 0
        group_acc_sum    = 0.0
        n_batches        = 0

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
                e2e_correct_sum  += loss_dict['e2e_correct_count']
                e2e_total_sum    += loss_dict['e2e_total_count']
                s1_recall_sum    += loss_dict['stage1_recall']
                group_acc_sum    += loss_dict['group_acc']

                prec = loss_dict['stage1_precision']
                if not (isinstance(prec, float) and prec != prec):  # skip NaN
                    s1_prec_sum += prec
                    s1_prec_n   += 1

                n_batches += 1
                preds = out['behavior_logits'].argmax(dim=1)
                evaluator.update(preds.cpu().numpy(), labels.cpu().numpy())
            pbar.close()

        n = max(n_batches, 1)
        avg_loss      = total_loss   / n
        avg_b_loss    = total_b_loss / n
        avg_g_loss    = total_g_loss / n
        e2e_acc       = e2e_correct_sum / max(e2e_total_sum, 1)
        avg_s1_recall = s1_recall_sum / n
        avg_s1_prec   = s1_prec_sum / max(s1_prec_n, 1)
        avg_g_acc     = group_acc_sum / n

        metrics  = evaluator.compute_metrics()
        acc      = metrics.get('overall_accuracy', 0.0)
        mpca     = metrics.get('mpca', 0.0)
        macro_f1 = metrics.get('macro_f1', 0.0)

        print(f"Val   Epoch {epoch}: Loss={avg_loss:.4f} "
              f"(beh={avg_b_loss:.4f}, grp={avg_g_loss:.4f}) | "
              f"BehAcc={acc:.4f} MPCA={mpca:.4f} MacroF1={macro_f1:.4f} | "
              f"E2E_Acc={e2e_acc:.4f} S1_Recall={avg_s1_recall:.4f} "
              f"S1_Prec={avg_s1_prec:.4f} GrpAcc={avg_g_acc:.4f}")
        return avg_loss, acc, mpca, e2e_acc, macro_f1

    # ------------------------------------------------------------------

    def test(self, loader):
        self.model.eval()
        evaluator = Stage2Evaluator(CLASS_NAMES)

        e2e_correct_sum = 0
        e2e_total_sum   = 0
        s1_recall_list  = []
        s1_prec_list    = []

        print(f"\n{'='*80}\nFINAL TEST EVALUATION (End-to-End Stage 1+2)\n{'='*80}")

        with torch.no_grad():
            for batch in tqdm(loader, desc="Testing"):
                if not batch:
                    continue
                scene_data, labels = self._prepare_batch(batch)
                out = self.model(scene_data)

                _, loss_dict = self.criterion(
                    out['behavior_logits'], labels,
                    out['group_logits'],   out['group_labels'],
                )

                preds = out['behavior_logits'].argmax(dim=1)
                evaluator.update(preds.cpu().numpy(), labels.cpu().numpy())

                e2e_correct_sum += loss_dict['e2e_correct_count']
                e2e_total_sum   += loss_dict['e2e_total_count']
                s1_recall_list.append(loss_dict['stage1_recall'])
                prec = loss_dict['stage1_precision']
                if not (isinstance(prec, float) and prec != prec):
                    s1_prec_list.append(prec)

        evaluator.print_evaluation_report()

        metrics   = evaluator.compute_metrics()
        beh_acc   = metrics.get('overall_accuracy', 0.0)
        mpca      = metrics.get('mpca', 0.0)
        macro_f1  = metrics.get('macro_f1', 0.0)
        e2e_acc   = e2e_correct_sum / max(e2e_total_sum, 1)
        s1_recall = float(np.mean(s1_recall_list)) if s1_recall_list else 0.0
        s1_prec   = float(np.mean(s1_prec_list))   if s1_prec_list   else float('nan')

        print(f"\n{'-'*60}")
        print(f"[传统指标]  BehaviorAcc={beh_acc:.4f}  MPCA={mpca:.4f}  MacroF1={macro_f1:.4f}")
        print(f"            （假设 Stage1 分组完全正确）")
        print(f"\n[端到端指标] E2E_Acc={e2e_acc:.4f}")
        print(f"            （Stage1 漏检的 pair 计入错误；{e2e_correct_sum}/{e2e_total_sum} GT对正确）")
        print(f"            Stage1_Recall={s1_recall:.4f}  Stage1_Precision={s1_prec:.4f}")
        print(f"{'-'*60}")
        return metrics, e2e_acc

    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader, test_loader=None):
        print(f"\n{'='*80}")
        print(f"STARTING GNN END-TO-END TRAINING (Stage 1 + Stage 2 Joint)")
        print(f"  软门控行为损失：梯度从行为分类回流到 group_head（Stage 1）")
        print(f"  模型选择标准：E2E_Acc（端到端行为准确率）")
        print(f"{'='*80}\n")

        warmup = self.config.e2e_warmup_epochs
        for epoch in range(1, self.config.epochs + 1):
            # 线性 warmup：前 warmup 个 epoch gate=0（无软门控），之后线性升到 1.0
            gate = max(0.0, min(1.0, (epoch - warmup) / max(warmup, 1)))
            self.train_epoch(train_loader, epoch, warmup_gate=gate)
            _, val_acc, val_mpca, val_e2e, val_macro_f1 = self.validate_epoch(val_loader, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            # 始终更新 best_e2e_acc（副指标，仅报告）
            if val_e2e > self.best_e2e_acc:
                self.best_e2e_acc = val_e2e

            # 以 MPCA 为主指标决定是否保存模型（与 nobackbone 脚本一致）
            if val_mpca > self.best_val_mpca:
                self.best_val_mpca     = val_mpca
                self.best_val_acc      = val_acc
                self.best_val_macro_f1 = val_macro_f1
                self.epochs_no_improve = 0

                ckpt = os.path.join(self.config.checkpoint_dir,
                                    'best_model_gnn_e2e.pth')
                torch.save({
                    'epoch':              epoch,
                    'model_state_dict':   self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_e2e_acc':       self.best_e2e_acc,
                    'best_val_mpca':      self.best_val_mpca,
                    'config':             self.config.__dict__,
                }, ckpt)
                print(f"  [Saved] MPCA={self.best_val_mpca:.4f} "
                      f"E2E_Acc={self.best_e2e_acc:.4f} → {ckpt}")
            else:
                self.epochs_no_improve += 1

            print(f"  Best MPCA={self.best_val_mpca:.4f} MacroF1={self.best_val_macro_f1:.4f} | "
                  f"Best E2E_Acc={self.best_e2e_acc:.4f} | "
                  f"No-improve={self.epochs_no_improve}")

            if self.epochs_no_improve >= self.config.early_stopping_patience:
                print("\nEarly stopping triggered.")
                break

        if test_loader is not None:
            ckpt = os.path.join(self.config.checkpoint_dir, 'best_model_gnn_e2e.pth')
            if os.path.exists(ckpt):
                data = torch.load(ckpt, map_location=self.device)
                self.model.load_state_dict(data['model_state_dict'])
                print(f"Loaded best model from epoch {data['epoch']}")
            self.test(test_loader)

        print(f"\n{'='*80}\nTRAINING DONE  "
              f"Best MPCA={self.best_val_mpca:.4f}  "
              f"MacroF1={self.best_val_macro_f1:.4f}  "
              f"BehAcc={self.best_val_acc:.4f}  "
              f"Best E2E_Acc={self.best_e2e_acc:.4f}\n{'='*80}")


# ============================================================================
# FLOPs
# ============================================================================

def estimate_flops(config, device):
    if not THOP_AVAILABLE:
        print("thop not available – skip FLOPs")
        return
    model = create_gnn_geometric_model(config).to(device).eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"\nGNN E2E model: {total:,} parameters (same as nobackbone)")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='GNN 端到端 Stage1+2 联合训练（软门控行为损失）')
    p.add_argument('--data_path',      type=str,   default='../dataset')
    p.add_argument('--gnn_hidden',     type=int,   default=256)
    p.add_argument('--gnn_layers',     type=int,   default=2)
    p.add_argument('--gnn_heads',      type=int,   default=4)
    p.add_argument('--batch_size',     type=int,   default=8)
    p.add_argument('--epochs',         type=int,   default=50)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--weight_decay',   type=float, default=1e-4)
    p.add_argument('--frame_interval', type=int,   default=1)
    p.add_argument('--checkpoint_dir', type=str,
                   default='./checkpoints/gnn_e2e')
    p.add_argument('--num_workers',    type=int,   default=4)
    # 损失权重
    p.add_argument('--lambda_behavior', type=float, default=0.8,
                   help='行为分类损失权重')
    p.add_argument('--lambda_group',    type=float, default=0.2,
                   help='分组检测损失权重')
    p.add_argument('--group_neg_ratio', type=float, default=2.0,
                   help='每个场景的负对采样倍率')
    # Stage 1 评估阈值
    p.add_argument('--e2e_threshold',   type=float, default=0.5,
                   help='Stage1 pair 检测二值化阈值（仅影响评估指标，不影响梯度）')
    # 消融选项
    p.add_argument('--nobackbone', action='store_true',
                   help='Pure bbox-geometry mode: disable all optical flow features '
                        '(equivalent to --no_flow_node_feats + --no_inject_flow_to_edges). '
                        'No image files required. 5D nodes / 7D edges / 10D pairs.')
    p.add_argument('--no_graph_transformer', action='store_true',
                   help='消融：禁用 Graph Transformer（使用静态边 GAT）')
    p.add_argument('--no_edge_in_cls', action='store_true',
                   help='消融：行为分类器不使用演化边特征')
    p.add_argument('--flops_only',     action='store_true')
    # 模型容量
    p.add_argument('--edge_hidden_dims', nargs='+', type=int, default=[256, 128, 64],
                   help='行为分类头隐藏层维度列表，如 256 128 64')
    # E2E warmup
    p.add_argument('--e2e_warmup_epochs', type=int, default=10,
                   help='软门控预热 epoch 数：前 N epoch 不启用软门控，之后线性升至完全启用')
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

    # --nobackbone shorthand: disable all optical-flow features (no images needed)
    if args.nobackbone:
        args.no_flow_node_feats      = True
        args.no_inject_flow_to_edges = True
        if args.checkpoint_dir == './checkpoints/gnn_e2e':
            args.checkpoint_dir = './checkpoints/gnn_e2e_nobackbone'
        print("Nobackbone mode: pure bbox geometry (5D nodes, 7D edges, 10D pairs) — no images needed")

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
    config.gnn_hidden_dim        = args.gnn_hidden
    config.gnn_num_layers        = args.gnn_layers
    config.gnn_num_heads         = args.gnn_heads
    config.batch_size            = args.batch_size
    config.epochs                = args.epochs
    config.learning_rate         = args.lr
    config.weight_decay          = args.weight_decay
    config.frame_interval        = args.frame_interval
    config.checkpoint_dir        = args.checkpoint_dir
    config.num_workers           = args.num_workers
    config.lambda_behavior       = args.lambda_behavior
    config.lambda_group          = args.lambda_group
    config.group_neg_ratio       = args.group_neg_ratio
    config.e2e_threshold         = args.e2e_threshold
    config.use_graph_transformer = not args.no_graph_transformer
    config.use_edge_in_cls       = not args.no_edge_in_cls
    config.edge_hidden_dims      = args.edge_hidden_dims
    config.e2e_warmup_epochs     = args.e2e_warmup_epochs
    config.graph_knn             = args.graph_knn
    config.inject_flow_to_edges  = not args.no_inject_flow_to_edges
    config.flow_node_feats       = not args.no_flow_node_feats
    config.label_smoothing       = args.label_smoothing
    config.drop_edge_rate        = args.drop_edge_rate
    config.use_virtual_node      = args.use_virtual_node
    config.use_cross_pair_attn   = args.use_cross_pair_attn
    config.cross_pair_dim        = args.cross_pair_dim
    config.random_seed           = args.random_seed
    config.cache_dir             = args.cache_dir
    # Re-derive feature dims after flags may have changed
    config.edge_feat_dim         = 17 if config.inject_flow_to_edges else 7
    config.node_feat_dim         = 13 if config.flow_node_feats else 5
    config.pair_feat_dim         = 11 if config.flow_node_feats else 10

    print(f"\nGNN E2E Config:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    if args.flops_only:
        estimate_flops(config, device)
        return

    # ---- Data ----
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_gnn_geometric_data_loaders(config)

    # 自动类权重
    dist = train_loader.dataset.get_class_distribution()
    if dist.get('class_counts'):
        counts = dist['class_counts']
        total  = dist['total_pairs']
        n_cls  = len(counts)
        config.class_weights = {
            int(c): total / (n_cls * cnt) for c, cnt in counts.items()
        }
        print(f"Auto class weights: {config.class_weights}")
        # 更新 criterion 的 alpha_weights
    estimate_flops(config, device)

    # ---- Train ----
    trainer = GNNEndToEndTrainer(config, device)
    # 若已有自动类权重，重新创建 criterion 以应用
    if hasattr(config, 'class_weights') and config.class_weights:
        trainer.criterion = create_e2e_loss(config).to(device)

    start = time.time()
    trainer.train(train_loader, val_loader, test_loader)
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")


if __name__ == '__main__':
    main()
