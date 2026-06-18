#!/usr/bin/env python3
"""
GNN-based Stage2 Behavior Classification Model
Graph Attention Network (GAT) for scene-level pedestrian interaction classification.

Architecture overview:
  1. ResNet backbone  → visual features for all N persons in a scene
  2. PersonGraphBuilder → build fully-connected scene graph with geometric edge features
  3. Multi-layer GATLayer → propagate context across all nodes (persons)
  4. Edge classifier  → classify labeled target pairs into 3 interaction classes

Pure PyTorch implementation – no torch_geometric dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import sys
import os

# Allow running this file directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from models.resnet_feature_extractors import ResNetBackbone
    from models.resnet_stage2_classifier import ResNetStage2Loss
    from src.features.geometric_features import extract_geometric_features, extract_geometric_features_batch
except ImportError:
    from resnet_feature_extractors import ResNetBackbone
    from resnet_stage2_classifier import ResNetStage2Loss
    sys.path.insert(0, os.path.join(project_root, 'src', 'features'))
    from geometric_features import extract_geometric_features, extract_geometric_features_batch


# ============================================================================
# Graph Builder
# ============================================================================

class PersonGraphBuilder:
    """
    Builds a fully-connected directed scene graph from N detected persons.

    Nodes: each person, with features = visual_feat cat position_encoding
    Edges: all ordered pairs (i, j) with i != j  →  N*(N-1) directed edges
    Edge features: 7D geometric features from extract_geometric_features()
    """

    def __init__(self, image_width: int = 3760, image_height: int = 480):
        self.image_width = image_width
        self.image_height = image_height

    def build_graph(
        self,
        visual_feats: torch.Tensor,             # [N, visual_dim]
        person_boxes: torch.Tensor,             # [N, 4] – [x, y, w, h]
        precomputed_edges: Optional[Dict] = None,  # {'edge_index': [2,E], 'edge_feats': [E,7]}
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_feats:       [N, visual_dim] – already extracted by backbone
            person_boxes:       [N, 4]
            precomputed_edges:  optional dict with pre_edge_index [2,E] and
                                pre_edge_feats [E,7] from the dataset.
                                When provided, skips the slow Python edge loop.

        Returns dict with:
            node_feats:  [N, visual_dim + 4]   (visual + normalised position)
            edge_index:  [2, E]               E = N*(N-1)
            edge_feats:  [E, 7]
        """
        N = visual_feats.size(0)
        device = visual_feats.device

        # ---- Position encoding [cx_norm, cy_norm, w_norm, h_norm] ----
        x = person_boxes[:, 0]
        y = person_boxes[:, 1]
        w = person_boxes[:, 2]
        h = person_boxes[:, 3]
        cx_norm = (x + w / 2) / self.image_width
        cy_norm = (y + h / 2) / self.image_height
        w_norm  = w / self.image_width
        h_norm  = h / self.image_height
        pos_enc = torch.stack([cx_norm, cy_norm, w_norm, h_norm], dim=1)  # [N, 4]

        node_feats = torch.cat([visual_feats, pos_enc], dim=1)  # [N, visual_dim+4]

        # ---- Degenerate case ----
        if N <= 1:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_feats = torch.zeros(0, 7, dtype=torch.float32, device=device)
            return {'node_feats': node_feats, 'edge_index': edge_index,
                    'edge_feats': edge_feats}

        # ---- Use precomputed edges when available (fast path) ----
        if precomputed_edges is not None:
            edge_index = precomputed_edges['edge_index'].to(device)
            edge_feats = precomputed_edges['edge_feats'].to(device)
            return {
                'node_feats': node_feats,
                'edge_index': edge_index,
                'edge_feats': edge_feats,
            }

        # ---- Fallback: compute edges on-the-fly (slow path, kept for compatibility) ----
        src_list = [i for i in range(N) for j in range(N) if i != j]
        dst_list = [j for i in range(N) for j in range(N) if i != j]

        src_t = torch.tensor(src_list, dtype=torch.long)
        dst_t = torch.tensor(dst_list, dtype=torch.long)
        edge_index = torch.stack([src_t, dst_t], dim=0).to(device)  # [2, E]

        boxes_cpu = person_boxes.cpu()
        edge_feats = extract_geometric_features_batch(
            boxes_cpu[src_t], boxes_cpu[dst_t],
            self.image_width, self.image_height,
        ).to(device)  # [E, 7]

        return {
            'node_feats': node_feats,   # [N, visual_dim+4]
            'edge_index': edge_index,   # [2, E]
            'edge_feats': edge_feats,   # [E, 7]
        }


# ============================================================================
# Graph Attention Layer (hand-implemented, no torch_geometric)
# ============================================================================

class GATLayer(nn.Module):
    """
    Single Graph Attention Network layer.

    For each directed edge (src → dst):
      e_ij = LeakyReLU( a^T [W·h_src || W·h_dst || W_e·edge_feat] )
      α_ij = softmax over all in-edges of dst
      h_dst_new = ELU( Σ_j α_ij · W·h_src_j )

    Multi-head: concat (non-final layers) or mean (final layer).
    Residual connection + LayerNorm applied when dimensions match.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,          # per-head output dim
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat_heads: bool = True,   # True → output [N, H*D]; False → [N, D]
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        H, D = num_heads, out_dim

        # Node projection: in_dim → H*D
        self.W_node = nn.Linear(in_dim, H * D, bias=False)
        # Edge projection: edge_dim → H*D
        self.W_edge = nn.Linear(edge_dim, H * D, bias=False)
        # Attention vector: for each head, 3*D → scalar
        self.attn_vec = nn.Parameter(torch.empty(H, 3 * D))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)

        out_total = H * D if concat_heads else D
        self.norm = nn.LayerNorm(out_total)

        # Residual projection when dims mismatch
        self.res_proj = (
            nn.Linear(in_dim, out_total, bias=False)
            if in_dim != out_total else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.W_node.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.W_edge.weight, mode='fan_out')

    def forward(
        self,
        node_feats: torch.Tensor,    # [N, in_dim]
        edge_index: torch.Tensor,    # [2, E]
        edge_feats: torch.Tensor,    # [E, edge_dim]
    ) -> torch.Tensor:
        N = node_feats.size(0)
        E = edge_index.size(1)
        H, D = self.num_heads, self.out_dim
        device = node_feats.device

        # Handle degenerate case (single node or no edges)
        if E == 0 or N <= 1:
            out_total = H * D if self.concat_heads else D
            out = torch.zeros(N, out_total, device=device)
            return F.elu(self.norm(out + self.res_proj(node_feats)))

        # 1. Project all node features: [N, H*D] → [N, H, D]
        h = self.W_node(node_feats).view(N, H, D)

        # 2. Project all edge features: [E, H*D] → [E, H, D]
        e_proj = self.W_edge(edge_feats).view(E, H, D)

        src = edge_index[0]   # [E]
        dst = edge_index[1]   # [E]

        # 3. Attention logits: [E, H, 3D] → [E, H]
        h_src = h[src]   # [E, H, D]
        h_dst = h[dst]   # [E, H, D]
        cat_feat = torch.cat([h_src, h_dst, e_proj], dim=-1)   # [E, H, 3D]
        # attn_vec [H, 3D] broadcast to [E, H, 3D]
        e_scores = (cat_feat * self.attn_vec.unsqueeze(0)).sum(dim=-1)  # [E, H]
        e_scores = self.leaky_relu(e_scores)

        # 4. Sparse softmax (per destination node)
        alpha = self._sparse_softmax(e_scores, dst, N)   # [E, H]
        alpha = self.dropout(alpha)

        # 5. Weighted aggregation: messages flow src → dst
        #    weighted_msg[e] = α[e] * h[src[e]]
        weighted_msg = alpha.unsqueeze(-1) * h[src]   # [E, H, D]
        agg = torch.zeros(N, H, D, device=device)
        agg.index_add_(0, dst, weighted_msg)           # [N, H, D]

        # 6. Concat / mean heads + residual + norm
        if self.concat_heads:
            out = agg.view(N, H * D)
        else:
            out = agg.mean(dim=1)   # [N, D]

        residual = self.res_proj(node_feats)
        out = self.norm(out + residual)
        return F.elu(out)

    @staticmethod
    def _sparse_softmax(
        scores: torch.Tensor,     # [E, H]
        dst_idx: torch.Tensor,    # [E]
        N: int,
    ) -> torch.Tensor:
        """
        Softmax grouped by destination node.
        Avoids torch_scatter by using index_reduce / manual scatter.
        """
        H = scores.size(1)
        device = scores.device

        # Numerical stability: subtract per-node max
        node_max = torch.full((N, H), float('-inf'), device=device)
        node_max.scatter_reduce_(
            0,
            dst_idx.unsqueeze(1).expand(-1, H),
            scores,
            reduce='amax',
            include_self=True,
        )
        scores_stable = scores - node_max[dst_idx]   # [E, H]

        exp_scores = torch.exp(scores_stable)

        # Sum exp per node
        exp_sum = torch.zeros(N, H, device=device)
        exp_sum.index_add_(0, dst_idx, exp_scores)

        alpha = exp_scores / (exp_sum[dst_idx] + 1e-8)
        return alpha


# ============================================================================
# Graph Transformer Layer (edge update + node update)
# ============================================================================

class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer that updates BOTH edge features and node features.

    Step 1 – Edge Update (before node aggregation):
        e'_ij = LayerNorm( e_ij + SiLU(W_es(h_i) + W_ed(h_j) + W_ee(e_ij)) )

    Step 2 – Node Update (same multi-head attention as GATLayer, using e'_ij):
        α_ij  = softmax_j( a^T [W·h_i || W·h_j || W_edge·e'_ij] )
        h'_i  = ELU( LayerNorm( Σ_j α_ij · W·h_j  +  residual(h_i) ) )

    Returns (node_feats_updated, edge_feats_updated) – both dimensions preserved.

    Assumptions:
        • node_dim is consistent across all layers (= gnn_hidden_dim)
        • edge_dim is consistent across all layers (= edge_hidden_dim);
          caller is responsible for projecting raw 7D features → edge_hidden_dim
          before the first layer.
    """

    def __init__(
        self,
        node_dim: int,           # node feature dimension (consistent across layers)
        edge_dim: int,           # edge feature dimension (= edge_hidden_dim)
        num_heads: int = 4,
        dropout: float = 0.1,
        concat_heads: bool = True,
    ):
        super().__init__()
        self.node_dim   = node_dim
        self.edge_dim   = edge_dim
        self.num_heads  = num_heads
        self.concat_heads = concat_heads

        # ---- Step 1: Edge update projections ----
        self.W_e_src  = nn.Linear(node_dim, edge_dim, bias=False)
        self.W_e_dst  = nn.Linear(node_dim, edge_dim, bias=False)
        self.W_e_self = nn.Linear(edge_dim, edge_dim, bias=False)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.edge_act  = nn.SiLU()

        # ---- Step 2: Node update (multi-head attention, same pattern as GATLayer) ----
        per_head = node_dim // num_heads
        assert node_dim % num_heads == 0, \
            f"node_dim {node_dim} must be divisible by num_heads {num_heads}"
        self.per_head = per_head
        H, D = num_heads, per_head

        self.W_node  = nn.Linear(node_dim, H * D, bias=False)
        self.W_edge  = nn.Linear(edge_dim,  H * D, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(H, 3 * D))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout    = nn.Dropout(dropout)

        out_total = H * D if concat_heads else D
        self.node_norm = nn.LayerNorm(out_total)
        self.res_proj  = (
            nn.Linear(node_dim, out_total, bias=False)
            if node_dim != out_total else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_e_src, self.W_e_dst, self.W_e_self,
                  self.W_node, self.W_edge]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(
        self,
        node_feats: torch.Tensor,   # [N, node_dim]
        edge_index: torch.Tensor,   # [2, E]
        edge_feats: torch.Tensor,   # [E, edge_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_feats_new: [N, node_dim]   (out_total = node_dim since concat H*(D=node/H))
            edge_feats_new: [E, edge_dim]
        """
        N = node_feats.size(0)
        E = edge_index.size(1)
        H, D = self.num_heads, self.per_head
        device = node_feats.device

        # ---- Degenerate case ----
        if E == 0 or N <= 1:
            out_total = H * D if self.concat_heads else D
            h_out = F.elu(self.node_norm(
                torch.zeros(N, out_total, device=device) + self.res_proj(node_feats)))
            return h_out, edge_feats

        src = edge_index[0]
        dst = edge_index[1]

        # ============================================================
        # Step 1 – Edge Update
        # ============================================================
        e_src  = self.W_e_src(node_feats[src])    # [E, edge_dim]
        e_dst  = self.W_e_dst(node_feats[dst])    # [E, edge_dim]
        e_self = self.W_e_self(edge_feats)        # [E, edge_dim]
        e_new  = self.edge_norm(
            edge_feats + self.edge_act(e_src + e_dst + e_self)
        )                                          # [E, edge_dim]  residual

        # ============================================================
        # Step 2 – Node Update (multi-head attention using e_new)
        # ============================================================
        h = self.W_node(node_feats).view(N, H, D)     # [N, H, D]
        e_proj = self.W_edge(e_new).view(E, H, D)     # [E, H, D]

        h_src = h[src]   # [E, H, D]
        h_dst = h[dst]   # [E, H, D]
        cat_feat = torch.cat([h_src, h_dst, e_proj], dim=-1)   # [E, H, 3D]
        e_scores = (cat_feat * self.attn_vec.unsqueeze(0)).sum(dim=-1)  # [E, H]
        e_scores = self.leaky_relu(e_scores)

        alpha = GATLayer._sparse_softmax(e_scores, dst, N)   # [E, H]
        alpha = self.dropout(alpha)

        weighted_msg = alpha.unsqueeze(-1) * h[src]    # [E, H, D]
        agg = torch.zeros(N, H, D, device=device)
        agg.index_add_(0, dst, weighted_msg)            # [N, H, D]

        if self.concat_heads:
            out = agg.view(N, H * D)
        else:
            out = agg.mean(dim=1)

        h_new = F.elu(self.node_norm(out + self.res_proj(node_feats)))

        return h_new, e_new


# ============================================================================
# Full GNN Classifier
# ============================================================================

class GNNStage2Classifier(nn.Module):
    """
    Full scene-level GNN classifier for Stage2 behavior classification.

    Forward input:  List[Dict] – one dict per frame/scene
    Forward output: Dict with 'logits' [P_total, 3]

    Batch strategy: super-graph concatenation
      All N_i nodes from all scenes in the batch are concatenated into
      one large graph. Edges are offset so cross-scene edges don't exist.
      The GAT operates on this single large graph efficiently.
    """

    def __init__(
        self,
        backbone_name: str = 'resnet18',
        visual_feature_dim: int = 128,
        gnn_hidden_dim: int = 256,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
        edge_feat_dim: int = 7,
        edge_hidden_dims: Optional[List[int]] = None,
        num_classes: int = 3,
        gnn_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        crop_size: int = 112,
        image_width: int = 3760,
        image_height: int = 480,
    ):
        super().__init__()

        if edge_hidden_dims is None:
            edge_hidden_dims = [512, 256, 128]

        self.visual_feature_dim = visual_feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.freeze_backbone = freeze_backbone

        # ---- ResNet Backbone (reuse existing class) ----
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            feature_dim=visual_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            input_size=crop_size,
        )

        # ---- Graph Builder ----
        self.graph_builder = PersonGraphBuilder(
            image_width=image_width,
            image_height=image_height,
        )

        # ---- Multi-layer GAT ----
        node_dim = visual_feature_dim + 4   # visual + position encoding
        self.gat_layers = nn.ModuleList()
        in_dim = node_dim
        for layer_idx in range(gnn_num_layers):
            is_last = (layer_idx == gnn_num_layers - 1)
            # All layers except the last concat heads; last layer uses mean
            concat = not is_last
            per_head_dim = gnn_hidden_dim // gnn_num_heads
            self.gat_layers.append(GATLayer(
                in_dim=in_dim,
                out_dim=per_head_dim,
                edge_dim=edge_feat_dim,
                num_heads=gnn_num_heads,
                dropout=gnn_dropout,
                concat_heads=concat,
            ))
            in_dim = gnn_hidden_dim if concat else per_head_dim

        # Final node embedding dimension
        node_emb_dim = in_dim

        # ---- Edge Classification Head ----
        # Input: [emb_A, emb_B, |A-B|, A*B]  →  4 * node_emb_dim
        edge_cls_input = 4 * node_emb_dim
        layers = []
        prev_dim = edge_cls_input
        for h_dim in edge_hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(classifier_dropout),
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.edge_classifier = nn.Sequential(*layers)

        self._init_weights()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GNNStage2Classifier created:")
        print(f"  Backbone: {backbone_name}, visual_dim={visual_feature_dim}, frozen={freeze_backbone}")
        print(f"  Node dim: {node_dim} → GAT hidden {gnn_hidden_dim} ({gnn_num_layers} layers, {gnn_num_heads} heads)")
        print(f"  Edge classifier input: {edge_cls_input}D → {edge_hidden_dims} → {num_classes}")
        print(f"  Total params: {total:,}  |  Trainable: {trainable:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, scene_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            scene_data: list of dicts (one per frame/scene), each containing:
                'person_crops':  Tensor [N_i, 3, H, W]
                'person_boxes':  Tensor [N_i, 4]
                'target_pairs':  Tensor [P_i, 2]  local indices

        Returns:
            {
                'logits':         [P_total, num_classes]
                'pair_scene_idx': [P_total]  which scene each pair belongs to
            }
        """
        device = next(self.parameters()).device

        # ----------------------------------------------------------------
        # Step 1: Batch all person crops → single backbone forward pass
        # ----------------------------------------------------------------
        all_crops = []
        scene_N = []   # number of persons per scene

        for scene in scene_data:
            crops = scene['person_crops'].to(device)   # [N_i, 3, H, W]
            all_crops.append(crops)
            scene_N.append(crops.size(0))

        all_crops_cat = torch.cat(all_crops, dim=0)   # [sum(N_i), 3, H, W]

        ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with ctx:
            all_visual = self.backbone(all_crops_cat)  # [sum(N_i), visual_dim]

        # ----------------------------------------------------------------
        # Step 2: Build per-scene graphs; assemble super-graph
        # ----------------------------------------------------------------
        all_node_feats = []
        all_edge_indices = []
        all_edge_feats = []
        target_pairs_global = []   # pairs with global node indices
        pair_scene_idx = []

        node_offset = 0
        feat_offset = 0

        for scene_idx, scene in enumerate(scene_data):
            N_i = scene_N[scene_idx]
            vis_i = all_visual[feat_offset: feat_offset + N_i]   # [N_i, visual_dim]
            feat_offset += N_i

            boxes_i = scene['person_boxes'].to(device)   # [N_i, 4]

            # Use precomputed edges if available (fast path)
            precomp = None
            if 'pre_edge_index' in scene and scene['pre_edge_index'].size(1) > 0:
                precomp = {
                    'edge_index': scene['pre_edge_index'],
                    'edge_feats': scene['pre_edge_feats'],
                }
            graph_i = self.graph_builder.build_graph(vis_i, boxes_i, precomp)

            all_node_feats.append(graph_i['node_feats'])
            all_edge_indices.append(graph_i['edge_index'] + node_offset)
            all_edge_feats.append(graph_i['edge_feats'])

            pairs_i = scene['target_pairs'].to(device)   # [P_i, 2]
            target_pairs_global.append(pairs_i + node_offset)
            pair_scene_idx.extend([scene_idx] * pairs_i.size(0))

            node_offset += N_i

        # ----------------------------------------------------------------
        # Step 3: Super-graph GAT propagation
        # ----------------------------------------------------------------
        all_node_feats_cat = torch.cat(all_node_feats, dim=0)        # [total_N, node_dim]
        all_edge_idx_cat   = torch.cat(all_edge_indices, dim=1)      # [2, total_E]
        all_edge_feats_cat = torch.cat(all_edge_feats, dim=0)        # [total_E, 7]

        h = all_node_feats_cat
        for gat_layer in self.gat_layers:
            h = gat_layer(h, all_edge_idx_cat, all_edge_feats_cat)   # [total_N, hidden]

        # ----------------------------------------------------------------
        # Step 4: Edge classification for target pairs
        # ----------------------------------------------------------------
        all_pairs = torch.cat(target_pairs_global, dim=0)   # [P_total, 2]

        emb_A = h[all_pairs[:, 0]]   # [P_total, node_emb_dim]
        emb_B = h[all_pairs[:, 1]]   # [P_total, node_emb_dim]

        # Relation fusion identical to Relation Network baseline
        abs_diff  = torch.abs(emb_A - emb_B)
        elem_prod = emb_A * emb_B
        edge_repr = torch.cat([emb_A, emb_B, abs_diff, elem_prod], dim=1)  # [P_total, 4*H]

        logits = self.edge_classifier(edge_repr)   # [P_total, num_classes]

        pair_scene_tensor = torch.tensor(pair_scene_idx, dtype=torch.long, device=device)

        return {
            'logits': logits,
            'pair_scene_idx': pair_scene_tensor,
        }

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'model_type': 'GNNStage2Classifier',
            'total_params': total,
            'trainable_params': trainable,
            'gnn_layers': len(self.gat_layers),
            'gnn_hidden_dim': self.gnn_hidden_dim,
        }


# ============================================================================
# Factory
# ============================================================================

def create_gnn_stage2_model(config) -> GNNStage2Classifier:
    """
    Instantiate GNNStage2Classifier from a GNNStage2Config.
    """
    return GNNStage2Classifier(
        backbone_name=config.backbone_name,
        visual_feature_dim=config.visual_feature_dim,
        gnn_hidden_dim=config.gnn_hidden_dim,
        gnn_num_layers=config.gnn_num_layers,
        gnn_num_heads=config.gnn_num_heads,
        edge_feat_dim=config.edge_feat_dim,
        edge_hidden_dims=list(config.edge_hidden_dims),
        num_classes=3,
        gnn_dropout=config.gnn_dropout,
        classifier_dropout=config.classifier_dropout,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        crop_size=config.crop_size,
        image_width=config.image_width,
        image_height=config.image_height,
    )


# ============================================================================
# Cross-Pair Attention
# ============================================================================

class CrossPairAttention(nn.Module):
    """
    1-layer self-attention over all target-pair embeddings within a scene.

    Each pair's classification representation is updated with context from
    all other pairs in the same scene.  This captures intra-scene correlations
    (e.g. if one pair is Sitting Together, others in the same room likely are too).

    Applied AFTER GNN propagation but BEFORE the behavior MLP classifier.
    The module uses a low-dimensional projection to keep attention cheap:
        cls_input (D_in) → proj_in (256) → self-attn → proj_out (D_in) [residual]

    When only 1 pair exists in a scene the module is a no-op.
    """

    def __init__(self, d_in: int, d_attn: int = 256, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert d_attn % num_heads == 0, "d_attn must be divisible by num_heads"
        self.proj_in  = nn.Linear(d_in, d_attn)
        self.attn     = nn.MultiheadAttention(d_attn, num_heads,
                                               dropout=dropout, batch_first=True)
        self.proj_out = nn.Linear(d_attn, d_in)
        self.norm     = nn.LayerNorm(d_in)
        self.ff       = nn.Sequential(
            nn.Linear(d_in, d_in * 2), nn.SiLU(), nn.Linear(d_in * 2, d_in))
        self.norm2    = nn.LayerNorm(d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [P, D_in] – all pair embeddings in the scene
        Returns:
            [P, D_in] – context-enriched embeddings (residual)
        """
        if x.size(0) <= 1:
            return x                            # single pair: skip (no-op)
        q   = self.proj_in(x).unsqueeze(0)      # [1, P, d_attn]
        out, _ = self.attn(q, q, q)             # [1, P, d_attn]
        x = self.norm(x + self.proj_out(out.squeeze(0)))   # [P, D_in]
        x = self.norm2(x + self.ff(x))
        return x


# ============================================================================
# Sanity test
# ============================================================================

if __name__ == '__main__':
    print("Testing GNNStage2Classifier with fake data...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = GNNStage2Classifier(
        backbone_name='resnet18',
        visual_feature_dim=128,
        gnn_hidden_dim=256,
        gnn_num_layers=2,
        gnn_num_heads=4,
        freeze_backbone=True,
    ).to(device)

    # Fake 2 scenes: scene0 has 4 persons + 2 pairs; scene1 has 3 persons + 1 pair
    def fake_crops(N, C=3, H=112, W=112):
        return torch.randn(N, C, H, W)

    def fake_boxes(N, img_w=3760, img_h=480):
        x = torch.rand(N) * (img_w - 200) + 200
        y = torch.rand(N) * (img_h - 100)
        w = torch.rand(N) * 100 + 50
        h = torch.rand(N) * 150 + 80
        return torch.stack([x, y, w, h], dim=1)

    scene0 = {
        'person_crops': fake_crops(4),
        'person_boxes': fake_boxes(4),
        'target_pairs': torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
    }
    scene1 = {
        'person_crops': fake_crops(3),
        'person_boxes': fake_boxes(3),
        'target_pairs': torch.tensor([[0, 2]], dtype=torch.long),
    }

    scene_data = [scene0, scene1]

    with torch.no_grad():
        output = model(scene_data)

    logits = output['logits']
    print(f"\nOutput logits shape: {logits.shape}")   # Expected [3, 3]
    print(f"pair_scene_idx:      {output['pair_scene_idx'].tolist()}")  # [0,0,1]
    assert logits.shape == (3, 3), f"Expected [3,3], got {logits.shape}"

    # Test loss
    all_labels = torch.tensor([0, 2, 1], dtype=torch.long, device=device)
    criterion = ResNetStage2Loss(class_weights={0: 1.0, 1: 1.4, 2: 6.1}, gamma=2.0)
    loss, loss_dict = criterion(logits, all_labels)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")

    print("\nAll assertions passed!")
    print(f"Model info: {model.get_model_info()}")
