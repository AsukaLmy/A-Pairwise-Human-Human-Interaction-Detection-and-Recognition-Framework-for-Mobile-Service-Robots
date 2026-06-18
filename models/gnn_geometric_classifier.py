#!/usr/bin/env python3
"""
GNN Geometric-Only Stage2 Classifier – Graph Transformer + Multi-task
No visual backbone. Node features from bounding-box geometry.

Architecture overview:
  1. Node projection:  5D bbox → gnn_hidden_dim
  2. Edge projection:  7D geometric → edge_hidden_dim  (static → learned space)
  3. GraphTransformerLayer × L:
       Step A – Edge update: e'_ij = LN(e + SiLU(W_s·h_i + W_d·h_j + W_e·e))
       Step B – Node update: attention using e'_ij → h'_i  (same as GATLayer)
       → both node AND edge features evolve each layer
  4. Evolved edge extraction for target pairs (symmetric: (A→B + B→A) / 2)
  5. Behavior head (positive pairs only):
       [emb_A, emb_B, |A-B|, A*B, flow_feats_10D, evolved_e] → 3 classes
  6. Group head (positive + negative pairs):
       [emb_A, emb_B, |A-B|, A*B, evolved_e] → 1 binary logit

Multi-task outputs:
    behavior_logits: [P, 3]     – for labeled positive pairs
    group_logits:    [P+Q, 1]   – for all pairs (positive=1, negative=0)
    group_labels:    [P+Q]      – derived from pair membership (no extra annotation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from models.resnet_stage2_classifier import ResNetStage2Loss
    from models.gnn_stage2_classifier import GATLayer, GraphTransformerLayer, CrossPairAttention
    from src.features.geometric_features import extract_geometric_features
except ImportError:
    from resnet_stage2_classifier import ResNetStage2Loss
    from gnn_stage2_classifier import GATLayer, GraphTransformerLayer, CrossPairAttention
    sys.path.insert(0, os.path.join(project_root, 'src', 'features'))
    from geometric_features import extract_geometric_features


# ============================================================================
# Geometric Graph Builder
# ============================================================================

class GeometricGraphBuilder:
    """
    Builds a fully-connected scene graph from bounding boxes.

    Nodes: each person, features = projected from 5D bbox geometry
    Edges: all ordered (i, j) pairs with i != j
    Edge features: 7D from extract_geometric_features() (spatial context)

    Also builds an edge_map: (i, j) → global edge index, for extracting
    evolved edge features for specific pairs after GraphTransformer layers.
    """

    def __init__(self, image_width: int = 3760, image_height: int = 480):
        self.W = image_width
        self.H = image_height

    def build_graph(
        self,
        node_feats: torch.Tensor,    # [N, hidden]  projected node features
        person_boxes: torch.Tensor,  # [N, 4]
    ) -> Dict:
        """
        Returns dict with:
            node_feats:  [N, hidden]
            edge_index:  [2, E]
            edge_feats:  [E, 7]
            src_list, dst_list: List[int] for edge_map construction
        """
        N = node_feats.size(0)
        device = node_feats.device

        if N <= 1:
            return {
                'node_feats': node_feats,
                'edge_index': torch.zeros(2, 0, dtype=torch.long, device=device),
                'edge_feats': torch.zeros(0, 7, dtype=torch.float32, device=device),
                'src_list': [], 'dst_list': [],
            }

        src_list = [i for i in range(N) for j in range(N) if i != j]
        dst_list = [j for i in range(N) for j in range(N) if i != j]
        edge_index = torch.tensor([src_list, dst_list],
                                  dtype=torch.long, device=device)

        boxes_cpu = person_boxes.cpu()
        edge_feats_list = []
        for e in range(len(src_list)):
            geom = extract_geometric_features(
                boxes_cpu[src_list[e]], boxes_cpu[dst_list[e]], self.W, self.H)
            edge_feats_list.append(geom)
        edge_feats = torch.stack(edge_feats_list, dim=0).to(device)

        return {
            'node_feats': node_feats,
            'edge_index': edge_index,
            'edge_feats': edge_feats,
            'src_list':   src_list,
            'dst_list':   dst_list,
        }


# ============================================================================
# Full Geometric GNN Classifier (Graph Transformer + Multi-task)
# ============================================================================

class GNNGeometricClassifier(nn.Module):
    """
    No-backbone GNN Stage2 classifier.

    Forward input:  List[Dict] – one dict per frame, from GNNGeometricDataset
    Forward output: Dict with:
        'behavior_logits': [P_total, 3]      – 3-class interaction
        'group_logits':    [P+Q_total, 1]    – binary group detection
        'group_labels':    [P+Q_total]       – 1 for positive, 0 for negative
        'pair_scene_idx':  [P_total]         – scene index per positive pair
    """

    def __init__(
        self,
        node_feat_dim: int = 5,
        edge_feat_dim: int = 7,
        pair_feat_dim: int = 10,
        edge_hidden_dim: int = 64,
        gnn_hidden_dim: int = 256,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
        edge_hidden_dims: Optional[List[int]] = None,
        num_classes: int = 3,
        gnn_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
        use_graph_transformer: bool = True,
        use_edge_in_cls: bool = True,
        image_width: int = 3760,
        image_height: int = 480,
        # === Improvement II: DropEdge ===
        drop_edge_rate: float = 0.0,
        # === Improvement III: Virtual Global Node ===
        use_virtual_node: bool = False,
        # === Improvement IV: Cross-Pair Attention ===
        use_cross_pair_attn: bool = False,
        cross_pair_dim: int = 256,
    ):
        super().__init__()

        if edge_hidden_dims is None:
            edge_hidden_dims = [512, 256, 128]

        self.gnn_hidden_dim        = gnn_hidden_dim
        self.edge_hidden_dim       = edge_hidden_dim
        self.pair_feat_dim         = pair_feat_dim
        self.use_graph_transformer = use_graph_transformer
        self.use_edge_in_cls       = use_edge_in_cls
        self.drop_edge_rate        = drop_edge_rate
        self.use_virtual_node      = use_virtual_node
        self.use_cross_pair_attn   = use_cross_pair_attn

        # ---- Graph builder ----
        self.graph_builder = GeometricGraphBuilder(image_width, image_height)

        # ---- Node projection: node_feat_dim → gnn_hidden_dim ----
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, gnn_hidden_dim),
            nn.LayerNorm(gnn_hidden_dim),
            nn.SiLU(),
        )

        # ---- Edge projection: 7D static → edge_hidden_dim ----
        # Unifies raw geometric features into the learned edge space
        # before the first transformer layer.
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.SiLU(),
        )

        # ---- Message-passing layers ----
        # GraphTransformerLayer requires node_dim constant across all layers:
        #   concat_heads=True → output H*(D=hidden/H) = hidden  (invariant)
        #   concat_heads=False → output D = hidden/H            (breaks invariant)
        # Therefore always use concat_heads=True.
        self.gt_layers = nn.ModuleList()
        for _ in range(gnn_num_layers):
            if use_graph_transformer:
                self.gt_layers.append(GraphTransformerLayer(
                    node_dim=gnn_hidden_dim,
                    edge_dim=edge_hidden_dim,
                    num_heads=gnn_num_heads,
                    dropout=gnn_dropout,
                    concat_heads=True,   # keeps node_dim = gnn_hidden_dim every layer
                ))
            else:
                # Fallback: original GATLayer (static edges, ablation)
                per_head = gnn_hidden_dim // gnn_num_heads
                self.gt_layers.append(GATLayer(
                    in_dim=gnn_hidden_dim,
                    out_dim=per_head,
                    edge_dim=edge_hidden_dim,
                    num_heads=gnn_num_heads,
                    dropout=gnn_dropout,
                    concat_heads=True,   # keeps output = gnn_hidden_dim
                ))

        node_emb_dim = gnn_hidden_dim   # stays constant (concat H*(D=hidden/H) = hidden)

        # ---- Improvement III: virtual node appends gnn_hidden_dim to behavior input ----
        vn_dim = gnn_hidden_dim if use_virtual_node else 0

        # ---- Behavior classifier (positive pairs only) ----
        # Input: [emb_A, emb_B, |A-B|, A*B] + flow_feats + (evolved_e if enabled)
        #        + (vn_emb if use_virtual_node)
        evolved_dim    = edge_hidden_dim if (use_graph_transformer and use_edge_in_cls) else 0
        edge_cls_input = 4 * node_emb_dim + pair_feat_dim + evolved_dim + vn_dim

        # ---- Improvement IV: Cross-Pair Attention (residual, no dim change) ----
        if use_cross_pair_attn:
            self.cross_pair_attn = CrossPairAttention(
                d_in=edge_cls_input, d_attn=cross_pair_dim,
                num_heads=4, dropout=gnn_dropout)
        else:
            self.cross_pair_attn = None

        layers = []
        prev = edge_cls_input
        for h in edge_hidden_dims:
            layers += [
                nn.Linear(prev, h), nn.LayerNorm(h),
                nn.SiLU(), nn.Dropout(classifier_dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.edge_classifier = nn.Sequential(*layers)

        # ---- Group head (all pairs: positive + negative) ----
        # Input: [emb_A, emb_B, |A-B|, A*B] + (evolved_e if enabled)
        # Does NOT use flow_feats or virtual node (negative pairs have none)
        group_input = 4 * node_emb_dim + evolved_dim
        self.group_head = nn.Sequential(
            nn.Linear(group_input, 256), nn.LayerNorm(256),
            nn.SiLU(), nn.Dropout(classifier_dropout),
            nn.Linear(256, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

        total = sum(p.numel() for p in self.parameters())
        layer_type = "GraphTransformer" if use_graph_transformer else "GAT (ablation)"
        extras = []
        if drop_edge_rate > 0:
            extras.append(f"DropEdge(p={drop_edge_rate})")
        if use_virtual_node:
            extras.append("VirtualNode")
        if use_cross_pair_attn:
            extras.append(f"CrossPairAttn(d={cross_pair_dim})")
        extras_str = " + ".join(extras) if extras else "none"
        print(f"GNNGeometricClassifier [{layer_type}] created:")
        print(f"  Node features:    {node_feat_dim}D → {gnn_hidden_dim}D")
        print(f"  Edge features:    {edge_feat_dim}D → {edge_hidden_dim}D (evolved)")
        print(f"  Pair flow feats:  {pair_feat_dim}D (optical flow, labeled pairs)")
        print(f"  Layers:           {gnn_num_layers} × {gnn_num_heads} heads → {gnn_hidden_dim}D")
        print(f"  Behavior cls in:  {edge_cls_input}D → {edge_hidden_dims} → {num_classes}")
        print(f"  Group head in:    {group_input}D → 256 → 1")
        print(f"  Structural extras: {extras_str}")
        print(f"  Total params:     {total:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Evolved edge extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pair_edges(
        evolved_edges: torch.Tensor,   # [total_E, edge_dim]
        edge_idx:      torch.Tensor,   # [2, total_E]
        pairs:         torch.Tensor,   # [P, 2]  global node indices
        device,
    ) -> torch.Tensor:
        """
        Extract evolved edge features for given pairs, symmetrized.
        Fully vectorized via [P, 1] vs [1, E] broadcasting – no Python loop.

        Returns: [P, edge_dim]
        """
        if evolved_edges.size(0) == 0 or pairs.size(0) == 0:
            return torch.zeros(
                pairs.size(0), evolved_edges.size(1), device=device)

        src_idx  = edge_idx[0]   # [E]
        dst_idx  = edge_idx[1]   # [E]
        A = pairs[:, 0]          # [P]
        B = pairs[:, 1]          # [P]

        # Broadcasting: [P, 1] == [1, E]  →  [P, E] bool
        match_AB = (A[:, None] == src_idx[None, :]) & (B[:, None] == dst_idx[None, :])
        match_BA = (B[:, None] == src_idx[None, :]) & (A[:, None] == dst_idx[None, :])

        idx_AB = match_AB.int().argmax(dim=1)   # [P]  (0 when no match)
        idx_BA = match_BA.int().argmax(dim=1)   # [P]

        has_AB = match_AB.any(dim=1).float().unsqueeze(1)   # [P, 1]
        has_BA = match_BA.any(dim=1).float().unsqueeze(1)   # [P, 1]

        e_AB = evolved_edges[idx_AB] * has_AB   # [P, edge_dim]
        e_BA = evolved_edges[idx_BA] * has_BA   # [P, edge_dim]

        return (e_AB + e_BA) * 0.5   # symmetric

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, scene_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            scene_data: List[Dict], each dict:
                'node_feats':      [N_i, 5]
                'person_boxes':    [N_i, 4]
                'target_pairs':    [P_i, 2]    positive (labeled) pairs
                'pair_flow_feats': [P_i, 10]
                'negative_pairs':  [Q_i, 2]    unlabeled pairs (group=0)

        Returns dict:
            'behavior_logits': [P_total, 3]
            'group_logits':    [P+Q_total, 1]
            'group_labels':    [P+Q_total]  float32
            'pair_scene_idx':  [P_total]
        """
        device = next(self.parameters()).device

        # ----------------------------------------------------------------
        # Step 1: Build super-graph
        # ----------------------------------------------------------------
        all_node_feats      = []
        all_edge_indices    = []
        all_edge_feats      = []
        target_pairs_global = []
        negative_pairs_global = []
        all_pair_flow_feats = []
        pair_scene_idx      = []
        node_offset         = 0

        for scene_idx, scene in enumerate(scene_data):
            nf         = scene['node_feats'].to(device)         # [N_i, 5]
            pos_pairs  = scene['target_pairs'].to(device)       # [P_i, 2]
            flow_feats = scene['pair_flow_feats'].to(device)    # [P_i, 10]
            neg_pairs  = scene['negative_pairs'].to(device)     # [Q_i, 2]

            # Project 5D → hidden_dim
            nf_proj = self.node_proj(nf)

            # Use precomputed edge_index / edge_feats when available (avoids
            # recomputing N*(N-1) geometric features on every forward pass).
            if 'pre_edge_index' in scene and 'pre_edge_feats' in scene:
                ei = scene['pre_edge_index'].to(device)         # [2, E_i]
                ef_raw = scene['pre_edge_feats'].to(device)     # [E_i, 7]
            else:
                # Fallback: compute on-the-fly (slower)
                boxes = scene['person_boxes'].to(device)
                graph = self.graph_builder.build_graph(nf_proj, boxes)
                ei     = graph['edge_index']
                ef_raw = graph['edge_feats']

            # Project raw 7D edge features → edge_hidden_dim
            ef_proj = self.edge_proj(ef_raw)                    # [E_i, edge_hidden]

            all_node_feats.append(nf_proj)
            all_edge_indices.append(ei + node_offset)
            all_edge_feats.append(ef_proj)

            target_pairs_global.append(pos_pairs + node_offset)
            negative_pairs_global.append(neg_pairs + node_offset)
            all_pair_flow_feats.append(flow_feats)
            pair_scene_idx.extend([scene_idx] * pos_pairs.size(0))

            node_offset += nf.size(0)

        # ----------------------------------------------------------------
        # Step 2: Message passing (Graph Transformer or GAT)
        # ----------------------------------------------------------------
        h        = torch.cat(all_node_feats,   dim=0)          # [total_N, hidden]
        edge_idx = torch.cat(all_edge_indices, dim=1)          # [2, total_E]
        ef       = torch.cat(all_edge_feats,   dim=0)          # [total_E, edge_hidden]

        # ---- Improvement II: DropEdge ----
        # Randomly drop non-target-pair edges during training to regularize GNN.
        # Target pair edges (both directions) are always kept so that
        # _extract_pair_edges can still find evolved edge features for them.
        if self.training and self.drop_edge_rate > 0 and edge_idx.size(1) > 0:
            all_pos = torch.cat(target_pairs_global, dim=0)    # [P_total, 2]
            t_src = torch.cat([all_pos[:, 0], all_pos[:, 1]])  # [2P]
            t_dst = torch.cat([all_pos[:, 1], all_pos[:, 0]])  # [2P]
            e_src = edge_idx[0]; e_dst = edge_idx[1]           # [E]
            # is_target[e] = True if edge e connects a target pair (either direction)
            is_target = ((e_src[:, None] == t_src[None, :]) &
                         (e_dst[:, None] == t_dst[None, :])).any(dim=1)  # [E]
            rand_keep = torch.rand(e_src.size(0), device=device) > self.drop_edge_rate
            keep = rand_keep | is_target
            edge_idx = edge_idx[:, keep]
            ef = ef[keep]

        # ---- Improvement III: Virtual Global Node ----
        # Add one virtual node per scene (at end of each scene's node block),
        # connected bidirectionally to all person nodes in that scene.
        # After GNN propagation the virtual node's embedding encodes the full
        # scene context and is appended to each behavior pair's representation.
        vn_indices = []   # virtual node global index per scene, for retrieval later
        if self.use_virtual_node:
            augmented_node_feats = []
            augmented_edge_idx   = []
            augmented_ef         = []
            vn_offset = 0    # running count of nodes seen so far (including vn's added)
            orig_offsets = []   # original node count per scene
            for s_idx, scene in enumerate(scene_data):
                nf_proj = all_node_feats[s_idx]   # [N_s, hidden]
                N_s = nf_proj.size(0)
                # Virtual node feature = mean of scene's projected node feats
                vn_feat = nf_proj.mean(dim=0, keepdim=True)   # [1, hidden]
                augmented_node_feats.append(nf_proj)
                augmented_node_feats.append(vn_feat)
                vn_global_idx = vn_offset + N_s   # index of vn in augmented super-graph
                vn_indices.append(vn_global_idx)
                orig_offsets.append(N_s)
                vn_offset += N_s + 1  # +1 for the vn itself

            # Rebuild node tensor from augmented list
            h = torch.cat(augmented_node_feats, dim=0)

            # Rebuild edges: offset all existing edges to account for vn insertions,
            # then add bidirectional vn↔person edges for each scene.
            # We need to remap edge_idx which was built without vn nodes.
            # Recompute cumulative offsets with vn included.
            # Original scene node counts (without vn):
            scene_n = [all_node_feats[s].size(0) for s in range(len(scene_data))]
            # New scene boundaries (with vn): scene s occupies [vn_cum[s], vn_cum[s]+n[s]+1)
            vn_cum = [0]
            for n in scene_n:
                vn_cum.append(vn_cum[-1] + n + 1)
            # Original boundaries (without vn):
            orig_cum = [0]
            for n in scene_n:
                orig_cum.append(orig_cum[-1] + n)

            # Remap existing edge_idx: for each edge, find which scene it belongs to,
            # then add the number of vn's inserted before that scene.
            e_src_orig = edge_idx[0]; e_dst_orig = edge_idx[1]
            def remap(idx_t):
                new_idx = idx_t.clone()
                for s in range(len(scene_data)):
                    # Nodes in scene s originally occupy [orig_cum[s], orig_cum[s+1])
                    mask = (idx_t >= orig_cum[s]) & (idx_t < orig_cum[s + 1])
                    new_idx[mask] = idx_t[mask] + s   # add s vn's inserted before scene s
                return new_idx
            new_src = remap(e_src_orig)
            new_dst = remap(e_dst_orig)
            new_edge_idx = torch.stack([new_src, new_dst])  # [2, E]

            # Build vn↔person edges for each scene
            vn_edge_src = []; vn_edge_dst = []
            for s in range(len(scene_data)):
                N_s = scene_n[s]
                base = vn_cum[s]
                vn_idx = base + N_s
                person_range = torch.arange(N_s, device=device) + base
                vn_src_s  = torch.full((N_s,), vn_idx, dtype=torch.long, device=device)
                vn_dst_s  = person_range
                p_src_s   = person_range
                p_dst_s   = torch.full((N_s,), vn_idx, dtype=torch.long, device=device)
                vn_edge_src.append(torch.cat([vn_src_s, p_src_s]))
                vn_edge_dst.append(torch.cat([vn_dst_s, p_dst_s]))

            vn_src_all = torch.cat(vn_edge_src)   # [2 * sum(N_s)]
            vn_dst_all = torch.cat(vn_edge_dst)
            vn_edge_all = torch.stack([vn_src_all, vn_dst_all])  # [2, 2*sum(N_s)]
            zero_vn_ef  = torch.zeros(vn_src_all.size(0), ef.size(1), device=device)

            edge_idx = torch.cat([new_edge_idx, vn_edge_all], dim=1)
            ef       = torch.cat([ef, zero_vn_ef], dim=0)

            # Remap target/negative pair global indices too
            all_pos_remap = []
            all_neg_remap = []
            for s in range(len(scene_data)):
                pos_g = target_pairs_global[s]    # already offset by orig node offsets
                neg_g = negative_pairs_global[s]
                delta = s  # number of vn's inserted before scene s
                all_pos_remap.append(pos_g + delta)
                all_neg_remap.append(neg_g + delta)
            target_pairs_global  = all_pos_remap
            negative_pairs_global = all_neg_remap

        if self.use_graph_transformer:
            # GraphTransformerLayer: returns (h, ef) with BOTH updated
            for gt in self.gt_layers:
                h, ef = gt(h, edge_idx, ef)
            evolved_edges = ef                                  # [total_E, edge_hidden]
        else:
            # GATLayer fallback (static edges for ablation)
            for gat in self.gt_layers:
                h = gat(h, edge_idx, ef)
            evolved_edges = ef                                  # unchanged

        # ----------------------------------------------------------------
        # Step 3: Extract positive pair embeddings
        # ----------------------------------------------------------------
        all_pos_pairs  = torch.cat(target_pairs_global,  dim=0)  # [P_total, 2]
        all_neg_pairs  = torch.cat(negative_pairs_global, dim=0) # [Q_total, 2]
        flow_feats_cat = torch.cat(all_pair_flow_feats,   dim=0) # [P_total, 10]

        emb_A_pos = h[all_pos_pairs[:, 0]]
        emb_B_pos = h[all_pos_pairs[:, 1]]

        # ---- Evolved edge for positive pairs (symmetric) ----
        if self.use_graph_transformer and self.use_edge_in_cls:
            e_pos = self._extract_pair_edges(
                evolved_edges, edge_idx, all_pos_pairs, device)   # [P, edge_hidden]
        else:
            e_pos = None

        # ----------------------------------------------------------------
        # Step 4: Behavior classification (positive pairs only)
        # ----------------------------------------------------------------
        parts = [emb_A_pos, emb_B_pos,
                 torch.abs(emb_A_pos - emb_B_pos),
                 emb_A_pos * emb_B_pos,
                 flow_feats_cat]
        if e_pos is not None:
            parts.append(e_pos)

        # ---- Improvement III: append virtual node embedding ----
        if self.use_virtual_node and vn_indices:
            # Build [P_total, gnn_hidden_dim] by repeating each scene's vn_emb
            vn_parts = []
            for s_idx, scene in enumerate(scene_data):
                n_pos = scene['target_pairs'].size(0)
                if n_pos > 0:
                    vn_emb_s = h[vn_indices[s_idx]].unsqueeze(0).expand(n_pos, -1)
                    vn_parts.append(vn_emb_s)
            if vn_parts:
                parts.append(torch.cat(vn_parts, dim=0))     # [P_total, hidden]

        behavior_repr = torch.cat(parts, dim=1)               # [P, edge_cls_input]

        # ---- Improvement IV: Cross-Pair Attention (per-scene) ----
        # Apply self-attention over all pairs in each scene separately, then
        # reassemble into the full [P_total, D] tensor.
        if self.cross_pair_attn is not None:
            # Build scene boundaries for positive pairs
            scene_sizes = [scene['target_pairs'].size(0) for scene in scene_data]
            updated_parts = []
            offset = 0
            for n_pos in scene_sizes:
                if n_pos > 0:
                    chunk = behavior_repr[offset: offset + n_pos]   # [n_pos, D]
                    updated_parts.append(self.cross_pair_attn(chunk))
                    offset += n_pos
            behavior_repr = torch.cat(updated_parts, dim=0) if updated_parts \
                else behavior_repr                                # [P, D] – residual

        behavior_logits = self.edge_classifier(behavior_repr)  # [P, 3]

        # ----------------------------------------------------------------
        # Step 5: Group detection (positive + negative pairs)
        # ----------------------------------------------------------------
        # Positive pairs: group_label = 1
        emb_A_neg = h[all_neg_pairs[:, 0]] if all_neg_pairs.size(0) > 0 \
            else torch.zeros(0, self.gnn_hidden_dim, device=device)
        emb_B_neg = h[all_neg_pairs[:, 1]] if all_neg_pairs.size(0) > 0 \
            else torch.zeros(0, self.gnn_hidden_dim, device=device)

        # Evolved edge for negative pairs (symmetric)
        if self.use_graph_transformer and all_neg_pairs.size(0) > 0:
            e_neg = self._extract_pair_edges(
                evolved_edges, edge_idx, all_neg_pairs, device)   # [Q, edge_hidden]
        else:
            e_neg = (torch.zeros(all_neg_pairs.size(0), self.edge_hidden_dim, device=device)
                     if self.use_graph_transformer else None)

        def _group_repr(emb_A, emb_B, e_ij):
            parts = [emb_A, emb_B, torch.abs(emb_A - emb_B), emb_A * emb_B]
            if e_ij is not None and e_ij.size(0) > 0:
                parts.append(e_ij)
            return torch.cat(parts, dim=1)

        if e_pos is not None:
            e_pos_group = e_pos
        elif self.use_graph_transformer:
            e_pos_group = torch.zeros(
                all_pos_pairs.size(0), self.edge_hidden_dim, device=device)
        else:
            e_pos_group = None

        gr_pos = _group_repr(emb_A_pos, emb_B_pos, e_pos_group)  # [P, group_input]
        gr_neg = _group_repr(emb_A_neg, emb_B_neg, e_neg)        # [Q, group_input]

        # Concatenate positive + negative for group head
        if gr_neg.size(0) > 0:
            group_repr = torch.cat([gr_pos, gr_neg], dim=0)       # [P+Q, group_input]
        else:
            group_repr = gr_pos

        group_logits = self.group_head(group_repr)                 # [P+Q, 1]

        # Derive group labels (no extra annotation needed):
        # positive pairs (labeled) → 1, negative pairs (unlabeled) → 0
        P = all_pos_pairs.size(0)
        Q = all_neg_pairs.size(0)
        group_labels = torch.cat([
            torch.ones(P,  dtype=torch.float32, device=device),
            torch.zeros(Q, dtype=torch.float32, device=device),
        ])

        return {
            'behavior_logits': behavior_logits,
            'group_logits':    group_logits,
            'group_labels':    group_labels,
            'pair_scene_idx':  torch.tensor(
                pair_scene_idx, dtype=torch.long, device=device),
        }

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            'model_type':          'GNNGeometricClassifier',
            'total_params':        total,
            'use_graph_transformer': self.use_graph_transformer,
            'gnn_layers':          len(self.gt_layers),
            'gnn_hidden_dim':      self.gnn_hidden_dim,
            'edge_hidden_dim':     self.edge_hidden_dim,
            'pair_feat_dim':       self.pair_feat_dim,
        }


# ============================================================================
# Factory
# ============================================================================

def create_gnn_geometric_model(config) -> GNNGeometricClassifier:
    return GNNGeometricClassifier(
        node_feat_dim=config.node_feat_dim,
        edge_feat_dim=config.edge_feat_dim,
        pair_feat_dim=config.pair_feat_dim,
        edge_hidden_dim=config.edge_hidden_dim,
        gnn_hidden_dim=config.gnn_hidden_dim,
        gnn_num_layers=config.gnn_num_layers,
        gnn_num_heads=config.gnn_num_heads,
        edge_hidden_dims=list(config.edge_hidden_dims),
        num_classes=config.num_classes,
        gnn_dropout=config.gnn_dropout,
        classifier_dropout=config.classifier_dropout,
        use_graph_transformer=config.use_graph_transformer,
        use_edge_in_cls=config.use_edge_in_cls,
        image_width=config.image_width,
        image_height=config.image_height,
        drop_edge_rate=getattr(config, 'drop_edge_rate', 0.0),
        use_virtual_node=getattr(config, 'use_virtual_node', False),
        use_cross_pair_attn=getattr(config, 'use_cross_pair_attn', False),
        cross_pair_dim=getattr(config, 'cross_pair_dim', 256),
    )


# ============================================================================
# Sanity test
# ============================================================================

if __name__ == '__main__':
    print("Testing GNNGeometricClassifier (Graph Transformer + Multi-task)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GNNGeometricClassifier(
        node_feat_dim=5, pair_feat_dim=10,
        edge_hidden_dim=64,
        gnn_hidden_dim=256, gnn_num_layers=2, gnn_num_heads=4,
        use_graph_transformer=True, use_edge_in_cls=True,
    ).to(device)

    def fake_scene(N, P, Q, W=3760, H=480):
        x = torch.rand(N) * (W - 300) + 200
        y = torch.rand(N) * (H - 100)
        w = torch.rand(N) * 80 + 40
        h = torch.rand(N) * 120 + 80
        node_feats   = torch.stack(
            [(x+w/2)/W, (y+h/2)/H, w/W, h/H, h/(w+1e-6)], dim=1)
        person_boxes = torch.stack([x, y, w, h], dim=1)
        pos_pairs = torch.randint(0, N, (P, 2))
        for i in range(P):
            while pos_pairs[i, 0] == pos_pairs[i, 1]:
                pos_pairs[i, 1] = torch.randint(0, N, (1,))
        neg_pairs = torch.randint(0, N, (Q, 2))
        for i in range(Q):
            while neg_pairs[i, 0] == neg_pairs[i, 1]:
                neg_pairs[i, 1] = torch.randint(0, N, (1,))
        return {
            'node_feats':      node_feats,
            'person_boxes':    person_boxes,
            'target_pairs':    pos_pairs,
            'pair_flow_feats': torch.rand(P, 10),
            'negative_pairs':  neg_pairs,
        }

    scenes = [fake_scene(5, 3, 6), fake_scene(4, 2, 4)]
    out = model(scenes)

    P_total = 3 + 2   # 5 positive pairs total
    Q_total = 6 + 4   # 10 negative pairs total
    print(f"behavior_logits: {out['behavior_logits'].shape}")   # [5, 3]
    print(f"group_logits:    {out['group_logits'].shape}")      # [15, 1]
    print(f"group_labels:    {out['group_labels'].shape}")      # [15]
    assert out['behavior_logits'].shape == (P_total, 3)
    assert out['group_logits'].shape == (P_total + Q_total, 1)

    # Test multi-task loss
    from models.gnn_multitask_loss import GNNMultiTaskLoss
    criterion = GNNMultiTaskLoss()
    behavior_labels = torch.randint(0, 3, (P_total,)).to(device)
    loss, d = criterion(
        out['behavior_logits'], behavior_labels,
        out['group_logits'],    out['group_labels'],
    )
    loss.backward()
    print(f"total_loss:    {d['total_loss']:.4f}")
    print(f"behavior_loss: {d['behavior_loss']:.4f}")
    print(f"group_loss:    {d['group_loss']:.4f}")
    print(f"MPCA:          {d['mpca']:.3f}")
    print(f"Model info: {model.get_model_info()}")
    print("All tests passed!")
