#!/usr/bin/env python3
"""
CAD GNN Geometric Dataset — scene-level (frame-level) batching.

Mirrors gnn_geometric_dataset.py but uses CADAnnotationParser instead of JRDB
JSON labels.  Two modes:

  stage='stage1'  Binary group detection.
                  target_pairs  = positive pairs (same social_group_id)
                  pair_labels   = zeros (dummy; lambda_behavior=0 in training)
                  pair_flow_feats = zeros
                  negative_pairs = non-positive pairs (sampled at neg_ratio;
                                   neg_ratio<=0 → keep ALL for test clustering)

  stage='stage2'  Activity classification.
                  target_pairs  = positive pairs with social_activity_id labels
                  pair_labels   = activity class (0-5 raw, or 0-2 merged)
                  pair_flow_feats = 10D from GeometricFlowExtractor
                  negative_pairs = sampled at neg_ratio

CAD-specific notes:
  - Bboxes: [x1,y1,x2,y2] — converted to [x,y,w,h] for all feature calls.
  - Images: cad_root/seqXX/frameYYYY.jpg
  - Previous frame: frame_id - 1 (via parser.get_prev_frame_path).
  - Image size: 720 × 480 (no panoramic wraparound).
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from collections import Counter
from itertools import combinations

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from datasets.cad_annotation_parser import CADAnnotationParser
from src.features.geometric_features import extract_geometric_features_batch

try:
    from PIL import Image
    from src.features.geometric_flow_extractor import GeometricFlowExtractor
    from src.features.interaction_synchrony import compute_interaction_synchrony
    _FLOW_AVAILABLE = True
except ImportError:
    _FLOW_AVAILABLE = False


class CADGNNGeometricDataset(Dataset):
    """
    Scene-level GNN dataset for the CAD (Cornell Activity Dataset).

    Node features per person (5D, from bounding box only):
        [cx_norm, cy_norm, w_norm, h_norm, aspect_ratio]
    Optionally extended to 13D by appending 8D per-node optical flow
    when flow_node_feats=True.

    Pair flow features (10D / 11D) — Stage 2 only.
    Edge features: 7D geometric, optionally 17D with flow injection.
    """

    # Class merge: 6 classes → 3 classes
    # Original: NA(0), Crossing(1), Waiting(2), Queuing(3), Walking(4), Talking(5)
    # Merged:   Moving(0), Standing(1), Talking(2)   [NA is skipped]
    CLASS_MERGE_MAP = {1: 0, 2: 1, 3: 1, 4: 0, 5: 2}
    CLASS_NAMES_6 = ['NA', 'Crossing', 'Waiting', 'Queuing', 'Walking', 'Talking']
    CLASS_NAMES_3 = ['Moving', 'Standing', 'Talking']

    def __init__(
        self,
        cad_root: str,
        sequences: List[int],
        stage: str = 'stage2',          # 'stage1' or 'stage2'
        image_width: int = 720,
        image_height: int = 480,
        class_merge: bool = False,       # stage2 only: merge to 3 classes
        group_neg_ratio: float = 2.0,    # <=0 → keep all negatives
        graph_knn: int = 0,
        inject_flow_to_edges: bool = True,
        flow_node_feats: bool = True,
        cache_dir: Optional[str] = None,
        use_individual_action_feat: bool = False,
        use_extra_pair_feats: bool = False,
    ):
        assert stage in ('stage1', 'stage2'), f"stage must be 'stage1' or 'stage2', got {stage}"

        self.cad_root          = cad_root
        self.sequences         = sequences
        self.stage             = stage
        self.image_width       = image_width
        self.image_height      = image_height
        self.class_merge       = class_merge
        self.group_neg_ratio   = group_neg_ratio
        self.graph_knn         = graph_knn
        self.inject_flow_to_edges = inject_flow_to_edges
        self.flow_node_feats   = flow_node_feats
        self.cache_dir         = cache_dir
        self.use_individual_action_feat = use_individual_action_feat
        self.use_extra_pair_feats = use_extra_pair_feats

        self.parser = CADAnnotationParser(cad_root)

        # Image / flow availability
        self._images_available = _FLOW_AVAILABLE
        if self._images_available:
            self.flow_extractor = GeometricFlowExtractor(
                flow_bound=20.0, cache_enabled=True)
        else:
            self.flow_extractor = None

        self.frames: List[Dict] = []
        self.all_pair_labels: List[int] = []

        self._load_data()

        total_pairs = sum(f['num_pairs'] for f in self.frames)
        print(f"CADGNNGeometricDataset ({stage}, seqs={sequences[:5]}{'...' if len(sequences)>5 else ''}): "
              f"{len(self.frames)} frames, {total_pairs} target pairs")
        if self.all_pair_labels:
            print(f"  Label distribution: {dict(Counter(self.all_pair_labels))}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load all frames from the given CAD sequences."""
        from tqdm import tqdm

        for seq_num in tqdm(self.sequences, desc="[CAD GNN] loading sequences"):
            try:
                df = self.parser.load_sequence_annotations(seq_num)
            except FileNotFoundError as e:
                print(f"  Warning: {e}")
                continue

            for frame_id, frame_df in df.groupby('frame_id'):
                persons = frame_df.to_dict('records')
                if len(persons) < 2:
                    continue

                # Check for previous frame (required for optical flow in stage2)
                has_prev = False
                prev_frame_path = self.parser.get_prev_frame_path(seq_num, frame_id)
                curr_frame_path = self.parser.get_frame_path(seq_num, frame_id)
                if prev_frame_path is not None and prev_frame_path.exists() and curr_frame_path.exists():
                    has_prev = True

                if self.stage == 'stage2' and not has_prev:
                    continue  # optical flow is needed for stage2 pair features

                # Build person lookup indexed by position in sorted list
                sorted_persons = sorted(persons, key=lambda p: p['track_id'])
                pid_to_local = {p['track_id']: i for i, p in enumerate(sorted_persons)}

                # CAD bboxes are [x1, y1, x2, y2]; convert to [x, y, w, h]
                all_boxes_xywh = [
                    _x1y1x2y2_to_xywh(p['x1'], p['y1'], p['x2'], p['y2'])
                    for p in sorted_persons
                ]

                # -- Find target pairs --
                target_pairs_info: List[Tuple[int, int, int]] = []
                pair_boxes: List[Tuple[list, list]] = []

                for i, j in combinations(range(len(sorted_persons)), 2):
                    pA = sorted_persons[i]
                    pB = sorted_persons[j]

                    same_group = (pA['social_group_id'] == pB['social_group_id'])

                    if self.stage == 'stage1':
                        if not same_group:
                            continue  # only interacting pairs are target_pairs
                        label = 0   # dummy label for stage1 (lambda_behavior=0)

                    else:  # stage2
                        if not same_group:
                            continue  # only positive pairs for activity classification
                        act_id = pA['social_activity_id']
                        if self.class_merge:
                            if act_id == 0:
                                continue  # skip NA
                            if act_id not in self.CLASS_MERGE_MAP:
                                continue
                            label = self.CLASS_MERGE_MAP[act_id]
                        else:
                            if act_id < 0 or act_id > 5:
                                continue
                            label = act_id

                    local_A = pid_to_local[pA['track_id']]
                    local_B = pid_to_local[pB['track_id']]
                    target_pairs_info.append((local_A, local_B, label))
                    pair_boxes.append((all_boxes_xywh[local_A], all_boxes_xywh[local_B]))

                if not target_pairs_info:
                    continue

                # -- Node features (5D bbox) --
                node_feats = self._compute_node_feats(all_boxes_xywh)

                # -- Pre-compute edge index + 7D geometric edge features --
                N = len(sorted_persons)
                boxes_t = torch.tensor(all_boxes_xywh, dtype=torch.float32)  # [N, 4]
                pre_edge_index, pre_edge_feats, src_list, dst_list = \
                    self._build_edges(N, boxes_t, target_pairs_info)

                # -- Load images once (shared by pair flow + node flow) --
                prev_img, curr_img, has_images = self._open_images(
                    str(prev_frame_path) if has_prev else '',
                    str(curr_frame_path) if has_prev else '',
                )

                # -- Pair flow features (10D, stage2 only) --
                if self.stage == 'stage2':
                    pair_flow_feats, _ = self._compute_all_pair_flow_feats_from_images(
                        prev_img, curr_img, pair_boxes)
                else:
                    # Stage 1: zeros placeholder
                    p_dim = 11 if self.flow_node_feats else 10
                    pair_flow_feats = torch.zeros(len(target_pairs_info), p_dim,
                                                  dtype=torch.float32)

                # -- Per-node 8D optical flow features (5D → 13D node) --
                if self.flow_node_feats and has_images:
                    node_flow = self._compute_node_flow_feats(
                        prev_img, curr_img, all_boxes_xywh)
                    node_feats = torch.cat([node_feats, node_flow], dim=1)  # [N, 13]
                elif self.flow_node_feats:
                    node_feats = torch.cat(
                        [node_feats, torch.zeros(N, 8, dtype=torch.float32)], dim=1)

                # -- Extra node features: individual_action_id + group_size_norm (+2D) --
                # Appended after flow feats so h_motion/v_motion stay at indices 9,10.
                if self.use_individual_action_feat:
                    grp_counts = Counter(p['social_group_id'] for p in sorted_persons)
                    n_pers = len(sorted_persons)
                    extra_node = torch.tensor([[
                        p.get('individual_action_id', 0) / 5.0,
                        grp_counts[p['social_group_id']] / n_pers,
                    ] for p in sorted_persons], dtype=torch.float32)  # [N, 2]
                    node_feats = torch.cat([node_feats, extra_node], dim=1)

                # -- Approach rate (→ 11D pair feature) --
                if self.flow_node_feats and self.stage == 'stage2':
                    approach_t = self._compute_approach_rates_cad(
                        node_feats, target_pairs_info)
                    pair_flow_feats = torch.cat([pair_flow_feats, approach_t], dim=1)

                # -- Extra pair features: axis angle + lateral rate (+2D or +3D) --
                if self.use_extra_pair_feats and self.stage == 'stage2':
                    pair_geom_extras = self._compute_pair_geometry_extras(
                        node_feats, target_pairs_info)
                    pair_flow_feats = torch.cat([pair_flow_feats, pair_geom_extras], dim=1)

                # -- Close images --
                if has_images:
                    if prev_img is not None:
                        prev_img.close()
                    if curr_img is not None:
                        curr_img.close()

                # -- Flow injection into edge features (7D → 17D) --
                if self.inject_flow_to_edges and pre_edge_feats.size(0) > 0:
                    zero_flow = torch.zeros(10, dtype=torch.float32)
                    flow_map: dict = {}
                    for k, (A, B, _) in enumerate(target_pairs_info):
                        fv = pair_flow_feats[k, :10]
                        flow_map[(A, B)] = fv
                        flow_map[(B, A)] = fv   # symmetric
                    flow_rows = [
                        flow_map.get((src_list[e], dst_list[e]), zero_flow)
                        for e in range(len(src_list))
                    ]
                    flow_part = torch.stack(flow_rows, dim=0)
                    pre_edge_feats = torch.cat([pre_edge_feats, flow_part], dim=1)

                # -- Negative pair sampling --
                negative_pairs = self._sample_negatives(N, target_pairs_info)

                # -- Track metadata for test-time evaluation --
                track_ids = [p['track_id'] for p in sorted_persons]
                social_group_ids = [p['social_group_id'] for p in sorted_persons]
                social_activity_ids = [p['social_activity_id'] for p in sorted_persons]

                self.frames.append({
                    'node_feats':          node_feats,
                    'person_boxes':        boxes_t,
                    'pre_edge_index':      pre_edge_index,
                    'pre_edge_feats':      pre_edge_feats,
                    'target_pairs_info':   target_pairs_info,
                    'pair_flow_feats':     pair_flow_feats,
                    'negative_pairs':      negative_pairs,
                    'num_persons':         N,
                    'num_pairs':           len(target_pairs_info),
                    # Test-time metadata
                    'seq_num':             seq_num,
                    'frame_id':            frame_id,
                    'track_ids':           track_ids,
                    'social_group_ids':    social_group_ids,
                    'social_activity_ids': social_activity_ids,
                })
                self.all_pair_labels.extend(p[2] for p in target_pairs_info)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_edges(
        self,
        N: int,
        boxes_t: torch.Tensor,
        target_pairs_info: List[Tuple[int, int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
        """Build fully-connected or K-NN edge index + 7D geometric edge features."""
        if N <= 1:
            empty_idx  = torch.zeros(2, 0, dtype=torch.long)
            empty_feat = torch.zeros(0, 7, dtype=torch.float32)
            return empty_idx, empty_feat, [], []

        if self.graph_knn > 0 and N > self.graph_knn + 1:
            # K-NN by normalised centre-point distance; target pair edges forced
            node_f5 = self._compute_node_feats(boxes_t.tolist())   # [N, 5]
            positions = node_f5[:, 0:2]                             # cx_norm, cy_norm
            dists = torch.cdist(positions, positions)
            dists.fill_diagonal_(float('inf'))
            K = min(self.graph_knn, N - 1)
            _, knn_idx = dists.topk(K, dim=1, largest=False)
            edge_set: set = set()
            src_list: List[int] = []
            dst_list: List[int] = []
            for i in range(N):
                for j in knn_idx[i].tolist():
                    for s, d in [(i, j), (j, i)]:
                        if (s, d) not in edge_set:
                            src_list.append(s)
                            dst_list.append(d)
                            edge_set.add((s, d))
            for A, B, _ in target_pairs_info:
                for s, d in [(A, B), (B, A)]:
                    if (s, d) not in edge_set:
                        src_list.append(s)
                        dst_list.append(d)
                        edge_set.add((s, d))
        else:
            src_list = [i for i in range(N) for j in range(N) if i != j]
            dst_list = [j for i in range(N) for j in range(N) if i != j]

        src_t = torch.tensor(src_list, dtype=torch.long)
        dst_t = torch.tensor(dst_list, dtype=torch.long)
        pre_edge_index = torch.stack([src_t, dst_t], dim=0)
        pre_edge_feats = extract_geometric_features_batch(
            boxes_t[src_t], boxes_t[dst_t],
            self.image_width, self.image_height,
        )   # [E, 7]
        return pre_edge_index, pre_edge_feats, src_list, dst_list

    # ------------------------------------------------------------------
    # Negative pair sampling
    # ------------------------------------------------------------------

    def _sample_negatives(
        self,
        N: int,
        target_pairs_info: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """Sample non-positive pairs. group_neg_ratio<=0 → keep all."""
        pos_set = {(A, B) for A, B, _ in target_pairs_info}
        all_neg = [
            (i, j) for i in range(N) for j in range(i + 1, N)
            if (i, j) not in pos_set
        ]
        if self.group_neg_ratio > 0:
            max_neg = max(1, int(self.group_neg_ratio * len(target_pairs_info)))
            if len(all_neg) > max_neg:
                all_neg = random.sample(all_neg, max_neg)
        return (torch.tensor(all_neg, dtype=torch.long)
                if all_neg else torch.zeros(0, 2, dtype=torch.long))

    def _resample_negative_pairs(self):
        """Re-sample negative pairs with current random state (for caching)."""
        for frame in self.frames:
            frame['negative_pairs'] = self._sample_negatives(
                frame['num_persons'], frame['target_pairs_info'])

    # ------------------------------------------------------------------
    # Node features
    # ------------------------------------------------------------------

    def _compute_node_feats(self, boxes_xywh) -> torch.Tensor:
        """5D geometric node features: [cx_norm, cy_norm, w_norm, h_norm, aspect_ratio]."""
        W, H = float(self.image_width), float(self.image_height)
        t = torch.tensor(boxes_xywh, dtype=torch.float32) if not isinstance(boxes_xywh, torch.Tensor) else boxes_xywh
        x, y, w, h = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        return torch.stack([
            (x + w * 0.5) / W,
            (y + h * 0.5) / H,
            w / W,
            h / H,
            h / (w + 1e-6),
        ], dim=1)   # [N, 5]

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _open_images(self, prev_path: str, curr_path: str):
        if not self._images_available or not prev_path or not curr_path:
            return None, None, False
        if not (os.path.exists(prev_path) and os.path.exists(curr_path)):
            return None, None, False
        try:
            prev_img = Image.open(prev_path).convert('RGB')
            curr_img = Image.open(curr_path).convert('RGB')
            return prev_img, curr_img, True
        except Exception:
            return None, None, False

    # ------------------------------------------------------------------
    # Pair flow features (10D, same as JRDB GNN dataset)
    # ------------------------------------------------------------------

    def _compute_all_pair_flow_feats_from_images(
        self,
        prev_img,
        curr_img,
        pair_boxes: List[Tuple[list, list]],
    ) -> Tuple[torch.Tensor, bool]:
        P = len(pair_boxes)
        zero = torch.zeros(P, 10, dtype=torch.float32)
        if not self._images_available or P == 0 or prev_img is None or curr_img is None:
            return zero, False
        try:
            rows = [self._compute_pair_flow_feat(prev_img, curr_img, bA, bB)
                    for bA, bB in pair_boxes]
            return torch.stack(rows, dim=0), True
        except Exception:
            return zero, False

    def _compute_pair_flow_feat(self, prev_img, curr_img, box_A, box_B) -> torch.Tensor:
        """10D symmetrised pair flow feature (same construction as JRDB GNN)."""
        try:
            feat_a = self.flow_extractor.extract_geometric_features(
                prev_img, curr_img, box_A, box_B)
            feat_b = self.flow_extractor.extract_geometric_features(
                prev_img, curr_img, box_B, box_A)
            sym = torch.tensor([
                feat_a[0],
                feat_a[1],
                (feat_a[2] + feat_b[2]) * 0.5,
                (feat_a[3] + feat_b[3]) * 0.5,
                (feat_a[4] + feat_b[4]) * 0.5,
                feat_a[5],
                feat_a[6],
                feat_a[7],
                (feat_a[8] + feat_b[8]) * 0.5,
            ], dtype=torch.float32)
            sync = compute_interaction_synchrony(feat_a.unsqueeze(0), feat_b.unsqueeze(0))
            sync_val = float(sync.item() if hasattr(sync, 'item') else sync)
            return torch.cat([sym, torch.tensor([sync_val], dtype=torch.float32)])
        except Exception:
            return torch.zeros(10, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Per-node 8D optical flow features
    # ------------------------------------------------------------------

    def _compute_node_flow_feats(
        self, prev_img, curr_img, all_boxes_xywh: List[list]
    ) -> torch.Tensor:
        """8D per-person optical flow statistics (same as JRDB GNN dataset)."""
        import cv2
        import numpy as np

        N = len(all_boxes_xywh)
        zero = torch.zeros(N, 8, dtype=torch.float32)
        if not self._images_available or N == 0 or prev_img is None or curr_img is None:
            return zero
        try:
            prev_gray = np.array(prev_img.convert('L'))
            curr_gray = np.array(curr_img.convert('L'))
        except Exception:
            return zero
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        except Exception:
            return zero

        H, W = flow.shape[:2]
        flow_bound = 20.0
        rows = []
        for box in all_boxes_xywh:
            x, y, w, h = box
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(W, int(x + w)), min(H, int(y + h))
            if x2 <= x1 or y2 <= y1:
                rows.append(torch.zeros(8, dtype=torch.float32))
                continue
            roi_x = flow[y1:y2, x1:x2, 0]
            roi_y = flow[y1:y2, x1:x2, 1]
            n_pix = roi_x.size
            if n_pix == 0:
                rows.append(torch.zeros(8, dtype=torch.float32))
                continue
            roi_x_abs = np.abs(roi_x)
            mag = np.sqrt(roi_x_abs ** 2 + roi_y ** 2)
            mag_mean  = float(np.mean(mag))  / flow_bound
            mag_std   = float(np.std(mag))   / flow_bound
            mag_max   = float(np.max(mag))   / flow_bound
            angles    = np.arctan2(roi_y, roi_x)
            dir_cons  = float(np.sqrt(np.mean(np.cos(angles)) ** 2 +
                                      np.mean(np.sin(angles)) ** 2))
            h_motion  = float(np.mean(roi_x)) / flow_bound
            v_motion  = float(np.mean(roi_y)) / flow_bound
            energy    = float(np.sum(mag ** 2)) / (n_pix * flow_bound ** 2)
            stat_r    = float(np.sum(mag < 0.5)) / n_pix
            feat = torch.tensor([mag_mean, mag_std, mag_max, dir_cons,
                                  h_motion, v_motion, energy, stat_r],
                                 dtype=torch.float32)
            rows.append(torch.clamp(feat, -10.0, 10.0))
        return torch.stack(rows, dim=0)   # [N, 8]

    # ------------------------------------------------------------------
    # Approach rate
    # ------------------------------------------------------------------

    def _compute_approach_rates_cad(
        self,
        node_feats: torch.Tensor,
        target_pairs_info: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """
        1D approach rate per pair: dot(mv_A - mv_B, unit(pos_B - pos_A)).
        CAD images are NOT panoramic, so no wraparound correction.
        node_feats cols [0,1] = (cx_norm, cy_norm); cols [9,10] = (h_motion, v_motion).
        Returns [P, 1].
        """
        import numpy as np
        P = len(target_pairs_info)
        if P == 0:
            return torch.zeros(0, 1, dtype=torch.float32)
        rates = []
        nf = node_feats.cpu().numpy()
        for A, B, _ in target_pairs_info:
            pos_A  = nf[A, 0:2]
            pos_B  = nf[B, 0:2]
            mv_A   = nf[A, 9:11]
            mv_B   = nf[B, 9:11]
            pos_diff = pos_B - pos_A
            dist = np.linalg.norm(pos_diff)
            unit = pos_diff / dist if dist > 1e-6 else np.zeros(2, dtype=np.float32)
            rate = float(np.dot(mv_A - mv_B, unit))
            rates.append(max(-1.0, min(1.0, rate)))
        return torch.tensor(rates, dtype=torch.float32).unsqueeze(1)   # [P, 1]

    # ------------------------------------------------------------------
    # Extra pair geometry features (CAD-specific)
    # ------------------------------------------------------------------

    def _compute_pair_geometry_extras(
        self,
        node_feats: torch.Tensor,
        target_pairs_info: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """
        Pair axis angle (cos, sin) + optional lateral relative velocity.
        Returns [P, 3] when flow_node_feats=True, else [P, 2].

        cos_angle / sin_angle encode side-by-side (Queuing) vs.
        face-to-face or random arrangement (Talking/Waiting).
        lateral_rate captures perpendicular motion divergence (Crossing).
        node_feats expected cols: 0,1 = cx_norm, cy_norm;
                                  9,10 = h_motion, v_motion (if flow).
        """
        import numpy as np
        P = len(target_pairs_info)
        k = 3 if self.flow_node_feats else 2
        if P == 0:
            return torch.zeros(0, k, dtype=torch.float32)
        nf = node_feats.cpu().numpy()
        rows = []
        for A, B, _ in target_pairs_info:
            dx = float(nf[A, 0]) - float(nf[B, 0])
            dy = float(nf[A, 1]) - float(nf[B, 1])
            d  = float(np.sqrt(dx ** 2 + dy ** 2)) + 1e-6
            cos_a = dx / d
            sin_a = dy / d
            if self.flow_node_feats and nf.shape[1] >= 11:
                mv_A = nf[A, 9:11]   # h_motion, v_motion
                mv_B = nf[B, 9:11]
                perp = np.array([-sin_a, cos_a])
                lat  = float(np.dot(mv_A - mv_B, perp))
                lat  = float(np.clip(lat, -1.0, 1.0))
                rows.append([cos_a, sin_a, lat])
            else:
                rows.append([cos_a, sin_a])
        return torch.tensor(rows, dtype=torch.float32)   # [P, k]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict:
        frame = self.frames[idx]
        pairs_info = frame['target_pairs_info']
        target_pairs = torch.tensor([[p[0], p[1]] for p in pairs_info], dtype=torch.long)
        pair_labels  = torch.tensor([p[2] for p in pairs_info], dtype=torch.long)
        return {
            'node_feats':          frame['node_feats'],
            'person_boxes':        frame['person_boxes'],
            'pre_edge_index':      frame['pre_edge_index'],
            'pre_edge_feats':      frame['pre_edge_feats'],
            'target_pairs':        target_pairs,
            'pair_labels':         pair_labels,
            'pair_flow_feats':     frame['pair_flow_feats'],
            'negative_pairs':      frame['negative_pairs'],
            'num_persons':         frame['num_persons'],
            'num_pairs':           frame['num_pairs'],
            # Test-time metadata
            'seq_num':             frame['seq_num'],
            'frame_id':            frame['frame_id'],
            'track_ids':           frame['track_ids'],
            'social_group_ids':    frame['social_group_ids'],
            'social_activity_ids': frame['social_activity_ids'],
        }

    def get_class_distribution(self) -> Dict:
        counts = Counter(self.all_pair_labels)
        if self.stage == 'stage2' and self.class_merge:
            names = self.CLASS_NAMES_3
        elif self.stage == 'stage2':
            names = self.CLASS_NAMES_6
        else:
            names = ['Interacting']
        return {
            'total_pairs':  len(self.all_pair_labels),
            'total_frames': len(self.frames),
            'class_counts': dict(counts),
            'class_names':  names,
        }


# ============================================================================
# Conversion helper
# ============================================================================

def _x1y1x2y2_to_xywh(x1, y1, x2, y2) -> list:
    """Convert CAD bbox [x1,y1,x2,y2] to [x,y,w,h]."""
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


# ============================================================================
# Collate function
# ============================================================================

def cad_gnn_collate_fn(batch: List) -> List[Dict]:
    """Return list of dicts (variable N per frame — cannot stack)."""
    return [item for item in batch if item is not None]


# ============================================================================
# DataLoader factory
# ============================================================================

def create_cad_gnn_data_loaders(
    config,
    stage: str,
    seqs_train: List[int],
    seqs_val: List[int],
    seqs_test: List[int],
    test_all_negatives: bool = True,
):
    """
    Create train / val / test DataLoaders for the CAD GNN pipeline.

    test_all_negatives: if True, the test dataset uses group_neg_ratio=0
    (keeps all negative pairs) so that the clustering evaluation can score
    every pair in each frame.
    """
    common = dict(
        cad_root=config.cad_root,
        stage=stage,
        image_width=getattr(config, 'image_width', 720),
        image_height=getattr(config, 'image_height', 480),
        class_merge=getattr(config, 'class_merge', False),
        group_neg_ratio=config.group_neg_ratio,
        graph_knn=getattr(config, 'graph_knn', 0),
        inject_flow_to_edges=getattr(config, 'inject_flow_to_edges', True),
        flow_node_feats=getattr(config, 'flow_node_feats', True),
        cache_dir=getattr(config, 'cache_dir', None),
        use_individual_action_feat=getattr(config, 'use_individual_action_feat', False),
        use_extra_pair_feats=getattr(config, 'use_extra_pair_feats', False),
    )

    train_ds = CADGNNGeometricDataset(sequences=seqs_train, **common)
    val_ds   = CADGNNGeometricDataset(sequences=seqs_val,   **common)

    # Test: keep all negatives for clustering / voting evaluation
    test_neg_ratio = 0 if test_all_negatives else config.group_neg_ratio
    test_ds = CADGNNGeometricDataset(
        sequences=seqs_test,
        **{**common, 'group_neg_ratio': test_neg_ratio}
    )

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=config.batch_size, shuffle=shuffle,
            num_workers=getattr(config, 'num_workers', 4),
            collate_fn=cad_gnn_collate_fn, pin_memory=True,
        )

    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False)
