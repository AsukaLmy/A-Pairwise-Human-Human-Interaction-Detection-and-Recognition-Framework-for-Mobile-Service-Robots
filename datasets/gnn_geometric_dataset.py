#!/usr/bin/env python3
"""
GNN Geometric-Only Stage2 Dataset
No visual backbone (no ResNet). Analogous to train_jrdb_stage2_nobackbone.py.

Each __getitem__ returns one frame's complete scene graph:
    node_feats:      [N, 5]   per-person geometric node features (bbox only)
    person_boxes:    [N, 4]   raw boxes (for GAT edge feature computation)
    target_pairs:    [P, 2]   local pair indices
    pair_labels:     [P]      class labels
    pair_flow_feats: [P, 10]  pair-level features matching original nobackbone:
                               9D from GeometricFlowExtractor (geometric distances
                               + Farneback optical flow stats, symmetrically
                               averaged over A→B and B→A) + 1D interaction sync

Node features are purely bbox-derived (no images needed).
Pair flow features require loading prev+curr frame images (same as original
nobackbone MLP pipeline – no ResNet, but does compute optical flow).
"""

import os
import json
import random
import hashlib
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from collections import Counter
from tqdm import tqdm

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from datasets.stage2_dataset import Stage2LabelMapper
from src.features.geometric_features import extract_geometric_features_batch

try:
    from PIL import Image
    from src.features.geometric_flow_extractor import GeometricFlowExtractor
    from src.features.interaction_synchrony import compute_interaction_synchrony
    _FLOW_AVAILABLE = True
except ImportError:
    _FLOW_AVAILABLE = False


class GNNGeometricDataset(Dataset):
    """
    Scene-level dataset for no-backbone GNN training.

    Node features per person (5D, derived from bounding box only):
        [cx_norm, cy_norm, w_norm, h_norm, aspect_ratio]

    Pair flow features per labeled pair (10D, same as original nobackbone MLP):
        Computed by GeometricFlowExtractor over prev+curr frame images.
        Features are symmetrically averaged over A→B and B→A perspectives,
        then appended with interaction synchrony – identical to the 10D vector
        used by the MLP nobackbone classifier.
        Falls back to zeros when images are unavailable.
    """

    TRAINSET = [
        'bytes-cafe-2019-02-07_0',
        'clark-center-2019-02-28_1',
        'cubberly-auditorium-2019-04-22_0',
        'discovery-walk-2019-02-28_1',
        'food-trucks-2019-02-12_0',
        'forbes-cafe-2019-01-22_0',
        'gates-159-group-meeting-2019-04-03_0',
        'gates-to-clark-2019-02-28_1',
        'gates-ai-lab-2019-04-17_0',
        'gates-basement-elevators-2019-01-17_1',
        'hewlett-class-2019-01-23_0',
        'hewlett-packard-intersection-2019-01-24_0',
        'huang-2-2019-01-25_0',
        'huang-2-2019-01-25_1',
        'huang-basement-2019-01-25_0',
        'huang-intersection-2019-01-22_0',
        'indoor-coupa-cafe-2019-02-06_0',
        'lomita-serra-intersection-2019-01-30_0',
        'memorial-court-2019-03-16_0',
        'meyer-green-2019-03-16_1',
        'nvidia-aud-2019-04-18_0',
        'nvidia-aud-2019-01-25_0',
        'outdoor-coupa-cafe-2019-02-06_0',
        'quarry-road-2019-02-28_0',
        'serra-street-2019-01-30_0',
        'stlc-111-2019-04-19_0',
        'packard-poster-session-2019-03-20_2',
        'svl-meeting-gates-2-2019-04-08_1',
        'tressider-2019-03-16_2',
        'tressider-2019-04-26_1',
        'tressider-2019-04-26_2',
    ]

    VALSET = [
        'clark-center-2019-02-28_0',
        'discovery-walk-2019-02-28_0',
        'gates-ai-lab-2019-02-08_0',
        'gates-foyer-2019-01-17_0',
        'hewlett-class-2019-01-23_1',
        'jordan-hall-2019-04-22_0',
        'nvidia-aud-2019-04-18_1',
        'packard-poster-session-2019-03-20_1',
        'stlc-111-2019-04-19_1',
        'svl-meeting-gates-2-2019-04-08_0',
        'tressider-2019-03-16_1',
        'tressider-2019-04-26_3',
    ]

    TESTSET = [
        'clark-center-intersection-2019-02-28_0',
        'cubberly-auditorium-2019-04-22_1',
        'gates-basement-elevators-2019-01-17_0',
        'gates-to-clark-2019-02-28_0',
        'meyer-green-2019-03-16_0',
        'nvidia-aud-2019-04-18_2',
        'packard-poster-session-2019-03-20_0',
        'stlc-111-2019-04-19_2',
        'tressider-2019-03-16_0',
        'tressider-2019-04-26_0',
    ]

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        frame_interval: int = 1,
        filter_occlusion: bool = True,
        filter_edge_cases: bool = True,
        edge_threshold: int = 200,
        image_width: int = 3760,
        image_height: int = 480,
        group_neg_ratio: float = 2.0,       # max negatives = neg_ratio × n_positives
        cache_dir: Optional[str] = None,    # 预计算特征缓存目录，None=不缓存
        graph_knn: int = 0,                 # 0=全连接，>0=K-NN 稀疏图
        inject_flow_to_edges: bool = True,  # 将 10D 流特征注入边初始化（→17D 边）
        flow_node_feats: bool = True,       # 每节点追加 8D 光流统计（5D→13D 节点）
    ):
        self.data_path = data_path
        self.split = split
        self.frame_interval = frame_interval
        self.filter_occlusion = filter_occlusion
        self.filter_edge_cases = filter_edge_cases
        self.edge_threshold = edge_threshold
        self.image_width    = image_width
        self.image_height   = image_height
        self.group_neg_ratio = group_neg_ratio
        self.cache_dir      = cache_dir
        self.graph_knn      = graph_knn
        self.inject_flow_to_edges = inject_flow_to_edges
        self.flow_node_feats = flow_node_feats

        self.label_mapper = Stage2LabelMapper()

        # Image directory (same layout as resnet_stage2_dataset)
        self.images_dir = os.path.join(data_path, 'images', 'image_stitched')
        self._images_available = os.path.exists(self.images_dir) and _FLOW_AVAILABLE

        if self._images_available:
            self.flow_extractor = GeometricFlowExtractor(
                flow_bound=20.0, cache_enabled=True)
            print(f"GNNGeometricDataset: image dir found – "
                  f"will pre-compute 10D pair flow features")
        else:
            self.flow_extractor = None
            print(f"GNNGeometricDataset: image dir not found or PIL/cv2 unavailable – "
                  f"pair_flow_feats will be zeros [P, 10]")

        self.frames: List[Dict] = []
        self.all_pair_labels: List[int] = []
        self._load_data()

        total_pairs = sum(f['num_pairs'] for f in self.frames)
        flow_ok = sum(1 for f in self.frames if f.get('has_flow', False))
        print(f"GNN Geometric Dataset ({split}): "
              f"{len(self.frames)} frames, {total_pairs} target pairs "
              f"({flow_ok} frames with optical flow features)")
        if self.all_pair_labels:
            print(f"  Label distribution: {dict(Counter(self.all_pair_labels))}")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cache_path(self) -> Optional[str]:
        """
        返回当前 split 对应的缓存文件路径。
        子文件夹按影响预计算输出的数据参数命名，不同 seed 共享同一缓存。
        格式：<cache_dir>/fi<fi>_occ<0/1>_edge<0/1>_et<et>_<datahash6>/<split>.pt
        """
        if self.cache_dir is None:
            return None
        data_hash = hashlib.md5(
            os.path.abspath(self.data_path).encode()
        ).hexdigest()[:6]
        folder_name = (
            f"fi{self.frame_interval}"
            f"_occ{int(self.filter_occlusion)}"
            f"_edge{int(self.filter_edge_cases)}"
            f"_et{self.edge_threshold}"
            f"_knn{self.graph_knn}"
            f"_fl{int(self.inject_flow_to_edges)}"
            f"_nfl{int(self.flow_node_feats)}"
            f"_{data_hash}"
        )
        return os.path.join(self.cache_dir, folder_name, f"{self.split}.pt")

    def _resample_negative_pairs(self):
        """
        用当前 random 状态为所有帧重采样负样本对。
        从缓存加载帧数据后调用，以支持不同 seed 产生不同的负样本集合。
        """
        for frame in self.frames:
            N = frame['num_persons']
            pos_set = {(A, B) for A, B, _ in frame['target_pairs_info']}
            all_undirected = [
                (i, j) for i in range(N)
                for j in range(i + 1, N)
                if (i, j) not in pos_set
            ]
            max_neg = max(1, int(self.group_neg_ratio * len(frame['target_pairs_info'])))
            if len(all_undirected) > max_neg:
                sampled = random.sample(all_undirected, max_neg)
            else:
                sampled = all_undirected
            frame['negative_pairs'] = (
                torch.tensor(sampled, dtype=torch.long)
                if sampled else torch.zeros(0, 2, dtype=torch.long)
            )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        split_map = {'train': self.TRAINSET, 'val': self.VALSET, 'test': self.TESTSET}
        if self.split not in split_map:
            raise ValueError(f"Unknown split: {self.split}")
        scene_list = split_map[self.split]

        # ---- 缓存加载（命中则跳过全部特征计算） ----
        cache_path = self._get_cache_path()
        if cache_path and os.path.exists(cache_path):
            print(f"[Cache] 从缓存加载 {cache_path}，重采样负样本对…")
            self.frames = torch.load(cache_path, weights_only=False)
            self._resample_negative_pairs()
            self.all_pair_labels = [
                lbl for f in self.frames for _, _, lbl in f['target_pairs_info']
            ]
            return

        labels_dir = os.path.join(
            self.data_path, 'labels', 'labels_2d_activity_social_stitched'
        )
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

        available = {f for f in os.listdir(labels_dir) if f.endswith('.json')}
        selected = [s for s in scene_list if f"{s}.json" in available]
        missing  = [s for s in scene_list if f"{s}.json" not in available]
        if missing:
            print(f"Warning: {len(missing)} scene(s) missing")
        print(f"Loading {len(selected)}/{len(scene_list)} scenes for '{self.split}'")

        filtered_occ  = 0
        filtered_edge = 0

        for scene_name in tqdm(selected, desc=f"[{self.split}] scenes", unit="scene"):
            scene_path = os.path.join(labels_dir, f"{scene_name}.json")
            try:
                with open(scene_path, 'r') as f:
                    scene_data = json.load(f)
            except Exception as e:
                print(f"Warning: {scene_name}: {e}")
                continue

            frame_names = sorted(scene_data.get('labels', {}).keys())
            # Start at index 1 so every frame has a prev frame for optical flow
            sampled_indices = list(range(1, len(frame_names), self.frame_interval))

            for idx in sampled_indices:
                image_name      = frame_names[idx]
                prev_image_name = frame_names[idx - 1]   # immediately preceding frame
                annotations = scene_data['labels'][image_name]

                # ---- Collect all valid persons in this frame ----
                person_dict: Dict[int, Dict] = {}
                for ann in annotations:
                    lid = ann.get('label_id', '')
                    if not lid.startswith('pedestrian:'):
                        continue
                    pid = int(lid.split(':')[1])
                    box = ann.get('box', [0, 0, 100, 100])
                    if not self._valid_box(box):
                        continue
                    occ = ann.get('attributes', {}).get('occlusion', 'unknown')
                    person_dict[pid] = {'box': box, 'occlusion': occ}

                if not person_dict:
                    continue

                sorted_pids   = sorted(person_dict.keys())
                pid_to_local  = {pid: i for i, pid in enumerate(sorted_pids)}

                # ---- Collect labeled pairs ----
                target_pairs_info: List[Tuple[int, int, int]] = []
                pair_boxes: List[Tuple[list, list]] = []   # (box_A, box_B) per pair

                for ann in annotations:
                    lid = ann.get('label_id', '')
                    if not lid.startswith('pedestrian:'):
                        continue
                    pid_A = int(lid.split(':')[1])
                    if pid_A not in person_dict:
                        continue
                    box_A = person_dict[pid_A]['box']
                    occ_A = person_dict[pid_A]['occlusion']

                    for iact in (ann.get('H-interaction', []) or ann.get('HHI', [])):
                        pair_id = iact.get('pair', '')
                        if not pair_id.startswith('pedestrian:'):
                            continue
                        pid_B = int(pair_id.split(':')[1])
                        if pid_B not in person_dict or pid_A >= pid_B:
                            continue

                        box_B = person_dict[pid_B]['box']
                        occ_B = person_dict[pid_B]['occlusion']

                        il = iact.get('inter_labels', {})
                        if not isinstance(il, dict) or not il:
                            continue
                        lbl = self.label_mapper.map_label(list(il.keys())[0])
                        if lbl is None:
                            continue

                        if self.filter_occlusion:
                            if not (self._valid_occ(occ_A) and self._valid_occ(occ_B)):
                                filtered_occ += 1
                                continue
                        if self.filter_edge_cases:
                            if self._edge_box(box_A) or self._edge_box(box_B):
                                filtered_edge += 1
                                continue

                        target_pairs_info.append(
                            (pid_to_local[pid_A], pid_to_local[pid_B], lbl)
                        )
                        pair_boxes.append((box_A, box_B))

                if not target_pairs_info:
                    continue

                # ---- Pre-compute node features (bbox only, no images) ----
                all_boxes = [person_dict[pid]['box'] for pid in sorted_pids]
                node_feats = self._compute_node_feats(all_boxes)

                # ---- Pre-compute edge index + 7D edge features ----
                # Uses K-NN sparsification when graph_knn > 0; target pair edges
                # are always force-included to guarantee flow feature extraction.
                N_all = len(sorted_pids)
                boxes_t = torch.tensor(all_boxes, dtype=torch.float32)  # [N, 4]
                if N_all > 1:
                    if self.graph_knn > 0 and N_all > self.graph_knn + 1:
                        # K-NN edges by normalised centre-point distance
                        positions = node_feats[:, 0:2]  # [N, 2] (cx_norm, cy_norm)
                        dists = torch.cdist(positions, positions)   # [N, N]
                        dists.fill_diagonal_(float('inf'))
                        K = min(self.graph_knn, N_all - 1)
                        _, knn_idx = dists.topk(K, dim=1, largest=False)  # [N, K]
                        edge_set: set = set()
                        src_list: List[int] = []
                        dst_list: List[int] = []
                        for i in range(N_all):
                            for j in knn_idx[i].tolist():
                                for s, d in [(i, j), (j, i)]:
                                    if (s, d) not in edge_set:
                                        src_list.append(s)
                                        dst_list.append(d)
                                        edge_set.add((s, d))
                        # Force-include directed edges for every target pair
                        for A, B, _ in target_pairs_info:
                            for s, d in [(A, B), (B, A)]:
                                if (s, d) not in edge_set:
                                    src_list.append(s)
                                    dst_list.append(d)
                                    edge_set.add((s, d))
                    else:
                        # Fully-connected (original behaviour)
                        src_list = [i for i in range(N_all) for j in range(N_all) if i != j]
                        dst_list = [j for i in range(N_all) for j in range(N_all) if i != j]

                    src_t = torch.tensor(src_list, dtype=torch.long)
                    dst_t = torch.tensor(dst_list, dtype=torch.long)
                    pre_edge_index = torch.stack([src_t, dst_t], dim=0)       # [2, E]
                    pre_edge_feats = extract_geometric_features_batch(
                        boxes_t[src_t], boxes_t[dst_t],
                        self.image_width, self.image_height,
                    )                                                          # [E, 7]
                else:
                    src_list = []
                    dst_list = []
                    pre_edge_index = torch.zeros(2, 0, dtype=torch.long)
                    pre_edge_feats = torch.zeros(0, 7, dtype=torch.float32)

                # ---- Open images once (shared by pair flow + node flow) ----
                curr_image_path = os.path.join(
                    self.images_dir, scene_name, image_name)
                prev_image_path = os.path.join(
                    self.images_dir, scene_name, prev_image_name)

                prev_img_f, curr_img_f, has_images = self._open_images(
                    prev_image_path, curr_image_path)

                # ---- Pre-compute 10D pair flow features ----
                pair_flow_feats, has_flow = self._compute_all_pair_flow_feats_from_images(
                    prev_img_f, curr_img_f, pair_boxes
                )

                # ---- Per-node 8D optical flow features (5D → 13D node) ----
                if self.flow_node_feats:
                    node_flow = self._compute_node_flow_feats(
                        prev_img_f, curr_img_f, all_boxes)
                    node_feats = torch.cat([node_feats, node_flow], dim=1)  # [N, 13]

                # ---- Approach rate (10D → 11D pair feature) ----
                if self.flow_node_feats:
                    approach_t = self._compute_approach_rates(node_feats, target_pairs_info)
                    pair_flow_feats = torch.cat([pair_flow_feats, approach_t], dim=1)  # [P, 11]

                # ---- Close images ----
                if has_images:
                    if prev_img_f is not None:
                        prev_img_f.close()
                    if curr_img_f is not None:
                        curr_img_f.close()

                # ---- Flow injection into edge features ----
                # Append 10D flow vector to each edge: target pair edges get the
                # actual (symmetrised) flow; all other edges get zero-padding.
                # Result: pre_edge_feats → [E, 17] when inject_flow_to_edges=True.
                if self.inject_flow_to_edges and pre_edge_feats.size(0) > 0:
                    # Only inject the base 10D geometric flow features (not approach_rate).
                    # approach_rate is a pair-level feature appended after this block.
                    zero_flow = torch.zeros(10, dtype=torch.float32)
                    # Build lookup: (i,j) → flow tensor [10] (symmetric)
                    flow_map: dict = {}
                    for k, (A, B, _) in enumerate(target_pairs_info):
                        fv = pair_flow_feats[k, :10]   # [10] — base flow only
                        flow_map[(A, B)] = fv
                        flow_map[(B, A)] = fv     # symmetric
                    flow_rows = [
                        flow_map.get((src_list[e], dst_list[e]), zero_flow)
                        for e in range(len(src_list))
                    ]
                    flow_part = torch.stack(flow_rows, dim=0)            # [E, 10]
                    pre_edge_feats = torch.cat(
                        [pre_edge_feats, flow_part], dim=1)              # [E, 17]

                # ---- Negative pair sampling (for grouping auxiliary task) ----
                # Positive pairs: labeled interacting pairs (group_label = 1)
                # Negative pairs: unlabeled pairs in same scene (group_label = 0)
                N_persons = len(sorted_pids)
                pos_set = {(A, B) for A, B, _ in target_pairs_info}
                # Undirected: only (i,j) with i<j to avoid duplicates
                all_undirected = [
                    (i, j) for i in range(N_persons)
                    for j in range(i + 1, N_persons)
                    if (i, j) not in pos_set
                ]
                max_neg = max(1, int(self.group_neg_ratio * len(target_pairs_info)))
                if len(all_undirected) > max_neg:
                    sampled_neg = random.sample(all_undirected, max_neg)
                else:
                    sampled_neg = all_undirected
                negative_pairs = torch.tensor(sampled_neg, dtype=torch.long) \
                    if sampled_neg else torch.zeros(0, 2, dtype=torch.long)

                frame_id = f"{scene_name}_{os.path.splitext(image_name)[0]}"
                self.frames.append({
                    'node_feats':        node_feats,           # [N, 5]
                    'person_boxes':      torch.tensor(
                        all_boxes, dtype=torch.float32),       # [N, 4]
                    'pre_edge_index':    pre_edge_index,       # [2, E]  precomputed
                    'pre_edge_feats':    pre_edge_feats,       # [E, 7]  precomputed
                    'target_pairs_info': target_pairs_info,    # List[(A,B,lbl)]
                    'pair_flow_feats':   pair_flow_feats,      # [P, 10]
                    'negative_pairs':    negative_pairs,       # [Q, 2]
                    'has_flow':          has_flow,
                    'num_persons':       N_persons,
                    'num_pairs':         len(target_pairs_info),
                    'scene_name':        scene_name,
                    'frame_id':          frame_id,
                })
                self.all_pair_labels.extend(p[2] for p in target_pairs_info)

        if filtered_occ + filtered_edge > 0:
            print(f"  Filtered: {filtered_occ} occlusion + "
                  f"{filtered_edge} edge-case pairs")

        # ---- 缓存保存（保存种子无关的特征，不含 negative_pairs） ----
        if cache_path:
            frames_to_cache = [
                {k: v for k, v in f.items() if k != 'negative_pairs'}
                for f in self.frames
            ]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(frames_to_cache, cache_path)
            print(f"[Cache] 已保存至 {cache_path}")

    # ------------------------------------------------------------------
    # Pair flow feature computation (10D, same as original nobackbone MLP)
    # ------------------------------------------------------------------

    def _compute_all_pair_flow_feats(
        self,
        prev_path: str,
        curr_path: str,
        pair_boxes: List[Tuple[list, list]],
    ) -> Tuple[torch.Tensor, bool]:
        """
        Pre-compute 10D pair features for all labeled pairs in one frame.

        Returns:
            pair_flow_feats: [P, 10]
            has_flow: bool – True if optical flow was actually computed
        """
        P = len(pair_boxes)
        zero_feats = torch.zeros(P, 10, dtype=torch.float32)

        if not self._images_available or P == 0:
            return zero_feats, False

        if not (os.path.exists(prev_path) and os.path.exists(curr_path)):
            return zero_feats, False

        try:
            prev_img = Image.open(prev_path).convert('RGB')
            curr_img = Image.open(curr_path).convert('RGB')
        except Exception:
            return zero_feats, False

        try:
            rows = []
            for box_A, box_B in pair_boxes:
                feat = self._compute_pair_flow_feat(prev_img, curr_img, box_A, box_B)
                rows.append(feat)
            return torch.stack(rows, dim=0), True   # [P, 10]
        finally:
            prev_img.close()
            curr_img.close()

    # ------------------------------------------------------------------
    # Image-sharing helpers (avoid double I/O for node + pair flow)
    # ------------------------------------------------------------------

    def _open_images(self, prev_path: str, curr_path: str):
        """
        Open prev/curr PIL images. Returns (prev_img, curr_img, success).
        Returns (None, None, False) on any failure.
        """
        if not self._images_available:
            return None, None, False
        if not (os.path.exists(prev_path) and os.path.exists(curr_path)):
            return None, None, False
        try:
            prev_img = Image.open(prev_path).convert('RGB')
            curr_img = Image.open(curr_path).convert('RGB')
            return prev_img, curr_img, True
        except Exception:
            return None, None, False

    def _compute_all_pair_flow_feats_from_images(
        self,
        prev_img,   # PIL.Image or None
        curr_img,   # PIL.Image or None
        pair_boxes: List[Tuple[list, list]],
    ) -> Tuple[torch.Tensor, bool]:
        """
        Compute 10D pair features from already-loaded PIL images.
        Avoids re-opening images when they were already loaded for node flow.
        """
        P = len(pair_boxes)
        zero_feats = torch.zeros(P, 10, dtype=torch.float32)
        if not self._images_available or P == 0 or prev_img is None or curr_img is None:
            return zero_feats, False
        try:
            rows = [self._compute_pair_flow_feat(prev_img, curr_img, bA, bB)
                    for bA, bB in pair_boxes]
            return torch.stack(rows, dim=0), True
        except Exception:
            return zero_feats, False

    def _compute_node_flow_feats(
        self,
        prev_img,               # PIL.Image or None
        curr_img,               # PIL.Image or None
        all_boxes: List[List[float]],
    ) -> torch.Tensor:
        """
        Compute 8D per-person optical flow statistics for all N persons.

        One Farneback call covers the full frame; all N persons share it.
        Farneback params identical to GeometricFlowExtractor to keep scales consistent.

        Returns [N, 8]:
            [0] magnitude_mean      (/ flow_bound=20.0)
            [1] magnitude_std       (/ flow_bound)
            [2] magnitude_max       (/ flow_bound)
            [3] direction_consistency  [0, 1]  (circular-variance R)
            [4] horizontal_motion   (signed mean flow_x / flow_bound)
            [5] vertical_motion     (signed mean flow_y / flow_bound)
            [6] motion_energy       sum(mag²) / (N_pixels × flow_bound²)
            [7] stationary_ratio    fraction of pixels with |flow| < 0.5 px
        """
        import cv2
        import numpy as np

        N = len(all_boxes)
        zero_feats = torch.zeros(N, 8, dtype=torch.float32)

        if not self._images_available or N == 0 or prev_img is None or curr_img is None:
            return zero_feats

        try:
            prev_gray = np.array(prev_img.convert('L'))
            curr_gray = np.array(curr_img.convert('L'))
        except Exception:
            return zero_feats

        # Identical Farneback params to GeometricFlowExtractor
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )   # [H, W, 2]
        except Exception:
            return zero_feats

        H, W = flow.shape[:2]
        flow_bound = 20.0
        rows = []

        for box in all_boxes:
            x, y, w, h = box
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(W, int(x + w))
            y2 = min(H, int(y + h))

            if x2 <= x1 or y2 <= y1:
                rows.append(torch.zeros(8, dtype=torch.float32))
                continue

            roi_x = flow[y1:y2, x1:x2, 0]   # [roi_H, roi_W]  signed
            roi_y = flow[y1:y2, x1:x2, 1]
            n_pix = roi_x.size

            if n_pix == 0:
                rows.append(torch.zeros(8, dtype=torch.float32))
                continue

            # Magnitude: use |flow_x| for consistency with GeometricFlowExtractor
            roi_x_abs = np.abs(roi_x)
            mag = np.sqrt(roi_x_abs ** 2 + roi_y ** 2)

            mag_mean  = float(np.mean(mag))  / flow_bound
            mag_std   = float(np.std(mag))   / flow_bound
            mag_max   = float(np.max(mag))   / flow_bound

            # Direction consistency (circular variance R)
            angles     = np.arctan2(roi_y, roi_x)
            mean_cos   = float(np.mean(np.cos(angles)))
            mean_sin   = float(np.mean(np.sin(angles)))
            dir_cons   = float(np.sqrt(mean_cos ** 2 + mean_sin ** 2))

            # Signed mean motion (direction awareness)
            h_motion   = float(np.mean(roi_x))  / flow_bound
            v_motion   = float(np.mean(roi_y))  / flow_bound

            # Motion energy and stationary ratio
            energy     = float(np.sum(mag ** 2)) / (n_pix * flow_bound ** 2)
            stat_ratio = float(np.sum(mag < 0.5)) / n_pix

            feat = torch.tensor(
                [mag_mean, mag_std, mag_max, dir_cons,
                 h_motion, v_motion, energy, stat_ratio],
                dtype=torch.float32,
            )
            rows.append(torch.clamp(feat, -10.0, 10.0))

        return torch.stack(rows, dim=0)   # [N, 8]

    def _compute_approach_rates(
        self,
        node_feats: torch.Tensor,          # [N, 13]  (5D static + 8D flow)
        target_pairs_info: List,           # [(A, B, lbl), ...]
    ) -> torch.Tensor:
        """
        Compute 1D approach-rate for each target pair.

        approach_rate = dot(mv_A - mv_B,  unit(pos_B - pos_A))
          >0: converging  <0: diverging  ~0: parallel motion or both stationary

        Uses node_feats cols [0,1] = (cx_norm, cy_norm) for position,
              node_feats cols [9,10] = (h_motion, v_motion) for motion vector.
        Handles panoramic boundary wraparound on horizontal axis.
        Returns [P, 1] float32 tensor, clamped to [-1, 1].
        """
        import numpy as np
        P = len(target_pairs_info)
        if P == 0:
            return torch.zeros(0, 1, dtype=torch.float32)

        rates = []
        nf = node_feats.cpu().numpy()
        for A, B, _ in target_pairs_info:
            pos_A = nf[A, 0:2]   # [cx_norm, cy_norm]
            pos_B = nf[B, 0:2]
            mv_A  = nf[A, 9:11]  # [h_motion, v_motion] — cols 5+4, 5+5
            mv_B  = nf[B, 9:11]

            dx = pos_B[0] - pos_A[0]
            # Panoramic wraparound correction
            if abs(dx) > 0.5:
                dx -= np.sign(dx)
            pos_diff = np.array([dx, pos_B[1] - pos_A[1]], dtype=np.float32)
            dist = np.linalg.norm(pos_diff)
            unit_vec = pos_diff / dist if dist > 1e-6 else np.zeros(2, dtype=np.float32)

            rate = float(np.dot(mv_A - mv_B, unit_vec))
            rates.append(max(-1.0, min(1.0, rate)))

        return torch.tensor(rates, dtype=torch.float32).unsqueeze(1)  # [P, 1]

    def _compute_pair_flow_feat(
        self,
        prev_img,
        curr_img,
        box_A: list,
        box_B: list,
    ) -> torch.Tensor:
        """
        Compute one 10D pair feature vector.

        Matches the feature construction in train_jrdb_stage2_nobackbone.py:
          1. Extract 9D from GeometricFlowExtractor (A→B perspective)
          2. Extract 9D from GeometricFlowExtractor (B→A perspective)
          3. Symmetric averaging:
             - Symmetric features (f0,f1,f5,f6,f7): keep A→B value
             - Asymmetric features (f2,f3,f4,f8): average A→B and B→A
          4. Append interaction synchrony (1D) → total 10D

        Returns:
            Tensor [10], zeros on error.
        """
        try:
            feat_a = self.flow_extractor.extract_geometric_features(
                prev_img, curr_img, box_A, box_B)   # [9]
            feat_b = self.flow_extractor.extract_geometric_features(
                prev_img, curr_img, box_B, box_A)   # [9]

            # Symmetric 9D vector (indices match GeometricFlowExtractor output order)
            sym = torch.tensor([
                feat_a[0],                          # [0] distance/avg_height  (symm)
                feat_a[1],                          # [1] distance/avg_width   (symm)
                (feat_a[2] + feat_b[2]) * 0.5,     # [2] flow_mean/area       (avg)
                (feat_a[3] + feat_b[3]) * 0.5,     # [3] flow_std/area        (avg)
                (feat_a[4] + feat_b[4]) * 0.5,     # [4] vertical_dominance   (avg)
                feat_a[5],                          # [5] avg_aspect_ratio     (symm)
                feat_a[6],                          # [6] avg_height/img_h     (symm)
                feat_a[7],                          # [7] avg_bottom/img_h     (symm)
                (feat_a[8] + feat_b[8]) * 0.5,     # [8] direction_consistency(avg)
            ], dtype=torch.float32)

            # Interaction synchrony – passes raw (asymmetric) A and B features
            sync = compute_interaction_synchrony(
                feat_a.unsqueeze(0), feat_b.unsqueeze(0))   # scalar or Tensor[1]
            sync_val = float(sync.item() if hasattr(sync, 'item') else sync)

            return torch.cat([sym, torch.tensor([sync_val], dtype=torch.float32)])
        except Exception:
            return torch.zeros(10, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Node feature computation (pure geometry, no images)
    # ------------------------------------------------------------------

    def _compute_node_feats(self, boxes: List[List[float]]) -> torch.Tensor:
        """
        Compute 5D geometric node features for each person.

        Features: [cx_norm, cy_norm, w_norm, h_norm, aspect_ratio]
        All values in [0, ~1] range for stable training.
        """
        W, H = float(self.image_width), float(self.image_height)
        t = torch.tensor(boxes, dtype=torch.float32)     # [N, 4]
        x, y, w, h = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        return torch.stack([
            (x + w * 0.5) / W,   # cx_norm
            (y + h * 0.5) / H,   # cy_norm
            w / W,               # w_norm
            h / H,               # h_norm
            h / (w + 1e-6),      # aspect_ratio
        ], dim=1)                                        # [N, 5]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _valid_box(self, box) -> bool:
        if len(box) != 4:
            return False
        x, y, w, h = box
        return w > 0 and h > 0 and x >= 0 and y >= 0 and w <= 5000 and h <= 5000

    def _valid_occ(self, occ: str) -> bool:
        return occ in ['Fully_visible', 'Mostly_visible']

    def _edge_box(self, box) -> bool:
        x, y, w, h = box
        return x < self.edge_threshold or (x + w) > (self.image_width - self.edge_threshold)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict:
        frame = self.frames[idx]
        pairs_info = frame['target_pairs_info']
        target_pairs = torch.tensor(
            [[p[0], p[1]] for p in pairs_info], dtype=torch.long)
        pair_labels = torch.tensor(
            [p[2] for p in pairs_info], dtype=torch.long)

        return {
            'node_feats':      frame['node_feats'],       # [N, 5]  – bbox geometry
            'person_boxes':    frame['person_boxes'],     # [N, 4]
            'pre_edge_index':  frame['pre_edge_index'],   # [2, E]  – precomputed
            'pre_edge_feats':  frame['pre_edge_feats'],   # [E, 7]  – precomputed
            'target_pairs':    target_pairs,              # [P, 2]
            'pair_labels':     pair_labels,               # [P]
            'pair_flow_feats': frame['pair_flow_feats'],  # [P, 10] – optical flow pair feats
            'negative_pairs':  frame['negative_pairs'],   # [Q, 2]  – unlabeled (group=0)
            'num_persons':     frame['num_persons'],
            'num_pairs':       frame['num_pairs'],
            'scene_name':      frame['scene_name'],
            'frame_id':        frame['frame_id'],
        }

    def get_labels(self) -> List[int]:
        return list(self.all_pair_labels)

    def get_class_distribution(self) -> Dict:
        counts = Counter(self.all_pair_labels)
        return {
            'total_pairs':  len(self.all_pair_labels),
            'total_frames': len(self.frames),
            'class_counts': dict(counts),
            'class_names':  self.label_mapper.class_names,
        }


# ============================================================================
# Collate & DataLoader factory
# ============================================================================

def gnn_geometric_collate_fn(batch: List) -> List[Dict]:
    """Return list of dicts (N_i differs per frame – cannot stack)."""
    return [item for item in batch if item is not None]


def create_gnn_geometric_data_loaders(config) -> Tuple:
    from torch.utils.data import DataLoader

    kwargs = dict(
        data_path=config.data_path,
        frame_interval=config.frame_interval,
        filter_occlusion=config.filter_occlusion,
        filter_edge_cases=config.filter_edge_cases,
        edge_threshold=config.edge_threshold,
        image_width=config.image_width,
        image_height=config.image_height,
        group_neg_ratio=config.group_neg_ratio,
        cache_dir=getattr(config, 'cache_dir', None),
        graph_knn=getattr(config, 'graph_knn', 0),
        inject_flow_to_edges=getattr(config, 'inject_flow_to_edges', True),
        flow_node_feats=getattr(config, 'flow_node_feats', True),
    )

    train_ds = GNNGeometricDataset(split='train', **kwargs)
    val_ds   = GNNGeometricDataset(split='val',   **kwargs)
    test_ds  = GNNGeometricDataset(split='test',  **kwargs)

    loader_kw = dict(
        collate_fn=gnn_geometric_collate_fn,
        num_workers=config.num_workers,
        pin_memory=(config.num_workers > 0),
        persistent_workers=(config.num_workers > 0),
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size,
                              shuffle=False, drop_last=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size,
                              shuffle=False, drop_last=False, **loader_kw)

    print(f"\nGNN Geometric DataLoaders: "
          f"train={len(train_ds)} frames / "
          f"val={len(val_ds)} frames / "
          f"test={len(test_ds)} frames")
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../../dataset"
    ds = GNNGeometricDataset(data_path=data_path, split='train', frame_interval=30)
    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        s = ds[0]
        print(f"node_feats:      {s['node_feats'].shape}")
        print(f"person_boxes:    {s['person_boxes'].shape}")
        print(f"target_pairs:    {s['target_pairs'].shape}")
        print(f"pair_labels:     {s['pair_labels'].shape}")
        print(f"pair_flow_feats: {s['pair_flow_feats'].shape}")
        print(f"pair_flow_feats[0]: {s['pair_flow_feats'][0]}")
