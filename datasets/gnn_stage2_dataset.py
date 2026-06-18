#!/usr/bin/env python3
"""
GNN-based Stage2 Dataset
Scene-level dataset for Graph Attention Network interaction classification.

Key difference from ResNetStage2Dataset:
  OLD: Each sample = 1 interaction pair  → len = total pairs (~tens of thousands)
  NEW: Each sample = 1 frame (scene)     → len = frames with target pairs (~thousands)

Each scene sample contains ALL persons in the frame as graph nodes,
plus the labeled interaction pairs as classification targets.
"""

import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from tqdm import tqdm

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from datasets.stage2_dataset import Stage2LabelMapper
from src.features.geometric_features import extract_geometric_features_batch


# ============================================================================
# Scene-level Dataset
# ============================================================================

class GNNStage2Dataset(Dataset):
    """
    Scene-level dataset for GNN-based Stage2 behavior classification.

    Each __getitem__ returns one frame containing:
      - person_crops:   [N, 3, H, W]  cropped images for all N persons
      - person_boxes:   [N, 4]        bounding boxes [x, y, w, h]
      - target_pairs:   [P, 2]        local (within-frame) index pairs (A_idx, B_idx)
      - pair_labels:    [P]           interaction class labels (0/1/2)
    """

    # Scene splits (identical to ResNetStage2Dataset for fair comparison)
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
        crop_size: int = 112,
        frame_interval: int = 1,
        filter_occlusion: bool = True,
        filter_edge_cases: bool = True,
        edge_threshold: int = 200,
        image_width: int = 3760,
        image_height: int = 480,
    ):
        self.data_path = data_path
        self.split = split
        self.crop_size = crop_size
        self.frame_interval = frame_interval
        self.filter_occlusion = filter_occlusion
        self.filter_edge_cases = filter_edge_cases
        self.edge_threshold = edge_threshold
        self.image_width = image_width
        self.image_height = image_height

        self.label_mapper = Stage2LabelMapper()

        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # self.frames: List[Dict] – one entry per frame with target pairs
        self.frames = []
        self.all_pair_labels = []   # flat list of labels (for class distribution)
        self._load_data()

        print(f"GNN Stage2 Dataset created:")
        print(f"  Split: {self.split}")
        print(f"  Frames with target pairs: {len(self.frames)}")
        total_pairs = sum(f['num_pairs'] for f in self.frames)
        print(f"  Total target pairs: {total_pairs}")
        if self.all_pair_labels:
            print(f"  Label distribution: {dict(Counter(self.all_pair_labels))}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        split_map = {'train': self.TRAINSET, 'val': self.VALSET, 'test': self.TESTSET}
        if self.split not in split_map:
            raise ValueError(f"Unknown split: {self.split}")
        scene_list = split_map[self.split]

        social_labels_dir = os.path.join(
            self.data_path, 'labels', 'labels_2d_activity_social_stitched'
        )
        images_dir = os.path.join(self.data_path, 'images', 'image_stitched')

        if not os.path.exists(social_labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {social_labels_dir}")

        available = {f for f in os.listdir(social_labels_dir) if f.endswith('.json')}
        selected = [s for s in scene_list if f"{s}.json" in available]
        missing = [s for s in scene_list if f"{s}.json" not in available]
        if missing:
            print(f"Warning: {len(missing)} scene(s) not found: {missing[:3]}...")
        print(f"Loading {len(selected)}/{len(scene_list)} scenes for '{self.split}' split")

        filtered_occlusion = 0
        filtered_edge = 0

        for scene_name in tqdm(selected, desc=f"[{self.split}] scenes", unit="scene"):
            scene_path = os.path.join(social_labels_dir, f"{scene_name}.json")
            try:
                with open(scene_path, 'r') as f:
                    scene_data = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {scene_name}: {e}")
                continue

            frame_names = sorted(scene_data.get('labels', {}).keys())
            # Start at index 1 so every frame has a potential previous frame
            sampled_indices = list(range(1, len(frame_names), self.frame_interval))

            for idx in sampled_indices:
                image_name = frame_names[idx]
                annotations = scene_data['labels'][image_name]
                image_path = os.path.join(images_dir, scene_name, image_name)

                # ------ Collect all persons in this frame ------
                person_dict = {}   # pid -> {box, occlusion}
                for ann in annotations:
                    label_id = ann.get('label_id', '')
                    if not label_id.startswith('pedestrian:'):
                        continue
                    pid = int(label_id.split(':')[1])
                    box = ann.get('box', [0, 0, 100, 100])
                    if not self._is_valid_box(box):
                        continue
                    occ = ann.get('attributes', {}).get('occlusion', 'unknown')
                    person_dict[pid] = {
                        'box': box,
                        'occlusion': occ,
                    }

                if not person_dict:
                    continue

                # Assign stable local indices (sorted by pid for reproducibility)
                sorted_pids = sorted(person_dict.keys())
                pid_to_local = {pid: i for i, pid in enumerate(sorted_pids)}

                # ------ Collect labeled interaction pairs ------
                target_pairs_info = []   # (local_A, local_B, label)

                for ann in annotations:
                    label_id = ann.get('label_id', '')
                    if not label_id.startswith('pedestrian:'):
                        continue
                    pid_A = int(label_id.split(':')[1])
                    if pid_A not in person_dict:
                        continue
                    box_A = person_dict[pid_A]['box']
                    occ_A = person_dict[pid_A]['occlusion']

                    interactions = ann.get('H-interaction', []) or ann.get('HHI', [])
                    for interaction in interactions:
                        pair_id = interaction.get('pair', '')
                        if not pair_id.startswith('pedestrian:'):
                            continue
                        pid_B = int(pair_id.split(':')[1])
                        if pid_B not in person_dict:
                            continue
                        # Only keep A < B to avoid duplicates
                        if pid_A >= pid_B:
                            continue

                        box_B = person_dict[pid_B]['box']
                        occ_B = person_dict[pid_B]['occlusion']

                        inter_labels = interaction.get('inter_labels', {})
                        if not isinstance(inter_labels, dict) or not inter_labels:
                            continue
                        interaction_type = list(inter_labels.keys())[0]
                        stage2_label = self.label_mapper.map_label(interaction_type)
                        if stage2_label is None:
                            continue

                        # Data filtering
                        if self.filter_occlusion:
                            if not (self._is_valid_occlusion(occ_A) and
                                    self._is_valid_occlusion(occ_B)):
                                filtered_occlusion += 1
                                continue
                        if self.filter_edge_cases:
                            if (self._is_edge_box(box_A) or self._is_edge_box(box_B)):
                                filtered_edge += 1
                                continue

                        local_A = pid_to_local[pid_A]
                        local_B = pid_to_local[pid_B]
                        target_pairs_info.append((local_A, local_B, stage2_label))

                if not target_pairs_info:
                    continue

                # All persons' boxes (in sorted-pid order)
                all_boxes = [person_dict[pid]['box'] for pid in sorted_pids]

                # ---- Pre-compute edge_index + 7D edge features (once, not per forward) ----
                N_all = len(sorted_pids)
                if N_all > 1:
                    src_list_e = [i for i in range(N_all) for j in range(N_all) if i != j]
                    dst_list_e = [j for i in range(N_all) for j in range(N_all) if i != j]
                    src_t      = torch.tensor(src_list_e, dtype=torch.long)
                    dst_t      = torch.tensor(dst_list_e, dtype=torch.long)
                    boxes_t    = torch.tensor(all_boxes, dtype=torch.float32)   # [N, 4]
                    pre_edge_index = torch.stack([src_t, dst_t], dim=0)          # [2, E]
                    pre_edge_feats = extract_geometric_features_batch(
                        boxes_t[src_t], boxes_t[dst_t],
                        self.image_width, self.image_height,
                    )                                                             # [E, 7]
                else:
                    pre_edge_index = torch.zeros(2, 0, dtype=torch.long)
                    pre_edge_feats = torch.zeros(0, 7, dtype=torch.float32)

                frame_id = f"{scene_name}_{os.path.splitext(image_name)[0]}"

                self.frames.append({
                    'image_path':    image_path if os.path.exists(image_path) else None,
                    'all_boxes':     all_boxes,            # List[List[4]]
                    'pre_edge_index': pre_edge_index,      # [2, E]  precomputed
                    'pre_edge_feats': pre_edge_feats,      # [E, 7]  precomputed
                    'target_pairs_info': target_pairs_info,
                    'num_persons':   len(sorted_pids),
                    'num_pairs':     len(target_pairs_info),
                    'scene_name':    scene_name,
                    'frame_id':      frame_id,
                })
                self.all_pair_labels.extend(p[2] for p in target_pairs_info)

        total_filtered = filtered_occlusion + filtered_edge
        if total_filtered > 0:
            print(f"Filtered: {filtered_occlusion} occlusion + {filtered_edge} edge = {total_filtered} pairs")

    # ------------------------------------------------------------------
    # Validation helpers (identical to ResNetStage2Dataset)
    # ------------------------------------------------------------------

    def _is_valid_box(self, box) -> bool:
        if len(box) != 4:
            return False
        x, y, w, h = box
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return False
        if w > 5000 or h > 5000:
            return False
        return True

    def _is_valid_occlusion(self, occlusion: str) -> bool:
        return occlusion in ['Fully_visible', 'Mostly_visible']

    def _is_edge_box(self, box) -> bool:
        x, y, w, h = box
        return x < self.edge_threshold or (x + w) > (self.image_width - self.edge_threshold)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        frame = self.frames[idx]

        image_path = frame['image_path']
        if image_path is None or not os.path.exists(image_path):
            return None

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            return None

        all_boxes = frame['all_boxes']
        N = len(all_boxes)

        # Crop each person
        crops = []
        valid_boxes = []
        for box in all_boxes:
            crop_tensor = self._crop_person(image, box)
            crops.append(crop_tensor)
            valid_boxes.append(box)

        person_crops = torch.stack(crops, dim=0)          # [N, 3, H, W]
        person_boxes = torch.tensor(valid_boxes, dtype=torch.float32)  # [N, 4]

        # Build target pairs tensors
        pairs_info = frame['target_pairs_info']           # List[(local_A, local_B, label)]
        target_pairs = torch.tensor(
            [[p[0], p[1]] for p in pairs_info], dtype=torch.long
        )                                                  # [P, 2]
        pair_labels = torch.tensor(
            [p[2] for p in pairs_info], dtype=torch.long
        )                                                  # [P]

        return {
            'person_crops':    person_crops,              # [N, 3, H, W]
            'person_boxes':    person_boxes,              # [N, 4]
            'pre_edge_index':  frame['pre_edge_index'],   # [2, E]  precomputed
            'pre_edge_feats':  frame['pre_edge_feats'],   # [E, 7]  precomputed
            'target_pairs':    target_pairs,              # [P, 2]
            'pair_labels':     pair_labels,               # [P]
            'num_persons':     N,
            'num_pairs':       len(pairs_info),
            'scene_name':      frame['scene_name'],
            'frame_id':        frame['frame_id'],
        }

    def _crop_person(self, image: Image.Image, box: List[float]) -> torch.Tensor:
        """Crop person region from panoramic image and apply transforms."""
        x, y, w, h = box
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(image.width, int(x + w))
        y2 = min(image.height, int(y + h))

        if x2 <= x1 or y2 <= y1:
            # Invalid crop → grey placeholder
            crop = Image.new('RGB', (self.crop_size, self.crop_size), (128, 128, 128))
        else:
            crop = image.crop((x1, y1, x2, y2))

        return self.transform(crop)  # [3, H, W]

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def get_labels(self) -> List[int]:
        """Flat list of all pair labels (for class distribution analysis)."""
        return list(self.all_pair_labels)

    def get_class_distribution(self) -> Dict:
        if not self.all_pair_labels:
            return {}
        counts = Counter(self.all_pair_labels)
        total = len(self.all_pair_labels)
        return {
            'total_pairs': total,
            'total_frames': len(self.frames),
            'class_counts': dict(counts),
            'class_names': self.label_mapper.class_names,
        }


# ============================================================================
# Custom collate function (required: each frame has different N persons)
# ============================================================================

def gnn_collate_fn(batch: List) -> List[Dict]:
    """
    GNN-specific DataLoader collate function.

    Because each frame has a different number of persons (N_i varies),
    we cannot stack tensors into a uniform batch. Instead, return a
    list of dicts (filtering out None items from failed loads).

    The model's forward() handles the 'super-graph' concatenation internally.

    Usage:
        DataLoader(dataset, batch_size=4, collate_fn=gnn_collate_fn)
    """
    return [item for item in batch if item is not None]


# ============================================================================
# DataLoader factory
# ============================================================================

def create_gnn_stage2_data_loaders(config) -> Tuple:
    """
    Create train/val/test DataLoaders for GNN Stage2.

    Args:
        config: GNNStage2Config instance

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    common_kwargs = dict(
        data_path=config.data_path,
        crop_size=config.crop_size,
        frame_interval=config.frame_interval,
        filter_occlusion=config.filter_occlusion,
        filter_edge_cases=config.filter_edge_cases,
        edge_threshold=config.edge_threshold,
        image_width=config.image_width,
        image_height=config.image_height,
    )

    train_dataset = GNNStage2Dataset(split='train', **common_kwargs)
    val_dataset   = GNNStage2Dataset(split='val',   **common_kwargs)
    test_dataset  = GNNStage2Dataset(split='test',  **common_kwargs)

    loader_kwargs = dict(
        collate_fn=gnn_collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    print(f"\nGNN DataLoaders created:")
    print(f"  Train: {len(train_dataset)} frames, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} frames, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} frames, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../../dataset"

    print("Testing GNNStage2Dataset...")
    ds = GNNStage2Dataset(data_path=data_path, split='train', frame_interval=10)
    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        sample = None
        for i in range(len(ds)):
            sample = ds[i]
            if sample is not None:
                break
        if sample:
            print(f"Sample keys: {list(sample.keys())}")
            print(f"  person_crops: {sample['person_crops'].shape}")
            print(f"  person_boxes: {sample['person_boxes'].shape}")
            print(f"  target_pairs: {sample['target_pairs'].shape}")
            print(f"  pair_labels:  {sample['pair_labels'].shape}")
            print(f"  scene: {sample['scene_name']}, frame: {sample['frame_id']}")
    print("Done.")
