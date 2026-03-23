"""
CAD Dataset for Stage1: Binary Interaction Detection

Uses 7D geometric features to detect if two persons are interacting.
Generates pairwise samples from CAD group annotations.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from itertools import combinations
from typing import Optional, List
import sys
import os
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.cad_annotation_parser import CADAnnotationParser
from src.features.geometric_features import extract_geometric_features, compute_scene_context


class CADGeometricStage1Dataset(Dataset):
    """
    CAD Dataset for Stage1: Binary interaction detection using 7D geometric features

    Label generation rule:
    - Positive (label=1): same social_group_id
    - Negative (label=0): different social_group_id
    """

    def __init__(self,
                 cad_root: str,
                 split: str = 'train',
                 sequences: Optional[List[int]] = None,
                 train_sequences: Optional[List[int]] = None,
                 val_sequences: Optional[List[int]] = None,
                 test_sequences: Optional[List[int]] = None,
                 image_width: int = 720,
                 image_height: int = 480,
                 negative_ratio: float = 1.0,
                 use_scene_context: bool = True):
        """
        Args:
            cad_root: Path to CAD ActivityDataset directory
            split: 'train', 'val', or 'test'
            sequences: Explicit list of sequences to use (overrides split-based selection)
            train_sequences: Default sequences for 'train' split if sequences=None
            val_sequences: Default sequences for 'val' split if sequences=None
            test_sequences: Default sequences for 'test' split if sequences=None
            image_width: CAD image width (default 720)
            image_height: CAD image height (default 480)
            negative_ratio: Ratio of negative to positive samples (1.0 = balanced)
            use_scene_context: Whether to compute scene context features
        """
        self.cad_root = cad_root
        self.split = split
        self.image_width = image_width
        self.image_height = image_height
        self.negative_ratio = negative_ratio
        self.use_scene_context = use_scene_context

        # Initialize parser
        self.parser = CADAnnotationParser(cad_root)

        # Determine which sequences to use
        if sequences is not None:
            # User explicitly specified sequences
            self.sequences = sequences
        else:
            # Use split-based defaults
            split_map = {
                'train': train_sequences or list(range(1, 31)),  # Default: seq01-30
                'val': val_sequences or list(range(31, 38)),     # Default: seq31-37
                'test': test_sequences or list(range(38, 45))    # Default: seq38-44
            }
            self.sequences = split_map.get(split, [])

        if not self.sequences:
            raise ValueError(f"No sequences specified for split '{split}'")

        # Load data
        self.samples = []
        self.scene_data = {}  # For scene context: {(seq_num, frame_id): num_people}

        self._load_data()

        print(f"CADGeometricStage1Dataset loaded: {len(self.samples)} samples ({split})")
        print(f"  Sequences: {self.sequences}")
        print(f"  Image dimensions: {image_width}x{image_height}")
        print(f"  Scene context: {use_scene_context}")

    def _load_data(self):
        """Load and generate pairwise samples from CAD annotations"""
        all_samples = []

        for seq_num in self.sequences:
            df = self.parser.load_sequence_annotations(seq_num)

            # Group by frame_id
            for frame_id, frame_df in df.groupby('frame_id'):
                persons = frame_df.to_dict('records')

                # Store scene context (number of people in frame)
                self.scene_data[(seq_num, frame_id)] = len(persons)

                # Generate all person pairs
                for i, j in combinations(range(len(persons)), 2):
                    person_A = persons[i]
                    person_B = persons[j]

                    # Determine label based on social_group_id
                    if person_A['social_group_id'] == person_B['social_group_id']:
                        label = 1  # Has interaction
                    else:
                        label = 0  # No interaction

                    sample = {
                        'seq_num': seq_num,
                        'frame_id': frame_id,
                        'bbox_A': [person_A['x1'], person_A['y1'],
                                  person_A['x2'], person_A['y2']],
                        'bbox_B': [person_B['x1'], person_B['y1'],
                                  person_B['x2'], person_B['y2']],
                        'track_id_A': person_A['track_id'],
                        'track_id_B': person_B['track_id'],
                        'label': label,
                        'social_group_id_A': person_A['social_group_id'],
                        'social_group_id_B': person_B['social_group_id']
                    }
                    all_samples.append(sample)

        # Balance positive/negative samples
        self.samples = self._balance_samples(all_samples)

    def _balance_samples(self, samples: List[dict]) -> List[dict]:
        """Balance positive and negative samples based on negative_ratio"""
        positive_samples = [s for s in samples if s['label'] == 1]
        negative_samples = [s for s in samples if s['label'] == 0]

        num_positive = len(positive_samples)
        num_negative = len(negative_samples)

        print(f"  Raw samples: {num_positive} positive, {num_negative} negative")

        if num_positive == 0:
            print("  Warning: No positive samples found!")
            return samples

        # If negative_ratio is 0, keep all samples (Focal Loss will handle imbalance)
        if self.negative_ratio == 0:
            print(f"  Keeping all samples (negative_ratio=0, Focal Loss handles imbalance)")
            print(f"  Final: {num_positive} positive, {num_negative} negative")
            balanced_samples = samples.copy()
            random.seed(42)
            random.shuffle(balanced_samples)
            return balanced_samples

        # Calculate target number of negative samples
        target_negative = int(num_positive * self.negative_ratio)

        if target_negative < num_negative:
            # Randomly sample negative samples
            random.seed(42)  # For reproducibility
            negative_samples = random.sample(negative_samples, target_negative)
            print(f"  Balanced to: {num_positive} positive, {target_negative} negative")
        else:
            print(f"  Keeping all samples: {num_positive} positive, {num_negative} negative")

        # Combine and shuffle
        balanced_samples = positive_samples + negative_samples
        random.shuffle(balanced_samples)

        return balanced_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns batch compatible with train_geometric_stage1.py

        Returns:
            {
                'geometric_features': torch.Tensor [7],  # 7D geometric features
                'stage1_label': torch.Tensor [],         # 0 or 1
                'scene_context': torch.Tensor [1],       # crowd level
                'person_A_box': torch.Tensor [4],        # [x1, y1, x2, y2]
                'person_B_box': torch.Tensor [4],
                'sequence': str,                          # 'seq01'
                'frame': int                              # frame_id
            }
        """
        sample = self.samples[idx]

        # Extract bounding boxes
        bbox_A = sample['bbox_A']  # [x1, y1, x2, y2]
        bbox_B = sample['bbox_B']

        # Convert to [x, y, w, h] format for extract_geometric_features
        bbox_A_xywh = [bbox_A[0], bbox_A[1], bbox_A[2] - bbox_A[0], bbox_A[3] - bbox_A[1]]
        bbox_B_xywh = [bbox_B[0], bbox_B[1], bbox_B[2] - bbox_B[0], bbox_B[3] - bbox_B[1]]

        # Extract 7D geometric features
        geometric_features = extract_geometric_features(
            bbox_A_xywh, bbox_B_xywh,
            image_width=self.image_width,
            image_height=self.image_height
        )

        # Compute scene context (crowd level)
        if self.use_scene_context:
            num_people = self.scene_data.get((sample['seq_num'], sample['frame_id']), 0)
            all_boxes = [bbox_A_xywh, bbox_B_xywh]  # Simplified: just the pair
            scene_context = compute_scene_context(
                all_boxes,
                image_width=self.image_width,
                image_height=self.image_height
            )
        else:
            scene_context = torch.tensor([1.0], dtype=torch.float32)

        return {
            'geometric_features': geometric_features,  # [7]
            'stage1_label': torch.tensor(sample['label'], dtype=torch.long),
            'scene_context': scene_context,  # [1]
            'person_A_box': torch.tensor(bbox_A, dtype=torch.float32),
            'person_B_box': torch.tensor(bbox_B, dtype=torch.float32),
            'sequence': sample['seq_num'],  # seq_num as int
            'frame': sample['frame_id'],
            'track_id_A': torch.tensor(sample['track_id_A'], dtype=torch.long),
            'track_id_B': torch.tensor(sample['track_id_B'], dtype=torch.long),
            'social_group_id_A': torch.tensor(sample['social_group_id_A'], dtype=torch.long),
            'social_group_id_B': torch.tensor(sample['social_group_id_B'], dtype=torch.long)
        }


def cad_stage1_collate_fn(batch):
    """
    Custom collate function for CAD Stage1 dataset

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary
    """
    return {
        'geometric_features': torch.stack([item['geometric_features'] for item in batch]),
        'stage1_label': torch.stack([item['stage1_label'] for item in batch]),
        'scene_context': torch.stack([item['scene_context'] for item in batch]),
        'person_A_box': torch.stack([item['person_A_box'] for item in batch]),
        'person_B_box': torch.stack([item['person_B_box'] for item in batch]),
        'sequence': torch.tensor([item['sequence'] for item in batch], dtype=torch.long),
        'frame': torch.tensor([item['frame'] for item in batch], dtype=torch.long),
        'track_id_A': torch.stack([item['track_id_A'] for item in batch]),
        'track_id_B': torch.stack([item['track_id_B'] for item in batch]),
        'social_group_id_A': torch.stack([item['social_group_id_A'] for item in batch]),
        'social_group_id_B': torch.stack([item['social_group_id_B'] for item in batch])
    }


if __name__ == '__main__':
    # Test the dataset
    import sys

    cad_root = "../dataset/cad/ActivityDataset"

    try:
        # Test with seq01 only
        dataset = CADGeometricStage1Dataset(
            cad_root=cad_root,
            split='train',
            sequences=[1, 2],  # Test with first 2 sequences
            image_width=720,
            image_height=480,
            negative_ratio=1.0
        )

        print(f"\nDataset size: {len(dataset)}")

        # Test first sample
        print("\n--- Testing first sample ---")
        sample = dataset[0]
        print(f"Keys: {sample.keys()}")
        print(f"Geometric features shape: {sample['geometric_features'].shape}")
        print(f"Label: {sample['stage1_label'].item()}")
        print(f"Scene context: {sample['scene_context']}")
        print(f"Sequence: {sample['sequence']}, Frame: {sample['frame']}")

        # Test dataloader
        print("\n--- Testing dataloader ---")
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=cad_stage1_collate_fn)
        batch = next(iter(loader))
        print(f"Batch geometric_features shape: {batch['geometric_features'].shape}")
        print(f"Batch stage1_label shape: {batch['stage1_label'].shape}")
        print(f"Batch scene_context shape: {batch['scene_context'].shape}")

        # Check label distribution
        labels = [dataset[i]['stage1_label'].item() for i in range(len(dataset))]
        num_positive = sum(labels)
        num_negative = len(labels) - num_positive
        print(f"\nLabel distribution: {num_positive} positive, {num_negative} negative")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
