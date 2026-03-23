"""
CAD Dataset for Stage2: Interaction Type Classification

Uses visual features + 10D OpGeo features to classify interaction types.
Filters to only interacting pairs (same social_group_id) and uses social_activity_id (0-5) as labels.
"""

import torch
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from itertools import combinations
from typing import Optional, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.cad_annotation_parser import CADAnnotationParser
from src.features.geometric_flow_extractor import GeometricFlowExtractor


class CADResNetStage2Dataset(Dataset):
    """
    CAD Dataset for Stage2: Interaction type classification

    Label generation rule:
    - Filter to only interacting pairs (same social_group_id)
    - Use social_activity_id (0-5) directly as class label
    """

    # Class merge mapping: 6-class -> 3-class
    # Original: NA(0), Crossing(1), Waiting(2), Queuing(3), Walking(4), Talking(5)
    # Merged: Moving(0), Standing(1), Talking(2)
    # - NA(0) -> skip (not included)
    # - Crossing(1), Walking(4) -> Moving(0)
    # - Waiting(2), Queuing(3) -> Standing(1)
    # - Talking(5) -> Talking(2)
    CLASS_MERGE_MAP = {
        1: 0,  # Crossing -> Moving
        2: 1,  # Waiting -> Standing
        3: 1,  # Queuing -> Standing
        4: 0,  # Walking -> Moving
        5: 2,  # Talking -> Talking
    }
    CLASS_MERGE_NAMES = ['Moving', 'Standing', 'Talking']

    def __init__(self,
                 cad_root: str,
                 split: str = 'train',
                 sequences: Optional[List[int]] = None,
                 train_sequences: Optional[List[int]] = None,
                 val_sequences: Optional[List[int]] = None,
                 test_sequences: Optional[List[int]] = None,
                 num_classes: int = 6,
                 image_width: int = 720,
                 image_height: int = 480,
                 backbone_name: str = 'resnet18',
                 feature_mode: str = 'both',
                 visual_feature_dim: int = 512,
                 class_merge: bool = False):
        """
        Args:
            cad_root: Path to CAD ActivityDataset directory
            split: 'train', 'val', or 'test'
            sequences: Explicit list of sequences to use
            train_sequences: Default train sequences
            val_sequences: Default val sequences
            test_sequences: Default test sequences
            num_classes: Number of social activity classes (6 for CAD, 3 for merged)
            image_width: CAD image width (default 720)
            image_height: CAD image height (default 480)
            backbone_name: ResNet variant ('resnet18', 'resnet50')
            feature_mode: 'both', 'opticalflow_only', 'bboxposition_only'
            visual_feature_dim: Dimension of visual features (512 for resnet18, 2048 for resnet50)
            class_merge: If True, merge classes: NA->skip, Crossing+Walking->Moving,
                        Waiting+Queuing->Standing, Talking->Talking (3 classes total)
        """
        self.cad_root = cad_root
        self.split = split
        self.class_merge = class_merge
        # Override num_classes if class_merge is enabled
        self.num_classes = 3 if class_merge else num_classes
        self.image_width = image_width
        self.image_height = image_height
        self.feature_mode = feature_mode
        self.visual_feature_dim = visual_feature_dim

        # Initialize parser
        self.parser = CADAnnotationParser(cad_root)

        # Determine which sequences to use
        if sequences is not None:
            self.sequences = sequences
        else:
            split_map = {
                'train': train_sequences or list(range(1, 31)),
                'val': val_sequences or list(range(31, 38)),
                'test': test_sequences or list(range(38, 45))
            }
            self.sequences = split_map.get(split, [])

        if not self.sequences:
            raise ValueError(f"No sequences specified for split '{split}'")

        # Initialize ResNet backbone for visual features
        self._initialize_backbone(backbone_name)

        # Initialize GeometricFlowExtractor for 10D OpGeo features
        self.geo_extractor = GeometricFlowExtractor(flow_bound=20.0, cache_enabled=False)

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Load data
        self.samples = []
        self._load_data()

        print(f"CADResNetStage2Dataset loaded: {len(self.samples)} samples ({split})")
        print(f"  Sequences: {self.sequences}")
        if self.class_merge:
            print(f"  Num classes: {self.num_classes} (merged: {self.CLASS_MERGE_NAMES})")
        else:
            print(f"  Num classes: {self.num_classes} (social_activity_id: 0-{self.num_classes - 1})")
        print(f"  Image dimensions: {image_width}x{image_height}")
        print(f"  Feature mode: {feature_mode}")

    def _initialize_backbone(self, backbone_name: str):
        """Initialize ResNet backbone for visual feature extraction"""
        if backbone_name == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.visual_feature_dim = 512
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.visual_feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Remove the final classification layer
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.backbone.eval()  # Set to evaluation mode

        # Move to GPU if available
        if torch.cuda.is_available():
            self.backbone = self.backbone.cuda()

        print(f"  Visual backbone: {backbone_name} (feature_dim={self.visual_feature_dim})")

    def _load_data(self):
        """Load positive interaction samples with social_activity_id labels"""
        all_samples = []

        for seq_num in self.sequences:
            df = self.parser.load_sequence_annotations(seq_num)

            # Group by frame_id
            for frame_id, frame_df in df.groupby('frame_id'):
                # Check if prev_frame exists (required for optical flow)
                prev_frame_path = self.parser.get_prev_frame_path(seq_num, frame_id)
                if prev_frame_path is None or not prev_frame_path.exists():
                    continue  # Skip frames without previous frame

                curr_frame_path = self.parser.get_frame_path(seq_num, frame_id)
                if not curr_frame_path.exists():
                    continue  # Skip if current frame doesn't exist

                persons = frame_df.to_dict('records')

                # Generate positive pairs only (same social_group_id)
                for i, j in combinations(range(len(persons)), 2):
                    person_A = persons[i]
                    person_B = persons[j]

                    # Only interacting pairs
                    if person_A['social_group_id'] != person_B['social_group_id']:
                        continue

                    # Use social_activity_id directly as label (already 0-indexed: 0-5)
                    activity_id = person_A['social_activity_id']

                    # Handle class merge mode
                    if self.class_merge:
                        # Skip NA (activity_id=0)
                        if activity_id == 0:
                            continue
                        # Map to merged class (3-class)
                        if activity_id not in self.CLASS_MERGE_MAP:
                            print(f"Warning: activity_id {activity_id} not in CLASS_MERGE_MAP, skipping")
                            continue
                        label = self.CLASS_MERGE_MAP[activity_id]
                    else:
                        # Validate activity_id is in range for 6-class mode
                        if activity_id < 0 or activity_id >= 6:
                            print(f"Warning: Invalid activity_id {activity_id} (expected 0-5)")
                            continue
                        label = activity_id

                    sample = {
                        'seq_num': seq_num,
                        'frame_id': frame_id,
                        'prev_frame_path': str(prev_frame_path),
                        'curr_frame_path': str(curr_frame_path),
                        'bbox_A': [person_A['x1'], person_A['y1'],
                                  person_A['x2'], person_A['y2']],
                        'bbox_B': [person_B['x1'], person_B['y1'],
                                  person_B['x2'], person_B['y2']],
                        'track_id_A': person_A['track_id'],
                        'track_id_B': person_B['track_id'],
                        'social_group_id_A': person_A['social_group_id'],
                        'social_group_id_B': person_B['social_group_id'],
                        'label': label,  # Merged (0-2) or original (0-5)
                        'social_activity_id': activity_id  # Keep original for reference
                    }
                    all_samples.append(sample)

        self.samples = all_samples

        # Print label distribution with class names
        label_counts = {}
        for sample in self.samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        if self.class_merge:
            print(f"  Label distribution (3-class merged):")
            for label_id, count in sorted(label_counts.items()):
                class_name = self.CLASS_MERGE_NAMES[label_id] if label_id < len(self.CLASS_MERGE_NAMES) else f"Unknown({label_id})"
                print(f"    {label_id} ({class_name}): {count}")
        else:
            print(f"  Label distribution (6-class): {label_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Extract visual features + 10D OpGeo features

        Returns:
            {
                'visual_A': torch.Tensor [visual_dim],  # ResNet features for person A
                'visual_B': torch.Tensor [visual_dim],  # ResNet features for person B
                'geometric': torch.Tensor [10],         # 10D OpGeo features
                'label': torch.Tensor [],               # 0-5 for 6-class
                'sequence': str,
                'frame': int
            }
        """
        sample = self.samples[idx]

        # Load images
        prev_image = Image.open(sample['prev_frame_path']).convert('RGB')
        curr_image = Image.open(sample['curr_frame_path']).convert('RGB')

        # Extract bounding boxes
        bbox_A = sample['bbox_A']  # [x1, y1, x2, y2]
        bbox_B = sample['bbox_B']

        # Convert to [x, y, w, h] format for feature extractors
        bbox_A_xywh = [bbox_A[0], bbox_A[1], bbox_A[2] - bbox_A[0], bbox_A[3] - bbox_A[1]]
        bbox_B_xywh = [bbox_B[0], bbox_B[1], bbox_B[2] - bbox_B[0], bbox_B[3] - bbox_B[1]]

        # Extract 10D OpGeo features
        geometric_features = self.geo_extractor.extract_geometric_features(
            prev_image, curr_image,
            bbox_A_xywh, bbox_B_xywh
        )

        # Extract visual features
        visual_A = self._extract_visual_features(curr_image, bbox_A)
        visual_B = self._extract_visual_features(curr_image, bbox_B)

        return {
            'visual_A': visual_A,
            'visual_B': visual_B,
            'geometric': geometric_features,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'sequence': sample['seq_num'],  # Return as int
            'frame': sample['frame_id'],
            'track_id_A': torch.tensor(sample['track_id_A'], dtype=torch.long),
            'track_id_B': torch.tensor(sample['track_id_B'], dtype=torch.long),
            'bbox_A': torch.tensor(bbox_A, dtype=torch.float32),
            'bbox_B': torch.tensor(bbox_B, dtype=torch.float32),
            'social_group_id_A': torch.tensor(sample['social_group_id_A'], dtype=torch.long),
            'social_group_id_B': torch.tensor(sample['social_group_id_B'], dtype=torch.long),
            'social_activity_id': torch.tensor(sample['social_activity_id'], dtype=torch.long)
        }

    def _extract_visual_features(self, image: Image, bbox: List[int]) -> torch.Tensor:
        """
        Crop person region and extract ResNet features

        Args:
            image: PIL Image
            bbox: [x1, y1, x2, y2]

        Returns:
            features: [visual_dim] tensor
        """
        # Crop person region
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)

        if x2 <= x1 or y2 <= y1:
            # Invalid bbox, return zero features
            return torch.zeros(self.visual_feature_dim, dtype=torch.float32)

        crop = image.crop((x1, y1, x2, y2))

        # Transform and extract features
        crop_tensor = self.transform(crop).unsqueeze(0)  # [1, 3, 224, 224]

        if torch.cuda.is_available():
            crop_tensor = crop_tensor.cuda()

        with torch.no_grad():
            features = self.backbone(crop_tensor)  # [1, feature_dim, 1, 1]
            features = features.squeeze()  # [feature_dim]

        # Move to CPU
        features = features.cpu()

        return features


def cad_stage2_collate_fn(batch):
    """
    Custom collate function for CAD Stage2 dataset

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary
    """
    return {
        'visual_A': torch.stack([item['visual_A'] for item in batch]),
        'visual_B': torch.stack([item['visual_B'] for item in batch]),
        'geometric': torch.stack([item['geometric'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'sequence': torch.tensor([item['sequence'] for item in batch], dtype=torch.long),
        'frame': torch.tensor([item['frame'] for item in batch], dtype=torch.long),
        'track_id_A': torch.stack([item['track_id_A'] for item in batch]),
        'track_id_B': torch.stack([item['track_id_B'] for item in batch]),
        'bbox_A': torch.stack([item['bbox_A'] for item in batch]),
        'bbox_B': torch.stack([item['bbox_B'] for item in batch]),
        'social_group_id_A': torch.stack([item['social_group_id_A'] for item in batch]),
        'social_group_id_B': torch.stack([item['social_group_id_B'] for item in batch]),
        'social_activity_id': torch.stack([item['social_activity_id'] for item in batch])
    }


if __name__ == '__main__':
    # Test the dataset
    import sys

    cad_root = "../dataset/cad/ActivityDataset"

    try:
        # Test with seq01-02
        dataset = CADResNetStage2Dataset(
            cad_root=cad_root,
            split='train',
            sequences=[1, 2],
            num_classes=6,
            image_width=720,
            image_height=480,
            backbone_name='resnet18',
            feature_mode='both'
        )

        print(f"\nDataset size: {len(dataset)}")

        if len(dataset) > 0:
            # Test first sample
            print("\n--- Testing first sample ---")
            sample = dataset[0]
            print(f"Keys: {sample.keys()}")
            print(f"Visual A shape: {sample['visual_A'].shape}")
            print(f"Visual B shape: {sample['visual_B'].shape}")
            print(f"Geometric shape: {sample['geometric'].shape}")
            print(f"Label: {sample['label'].item()}")
            print(f"Sequence: {sample['sequence']}, Frame: {sample['frame']}")

            # Test dataloader
            print("\n--- Testing dataloader ---")
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=cad_stage2_collate_fn)
            batch = next(iter(loader))
            print(f"Batch visual_A shape: {batch['visual_A'].shape}")
            print(f"Batch visual_B shape: {batch['visual_B'].shape}")
            print(f"Batch geometric shape: {batch['geometric'].shape}")
            print(f"Batch label shape: {batch['label'].shape}")
            print(f"Batch labels: {batch['label']}")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
