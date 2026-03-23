#!/usr/bin/env python3
"""
Visual Stage1 Dataset
Dataset for Stage1 interaction detection using both visual and geometric features
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

from src.features.geometric_features import extract_geometric_features, compute_scene_context


class VisualStage1Dataset(Dataset):
    """
    Dataset for Stage1 interaction detection using visual + geometric features
    """
    
    def __init__(self, data_path: str, split: str = 'train', backbone_name: str = 'resnet18',
                 crop_size: int = 224, frame_interval: int = 1, use_geometric: bool = True,
                 interaction_threshold: float = 2.0):
        """
        Args:
            data_path: Path to JRDB dataset
            split: 'train', 'val', or 'test' 
            backbone_name: ResNet backbone for feature extraction
            crop_size: Size for cropped person images
            frame_interval: Frame sampling interval
            use_geometric: Whether to include geometric features
            interaction_threshold: Distance threshold for positive/negative sampling
        """
        self.data_path = data_path
        self.split = split
        self.backbone_name = backbone_name
        self.crop_size = crop_size
        self.frame_interval = frame_interval
        self.use_geometric = use_geometric
        self.interaction_threshold = interaction_threshold
        
        # Dataset splits (reuse Stage2 splits)
        self.trainset_split = [
        'bytes-cafe-2019-02-07_0', 
        # 'clark-center-2019-02-28_0', 
        'clark-center-2019-02-28_1',
        # 'clark-center-intersection-2019-02-28_0', 
        'cubberly-auditorium-2019-04-22_0',
        # 'cubberly-auditorium-2019-04-22_1',
        # 'discovery-walk-2019-02-28_0',
        'discovery-walk-2019-02-28_1',
        'food-trucks-2019-02-12_0',
        'forbes-cafe-2019-01-22_0', 
        'gates-159-group-meeting-2019-04-03_0',
        'gates-to-clark-2019-02-28_1',
        # 'gates-to-clark-2019-02-28_0',
        # 'gates-ai-lab-2019-02-08_0',
        'gates-ai-lab-2019-04-17_0',
        # 'gates-basement-elevators-2019-01-17_0',
        'gates-basement-elevators-2019-01-17_1', 
        # 'gates-foyer-2019-01-17_0',
        'hewlett-class-2019-01-23_0',
        # 'hewlett-class-2019-01-23_1',
        'hewlett-packard-intersection-2019-01-24_0', 
        'huang-2-2019-01-25_0', 
        'huang-2-2019-01-25_1',
        'huang-basement-2019-01-25_0',
        # 'huang-lane-2019-02-12_0', 
        'huang-intersection-2019-01-22_0',
        'indoor-coupa-cafe-2019-02-06_0',
        'lomita-serra-intersection-2019-01-30_0',
        'memorial-court-2019-03-16_0', 
        # 'meyer-green-2019-03-16_0',
        'meyer-green-2019-03-16_1',
        'nvidia-aud-2019-04-18_0',
        # 'nvidia-aud-2019-04-18_1', 
        # 'nvidia-aud-2019-04-18_2',
        'nvidia-aud-2019-01-25_0',
        'outdoor-coupa-cafe-2019-02-06_0',
        'quarry-road-2019-02-28_0',
        'serra-street-2019-01-30_0',
        'stlc-111-2019-04-19_0', 
        # 'stlc-111-2019-04-19_1', 
        # 'stlc-111-2019-04-19_2',
        'packard-poster-session-2019-03-20_2', 
        # 'packard-poster-session-2019-03-20_0',
        # 'packard-poster-session-2019-03-20_1',
        # 'svl-meeting-gates-2-2019-04-08_0',
        'svl-meeting-gates-2-2019-04-08_1',
        # 'tressider-2019-03-16_0',
        'tressider-2019-03-16_2',
        # 'tressider-2019-03-16_1',
        # 'tressider-2019-04-26_0',
        'tressider-2019-04-26_1',
        'tressider-2019-04-26_2',
        # 'tressider-2019-04-26_3',
        # 'jordan-hall-2019-04-22_0', 
        ]
        
        self.valset_split = [
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
        
        self.testset_split = [
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
        # Image preprocessing
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load data
        self.samples = []
        self.labels = []
        self._load_data()
        
        print(f"VisualStage1Dataset created:")
        print(f"  Split: {self.split}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Positive/Negative ratio: {self._get_label_distribution()}")
    
    def _load_data(self):
        """Load JRDB data for Stage1 interaction detection"""
        social_labels_dir = os.path.join(self.data_path, 'labels', 'labels_2d_activity_social_stitched')
        images_dir = os.path.join(self.data_path, 'images', 'image_stitched')
        
        # Select scenes based on split
        if self.split == 'train':
            scene_splits = self.trainset_split
        elif self.split == 'val':
            scene_splits = self.valset_split
        elif self.split == 'test':
            scene_splits = self.testset_split
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        positive_samples = 0
        negative_samples = 0
        
        for scene_name in scene_splits:
            scene_file = f"{scene_name}.json"
            scene_path = os.path.join(social_labels_dir, scene_file)
            
            if not os.path.exists(scene_path):
                print(f"Warning: Scene file {scene_file} not found")
                continue
            
            try:
                with open(scene_path, 'r') as f:
                    scene_data = json.load(f)
                
                # Process frames with interval sampling
                frame_names = list(scene_data.get('labels', {}).keys())
                frame_names.sort()
                selected_frames = frame_names[::self.frame_interval]
                
                for frame_name in selected_frames:
                    annotations = scene_data['labels'][frame_name]
                    image_path = os.path.join(images_dir, scene_name, frame_name)
                    
                    # Collect all person info
                    person_dict = {}
                    all_boxes = []
                    
                    for ann in annotations:
                        person_id = ann.get('label_id', '')
                        if person_id.startswith('pedestrian:'):
                            pid = int(person_id.split(':')[1])
                            box = ann.get('box', [0, 0, 100, 100])
                            
                            if self._is_valid_box(box):
                                all_boxes.append(box)
                                person_dict[pid] = {
                                    'box': box,
                                    'interactions': ann.get('H-interaction', []) or ann.get('HHI', [])
                                }
                    
                    # Generate positive samples (people with interactions)
                    for ann in annotations:
                        person_id = ann.get('label_id', '')
                        if not person_id.startswith('pedestrian:'):
                            continue
                        
                        person_A_id = int(person_id.split(':')[1])
                        if person_A_id not in person_dict:
                            continue
                        
                        person_A_box = person_dict[person_A_id]['box']
                        
                        for interaction in (ann.get('H-interaction', []) or ann.get('HHI', [])):
                            pair_id = interaction.get('pair', '')
                            if pair_id.startswith('pedestrian:'):
                                person_B_id = int(pair_id.split(':')[1])
                                
                                if person_B_id in person_dict and person_A_id < person_B_id:
                                    person_B_box = person_dict[person_B_id]['box']
                                    
                                    # Positive sample (has interaction)
                                    sample = {
                                        'image_path': image_path,
                                        'person_A_box': person_A_box,
                                        'person_B_box': person_B_box,
                                        'all_boxes': all_boxes,
                                        'scene_name': scene_name,
                                        'frame_name': frame_name,
                                        'stage1_label': 1,  # Has interaction
                                        'interaction_type': list(interaction.get('inter_labels', {}).keys())[0] if interaction.get('inter_labels') else 'unknown'
                                    }
                                    
                                    self.samples.append(sample)
                                    self.labels.append(1)
                                    positive_samples += 1
                    
                    # Generate negative samples (people without interactions)
                    # Use geometric distance to generate hard negatives
                    person_ids = list(person_dict.keys())
                    for i in range(len(person_ids)):
                        for j in range(i+1, len(person_ids)):
                            person_A_id, person_B_id = person_ids[i], person_ids[j]
                            
                            # Check if they already have interaction
                            has_interaction = False
                            for ann in annotations:
                                if ann.get('label_id') == f'pedestrian:{person_A_id}':
                                    for interaction in (ann.get('H-interaction', []) or ann.get('HHI', [])):
                                        if interaction.get('pair') == f'pedestrian:{person_B_id}':
                                            has_interaction = True
                                            break
                                if has_interaction:
                                    break
                            
                            if not has_interaction:
                                person_A_box = person_dict[person_A_id]['box']
                                person_B_box = person_dict[person_B_id]['box']
                                
                                # Calculate distance for hard negative mining
                                geometric_feats = extract_geometric_features(
                                    person_A_box, person_B_box, 3760, 480
                                )
                                center_dist = geometric_feats[5].item()  # center_dist_norm
                                
                                # Only include as negative if distance is reasonable
                                # (not too far apart, to create meaningful negatives)
                                if 0.1 < center_dist < self.interaction_threshold:
                                    sample = {
                                        'image_path': image_path,
                                        'person_A_box': person_A_box,
                                        'person_B_box': person_B_box,
                                        'all_boxes': all_boxes,
                                        'scene_name': scene_name,
                                        'frame_name': frame_name,
                                        'stage1_label': 0,  # No interaction
                                        'interaction_type': 'none'
                                    }
                                    
                                    self.samples.append(sample)
                                    self.labels.append(0)
                                    negative_samples += 1
                            
                            # Balance positive/negative samples
                            if negative_samples >= positive_samples * 2:
                                break
                        if negative_samples >= positive_samples * 2:
                            break
                
            except Exception as e:
                print(f"Error loading scene {scene_name}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} samples:")
        print(f"  Positive samples: {positive_samples}")
        print(f"  Negative samples: {negative_samples}")
        print(f"  Ratio: {negative_samples/max(positive_samples, 1):.2f}:1")
    
    def _is_valid_box(self, box: List[float]) -> bool:
        """Validate bounding box"""
        if len(box) != 4:
            return False
        x, y, w, h = box
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return False
        if w > 5000 or h > 5000:
            return False
        return True
    
    def _get_label_distribution(self) -> str:
        """Get label distribution string"""
        if not self.labels:
            return "No labels"
        
        counter = Counter(self.labels)
        pos = counter.get(1, 0)
        neg = counter.get(0, 0)
        total = pos + neg
        
        if total == 0:
            return "No samples"
        
        return f"{pos}/{total} ({pos/total*100:.1f}%) positive"
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample"""
        sample = self.samples[idx]
        
        # Load image
        image = None
        if os.path.exists(sample['image_path']):
            try:
                image = Image.open(sample['image_path']).convert('RGB')
            except Exception as e:
                print(f"Warning: Failed to load image {sample['image_path']}: {e}")
        
        # Crop person regions
        if image is not None:
            person_A_img = self._crop_person(image, sample['person_A_box'])
            person_B_img = self._crop_person(image, sample['person_B_box'])
        else:
            # Fallback: create dummy images
            person_A_img = Image.new('RGB', (self.crop_size, self.crop_size), color='gray')
            person_B_img = Image.new('RGB', (self.crop_size, self.crop_size), color='gray')
        
        # Transform images
        person_A_tensor = self.transform(person_A_img)
        person_B_tensor = self.transform(person_B_img)
        
        # Extract geometric features
        geometric_features = torch.zeros(7, dtype=torch.float32)
        if self.use_geometric:
            try:
                geom_feats = extract_geometric_features(
                    sample['person_A_box'], sample['person_B_box'], 3760, 480
                )
                if isinstance(geom_feats, torch.Tensor):
                    geometric_features = geom_feats.float()
                else:
                    geometric_features = torch.tensor(geom_feats, dtype=torch.float32)
            except Exception:
                pass
        
        # Scene context
        scene_context = torch.tensor([0.0], dtype=torch.float32)
        try:
            scene_ctx = compute_scene_context(sample['all_boxes'], 3760, 480)
            if isinstance(scene_ctx, torch.Tensor):
                scene_context = scene_ctx.float()
            else:
                scene_context = torch.tensor(scene_ctx, dtype=torch.float32)
        except Exception:
            pass
        
        return {
            'person_A_image': person_A_tensor,
            'person_B_image': person_B_tensor,
            'geometric_features': geometric_features,
            'scene_context': scene_context,
            'stage1_label': torch.tensor(sample['stage1_label'], dtype=torch.long),
            'scene_name': sample['scene_name']
        }
    
    def _crop_person(self, image, box):
        """Crop person region from image"""
        x, y, w, h = box
        
        # Add padding
        padding = 0.1
        x_pad = max(0, x - w * padding)
        y_pad = max(0, y - h * padding)
        w_pad = min(image.width - x_pad, w * (1 + 2 * padding))
        h_pad = min(image.height - y_pad, h * (1 + 2 * padding))
        
        # Crop
        crop_region = (int(x_pad), int(y_pad), int(x_pad + w_pad), int(y_pad + h_pad))
        cropped = image.crop(crop_region)
        
        return cropped
    
    def get_class_weights(self):
        """Get class weights for balancing"""
        if not self.labels:
            return None
        
        counter = Counter(self.labels)
        total = len(self.labels)
        weights = {cls: total / (len(counter) * count) for cls, count in counter.items()}
        return weights


def create_visual_stage1_data_loaders(data_path: str, batch_size: int = 32, 
                                    num_workers: int = 2, backbone_name: str = 'resnet18',
                                    crop_size: int = 224, frame_interval: int = 1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for visual Stage1 dataset"""
    
    print(f"Creating Visual Stage1 data loaders...")
    print(f"  Backbone: {backbone_name}")
    print(f"  Crop size: {crop_size}")
    print(f"  Frame interval: {frame_interval}")
    
    # Create datasets
    train_dataset = VisualStage1Dataset(
        data_path=data_path, split='train', backbone_name=backbone_name,
        crop_size=crop_size, frame_interval=frame_interval
    )
    
    val_dataset = VisualStage1Dataset(
        data_path=data_path, split='val', backbone_name=backbone_name,
        crop_size=crop_size, frame_interval=frame_interval
    )
    
    test_dataset = VisualStage1Dataset(
        data_path=data_path, split='test', backbone_name=backbone_name,
        crop_size=crop_size, frame_interval=frame_interval
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    
    print(f"✅ Visual Stage1 data loaders created:")
    print(f"   Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset):,} samples, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset):,} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    print("Testing VisualStage1Dataset...")
    
    dataset = VisualStage1Dataset(
        data_path="../dataset",
        split='train',
        crop_size=224,
        frame_interval=10  # Faster testing
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Person A image shape: {sample['person_A_image'].shape}")
        print(f"Person B image shape: {sample['person_B_image'].shape}")
        print(f"Geometric features shape: {sample['geometric_features'].shape}")
        print(f"Stage1 label: {sample['stage1_label']}")
        
        # Test data loader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        print(f"Batch person A images shape: {batch['person_A_image'].shape}")
        print(f"Batch labels: {batch['stage1_label']}")
    
    print("Test completed!")