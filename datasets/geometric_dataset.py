import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from collections import defaultdict
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.geometric_features import extract_geometric_features, extract_causal_motion_features, compute_scene_context
from torch.nn.utils.rnn import pad_sequence
from src.data_loaders.temporal_buffer import CausalTemporalBuffer, TemporalPairManager

def _get_frame_id_sort_key(sample):
    """Helper function to get frame_id for sorting (pickle-friendly)"""
    return sample['frame_id']


class GeometricDualPersonDataset(Dataset):
    """
    Dataset for geometric dual-person interaction detection
    Focuses on geometric features rather than visual features
    """
    
    def __init__(self, data_path, split='train', history_length=5,
                 use_temporal=True, use_scene_context=True,
                 trainset_split=None, valset_split=None, testset_split=None,
                 use_custom_splits=False, frame_interval=1, feature_mode='legacy'):
        """
        Args:
            data_path: Path to dataset root directory
            split: 'train', 'val', or 'test'
            history_length: Number of historical frames to use
            use_temporal: Whether to use temporal features
            use_scene_context: Whether to use scene context features
            trainset_split: List of scene names for training split
            valset_split: List of scene names for validation split
            testset_split: List of scene names for test split
            use_custom_splits: Whether to use custom scene splits instead of percentage-based
            frame_interval: Frame sampling interval (1=every frame, 5=every 5th frame)
            feature_mode: Feature extraction mode ('legacy' for 7D, 'both' for 10D,
                         'opticalflow_only' for 5D optical flow, 'bboxposition_only' for 5D position)
        """
        self.data_path = data_path
        self.split = split
        self.history_length = history_length
        self.use_temporal = use_temporal
        self.use_scene_context = use_scene_context
        self.use_custom_splits = use_custom_splits
        self.frame_interval = frame_interval
        self.feature_mode = feature_mode

        # Initialize 10D feature extractors for non-legacy modes
        if feature_mode in ['both', 'opticalflow_only', 'bboxposition_only']:
            from src.features.geometric_flow_extractor import GeometricFlowExtractor
            from src.features.interaction_synchrony import compute_interaction_synchrony
            self.geometric_flow_extractor = GeometricFlowExtractor(flow_bound=20.0, cache_enabled=True)
            self.compute_synchrony = compute_interaction_synchrony
        else:
            self.geometric_flow_extractor = None
            self.compute_synchrony = None

        # Store custom splits
        self.trainset_split = trainset_split or []
        self.valset_split = valset_split or []
        self.testset_split = testset_split or []

        # Initialize temporal manager
        if use_temporal:
            self.temporal_manager = TemporalPairManager(history_length=history_length)
        else:
            self.temporal_manager = None

        # Load data
        self.samples = []
        self.scene_data = {}  # For scene context computation
        self.feature_cache = {}  # Cache for precomputed features

        self._load_data()
        self._precompute_scene_context()

        # Precompute 10D features if using non-legacy mode
        if self.feature_mode in ['both', 'opticalflow_only', 'bboxposition_only']:
            self._precompute_10d_features()
        
        print(f"GeometricDualPersonDataset loaded: {len(self.samples)} samples ({split})")
        print(f"  Temporal features: {use_temporal}")
        print(f"  Scene context: {use_scene_context}")
        print(f"  Frame sampling interval: {frame_interval} (every {frame_interval} frame{'s' if frame_interval > 1 else ''})")
    
    def _load_data(self):
        """Load geometric interaction data from JRDB format"""
        # JRDB format: separate JSON files for each scene
        social_labels_dir = os.path.join(self.data_path, 'labels', 'labels_2d_activity_social_stitched')
        images_dir = os.path.join(self.data_path, 'images', 'image_stitched')  # For 10D features

        if not os.path.exists(social_labels_dir):
            raise FileNotFoundError(f"Social labels directory not found: {social_labels_dir}")

        # Get all scene files
        scene_files = [f for f in os.listdir(social_labels_dir) if f.endswith('.json')]

        # Split scenes for train/val/test
        scene_files.sort()  # Ensure consistent ordering
        total_scenes = len(scene_files)

        if self.split == 'train':
            selected_files = scene_files[:int(0.7 * total_scenes)]
        elif self.split == 'val':
            selected_files = scene_files[int(0.7 * total_scenes):int(0.85 * total_scenes)]
        else:  # test
            selected_files = scene_files[int(0.85 * total_scenes):]

        print(f"Loading {len(selected_files)} scenes for {self.split} split")

        # Track frame sequence for previous frame lookup (for 10D features)
        self.scene_frames = {}  # scene_name -> sorted list of frame_names

        # Load data from selected scenes
        all_social_data = {}
        for scene_file in selected_files:
            scene_path = os.path.join(social_labels_dir, scene_file)
            scene_name = os.path.splitext(scene_file)[0]
            
            try:
                with open(scene_path, 'r') as f:
                    scene_data = json.load(f)
                all_social_data[scene_name] = scene_data
            except Exception as e:
                print(f"Error loading scene {scene_file}: {e}")
                continue
        
        # Process social annotations to extract geometric pairs
        frame_count = 0

        for scene_name, scene_data in all_social_data.items():
            # Build sorted frame list for this scene (for previous frame lookup)
            frame_names = sorted(scene_data.get('labels', {}).keys(),
                               key=lambda x: int(self._extract_frame_id(x)))
            self.scene_frames[scene_name] = frame_names

            for image_name, annotations in scene_data.get('labels', {}).items():
                # Apply frame interval sampling
                frame_number = int(self._extract_frame_id(image_name))
                if frame_number % self.frame_interval != 0:
                    continue  # Skip this frame based on sampling interval

                # Create unique frame_id combining scene and image
                frame_id = f"{scene_name}_{self._extract_frame_id(image_name)}"

                # Find previous frame for optical flow (for 10D features)
                prev_image_path = None
                prev_image_name = None
                if scene_name in self.scene_frames:
                    frame_list = self.scene_frames[scene_name]
                    try:
                        current_idx = frame_list.index(image_name)
                        if current_idx > 0:
                            prev_image_name = frame_list[current_idx - 1]
                            prev_image_path = os.path.join(images_dir, scene_name, prev_image_name)
                    except ValueError:
                        pass  # image_name not in list

                # Collect all person boxes in this frame for scene context
                all_boxes = []
                person_dict = {}
                
                for ann in annotations:
                    person_id = ann.get('label_id', '')
                    if person_id.startswith('pedestrian:'):
                        pid = int(person_id.split(':')[1])
                        box = ann.get('box', [0, 0, 100, 100])
                        all_boxes.append(box)
                        person_dict[pid] = {
                            'box': box,
                            'actions': ann.get('action_label', {}),
                            'interactions': ann.get('H-interaction', []) or ann.get('HHI', [])
                        }
                
                # Store scene information
                self.scene_data[frame_id] = {
                    'scene_name': scene_name,
                    'image_name': image_name,
                    'all_boxes': all_boxes,
                    'persons': person_dict
                }
                
                # Generate positive samples (has interaction)
                for ann in annotations:
                    person_id = ann.get('label_id', '')
                    if not person_id.startswith('pedestrian:'):
                        continue
                    
                    person_A_id = int(person_id.split(':')[1])
                    person_A_box = ann.get('box', [0, 0, 100, 100])
                    
                    # Process H-interaction (JRDB format)
                    for interaction in (ann.get('H-interaction', []) or ann.get('HHI', [])):
                        pair_id = interaction.get('pair', '')
                        if pair_id.startswith('pedestrian:'):
                            person_B_id = int(pair_id.split(':')[1])
                            
                            if person_B_id in person_dict:
                                person_B_box = person_dict[person_B_id]['box']
                                interaction_labels = interaction.get('inter_labels', {})
                                
                                # Create positive sample
                                sample = {
                                    'frame_id': frame_id,
                                    'scene_name': scene_name,
                                    'image_name': image_name,
                                    'image_path': os.path.join(images_dir, scene_name, image_name),  # NEW: for optical flow
                                    'prev_image_name': prev_image_name,  # NEW: for optical flow
                                    'prev_image_path': prev_image_path,  # NEW: for optical flow
                                    'person_A_id': person_A_id,
                                    'person_B_id': person_B_id,
                                    'person_A_box': person_A_box,
                                    'person_B_box': person_B_box,
                                    'has_interaction': 1,
                                    'interaction_labels': interaction_labels,
                                    'sample_type': 'positive'
                                }
                                self.samples.append(sample)
                
                # Generate negative samples (no interaction)
                person_ids = list(person_dict.keys())
                if len(person_ids) >= 2:
                    # Find pairs without interactions
                    interacting_pairs = set()
                    for ann in annotations:
                        person_id = ann.get('label_id', '')
                        if person_id.startswith('pedestrian:'):
                            person_A_id = int(person_id.split(':')[1])
                            for interaction in (ann.get('H-interaction', []) or ann.get('HHI', [])):
                                pair_id = interaction.get('pair', '')
                                if pair_id.startswith('pedestrian:'):
                                    person_B_id = int(pair_id.split(':')[1])
                                    interacting_pairs.add(tuple(sorted([person_A_id, person_B_id])))
                    
                    # Generate negative samples
                    neg_count = 0
                    max_neg_per_frame = min(len(person_ids) * 2, 10)
                    
                    for i, person_A_id in enumerate(person_ids):
                        for person_B_id in person_ids[i+1:]:
                            pair = tuple(sorted([person_A_id, person_B_id]))
                            if pair not in interacting_pairs and neg_count < max_neg_per_frame:
                                sample = {
                                    'frame_id': frame_id,
                                    'scene_name': scene_name,
                                    'image_name': image_name,
                                    'image_path': os.path.join(images_dir, scene_name, image_name),  # NEW: for optical flow
                                    'prev_image_name': prev_image_name,  # NEW: for optical flow
                                    'prev_image_path': prev_image_path,  # NEW: for optical flow
                                    'person_A_id': person_A_id,
                                    'person_B_id': person_B_id,
                                    'person_A_box': person_dict[person_A_id]['box'],
                                    'person_B_box': person_dict[person_B_id]['box'],
                                    'has_interaction': 0,
                                    'interaction_labels': {},
                                    'sample_type': 'negative'
                                }
                                self.samples.append(sample)
                                neg_count += 1
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames, {len(self.samples)} samples")
    
    def _extract_frame_id(self, image_name):
        """Extract frame ID from image name (JRDB format)"""
        # JRDB format: "000000.jpg" -> "000000"
        return os.path.splitext(image_name)[0]
    
    def _precompute_scene_context(self):
        """Precompute scene context for all frames"""
        total_frames = len(self.scene_data)
        print(f"Precomputing scene context for {total_frames} frames...")
        
        processed = 0
        for frame_id, scene_info in self.scene_data.items():
            all_boxes = scene_info['all_boxes']
            if len(all_boxes) > 0:
                # Use JRDB standard image dimensions
                scene_context = compute_scene_context(all_boxes, 3760, 480)
                self.scene_data[frame_id]['scene_context'] = scene_context
            else:
                self.scene_data[frame_id]['scene_context'] = torch.tensor([0.0], dtype=torch.float32)  # Empty scene
            
            processed += 1
            if processed % 1000 == 0:  # Every 1000 frames
                print(f"  Progress: {processed}/{total_frames} frames ({100*processed/total_frames:.1f}%)")
        
        print(f"Scene context precomputation completed: {total_frames} frames processed")

    def _precompute_10d_features(self):
        """
        Precompute 10D features for all samples to avoid repeated optical flow computation
        Features are cached in memory and saved to disk for future runs
        """
        # Generate cache filename
        cache_file = os.path.join(
            self.data_path,
            'feature_cache',
            f'opgeo_features_{self.split}_{self.feature_mode}_interval{self.frame_interval}.pt'
        )

        # Create cache directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        # Try to load existing cache
        if os.path.exists(cache_file):
            print(f"\nLoading precomputed features from cache...")
            print(f"  Cache file: {cache_file}")
            try:
                self.feature_cache = torch.load(cache_file)
                print(f"  Successfully loaded {len(self.feature_cache)} cached features")

                # Verify cache matches current samples
                if len(self.feature_cache) == len(self.samples):
                    print(f"  Cache validation: PASSED")
                    return
                else:
                    print(f"  Cache validation: FAILED (size mismatch)")
                    print(f"  Expected {len(self.samples)}, got {len(self.feature_cache)}")
                    print(f"  Recomputing features...")
                    self.feature_cache = {}
            except Exception as e:
                print(f"  Error loading cache: {e}")
                print(f"  Recomputing features...")
                self.feature_cache = {}

        # Precompute features for all samples
        print(f"\n{'='*60}")
        print(f"Precomputing {self.feature_mode} features for {len(self.samples)} interaction pairs")
        print(f"  Split: {self.split}")
        print(f"  Frame interval: {self.frame_interval}")
        print(f"  This may take 10-30 minutes depending on dataset size...")
        print(f"{'='*60}\n")

        import time
        start_time = time.time()

        # Track progress
        total_samples = len(self.samples)
        checkpoint_interval = max(50, total_samples // 20)  # Report every 5%

        for idx in range(total_samples):
            sample = self.samples[idx]

            # Extract 10D features for this interaction pair
            try:
                features = self._extract_10d_features(sample)
                self.feature_cache[idx] = features
            except Exception as e:
                print(f"\n  Warning: Failed to extract features for sample {idx}: {e}")
                # Use zero features as fallback
                dim = 10 if self.feature_mode == 'both' else 5
                self.feature_cache[idx] = torch.zeros(dim, dtype=torch.float32)

            # Progress reporting
            if (idx + 1) % checkpoint_interval == 0 or (idx + 1) == total_samples:
                elapsed = time.time() - start_time
                progress_pct = 100 * (idx + 1) / total_samples
                samples_per_sec = (idx + 1) / elapsed
                eta_sec = (total_samples - idx - 1) / samples_per_sec if samples_per_sec > 0 else 0

                print(f"  Progress: {idx+1:6d}/{total_samples} ({progress_pct:5.1f}%) | "
                      f"Speed: {samples_per_sec:5.1f} pairs/s | "
                      f"Elapsed: {elapsed/60:5.1f}m | "
                      f"ETA: {eta_sec/60:5.1f}m")

        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Feature precomputation completed!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Average speed: {total_samples/total_time:.1f} pairs/second")
        print(f"  Cached features: {len(self.feature_cache)}")
        print(f"{'='*60}\n")

        # Save cache to disk
        print(f"Saving feature cache to disk...")
        print(f"  File: {cache_file}")
        try:
            torch.save(self.feature_cache, cache_file)
            cache_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"  Cache saved successfully ({cache_size_mb:.1f} MB)")
            print(f"  Future runs will load from cache and skip precomputation")
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")
            print(f"  Features are still cached in memory for this run")

    def _extract_10d_features(self, sample):
        """
        Extract 10D geometric features following stage2's OfflineFeatureExtractor logic
        Returns [10] for 'both' mode or [5] for subset modes
        """
        prev_image_path = sample.get('prev_image_path')
        image_path = sample.get('image_path')

        # Handle missing previous frame (first frame or missing file)
        if not prev_image_path or not os.path.exists(prev_image_path):
            dim = 10 if self.feature_mode == 'both' else 5
            return torch.zeros(dim, dtype=torch.float32)

        # Verify current image exists
        if not os.path.exists(image_path):
            dim = 10 if self.feature_mode == 'both' else 5
            return torch.zeros(dim, dtype=torch.float32)

        # Load images
        from PIL import Image
        try:
            prev_image = Image.open(prev_image_path).convert('RGB')
            curr_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}")
            dim = 10 if self.feature_mode == 'both' else 5
            return torch.zeros(dim, dtype=torch.float32)

        # Extract 9D features from both perspectives (A->B and B->A)
        features_A_to_B = self.geometric_flow_extractor.extract_geometric_features(
            prev_image, curr_image, sample['person_A_box'], sample['person_B_box']
        )  # [9]

        features_B_to_A = self.geometric_flow_extractor.extract_geometric_features(
            prev_image, curr_image, sample['person_B_box'], sample['person_A_box']
        )  # [9]

        # Convert to numpy for symmetric averaging
        feat_A = features_A_to_B.cpu().numpy()
        feat_B = features_B_to_A.cpu().numpy()

        # Select feature subset based on mode
        feat_A_selected = self._select_features_by_mode(feat_A)
        feat_B_selected = self._select_features_by_mode(feat_B)

        # Symmetric averaging
        if self.feature_mode == 'both':
            # Indices: 0,1,5,6,7 symmetric; 2,3,4,8 averaged
            symmetric_features = np.array([
                feat_A_selected[0],  # f0: distance/height
                feat_A_selected[1],  # f1: distance/width
                (feat_A_selected[2] + feat_B_selected[2]) / 2.0,  # f2: flow_mean/area
                (feat_A_selected[3] + feat_B_selected[3]) / 2.0,  # f3: flow_std/area
                (feat_A_selected[4] + feat_B_selected[4]) / 2.0,  # f4: vertical_dominance
                feat_A_selected[5],  # f5: aspect_ratio
                feat_A_selected[6],  # f7: relative_height
                feat_A_selected[7],  # f8: relative_bottom
                (feat_A_selected[8] + feat_B_selected[8]) / 2.0,  # f9: direction_consistency
            ], dtype=np.float32)
        else:
            # For 5D modes, average all features
            symmetric_features = (feat_A_selected + feat_B_selected) / 2.0

        # Compute interaction synchrony (10th dimension) - only for optical flow modes
        # bboxposition_only mode doesn't use synchrony (static features only)
        if self.feature_mode in ['both', 'opticalflow_only']:
            sync_score = self.compute_synchrony(
                torch.tensor(feat_A[:6], dtype=torch.float32),  # First 6 features
                torch.tensor(feat_B[:6], dtype=torch.float32)
            )

            # Concatenate to form final feature vector
            features_10d = torch.cat([
                torch.tensor(symmetric_features, dtype=torch.float32),
                sync_score.unsqueeze(0) if sync_score.dim() == 0 else sync_score
            ], dim=0)
        else:
            # bboxposition_only: no synchrony, just return the 5D static features
            features_10d = torch.tensor(symmetric_features, dtype=torch.float32)

        return features_10d

    def _select_features_by_mode(self, full_features_9d):
        """
        Select feature subset based on feature_mode
        Mirrors stage2's select_features_by_mode function
        """
        if self.feature_mode == 'both':
            return full_features_9d  # All 9 features
        elif self.feature_mode == 'opticalflow_only':
            return np.array([
                full_features_9d[2],  # f2: flow_mean/area
                full_features_9d[3],  # f3: flow_std/area
                full_features_9d[4],  # f4: vertical_dominance
                full_features_9d[8],  # f9: direction_consistency
            ], dtype=np.float32)
        elif self.feature_mode == 'bboxposition_only':
            return np.array([
                full_features_9d[0],  # f0: distance/height
                full_features_9d[1],  # f1: distance/width
                full_features_9d[5],  # f5: aspect_ratio
                full_features_9d[6],  # f7: relative_height
                full_features_9d[7],  # f8: relative_bottom
            ], dtype=np.float32)
        else:
            return full_features_9d

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample with geometric features and optional temporal information
        """
        sample = self.samples[idx]

        # Extract basic information
        frame_id = sample['frame_id']
        person_A_id = sample['person_A_id']
        person_B_id = sample['person_B_id']
        person_A_box = torch.tensor(sample['person_A_box'], dtype=torch.float32)
        person_B_box = torch.tensor(sample['person_B_box'], dtype=torch.float32)
        has_interaction = torch.tensor(sample['has_interaction'], dtype=torch.long)

        # Extract geometric features based on mode
        if self.feature_mode == 'legacy':
            # Original 7D extraction (computed on-the-fly, fast)
            geometric_features = extract_geometric_features(
                person_A_box, person_B_box, 3760, 480  # JRDB standard dimensions
            )  # [7]
        else:
            # 10D extraction - use precomputed cache
            if idx in self.feature_cache:
                geometric_features = self.feature_cache[idx].clone()  # Clone to avoid in-place modifications
            else:
                # Fallback: compute on-the-fly if cache miss (shouldn't happen)
                print(f"Warning: Cache miss for sample {idx}, computing on-the-fly")
                geometric_features = self._extract_10d_features(sample)  # [10] or [5]
        
        # Prepare result
        result = {
            'geometric_features': geometric_features.clone(),
            'stage1_label': has_interaction,
            'person_A_id': person_A_id,
            'person_B_id': person_B_id,
            'frame_id': frame_id,
            'person_A_box': person_A_box,
            'person_B_box': person_B_box
        }
        
        # Add scene context if enabled
        if self.use_scene_context and frame_id in self.scene_data:
            result['scene_context'] = self.scene_data[frame_id]['scene_context'].clone()
        else:
            result['scene_context'] = torch.tensor([1.0], dtype=torch.float32)  # Default: sparse scene
        
        # Add temporal features if enabled
        if self.use_temporal and self.temporal_manager:
            temporal_features = self.temporal_manager.get_temporal_features(person_A_id, person_B_id)
            
            # Use pair interaction history as the main temporal signal
            result['history_geometric'] = temporal_features['pair_interaction_history'].clone()
            result['has_sufficient_history'] = temporal_features['has_sufficient_history']
            
            # Extract motion features from historical data
            if temporal_features['has_sufficient_history']:
                history_data = temporal_features['pair_interaction_history']
                if history_data.size(0) >= 2:  # Need at least 2 time steps
                    motion_features = extract_causal_motion_features(history_data.unsqueeze(0))
                    result['motion_features'] = motion_features.squeeze(0).clone()
                else:
                    result['motion_features'] = torch.zeros(4, dtype=torch.float32)
            else:
                result['motion_features'] = torch.zeros(4, dtype=torch.float32)
        else:
            result['history_geometric'] = torch.zeros(self.history_length, 7, dtype=torch.float32)
            result['has_sufficient_history'] = False
            result['motion_features'] = torch.zeros(4, dtype=torch.float32)
        
        return result
    
    def update_temporal_buffer(self):
        """Update temporal buffer with all data (for proper temporal modeling)"""
        if not self.temporal_manager:
            return
        
        print("Updating temporal buffer...")
        
        # Sort samples by frame_id for temporal consistency
        sorted_samples = sorted(self.samples, key=_get_frame_id_sort_key)
        
        # Group by frame
        frames = {}
        for sample in sorted_samples:
            frame_id = sample['frame_id']
            if frame_id not in frames:
                frames[frame_id] = []
            frames[frame_id].append(sample)
        
        # Process frames in order
        for frame_id in sorted(frames.keys()):
            frame_samples = frames[frame_id]
            frame_data = []
            
            for sample in frame_samples:
                geometric_features = extract_geometric_features(
                    torch.tensor(sample['person_A_box'], dtype=torch.float32),
                    torch.tensor(sample['person_B_box'], dtype=torch.float32),
                    3760, 480  # JRDB standard dimensions
                )
                
                frame_data.append({
                    'person_A_id': sample['person_A_id'],
                    'person_B_id': sample['person_B_id'],
                    'geometric_features': geometric_features
                })
            
            # Update temporal manager
            self.temporal_manager.update_frame(frame_data, frame_id)
        
        print("Temporal buffer updated!")
    
    def get_class_distribution(self):
        """Get class distribution for balancing"""
        positive = sum(1 for s in self.samples if s['has_interaction'] == 1)
        negative = len(self.samples) - positive
        return {'positive': positive, 'negative': negative, 'total': len(self.samples)}

    @staticmethod
    def clear_feature_cache(data_path, split=None, feature_mode=None, frame_interval=None):
        """
        Utility method to clear cached features

        Args:
            data_path: Path to dataset root
            split: Specific split to clear ('train', 'val', 'test'), or None for all
            feature_mode: Specific mode to clear, or None for all
            frame_interval: Specific interval to clear, or None for all

        Returns:
            Number of cache files removed
        """
        cache_dir = os.path.join(data_path, 'feature_cache')

        if not os.path.exists(cache_dir):
            print(f"No cache directory found at {cache_dir}")
            return 0

        removed_count = 0

        # Get all cache files
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pt')]

        for cache_file in cache_files:
            should_remove = True

            # Filter by split if specified
            if split is not None and f"_{split}_" not in cache_file:
                should_remove = False

            # Filter by feature_mode if specified
            if feature_mode is not None and f"_{feature_mode}_" not in cache_file:
                should_remove = False

            # Filter by frame_interval if specified
            if frame_interval is not None and f"interval{frame_interval}.pt" not in cache_file:
                should_remove = False

            if should_remove:
                cache_path = os.path.join(cache_dir, cache_file)
                try:
                    os.remove(cache_path)
                    print(f"Removed: {cache_file}")
                    removed_count += 1
                except Exception as e:
                    print(f"Failed to remove {cache_file}: {e}")

        print(f"\nTotal cache files removed: {removed_count}")
        return removed_count


def temporal_collate_fn(batch):
    """
    Custom collate function to handle variable-length temporal sequences
    """
    # Separate fields that need special handling
    geometric_features = torch.stack([item['geometric_features'] for item in batch])
    stage1_labels = torch.stack([item['stage1_label'] for item in batch])
    scene_contexts = torch.stack([item['scene_context'] for item in batch])
    motion_features = torch.stack([item['motion_features'] for item in batch])

    # Handle variable-length history_geometric with padding
    history_seqs = [item['history_geometric'] for item in batch]

    # Pad sequences to the same length (pad to max length in batch)
    max_len = max(seq.size(0) for seq in history_seqs)
    padded_histories = []
    seq_lengths = []

    for seq in history_seqs:
        seq_len = seq.size(0)
        seq_lengths.append(seq_len)

        if seq_len < max_len:
            # Pad with zeros at the beginning (causal padding)
            padding = torch.zeros(max_len - seq_len, seq.size(1), dtype=torch.float32)
            padded_seq = torch.cat([padding, seq], dim=0)
        else:
            padded_seq = seq

        padded_histories.append(padded_seq)

    history_geometric = torch.stack(padded_histories)
    seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)

    # Other scalar fields
    has_sufficient_history = [item['has_sufficient_history'] for item in batch]
    person_A_ids = [item['person_A_id'] for item in batch]
    person_B_ids = [item['person_B_id'] for item in batch]
    frame_ids = [item['frame_id'] for item in batch]
    person_A_boxes = torch.stack([item['person_A_box'] for item in batch])
    person_B_boxes = torch.stack([item['person_B_box'] for item in batch])

    return {
        'geometric_features': geometric_features,
        'stage1_label': stage1_labels,
        'scene_context': scene_contexts,
        'motion_features': motion_features,
        'history_geometric': history_geometric,
        'seq_lengths': seq_lengths,  # Add sequence lengths for proper masking
        'has_sufficient_history': has_sufficient_history,
        'person_A_id': person_A_ids,
        'person_B_id': person_B_ids,
        'frame_id': frame_ids,
        'person_A_box': person_A_boxes,
        'person_B_box': person_B_boxes
    }


def create_geometric_data_loaders(data_path, batch_size=32, num_workers=4,
                                history_length=5, use_temporal=False, use_scene_context=True,
                                trainset_split=None, valset_split=None, testset_split=None,
                                use_custom_splits=False, frame_interval=1, feature_mode='legacy'):
    """
    Create data loaders for geometric dual-person interaction detection

    Args:
        data_path: Path to dataset root
        batch_size: Batch size for data loaders
        num_workers: Number of data loading workers
        history_length: Number of historical frames
        use_temporal: Whether to use temporal features
        use_scene_context: Whether to use scene context
        trainset_split: List of scene names for training split
        valset_split: List of scene names for validation split
        testset_split: List of scene names for test split
        use_custom_splits: Whether to use custom scene splits instead of percentage-based
        frame_interval: Frame sampling interval (1=every frame, 5=every 5th frame)
        feature_mode: Feature extraction mode ('legacy', 'both', 'opticalflow_only', 'bboxposition_only')

    Returns:
        train_loader, val_loader, test_loader
    """

    # Create datasets
    train_dataset = GeometricDualPersonDataset(
        data_path, split='train', history_length=history_length,
        use_temporal=use_temporal, use_scene_context=use_scene_context,
        trainset_split=trainset_split, valset_split=valset_split, testset_split=testset_split,
        use_custom_splits=use_custom_splits, frame_interval=frame_interval,
        feature_mode=feature_mode
    )

    val_dataset = GeometricDualPersonDataset(
        data_path, split='val', history_length=history_length,
        use_temporal=use_temporal, use_scene_context=use_scene_context,
        trainset_split=trainset_split, valset_split=valset_split, testset_split=testset_split,
        use_custom_splits=use_custom_splits, frame_interval=frame_interval,
        feature_mode=feature_mode
    )

    test_dataset = GeometricDualPersonDataset(
        data_path, split='test', history_length=history_length,
        use_temporal=use_temporal, use_scene_context=use_scene_context,
        trainset_split=trainset_split, valset_split=valset_split, testset_split=testset_split,
        use_custom_splits=use_custom_splits, frame_interval=frame_interval,
        feature_mode=feature_mode
    )
    
    # Update temporal buffers for proper temporal modeling
    if use_temporal:
        print("Initializing temporal modeling...")
        train_dataset.update_temporal_buffer()
        val_dataset.update_temporal_buffer()
        test_dataset.update_temporal_buffer()
    
    # Choose collate function based on temporal usage
    collate_fn = temporal_collate_fn if use_temporal else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    # Print dataset statistics
    train_dist = train_dataset.get_class_distribution()
    val_dist = val_dataset.get_class_distribution()
    test_dist = test_dataset.get_class_distribution()
    
    print(f"\nDataset Statistics:")
    print(f"Train: {train_dist}")
    print(f"Val: {val_dist}")
    print(f"Test: {test_dist}")
    
    return train_loader, val_loader, test_loader


# Default scene splits (consistent with resnet_stage2_dataset.py)
DEFAULT_TRAINSET_SPLIT = [
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

DEFAULT_VALSET_SPLIT = [
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

DEFAULT_TESTSET_SPLIT = [
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


def create_geometric_data_loaders_with_custom_splits(data_path, batch_size=32, num_workers=4,
                                                   history_length=5, use_temporal=False, use_scene_context=True,
                                                   trainset_split=None, valset_split=None, testset_split=None,
                                                   frame_interval=1, feature_mode='legacy'):
    """
    Create data loaders with custom scene splits (convenience function)

    Args:
        data_path: Path to dataset root
        batch_size: Batch size for data loaders
        num_workers: Number of data loading workers
        history_length: Number of historical frames
        use_temporal: Whether to use temporal features
        use_scene_context: Whether to use scene context
        trainset_split: List of scene names for training split (uses default if None)
        valset_split: List of scene names for validation split (uses default if None)
        testset_split: List of scene names for test split (uses default if None)
        frame_interval: Frame sampling interval (1=every frame, 5=every 5th frame)
        feature_mode: Feature extraction mode ('legacy', 'both', 'opticalflow_only', 'bboxposition_only')

    Returns:
        train_loader, val_loader, test_loader
    """
    # Use default splits if not provided
    if trainset_split is None:
        trainset_split = DEFAULT_TRAINSET_SPLIT
    if valset_split is None:
        valset_split = DEFAULT_VALSET_SPLIT
    if testset_split is None:
        testset_split = DEFAULT_TESTSET_SPLIT

    return create_geometric_data_loaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        history_length=history_length,
        use_temporal=use_temporal,
        use_scene_context=use_scene_context,
        trainset_split=trainset_split,
        valset_split=valset_split,
        testset_split=testset_split,
        use_custom_splits=True,
        frame_interval=frame_interval,
        feature_mode=feature_mode
    )


if __name__ == '__main__':
    # Test dataset
    print("Testing GeometricDualPersonDataset...")
    
    # You would need to provide actual data path
    # data_path = 'D:/1data/imagedata'
    
    # For testing with dummy data
    print("This is a code structure test. Replace with actual data_path for real testing.")
    print("Dataset implementation completed!")