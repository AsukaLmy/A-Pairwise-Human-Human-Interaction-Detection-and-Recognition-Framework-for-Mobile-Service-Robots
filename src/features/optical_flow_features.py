#!/usr/bin/env python3
"""
Optical Flow Feature Extraction for Social Interaction Classification
Provides motion-based features to improve Walking/Standing/Sitting Together classification
"""

import numpy as np
import cv2
import torch
from typing import Dict, Tuple, Optional, Union
from PIL import Image
import warnings


class OpticalFlowExtractor:
    """
    Extract optical flow and motion statistics from consecutive frames
    Helps distinguish Walking Together (high motion) from Standing/Sitting (low motion)
    """

    def __init__(self, method='farneback', flow_bound=20.0, cache_enabled=True,
                 compensate_ego_motion=True, use_blockwise_compensation=False,
                 num_blocks=8):
        """
        Args:
            method: Optical flow method ('farneback' or 'lucas_kanade')
            flow_bound: Maximum expected flow magnitude (for normalization)
            cache_enabled: Whether to cache flow computations
            compensate_ego_motion: Whether to compensate for camera/robot motion
            use_blockwise_compensation: Whether to use block-wise compensation (for panoramic cameras)
            num_blocks: Number of horizontal blocks for block-wise compensation
        """
        self.method = method
        self.flow_bound = flow_bound
        self.cache_enabled = cache_enabled
        self.compensate_ego_motion = compensate_ego_motion
        self.use_blockwise_compensation = use_blockwise_compensation
        self.num_blocks = num_blocks

        # Farneback parameters (tuned for person tracking)
        self.farneback_params = {
            'pyr_scale': 0.5,        # Pyramid scale
            'levels': 3,              # Number of pyramid levels
            'winsize': 15,            # Averaging window size
            'iterations': 3,          # Iterations at each pyramid level
            'poly_n': 5,              # Neighborhood size
            'poly_sigma': 1.2,        # Gaussian sigma
            'flags': 0
        }

        # Lucas-Kanade parameters
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }

        # Frame cache: (image_path, box) -> flow
        self.flow_cache = {}

        comp_info = "disabled"
        if compensate_ego_motion:
            if use_blockwise_compensation:
                comp_info = f"block-wise ({num_blocks} blocks)"
            else:
                comp_info = "global median"

        print(f"Optical Flow Extractor initialized: method={method}, flow_bound={flow_bound}, "
              f"ego-motion compensation={comp_info}")

    def compute_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow between two frames

        Args:
            prev_frame: Previous frame (grayscale, H x W)
            curr_frame: Current frame (grayscale, H x W)

        Returns:
            flow: Optical flow field (H x W x 2), where flow[:,:,0] is x-flow, flow[:,:,1] is y-flow
        """
        if self.method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame,
                None,
                **self.farneback_params
            )
        elif self.method == 'lucas_kanade':
            # For Lucas-Kanade, we need feature points
            # Use good features to track
            p0 = cv2.goodFeaturesToTrack(prev_frame, maxCorners=200,
                                         qualityLevel=0.01, minDistance=7, blockSize=7)

            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, p0, None, **self.lk_params)

                # Create dense flow field from sparse points (interpolation)
                flow = self._sparse_to_dense_flow(p0, p1, st, prev_frame.shape)
            else:
                # No features found, return zero flow
                flow = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 2), dtype=np.float32)
        else:
            raise ValueError(f"Unknown optical flow method: {self.method}")

        return flow

    def _sparse_to_dense_flow(self, p0, p1, st, shape):
        """Convert sparse optical flow to dense flow field using interpolation"""
        # This is a simple implementation - in practice, Farneback is better for dense flow
        flow = np.zeros((shape[0], shape[1], 2), dtype=np.float32)

        # Filter good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) == 0:
            return flow

        # Calculate displacement
        displacement = good_new - good_old

        # Simple approach: fill flow with nearest neighbor interpolation
        # For simplicity, we just use mean flow (can be improved with interpolation)
        mean_flow = np.mean(displacement, axis=0)
        flow[:, :, 0] = mean_flow[0]
        flow[:, :, 1] = mean_flow[1]

        return flow

    def estimate_background_motion(self, flow: np.ndarray,
                                   foreground_masks: list = None) -> np.ndarray:
        """
        Estimate camera/robot ego-motion from background regions

        Args:
            flow: Full optical flow field (H x W x 2)
            foreground_masks: List of binary masks for foreground objects (persons)
                             to exclude from background estimation

        Returns:
            background_flow: Estimated background motion vector [dx, dy]
        """
        # Create background mask (everything except foreground objects)
        if foreground_masks is not None and len(foreground_masks) > 0:
            background_mask = np.ones(flow.shape[:2], dtype=np.uint8)
            for fg_mask in foreground_masks:
                background_mask = background_mask * (1 - fg_mask)
        else:
            # No foreground specified, use entire image
            background_mask = np.ones(flow.shape[:2], dtype=np.uint8)

        # Extract background flow
        bg_flow_x = flow[:, :, 0][background_mask > 0]
        bg_flow_y = flow[:, :, 1][background_mask > 0]

        if len(bg_flow_x) == 0:
            # No background pixels, assume zero ego-motion
            return np.array([0.0, 0.0], dtype=np.float32)

        # Use median to robustly estimate background motion
        # (median is robust to outliers/moving objects in background)
        bg_motion_x = np.median(bg_flow_x)
        bg_motion_y = np.median(bg_flow_y)

        return np.array([bg_motion_x, bg_motion_y], dtype=np.float32)

    def compensate_ego_motion_in_flow(self, flow: np.ndarray,
                                     background_motion: np.ndarray) -> np.ndarray:
        """
        Subtract background motion from optical flow to get relative motion

        Args:
            flow: Optical flow field (H x W x 2)
            background_motion: Background motion vector [dx, dy]

        Returns:
            compensated_flow: Flow with ego-motion removed (H x W x 2)
        """
        compensated_flow = flow.copy()
        compensated_flow[:, :, 0] -= background_motion[0]
        compensated_flow[:, :, 1] -= background_motion[1]
        return compensated_flow

    def compute_block_statistics(self, flow: np.ndarray, background_mask: np.ndarray,
                                 num_blocks: Optional[int] = None) -> list:
        """
        Compute median flow vector for each horizontal block of the image
        (For handling panoramic camera with spatially-varying background motion)

        Args:
            flow: Optical flow field (H x W x 2)
            background_mask: Binary mask for background regions (H x W)
            num_blocks: Number of horizontal blocks (defaults to self.num_blocks)

        Returns:
            block_stats: List of dicts with 'col', 'bounds', 'median_flow', 'num_pixels'
        """
        if num_blocks is None:
            num_blocks = self.num_blocks

        height, width = flow.shape[:2]
        block_width = width // num_blocks

        block_stats = []

        for col in range(num_blocks):
            # Define block region (horizontal strip)
            x1 = col * block_width
            x2 = (col + 1) * block_width if col < num_blocks - 1 else width

            # Extract block flow and mask
            block_flow = flow[:, x1:x2, :]
            block_mask = background_mask[:, x1:x2]

            # Extract background flow in this block
            bg_flow_x = block_flow[:, :, 0][block_mask > 0]
            bg_flow_y = block_flow[:, :, 1][block_mask > 0]

            if len(bg_flow_x) > 10:  # Require at least 10 background pixels
                median_flow = np.array([np.median(bg_flow_x), np.median(bg_flow_y)], dtype=np.float32)
                num_pixels = len(bg_flow_x)
            else:
                # Not enough background pixels, use zero compensation
                median_flow = np.array([0.0, 0.0], dtype=np.float32)
                num_pixels = 0

            block_stats.append({
                'col': col,
                'bounds': (x1, 0, x2, height),  # (x1, y1, x2, y2)
                'median_flow': median_flow,
                'num_pixels': num_pixels
            })

        return block_stats

    def get_block_index_for_person(self, person_box: np.ndarray, num_blocks: Optional[int] = None,
                                   image_width: int = 3760) -> int:
        """
        Get the block index for a person based on their bounding box center

        Args:
            person_box: Bounding box [x, y, w, h]
            num_blocks: Number of horizontal blocks (defaults to self.num_blocks)
            image_width: Image width

        Returns:
            block_idx: Index of the block containing the person's center
        """
        if num_blocks is None:
            num_blocks = self.num_blocks

        # Calculate person center X coordinate
        person_center_x = person_box[0] + person_box[2] / 2

        # Calculate block index
        block_width = image_width / num_blocks
        block_idx = int(person_center_x / block_width)

        # Clamp to valid range
        block_idx = min(max(block_idx, 0), num_blocks - 1)

        return block_idx

    def extract_motion_features(self, flow: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Extract statistical motion features from optical flow

        Args:
            flow: Optical flow field (H x W x 2)
            mask: Optional binary mask to focus on specific region (H x W)

        Returns:
            features: 8D motion feature vector:
                [0] magnitude_mean: Average motion magnitude
                [1] magnitude_std: Motion magnitude variance
                [2] magnitude_max: Maximum motion magnitude
                [3] direction_consistency: How consistent the motion direction is
                [4] horizontal_motion: Average horizontal motion (positive=right, negative=left)
                [5] vertical_motion: Average vertical motion (positive=down, negative=up)
                [6] motion_energy: Total motion energy
                [7] stationary_ratio: Ratio of pixels with near-zero motion
        """
        if mask is not None:
            # Apply mask
            flow_x = flow[:, :, 0] * mask
            flow_y = flow[:, :, 1] * mask
            valid_pixels = np.sum(mask > 0)
        else:
            flow_x = flow[:, :, 0]
            flow_y = flow[:, :, 1]
            valid_pixels = flow.shape[0] * flow.shape[1]

        if valid_pixels == 0:
            # No valid pixels, return zero features
            return torch.zeros(8, dtype=torch.float32)

        # Compute magnitude and angle
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        angle = np.arctan2(flow_y, flow_x)  # Range: [-pi, pi]

        # Feature 0-2: Magnitude statistics
        magnitude_mean = np.mean(magnitude)
        magnitude_std = np.std(magnitude)
        magnitude_max = np.max(magnitude)

        # Normalize by flow_bound
        magnitude_mean_norm = magnitude_mean / self.flow_bound
        magnitude_std_norm = magnitude_std / self.flow_bound
        magnitude_max_norm = magnitude_max / self.flow_bound

        # Feature 3: Direction consistency (using circular variance)
        # High consistency = vectors point in similar direction (walking together)
        # Low consistency = random directions (standing, fidgeting)
        cos_angles = np.cos(angle)
        sin_angles = np.sin(angle)
        mean_cos = np.mean(cos_angles)
        mean_sin = np.mean(sin_angles)
        direction_consistency = np.sqrt(mean_cos**2 + mean_sin**2)  # Range: [0, 1]

        # Feature 4-5: Average directional motion
        horizontal_motion = np.mean(flow_x) / self.flow_bound
        vertical_motion = np.mean(flow_y) / self.flow_bound

        # Feature 6: Total motion energy (sum of squared magnitudes)
        motion_energy = np.sum(magnitude**2) / (valid_pixels * self.flow_bound**2)

        # Feature 7: Stationary ratio (percentage of nearly-stationary pixels)
        stationary_threshold = 0.5  # pixels with magnitude < 0.5 are considered stationary
        stationary_ratio = np.sum(magnitude < stationary_threshold) / valid_pixels

        # Combine into feature vector
        features = torch.tensor([
            magnitude_mean_norm,
            magnitude_std_norm,
            magnitude_max_norm,
            direction_consistency,
            horizontal_motion,
            vertical_motion,
            motion_energy,
            stationary_ratio
        ], dtype=torch.float32)

        # Clamp to reasonable range
        features = torch.clamp(features, -10.0, 10.0)

        return features

    def extract_person_pair_optical_flow(self,
                                         prev_image: Union[Image.Image, np.ndarray],
                                         curr_image: Union[Image.Image, np.ndarray],
                                         person_A_box: Union[torch.Tensor, np.ndarray, list],
                                         person_B_box: Union[torch.Tensor, np.ndarray, list],
                                         image_width: int = 3760,
                                         image_height: int = 480) -> Dict[str, torch.Tensor]:
        """
        Extract optical flow features for a person pair

        Args:
            prev_image: Previous frame (PIL Image or numpy array)
            curr_image: Current frame (PIL Image or numpy array)
            person_A_box: Bounding box for person A [x, y, w, h]
            person_B_box: Bounding box for person B [x, y, w, h]
            image_width: Image width (default 3760 for JRDB)
            image_height: Image height (default 480 for JRDB)

        Returns:
            Dictionary with:
                - person_A_flow_features: 8D flow features for person A
                - person_B_flow_features: 8D flow features for person B
                - pair_flow_features: 8D flow features for the pair region
                - combined_flow_features: 8D combined features (for easy integration)
        """
        # Convert images to numpy arrays and grayscale
        if isinstance(prev_image, Image.Image):
            prev_frame = np.array(prev_image.convert('L'))
        else:
            prev_frame = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY) if len(prev_image.shape) == 3 else prev_image

        if isinstance(curr_image, Image.Image):
            curr_frame = np.array(curr_image.convert('L'))
        else:
            curr_frame = cv2.cvtColor(curr_image, cv2.COLOR_RGB2GRAY) if len(curr_image.shape) == 3 else curr_image

        # Ensure consistent size
        if prev_frame.shape != curr_frame.shape:
            warnings.warn(f"Frame size mismatch: prev {prev_frame.shape} vs curr {curr_frame.shape}")
            # Resize to match
            curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))

        # Compute optical flow
        flow = self.compute_optical_flow(prev_frame, curr_frame)

        # Convert boxes to numpy if needed
        if isinstance(person_A_box, torch.Tensor):
            person_A_box = person_A_box.cpu().numpy()
        if isinstance(person_B_box, torch.Tensor):
            person_B_box = person_B_box.cpu().numpy()

        person_A_box = np.array(person_A_box, dtype=np.int32)
        person_B_box = np.array(person_B_box, dtype=np.int32)

        # Create person masks
        person_A_mask = self._create_box_mask(person_A_box, flow.shape[:2])
        person_B_mask = self._create_box_mask(person_B_box, flow.shape[:2])

        # === EGO-MOTION COMPENSATION ===
        if self.compensate_ego_motion:
            if self.use_blockwise_compensation:
                # === BLOCK-WISE COMPENSATION (for panoramic cameras) ===
                # Create background mask (exclude persons)
                background_mask = np.ones(flow.shape[:2], dtype=np.uint8)
                background_mask[person_A_mask > 0] = 0
                background_mask[person_B_mask > 0] = 0

                # Compute block statistics
                block_stats = self.compute_block_statistics(flow, background_mask, num_blocks=self.num_blocks)

                # Get block indices for each person
                block_idx_A = self.get_block_index_for_person(person_A_box, self.num_blocks, image_width)
                block_idx_B = self.get_block_index_for_person(person_B_box, self.num_blocks, image_width)

                # Get compensation vectors for each person
                compensation_A = block_stats[block_idx_A]['median_flow']
                compensation_B = block_stats[block_idx_B]['median_flow']

                # Create compensated flow
                flow_compensated = flow.copy()

                # Apply compensation to person A's region
                y1_A = max(0, int(person_A_box[1]))
                y2_A = min(flow.shape[0], int(person_A_box[1] + person_A_box[3]))
                x1_A = max(0, int(person_A_box[0]))
                x2_A = min(flow.shape[1], int(person_A_box[0] + person_A_box[2]))
                flow_compensated[y1_A:y2_A, x1_A:x2_A, 0] -= compensation_A[0]
                flow_compensated[y1_A:y2_A, x1_A:x2_A, 1] -= compensation_A[1]

                # Apply compensation to person B's region
                y1_B = max(0, int(person_B_box[1]))
                y2_B = min(flow.shape[0], int(person_B_box[1] + person_B_box[3]))
                x1_B = max(0, int(person_B_box[0]))
                x2_B = min(flow.shape[1], int(person_B_box[0] + person_B_box[2]))
                flow_compensated[y1_B:y2_B, x1_B:x2_B, 0] -= compensation_B[0]
                flow_compensated[y1_B:y2_B, x1_B:x2_B, 1] -= compensation_B[1]

                # For return info: use average of the two compensations
                background_motion = (compensation_A + compensation_B) / 2.0

            else:
                # === GLOBAL COMPENSATION (original method) ===
                # Estimate background motion (camera/robot movement)
                background_motion = self.estimate_background_motion(
                    flow,
                    foreground_masks=[person_A_mask, person_B_mask]
                )

                # Compensate for ego-motion: subtract background motion from flow
                flow_compensated = self.compensate_ego_motion_in_flow(flow, background_motion)

                # Debug info (optional)
                bg_magnitude = np.linalg.norm(background_motion)
                if bg_magnitude > 1.0:  # Only print if significant camera motion
                    # print(f"[EGO-MOTION] Background motion: [{background_motion[0]:.2f}, {background_motion[1]:.2f}], "
                    #       f"magnitude: {bg_magnitude:.2f}")
                    pass
        else:
            # No compensation, use original flow
            flow_compensated = flow
            background_motion = np.array([0.0, 0.0], dtype=np.float32)
        # ======================================

        # Extract flow features for person A (using compensated flow)
        person_A_flow_features = self.extract_motion_features(flow_compensated, person_A_mask)

        # Extract flow features for person B (using compensated flow)
        person_B_flow_features = self.extract_motion_features(flow_compensated, person_B_mask)

        # Extract flow features for the pair region (union of both boxes)
        pair_mask = np.maximum(person_A_mask, person_B_mask)
        pair_flow_features = self.extract_motion_features(flow_compensated, pair_mask)

        # Combined features: average of individual and pair
        # This captures both individual motion and joint motion patterns
        combined_flow_features = (person_A_flow_features + person_B_flow_features + pair_flow_features) / 3.0

        return {
            'person_A_flow_features': person_A_flow_features,
            'person_B_flow_features': person_B_flow_features,
            'pair_flow_features': pair_flow_features,
            'combined_flow_features': combined_flow_features,
            'background_motion': torch.tensor(background_motion if self.compensate_ego_motion else [0.0, 0.0],
                                            dtype=torch.float32)  # Return for analysis
        }

    def _create_box_mask(self, box: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a binary mask for a bounding box

        Args:
            box: [x, y, w, h]
            shape: (height, width)

        Returns:
            mask: Binary mask (H x W)
        """
        mask = np.zeros(shape, dtype=np.uint8)

        x, y, w, h = box
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(shape[1], int(x + w))
        y2 = min(shape[0], int(y + h))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

        return mask

    def get_feature_dim(self) -> int:
        """Get the dimension of extracted flow features"""
        return 8

    def get_feature_names(self) -> list:
        """Get names of flow features for interpretability"""
        return [
            'magnitude_mean',
            'magnitude_std',
            'magnitude_max',
            'direction_consistency',
            'horizontal_motion',
            'vertical_motion',
            'motion_energy',
            'stationary_ratio'
        ]

    def clear_cache(self):
        """Clear the flow cache"""
        self.flow_cache.clear()


# Convenience function for quick usage
def extract_optical_flow_features(prev_image, curr_image, person_A_box, person_B_box,
                                  method='farneback', **kwargs) -> torch.Tensor:
    """
    Quick function to extract optical flow features for a person pair

    Returns:
        8D tensor of combined optical flow features
    """
    extractor = OpticalFlowExtractor(method=method)
    result = extractor.extract_person_pair_optical_flow(
        prev_image, curr_image, person_A_box, person_B_box, **kwargs
    )
    return result['combined_flow_features']


if __name__ == '__main__':
    print("Testing Optical Flow Feature Extraction...")

    # Create dummy images for testing
    print("\n1. Creating test frames...")
    height, width = 480, 640

    # Frame 1: static scene
    frame1 = np.ones((height, width, 3), dtype=np.uint8) * 128
    # Add person A (moving right)
    frame1[100:200, 100:200] = 255

    # Frame 2: person moved
    frame2 = np.ones((height, width, 3), dtype=np.uint8) * 128
    frame2[100:200, 120:220] = 255  # Moved 20 pixels right

    # Add person B (stationary)
    frame1[100:200, 400:500] = 200
    frame2[100:200, 400:500] = 200

    print(f"Frame shapes: {frame1.shape}, {frame2.shape}")

    # Test optical flow extraction
    print("\n2. Testing optical flow extraction...")
    extractor = OpticalFlowExtractor(method='farneback')

    person_A_box = [100, 100, 100, 100]  # Moving person
    person_B_box = [400, 100, 100, 100]  # Stationary person

    flow_features = extractor.extract_person_pair_optical_flow(
        Image.fromarray(frame1),
        Image.fromarray(frame2),
        person_A_box,
        person_B_box,
        image_width=width,
        image_height=height
    )

    print(f"\n3. Extracted features:")
    print(f"Feature dimension: {extractor.get_feature_dim()}")
    print(f"Feature names: {extractor.get_feature_names()}")
    print(f"\nPerson A (moving) flow features:\n{flow_features['person_A_flow_features']}")
    print(f"\nPerson B (stationary) flow features:\n{flow_features['person_B_flow_features']}")
    print(f"\nCombined flow features:\n{flow_features['combined_flow_features']}")

    # Verify person A has higher motion than person B
    person_A_magnitude = flow_features['person_A_flow_features'][0].item()
    person_B_magnitude = flow_features['person_B_flow_features'][0].item()

    print(f"\n4. Validation:")
    print(f"Person A motion magnitude: {person_A_magnitude:.4f}")
    print(f"Person B motion magnitude: {person_B_magnitude:.4f}")

    if person_A_magnitude > person_B_magnitude:
        print("✅ SUCCESS: Moving person has higher motion magnitude!")
    else:
        print("⚠️  WARNING: Expected person A to have higher motion")

    print("\n✅ Optical Flow Feature Extraction test completed!")
