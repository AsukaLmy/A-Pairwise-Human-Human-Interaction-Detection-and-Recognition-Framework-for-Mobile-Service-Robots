#!/usr/bin/env python3
"""
Geometric Flow Feature Extractor
Extracts 3D geometric features using Farneback optical flow for hierarchical classification
"""

import numpy as np
import cv2
import torch
from typing import Dict, Tuple, Optional, Union
from PIL import Image


class GeometricFlowExtractor:
    """
    Extract 10D geometric features for sitting vs not-sitting classification

    Features:
    [0] inter_person_distance / avg_height
    [1] inter_person_distance / avg_width
    [2] flow_magnitude_mean / avg_area
    [3] flow_magnitude_std / avg_area (motion variability)
    [4] vertical_flow_dominance (vertical/horizontal ratio)
    [5] avg_aspect_ratio (height/width of bounding boxes)
    [6] interaction_synchrony (computed separately in train_moe_geometric.py)
    [7] avg_bbox_height / image_height (relative height - sitting lower)
    [8] avg_bbox_bottom / image_height (relative position)
    [9] motion_direction_consistency (directional coherence - walking high, sitting low)
    """

    def __init__(self, flow_bound=20.0, cache_enabled=True):
        """
        Args:
            flow_bound: Maximum expected flow magnitude (for normalization)
            cache_enabled: Whether to cache flow computations
        """
        self.flow_bound = flow_bound
        self.cache_enabled = cache_enabled

        # Farneback parameters (optimized for person tracking)
        self.farneback_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }

        # Flow cache: key -> flow
        self.flow_cache = {}

        print(f"GeometricFlowExtractor initialized:")
        print(f"  Flow method: Farneback (OpenCV)")
        print(f"  Flow bound: {flow_bound}")
        print(f"  Cache enabled: {cache_enabled}")

    def compute_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow using Farneback method

        Args:
            prev_frame: Previous frame (grayscale, H x W)
            curr_frame: Current frame (grayscale, H x W)

        Returns:
            flow: Optical flow field (H x W x 2), where flow[:,:,0] is x-flow, flow[:,:,1] is y-flow
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, curr_frame,
            None,
            **self.farneback_params
        )
        return flow

    def compute_motion_direction_consistency(self,
                                             flow: np.ndarray,
                                             person_A_box: np.ndarray,
                                             person_B_box: np.ndarray,
                                             magnitude_threshold: float = 1.0) -> float:
        """
        Compute motion direction consistency in the combined person regions

        High consistency (near 1.0) indicates all motion is toward the same direction (walking)
        Low consistency (near 0.0) indicates scattered motion directions (sitting, random gestures)

        Args:
            flow: Optical flow field [H, W, 2]
            person_A_box: Bounding box [x, y, w, h]
            person_B_box: Bounding box [x, y, w, h]
            magnitude_threshold: Minimum flow magnitude to consider (filter out noise)

        Returns:
            consistency: Direction consistency score [0, 1]
        """
        # Create masks for both persons
        mask_A = self._create_box_mask(person_A_box, flow.shape[:2])
        mask_B = self._create_box_mask(person_B_box, flow.shape[:2])
        combined_mask = np.maximum(mask_A, mask_B)

        # Extract flow in person regions
        flow_x = flow[:, :, 0][combined_mask > 0]
        flow_y = flow[:, :, 1][combined_mask > 0]

        if len(flow_x) == 0:
            return 0.0

        # Compute flow magnitudes
        flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # Filter out low-magnitude motion (noise)
        valid_mask = flow_magnitude > magnitude_threshold

        if np.sum(valid_mask) < 10:  # Need at least 10 valid points
            return 0.0

        # Extract valid flow vectors
        flow_x_valid = flow_x[valid_mask]
        flow_y_valid = flow_y[valid_mask]

        # Compute motion angles for each valid pixel
        angles = np.arctan2(flow_y_valid, flow_x_valid)  # Range: [-π, π]

        # Method: Compute circular variance
        # Convert angles to unit vectors and average
        mean_cos = np.mean(np.cos(angles))
        mean_sin = np.mean(np.sin(angles))

        # Mean resultant length R (measure of concentration)
        # R = 1: all angles identical (perfect consistency)
        # R = 0: angles uniformly distributed (no consistency)
        R = np.sqrt(mean_cos**2 + mean_sin**2)

        # R is already in [0, 1], perfect for our use
        consistency = float(R)

        return consistency

    def extract_geometric_features(self,
                                   prev_image: Union[Image.Image, np.ndarray],
                                   curr_image: Union[Image.Image, np.ndarray],
                                   person_A_box: Union[torch.Tensor, np.ndarray, list],
                                   person_B_box: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
        """
        Extract 10D geometric features for a person pair

        Args:
            prev_image: Previous frame (PIL Image or numpy array)
            curr_image: Current frame (PIL Image or numpy array)
            person_A_box: Bounding box for person A [x, y, w, h]
            person_B_box: Bounding box for person B [x, y, w, h]

        Returns:
            features: 10D geometric feature vector [
                [0] inter_distance / avg_height,
                [1] inter_distance / avg_width,
                [2] flow_magnitude_mean / avg_area,
                [3] flow_magnitude_std / avg_area,
                [4] vertical_flow_dominance,
                [5] avg_aspect_ratio,
                [6] interaction_synchrony (computed separately),
                [7] avg_bbox_height / image_height,
                [8] avg_bbox_bottom / image_height,
                [9] motion_direction_consistency
            ]
        """
        # Convert images to grayscale numpy arrays
        if isinstance(prev_image, Image.Image):
            prev_frame = np.array(prev_image.convert('L'))
        else:
            prev_frame = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY) if len(prev_image.shape) == 3 else prev_image

        if isinstance(curr_image, Image.Image):
            curr_frame = np.array(curr_image.convert('L'))
        else:
            curr_frame = cv2.cvtColor(curr_image, cv2.COLOR_RGB2GRAY) if len(curr_image.shape) == 3 else curr_image

        # Compute optical flow
        flow = self.compute_optical_flow(prev_frame, curr_frame)

        # Convert boxes to numpy
        if isinstance(person_A_box, torch.Tensor):
            person_A_box = person_A_box.cpu().numpy()
        if isinstance(person_B_box, torch.Tensor):
            person_B_box = person_B_box.cpu().numpy()

        person_A_box = np.array(person_A_box, dtype=np.float32)
        person_B_box = np.array(person_B_box, dtype=np.float32)

        # === Calculate geometric measurements ===

        # Person centers
        center_A_x = person_A_box[0] + person_A_box[2] / 2.0
        center_A_y = person_A_box[1] + person_A_box[3] / 2.0
        center_B_x = person_B_box[0] + person_B_box[2] / 2.0
        center_B_y = person_B_box[1] + person_B_box[3] / 2.0

        # Euclidean distance between person centers
        inter_person_distance = np.sqrt((center_A_x - center_B_x)**2 + (center_A_y - center_B_y)**2)

        # Average box dimensions
        avg_height = (person_A_box[3] + person_B_box[3]) / 2.0
        avg_width = (person_A_box[2] + person_B_box[2]) / 2.0

        # Average box area
        area_A = person_A_box[2] * person_A_box[3]
        area_B = person_B_box[2] * person_B_box[3]
        avg_area = (area_A + area_B) / 2.0

        # === Calculate flow magnitude in person regions ===

        # Create masks for both persons
        mask_A = self._create_box_mask(person_A_box, flow.shape[:2])
        mask_B = self._create_box_mask(person_B_box, flow.shape[:2])
        combined_mask = np.maximum(mask_A, mask_B)

        # Extract flow in person regions
        flow_x = flow[:, :, 0][combined_mask > 0]
        flow_y = flow[:, :, 1][combined_mask > 0]

        if len(flow_x) > 0:
            # Compute magnitude (using absolute value for horizontal flow symmetry)
            flow_x_abs = np.abs(flow_x)
            magnitude = np.sqrt(flow_x_abs**2 + flow_y**2)
            avg_magnitude = np.mean(magnitude)
            std_magnitude = np.std(magnitude)

            # Compute vertical vs horizontal flow dominance
            avg_vertical_flow = np.mean(np.abs(flow_y))
            avg_horizontal_flow = np.mean(flow_x_abs)
            vertical_dominance = avg_vertical_flow / max(avg_horizontal_flow, 1e-5)
        else:
            avg_magnitude = 0.0
            std_magnitude = 0.0
            vertical_dominance = 0.0

        # === Calculate bounding box aspect ratios ===
        aspect_ratio_A = person_A_box[3] / max(person_A_box[2], 1e-5)  # height / width
        aspect_ratio_B = person_B_box[3] / max(person_B_box[2], 1e-5)
        avg_aspect_ratio = (aspect_ratio_A + aspect_ratio_B) / 2.0

        # === Get image dimensions ===
        image_height = float(curr_frame.shape[0])
        image_width = float(curr_frame.shape[1])

        # === Calculate NEW position features (Feature 7-8) ===
        # Symmetric aggregation: average of both persons
        height_A = person_A_box[3]
        height_B = person_B_box[3]

        bottom_A = person_A_box[1] + person_A_box[3]  # y + height
        bottom_B = person_B_box[1] + person_B_box[3]

        avg_bbox_height = (height_A + height_B) / 2.0
        avg_bbox_bottom = (bottom_A + bottom_B) / 2.0

        # === Calculate NEW motion direction consistency (Feature 9) ===
        direction_consistency = self.compute_motion_direction_consistency(
            flow, person_A_box, person_B_box
        )

        # === Construct 10D feature vector ===

        # [0] Inter-person distance / average height ratio
        feature_0 = inter_person_distance / max(avg_height, 1e-5)

        # [1] Inter-person distance / average width ratio
        feature_1 = inter_person_distance / max(avg_width, 1e-5)

        # [2] Flow mean intensity / average area ratio
        feature_2 = avg_magnitude / max(avg_area, 1e-5)

        # [3] Flow std intensity / average area ratio (motion variability)
        feature_3 = std_magnitude / max(avg_area, 1e-5)

        # [4] Vertical flow dominance (vertical/horizontal ratio)
        feature_4 = vertical_dominance

        # [5] Average bounding box aspect ratio (height/width)
        feature_5 = avg_aspect_ratio

        # [6] Interaction synchrony (NOTE: computed separately in train_moe_geometric.py)
        # This is a placeholder - actual value added later
        # feature_6 = interaction_synchrony (not computed here)

        # [7] Average bbox height / image height (relative height - sitting lower)
        feature_7 = avg_bbox_height / max(image_height, 1e-5)

        # [8] Average bbox bottom / image height (relative position)
        feature_8 = avg_bbox_bottom / max(image_height, 1e-5)

        # [9] Motion direction consistency (walking high, sitting low)
        feature_9 = direction_consistency

        # Return as torch tensor (note: feature_6 added later, so this returns 9D)
        features = torch.tensor([
            feature_0, feature_1, feature_2, feature_3, feature_4, feature_5,
            feature_7, feature_8, feature_9
        ], dtype=torch.float32)

        # Clamp to reasonable range
        features = torch.clamp(features, -10.0, 10.0)

        return features

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

    def clear_cache(self):
        """Clear the flow cache"""
        self.flow_cache.clear()


if __name__ == '__main__':
    print("Testing Geometric Flow Extractor...")

    # Create dummy test images
    height, width = 480, 640

    # Frame 1: two people
    frame1 = np.ones((height, width, 3), dtype=np.uint8) * 128
    frame1[100:200, 100:200] = 255  # Person A
    frame1[100:200, 400:500] = 200  # Person B

    # Frame 2: person A moved right
    frame2 = np.ones((height, width, 3), dtype=np.uint8) * 128
    frame2[100:200, 120:220] = 255  # Person A moved
    frame2[100:200, 400:500] = 200  # Person B (static)

    # Test extraction
    extractor = GeometricFlowExtractor()

    person_A_box = [100, 100, 100, 100]
    person_B_box = [400, 100, 100, 100]

    features = extractor.extract_geometric_features(
        Image.fromarray(frame1),
        Image.fromarray(frame2),
        person_A_box,
        person_B_box
    )

    print(f"\nExtracted {len(features)}D geometric features:")
    print(f"  [0] distance/height ratio: {features[0]:.4f}")
    print(f"  [1] distance/width ratio: {features[1]:.4f}")
    print(f"  [2] flow_mean/area ratio: {features[2]:.4f}")
    print(f"  [3] flow_std/area ratio: {features[3]:.4f}")
    print(f"  [4] vertical flow dominance: {features[4]:.4f}")
    print(f"  [5] avg aspect ratio: {features[5]:.4f}")
    print(f"  [6] avg height / image_height: {features[6]:.4f}")
    print(f"  [7] avg bottom / image_height: {features[7]:.4f}")
    print(f"  [8] motion direction consistency: {features[8]:.4f}")

    print("\n[PASS] Geometric Flow Extractor test completed!")
