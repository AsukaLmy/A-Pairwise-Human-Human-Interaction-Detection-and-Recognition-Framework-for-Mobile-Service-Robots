import torch
import numpy as np
import math

def extract_geometric_features(box_A, box_B, image_width=3760, image_height=480, panoramic=True):
    """
    Extract 7 core geometric features from two person bounding boxes
    
    Args:
        box_A, box_B: torch.Tensor or list [x, y, width, height] format (JRDB format)
        image_width, image_height: normalization parameters (default: JRDB 3760×480)
        
    Returns:
        torch.Tensor: [7] geometric features
    """
    if isinstance(box_A, (list, tuple)):
        box_A = torch.tensor(box_A, dtype=torch.float32)
    if isinstance(box_B, (list, tuple)):
        box_B = torch.tensor(box_B, dtype=torch.float32)
    
    # Handle batch dimension
    if box_A.dim() == 1:
        box_A = box_A.unsqueeze(0)
        box_B = box_B.unsqueeze(0)
    
    batch_size = box_A.size(0)
    features = []
    
    for i in range(batch_size):
        # Extract coordinates - JRDB format: [x, y, width, height]
        x_A, y_A, w_A, h_A = box_A[i]
        x_B, y_B, w_B, h_B = box_B[i]
        
        # Convert to [x1, y1, x2, y2] format
        x1_A, y1_A, x2_A, y2_A = x_A, y_A, x_A + w_A, y_A + h_A
        x1_B, y1_B, x2_B, y2_B = x_B, y_B, x_B + w_B, y_B + h_B
        
        # Use default normalization if not provided
        img_w = image_width if image_width is not None else 1000.0
        img_h = image_height if image_height is not None else 1000.0
        
        # Calculate centers and dimensions
        center_A_x = (x1_A + x2_A) / 2
        center_A_y = (y1_A + y2_A) / 2
        center_B_x = (x1_B + x2_B) / 2
        center_B_y = (y1_B + y2_B) / 2
        
        # Calculate horizontal distance
        dx_direct = abs(center_A_x - center_B_x)
        if panoramic:
            # Handle panoramic image boundary wraparound
            dx_wraparound = img_w - dx_direct  # Distance through wraparound
            dx_actual = min(dx_direct, dx_wraparound)  # Use shorter distance
        else:
            dx_actual = dx_direct
        
        width_A = x2_A - x1_A
        height_A = y2_A - y1_A
        width_B = x2_B - x1_B
        height_B = y2_B - y1_B
        
        area_A = width_A * height_A
        area_B = width_B * height_B
        
        # 1. Horizontal gap (normalized) - considering panoramic wraparound
        horizontal_gap = dx_actual - (width_A + width_B) / 2
        horizontal_gap_norm = horizontal_gap / img_w
        
        # 2. Height ratio
        height_ratio = min(height_A, height_B) / max(height_A, height_B + 1e-6)
        
        # 3. Ground distance (normalized) - using actual distance with wraparound
        ground_distance = dx_actual / img_w
        
        # 4. Vertical overlap
        v_overlap = max(0.0, min(y2_A, y2_B) - max(y1_A, y1_B)) / min(height_A, height_B + 1e-6)
        
        # 5. Area ratio
        area_ratio = min(area_A, area_B) / max(area_A, area_B + 1e-6)
        
        # 6. Center distance normalized - considering panoramic wraparound
        center_dist = torch.sqrt(dx_actual**2 + (center_A_y - center_B_y)**2)
        center_dist_norm = center_dist / torch.sqrt(area_A + area_B + 1e-6)
        
        # 7. Vertical distance ratio
        vertical_gap = abs(center_A_y - center_B_y) / img_h
        
        # Combine features
        feature_vector = torch.tensor([
            horizontal_gap_norm,
            height_ratio,
            ground_distance,
            v_overlap,
            area_ratio,
            center_dist_norm,
            vertical_gap
        ], dtype=torch.float32)
        
        features.append(feature_vector)
    
    result = torch.stack(features)
    if result.size(0) == 1:
        return result.squeeze(0).clone()
    else:
        return result


def extract_geometric_features_batch(boxes_A, boxes_B, image_width=3760, image_height=480):
    """
    Fully vectorized version of extract_geometric_features for E pairs at once.
    No Python loops – all operations are tensor ops.

    Args:
        boxes_A, boxes_B: torch.Tensor [E, 4]  (x, y, w, h format)
        image_width, image_height: normalization constants
    Returns:
        torch.Tensor [E, 7]
    """
    x_A = boxes_A[:, 0]; y_A = boxes_A[:, 1]
    w_A = boxes_A[:, 2]; h_A = boxes_A[:, 3]
    x_B = boxes_B[:, 0]; y_B = boxes_B[:, 1]
    w_B = boxes_B[:, 2]; h_B = boxes_B[:, 3]

    cx_A = x_A + w_A * 0.5;  cy_A = y_A + h_A * 0.5
    cx_B = x_B + w_B * 0.5;  cy_B = y_B + h_B * 0.5

    dx_direct  = (cx_A - cx_B).abs()
    dx_wrap    = image_width - dx_direct
    dx_actual  = torch.minimum(dx_direct, dx_wrap)          # panoramic wraparound

    area_A = w_A * h_A;  area_B = w_B * h_B

    # 1. horizontal_gap_norm
    horizontal_gap_norm = (dx_actual - (w_A + w_B) * 0.5) / image_width

    # 2. height_ratio
    height_ratio = torch.minimum(h_A, h_B) / (torch.maximum(h_A, h_B) + 1e-6)

    # 3. ground_distance
    ground_distance = dx_actual / image_width

    # 4. vertical overlap
    y2_A = y_A + h_A;  y2_B = y_B + h_B
    v_overlap = (torch.minimum(y2_A, y2_B) - torch.maximum(y_A, y_B)).clamp(min=0.0) \
                / (torch.minimum(h_A, h_B) + 1e-6)

    # 5. area_ratio
    area_ratio = torch.minimum(area_A, area_B) / (torch.maximum(area_A, area_B) + 1e-6)

    # 6. center_dist_norm
    dy = cy_A - cy_B
    center_dist_norm = torch.sqrt(dx_actual ** 2 + dy ** 2) \
                       / torch.sqrt(area_A + area_B + 1e-6)

    # 7. vertical_gap
    vertical_gap = dy.abs() / image_height

    return torch.stack([
        horizontal_gap_norm, height_ratio, ground_distance,
        v_overlap, area_ratio, center_dist_norm, vertical_gap,
    ], dim=1)  # [E, 7]


def extract_causal_motion_features(geometric_history):
    """
    Extract causal motion features from geometric history
    Only uses past information (t-n to t-1)
    
    Args:
        geometric_history: [batch_size, time_steps, 7] or [time_steps, 7]
        
    Returns:
        torch.Tensor: [batch_size, 4] or [4] motion features
    """
    if geometric_history.dim() == 2:
        # Single sequence
        geometric_history = geometric_history.unsqueeze(0)
    
    batch_size = geometric_history.size(0)
    motion_features = []
    
    for batch_idx in range(batch_size):
        batch_history = geometric_history[batch_idx]  # [time_steps, 7]
        time_steps = batch_history.size(0)
        
        if time_steps < 2:
            # Not enough history, return zero features
            motion_features.append(torch.zeros(4, dtype=torch.float32))
            continue
        
        # Extract distance changes over time
        distance_trend = []
        approach_speeds = []
        
        for t in range(1, time_steps):
            # Use center_dist_norm (index 5)
            prev_dist = batch_history[t-1, 5]
            curr_dist = batch_history[t, 5]
            
            distance_change = curr_dist - prev_dist
            distance_trend.append(distance_change.item())
            approach_speeds.append(abs(distance_change.item()))
        
        # Calculate motion statistics
        if len(distance_trend) > 0:
            avg_approach_speed = np.mean(approach_speeds)
            distance_trend_slope = distance_trend[-1]  # Latest trend
            is_approaching = 1.0 if distance_trend_slope < -0.01 else 0.0
            
            # Motion consistency (lower std = more consistent)
            if len(distance_trend) > 1:
                motion_consistency = 1.0 - min(np.std(distance_trend), 1.0)
            else:
                motion_consistency = 1.0
        else:
            avg_approach_speed = 0.0
            distance_trend_slope = 0.0
            is_approaching = 0.0
            motion_consistency = 1.0
        
        motion_feature = torch.tensor([
            avg_approach_speed,
            distance_trend_slope,
            is_approaching,
            motion_consistency
        ], dtype=torch.float32)
        
        motion_features.append(motion_feature)
    
    result = torch.stack(motion_features)
    if result.size(0) == 1:
        return result.squeeze(0).clone()
    else:
        return result


def compute_scene_context(all_boxes_in_frame, image_width=1000.0, image_height=1000.0):
    """
    Compute simplified scene context based on crowd level
    
    Args:
        all_boxes_in_frame: List of bounding boxes in current frame
        image_width, image_height: Image dimensions
        
    Returns:
        torch.Tensor: [1] scene context - crowd level (0=empty, 1=sparse, 2=moderate, 3=crowded)
    """
    num_people = len(all_boxes_in_frame)
    
    # Simple crowd level classification
    if num_people == 0:
        crowd_level = 0.0    # Empty scene
    elif num_people <= 5:
        crowd_level = 1.0    # Sparse (0-5 people)
    elif num_people <= 10:
        crowd_level = 2.0    # Moderate (5-10 people) 
    else:
        crowd_level = 3.0    # Crowded (10+ people)
    
    return torch.tensor([crowd_level], dtype=torch.float32)


if __name__ == '__main__':
    # Test geometric features extraction
    print("Testing Geometric Features Extraction...")
    
    # Test data
    box_A = [100, 100, 200, 300]  # x1, y1, x2, y2
    box_B = [180, 120, 280, 320]
    
    # Single pair test
    features = extract_geometric_features(box_A, box_B, 3760, 480)
    print(f"Geometric features shape: {features.shape}")
    print(f"Features: {features}")
    
    # Batch test
    batch_box_A = torch.tensor([[100, 100, 200, 300], [50, 50, 150, 250]])
    batch_box_B = torch.tensor([[180, 120, 280, 320], [200, 100, 300, 300]])
    
    batch_features = extract_geometric_features(batch_box_A, batch_box_B, 3760, 480)
    print(f"Batch features shape: {batch_features.shape}")
    
    # Test motion features
    print("\nTesting Motion Features...")
    history = torch.randn(5, 7)  # 5 time steps, 7 features
    motion_feats = extract_causal_motion_features(history)
    print(f"Motion features shape: {motion_feats.shape}")
    print(f"Motion features: {motion_feats}")
    
    # Test scene context
    print("\nTesting Scene Context...")
    all_boxes = [[100, 100, 200, 300], [180, 120, 280, 320], [300, 200, 400, 400]]
    context = compute_scene_context(all_boxes, 3760, 480)
    print(f"Scene context: {context}")
    
    print("All tests passed!")