#!/usr/bin/env python3
"""
Interaction Synchrony Feature Extractor
计算两人社交互动中的运动同步性特征
"""

import numpy as np
import torch
from typing import Union


def compute_interaction_synchrony(current_features: Union[np.ndarray, torch.Tensor],
                                   partner_features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    计算交互同步性特征

    核心假设:
    - Sitting together: 高度同步 (0.7-0.9)
      两人同时喝咖啡、同时调整姿势、运动模式一致

    - Walking together: 中等同步 (0.5-0.7)
      步态有周期性，但存在相位差

    - Standing together: 低同步 (0.3-0.6)
      各自独立移动，同步性较低

    Args:
        current_features: 当前人的6D特征 [N, 6] or [6]
            [0] inter_distance / avg_height
            [1] inter_distance / avg_width
            [2] flow_magnitude_mean / avg_area  ← 运动强度
            [3] flow_magnitude_std / avg_area   ← 运动变化
            [4] vertical_flow_dominance
            [5] avg_aspect_ratio                ← 姿态

        partner_features: 交互对象的6D特征 [N, 6] or [6]

    Returns:
        sync_score: 同步性分数 [N] or scalar
            范围: [0, 1]
            0 = 完全不同步
            1 = 完全同步
    """
    # 转换为numpy处理
    is_torch = isinstance(current_features, torch.Tensor)
    if is_torch:
        current = current_features.cpu().numpy()
        partner = partner_features.cpu().numpy()
    else:
        current = current_features
        partner = partner_features

    # 处理单样本情况
    single_sample = (len(current.shape) == 1)
    if single_sample:
        current = current.reshape(1, -1)
        partner = partner.reshape(1, -1)

    # === 1. 运动强度同步性 (Motion Intensity Synchrony) ===
    # 使用 f2: flow_magnitude_mean / avg_area
    flow_intensity_curr = current[:, 2]
    flow_intensity_partner = partner[:, 2]

    # 差异越小，同步性越高
    flow_diff = np.abs(flow_intensity_curr - flow_intensity_partner)
    # 归一化：假设最大差异为0.5
    flow_sync = 1.0 - np.clip(flow_diff / 0.5, 0, 1)

    # === 2. 运动模式同步性 (Motion Pattern Synchrony) ===
    # 使用 f3: flow_magnitude_std / avg_area (运动变化性)
    flow_variability_curr = current[:, 3]
    flow_variability_partner = partner[:, 3]

    pattern_diff = np.abs(flow_variability_curr - flow_variability_partner)
    # 归一化：假设最大差异为0.3
    pattern_sync = 1.0 - np.clip(pattern_diff / 0.3, 0, 1)

    # === 3. 姿态同步性 (Posture Synchrony) ===
    # 使用 f5: avg_aspect_ratio (长宽比)
    aspect_ratio_curr = current[:, 5]
    aspect_ratio_partner = partner[:, 5]

    posture_diff = np.abs(aspect_ratio_curr - aspect_ratio_partner)
    # 归一化：假设最大差异为1.0
    posture_sync = 1.0 - np.clip(posture_diff / 1.0, 0, 1)

    # === 4. 运动方向同步性 (Motion Direction Synchrony) ===
    # 使用 f4: vertical_flow_dominance
    direction_curr = current[:, 4]
    direction_partner = partner[:, 4]

    direction_diff = np.abs(direction_curr - direction_partner)
    # 归一化：假设最大差异为2.0
    direction_sync = 1.0 - np.clip(direction_diff / 2.0, 0, 1)

    # === 综合同步分数 ===
    # 使用加权几何平均（更敏感于低分）
    # sitting需要所有维度都高度同步
    sync_score = (flow_sync ** 0.4) * (pattern_sync ** 0.3) * (posture_sync ** 0.2) * (direction_sync ** 0.1)

    # 恢复形状和转回torch
    if single_sample:
        sync_score = sync_score[0]
        if is_torch:
            # Handle scalar case
            sync_score = torch.tensor(sync_score, dtype=torch.float32)
    else:
        if is_torch:
            # Handle array case
            sync_score = torch.from_numpy(sync_score).float()

    return sync_score


def compute_enhanced_synchrony(current_features: Union[np.ndarray, torch.Tensor],
                                partner_features: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    计算增强的同步性特征（返回多个维度）

    Returns:
        dict with keys:
            - overall_sync: 总体同步性
            - motion_sync: 运动同步性
            - posture_sync: 姿态同步性
            - direction_sync: 方向同步性
    """
    # 转换为numpy
    is_torch = isinstance(current_features, torch.Tensor)
    if is_torch:
        current = current_features.cpu().numpy()
        partner = partner_features.cpu().numpy()
    else:
        current = current_features
        partner = partner_features

    # 处理单样本
    single_sample = (len(current.shape) == 1)
    if single_sample:
        current = current.reshape(1, -1)
        partner = partner.reshape(1, -1)

    # 计算各维度同步性
    # Motion sync (f2 + f3)
    flow_diff = np.abs(current[:, 2] - partner[:, 2])
    pattern_diff = np.abs(current[:, 3] - partner[:, 3])
    motion_sync = 1.0 - np.clip((flow_diff + pattern_diff) / 0.8, 0, 1)

    # Posture sync (f5)
    posture_diff = np.abs(current[:, 5] - partner[:, 5])
    posture_sync = 1.0 - np.clip(posture_diff / 1.0, 0, 1)

    # Direction sync (f4)
    direction_diff = np.abs(current[:, 4] - partner[:, 4])
    direction_sync = 1.0 - np.clip(direction_diff / 2.0, 0, 1)

    # Overall sync (weighted combination)
    overall_sync = 0.5 * motion_sync + 0.3 * posture_sync + 0.2 * direction_sync

    # 恢复形状
    if single_sample:
        motion_sync = motion_sync[0]
        posture_sync = posture_sync[0]
        direction_sync = direction_sync[0]
        overall_sync = overall_sync[0]

    result = {
        'overall_sync': overall_sync,
        'motion_sync': motion_sync,
        'posture_sync': posture_sync,
        'direction_sync': direction_sync
    }

    # 转回torch
    if is_torch:
        if single_sample:
            # Handle scalar case
            result = {k: torch.tensor(v, dtype=torch.float32) for k, v in result.items()}
        else:
            # Handle array case
            result = {k: torch.from_numpy(v).float() for k, v in result.items()}

    return result


def analyze_synchrony_pattern(sync_score: float) -> str:
    """
    根据同步分数判断交互模式

    Args:
        sync_score: 同步性分数 [0, 1]

    Returns:
        pattern: 交互模式描述
    """
    if sync_score >= 0.75:
        return "高度同步 - 可能是sitting together (聊天、用餐等亲密互动)"
    elif sync_score >= 0.55:
        return "中度同步 - 可能是walking together (并行行走)"
    elif sync_score >= 0.35:
        return "低度同步 - 可能是standing together (独立但邻近)"
    else:
        return "不同步 - 可能无互动或是偶然邻近"


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """示例：如何使用同步性特征"""
    print("="*80)
    print("交互同步性特征使用示例")
    print("="*80)

    # === 场景1: 坐着聊天 ===
    print("\n场景1: 两人坐在咖啡馆聊天")
    person_A_features = np.array([
        0.6,   # f0: 距离/高度 (近)
        0.8,   # f1: 距离/宽度
        0.02,  # f2: 光流强度 (静止)
        0.01,  # f3: 光流变化 (一致)
        0.7,   # f4: 垂直主导度
        1.8    # f5: 长宽比 (坐姿)
    ])

    person_B_features = np.array([
        0.6,   # 相似的距离
        0.8,
        0.025, # 相似的运动强度
        0.012, # 相似的运动变化
        0.75,  # 相似的方向
        1.85   # 相似的姿态
    ])

    sync_score = compute_interaction_synchrony(person_A_features, person_B_features)
    print(f"同步分数: {sync_score:.3f}")
    print(f"分析: {analyze_synchrony_pattern(sync_score)}")

    # 详细分析
    detailed = compute_enhanced_synchrony(person_A_features, person_B_features)
    print(f"详细分析:")
    print(f"  运动同步性: {detailed['motion_sync']:.3f}")
    print(f"  姿态同步性: {detailed['posture_sync']:.3f}")
    print(f"  方向同步性: {detailed['direction_sync']:.3f}")

    # === 场景2: 并行行走 ===
    print("\n场景2: 两人并排行走")
    person_A_walking = np.array([
        2.0,   # f0: 较远距离
        2.5,   # f1
        0.18,  # f2: 显著运动
        0.10,  # f3: 运动变化大
        1.2,   # f4: 步态上下运动
        2.2    # f5: 行走姿态
    ])

    person_B_walking = np.array([
        2.0,
        2.5,
        0.15,  # 运动强度稍不同（步态相位差）
        0.12,  # 变化稍不同
        1.4,   # 方向稍不同
        2.3    # 姿态相似
    ])

    sync_score = compute_interaction_synchrony(person_A_walking, person_B_walking)
    print(f"同步分数: {sync_score:.3f}")
    print(f"分析: {analyze_synchrony_pattern(sync_score)}")

    # === 场景3: 站着排队 ===
    print("\n场景3: 两人站着排队（各自玩手机）")
    person_A_standing = np.array([
        1.2,   # f0: 中等距离
        1.5,   # f1
        0.05,  # f2: 轻微移动
        0.03,  # f3: 小幅变化
        0.5,   # f4: 水平晃动
        2.6    # f5: 站立姿态
    ])

    person_B_standing = np.array([
        1.2,
        1.5,
        0.08,  # 运动强度不同（独立移动）
        0.06,  # 变化不同
        0.3,   # 方向不同
        2.7    # 姿态略不同
    ])

    sync_score = compute_interaction_synchrony(person_A_standing, person_B_standing)
    print(f"同步分数: {sync_score:.3f}")
    print(f"分析: {analyze_synchrony_pattern(sync_score)}")

    print("\n" + "="*80)

    # === 批量处理示例 ===
    print("\n批量处理示例:")
    batch_current = np.array([person_A_features, person_A_walking, person_A_standing])
    batch_partner = np.array([person_B_features, person_B_walking, person_B_standing])

    batch_sync = compute_interaction_synchrony(batch_current, batch_partner)
    print(f"批量同步分数: {batch_sync}")


if __name__ == '__main__':
    example_usage()
