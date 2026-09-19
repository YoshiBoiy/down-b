from __future__ import annotations

import numpy as np

from down_b.capture.calibration import StereoCalibration
from down_b.geometry.disparity import sample_disparity
from down_b.types import JOINT_NAMES, Pose2D


def lift_pose_3d(
    pose: Pose2D,
    disparity_map: np.ndarray,
    calibration: StereoCalibration,
    min_joint_confidence: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    joints: dict[str, np.ndarray] = {}
    confidence: dict[str, float] = {}
    for index, name in enumerate(JOINT_NAMES):
        if index >= len(pose.keypoints_xy):
            break
        joint_conf = float(pose.keypoint_confidence[index])
        if joint_conf < min_joint_confidence:
            confidence[name] = 0.0
            continue
        u, v = pose.keypoints_xy[index]
        disparity, depth_conf = sample_disparity(
            disparity_map,
            float(u),
            float(v),
            min_disparity=calibration.min_disparity_px,
        )
        if disparity is None:
            confidence[name] = 0.0
            continue
        point = calibration.reproject(float(u), float(v), disparity)
        if point is None:
            confidence[name] = 0.0
            continue
        joints[name] = point
        confidence[name] = min(joint_conf, depth_conf)
    return joints, confidence
