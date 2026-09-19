from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from down_b.types import JOINT_NAMES, Pose2D


class PoseEstimator(ABC):
    @abstractmethod
    def infer(self, image: np.ndarray) -> list[Pose2D]:
        raise NotImplementedError


class BpuPoseEstimator(PoseEstimator):
    def __init__(self, model_path: str):
        self.model_path = model_path
        raise RuntimeError(
            "BpuPoseEstimator is an integration adapter. Install the RDK X5 "
            "hbm_runtime/model-zoo stack and implement model-specific parsing here."
        )

    def infer(self, image: np.ndarray) -> list[Pose2D]:
        raise NotImplementedError


class MockPoseEstimator(PoseEstimator):
    def infer(self, image: np.ndarray) -> list[Pose2D]:
        height, width = image.shape[:2]
        center_x = width / 2
        center_y = height / 2
        keypoints = np.zeros((len(JOINT_NAMES), 2), dtype=np.float32)
        layout = {
            "nose": (0.0, -0.42),
            "left_shoulder": (-0.18, -0.22),
            "right_shoulder": (0.18, -0.22),
            "left_elbow": (-0.28, 0.02),
            "right_elbow": (0.28, 0.02),
            "left_wrist": (-0.22, -0.16),
            "right_wrist": (0.22, -0.16),
            "left_hip": (-0.14, 0.24),
            "right_hip": (0.14, 0.24),
            "left_knee": (-0.14, 0.54),
            "right_knee": (0.14, 0.54),
            "left_ankle": (-0.14, 0.86),
            "right_ankle": (0.14, 0.86),
        }
        for idx, name in enumerate(JOINT_NAMES):
            dx, dy = layout.get(name, (0.0, 0.0))
            keypoints[idx] = [center_x + dx * width, center_y + dy * height]
        conf = np.full(len(JOINT_NAMES), 0.9, dtype=np.float32)
        return [
            Pose2D(
                bbox_xyxy=np.array([center_x - 120, center_y - 220, center_x + 120, center_y + 240]),
                keypoints_xy=keypoints,
                keypoint_confidence=conf,
                detection_confidence=0.92,
            )
        ]
