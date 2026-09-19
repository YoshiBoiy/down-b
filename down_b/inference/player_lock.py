from __future__ import annotations

import numpy as np

from down_b.types import Pose2D


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else float(inter / denom)


class PlayerLock:
    def __init__(self, lost_frame_limit: int = 5):
        self.track_id = 0
        self.locked_bbox: np.ndarray | None = None
        self.lost_frames = 0
        self.lost_frame_limit = lost_frame_limit

    @property
    def is_lost(self) -> bool:
        return self.lost_frames > self.lost_frame_limit

    def select(self, poses: list[Pose2D]) -> Pose2D | None:
        if not poses:
            self.lost_frames += 1
            return None
        if self.locked_bbox is None:
            pose = max(poses, key=lambda p: p.detection_confidence)
            self.locked_bbox = pose.bbox_xyxy.copy()
            self.lost_frames = 0
            return pose
        pose = max(
            poses,
            key=lambda p: 0.7 * bbox_iou(self.locked_bbox, p.bbox_xyxy) + 0.3 * p.detection_confidence,
        )
        if bbox_iou(self.locked_bbox, pose.bbox_xyxy) < 0.05 and pose.detection_confidence < 0.7:
            self.lost_frames += 1
            return None
        self.locked_bbox = pose.bbox_xyxy.copy()
        self.lost_frames = 0
        return pose
