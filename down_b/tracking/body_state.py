from __future__ import annotations

import numpy as np

from down_b.boxing.guard import guard_scores
from down_b.tracking.filters import BodyFilter
from down_b.types import BodyState, Stance


def detect_stance(joints: dict[str, np.ndarray]) -> Stance:
    left = joints.get("left_ankle")
    right = joints.get("right_ankle")
    if left is None or right is None:
        return Stance.UNKNOWN
    return Stance.ORTHODOX if left[2] < right[2] else Stance.SOUTHPAW


class BodyStateBuilder:
    def __init__(self, body_filter: BodyFilter):
        self.body_filter = body_filter

    def build(
        self,
        timestamp_ns: int,
        track_id: int,
        raw_joints: dict[str, np.ndarray],
        raw_confidence: dict[str, float],
    ) -> BodyState:
        joints, confidence = self.body_filter.update(raw_joints, raw_confidence, timestamp_ns)
        pose_conf = float(np.mean(list(confidence.values()))) if confidence else 0.0
        stance = detect_stance(joints)
        return BodyState(
            timestamp_ns=timestamp_ns,
            track_id=track_id,
            pose_confidence=pose_conf,
            stance=stance,
            joints_3d=joints,
            joint_confidence=confidence,
            guard=guard_scores(joints),
            raw_joints_3d=raw_joints,
        )
