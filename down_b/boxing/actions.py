from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from down_b.boxing.features import arm_extension, pelvis
from down_b.boxing.guard import high_guard
from down_b.config import ThresholdConfig
from down_b.types import BodyState, PlayerAction, Stance


@dataclass(slots=True)
class ActionFeatures:
    left_wrist_speed: float = 0.0
    right_wrist_speed: float = 0.0
    left_extension: float = 0.0
    right_extension: float = 0.0
    head_offset_x: float = 0.0
    head_offset_y: float = 0.0
    pelvis_delta: np.ndarray | None = None


class ActionRecognizer:
    def __init__(self, thresholds: ThresholdConfig, history_size: int = 12):
        self.thresholds = thresholds
        self.history: deque[BodyState] = deque(maxlen=history_size)

    def update(self, state: BodyState) -> tuple[PlayerAction, ActionFeatures]:
        previous = self.history[-1] if self.history else None
        self.history.append(state)
        features = self._features(state, previous)
        if state.pose_confidence < self.thresholds.min_pose_confidence:
            return PlayerAction.LOST, features
        if features.head_offset_y > self.thresholds.duck_distance_m:
            return PlayerAction.DUCK, features
        if features.head_offset_x > self.thresholds.slip_distance_m:
            return PlayerAction.RIGHT_SLIP, features
        if features.head_offset_x < -self.thresholds.slip_distance_m:
            return PlayerAction.LEFT_SLIP, features
        footwork = self._footwork(features)
        if footwork is not None:
            return footwork, features
        punch = self._punch(state, features)
        if punch is not None:
            return punch, features
        if high_guard(state.guard):
            return PlayerAction.HIGH_GUARD, features
        return PlayerAction.NEUTRAL, features

    def _features(self, state: BodyState, previous: BodyState | None) -> ActionFeatures:
        left_speed = self._joint_speed("left_wrist", state, previous)
        right_speed = self._joint_speed("right_wrist", state, previous)
        left_extension = arm_extension(state.joints_3d, "left") or 0.0
        right_extension = arm_extension(state.joints_3d, "right") or 0.0
        head = state.joint("nose")
        pelv = pelvis(state.joints_3d)
        head_offset = head - pelv if head is not None and pelv is not None else np.zeros(3)
        prev_pelvis = pelvis(previous.joints_3d) if previous else None
        pelvis_delta = pelv - prev_pelvis if pelv is not None and prev_pelvis is not None else None
        return ActionFeatures(
            left_wrist_speed=left_speed,
            right_wrist_speed=right_speed,
            left_extension=left_extension,
            right_extension=right_extension,
            head_offset_x=float(head_offset[0]),
            head_offset_y=float(head_offset[1]),
            pelvis_delta=pelvis_delta,
        )

    def _joint_speed(self, joint: str, state: BodyState, previous: BodyState | None) -> float:
        if previous is None:
            return 0.0
        a = previous.joint(joint)
        b = state.joint(joint)
        if a is None or b is None:
            return 0.0
        dt = max(1e-6, (state.timestamp_ns - previous.timestamp_ns) / 1_000_000_000)
        return float(np.linalg.norm(b - a) / dt)

    def _punch(self, state: BodyState, features: ActionFeatures) -> PlayerAction | None:
        left_candidate = (
            features.left_wrist_speed >= self.thresholds.punch_speed_mps
            and features.left_extension >= self.thresholds.punch_extension
        )
        right_candidate = (
            features.right_wrist_speed >= self.thresholds.punch_speed_mps
            and features.right_extension >= self.thresholds.punch_extension
        )
        if not left_candidate and not right_candidate:
            return None
        moving_side = "left" if features.left_wrist_speed >= features.right_wrist_speed else "right"
        lead_side = "left" if state.stance == Stance.ORTHODOX else "right"
        return PlayerAction.JAB if moving_side == lead_side else PlayerAction.CROSS

    def _footwork(self, features: ActionFeatures) -> PlayerAction | None:
        delta = features.pelvis_delta
        if delta is None:
            return None
        if abs(float(delta[2])) >= self.thresholds.step_distance_m:
            return PlayerAction.STEP_FORWARD if delta[2] < 0 else PlayerAction.STEP_BACKWARD
        if abs(float(delta[0])) >= self.thresholds.step_distance_m:
            return PlayerAction.STEP_RIGHT if delta[0] > 0 else PlayerAction.STEP_LEFT
        return None
