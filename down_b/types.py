from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


JOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class Stance(str, Enum):
    ORTHODOX = "orthodox"
    SOUTHPAW = "southpaw"
    UNKNOWN = "unknown"


class PlayerAction(str, Enum):
    NEUTRAL = "neutral"
    HIGH_GUARD = "high_guard"
    JAB = "jab"
    CROSS = "cross"
    LEFT_SLIP = "left_slip"
    RIGHT_SLIP = "right_slip"
    DUCK = "duck"
    STEP_FORWARD = "step_forward"
    STEP_BACKWARD = "step_backward"
    STEP_LEFT = "step_left"
    STEP_RIGHT = "step_right"
    LOST = "lost"


class OpponentAction(str, Enum):
    WAIT = "wait"
    JAB = "jab"
    CROSS = "cross"
    JAB_CROSS = "jab_cross"
    STEP_FORWARD = "step_forward"
    STEP_BACKWARD = "step_backward"
    BLOCK = "block"
    SLIP = "slip"


class ExchangeResult(str, Enum):
    HIT = "hit"
    BLOCK = "block"
    MISS = "miss"
    UNCERTAIN = "uncertain"


@dataclass(slots=True)
class Pose2D:
    bbox_xyxy: np.ndarray
    keypoints_xy: np.ndarray
    keypoint_confidence: np.ndarray
    detection_confidence: float


@dataclass(slots=True)
class StereoFrame:
    timestamp_ns: int
    left_bgr: np.ndarray
    right_bgr: np.ndarray
    left_rectified: np.ndarray
    right_rectified: np.ndarray


@dataclass(slots=True)
class BodyState:
    timestamp_ns: int
    track_id: int
    pose_confidence: float
    stance: Stance
    joints_3d: dict[str, np.ndarray]
    joint_confidence: dict[str, float]
    guard: dict[str, float] = field(default_factory=lambda: {"left": 0.0, "right": 0.0})
    raw_joints_3d: dict[str, np.ndarray] = field(default_factory=dict)

    def joint(self, name: str) -> np.ndarray | None:
        return self.joints_3d.get(name)

    def confidence(self, name: str) -> float:
        return float(self.joint_confidence.get(name, 0.0))

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "track_id": self.track_id,
            "pose_confidence": self.pose_confidence,
            "stance": self.stance.value,
            "joints_3d": {k: [float(x) for x in v] for k, v in self.joints_3d.items()},
            "joint_confidence": {k: float(v) for k, v in self.joint_confidence.items()},
            "guard": {k: float(v) for k, v in self.guard.items()},
        }


@dataclass(slots=True)
class GameFrame:
    timestamp_ns: int
    player_action: PlayerAction
    opponent_action: OpponentAction
    exchange_result: ExchangeResult
    score: dict[str, int]
    confidence: float
    reason: str = ""
    adaptation: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "frame",
            "timestamp_ns": self.timestamp_ns,
            "player_action": self.player_action.value,
            "opponent_action": self.opponent_action.value,
            "exchange_result": self.exchange_result.value,
            "score": self.score,
            "confidence": self.confidence,
            "reason": self.reason,
            "adaptation": self.adaptation,
        }
