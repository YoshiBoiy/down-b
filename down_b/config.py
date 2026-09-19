from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CameraConfig:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    min_disparity_px: float
    max_pair_delta_ns: int


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    min_pose_confidence: float
    min_joint_confidence: float
    high_guard_distance: float
    punch_speed_mps: float
    punch_extension: float
    slip_distance_m: float
    duck_distance_m: float
    step_distance_m: float
    max_joint_speed_mps: float


@dataclass(frozen=True, slots=True)
class OpponentConfig:
    min_confident_samples: int
    guard_drop_rate_threshold: float
    base_jab_probability: float
    base_cross_probability: float
    adapted_counter_probability: float
    punch_reach_m: float


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_camera_config(path: str | Path) -> CameraConfig:
    data = _read_yaml(Path(path))
    return CameraConfig(**data)


def load_thresholds(path: str | Path) -> ThresholdConfig:
    data = _read_yaml(Path(path))
    return ThresholdConfig(**data)


def load_opponent_config(path: str | Path) -> OpponentConfig:
    data = _read_yaml(Path(path))
    return OpponentConfig(**data)
