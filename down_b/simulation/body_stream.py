from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from down_b.tracking.body_state import BodyStateBuilder
from down_b.tracking.filters import BodyFilter
from down_b.types import BodyState


def _base_joints() -> dict[str, np.ndarray]:
    return {
        "nose": np.array([0.0, -0.52, 2.0], dtype=np.float32),
        "left_shoulder": np.array([-0.22, -0.22, 2.05], dtype=np.float32),
        "right_shoulder": np.array([0.22, -0.22, 2.05], dtype=np.float32),
        "left_elbow": np.array([-0.34, -0.04, 1.98], dtype=np.float32),
        "right_elbow": np.array([0.34, -0.04, 1.98], dtype=np.float32),
        "left_wrist": np.array([-0.16, -0.42, 1.86], dtype=np.float32),
        "right_wrist": np.array([0.16, -0.42, 1.86], dtype=np.float32),
        "left_hip": np.array([-0.16, 0.22, 2.10], dtype=np.float32),
        "right_hip": np.array([0.16, 0.22, 2.16], dtype=np.float32),
        "left_knee": np.array([-0.16, 0.58, 2.06], dtype=np.float32),
        "right_knee": np.array([0.16, 0.58, 2.18], dtype=np.float32),
        "left_ankle": np.array([-0.16, 0.92, 2.02], dtype=np.float32),
        "right_ankle": np.array([0.16, 0.92, 2.22], dtype=np.float32),
    }


def simulated_body_states(fps: int = 30) -> Iterator[BodyState]:
    builder = BodyStateBuilder(BodyFilter(max_speed_mps=12.0))
    timestamp = 0
    frame = 0
    while True:
        joints = _base_joints()
        phase = frame % 180
        if 20 <= phase < 30:
            t = (phase - 20) / 10
            joints["left_wrist"] = np.array([-0.08, -0.37, 1.86 - 0.7 * math.sin(t * math.pi)], dtype=np.float32)
            joints["right_wrist"] = np.array([0.42, -0.12, 2.0], dtype=np.float32)
        elif 60 <= phase < 72:
            t = (phase - 60) / 12
            joints["nose"] = np.array([-0.24 * math.sin(t * math.pi), -0.52, 2.0], dtype=np.float32)
        elif 100 <= phase < 112:
            t = (phase - 100) / 12
            joints["right_wrist"] = np.array([0.08, -0.37, 1.86 - 0.72 * math.sin(t * math.pi)], dtype=np.float32)
        elif 140 <= phase < 152:
            t = (phase - 140) / 12
            drop = 0.34 * math.sin(t * math.pi)
            joints["left_hip"] += np.array([0.0, 0.0, -drop], dtype=np.float32)
            joints["right_hip"] += np.array([0.0, 0.0, -drop], dtype=np.float32)
            for key in list(joints):
                if key not in {"left_hip", "right_hip"}:
                    joints[key] += np.array([0.0, 0.0, -drop], dtype=np.float32)
        confidence = {name: 0.92 for name in joints}
        yield builder.build(timestamp, 0, joints, confidence)
        timestamp += int(1_000_000_000 / fps)
        frame += 1
