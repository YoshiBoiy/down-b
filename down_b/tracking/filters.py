from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class JointFilterState:
    value: np.ndarray | None = None
    timestamp_ns: int | None = None
    missing: int = 0


class JointFilter:
    def __init__(self, alpha: float, max_speed_mps: float, max_gap_frames: int = 2):
        self.alpha = alpha
        self.max_speed_mps = max_speed_mps
        self.max_gap_frames = max_gap_frames
        self.state = JointFilterState()

    def update(self, value: np.ndarray | None, confidence: float, timestamp_ns: int) -> tuple[np.ndarray | None, float]:
        if value is None or confidence <= 0.0:
            self.state.missing += 1
            if self.state.value is not None and self.state.missing <= self.max_gap_frames:
                return self.state.value.copy(), confidence * 0.5
            return None, 0.0
        if self.state.value is None or self.state.timestamp_ns is None:
            self.state = JointFilterState(value=value.copy(), timestamp_ns=timestamp_ns, missing=0)
            return value.copy(), confidence
        dt = max(1e-6, (timestamp_ns - self.state.timestamp_ns) / 1_000_000_000)
        speed = float(np.linalg.norm(value - self.state.value) / dt)
        if speed > self.max_speed_mps:
            self.state.missing += 1
            return self.state.value.copy(), confidence * 0.25
        filtered = self.alpha * value + (1.0 - self.alpha) * self.state.value
        self.state = JointFilterState(value=filtered.copy(), timestamp_ns=timestamp_ns, missing=0)
        return filtered, confidence


class BodyFilter:
    def __init__(self, max_speed_mps: float):
        self.filters: dict[str, JointFilter] = {}
        self.max_speed_mps = max_speed_mps

    def _alpha_for(self, joint: str) -> float:
        if "wrist" in joint:
            return 0.72
        if joint in {"nose", "left_hip", "right_hip"}:
            return 0.38
        return 0.55

    def update(
        self,
        joints: dict[str, np.ndarray],
        confidence: dict[str, float],
        timestamp_ns: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        filtered: dict[str, np.ndarray] = {}
        filtered_conf: dict[str, float] = {}
        names = set(joints) | set(self.filters)
        for name in names:
            filt = self.filters.setdefault(
                name,
                JointFilter(alpha=self._alpha_for(name), max_speed_mps=self.max_speed_mps),
            )
            value, conf = filt.update(joints.get(name), confidence.get(name, 0.0), timestamp_ns)
            if value is not None:
                filtered[name] = value
                filtered_conf[name] = conf
        return filtered, filtered_conf
