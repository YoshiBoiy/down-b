from __future__ import annotations

import numpy as np


def _side_guard_score(joints: dict[str, np.ndarray], side: str, max_distance: float = 0.32) -> float:
    wrist = joints.get(f"{side}_wrist")
    nose = joints.get("nose")
    shoulder = joints.get(f"{side}_shoulder")
    if wrist is None or nose is None or shoulder is None:
        return 0.0
    target = 0.72 * nose + 0.28 * shoulder
    dist = float(np.linalg.norm(wrist - target))
    return max(0.0, min(1.0, 1.0 - dist / max_distance))


def guard_scores(joints: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "left": _side_guard_score(joints, "left"),
        "right": _side_guard_score(joints, "right"),
    }


def high_guard(guard: dict[str, float], threshold: float = 0.55) -> bool:
    return guard.get("left", 0.0) >= threshold and guard.get("right", 0.0) >= threshold
