from __future__ import annotations

import numpy as np


def midpoint(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def pelvis(joints: dict[str, np.ndarray]) -> np.ndarray | None:
    return midpoint(joints.get("left_hip"), joints.get("right_hip"))


def shoulder_center(joints: dict[str, np.ndarray]) -> np.ndarray | None:
    return midpoint(joints.get("left_shoulder"), joints.get("right_shoulder"))


def arm_length(joints: dict[str, np.ndarray], side: str) -> float | None:
    shoulder = joints.get(f"{side}_shoulder")
    elbow = joints.get(f"{side}_elbow")
    wrist = joints.get(f"{side}_wrist")
    upper = distance(shoulder, elbow)
    lower = distance(elbow, wrist)
    if upper is None or lower is None:
        return None
    return max(upper + lower, 1e-6)


def arm_extension(joints: dict[str, np.ndarray], side: str) -> float | None:
    length = arm_length(joints, side)
    wrist_to_shoulder = distance(joints.get(f"{side}_wrist"), joints.get(f"{side}_shoulder"))
    if length is None or wrist_to_shoulder is None:
        return None
    return wrist_to_shoulder / length
