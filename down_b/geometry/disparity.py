from __future__ import annotations

import numpy as np


def sample_disparity(
    disparity_map: np.ndarray,
    u: float,
    v: float,
    radius: int = 2,
    min_disparity: float = 1.0,
    max_std: float = 4.0,
) -> tuple[float | None, float]:
    height, width = disparity_map.shape[:2]
    cx = int(round(u))
    cy = int(round(v))
    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius + 1)
    patch = disparity_map[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch >= min_disparity)]
    if valid.size == 0:
        return None, 0.0
    median = float(np.median(valid))
    std = float(np.std(valid))
    confidence = max(0.0, min(1.0, 1.0 - std / max_std))
    return median, confidence


def synthetic_constant_disparity(width: int, height: int, disparity_px: float) -> np.ndarray:
    return np.full((height, width), disparity_px, dtype=np.float32)
