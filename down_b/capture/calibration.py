from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from down_b.config import CameraConfig


@dataclass(frozen=True, slots=True)
class StereoCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    min_disparity_px: float = 1.0

    @classmethod
    def from_config(cls, config: CameraConfig) -> "StereoCalibration":
        return cls(
            fx=config.fx,
            fy=config.fy,
            cx=config.cx,
            cy=config.cy,
            baseline_m=config.baseline_m,
            min_disparity_px=config.min_disparity_px,
        )

    def reproject(self, u: float, v: float, disparity: float) -> np.ndarray | None:
        if not np.isfinite(disparity) or disparity < self.min_disparity_px:
            return None
        z = self.fx * self.baseline_m / disparity
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float32)


class Rectifier:
    """Placeholder rectifier for calibrated image maps.

    On the RDK path this class should be initialized with OpenCV remap matrices
    from camera.yaml. In simulator/tests the frames are already rectified.
    """

    def rectify_pair(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return left_bgr, right_bgr
