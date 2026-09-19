from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from down_b.capture.calibration import Rectifier
from down_b.types import StereoFrame


@dataclass(slots=True)
class CameraImage:
    timestamp_ns: int
    bgr: np.ndarray


class StereoSynchronizer:
    def __init__(self, max_delta_ns: int, rectifier: Rectifier | None = None, max_queue: int = 2):
        self.max_delta_ns = max_delta_ns
        self.rectifier = rectifier or Rectifier()
        self.left_queue: deque[CameraImage] = deque(maxlen=max_queue)
        self.right_queue: deque[CameraImage] = deque(maxlen=max_queue)

    def push_left(self, image: CameraImage) -> StereoFrame | None:
        self.left_queue.append(image)
        return self._match()

    def push_right(self, image: CameraImage) -> StereoFrame | None:
        self.right_queue.append(image)
        return self._match()

    def _match(self) -> StereoFrame | None:
        if not self.left_queue or not self.right_queue:
            return None
        best: tuple[int, int, int] | None = None
        for li, left in enumerate(self.left_queue):
            for ri, right in enumerate(self.right_queue):
                delta = abs(left.timestamp_ns - right.timestamp_ns)
                if best is None or delta < best[0]:
                    best = (delta, li, ri)
        if best is None or best[0] > self.max_delta_ns:
            self._drop_stale()
            return None
        _, li, ri = best
        left = self.left_queue[li]
        right = self.right_queue[ri]
        for _ in range(li + 1):
            self.left_queue.popleft()
        for _ in range(ri + 1):
            self.right_queue.popleft()
        left_rect, right_rect = self.rectifier.rectify_pair(left.bgr, right.bgr)
        return StereoFrame(
            timestamp_ns=max(left.timestamp_ns, right.timestamp_ns),
            left_bgr=left.bgr,
            right_bgr=right.bgr,
            left_rectified=left_rect,
            right_rectified=right_rect,
        )

    def _drop_stale(self) -> None:
        if self.left_queue and self.right_queue:
            if self.left_queue[0].timestamp_ns < self.right_queue[0].timestamp_ns:
                self.left_queue.popleft()
            else:
                self.right_queue.popleft()
