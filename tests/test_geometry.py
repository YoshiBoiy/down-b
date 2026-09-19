import numpy as np

from down_b.capture.calibration import StereoCalibration
from down_b.geometry.disparity import sample_disparity


def test_reproject_uses_metric_stereo_formula():
    calibration = StereoCalibration(fx=700, fy=700, cx=320, cy=240, baseline_m=0.07)
    point = calibration.reproject(320, 240, 49)
    assert point is not None
    assert np.allclose(point, [0.0, 0.0, 1.0], atol=1e-5)


def test_disparity_sampling_rejects_invalid_and_reports_confidence():
    disparity = np.array(
        [
            [0, 49, 50],
            [49, 50, 51],
            [0, 52, 50],
        ],
        dtype=np.float32,
    )
    value, confidence = sample_disparity(disparity, 1, 1, radius=1, min_disparity=1)
    assert value == 50
    assert 0.0 < confidence <= 1.0
