import numpy as np

from down_b.boxing.actions import ActionRecognizer
from down_b.config import ThresholdConfig
from down_b.types import BodyState, PlayerAction, Stance


def thresholds() -> ThresholdConfig:
    return ThresholdConfig(
        min_pose_confidence=0.45,
        min_joint_confidence=0.35,
        high_guard_distance=0.28,
        punch_speed_mps=1.0,
        punch_extension=0.75,
        slip_distance_m=0.18,
        duck_distance_m=0.16,
        step_distance_m=0.18,
        max_joint_speed_mps=8.0,
    )


def body(timestamp_ns: int, left_wrist_z: float = 1.9, nose_x: float = 0.0) -> BodyState:
    joints = {
        "nose": np.array([nose_x, -0.52, 2.0], dtype=np.float32),
        "left_shoulder": np.array([-0.22, -0.22, 2.0], dtype=np.float32),
        "right_shoulder": np.array([0.22, -0.22, 2.0], dtype=np.float32),
        "left_elbow": np.array([-0.28, -0.20, 1.6], dtype=np.float32),
        "right_elbow": np.array([0.28, -0.20, 1.9], dtype=np.float32),
        "left_wrist": np.array([-0.18, -0.20, left_wrist_z], dtype=np.float32),
        "right_wrist": np.array([0.16, -0.42, 1.86], dtype=np.float32),
        "left_hip": np.array([-0.16, 0.22, 2.0], dtype=np.float32),
        "right_hip": np.array([0.16, 0.22, 2.1], dtype=np.float32),
        "left_ankle": np.array([-0.16, 0.92, 2.0], dtype=np.float32),
        "right_ankle": np.array([0.16, 0.92, 2.2], dtype=np.float32),
    }
    return BodyState(
        timestamp_ns=timestamp_ns,
        track_id=0,
        pose_confidence=0.9,
        stance=Stance.ORTHODOX,
        joints_3d=joints,
        joint_confidence={key: 0.9 for key in joints},
        guard={"left": 0.2, "right": 0.9},
    )


def test_detects_orthodox_left_hand_jab():
    recognizer = ActionRecognizer(thresholds())
    recognizer.update(body(0, left_wrist_z=1.9))
    action, _ = recognizer.update(body(100_000_000, left_wrist_z=1.0))
    assert action == PlayerAction.JAB


def test_detects_head_slip_relative_to_pelvis():
    recognizer = ActionRecognizer(thresholds())
    action, _ = recognizer.update(body(0, nose_x=-0.25))
    assert action == PlayerAction.LEFT_SLIP
