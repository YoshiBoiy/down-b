import numpy as np

from down_b.boxing.adaptation import AdaptationEngine
from down_b.boxing.hits import score_robot_punch
from down_b.config import OpponentConfig
from down_b.types import BodyState, ExchangeResult, OpponentAction, PlayerAction, Stance


def opponent_config() -> OpponentConfig:
    return OpponentConfig(
        min_confident_samples=5,
        guard_drop_rate_threshold=0.6,
        base_jab_probability=0.35,
        base_cross_probability=0.25,
        adapted_counter_probability=0.7,
        punch_reach_m=1.15,
    )


def state(guard_left: float, guard_right: float) -> BodyState:
    joints = {
        "nose": np.array([0.0, -0.5, 2.0], dtype=np.float32),
        "left_shoulder": np.array([-0.2, -0.2, 2.1], dtype=np.float32),
        "right_shoulder": np.array([0.2, -0.2, 2.1], dtype=np.float32),
    }
    return BodyState(
        timestamp_ns=0,
        track_id=0,
        pose_confidence=0.9,
        stance=Stance.ORTHODOX,
        joints_3d=joints,
        joint_confidence={key: 0.9 for key in joints},
        guard={"left": guard_left, "right": guard_right},
    )


def test_high_guard_blocks_robot_jab():
    result, reason = score_robot_punch(
        state(0.8, 0.8),
        PlayerAction.HIGH_GUARD,
        OpponentAction.JAB,
        opponent_config(),
    )
    assert result == ExchangeResult.BLOCK
    assert "covered" in reason


def test_low_confidence_exchange_is_uncertain():
    low = state(0.1, 0.1)
    low.pose_confidence = 0.2
    result, _ = score_robot_punch(low, PlayerAction.NEUTRAL, OpponentAction.JAB, opponent_config())
    assert result == ExchangeResult.UNCERTAIN


def test_adaptation_triggers_cross_after_repeated_right_guard_drop():
    engine = AdaptationEngine(opponent_config())
    for _ in range(5):
        engine.observe(PlayerAction.JAB, {"left": 0.8, "right": 0.1})
    assert engine.preferred_counter() == OpponentAction.CROSS
    assert engine.to_json()["after_player_jab"]["right_guard_drop_rate"] == 1.0
