from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from down_b.boxing.guard import high_guard
from down_b.config import OpponentConfig
from down_b.types import BodyState, ExchangeResult, OpponentAction, PlayerAction


@dataclass(frozen=True, slots=True)
class TargetVolume:
    center: np.ndarray
    radius_m: float

    def contains(self, point: np.ndarray) -> bool:
        return float(np.linalg.norm(point - self.center)) <= self.radius_m


def head_volume(state: BodyState) -> TargetVolume | None:
    nose = state.joint("nose")
    if nose is None:
        return None
    return TargetVolume(center=nose, radius_m=0.16)


def score_robot_punch(
    state: BodyState,
    player_action: PlayerAction,
    opponent_action: OpponentAction,
    config: OpponentConfig,
) -> tuple[ExchangeResult, str]:
    if opponent_action not in {OpponentAction.JAB, OpponentAction.CROSS, OpponentAction.JAB_CROSS}:
        return ExchangeResult.MISS, "opponent did not punch"
    if state.pose_confidence < 0.45:
        return ExchangeResult.UNCERTAIN, "low perception confidence"
    if player_action in {PlayerAction.LEFT_SLIP, PlayerAction.RIGHT_SLIP, PlayerAction.DUCK}:
        return ExchangeResult.MISS, "player displaced target"
    if high_guard(state.guard):
        return ExchangeResult.BLOCK, "both wrists covered the head"
    volume = head_volume(state)
    if volume is None:
        return ExchangeResult.UNCERTAIN, "missing head joint"
    shoulder = state.joint("left_shoulder")
    if shoulder is None:
        shoulder = state.joint("right_shoulder")
    if shoulder is not None and float(np.linalg.norm(volume.center - shoulder)) > config.punch_reach_m:
        return ExchangeResult.MISS, "player out of range"
    return ExchangeResult.HIT, "virtual fist entered head volume"


def score_player_punch(player_action: PlayerAction, state: BodyState) -> tuple[ExchangeResult, str]:
    if player_action not in {PlayerAction.JAB, PlayerAction.CROSS}:
        return ExchangeResult.MISS, "player did not punch"
    if state.pose_confidence < 0.45:
        return ExchangeResult.UNCERTAIN, "low perception confidence"
    return ExchangeResult.HIT, "player punch reached virtual target"
