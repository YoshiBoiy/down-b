from __future__ import annotations

import random
from dataclasses import dataclass

from down_b.boxing.adaptation import AdaptationEngine
from down_b.config import OpponentConfig
from down_b.types import OpponentAction, PlayerAction


@dataclass(slots=True)
class OpponentState:
    action: OpponentAction = OpponentAction.WAIT
    phase: str = "IDLE"
    phase_until_ns: int = 0


class OpponentPolicy:
    def __init__(self, config: OpponentConfig, adaptation: AdaptationEngine, seed: int = 7):
        self.config = config
        self.adaptation = adaptation
        self.rng = random.Random(seed)
        self.state = OpponentState()

    def update(self, timestamp_ns: int, player_action: PlayerAction) -> OpponentAction:
        if timestamp_ns < self.state.phase_until_ns:
            return self.state.action
        if self.state.phase != "IDLE":
            self.state = OpponentState()
            return OpponentAction.WAIT
        action = self._choose_action(player_action)
        self.state = OpponentState(action=action, phase="TELEGRAPH", phase_until_ns=timestamp_ns + 280_000_000)
        return action

    def _choose_action(self, player_action: PlayerAction) -> OpponentAction:
        counter = self.adaptation.preferred_counter()
        if counter is not None and player_action == PlayerAction.JAB:
            if self.rng.random() < self.config.adapted_counter_probability:
                return counter
        roll = self.rng.random()
        if roll < self.config.base_jab_probability:
            return OpponentAction.JAB
        if roll < self.config.base_jab_probability + self.config.base_cross_probability:
            return OpponentAction.CROSS
        if player_action in {PlayerAction.JAB, PlayerAction.CROSS}:
            return OpponentAction.BLOCK
        return OpponentAction.WAIT
