from __future__ import annotations

from down_b.boxing.actions import ActionRecognizer
from down_b.boxing.adaptation import AdaptationEngine
from down_b.boxing.hits import score_player_punch, score_robot_punch
from down_b.boxing.opponent import OpponentPolicy
from down_b.config import OpponentConfig, ThresholdConfig
from down_b.types import BodyState, ExchangeResult, GameFrame


class GameEngine:
    def __init__(self, thresholds: ThresholdConfig, opponent_config: OpponentConfig):
        self.recognizer = ActionRecognizer(thresholds)
        self.adaptation = AdaptationEngine(opponent_config)
        self.policy = OpponentPolicy(opponent_config, self.adaptation)
        self.opponent_config = opponent_config
        self.score = {"player": 0, "opponent": 0}

    def update(self, state: BodyState) -> GameFrame:
        player_action, _features = self.recognizer.update(state)
        self.adaptation.observe(player_action, state.guard)
        opponent_action = self.policy.update(state.timestamp_ns, player_action)
        robot_result, robot_reason = score_robot_punch(state, player_action, opponent_action, self.opponent_config)
        player_result, player_reason = score_player_punch(player_action, state)
        result = robot_result
        reason = robot_reason
        if player_result == ExchangeResult.HIT:
            self.score["player"] += 1
            result = player_result
            reason = player_reason
        elif robot_result == ExchangeResult.HIT:
            self.score["opponent"] += 1
        return GameFrame(
            timestamp_ns=state.timestamp_ns,
            player_action=player_action,
            opponent_action=opponent_action,
            exchange_result=result,
            score=dict(self.score),
            confidence=state.pose_confidence,
            reason=reason,
            adaptation=self.adaptation.to_json(),
        )
