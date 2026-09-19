from __future__ import annotations

from dataclasses import dataclass

from down_b.config import OpponentConfig
from down_b.types import OpponentAction, PlayerAction


@dataclass(slots=True)
class ContextStats:
    samples: int = 0
    right_guard_drops: int = 0
    stationary_head: int = 0
    total_recovery_ms: float = 0.0

    @property
    def right_guard_drop_rate(self) -> float:
        return self.right_guard_drops / self.samples if self.samples else 0.0

    @property
    def mean_recovery_ms(self) -> float:
        return self.total_recovery_ms / self.samples if self.samples else 0.0


class AdaptationEngine:
    def __init__(self, config: OpponentConfig):
        self.config = config
        self.after_player_jab = ContextStats()
        self.active_reason = ""

    def observe(self, action: PlayerAction, guard: dict[str, float], recovery_ms: float = 0.0) -> None:
        if action != PlayerAction.JAB:
            return
        self.after_player_jab.samples += 1
        if guard.get("right", 1.0) < 0.35:
            self.after_player_jab.right_guard_drops += 1
        self.after_player_jab.total_recovery_ms += recovery_ms
        self._refresh_reason()

    def preferred_counter(self) -> OpponentAction | None:
        stats = self.after_player_jab
        if stats.samples < self.config.min_confident_samples:
            return None
        if stats.right_guard_drop_rate > self.config.guard_drop_rate_threshold:
            return OpponentAction.CROSS
        return None

    def multiplier(self, action: OpponentAction) -> float:
        counter = self.preferred_counter()
        if counter is None:
            return 1.0
        return self.config.adapted_counter_probability if action == counter else 1.0

    def to_json(self) -> dict[str, object]:
        stats = self.after_player_jab
        return {
            "after_player_jab": {
                "samples": stats.samples,
                "right_guard_drop_rate": round(stats.right_guard_drop_rate, 3),
                "mean_recovery_ms": round(stats.mean_recovery_ms, 1),
            },
            "active_reason": self.active_reason,
            "preferred_counter": self.preferred_counter().value if self.preferred_counter() else None,
        }

    def _refresh_reason(self) -> None:
        counter = self.preferred_counter()
        if counter == OpponentAction.CROSS:
            rate = self.after_player_jab.right_guard_drop_rate
            self.active_reason = f"right guard dropped after jab in {rate:.0%} of confident samples"
        else:
            self.active_reason = ""
