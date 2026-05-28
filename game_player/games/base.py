from __future__ import annotations

from typing import Protocol

Action = int
Player = int


class GameState(Protocol):
    current_player: Player

    @property
    def action_size(self) -> int:
        raise NotImplementedError

    @property
    def observation_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def legal_actions(self) -> list[Action]:
        raise NotImplementedError

    def apply_action(self, action: Action) -> "GameState":
        raise NotImplementedError

    def is_terminal(self) -> bool:
        raise NotImplementedError

    def rewards(self) -> dict[Player, float]:
        raise NotImplementedError

    def result_for_player(self, player: Player) -> float:
        raise NotImplementedError

    def area_scores(self) -> tuple[float, float]:
        raise NotImplementedError

    def ownership(self) -> tuple[Player, ...]:
        raise NotImplementedError

    def winner(self) -> Player:
        raise NotImplementedError

    def observation(self) -> list[float]:
        raise NotImplementedError
