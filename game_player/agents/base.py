from __future__ import annotations

from typing import Protocol

from game_player.games.base import Action, GameState


class Agent(Protocol):
    def select_action(self, state: GameState) -> Action:
        raise NotImplementedError
