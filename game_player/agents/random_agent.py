from __future__ import annotations

import random

from game_player.games.base import Action, GameState


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, state: GameState) -> Action:
        legal_actions = state.legal_actions()
        assert legal_actions
        return self.rng.choice(legal_actions)
