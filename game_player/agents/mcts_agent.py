from __future__ import annotations

import random
from collections.abc import Callable

from game_player.az.mcts import Evaluator, run_mcts
from game_player.games.base import Action, GameState


class MCTSAgent:
    def __init__(
        self,
        evaluator: Evaluator,
        simulations: int = 64,
        c_puct: float = 1.5,
        seed: int | None = None,
    ) -> None:
        assert simulations > 0
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.rng = random.Random(seed)

    def select_action(self, state: GameState) -> Action:
        policy = run_mcts(
            state=state,
            evaluator=self.evaluator,
            simulations=self.simulations,
            c_puct=self.c_puct,
            temperature=0.0,
            rng=self.rng,
        )
        best_action = max(range(len(policy)), key=policy.__getitem__)
        return int(best_action)
