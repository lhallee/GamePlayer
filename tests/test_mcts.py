import random
import unittest

from game_player.az.mcts import run_mcts, uniform_evaluator
from game_player.games.go import GoState


class MCTSTest(unittest.TestCase):
    def test_mcts_returns_policy_over_actions(self):
        state = GoState.new(board_size=3, komi=0.5)
        policy = run_mcts(
            state=state,
            evaluator=uniform_evaluator,
            simulations=4,
            rng=random.Random(0),
        )
        self.assertEqual(len(policy), state.action_size)
        self.assertAlmostEqual(sum(policy), 1.0)


if __name__ == "__main__":
    unittest.main()
