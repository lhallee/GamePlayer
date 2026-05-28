import unittest

from game_player.az.evaluation import (
    RandomRolloutEvaluator,
    TrompTaylorTacticalEvaluator,
)
from game_player.games.go import GoState


class RandomRolloutEvaluatorTest(unittest.TestCase):
    def test_rollout_evaluator_returns_uniform_priors_and_value(self):
        state = GoState.new(board_size=3, komi=0.5)
        evaluator = RandomRolloutEvaluator(
            rollouts=2,
            max_moves=8,
            seed=0,
        )

        priors, value = evaluator(state)

        self.assertEqual(set(priors), set(state.legal_actions()))
        self.assertAlmostEqual(sum(priors.values()), 1.0)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_tactical_evaluator_returns_priors_and_value(self):
        state = GoState.new(board_size=3, komi=0.5)
        evaluator = TrompTaylorTacticalEvaluator()

        priors, value = evaluator(state)

        self.assertEqual(set(priors), set(state.legal_actions()))
        self.assertAlmostEqual(sum(priors.values()), 1.0)
        self.assertGreater(priors[0], priors[state.pass_action])
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
