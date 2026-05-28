import random
import unittest

from game_player.az.mcts import uniform_evaluator
from game_player.az.self_play import play_self_play_game
from game_player.games.go import GoState


class SelfPlayTest(unittest.TestCase):
    def test_self_play_produces_samples(self):
        state = GoState.new(board_size=3, komi=0.5)
        samples, final_state = play_self_play_game(
            initial_state=state,
            evaluator=uniform_evaluator,
            simulations=2,
            max_moves=4,
            rng=random.Random(0),
        )
        self.assertGreater(len(samples), 0)
        self.assertLessEqual(len(samples), 4)
        self.assertEqual(len(samples[0].policy), state.action_size)
        self.assertEqual(len(samples[0].observation), 18)
        self.assertIsInstance(final_state, GoState)

    def test_self_play_can_use_score_value_target(self):
        state = GoState.new(board_size=3, komi=0.5)
        samples, _ = play_self_play_game(
            initial_state=state,
            evaluator=uniform_evaluator,
            simulations=2,
            max_moves=4,
            value_target="score",
            rng=random.Random(0),
        )
        self.assertGreater(len(samples), 0)
        self.assertGreaterEqual(samples[0].value, -1.0)
        self.assertLessEqual(samples[0].value, 1.0)

    def test_self_play_can_emit_ownership_targets(self):
        state = GoState.new(board_size=3, komi=0.5)
        samples, _ = play_self_play_game(
            initial_state=state,
            evaluator=uniform_evaluator,
            simulations=2,
            max_moves=4,
            ownership_target=True,
            rng=random.Random(0),
        )
        self.assertGreater(len(samples), 0)
        self.assertIsNotNone(samples[0].ownership)
        assert samples[0].ownership is not None
        self.assertEqual(len(samples[0].ownership), 9)
        self.assertTrue(
            all(value in (-1.0, 0.0, 1.0) for value in samples[0].ownership)
        )


if __name__ == "__main__":
    unittest.main()
