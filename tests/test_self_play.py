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


if __name__ == "__main__":
    unittest.main()
