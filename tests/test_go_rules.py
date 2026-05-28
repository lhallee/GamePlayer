import unittest

from game_player.games.go import BLACK, EMPTY, WHITE, GoState, point_to_action


class GoRulesTest(unittest.TestCase):
    def test_initial_legal_actions_include_pass(self):
        state = GoState.new(board_size=3, komi=0.5)
        self.assertEqual(len(state.legal_actions()), 10)
        self.assertEqual(state.pass_action, 9)

    def test_capture_removes_surrounded_stone(self):
        state = GoState.new(board_size=3, komi=0.5)
        center = point_to_action(1, 1, 3)
        state = state.apply_action(center)
        state = state.apply_action(point_to_action(0, 1, 3))
        state = state.apply_action(state.pass_action)
        state = state.apply_action(point_to_action(1, 0, 3))
        state = state.apply_action(state.pass_action)
        state = state.apply_action(point_to_action(1, 2, 3))
        state = state.apply_action(state.pass_action)
        state = state.apply_action(point_to_action(2, 1, 3))
        self.assertEqual(state.board[center], EMPTY)

    def test_captured_point_can_be_legal_again_when_it_has_a_liberty(self):
        previous_board = (
            EMPTY, BLACK, EMPTY,
            BLACK, WHITE, WHITE,
            EMPTY, BLACK, EMPTY,
        )
        current_board = (
            EMPTY, BLACK, EMPTY,
            BLACK, EMPTY, EMPTY,
            EMPTY, BLACK, EMPTY,
        )
        state = GoState(
            board_size=3,
            komi=0.5,
            board=current_board,
            current_player=WHITE,
            consecutive_passes=0,
            position_history=(previous_board, current_board),
        )
        self.assertTrue(state.is_legal_action(point_to_action(1, 1, 3)))

    def test_captured_point_can_be_replayed_as_tromp_taylor_suicide(self):
        state = GoState.new(board_size=3, komi=0.5)
        center = point_to_action(1, 1, 3)
        state = state.apply_action(center)
        state = state.apply_action(point_to_action(0, 1, 3))
        state = state.apply_action(state.pass_action)
        state = state.apply_action(point_to_action(1, 0, 3))
        state = state.apply_action(state.pass_action)
        state = state.apply_action(point_to_action(1, 2, 3))
        state = state.apply_action(state.pass_action)
        state = state.apply_action(point_to_action(2, 1, 3))
        self.assertEqual(state.board[center], EMPTY)
        self.assertTrue(state.is_legal_action(center))
        replayed = state.apply_action(center)
        self.assertEqual(replayed.board[center], EMPTY)

    def test_suicide_is_legal_under_tromp_taylor(self):
        board = (
            EMPTY, WHITE, EMPTY,
            WHITE, EMPTY, WHITE,
            EMPTY, WHITE, EMPTY,
        )
        state = GoState(
            board_size=3,
            komi=0.5,
            board=board,
            current_player=BLACK,
            consecutive_passes=0,
            position_history=(board,),
        )
        self.assertTrue(state.is_legal_action(point_to_action(1, 1, 3)))
        next_state = state.apply_action(point_to_action(1, 1, 3))
        self.assertEqual(next_state.board, board)

    def test_repeating_position_left_by_same_player_is_illegal(self):
        board = (
            EMPTY, WHITE, EMPTY,
            WHITE, EMPTY, WHITE,
            EMPTY, WHITE, EMPTY,
        )
        state = GoState(
            board_size=3,
            komi=0.5,
            board=board,
            current_player=BLACK,
            consecutive_passes=0,
            position_history=(board,),
            position_history_players=(BLACK,),
        )
        self.assertFalse(state.is_legal_action(point_to_action(1, 1, 3)))

    def test_repeating_position_left_by_other_player_is_legal(self):
        board = (
            EMPTY, WHITE, EMPTY,
            WHITE, EMPTY, WHITE,
            EMPTY, WHITE, EMPTY,
        )
        state = GoState(
            board_size=3,
            komi=0.5,
            board=board,
            current_player=BLACK,
            consecutive_passes=0,
            position_history=(board,),
            position_history_players=(WHITE,),
        )
        self.assertTrue(state.is_legal_action(point_to_action(1, 1, 3)))

    def test_two_passes_end_game(self):
        state = GoState.new(board_size=3, komi=0.5)
        state = state.apply_action(state.pass_action)
        self.assertFalse(state.is_terminal())
        state = state.apply_action(state.pass_action)
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.area_scores(), (9.0, 9.5))
        self.assertEqual(state.winner(), WHITE)

    def test_observation_is_current_player_perspective(self):
        state = GoState.new(board_size=3, komi=0.5)
        state = state.apply_action(point_to_action(0, 0, 3))
        observation = state.observation()
        self.assertEqual(len(observation), 18)
        own_channel = observation[:9]
        opponent_channel = observation[9:]
        self.assertEqual(sum(own_channel), 0.0)
        self.assertEqual(sum(opponent_channel), 1.0)


if __name__ == "__main__":
    unittest.main()
