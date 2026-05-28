from __future__ import annotations

import argparse
import json

from game_player.agents.base import Agent
from game_player.agents.random_agent import RandomAgent
from game_player.az.evaluation import TrompTaylorTacticalEvaluator, play_match
from game_player.games.base import Action, GameState
from game_player.games.go import BLACK, WHITE, GoState


class TacticalPolicyAgent(Agent):
    def __init__(self, evaluator: TrompTaylorTacticalEvaluator) -> None:
        self.evaluator = evaluator

    def select_action(self, state: GameState) -> Action:
        priors, _ = self.evaluator(state)
        assert priors
        return max(priors, key=lambda action: priors[action])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--board-size",
        type=int,
        default=19,
    )
    parser.add_argument(
        "--komi",
        type=float,
        default=7.5,
    )
    parser.add_argument(
        "--games",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--tactical-area-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--tactical-prior-temperature",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--tactical-capture-weight",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--tactical-liberty-weight",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--tactical-stone-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--tactical-adjacent-opponent-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--tactical-adjacent-own-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--tactical-opponent-liberty-reduction-weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--tactical-opponent-atari-weight",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--tactical-self-atari-weight",
        type=float,
        default=-15.0,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_tactical_against_random(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def evaluate_tactical_against_random(
    args: argparse.Namespace,
) -> dict[str, float]:
    assert args.games > 0
    tactical_agent = TacticalPolicyAgent(
        TrompTaylorTacticalEvaluator(
            prior_temperature=args.tactical_prior_temperature,
            area_weight=args.tactical_area_weight,
            capture_weight=args.tactical_capture_weight,
            liberty_weight=args.tactical_liberty_weight,
            stone_weight=args.tactical_stone_weight,
            adjacent_opponent_weight=args.tactical_adjacent_opponent_weight,
            adjacent_own_weight=args.tactical_adjacent_own_weight,
            opponent_liberty_reduction_weight=(
                args.tactical_opponent_liberty_reduction_weight
            ),
            opponent_atari_weight=args.tactical_opponent_atari_weight,
            self_atari_weight=args.tactical_self_atari_weight,
        )
    )
    random_agent = RandomAgent(seed=args.seed)
    candidate_wins = 0
    candidate_black_games = 0
    candidate_black_wins = 0
    candidate_white_games = 0
    candidate_white_wins = 0
    candidate_score_margin = 0.0
    total_moves = 0
    terminal_games = 0

    for game_idx in range(args.games):
        state = GoState.new(board_size=args.board_size, komi=args.komi)
        candidate_is_black = game_idx % 2 == 0
        if candidate_is_black:
            candidate_black_games += 1
            result = play_match(
                state=state,
                black_agent=tactical_agent,
                white_agent=random_agent,
                max_moves=args.max_moves,
            )
            win = int(result.winner == BLACK)
            candidate_black_wins += win
            candidate_wins += win
            candidate_score_margin += result.black_score - result.white_score
        else:
            candidate_white_games += 1
            result = play_match(
                state=state,
                black_agent=random_agent,
                white_agent=tactical_agent,
                max_moves=args.max_moves,
            )
            win = int(result.winner == WHITE)
            candidate_white_wins += win
            candidate_wins += win
            candidate_score_margin += result.white_score - result.black_score
        total_moves += result.moves
        terminal_games += int(result.terminal)

    return {
        "validation_games": float(args.games),
        "validation_candidate_wins": float(candidate_wins),
        "validation_candidate_win_rate": candidate_wins / args.games,
        "validation_candidate_avg_score_margin": (
            candidate_score_margin / args.games
        ),
        "validation_avg_moves": total_moves / args.games,
        "validation_terminal_games": float(terminal_games),
        "validation_terminal_rate": terminal_games / args.games,
        "validation_candidate_black_games": float(candidate_black_games),
        "validation_candidate_black_wins": float(candidate_black_wins),
        "validation_candidate_black_win_rate": _win_rate(
            candidate_black_wins,
            candidate_black_games,
        ),
        "validation_candidate_white_games": float(candidate_white_games),
        "validation_candidate_white_wins": float(candidate_white_wins),
        "validation_candidate_white_win_rate": _win_rate(
            candidate_white_wins,
            candidate_white_games,
        ),
    }


def _win_rate(wins: int, games: int) -> float:
    if games == 0:
        return 0.0
    return wins / games


if __name__ == "__main__":
    main()
