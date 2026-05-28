from __future__ import annotations

import argparse
import json
from pathlib import Path

from game_player.az.evaluation import evaluate_model_against_random_weights
from game_player.az.training import TrainingConfig, run_training
from game_player.hub import save_mlp_model
from game_player.models.mlp import MLPPolicyValueNet


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
        "--hidden-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--self-play-games",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--validation-games",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--validation-max-moves",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = MLPPolicyValueNet(
        board_size=args.board_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        include_pass=True,
    )
    random_weight_baseline = MLPPolicyValueNet(
        board_size=args.board_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        include_pass=True,
    )
    config = TrainingConfig(
        board_size=args.board_size,
        komi=args.komi,
        self_play_games=args.self_play_games,
        mcts_simulations=args.mcts_simulations,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        max_moves=args.max_moves,
        seed=args.seed,
    )
    metrics = run_training(
        model=model,
        config=config,
        device=args.device,
    )
    if args.validation_games > 0:
        metrics.update(
            evaluate_model_against_random_weights(
                candidate_model=model,
                baseline_model=random_weight_baseline,
                board_size=args.board_size,
                komi=args.komi,
                games=args.validation_games,
                max_moves=args.validation_max_moves,
                device=args.device,
            )
        )
    if args.output_dir is not None:
        save_mlp_model(model, args.output_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
