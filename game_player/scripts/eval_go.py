from __future__ import annotations

import argparse
import json

import torch

from game_player.az.evaluation import (
    evaluate_model_against_random_agent,
    evaluate_model_against_random_weights,
)
from game_player.models.mlp import MLPPolicyValueNet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--opponent",
        choices=["random-agent", "random-weights"],
        default="random-weights",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--komi",
        type=float,
        default=7.5,
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
        "--seed",
        type=int,
        default=0,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    model = MLPPolicyValueNet.from_pretrained(args.model)

    if args.opponent == "random-agent":
        metrics = evaluate_model_against_random_agent(
            candidate_model=model,
            board_size=model.board_size,
            komi=args.komi,
            games=args.games,
            max_moves=args.max_moves,
            device=args.device,
            seed=args.seed,
        )
    elif args.opponent == "random-weights":
        baseline = MLPPolicyValueNet(
            board_size=model.board_size,
            hidden_size=model.hidden_size,
            depth=model.depth,
            include_pass=model.include_pass,
        )
        metrics = evaluate_model_against_random_weights(
            candidate_model=model,
            baseline_model=baseline,
            board_size=model.board_size,
            komi=args.komi,
            games=args.games,
            max_moves=args.max_moves,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown opponent: {args.opponent}")

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
