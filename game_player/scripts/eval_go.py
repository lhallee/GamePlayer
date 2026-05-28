from __future__ import annotations

import argparse
import json

import torch

from game_player.az.evaluation import (
    evaluate_model_against_random_agent,
    evaluate_model_against_random_weights,
)
from game_player.models.conv import ConvPolicyValueNet
from game_player.models.mlp import MLPPolicyValueNet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model-type",
        choices=["mlp", "conv"],
        default="mlp",
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
    model = load_model(args.model, args.model_type)

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
        baseline = build_random_weight_baseline(model, args.model_type)
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


def load_model(path_or_repo_id: str, model_type: str) -> torch.nn.Module:
    if model_type == "mlp":
        return MLPPolicyValueNet.from_pretrained(path_or_repo_id)
    if model_type == "conv":
        return ConvPolicyValueNet.from_pretrained(path_or_repo_id)
    raise ValueError(f"Unknown model type: {model_type}")


def build_random_weight_baseline(
    model: torch.nn.Module,
    model_type: str,
) -> torch.nn.Module:
    if model_type == "mlp":
        assert isinstance(model, MLPPolicyValueNet)
        return MLPPolicyValueNet(
            board_size=model.board_size,
            hidden_size=model.hidden_size,
            depth=model.depth,
            include_pass=model.include_pass,
        )
    if model_type == "conv":
        assert isinstance(model, ConvPolicyValueNet)
        return ConvPolicyValueNet(
            board_size=model.board_size,
            channels=model.channels,
            blocks=model.blocks,
            include_pass=model.include_pass,
        )
    raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()
