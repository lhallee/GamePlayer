from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from game_player.az.evaluation import (
    HybridPolicyRolloutEvaluator,
    RandomRolloutEvaluator,
    evaluate_model_against_random_agent,
    evaluate_model_against_random_weights,
)
from game_player.az.mcts import Evaluator
from game_player.az.training import TrainingConfig, run_training_iterations
from game_player.hub import save_mlp_model
from game_player.models.mlp import MLPPolicyValueNet
from game_player.reporting import write_training_report


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
        "--iterations",
        type=int,
        default=1,
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
        "--c-puct",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--temperature-drop-move",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--value-target",
        choices=["outcome", "score"],
        default="outcome",
    )
    parser.add_argument(
        "--mcts-evaluator",
        choices=["neural", "rollout", "hybrid"],
        default="neural",
    )
    parser.add_argument(
        "--rollout-games",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--rollout-max-moves",
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
        "--validation-opponent",
        choices=["random-agent", "random-weights"],
        default="random-weights",
    )
    parser.add_argument(
        "--validation-max-moves",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--target-win-rate",
        type=float,
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
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--best-checkpoint-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="Go MLP AlphaZero Run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
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
        iterations=args.iterations,
        self_play_games=args.self_play_games,
        mcts_simulations=args.mcts_simulations,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        max_moves=args.max_moves,
        seed=args.seed,
        c_puct=args.c_puct,
        temperature=args.temperature,
        temperature_drop_move=args.temperature_drop_move,
        dirichlet_alpha=args.dirichlet_alpha,
        exploration_fraction=args.exploration_fraction,
        value_target=args.value_target,
    )
    metrics_by_iteration = run_training_cli(
        model=model,
        config=config,
        baseline_model=random_weight_baseline,
        validation_games=args.validation_games,
        validation_opponent=args.validation_opponent,
        validation_max_moves=args.validation_max_moves,
        target_win_rate=args.target_win_rate,
        checkpoint_dir=args.checkpoint_dir,
        best_checkpoint_dir=args.best_checkpoint_dir,
        save_every=args.save_every,
        metrics_path=args.metrics_path,
        report_dir=args.report_dir,
        run_config={
            "board_size": args.board_size,
            "komi": args.komi,
            "hidden_size": args.hidden_size,
            "depth": args.depth,
            "iterations": args.iterations,
            "self_play_games": args.self_play_games,
            "mcts_simulations": args.mcts_simulations,
            "batch_size": args.batch_size,
            "train_steps": args.train_steps,
            "learning_rate": args.learning_rate,
            "c_puct": args.c_puct,
            "temperature": args.temperature,
            "temperature_drop_move": args.temperature_drop_move,
            "dirichlet_alpha": args.dirichlet_alpha,
            "exploration_fraction": args.exploration_fraction,
            "max_moves": args.max_moves,
            "value_target": args.value_target,
            "mcts_evaluator": args.mcts_evaluator,
            "rollout_games": args.rollout_games,
            "rollout_max_moves": args.rollout_max_moves,
            "device": args.device,
            "validation_games": args.validation_games,
            "validation_opponent": args.validation_opponent,
            "validation_max_moves": args.validation_max_moves,
            "target_win_rate": args.target_win_rate,
            "seed": args.seed,
            "output_dir": args.output_dir,
            "checkpoint_dir": args.checkpoint_dir,
            "best_checkpoint_dir": args.best_checkpoint_dir,
            "save_every": args.save_every,
            "metrics_path": args.metrics_path,
            "report_dir": args.report_dir,
            "run_name": args.run_name,
        },
        run_name=args.run_name,
        device=args.device,
        mcts_evaluator=args.mcts_evaluator,
        rollout_games=args.rollout_games,
        rollout_max_moves=args.rollout_max_moves,
    )
    metrics = metrics_by_iteration[-1]
    if args.output_dir is not None:
        save_mlp_model(model, args.output_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def run_training_cli(
    model: MLPPolicyValueNet,
    config: TrainingConfig,
    baseline_model: MLPPolicyValueNet,
    validation_games: int,
    validation_opponent: str,
    validation_max_moves: int | None,
    target_win_rate: float | None,
    checkpoint_dir: Path | None,
    best_checkpoint_dir: Path | None,
    save_every: int,
    metrics_path: Path | None,
    report_dir: Path | None,
    run_config: dict[str, object],
    run_name: str,
    device: str,
    mcts_evaluator: str = "neural",
    rollout_games: int = 1,
    rollout_max_moves: int | None = None,
) -> list[dict[str, float]]:
    assert save_every > 0
    assert rollout_games > 0
    assert mcts_evaluator in ("neural", "rollout", "hybrid")
    if target_win_rate is not None:
        assert 0.0 <= target_win_rate <= 1.0

    best_win_rate = -1.0
    metrics_history: list[dict[str, float]] = []

    def validation_fn(candidate_model: MLPPolicyValueNet) -> dict[str, float]:
        if validation_games == 0:
            return {}
        if validation_opponent == "random-agent":
            return evaluate_model_against_random_agent(
                candidate_model=candidate_model,
                board_size=config.board_size,
                komi=config.komi,
                games=validation_games,
                max_moves=validation_max_moves,
                device=device,
                seed=config.seed,
            )
        if validation_opponent == "random-weights":
            return evaluate_model_against_random_weights(
                candidate_model=candidate_model,
                baseline_model=baseline_model,
                board_size=config.board_size,
                komi=config.komi,
                games=validation_games,
                max_moves=validation_max_moves,
                device=device,
            )
        raise ValueError(f"Unknown validation opponent: {validation_opponent}")

    def metrics_callback(metrics: dict[str, float]) -> None:
        nonlocal best_win_rate
        iteration = int(metrics["iteration"])
        if "validation_candidate_win_rate" in metrics:
            win_rate = metrics["validation_candidate_win_rate"]
            if best_checkpoint_dir is not None and win_rate > best_win_rate:
                best_win_rate = win_rate
                save_mlp_model(model, best_checkpoint_dir)
            if target_win_rate is not None and win_rate >= target_win_rate:
                metrics["target_reached"] = 1.0
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        metrics_history.append(dict(metrics))
        if report_dir is not None:
            write_training_report(
                metrics=metrics_history,
                report_dir=report_dir,
                run_config=run_config,
                title=run_name,
            )
        if checkpoint_dir is not None and iteration % save_every == 0:
            save_mlp_model(
                model,
                checkpoint_dir / f"iteration-{iteration:04d}",
            )

    search_evaluator = build_search_evaluator(
        model=model,
        mcts_evaluator=mcts_evaluator,
        rollout_games=rollout_games,
        rollout_max_moves=rollout_max_moves,
        seed=config.seed,
        device=device,
    )

    return run_training_iterations(
        model=model,
        config=config,
        device=device,
        validation_fn=validation_fn,
        metrics_callback=metrics_callback,
        search_evaluator=search_evaluator,
    )


def build_search_evaluator(
    model: MLPPolicyValueNet,
    mcts_evaluator: str,
    rollout_games: int,
    rollout_max_moves: int | None,
    seed: int,
    device: str,
) -> Evaluator | None:
    if mcts_evaluator == "neural":
        return None
    if mcts_evaluator == "rollout":
        return RandomRolloutEvaluator(
            rollouts=rollout_games,
            max_moves=rollout_max_moves,
            seed=seed,
        )
    if mcts_evaluator == "hybrid":
        return HybridPolicyRolloutEvaluator(
            model=model,
            rollouts=rollout_games,
            max_moves=rollout_max_moves,
            device=device,
            seed=seed,
        )
    raise ValueError(f"Unknown MCTS evaluator: {mcts_evaluator}")


if __name__ == "__main__":
    main()
