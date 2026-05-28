from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from math import isqrt

import torch
import torch.nn.functional as F

from game_player.az.evaluation import NeuralPolicyValueEvaluator
from game_player.az.mcts import Evaluator
from game_player.az.replay import ReplayBuffer
from game_player.az.self_play import SelfPlaySample, play_self_play_game
from game_player.games.go import GoState


@dataclass(frozen=True)
class TrainingConfig:
    board_size: int = 19
    komi: float = 7.5
    iterations: int = 1
    self_play_games: int = 1
    mcts_simulations: int = 64
    replay_capacity: int = 100_000
    batch_size: int = 64
    train_steps: int = 32
    learning_rate: float = 1e-3
    max_moves: int | None = None
    seed: int = 0
    c_puct: float = 1.5
    temperature: float = 1.0
    temperature_drop_move: int = 30
    dirichlet_alpha: float | None = 0.03
    exploration_fraction: float = 0.25
    value_target: str = "outcome"
    augment_symmetries: bool = False
    ownership_loss_weight: float = 0.0
    validation_interval: int = 1


ValidationFn = Callable[[torch.nn.Module], dict[str, float]]
MetricsCallback = Callable[[dict[str, float]], None]


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: list[SelfPlaySample],
    device: str = "cpu",
    augment_symmetries: bool = False,
    ownership_loss_weight: float = 0.0,
) -> dict[str, float]:
    assert batch
    assert ownership_loss_weight >= 0.0
    torch_device = torch.device(device)
    model.train()

    observations = torch.tensor(
        [sample.observation for sample in batch],
        dtype=torch.float32,
        device=torch_device,
    )
    target_policies = torch.tensor(
        [sample.policy for sample in batch],
        dtype=torch.float32,
        device=torch_device,
    )
    target_values = torch.tensor(
        [sample.value for sample in batch],
        dtype=torch.float32,
        device=torch_device,
    )
    if augment_symmetries:
        observations, target_policies, target_ownerships = _augment_go_symmetries(
            observations,
            target_policies,
            _ownership_tensor(batch, torch_device, ownership_loss_weight),
        )
    else:
        target_ownerships = _ownership_tensor(batch, torch_device, ownership_loss_weight)

    output = model(observations)
    log_policy = F.log_softmax(output.policy_logits, dim=-1)
    policy_loss = -(target_policies * log_policy).sum(dim=-1).mean()
    value_loss = F.mse_loss(output.value, target_values)
    ownership_loss = torch.tensor(0.0, device=torch_device)
    if ownership_loss_weight > 0.0:
        assert target_ownerships is not None
        assert output.ownership is not None
        ownership_loss = F.mse_loss(
            output.ownership.flatten(start_dim=1),
            target_ownerships,
        )
    loss = policy_loss + value_loss + ownership_loss_weight * ownership_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "ownership_loss": float(ownership_loss.detach().cpu().item()),
    }


def run_training(
    model: torch.nn.Module,
    config: TrainingConfig,
    device: str = "cpu",
) -> dict[str, float]:
    metrics = run_training_iterations(
        model=model,
        config=config,
        device=device,
    )
    assert metrics
    return metrics[-1]


def run_training_iterations(
    model: torch.nn.Module,
    config: TrainingConfig,
    device: str = "cpu",
    validation_fn: ValidationFn | None = None,
    metrics_callback: MetricsCallback | None = None,
    search_evaluator: Evaluator | None = None,
) -> list[dict[str, float]]:
    assert config.iterations > 0
    assert config.self_play_games > 0
    assert config.mcts_simulations > 0
    assert config.train_steps >= 0
    assert config.c_puct > 0.0
    assert config.temperature >= 0.0
    assert config.temperature_drop_move >= 0
    assert 0.0 <= config.exploration_fraction <= 1.0
    assert config.value_target in ("outcome", "score")
    assert config.ownership_loss_weight >= 0.0
    assert config.validation_interval > 0

    rng = random.Random(config.seed)
    model.to(torch.device(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(capacity=config.replay_capacity)
    if search_evaluator is None:
        evaluator = NeuralPolicyValueEvaluator(model=model, device=device)
    else:
        evaluator = search_evaluator

    all_metrics: list[dict[str, float]] = []
    for iteration in range(1, config.iterations + 1):
        metrics = _run_training_iteration(
            model=model,
            optimizer=optimizer,
            replay=replay,
            evaluator=evaluator,
            config=config,
            device=device,
            rng=rng,
        )
        metrics["iteration"] = float(iteration)
        should_validate = (
            iteration % config.validation_interval == 0
            or iteration == config.iterations
        )
        if validation_fn is not None and should_validate:
            metrics.update(validation_fn(model))
        if metrics_callback is not None:
            metrics_callback(metrics)
        all_metrics.append(metrics)
        if metrics["target_reached"] == 1.0:
            break

    return all_metrics


def _run_training_iteration(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    evaluator: NeuralPolicyValueEvaluator,
    config: TrainingConfig,
    device: str,
    rng: random.Random,
) -> dict[str, float]:
    total_samples = 0
    for _ in range(config.self_play_games):
        state = GoState.new(board_size=config.board_size, komi=config.komi)
        samples, _ = play_self_play_game(
            initial_state=state,
            evaluator=evaluator,
            simulations=config.mcts_simulations,
            c_puct=config.c_puct,
            temperature=config.temperature,
            temperature_drop_move=config.temperature_drop_move,
            dirichlet_alpha=config.dirichlet_alpha,
            exploration_fraction=config.exploration_fraction,
            max_moves=config.max_moves,
            value_target=config.value_target,
            ownership_target=config.ownership_loss_weight > 0.0,
            rng=rng,
        )
        replay.add_many(samples)
        total_samples += len(samples)

    metrics: dict[str, float] = {
        "self_play_samples": float(total_samples),
        "replay_size": float(len(replay)),
        "train_steps_completed": 0.0,
        "target_reached": 0.0,
    }
    if len(replay) < config.batch_size or config.train_steps == 0:
        return metrics

    running_loss = 0.0
    running_policy_loss = 0.0
    running_value_loss = 0.0
    running_ownership_loss = 0.0
    for _ in range(config.train_steps):
        batch = replay.sample(config.batch_size, rng=rng)
        step_metrics = train_step(
            model,
            optimizer,
            batch,
            device=device,
            augment_symmetries=config.augment_symmetries,
            ownership_loss_weight=config.ownership_loss_weight,
        )
        running_loss += step_metrics["loss"]
        running_policy_loss += step_metrics["policy_loss"]
        running_value_loss += step_metrics["value_loss"]
        running_ownership_loss += step_metrics["ownership_loss"]

    metrics["train_steps_completed"] = float(config.train_steps)
    metrics["loss"] = running_loss / config.train_steps
    metrics["policy_loss"] = running_policy_loss / config.train_steps
    metrics["value_loss"] = running_value_loss / config.train_steps
    metrics["ownership_loss"] = running_ownership_loss / config.train_steps
    metrics["replay_size"] = float(len(replay))
    return metrics


def _augment_go_symmetries(
    observations: torch.Tensor,
    target_policies: torch.Tensor,
    target_ownerships: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    assert observations.dim() == 2
    assert target_policies.dim() == 2
    batch_size = observations.shape[0]
    board_actions = target_policies.shape[1] - 1
    board_size = isqrt(board_actions)
    assert board_size * board_size == board_actions
    assert observations.shape[1] == 2 * board_actions

    observation_boards = observations.reshape(batch_size, 2, board_size, board_size)
    policy_boards = target_policies[:, :board_actions].reshape(
        batch_size,
        board_size,
        board_size,
    )
    pass_policy = target_policies[:, board_actions:]

    augmented_observations = []
    augmented_policy_boards = []
    augmented_ownership_boards = []
    for sample_idx in range(batch_size):
        transform_idx = int(torch.randint(0, 8, (1,)).item())
        augmented_observations.append(
            _apply_d4_transform(
                observation_boards[sample_idx],
                transform_idx,
            )
        )
        augmented_policy_boards.append(
            _apply_d4_transform(
                policy_boards[sample_idx],
                transform_idx,
            )
        )
        if target_ownerships is not None:
            ownership_board = target_ownerships[sample_idx].reshape(
                board_size,
                board_size,
            )
            augmented_ownership_boards.append(
                _apply_d4_transform(
                    ownership_board,
                    transform_idx,
                )
            )

    stacked_observations = torch.stack(augmented_observations).reshape(
        batch_size,
        2 * board_actions,
    )
    stacked_policy_boards = torch.stack(augmented_policy_boards).reshape(
        batch_size,
        board_actions,
    )
    augmented_ownerships = None
    if target_ownerships is not None:
        augmented_ownerships = torch.stack(augmented_ownership_boards).reshape(
            batch_size,
            board_actions,
        )
    return (
        stacked_observations,
        torch.cat([stacked_policy_boards, pass_policy], dim=1),
        augmented_ownerships,
    )


def _apply_d4_transform(tensor: torch.Tensor, transform_idx: int) -> torch.Tensor:
    assert 0 <= transform_idx < 8
    transformed = torch.rot90(tensor, k=transform_idx % 4, dims=(-2, -1))
    if transform_idx >= 4:
        transformed = torch.flip(transformed, dims=(-1,))
    return transformed


def _ownership_tensor(
    batch: list[SelfPlaySample],
    device: torch.device,
    ownership_loss_weight: float,
) -> torch.Tensor | None:
    if ownership_loss_weight == 0.0:
        return None
    assert all(sample.ownership is not None for sample in batch)
    return torch.tensor(
        [sample.ownership for sample in batch],
        dtype=torch.float32,
        device=device,
    )
