from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from game_player.az.evaluation import NeuralPolicyValueEvaluator
from game_player.az.replay import ReplayBuffer
from game_player.az.self_play import SelfPlaySample, play_self_play_game
from game_player.games.go import GoState


@dataclass(frozen=True)
class TrainingConfig:
    board_size: int = 19
    komi: float = 7.5
    self_play_games: int = 1
    mcts_simulations: int = 64
    replay_capacity: int = 100_000
    batch_size: int = 64
    train_steps: int = 32
    learning_rate: float = 1e-3
    max_moves: int | None = None
    seed: int = 0


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: list[SelfPlaySample],
    device: str = "cpu",
) -> dict[str, float]:
    assert batch
    torch_device = torch.device(device)

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

    output = model(observations)
    log_policy = F.log_softmax(output.policy_logits, dim=-1)
    policy_loss = -(target_policies * log_policy).sum(dim=-1).mean()
    value_loss = F.mse_loss(output.value, target_values)
    loss = policy_loss + value_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
    }


def run_training(
    model: torch.nn.Module,
    config: TrainingConfig,
    device: str = "cpu",
) -> dict[str, float]:
    assert config.self_play_games > 0
    assert config.mcts_simulations > 0
    assert config.train_steps >= 0

    rng = random.Random(config.seed)
    model.to(torch.device(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(capacity=config.replay_capacity)
    evaluator = NeuralPolicyValueEvaluator(model=model, device=device)

    total_samples = 0
    for _ in range(config.self_play_games):
        state = GoState.new(board_size=config.board_size, komi=config.komi)
        samples, _ = play_self_play_game(
            initial_state=state,
            evaluator=evaluator,
            simulations=config.mcts_simulations,
            max_moves=config.max_moves,
            rng=rng,
        )
        replay.add_many(samples)
        total_samples += len(samples)

    metrics: dict[str, float] = {
        "self_play_samples": float(total_samples),
        "replay_size": float(len(replay)),
    }
    if len(replay) < config.batch_size or config.train_steps == 0:
        return metrics

    for _ in range(config.train_steps):
        batch = replay.sample(config.batch_size, rng=rng)
        metrics.update(train_step(model, optimizer, batch, device=device))
    metrics["replay_size"] = float(len(replay))
    return metrics
