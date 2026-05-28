from __future__ import annotations

import random
from dataclasses import dataclass

from game_player.az.mcts import Evaluator, run_mcts
from game_player.games.base import Action, GameState, Player


@dataclass(frozen=True)
class SelfPlaySample:
    observation: list[float]
    policy: list[float]
    value: float


def play_self_play_game(
    initial_state: GameState,
    evaluator: Evaluator,
    simulations: int = 64,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    temperature_drop_move: int = 30,
    dirichlet_alpha: float | None = 0.03,
    exploration_fraction: float = 0.25,
    max_moves: int | None = None,
    value_target: str = "outcome",
    rng: random.Random | None = None,
) -> tuple[list[SelfPlaySample], GameState]:
    assert simulations > 0
    assert value_target in ("outcome", "score")
    if rng is None:
        rng = random.Random()

    state = initial_state
    trajectory: list[tuple[list[float], list[float], Player]] = []
    move_number = 0

    while not state.is_terminal():
        if max_moves is not None and move_number >= max_moves:
            break

        move_temperature = temperature if move_number < temperature_drop_move else 0.0
        policy = run_mcts(
            state=state,
            evaluator=evaluator,
            simulations=simulations,
            c_puct=c_puct,
            temperature=move_temperature,
            dirichlet_alpha=dirichlet_alpha,
            exploration_fraction=exploration_fraction,
            rng=rng,
        )
        action = sample_action(policy, rng)
        trajectory.append((state.observation(), policy, state.current_player))
        state = state.apply_action(action)
        move_number += 1

    samples = [
        SelfPlaySample(
            observation=observation,
            policy=policy,
            value=_final_value_for_player(state, player, value_target),
        )
        for observation, policy, player in trajectory
    ]
    return samples, state


def sample_action(policy: list[float], rng: random.Random) -> Action:
    assert policy
    total_probability = sum(policy)
    assert total_probability > 0.0

    threshold = rng.random() * total_probability
    cumulative = 0.0
    for action, probability in enumerate(policy):
        cumulative += probability
        if cumulative >= threshold:
            return action
    return len(policy) - 1


def _final_value_for_player(
    state: GameState,
    player: Player,
    value_target: str,
) -> float:
    if value_target == "outcome":
        return state.result_for_player(player)
    if value_target == "score":
        black_score, white_score = state.area_scores()
        score_margin = black_score - white_score
        if player < 0:
            score_margin = -score_margin
        board_area = state.action_size - 1
        value = score_margin / board_area
        return max(-1.0, min(1.0, value))
    raise ValueError(f"Unknown value target: {value_target}")
