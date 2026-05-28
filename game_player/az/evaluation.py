from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from game_player.agents.argmax_agent import ArgmaxPolicyAgent
from game_player.agents.base import Agent
from game_player.agents.random_agent import RandomAgent
from game_player.games.base import Action, GameState, Player
from game_player.games.go import BLACK, WHITE, GoState


class NeuralPolicyValueEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, state: GameState) -> tuple[dict[Action, float], float]:
        legal_actions = state.legal_actions()
        assert legal_actions

        self.model.eval()
        with torch.no_grad():
            observation = torch.tensor(
                state.observation(),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            output = self.model(observation)
            logits = output.policy_logits[0]

        assert logits.numel() == state.action_size
        legal_tensor = torch.tensor(legal_actions, dtype=torch.long, device=self.device)
        legal_logits = logits.index_select(dim=0, index=legal_tensor)
        legal_probabilities = F.softmax(legal_logits, dim=0).detach().cpu().tolist()
        priors = {
            action: float(probability)
            for action, probability in zip(legal_actions, legal_probabilities, strict=True)
        }
        return priors, float(output.value[0].detach().cpu().item())


class RandomRolloutEvaluator:
    def __init__(
        self,
        rollouts: int = 1,
        max_moves: int | None = None,
        seed: int = 0,
    ) -> None:
        assert rollouts > 0
        self.rollouts = rollouts
        self.max_moves = max_moves
        self.rng = random.Random(seed)

    def __call__(self, state: GameState) -> tuple[dict[Action, float], float]:
        legal_actions = state.legal_actions()
        assert legal_actions
        probability = 1.0 / len(legal_actions)
        priors = {action: probability for action in legal_actions}
        return priors, self.estimate_value(state)

    def estimate_value(self, state: GameState) -> float:
        total_value = 0.0
        for _ in range(self.rollouts):
            total_value += _random_rollout_result(
                state=state,
                max_moves=self.max_moves,
                rng=self.rng,
            )
        return total_value / self.rollouts


class HybridPolicyRolloutEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        rollouts: int = 1,
        max_moves: int | None = None,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.policy_evaluator = NeuralPolicyValueEvaluator(
            model=model,
            device=device,
        )
        self.rollout_evaluator = RandomRolloutEvaluator(
            rollouts=rollouts,
            max_moves=max_moves,
            seed=seed,
        )

    def __call__(self, state: GameState) -> tuple[dict[Action, float], float]:
        priors, _ = self.policy_evaluator(state)
        return priors, self.rollout_evaluator.estimate_value(state)


def _random_rollout_result(
    state: GameState,
    max_moves: int | None,
    rng: random.Random,
) -> float:
    root_player = state.current_player
    rollout_state = state
    moves = 0

    while not rollout_state.is_terminal():
        if max_moves is not None and moves >= max_moves:
            break
        legal_actions = rollout_state.legal_actions()
        assert legal_actions
        action = rng.choice(legal_actions)
        rollout_state = rollout_state.apply_action(action)
        moves += 1

    return rollout_state.result_for_player(root_player)


@dataclass(frozen=True)
class MatchResult:
    winner: Player
    black_score: float
    white_score: float
    moves: int


def play_match(
    state: GameState,
    black_agent: Agent,
    white_agent: Agent,
    max_moves: int | None = None,
) -> MatchResult:
    moves = 0
    while not state.is_terminal():
        if max_moves is not None and moves >= max_moves:
            break
        agent = black_agent if state.current_player == BLACK else white_agent
        action = agent.select_action(state)
        state = state.apply_action(action)
        moves += 1

    black_score, white_score = state.area_scores()
    return MatchResult(
        winner=state.winner(),
        black_score=black_score,
        white_score=white_score,
        moves=moves,
    )


def win_rate_against(
    states: list[GameState],
    candidate: Agent,
    baseline: Agent,
    candidate_player: Player,
    max_moves: int | None = None,
) -> float:
    assert states
    assert candidate_player in (BLACK, WHITE)
    wins = 0
    for state in states:
        if candidate_player == BLACK:
            result = play_match(
                state=state,
                black_agent=candidate,
                white_agent=baseline,
                max_moves=max_moves,
            )
        else:
            result = play_match(
                state=state,
                black_agent=baseline,
                white_agent=candidate,
                max_moves=max_moves,
            )
        wins += int(result.winner == candidate_player)
    return wins / len(states)


def evaluate_model_against_random_weights(
    candidate_model: torch.nn.Module,
    baseline_model: torch.nn.Module,
    board_size: int = 19,
    komi: float = 7.5,
    games: int = 100,
    max_moves: int | None = None,
    device: str = "cpu",
) -> dict[str, float]:
    assert games > 0

    candidate_agent = ArgmaxPolicyAgent(candidate_model, device=device)
    baseline_agent = ArgmaxPolicyAgent(baseline_model, device=device)
    candidate_wins = 0
    candidate_black_games = 0
    candidate_white_games = 0

    for game_idx in range(games):
        state = GoState.new(board_size=board_size, komi=komi)
        candidate_is_black = game_idx % 2 == 0
        if candidate_is_black:
            candidate_black_games += 1
            result = play_match(
                state=state,
                black_agent=candidate_agent,
                white_agent=baseline_agent,
                max_moves=max_moves,
            )
            candidate_wins += int(result.winner == BLACK)
        else:
            candidate_white_games += 1
            result = play_match(
                state=state,
                black_agent=baseline_agent,
                white_agent=candidate_agent,
                max_moves=max_moves,
            )
            candidate_wins += int(result.winner == WHITE)

    return {
        "validation_games": float(games),
        "validation_candidate_wins": float(candidate_wins),
        "validation_candidate_win_rate": candidate_wins / games,
        "validation_candidate_black_games": float(candidate_black_games),
        "validation_candidate_white_games": float(candidate_white_games),
    }


def evaluate_model_against_random_agent(
    candidate_model: torch.nn.Module,
    board_size: int = 19,
    komi: float = 7.5,
    games: int = 100,
    max_moves: int | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, float]:
    assert games > 0

    candidate_agent = ArgmaxPolicyAgent(candidate_model, device=device)
    random_agent = RandomAgent(seed=seed)
    candidate_wins = 0
    candidate_black_games = 0
    candidate_white_games = 0

    for game_idx in range(games):
        state = GoState.new(board_size=board_size, komi=komi)
        candidate_is_black = game_idx % 2 == 0
        if candidate_is_black:
            candidate_black_games += 1
            result = play_match(
                state=state,
                black_agent=candidate_agent,
                white_agent=random_agent,
                max_moves=max_moves,
            )
            candidate_wins += int(result.winner == BLACK)
        else:
            candidate_white_games += 1
            result = play_match(
                state=state,
                black_agent=random_agent,
                white_agent=candidate_agent,
                max_moves=max_moves,
            )
            candidate_wins += int(result.winner == WHITE)

    return {
        "validation_games": float(games),
        "validation_candidate_wins": float(candidate_wins),
        "validation_candidate_win_rate": candidate_wins / games,
        "validation_candidate_black_games": float(candidate_black_games),
        "validation_candidate_white_games": float(candidate_white_games),
    }
