from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from game_player.agents.argmax_agent import ArgmaxPolicyAgent
from game_player.agents.base import Agent
from game_player.agents.random_agent import RandomAgent
from game_player.games.base import Action, GameState, Player
from game_player.games.go import (
    BLACK,
    WHITE,
    GoState,
    _collect_group,
    _neighbors,
)


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


class TrompTaylorTacticalEvaluator:
    def __init__(
        self,
        prior_temperature: float = 2.0,
        pass_prior: float = 1e-6,
        area_weight: float = 1.0,
        capture_weight: float = 100.0,
        liberty_weight: float = 5.0,
        stone_weight: float = 0.25,
        adjacent_opponent_weight: float = 2.0,
        adjacent_own_weight: float = 1.0,
        opponent_liberty_reduction_weight: float = 10.0,
        opponent_atari_weight: float = 15.0,
        self_atari_weight: float = -15.0,
        suicide_weight: float = -1000.0,
    ) -> None:
        assert prior_temperature > 0.0
        assert pass_prior > 0.0
        self.prior_temperature = prior_temperature
        self.pass_prior = pass_prior
        self.area_weight = area_weight
        self.capture_weight = capture_weight
        self.liberty_weight = liberty_weight
        self.stone_weight = stone_weight
        self.adjacent_opponent_weight = adjacent_opponent_weight
        self.adjacent_own_weight = adjacent_own_weight
        self.opponent_liberty_reduction_weight = opponent_liberty_reduction_weight
        self.opponent_atari_weight = opponent_atari_weight
        self.self_atari_weight = self_atari_weight
        self.suicide_weight = suicide_weight

    def __call__(self, state: GameState) -> tuple[dict[Action, float], float]:
        assert isinstance(state, GoState)
        legal_actions = state.legal_actions()
        assert legal_actions

        action_scores = {
            action: self._action_score(state, action)
            for action in legal_actions
        }
        priors = _softmax_action_scores(
            action_scores=action_scores,
            temperature=self.prior_temperature,
        )
        return priors, _score_value_for_current_player(state)

    def _action_score(self, state: GoState, action: Action) -> float:
        if action == state.pass_action:
            if len(state.legal_actions()) == 1:
                return 0.0
            return math.log(self.pass_prior)

        before_opponent_stones = sum(
            1
            for point in state.board
            if point == -state.current_player
        )
        opponent_groups = _neighboring_opponent_groups(state, action)
        next_state = state.apply_action(action)
        after_opponent_stones = sum(
            1
            for point in next_state.board
            if point == -state.current_player
        )
        captured_stones = before_opponent_stones - after_opponent_stones
        own_stones = sum(
            1
            for point in next_state.board
            if point == state.current_player
        )
        opponent_stones = sum(
            1
            for point in next_state.board
            if point == -state.current_player
        )
        group_liberties = 0
        suicide = 0
        if next_state.board[action] == state.current_player:
            _, liberties = _collect_group(
                next_state.board,
                action,
                next_state.board_size,
            )
            group_liberties = len(liberties)
        else:
            suicide = 1

        adjacent_opponents = sum(
            1
            for neighbor in _neighbors(action, state.board_size)
            if state.board[neighbor] == -state.current_player
        )
        adjacent_own = sum(
            1
            for neighbor in _neighbors(action, state.board_size)
            if state.board[neighbor] == state.current_player
        )
        opponent_liberty_reduction = 0
        opponent_groups_in_atari = 0
        for group, before_liberties in opponent_groups:
            remaining_points = [
                point
                for point in group
                if next_state.board[point] == -state.current_player
            ]
            if not remaining_points:
                opponent_liberty_reduction += before_liberties
                continue
            _, after_liberties = _collect_group(
                next_state.board,
                remaining_points[0],
                next_state.board_size,
            )
            opponent_liberty_reduction += max(
                0,
                before_liberties - len(after_liberties),
            )
            opponent_groups_in_atari += int(len(after_liberties) == 1)
        self_atari = int(group_liberties == 1 and captured_stones == 0)
        area_margin = 0.0
        if self.area_weight != 0.0:
            black_score, white_score = next_state.area_scores()
            area_margin = (
                black_score - white_score
                if state.current_player == BLACK
                else white_score - black_score
            )

        return (
            self.area_weight * area_margin
            + self.capture_weight * captured_stones
            + self.liberty_weight * group_liberties
            + self.adjacent_opponent_weight * adjacent_opponents
            + self.adjacent_own_weight * adjacent_own
            + self.opponent_liberty_reduction_weight * opponent_liberty_reduction
            + self.opponent_atari_weight * opponent_groups_in_atari
            + self.self_atari_weight * self_atari
            + self.suicide_weight * suicide
            + self.stone_weight * (own_stones - opponent_stones)
        )


def _neighboring_opponent_groups(
    state: GoState,
    action: Action,
) -> list[tuple[set[int], int]]:
    seen: set[int] = set()
    groups: list[tuple[set[int], int]] = []
    for neighbor in _neighbors(action, state.board_size):
        if state.board[neighbor] != -state.current_player:
            continue
        if neighbor in seen:
            continue
        group, liberties = _collect_group(
            state.board,
            neighbor,
            state.board_size,
        )
        seen.update(group)
        groups.append((group, len(liberties)))
    return groups


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


def _score_value_for_current_player(state: GoState) -> float:
    black_score, white_score = state.area_scores()
    score_margin = (
        black_score - white_score
        if state.current_player == BLACK
        else white_score - black_score
    )
    board_area = state.board_size * state.board_size
    value = score_margin / board_area
    return max(-1.0, min(1.0, value))


def _softmax_action_scores(
    action_scores: dict[Action, float],
    temperature: float,
) -> dict[Action, float]:
    assert action_scores
    max_score = max(action_scores.values())
    exponentials = {
        action: math.exp((score - max_score) / temperature)
        for action, score in action_scores.items()
    }
    total = sum(exponentials.values())
    assert total > 0.0
    return {
        action: value / total
        for action, value in exponentials.items()
    }


@dataclass(frozen=True)
class MatchResult:
    winner: Player
    black_score: float
    white_score: float
    moves: int
    terminal: bool


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
        terminal=state.is_terminal(),
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
    candidate_black_wins = 0
    candidate_white_games = 0
    candidate_white_wins = 0
    candidate_score_margin = 0.0
    total_moves = 0
    terminal_games = 0

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
            win = int(result.winner == BLACK)
            candidate_black_wins += win
            candidate_wins += win
            candidate_score_margin += result.black_score - result.white_score
        else:
            candidate_white_games += 1
            result = play_match(
                state=state,
                black_agent=baseline_agent,
                white_agent=candidate_agent,
                max_moves=max_moves,
            )
            win = int(result.winner == WHITE)
            candidate_white_wins += win
            candidate_wins += win
            candidate_score_margin += result.white_score - result.black_score
        total_moves += result.moves
        terminal_games += int(result.terminal)

    return {
        "validation_games": float(games),
        "validation_candidate_wins": float(candidate_wins),
        "validation_candidate_win_rate": candidate_wins / games,
        "validation_candidate_avg_score_margin": candidate_score_margin / games,
        "validation_avg_moves": total_moves / games,
        "validation_terminal_games": float(terminal_games),
        "validation_terminal_rate": terminal_games / games,
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
    candidate_black_wins = 0
    candidate_white_games = 0
    candidate_white_wins = 0
    candidate_score_margin = 0.0
    total_moves = 0
    terminal_games = 0

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
            win = int(result.winner == BLACK)
            candidate_black_wins += win
            candidate_wins += win
            candidate_score_margin += result.black_score - result.white_score
        else:
            candidate_white_games += 1
            result = play_match(
                state=state,
                black_agent=random_agent,
                white_agent=candidate_agent,
                max_moves=max_moves,
            )
            win = int(result.winner == WHITE)
            candidate_white_wins += win
            candidate_wins += win
            candidate_score_margin += result.white_score - result.black_score
        total_moves += result.moves
        terminal_games += int(result.terminal)

    return {
        "validation_games": float(games),
        "validation_candidate_wins": float(candidate_wins),
        "validation_candidate_win_rate": candidate_wins / games,
        "validation_candidate_avg_score_margin": candidate_score_margin / games,
        "validation_avg_moves": total_moves / games,
        "validation_terminal_games": float(terminal_games),
        "validation_terminal_rate": terminal_games / games,
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
