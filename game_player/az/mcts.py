from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from game_player.games.base import Action, GameState

Evaluator = Callable[[GameState], tuple[dict[Action, float], float]]


@dataclass
class SearchNode:
    prior: float
    to_play: int
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[Action, "SearchNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def uniform_evaluator(state: GameState) -> tuple[dict[Action, float], float]:
    legal_actions = state.legal_actions()
    assert legal_actions
    probability = 1.0 / len(legal_actions)
    return {action: probability for action in legal_actions}, 0.0


def run_mcts(
    state: GameState,
    evaluator: Evaluator,
    simulations: int,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    dirichlet_alpha: float | None = None,
    exploration_fraction: float = 0.25,
    rng: random.Random | None = None,
) -> list[float]:
    assert simulations > 0
    assert c_puct > 0.0
    assert temperature >= 0.0

    if rng is None:
        rng = random.Random()

    root = SearchNode(prior=1.0, to_play=state.current_player)
    priors, _ = evaluator(state)
    _expand(root, state, priors)
    if dirichlet_alpha is not None:
        _add_exploration_noise(
            root,
            alpha=dirichlet_alpha,
            fraction=exploration_fraction,
            rng=rng,
        )

    for _ in range(simulations):
        node = root
        scratch_state = state
        search_path = [node]

        while node.children and not scratch_state.is_terminal():
            action, node = _select_child(node, c_puct=c_puct)
            scratch_state = scratch_state.apply_action(action)
            search_path.append(node)

        if scratch_state.is_terminal():
            value = scratch_state.result_for_player(scratch_state.current_player)
        else:
            priors, value = evaluator(scratch_state)
            _expand(node, scratch_state, priors)

        _backpropagate(
            search_path=search_path,
            value=value,
            value_player=scratch_state.current_player,
        )

    return _visit_count_policy(root, action_size=state.action_size, temperature=temperature)


def _expand(
    node: SearchNode,
    state: GameState,
    priors: dict[Action, float],
) -> None:
    legal_actions = state.legal_actions()
    assert legal_actions
    total_prior = sum(max(priors[action], 0.0) for action in legal_actions)
    if total_prior <= 0.0:
        probability = 1.0 / len(legal_actions)
        priors = {action: probability for action in legal_actions}
        total_prior = 1.0

    for action in legal_actions:
        prior = max(priors[action], 0.0) / total_prior
        child_state = state.apply_action(action)
        node.children[action] = SearchNode(
            prior=prior,
            to_play=child_state.current_player,
        )


def _select_child(node: SearchNode, c_puct: float) -> tuple[Action, SearchNode]:
    assert node.children
    _, action, child = max(
        (_puct_score(node, child, c_puct), action, child)
        for action, child in node.children.items()
    )
    return action, child


def _puct_score(parent: SearchNode, child: SearchNode, c_puct: float) -> float:
    prior_score = c_puct * child.prior * math.sqrt(parent.visit_count + 1)
    prior_score /= child.visit_count + 1
    value_score = -child.value
    return value_score + prior_score


def _backpropagate(
    search_path: list[SearchNode],
    value: float,
    value_player: int,
) -> None:
    for node in reversed(search_path):
        node.visit_count += 1
        if node.to_play == value_player:
            node.value_sum += value
        else:
            node.value_sum -= value


def _visit_count_policy(
    root: SearchNode,
    action_size: int,
    temperature: float,
) -> list[float]:
    policy = [0.0] * action_size
    if not root.children:
        return policy

    if temperature == 0.0:
        best_action = max(
            root.children,
            key=lambda action: root.children[action].visit_count,
        )
        policy[best_action] = 1.0
        return policy

    visits = {
        action: child.visit_count ** (1.0 / temperature)
        for action, child in root.children.items()
    }
    total_visits = sum(visits.values())
    assert total_visits > 0.0
    for action, visit_count in visits.items():
        policy[action] = visit_count / total_visits
    return policy


def _add_exploration_noise(
    root: SearchNode,
    alpha: float,
    fraction: float,
    rng: random.Random,
) -> None:
    assert alpha > 0.0
    assert 0.0 <= fraction <= 1.0
    actions = list(root.children)
    if not actions:
        return

    gamma_samples = [rng.gammavariate(alpha, 1.0) for _ in actions]
    total = sum(gamma_samples)
    assert total > 0.0
    noise = [sample / total for sample in gamma_samples]
    for action, sample in zip(actions, noise, strict=True):
        child = root.children[action]
        child.prior = child.prior * (1.0 - fraction) + sample * fraction
