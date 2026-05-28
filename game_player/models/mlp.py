from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


@dataclass(frozen=True)
class PolicyValueOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
    board_size: int
    include_pass: bool

    @property
    def board_logits(self) -> torch.Tensor:
        board_actions = self.board_size * self.board_size
        batch_shape = self.policy_logits.shape[:-1]
        return self.policy_logits[..., :board_actions].reshape(
            *batch_shape,
            self.board_size,
            self.board_size,
        )

    @property
    def pass_logit(self) -> torch.Tensor:
        assert self.include_pass
        return self.policy_logits[..., -1]


class MLPPolicyValueNet(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        board_size: int = 19,
        hidden_size: int = 512,
        depth: int = 3,
        include_pass: bool = True,
    ) -> None:
        super().__init__()
        assert board_size > 1
        assert hidden_size > 0
        assert depth > 0

        self.board_size = board_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.include_pass = include_pass

        self.input_size = 2 * board_size * board_size
        self.num_board_actions = board_size * board_size
        self.num_actions = self.num_board_actions + int(include_pass)

        layers: list[nn.Module] = []
        in_features = self.input_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        self.torso = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_size, self.num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observation: torch.Tensor) -> PolicyValueOutput:
        if observation.dim() > 2:
            observation = observation.flatten(start_dim=1)
        assert observation.shape[-1] == self.input_size

        hidden = self.torso(observation.float())
        policy_logits = self.policy_head(hidden)
        value = torch.tanh(self.value_head(hidden)).squeeze(-1)
        return PolicyValueOutput(
            policy_logits=policy_logits,
            value=value,
            board_size=self.board_size,
            include_pass=self.include_pass,
        )
