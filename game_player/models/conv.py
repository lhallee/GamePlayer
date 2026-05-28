from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


@dataclass(frozen=True)
class ConvPolicyValueOutput:
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


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class ConvPolicyValueNet(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        board_size: int = 19,
        channels: int = 64,
        blocks: int = 4,
        include_pass: bool = True,
    ) -> None:
        super().__init__()
        assert board_size > 1
        assert channels > 0
        assert blocks > 0

        self.board_size = board_size
        self.channels = channels
        self.blocks = blocks
        self.include_pass = include_pass

        self.input_size = 2 * board_size * board_size
        self.num_board_actions = board_size * board_size
        self.num_actions = self.num_board_actions + int(include_pass)

        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.torso = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> ConvPolicyValueOutput:
        if observation.dim() == 2:
            observation = observation.reshape(
                observation.shape[0],
                2,
                self.board_size,
                self.board_size,
            )
        assert tuple(observation.shape[-3:]) == (
            2,
            self.board_size,
            self.board_size,
        )

        hidden = self.torso(self.stem(observation.float()))
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return ConvPolicyValueOutput(
            policy_logits=policy_logits,
            value=value,
            board_size=self.board_size,
            include_pass=self.include_pass,
        )
