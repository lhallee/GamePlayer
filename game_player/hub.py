from __future__ import annotations

from pathlib import Path
from typing import Protocol

from game_player.models.mlp import MLPPolicyValueNet


class HubSaveableModel(Protocol):
    def save_pretrained(self, save_directory: str) -> None:
        ...


def load_mlp_model(path_or_repo_id: str) -> MLPPolicyValueNet:
    return MLPPolicyValueNet.from_pretrained(path_or_repo_id)


def save_policy_value_model(
    model: HubSaveableModel,
    output_dir: str | Path,
) -> None:
    model.save_pretrained(str(output_dir))


def save_mlp_model(model: MLPPolicyValueNet, output_dir: str | Path) -> None:
    save_policy_value_model(model, output_dir)


def push_mlp_model(model: MLPPolicyValueNet, repo_id: str) -> str:
    return model.push_to_hub(repo_id)
