from __future__ import annotations

import torch

from game_player.games.base import Action, GameState


class ArgmaxPolicyAgent:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def select_action(self, state: GameState) -> Action:
        legal_actions = state.legal_actions()
        assert legal_actions

        with torch.no_grad():
            observation = torch.tensor(
                state.observation(),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            output = self.model(observation)
            logits = output.policy_logits[0].detach().cpu()

        assert logits.numel() == state.action_size
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits[legal_actions] = logits[legal_actions]
        return int(torch.argmax(masked_logits).item())
