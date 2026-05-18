"""Masked recurrent policy used by AE behavior cloning and PPO fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ppo_preprocess import preprocess_observation


ACTION_DIM = 6


@dataclass
class LearnedAction:
    action: int
    confidence: float
    hidden_state: torch.Tensor


class MaskedRecurrentPolicy(nn.Module):
    """Small CNN + GRU policy with hard action masking."""

    def __init__(
        self,
        scalar_dim: int = 14,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.agent_branch = self._view_branch()
        self.base_branch = self._view_branch()
        self.scalar_branch = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(64 + 64 + 64, hidden_dim),
            nn.ReLU(),
        )
        self.memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _view_branch() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(25, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
        )

    def initial_state(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        agent_viewcone: torch.Tensor,
        base_viewcone: torch.Tensor,
        scalars: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_state is None:
            hidden_state = self.initial_state(agent_viewcone.shape[0], agent_viewcone.device)

        agent_features = self.agent_branch(agent_viewcone.float().permute(0, 3, 1, 2))
        base_features = self.base_branch(base_viewcone.float().permute(0, 3, 1, 2))
        scalar_features = self.scalar_branch(scalars.float())
        fused = self.fuse(torch.cat([agent_features, base_features, scalar_features], dim=1))
        next_hidden = self.memory(fused, hidden_state)
        return self.policy_head(next_hidden), self.value_head(next_hidden).squeeze(-1), next_hidden

    @staticmethod
    def mask_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        mask = action_mask.float() > 0.0
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        safe_mask = mask.clone()
        no_legal = ~safe_mask.any(dim=1)
        if no_legal.any():
            safe_mask[no_legal, 4] = True
        return logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)

    @torch.no_grad()
    def act(
        self,
        observation: dict[str, Any],
        hidden_state: torch.Tensor | None = None,
        deterministic: bool = True,
        device: torch.device | str = "cpu",
    ) -> LearnedAction:
        device = torch.device(device)
        self.to(device)
        self.eval()
        processed = preprocess_observation(observation)
        batch = {
            key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
            for key, value in processed.items()
        }
        if hidden_state is not None:
            hidden_state = hidden_state.to(device)
        logits, _value, next_hidden = self.forward(
            batch["agent_viewcone"],
            batch["base_viewcone"],
            batch["scalars"],
            hidden_state,
        )
        masked_logits = self.mask_logits(logits, batch["action_mask"])
        probs = torch.softmax(masked_logits, dim=1)
        if deterministic:
            action_tensor = torch.argmax(probs, dim=1)
        else:
            action_tensor = torch.distributions.Categorical(probs=probs).sample()
        action = int(action_tensor.item())
        confidence = float(probs[0, action].item())
        return LearnedAction(action, confidence, next_hidden.detach())


def load_masked_recurrent_policy(path: str | Path, device: str = "cpu") -> MaskedRecurrentPolicy:
    checkpoint = torch.load(Path(path), map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = MaskedRecurrentPolicy(
        scalar_dim=int(config.get("scalar_dim", 14)),
        action_dim=int(config.get("action_dim", ACTION_DIM)),
        hidden_dim=int(config.get("hidden_dim", 128)),
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def masked_cross_entropy(logits: torch.Tensor, action_mask: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    masked_logits = MaskedRecurrentPolicy.mask_logits(logits, action_mask)
    return nn.functional.cross_entropy(masked_logits, actions.long())


def action_accuracy(logits: torch.Tensor, action_mask: torch.Tensor, actions: torch.Tensor) -> float:
    masked_logits = MaskedRecurrentPolicy.mask_logits(logits, action_mask)
    predicted = torch.argmax(masked_logits, dim=1)
    return float((predicted == actions.long()).float().mean().item())
