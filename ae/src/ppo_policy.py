"""Stable-Baselines3 policy components for the AE PPO agent."""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class ViewconeExtractor(BaseFeaturesExtractor):
    """CNN + MLP extractor for Bomberman viewcones and scalar state."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)

        agent_shape = observation_space["agent_viewcone"].shape
        base_shape = observation_space["base_viewcone"].shape
        scalar_dim = int(observation_space["scalars"].shape[0])
        mask_dim = int(observation_space["action_mask"].shape[0])

        self.agent_cnn = self._view_branch(agent_shape)
        self.base_cnn = self._view_branch(base_shape)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim + mask_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        agent_out = self._cnn_output_dim(self.agent_cnn, agent_shape)
        base_out = self._cnn_output_dim(self.base_cnn, base_shape)
        self.project = nn.Sequential(
            nn.Linear(agent_out + base_out + 64, features_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _view_branch(shape: tuple[int, ...]) -> nn.Sequential:
        channels = int(shape[2])
        return nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    @staticmethod
    def _cnn_output_dim(cnn: nn.Module, shape: tuple[int, ...]) -> int:
        sample = torch.zeros(1, int(shape[2]), int(shape[0]), int(shape[1]))
        with torch.no_grad():
            return int(cnn(sample).shape[1])

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        agent_view = observations["agent_viewcone"].float().permute(0, 3, 1, 2)
        base_view = observations["base_viewcone"].float().permute(0, 3, 1, 2)
        scalars = observations["scalars"].float()
        action_mask = observations["action_mask"].float()

        features = torch.cat(
            [
                self.agent_cnn(agent_view),
                self.base_cnn(base_view),
                self.scalar_mlp(torch.cat([scalars, action_mask], dim=1)),
            ],
            dim=1,
        )
        return self.project(features)
