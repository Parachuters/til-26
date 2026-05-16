"""Observation preprocessing shared by AE PPO training and inference."""

from __future__ import annotations

from typing import Any

import numpy as np


AGENT_VIEW_SHAPE = (7, 5, 25)
BASE_VIEW_SHAPE = (7, 7, 25)
ACTION_MASK_SHAPE = (6,)
SCALAR_SHAPE = (14,)

GRID_SIZE = 16.0
MAX_HEALTH = 60.0
MAX_FREEZE_TICKS = 3.0
MAX_BASE_HEALTH = 100.0
MAX_TEAM_RESOURCES = 100.0
MAX_TEAM_BOMBS = 50.0
MAX_STEPS = 200.0


def preprocess_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert a competition observation into the PPO MultiInputPolicy format."""

    location = _array(observation.get("location", [0, 0]), shape=(2,))
    base_location = _array(observation.get("base_location", [0, 0]), shape=(2,))
    health = _scalar(observation.get("health", [MAX_HEALTH]), MAX_HEALTH)
    frozen_ticks = float(observation.get("frozen_ticks", 0.0))
    base_health = _scalar(observation.get("base_health", [MAX_BASE_HEALTH]), MAX_BASE_HEALTH)
    team_resources = _scalar(observation.get("team_resources", [0.0]), 0.0)
    team_bombs = float(observation.get("team_bombs", 0.0))
    step = float(observation.get("step", 0.0))
    direction = float(observation.get("direction", 0.0))

    rel_base = (base_location - location) / GRID_SIZE
    scalars = np.array(
        [
            direction / 3.0,
            location[0] / (GRID_SIZE - 1.0),
            location[1] / (GRID_SIZE - 1.0),
            base_location[0] / (GRID_SIZE - 1.0),
            base_location[1] / (GRID_SIZE - 1.0),
            rel_base[0],
            rel_base[1],
            health / MAX_HEALTH,
            frozen_ticks / MAX_FREEZE_TICKS,
            base_health / MAX_BASE_HEALTH,
            team_resources / MAX_TEAM_RESOURCES,
            team_bombs / MAX_TEAM_BOMBS,
            step / MAX_STEPS,
            1.0 if frozen_ticks > 0 else 0.0,
        ],
        dtype=np.float32,
    )

    return {
        "agent_viewcone": _fit_view(observation.get("agent_viewcone", []), AGENT_VIEW_SHAPE),
        "base_viewcone": _fit_view(observation.get("base_viewcone", []), BASE_VIEW_SHAPE),
        "scalars": np.clip(scalars, -1.0, 1.0).astype(np.float32),
        "action_mask": _fit_mask(observation.get("action_mask", [])),
    }


def _fit_view(value: Any, shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    out = np.zeros(shape, dtype=np.float32)
    if arr.ndim != 3:
        return out
    rows = min(shape[0], arr.shape[0])
    cols = min(shape[1], arr.shape[1])
    channels = min(shape[2], arr.shape[2])
    src_row = max(0, (arr.shape[0] - rows) // 2)
    src_col = max(0, (arr.shape[1] - cols) // 2)
    dst_row = max(0, (shape[0] - rows) // 2)
    dst_col = max(0, (shape[1] - cols) // 2)
    out[dst_row : dst_row + rows, dst_col : dst_col + cols, :channels] = arr[
        src_row : src_row + rows,
        src_col : src_col + cols,
        :channels,
    ]
    return out


def _fit_mask(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.zeros(ACTION_MASK_SHAPE, dtype=np.float32)
    length = min(out.shape[0], arr.shape[0])
    if length:
        out[:length] = arr[:length]
    return out


def _array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.zeros(shape, dtype=np.float32)
    length = min(out.size, arr.size)
    if length:
        out.reshape(-1)[:length] = arr[:length]
    return out


def _scalar(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[0]) if len(value) else float(default)
    return float(value)
