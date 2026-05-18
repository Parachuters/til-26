"""Train and evaluate a PPO policy for the AE Bomberman challenge."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


REPO_ROOT = Path(__file__).resolve().parents[2]
AE_SRC = REPO_ROOT / "ae" / "src"
TIL_ENV = REPO_ROOT / "til-26-ae"
for path in (AE_SRC, TIL_ENV):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ae_manager import AEManager  # noqa: E402
from ppo_preprocess import (  # noqa: E402
    ACTION_MASK_SHAPE,
    AGENT_VIEW_SHAPE,
    BASE_VIEW_SHAPE,
    SCALAR_SHAPE,
    preprocess_observation,
)
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


ActionFn = Callable[[dict[str, Any]], int]


@dataclass
class EpisodeStats:
    seed: int
    policy: str
    reward: float
    score: float
    frozen_ticks: int
    masked_actions: int
    steps: int


class BombermanSingleAgentEnv(gym.Env):
    """Gymnasium wrapper controlling agent_0 while opponents act randomly."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None, novice: bool = False) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.novice = novice
        self.env = self._make_env()
        self.controlled_agent = self.env.possible_agents[0]
        self.action_space = spaces.Discrete(6)
        self.observation_space = _ppo_observation_space()
        self.seen_cells: set[tuple[int, int]] = set()
        self.prev_health = 60.0
        self.prev_base_health = 100.0
        self.prev_frozen_ticks = 0
        self.masked_actions = 0

    def _make_env(self):
        cfg = default_config()
        cfg.env.novice = bool(self.novice)
        cfg.env.render_mode = None
        return bomberman_env.basic_env(cfg=cfg, env_wrappers=[])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        effective_seed = seed
        if effective_seed is None:
            effective_seed = int(self.rng.integers(0, 2**31 - 1))
        self.env.reset(seed=effective_seed)
        self.seen_cells.clear()
        self.masked_actions = 0
        observation, _reward, _termination, _truncation, _info = self.env.last()
        self._remember_observation(observation, include_start=True)
        return preprocess_observation(observation), {}

    def step(self, action: int):
        observation, _reward, termination, truncation, _info = self.env.last()
        filtered_action = self._apply_action_mask(int(action), observation)
        self.env.step(None if termination or truncation else filtered_action)

        while self.env.agents and self.env.agent_selection != self.controlled_agent:
            other_obs, _other_reward, other_term, other_trunc, _other_info = self.env.last()
            other_action = None
            if not other_term and not other_trunc:
                other_action = self.env.action_space(self.env.agent_selection).sample()
            self.env.step(other_action)

        if not self.env.agents:
            return preprocess_observation(observation), 0.0, True, False, {}

        next_obs, base_reward, next_term, next_trunc, info = self.env.last()
        shaped_reward = float(base_reward) + self._shaping_reward(next_obs)
        return (
            preprocess_observation(next_obs),
            shaped_reward,
            bool(next_term),
            bool(next_trunc),
            {
                **info,
                "base_reward": float(base_reward),
                "masked_actions": self.masked_actions,
            },
        )

    def close(self) -> None:
        self.env.close()

    def _apply_action_mask(self, action: int, observation: dict[str, Any]) -> int:
        mask = np.asarray(observation.get("action_mask", []), dtype=np.int8).reshape(-1)
        legal = [idx for idx, value in enumerate(mask[:6]) if value]
        if action in legal:
            return int(action)
        self.masked_actions += 1
        if 4 in legal:
            return 4
        return int(legal[0]) if legal else 4

    def _shaping_reward(self, observation: dict[str, Any]) -> float:
        reward = 0.0
        location = _location(observation)
        if location not in self.seen_cells:
            reward += 0.03
            self.seen_cells.add(location)

        frozen_ticks = int(observation.get("frozen_ticks", 0))
        if frozen_ticks > 0:
            reward -= 0.3

        health = _scalar(observation.get("health", [60.0]), 60.0)
        base_health = _scalar(observation.get("base_health", [100.0]), 100.0)
        if (self.prev_health > 0.0 and health <= 0.0) or (
            self.prev_frozen_ticks == 0 and frozen_ticks > 0
        ) or (
            base_health < self.prev_base_health
        ):
            reward -= 1.0

        self.prev_health = health
        self.prev_base_health = base_health
        self.prev_frozen_ticks = frozen_ticks
        return reward

    def _remember_observation(self, observation: dict[str, Any], include_start: bool) -> None:
        self.prev_health = _scalar(observation.get("health", [60.0]), 60.0)
        self.prev_base_health = _scalar(observation.get("base_health", [100.0]), 100.0)
        self.prev_frozen_ticks = int(observation.get("frozen_ticks", 0))
        if include_start:
            self.seen_cells.add(_location(observation))


def train(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_checker import check_env

    from ppo_policy import ViewconeExtractor

    env = BombermanSingleAgentEnv(seed=args.seed, novice=args.novice)
    if args.check_env:
        check_env(env, warn=True)

    callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(args.checkpoint_dir),
        name_prefix="ae_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    policy_kwargs = {
        "features_extractor_class": ViewconeExtractor,
        "features_extractor_kwargs": {"features_dim": args.features_dim},
    }
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        tensorboard_log=str(args.tensorboard_log),
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=args.progress_bar,
    )
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_out))
    env.close()


def evaluate(args: argparse.Namespace) -> None:
    rows: list[EpisodeStats] = []
    ppo_model = None
    if args.model is not None:
        from stable_baselines3 import PPO

        import ppo_policy  # noqa: F401  # Ensure custom extractor is importable.

        ppo_model = PPO.load(str(args.model), device=args.device)

    for idx in range(args.episodes):
        seed = args.seed + idx
        if args.include_planner:
            manager = AEManager()
            rows.append(
                run_episode(
                    seed=seed,
                    novice=args.novice,
                    policy_name="planner",
                    action_fn=manager.ae,
                )
            )
        if ppo_model is not None:
            rows.append(
                run_episode(
                    seed=seed,
                    novice=args.novice,
                    policy_name="ppo",
                    action_fn=lambda obs, model=ppo_model: _ppo_action(model, obs),
                )
            )

    _write_eval(rows, args.csv_out)
    _print_eval(rows)


def smoke(args: argparse.Namespace) -> None:
    env = BombermanSingleAgentEnv(seed=args.seed, novice=args.novice)
    obs, _info = env.reset(seed=args.seed)
    assert obs["agent_viewcone"].shape == AGENT_VIEW_SHAPE
    assert obs["base_viewcone"].shape == BASE_VIEW_SHAPE
    assert obs["scalars"].shape == SCALAR_SHAPE
    assert obs["action_mask"].shape == ACTION_MASK_SHAPE
    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _info = env.step(action)
        assert np.isfinite(reward)
        if terminated or truncated:
            obs, _info = env.reset()
    env.close()
    print(json.dumps({"status": "ok", "steps": args.steps}))


def run_episode(
    seed: int,
    novice: bool,
    policy_name: str,
    action_fn: ActionFn,
) -> EpisodeStats:
    cfg = default_config()
    cfg.env.novice = bool(novice)
    cfg.env.render_mode = None
    env = bomberman_env.basic_env(cfg=cfg, env_wrappers=[])
    controlled_agent = env.possible_agents[0]
    reward_total = 0.0
    frozen_ticks = 0
    masked_actions = 0
    steps = 0

    env.reset(seed=seed)
    for agent in env.agent_iter():
        observation, _reward, termination, truncation, _info = env.last()
        if controlled_agent in env.rewards:
            reward_total += float(env.rewards[controlled_agent])

        if termination or truncation:
            action = None
        elif agent == controlled_agent:
            frozen_ticks += int(observation.get("frozen_ticks", 0) > 0)
            raw_action = int(action_fn(observation))
            action = _mask_action(raw_action, observation)
            if action != raw_action:
                masked_actions += 1
            steps += 1
        else:
            action = env.action_space(agent).sample()
        env.step(action)

    env.close()
    return EpisodeStats(
        seed=seed,
        policy=policy_name,
        reward=reward_total,
        score=reward_total / 1000.0,
        frozen_ticks=frozen_ticks,
        masked_actions=masked_actions,
        steps=steps,
    )


def _ppo_action(model: Any, observation: dict[str, Any]) -> int:
    action, _state = model.predict(preprocess_observation(observation), deterministic=True)
    return int(np.asarray(action).item())


def _mask_action(action: int, observation: dict[str, Any]) -> int:
    mask = np.asarray(observation.get("action_mask", []), dtype=np.int8).reshape(-1)
    legal = [idx for idx, value in enumerate(mask[:6]) if value]
    if action in legal:
        return int(action)
    if 4 in legal:
        return 4
    return int(legal[0]) if legal else 4


def _ppo_observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "agent_viewcone": spaces.Box(0.0, 1.0, shape=AGENT_VIEW_SHAPE, dtype=np.float32),
            "base_viewcone": spaces.Box(0.0, 1.0, shape=BASE_VIEW_SHAPE, dtype=np.float32),
            "scalars": spaces.Box(-1.0, 1.0, shape=SCALAR_SHAPE, dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=ACTION_MASK_SHAPE, dtype=np.float32),
        }
    )


def _write_eval(rows: list[EpisodeStats], csv_out: Path | None) -> None:
    if csv_out is None:
        return
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EpisodeStats.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _print_eval(rows: list[EpisodeStats]) -> None:
    grouped: dict[str, list[EpisodeStats]] = {}
    for row in rows:
        grouped.setdefault(row.policy, []).append(row)
    for policy, policy_rows in grouped.items():
        rewards = np.array([row.reward for row in policy_rows], dtype=np.float32)
        print(
            json.dumps(
                {
                    "policy": policy,
                    "episodes": len(policy_rows),
                    "mean_reward": float(np.mean(rewards)),
                    "median_reward": float(np.median(rewards)),
                    "mean_score": float(np.mean(rewards) / 1000.0),
                    "total_masked_actions": int(sum(row.masked_actions for row in policy_rows)),
                    "total_frozen_ticks": int(sum(row.frozen_ticks for row in policy_rows)),
                }
            )
        )


def _location(observation: dict[str, Any]) -> tuple[int, int]:
    value = observation.get("location", [0, 0])
    return int(value[0]), int(value[1])


def _scalar(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[0]) if len(value) else float(default)
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train PPO")
    _add_common(train_parser)
    train_parser.add_argument("--timesteps", type=int, default=5_000_000)
    train_parser.add_argument("--n-steps", type=int, default=2048)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--n-epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    train_parser.add_argument("--clip-range", type=float, default=0.2)
    train_parser.add_argument("--ent-coef", type=float, default=0.01)
    train_parser.add_argument("--features-dim", type=int, default=256)
    train_parser.add_argument("--tensorboard-log", type=Path, default=REPO_ROOT / "ae" / "train" / "runs")
    train_parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "ae" / "train" / "checkpoints")
    train_parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    train_parser.add_argument("--model-out", type=Path, default=REPO_ROOT / "ae" / "train" / "checkpoints" / "ae_ppo")
    train_parser.add_argument("--check-env", action="store_true")
    train_parser.add_argument("--progress-bar", action="store_true")
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval", help="evaluate PPO and/or planner")
    _add_common(eval_parser)
    eval_parser.add_argument("--model", type=Path)
    eval_parser.add_argument("--episodes", type=int, default=60)
    eval_parser.add_argument("--include-planner", action="store_true", default=True)
    eval_parser.add_argument("--csv-out", type=Path, default=REPO_ROOT / "ae" / "train" / "eval.csv")
    eval_parser.set_defaults(func=evaluate)

    smoke_parser = subparsers.add_parser("smoke", help="run wrapper smoke test")
    _add_common(smoke_parser)
    smoke_parser.add_argument("--steps", type=int, default=20)
    smoke_parser.set_defaults(func=smoke)
    return parser.parse_args()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--novice", action="store_true", help="use fixed novice map")
    parser.add_argument("--device", default="auto")


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
