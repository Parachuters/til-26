"""Collect planner demonstrations for AE behavior cloning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AE_SRC = REPO_ROOT / "ae" / "src"
TIL_ENV = REPO_ROOT / "til-26-ae"
for path in (AE_SRC, TIL_ENV):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ae_manager import AEManager  # noqa: E402
from ppo_preprocess import preprocess_observation  # noqa: E402
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


def collect(args: argparse.Namespace) -> None:
    os.environ["AE_ENABLE_RL"] = "0"

    arrays: dict[str, list[Any]] = {
        "agent_viewcone": [],
        "base_viewcone": [],
        "scalars": [],
        "action_mask": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "sources": [],
        "seeds": [],
        "steps": [],
    }
    episode_rewards: list[float] = []

    for episode_idx in range(args.episodes):
        seed = args.seed_start + episode_idx
        manager = AEManager()
        reward_total = _collect_episode(seed, args.novice, manager, arrays)
        episode_rewards.append(reward_total)
        if (episode_idx + 1) % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "episodes": episode_idx + 1,
                        "last_seed": seed,
                        "mean_reward": float(np.mean(episode_rewards[-args.log_every :])),
                        "samples": len(arrays["actions"]),
                    }
                )
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        agent_viewcone=np.asarray(arrays["agent_viewcone"], dtype=np.float32),
        base_viewcone=np.asarray(arrays["base_viewcone"], dtype=np.float32),
        scalars=np.asarray(arrays["scalars"], dtype=np.float32),
        action_mask=np.asarray(arrays["action_mask"], dtype=np.float32),
        actions=np.asarray(arrays["actions"], dtype=np.int64),
        rewards=np.asarray(arrays["rewards"], dtype=np.float32),
        dones=np.asarray(arrays["dones"], dtype=np.bool_),
        sources=np.asarray(arrays["sources"], dtype="<U16"),
        seeds=np.asarray(arrays["seeds"], dtype=np.int64),
        steps=np.asarray(arrays["steps"], dtype=np.int64),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "episodes": args.episodes,
                "samples": len(arrays["actions"]),
                "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            }
        )
    )


def _collect_episode(
    seed: int,
    novice: bool,
    manager: AEManager,
    arrays: dict[str, list[Any]],
) -> float:
    cfg = default_config()
    cfg.env.novice = bool(novice)
    cfg.env.render_mode = None
    env = bomberman_env.basic_env(cfg=cfg, env_wrappers=[])
    controlled_agent = env.possible_agents[0]
    reward_total = 0.0

    env.reset(seed=seed)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, _info = env.last()
        if agent == controlled_agent:
            reward_total += float(reward)

        if termination or truncation:
            action = None
        elif agent == controlled_agent:
            action = int(manager.ae(observation))
            processed = preprocess_observation(observation)
            arrays["agent_viewcone"].append(processed["agent_viewcone"])
            arrays["base_viewcone"].append(processed["base_viewcone"])
            arrays["scalars"].append(processed["scalars"])
            arrays["action_mask"].append(processed["action_mask"])
            arrays["actions"].append(action)
            arrays["rewards"].append(float(reward))
            arrays["dones"].append(False)
            arrays["sources"].append(manager.last_action_source or "unknown")
            arrays["seeds"].append(seed)
            arrays["steps"].append(int(observation.get("step", 0)))
        else:
            action = env.action_space(agent).sample()
        env.step(action)

    if arrays["dones"]:
        arrays["dones"][-1] = True
    env.close()
    return reward_total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "ae" / "train" / "results" / "planner_demos.npz")
    parser.add_argument("--novice", action="store_true", help="collect on fixed novice maps")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    collect(parse_args())


if __name__ == "__main__":
    main()
