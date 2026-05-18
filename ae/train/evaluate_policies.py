"""Compare AE planner, BC, PPO, and hybrid policies on identical seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AE_SRC = REPO_ROOT / "ae" / "src"
TRAIN_DIR = REPO_ROOT / "ae" / "train"
for path in (AE_SRC, TRAIN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ae_manager import AEManager  # noqa: E402
from masked_recurrent_policy import load_masked_recurrent_policy  # noqa: E402
from train_ppo import EpisodeStats, run_episode  # noqa: E402


ActionFn = Callable[[dict[str, Any]], int]


def evaluate(args: argparse.Namespace) -> None:
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    rows: list[EpisodeStats] = []

    if args.planner:
        with _env({"AE_ENABLE_RL": "0"}):
            for seed in seeds:
                manager = AEManager()
                rows.append(run_episode(seed, args.novice, "planner", manager.ae))

    if args.bc_model is not None and args.bc_model.exists():
        rows.extend(_run_learned("bc", args.bc_model, seeds, args.novice))

    if args.ppo_model is not None and args.ppo_model.exists():
        rows.extend(_run_learned("ppo", args.ppo_model, seeds, args.novice))

    if args.hybrid_model is not None and args.hybrid_model.exists():
        with _env({"AE_ENABLE_RL": "1", "AE_POLICY_MODEL": str(args.hybrid_model)}):
            for seed in seeds:
                manager = AEManager()
                rows.append(run_episode(seed, args.novice, "hybrid", manager.ae))

    _write_rows(rows, args.csv_out)
    _print_summary(rows)


def _run_learned(
    name: str,
    model_path: Path,
    seeds: list[int],
    novice: bool,
) -> list[EpisodeStats]:
    policy = load_masked_recurrent_policy(model_path)
    rows: list[EpisodeStats] = []
    for seed in seeds:
        hidden_state = None

        def act(observation: dict[str, Any]) -> int:
            nonlocal hidden_state
            if int(observation.get("step", 0)) == 0:
                hidden_state = None
            result = policy.act(observation, hidden_state=hidden_state)
            hidden_state = result.hidden_state
            return result.action

        rows.append(run_episode(seed, novice, name, act))
    return rows


def _write_rows(rows: list[EpisodeStats], csv_out: Path | None) -> None:
    if csv_out is None:
        return
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EpisodeStats.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _print_summary(rows: list[EpisodeStats]) -> None:
    grouped: dict[str, list[EpisodeStats]] = {}
    for row in rows:
        grouped.setdefault(row.policy, []).append(row)
    for policy, policy_rows in grouped.items():
        rewards = np.asarray([row.reward for row in policy_rows], dtype=np.float32)
        print(
            json.dumps(
                {
                    "policy": policy,
                    "episodes": len(policy_rows),
                    "mean_reward": float(np.mean(rewards)),
                    "median_reward": float(np.median(rewards)),
                    "p10_reward": float(np.percentile(rewards, 10)),
                    "mean_score": float(np.mean(rewards) / 1000.0),
                    "masked_actions": int(sum(row.masked_actions for row in policy_rows)),
                    "frozen_ticks": int(sum(row.frozen_ticks for row in policy_rows)),
                }
            )
        )


@contextmanager
def _env(values: dict[str, str]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--novice", action="store_true")
    parser.add_argument("--planner", action="store_true", default=True)
    parser.add_argument("--bc-model", type=Path, default=REPO_ROOT / "ae" / "train" / "checkpoints" / "ae_bc.pt")
    parser.add_argument("--ppo-model", type=Path, default=REPO_ROOT / "ae" / "train" / "checkpoints" / "ae_ppo_recurrent.pt")
    parser.add_argument("--hybrid-model", type=Path, default=REPO_ROOT / "ae" / "train" / "checkpoints" / "ae_ppo_recurrent.pt")
    parser.add_argument("--csv-out", type=Path, default=REPO_ROOT / "ae" / "train" / "results" / "policy_eval.csv")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
