"""Fine-tune a behavior-cloned AE policy with masked recurrent PPO."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
AE_SRC = REPO_ROOT / "ae" / "src"
for path in (AE_SRC, REPO_ROOT / "ae" / "train"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from masked_recurrent_policy import MaskedRecurrentPolicy, load_masked_recurrent_policy  # noqa: E402
from train_ppo import BombermanSingleAgentEnv  # noqa: E402


@dataclass
class Transition:
    obs: dict[str, np.ndarray]
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    hidden: np.ndarray


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    init_path = args.init if args.init is not None and args.init.exists() else None
    policy = (
        load_masked_recurrent_policy(init_path, device=args.device)
        if init_path is not None
        else MaskedRecurrentPolicy()
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    env = BombermanSingleAgentEnv(seed=args.seed, novice=args.novice)

    best_mean = -float("inf")
    below_bc_count = 0
    for update in range(args.updates):
        transitions = _collect_rollout(policy, env, args.rollout_steps, device)
        metrics = _ppo_update(policy, optimizer, transitions, args, device)

        if (update + 1) % args.eval_every == 0:
            rewards = _quick_eval(policy, args.seed + 10000 + update * args.eval_episodes, args.eval_episodes, args.novice)
            mean_reward = float(np.mean(rewards))
            metrics["eval_mean_reward"] = mean_reward
            metrics["eval_median_reward"] = float(np.median(rewards))
            if mean_reward > best_mean:
                best_mean = mean_reward
                _save_checkpoint(policy, args.out)
            if args.bc_floor is not None and mean_reward < args.bc_floor:
                below_bc_count += 1
            else:
                below_bc_count = 0
            if below_bc_count >= 2:
                print(json.dumps({"early_stop": "validation_below_bc", **metrics}))
                break

        print(json.dumps({"update": update + 1, **metrics}))

    env.close()


def _collect_rollout(
    policy: MaskedRecurrentPolicy,
    env: BombermanSingleAgentEnv,
    rollout_steps: int,
    device: torch.device,
) -> list[Transition]:
    transitions: list[Transition] = []
    obs, _info = env.reset()
    hidden = policy.initial_state(1, device=device)
    while len(transitions) < rollout_steps:
        batch = _obs_to_tensors(obs, device)
        with torch.no_grad():
            logits, value, next_hidden = policy(
                batch["agent_viewcone"],
                batch["base_viewcone"],
                batch["scalars"],
                hidden,
            )
            masked_logits = policy.mask_logits(logits, batch["action_mask"])
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        next_obs, reward, terminated, truncated, _info = env.step(int(action.item()))
        done = bool(terminated or truncated)
        transitions.append(
            Transition(
                obs={key: value.copy() for key, value in obs.items()},
                action=int(action.item()),
                log_prob=float(log_prob.item()),
                value=float(value.item()),
                reward=float(reward),
                done=done,
                hidden=hidden.detach().cpu().numpy(),
            )
        )
        obs = next_obs
        hidden = policy.initial_state(1, device=device) if done else next_hidden.detach()
        if done:
            obs, _info = env.reset()
    return transitions


def _ppo_update(
    policy: MaskedRecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    returns, advantages = _returns_and_advantages(transitions, args.gamma, args.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    idxs = np.arange(len(transitions))
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropy_values: list[float] = []

    for _epoch in range(args.epochs):
        np.random.shuffle(idxs)
        for start in range(0, len(idxs), args.batch_size):
            batch_idxs = idxs[start : start + args.batch_size]
            batch = _transition_batch(transitions, batch_idxs, device)
            logits, values, _hidden = policy(
                batch["agent_viewcone"],
                batch["base_viewcone"],
                batch["scalars"],
                batch["hidden"],
            )
            masked_logits = policy.mask_logits(logits, batch["action_mask"])
            dist = torch.distributions.Categorical(logits=masked_logits)
            log_probs = dist.log_prob(batch["actions"])
            ratio = torch.exp(log_probs - batch["old_log_probs"])
            adv = torch.as_tensor(advantages[batch_idxs], dtype=torch.float32, device=device)
            ret = torch.as_tensor(returns[batch_idxs], dtype=torch.float32, device=device)

            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, ret)
            entropy = dist.entropy().mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            policy_losses.append(float(policy_loss.detach().cpu().item()))
            value_losses.append(float(value_loss.detach().cpu().item()))
            entropy_values.append(float(entropy.detach().cpu().item()))

    return {
        "loss": float(np.mean(losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropy_values)),
        "rollout_reward": float(sum(t.reward for t in transitions)),
    }


def _returns_and_advantages(
    transitions: list[Transition],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray([t.reward for t in transitions], dtype=np.float32)
    values = np.asarray([t.value for t in transitions], dtype=np.float32)
    dones = np.asarray([t.done for t in transitions], dtype=np.float32)
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    next_value = 0.0
    for idx in reversed(range(len(transitions))):
        non_terminal = 1.0 - dones[idx]
        delta = rewards[idx] + gamma * next_value * non_terminal - values[idx]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[idx] = last_gae
        next_value = values[idx]
    return advantages + values, advantages


def _quick_eval(policy: MaskedRecurrentPolicy, seed_start: int, episodes: int, novice: bool) -> list[float]:
    rewards: list[float] = []
    device = next(policy.parameters()).device
    for offset in range(episodes):
        env = BombermanSingleAgentEnv(seed=seed_start + offset, novice=novice)
        obs, _info = env.reset(seed=seed_start + offset)
        hidden = policy.initial_state(1, device=device)
        total = 0.0
        done = False
        while not done:
            batch = _obs_to_tensors(obs, device)
            with torch.no_grad():
                logits, _value, hidden = policy(
                    batch["agent_viewcone"],
                    batch["base_viewcone"],
                    batch["scalars"],
                    hidden,
                )
                masked_logits = policy.mask_logits(logits, batch["action_mask"])
                action = int(torch.argmax(masked_logits, dim=1).item())
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        env.close()
        rewards.append(total)
    return rewards



def _obs_to_tensors(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        for key, value in obs.items()
    }


def _transition_batch(
    transitions: list[Transition],
    idxs: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "agent_viewcone": torch.as_tensor(np.stack([transitions[i].obs["agent_viewcone"] for i in idxs]), dtype=torch.float32, device=device),
        "base_viewcone": torch.as_tensor(np.stack([transitions[i].obs["base_viewcone"] for i in idxs]), dtype=torch.float32, device=device),
        "scalars": torch.as_tensor(np.stack([transitions[i].obs["scalars"] for i in idxs]), dtype=torch.float32, device=device),
        "action_mask": torch.as_tensor(np.stack([transitions[i].obs["action_mask"] for i in idxs]), dtype=torch.float32, device=device),
        "actions": torch.as_tensor([transitions[i].action for i in idxs], dtype=torch.long, device=device),
        "old_log_probs": torch.as_tensor([transitions[i].log_prob for i in idxs], dtype=torch.float32, device=device),
        "hidden": torch.as_tensor(np.concatenate([transitions[i].hidden for i in idxs], axis=0), dtype=torch.float32, device=device),
    }


def _save_checkpoint(policy: MaskedRecurrentPolicy, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "config": {
                "hidden_dim": policy.hidden_dim,
                "action_dim": policy.action_dim,
                "scalar_dim": 14,
            },
        },
        out,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, default=REPO_ROOT / "ae" / "train" / "checkpoints" / "ae_bc.pt")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "ae" / "train" / "checkpoints" / "ae_ppo_recurrent.pt")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--bc-floor", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--novice", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
