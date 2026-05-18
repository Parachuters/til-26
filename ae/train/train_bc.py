"""Train the masked recurrent AE policy from planner demonstrations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
AE_SRC = REPO_ROOT / "ae" / "src"
for path in (AE_SRC,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from masked_recurrent_policy import (  # noqa: E402
    MaskedRecurrentPolicy,
    action_accuracy,
    masked_cross_entropy,
)


def train(args: argparse.Namespace) -> None:
    data = np.load(args.demos)
    seeds = data["seeds"]
    train_seeds, val_seeds = _seed_split(seeds, args.val_fraction)
    device = torch.device(args.device)
    model = MaskedRecurrentPolicy(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_accuracy = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for seed in np.random.default_rng(args.seed + epoch).permutation(train_seeds):
            losses.extend(_train_episode(model, optimizer, data, int(seed), args.bptt, device))

        val_metrics = _evaluate_accuracy(model, data, val_seeds, device)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": float(np.mean(losses)) if losses else 0.0,
                    **val_metrics,
                }
            )
        )
        if val_metrics["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["val_accuracy"]
            _save_checkpoint(model, args.out)


def _train_episode(
    model: MaskedRecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    data: np.lib.npyio.NpzFile,
    seed: int,
    bptt: int,
    device: torch.device,
) -> list[float]:
    idxs = np.flatnonzero(data["seeds"] == seed)
    if idxs.size == 0:
        return []
    idxs = idxs[np.argsort(data["steps"][idxs])]
    hidden = model.initial_state(1, device=device)
    losses: list[torch.Tensor] = []
    loss_values: list[float] = []

    for offset, idx in enumerate(idxs, start=1):
        batch = _sample_to_tensors(data, int(idx), device)
        logits, _value, hidden = model(
            batch["agent_viewcone"],
            batch["base_viewcone"],
            batch["scalars"],
            hidden,
        )
        losses.append(masked_cross_entropy(logits, batch["action_mask"], batch["actions"]))
        if offset % bptt == 0:
            loss_values.append(_optimize_chunk(optimizer, losses))
            hidden = hidden.detach()
            losses = []

    if losses:
        loss_values.append(_optimize_chunk(optimizer, losses))
    return loss_values


def _optimize_chunk(optimizer: torch.optim.Optimizer, losses: list[torch.Tensor]) -> float:
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [param for group in optimizer.param_groups for param in group["params"]],
        max_norm=1.0,
    )
    optimizer.step()
    return float(loss.detach().cpu().item())


@torch.no_grad()
def _evaluate_accuracy(
    model: MaskedRecurrentPolicy,
    data: np.lib.npyio.NpzFile,
    val_seeds: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    accuracies: list[float] = []
    safety_accuracies: list[float] = []
    for seed in val_seeds:
        idxs = np.flatnonzero(data["seeds"] == seed)
        if idxs.size == 0:
            continue
        idxs = idxs[np.argsort(data["steps"][idxs])]
        hidden = model.initial_state(1, device=device)
        for idx in idxs:
            batch = _sample_to_tensors(data, int(idx), device)
            logits, _value, hidden = model(
                batch["agent_viewcone"],
                batch["base_viewcone"],
                batch["scalars"],
                hidden,
            )
            acc = action_accuracy(logits, batch["action_mask"], batch["actions"])
            accuracies.append(acc)
            source_is_safety = "sources" in data and str(data["sources"][idx]) == "safety"
            if source_is_safety or float(batch["scalars"][0, 13].item()) > 0.5:
                safety_accuracies.append(acc)
    return {
        "val_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "val_safety_accuracy": float(np.mean(safety_accuracies)) if safety_accuracies else 0.0,
    }


def _sample_to_tensors(
    data: np.lib.npyio.NpzFile,
    idx: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "agent_viewcone": torch.as_tensor(data["agent_viewcone"][idx], dtype=torch.float32, device=device).unsqueeze(0),
        "base_viewcone": torch.as_tensor(data["base_viewcone"][idx], dtype=torch.float32, device=device).unsqueeze(0),
        "scalars": torch.as_tensor(data["scalars"][idx], dtype=torch.float32, device=device).unsqueeze(0),
        "action_mask": torch.as_tensor(data["action_mask"][idx], dtype=torch.float32, device=device).unsqueeze(0),
        "actions": torch.as_tensor([int(data["actions"][idx])], dtype=torch.long, device=device),
    }


def _seed_split(seeds: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(seeds)
    cutoff = max(1, int(round(unique.size * (1.0 - val_fraction))))
    return unique[:cutoff], unique[cutoff:]


def _save_checkpoint(model: MaskedRecurrentPolicy, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_dim": model.hidden_dim,
                "action_dim": model.action_dim,
                "scalar_dim": 14,
            },
        },
        out,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, default=REPO_ROOT / "ae" / "train" / "results" / "planner_demos.npz")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "ae" / "train" / "checkpoints" / "ae_bc.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bptt", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
