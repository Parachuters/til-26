"""Train YOLO11l for the TIL CV challenge and stage cv/best.pt."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/home/jupyter/advanced/cv_yolo/data.yaml"),
        help="Path to YOLO data.yaml.",
    )
    parser.add_argument("--model", default="yolo11l.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Ultralytics batch size. -1 enables auto-batch.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runs/cv")
    parser.add_argument("--name", default="yolo11l_adv")
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=Path("cv/best.pt"),
        help="Where to copy the trained best.pt.",
    )
    parser.add_argument("--no-copy", action="store_true", help="Do not stage best.pt.")
    parser.add_argument(
        "--cache",
        choices=["ram", "disk"],
        default=None,
        help="Optional Ultralytics dataset cache mode.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the previous run in project/name if supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} does not exist. Run cv/prepare_dataset.py first."
        )

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "resume": args.resume,
        "augment": True,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
        "degrees": 10,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "close_mosaic": 10,
    }
    if args.cache is not None:
        train_kwargs["cache"] = args.cache

    model.train(**train_kwargs)

    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    if not args.no_copy:
        if not best_path.exists():
            raise FileNotFoundError(f"Training finished but {best_path} was not found")
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, args.copy_to)
        print(f"Copied {best_path} -> {args.copy_to}")


if __name__ == "__main__":
    main()
