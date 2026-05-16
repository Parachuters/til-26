"""Run the ASR dataset profile and prediction analysis scripts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from asr_analysis_common import default_data_dir, default_predictions_path, load_env


def parse_args() -> argparse.Namespace:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="ASR data directory containing asr.jsonl and audio files.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="JSON predictions file produced by til test asr.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/asr"),
        help="Directory for generated report files.",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Only profile the dataset; do not analyze prediction output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or default_data_dir()
    predictions = args.predictions or default_predictions_path()
    script_dir = Path(__file__).resolve().parent

    run(
        [
            sys.executable,
            str(script_dir / "profile_dataset.py"),
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(args.out_dir),
        ]
    )

    if args.skip_predictions:
        return

    if not predictions.exists():
        raise SystemExit(
            f"Predictions file not found: {predictions}. Run `til test asr` first "
            "or pass --skip-predictions."
        )

    run(
        [
            sys.executable,
            str(script_dir / "analyze_predictions.py"),
            "--data-dir",
            str(data_dir),
            "--predictions",
            str(predictions),
            "--out-dir",
            str(args.out_dir),
        ]
    )


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
