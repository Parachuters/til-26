"""Shared helpers for ASR dataset and prediction analysis scripts."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - scripts can still use explicit paths.
    load_dotenv = None


LANGUAGES = ("english", "chinese", "malay", "tamil")


def load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()


def default_data_dir() -> Path:
    team_track = os.getenv("TEAM_TRACK")
    if not team_track:
        raise SystemExit(
            "TEAM_TRACK is not set. Pass --data-dir or run from an environment "
            "with TEAM_TRACK loaded."
        )
    return Path("/home/jupyter") / team_track / "asr"


def default_predictions_path() -> Path:
    team_name = os.getenv("TEAM_NAME")
    if not team_name:
        raise SystemExit(
            "TEAM_NAME is not set. Pass --predictions or run from an environment "
            "with TEAM_NAME loaded."
        )
    return Path("/home/jupyter") / team_name / "asr_results.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(clean),
        "mean": round(mean(clean), 6),
        "min": round(clean[0], 6),
        "p10": round(percentile(clean, 10), 6),
        "median": round(median(clean), 6),
        "p90": round(percentile(clean, 90), 6),
        "p95": round(percentile(clean, 95), 6),
        "max": round(clean[-1], 6),
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def group_by_language(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("language", "unknown")).lower()].append(row)
    return dict(grouped)


def counter_to_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=lambda item: str(item))}


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ")
