"""Profile the ASR training/evaluation dataset on GCP.

The script reads /home/jupyter/$TEAM_TRACK/asr/asr.jsonl by default and writes
compact reports under reports/asr so they can be committed and synced.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from asr_analysis_common import (
    counter_to_dict,
    default_data_dir,
    group_by_language,
    load_env,
    markdown_table,
    numeric_summary,
    read_jsonl,
    write_csv,
    write_json,
    write_text,
)


NEAR_SILENT_RMS = 1e-4
NEAR_SILENT_PEAK = 1e-3
CLIPPED_PEAK = 0.999


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
        "--out-dir",
        type=Path,
        default=Path("reports/asr"),
        help="Directory for generated report files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or default_data_dir()
    out_dir = args.out_dir
    metadata_path = data_dir / "asr.jsonl"

    records = read_jsonl(metadata_path)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for index, record in enumerate(records):
        row = profile_record(data_dir, record, index)
        rows.append(row)
        if row["status"] != "ok":
            errors.append(
                {
                    "index": str(index),
                    "key": str(record.get("key", "")),
                    "audio": str(record.get("audio", "")),
                    "status": row["status"],
                    "error": row.get("error", ""),
                }
            )

    summary = build_summary(rows, errors, data_dir)
    write_json(out_dir / "dataset_profile.json", summary)
    write_csv(out_dir / "dataset_profile_by_sample.csv", rows, sample_fieldnames())
    write_text(out_dir / "dataset_profile.md", render_markdown(summary))
    print(f"Wrote ASR dataset profile to {out_dir}")


def profile_record(data_dir: Path, record: dict[str, Any], index: int) -> dict[str, Any]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit(
            "soundfile is required for dataset profiling. Install dev dependencies "
            "with `pip install -r requirements-dev.txt` on GCP."
        ) from exc

    audio_rel = str(record.get("audio", ""))
    audio_path = data_dir / audio_rel
    transcript = str(record.get("transcript", ""))
    language = str(record.get("language", "unknown")).lower()

    row: dict[str, Any] = {
        "index": index,
        "key": record.get("key", ""),
        "audio": audio_rel,
        "language": language,
        "transcript_chars": len(transcript),
        "transcript_words": len(transcript.split()),
        "status": "ok",
    }

    if not audio_path.exists():
        row["status"] = "missing"
        row["error"] = "audio file not found"
        return row

    try:
        info = sf.info(str(audio_path))
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except Exception as exc:  # pragma: no cover - depends on corrupt data.
        row["status"] = "error"
        row["error"] = repr(exc)
        return row

    mono = audio.mean(axis=1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    mean_abs = float(np.mean(np.abs(mono))) if mono.size else 0.0
    dc_offset = float(np.mean(mono)) if mono.size else 0.0
    duration = float(len(mono) / sample_rate) if sample_rate else 0.0

    row.update(
        {
            "sample_rate": int(sample_rate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "duration_sec": round(duration, 6),
            "format": info.format,
            "subtype": info.subtype,
            "peak": round(peak, 8),
            "rms": round(rms, 8),
            "mean_abs": round(mean_abs, 8),
            "dc_offset": round(dc_offset, 8),
            "near_silent": bool(rms < NEAR_SILENT_RMS or peak < NEAR_SILENT_PEAK),
            "clipped": bool(peak >= CLIPPED_PEAK),
        }
    )
    return row


def build_summary(
    rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
    data_dir: Path,
) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    grouped = group_by_language(ok_rows)
    by_language: dict[str, Any] = {}

    for language, lang_rows in sorted(grouped.items()):
        by_language[language] = {
            "count": len(lang_rows),
            "duration_sec": numeric_summary(float(row["duration_sec"]) for row in lang_rows),
            "transcript_words": numeric_summary(
                float(row["transcript_words"]) for row in lang_rows
            ),
            "transcript_chars": numeric_summary(
                float(row["transcript_chars"]) for row in lang_rows
            ),
            "rms": numeric_summary(float(row["rms"]) for row in lang_rows),
            "peak": numeric_summary(float(row["peak"]) for row in lang_rows),
            "near_silent_count": sum(bool(row["near_silent"]) for row in lang_rows),
            "clipped_count": sum(bool(row["clipped"]) for row in lang_rows),
        }

    return {
        "data_dir": str(data_dir),
        "total_records": len(rows),
        "ok_records": len(ok_rows),
        "problem_records": len(errors),
        "language_counts": counter_to_dict(Counter(row["language"] for row in rows)),
        "sample_rates": counter_to_dict(Counter(row.get("sample_rate") for row in ok_rows)),
        "channels": counter_to_dict(Counter(row.get("channels") for row in ok_rows)),
        "formats": counter_to_dict(Counter(row.get("format") for row in ok_rows)),
        "subtypes": counter_to_dict(Counter(row.get("subtype") for row in ok_rows)),
        "duration_sec": numeric_summary(float(row["duration_sec"]) for row in ok_rows),
        "rms": numeric_summary(float(row["rms"]) for row in ok_rows),
        "peak": numeric_summary(float(row["peak"]) for row in ok_rows),
        "near_silent_count": sum(bool(row.get("near_silent")) for row in ok_rows),
        "clipped_count": sum(bool(row.get("clipped")) for row in ok_rows),
        "by_language": by_language,
        "errors": errors[:100],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    language_rows = []
    for language, data in summary["by_language"].items():
        language_rows.append(
            [
                language,
                data["count"],
                data["duration_sec"]["mean"],
                data["duration_sec"]["p95"],
                data["transcript_words"]["mean"],
                data["rms"]["median"],
                data["near_silent_count"],
                data["clipped_count"],
            ]
        )

    lines = [
        "# ASR Dataset Profile",
        "",
        f"- Data dir: `{summary['data_dir']}`",
        f"- Total records: {summary['total_records']}",
        f"- OK records: {summary['ok_records']}",
        f"- Problem records: {summary['problem_records']}",
        f"- Near-silent files: {summary['near_silent_count']}",
        f"- Clipped files: {summary['clipped_count']}",
        "",
        "## By Language",
        "",
        markdown_table(
            [
                "language",
                "count",
                "mean sec",
                "p95 sec",
                "mean words",
                "median rms",
                "near silent",
                "clipped",
            ],
            language_rows,
        ),
        "",
        "## Audio Format",
        "",
        f"- Sample rates: `{summary['sample_rates']}`",
        f"- Channels: `{summary['channels']}`",
        f"- Formats: `{summary['formats']}`",
        f"- Subtypes: `{summary['subtypes']}`",
    ]
    if summary["errors"]:
        lines.extend(
            [
                "",
                "## First Problem Records",
                "",
                markdown_table(
                    ["index", "key", "audio", "status", "error"],
                    (
                        [
                            item["index"],
                            item["key"],
                            item["audio"],
                            item["status"],
                            item["error"],
                        ]
                        for item in summary["errors"][:20]
                    ),
                ),
            ]
        )
    return "\n".join(lines)


def sample_fieldnames() -> list[str]:
    return [
        "index",
        "key",
        "audio",
        "language",
        "status",
        "error",
        "sample_rate",
        "channels",
        "frames",
        "duration_sec",
        "format",
        "subtype",
        "peak",
        "rms",
        "mean_abs",
        "dc_offset",
        "near_silent",
        "clipped",
        "transcript_chars",
        "transcript_words",
    ]


if __name__ == "__main__":
    main()
