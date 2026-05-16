"""Analyze ASR prediction quality against the GCP dataset labels."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import jiwer

from asr_analysis_common import (
    LANGUAGES,
    counter_to_dict,
    default_data_dir,
    default_predictions_path,
    load_env,
    markdown_table,
    read_jsonl,
    write_csv,
    write_json,
    write_text,
)


WER_TRANSFORMS = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"-": " ", "\u2014": " ", "\u2013": " "}),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

CER_TRANSFORMS = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"-": "", "\u2014": "", "\u2013": ""}),
        jiwer.RemoveWhiteSpace(replace_by_space=False),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)


def parse_args() -> argparse.Namespace:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="ASR data directory containing asr.jsonl.",
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
        "--worst-n",
        type=int,
        default=25,
        help="Worst samples to keep per language.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or default_data_dir()
    predictions_path = args.predictions or default_predictions_path()
    out_dir = args.out_dir

    records = read_jsonl(data_dir / "asr.jsonl")
    predictions = read_predictions(predictions_path)
    rows = build_rows(records, predictions, data_dir)
    summary = build_summary(rows, records, predictions, predictions_path, args.worst_n)

    write_json(out_dir / "prediction_analysis.json", summary)
    write_csv(out_dir / "worst_samples.csv", flatten_worst(summary), worst_fieldnames())
    write_text(out_dir / "prediction_analysis.md", render_markdown(summary))
    print(f"Wrote ASR prediction analysis to {out_dir}")


def read_predictions(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        data = data["predictions"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of predictions in {path}")
    return ["" if item is None else str(item) for item in data]


def build_rows(
    records: list[dict[str, Any]],
    predictions: list[str],
    data_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count = min(len(records), len(predictions))
    for index in range(count):
        record = records[index]
        reference = str(record.get("transcript", ""))
        prediction = predictions[index]
        language = str(record.get("language", "unknown")).lower()
        error_rate = score_single(language, reference, prediction)
        ref_words = len(reference.split())
        pred_words = len(prediction.split())
        ref_chars = len(reference)
        pred_chars = len(prediction)
        length_ratio = safe_ratio(pred_chars, ref_chars)
        rows.append(
            {
                "index": index,
                "key": record.get("key", ""),
                "audio": str(data_dir / str(record.get("audio", ""))),
                "language": language,
                "metric": "CER" if language == "chinese" else "WER",
                "error_rate": error_rate,
                "reference": reference,
                "prediction": prediction,
                "reference_words": ref_words,
                "prediction_words": pred_words,
                "reference_chars": ref_chars,
                "prediction_chars": pred_chars,
                "length_ratio": length_ratio,
                "empty_prediction": prediction.strip() == "",
                "long_prediction": length_ratio is not None and length_ratio >= 2.5,
                "short_prediction": length_ratio is not None and length_ratio <= 0.35,
                "repetition_candidate": has_repetition(prediction, language),
            }
        )
    return rows


def score_single(language: str, reference: str, prediction: str) -> float:
    scorer = (
        partial(
            jiwer.cer,
            reference_transform=CER_TRANSFORMS,
            hypothesis_transform=CER_TRANSFORMS,
        )
        if language == "chinese"
        else partial(
            jiwer.wer,
            reference_transform=WER_TRANSFORMS,
            hypothesis_transform=WER_TRANSFORMS,
        )
    )
    try:
        value = float(scorer(reference, prediction))
    except Exception:
        value = math.inf
    return round(value, 6) if math.isfinite(value) else value


def build_summary(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    predictions: list[str],
    predictions_path: Path,
    worst_n: int,
) -> dict[str, Any]:
    by_language_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language_rows[row["language"]].append(row)

    by_language: dict[str, Any] = {}
    language_error_rates: dict[str, float] = {}
    for language in LANGUAGES:
        lang_rows = by_language_rows.get(language, [])
        references = [row["reference"] for row in lang_rows]
        hypotheses = [row["prediction"] for row in lang_rows]
        error_rate = score_many(language, references, hypotheses) if lang_rows else None
        if error_rate is not None:
            language_error_rates[language] = error_rate
        by_language[language] = {
            "count": len(lang_rows),
            "metric": "CER" if language == "chinese" else "WER",
            "error_rate": error_rate,
            "empty_predictions": sum(bool(row["empty_prediction"]) for row in lang_rows),
            "long_predictions": sum(bool(row["long_prediction"]) for row in lang_rows),
            "short_predictions": sum(bool(row["short_prediction"]) for row in lang_rows),
            "repetition_candidates": sum(
                bool(row["repetition_candidate"]) for row in lang_rows
            ),
            "worst_samples": sorted(
                lang_rows,
                key=lambda row: row["error_rate"],
                reverse=True,
            )[:worst_n],
        }

    mean_error = (
        sum(language_error_rates.values()) / len(LANGUAGES)
        if len(language_error_rates) == len(LANGUAGES)
        else None
    )
    macro_score = max(0.0, 1.0 - mean_error) if mean_error is not None else None

    return {
        "predictions_path": str(predictions_path),
        "records_count": len(records),
        "predictions_count": len(predictions),
        "aligned_count": len(rows),
        "count_mismatch": len(records) != len(predictions),
        "language_counts": counter_to_dict(Counter(row["language"] for row in rows)),
        "language_error_rates": language_error_rates,
        "mean_error_rate": round(mean_error, 6) if mean_error is not None else None,
        "macro_score": round(macro_score, 6) if macro_score is not None else None,
        "empty_predictions": sum(bool(row["empty_prediction"]) for row in rows),
        "long_predictions": sum(bool(row["long_prediction"]) for row in rows),
        "short_predictions": sum(bool(row["short_prediction"]) for row in rows),
        "repetition_candidates": sum(bool(row["repetition_candidate"]) for row in rows),
        "by_language": by_language,
    }


def score_many(
    language: str,
    references: list[str],
    hypotheses: list[str],
) -> float:
    scorer = (
        partial(
            jiwer.cer,
            reference_transform=CER_TRANSFORMS,
            hypothesis_transform=CER_TRANSFORMS,
        )
        if language == "chinese"
        else partial(
            jiwer.wer,
            reference_transform=WER_TRANSFORMS,
            hypothesis_transform=WER_TRANSFORMS,
        )
    )
    return round(float(scorer(references, hypotheses)), 6)


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def has_repetition(text: str, language: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return False
    if language == "chinese":
        compact = re.sub(r"\s+", "", normalized)
        return bool(re.search(r"(.{2,8})\1{2,}", compact))
    tokens = normalized.split()
    if len(tokens) < 8:
        return False
    for span in (1, 2, 3, 4):
        for start in range(0, len(tokens) - span * 3 + 1):
            chunk = tokens[start : start + span]
            if (
                tokens[start + span : start + span * 2] == chunk
                and tokens[start + span * 2 : start + span * 3] == chunk
            ):
                return True
    return False


def flatten_worst(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for language, data in summary["by_language"].items():
        for row in data["worst_samples"]:
            rows.append(row)
    rows.sort(key=lambda row: (row["language"], -float(row["error_rate"])))
    return rows


def render_markdown(summary: dict[str, Any]) -> str:
    language_rows = []
    for language in LANGUAGES:
        data = summary["by_language"][language]
        language_rows.append(
            [
                language,
                data["count"],
                data["metric"],
                data["error_rate"],
                data["empty_predictions"],
                data["long_predictions"],
                data["short_predictions"],
                data["repetition_candidates"],
            ]
        )

    lines = [
        "# ASR Prediction Analysis",
        "",
        f"- Predictions: `{summary['predictions_path']}`",
        f"- Records: {summary['records_count']}",
        f"- Predictions: {summary['predictions_count']}",
        f"- Aligned pairs: {summary['aligned_count']}",
        f"- Count mismatch: {summary['count_mismatch']}",
        f"- Mean error rate: {summary['mean_error_rate']}",
        f"- Macro score: {summary['macro_score']}",
        "",
        "## By Language",
        "",
        markdown_table(
            [
                "language",
                "count",
                "metric",
                "error",
                "empty",
                "long",
                "short",
                "repeated",
            ],
            language_rows,
        ),
        "",
        "## Failure Buckets",
        "",
        f"- Empty predictions: {summary['empty_predictions']}",
        f"- Long predictions: {summary['long_predictions']}",
        f"- Short predictions: {summary['short_predictions']}",
        f"- Repetition candidates: {summary['repetition_candidates']}",
        "",
        "## Worst Samples",
        "",
        "See `worst_samples.csv` for references, predictions, and original audio paths.",
    ]
    return "\n".join(lines)


def worst_fieldnames() -> list[str]:
    return [
        "index",
        "key",
        "audio",
        "language",
        "metric",
        "error_rate",
        "reference",
        "prediction",
        "reference_words",
        "prediction_words",
        "reference_chars",
        "prediction_chars",
        "length_ratio",
        "empty_prediction",
        "long_prediction",
        "short_prediction",
        "repetition_candidate",
    ]


if __name__ == "__main__":
    main()
