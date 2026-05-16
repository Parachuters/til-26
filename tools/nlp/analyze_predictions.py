"""Analyze NLP prediction failures against the local TIL test data.

The official score uses a model-based answer equivalence check, so this script
does not try to reproduce the final score. It separates the cases that can be
diagnosed from JSON alone: retrieval misses, L4 formatting mistakes, and L5
false-premise mistakes.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def _normal_answer(answer: Any) -> str:
    if answer is None:
        return ""
    return str(answer).strip()


def _doc_overlap(expected_docs: list[str], predicted_docs: list[str]) -> bool:
    return bool(set(expected_docs).intersection(predicted_docs[:3]))


def classify_prediction(
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    expected_docs = list(ground_truth.get("source_docs") or [])
    predicted_docs = list(prediction.get("documents") or [])[:3]
    reference_answer = _normal_answer(ground_truth.get("answer"))
    candidate_answer = _normal_answer(prediction.get("answer"))
    overlap = _doc_overlap(expected_docs, predicted_docs)

    if not expected_docs and not reference_answer:
        correct = not predicted_docs and not candidate_answer
        return {
            "bucket": "l4_correct" if correct else "l4_false_positive",
            "doc_overlap": False,
            "score_floor": 1.0 if correct else 0.0,
        }

    if expected_docs and not reference_answer:
        if not overlap:
            return {
                "bucket": "l5_doc_miss",
                "doc_overlap": False,
                "score_floor": 0.0,
            }
        if candidate_answer:
            return {
                "bucket": "l5_non_empty_answer",
                "doc_overlap": True,
                "score_floor": 0.4,
            }
        return {
            "bucket": "l5_correct",
            "doc_overlap": True,
            "score_floor": 1.0,
        }

    if not overlap:
        return {
            "bucket": "answerable_doc_miss",
            "doc_overlap": False,
            "score_floor": 0.0,
        }

    return {
        "bucket": "answerable_retrieval_hit",
        "doc_overlap": True,
        "score_floor": 0.4,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of predictions in {path}")
    return data


def analyze(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    sample_limit: int = 8,
) -> dict[str, Any]:
    if len(ground_truth) != len(predictions):
        raise ValueError(
            f"Mismatched lengths: {len(ground_truth)} ground-truth rows, "
            f"{len(predictions)} predictions"
        )

    buckets: Counter[str] = Counter()
    floor_total = 0.0
    samples: dict[str, list[dict[str, Any]]] = {}

    for index, (gt, pred) in enumerate(zip(ground_truth, predictions)):
        result = classify_prediction(gt, pred)
        bucket = result["bucket"]
        buckets[bucket] += 1
        floor_total += float(result["score_floor"])
        bucket_samples = samples.setdefault(bucket, [])
        if len(bucket_samples) < sample_limit:
            bucket_samples.append(
                {
                    "index": index,
                    "question": gt.get("question"),
                    "expected_docs": gt.get("source_docs") or [],
                    "predicted_docs": (pred.get("documents") or [])[:3],
                    "reference_answer": _normal_answer(gt.get("answer")),
                    "predicted_answer": _normal_answer(pred.get("answer")),
                }
            )

    total = len(ground_truth)
    return {
        "total": total,
        "bucket_counts": dict(sorted(buckets.items())),
        "diagnostic_score_floor": round(floor_total / total, 4) if total else 0.0,
        "samples": samples,
    }


def _default_data_dir() -> Path:
    team_track = os.getenv("TEAM_TRACK")
    if team_track:
        return Path("/home/jupyter") / team_track / "nlp"
    return Path("/home/jupyter") / "zuckerberger" / "nlp"


def _default_results_path() -> Path:
    team_name = os.getenv("TEAM_NAME")
    if team_name:
        return Path("/home/jupyter") / team_name / "nlp_results.json"
    return Path("/home/jupyter") / "zuckerberger" / "nlp_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=_default_data_dir() / "nlp.jsonl",
        help="Path to nlp.jsonl with ground truth.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=_default_results_path(),
        help="Path to nlp_results.json.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=8,
        help="Number of examples to print for each bucket.",
    )
    args = parser.parse_args()

    report = analyze(
        load_jsonl(args.questions),
        load_json(args.predictions),
        sample_limit=args.sample_limit,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
