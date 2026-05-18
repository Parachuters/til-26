"""Analyze official til test cv output against local COCO annotations."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCOPatched(COCO):
    """COCO wrapper that accepts an already-loaded annotation dictionary."""

    def __init__(self, annotations: dict[str, Any]):
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.dataset = annotations
        self.createIndex()


def parse_args() -> argparse.Namespace:
    team_track = os.getenv("TEAM_TRACK", "advanced")
    team_name = os.getenv("TEAM_NAME", "")
    default_results = (
        Path(f"/home/jupyter/{team_name}/cv_results.json")
        if team_name
        else Path("cv_results.json")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(f"/home/jupyter/{team_track}/cv/annotations.json"),
    )
    parser.add_argument("--results", type=Path, default=default_results)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug/cv_til_results_summary.json"),
    )
    return parser.parse_args()


def evaluate(annotations: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, float]:
    if not predictions:
        return {"map_50_95": 0.0, "map_50": 0.0, "map_75": 0.0}
    ground_truth = COCOPatched(annotations)
    detections = ground_truth.loadRes(predictions)
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "map_50_95": float(evaluator.stats[0]),
        "map_50": float(evaluator.stats[1]),
        "map_75": float(evaluator.stats[2]),
        "map_small": float(evaluator.stats[3]),
        "map_medium": float(evaluator.stats[4]),
        "map_large": float(evaluator.stats[5]),
        "ar_1": float(evaluator.stats[6]),
        "ar_10": float(evaluator.stats[7]),
        "ar_100": float(evaluator.stats[8]),
    }


def summarize(annotations: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    image_ids = {int(image["id"]) for image in annotations["images"]}
    gt_class_counts = Counter(int(ann["category_id"]) for ann in annotations.get("annotations", []))
    pred_class_counts = Counter(int(pred["category_id"]) for pred in predictions)
    by_image = Counter(int(pred["image_id"]) for pred in predictions)
    scores = [float(pred.get("score", 1.0)) for pred in predictions]
    invalid = [
        pred for pred in predictions
        if pred["bbox"][2] <= 0 or pred["bbox"][3] <= 0 or pred["bbox"][0] < 0 or pred["bbox"][1] < 0
    ]
    unknown_images = sorted({int(pred["image_id"]) for pred in predictions} - image_ids)
    return {
        "ground_truth_images": len(image_ids),
        "ground_truth_boxes": len(annotations.get("annotations", [])),
        "ground_truth_class_counts": gt_class_counts.most_common(),
        "detections": len(predictions),
        "images_with_detections": len(by_image),
        "detections_per_image_mean": len(predictions) / max(1, len(image_ids)),
        "detections_per_detected_image_median": statistics.median(by_image.values()) if by_image else 0,
        "max_detections_single_image": max(by_image.values()) if by_image else 0,
        "prediction_class_counts": pred_class_counts.most_common(),
        "invalid_boxes": len(invalid),
        "unknown_image_ids": unknown_images[:20],
        "unknown_image_id_count": len(unknown_images),
        "score_min": min(scores) if scores else None,
        "score_median": statistics.median(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "all_scores_equal": len(set(scores)) <= 1,
    }


def main() -> None:
    args = parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    predictions = json.loads(args.results.read_text(encoding="utf-8"))
    summary = {
        "annotations": str(args.annotations),
        "results": str(args.results),
        **evaluate(annotations, predictions),
        **summarize(annotations, predictions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
