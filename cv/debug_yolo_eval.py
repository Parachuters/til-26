"""Offline YOLO diagnostics for the TIL CV challenge.

Run this on the GCP Workbench instance where the training/test data exists.
It evaluates cv/best.pt directly against the COCO annotations, preserving
model confidence scores so you can compare model quality against til test cv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable


class COCOPatched(COCO):
    """COCO wrapper that accepts an already-loaded annotation dictionary."""

    def __init__(self, annotations: dict[str, Any]):
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.dataset = annotations
        self.createIndex()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    team_track = os.getenv("TEAM_TRACK", "advanced")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(f"/home/jupyter/{team_track}/cv"),
        help="Directory containing annotations.json and images/.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("cv/best.pt"),
        help="Path to trained YOLO weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug/cv_yolo_eval"),
        help="Directory for JSON summaries, predictions, and overlays.",
    )
    parser.add_argument("--imgsz-list", default="1280")
    parser.add_argument("--conf-list", default="0.001,0.01,0.025,0.05,0.1,0.2,0.3")
    parser.add_argument("--iou-list", default="0.6")
    parser.add_argument("--device", default=None, help="Example: 0, cpu, cuda:0.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Use the first N images after sorting by id. 0 means all images.",
    )
    parser.add_argument(
        "--class-id-mode",
        choices=("identity", "coco-category-order"),
        default="identity",
        help=(
            "identity keeps YOLO class ids as category_id. "
            "coco-category-order maps class index i to the i-th sorted COCO category id."
        ),
    )
    parser.add_argument(
        "--score-mode",
        choices=("model", "constant"),
        default="model",
        help="constant reproduces til test cv's score=1.0 behavior.",
    )
    parser.add_argument(
        "--overlay-count",
        type=int,
        default=12,
        help="Number of images to render with GT/prediction overlays per run.",
    )
    parser.add_argument(
        "--overlay-conf",
        type=float,
        default=0.25,
        help="Only draw predictions at or above this confidence on overlays.",
    )
    return parser.parse_args()


def load_annotations(data_dir: Path, max_images: int) -> dict[str, Any]:
    with (data_dir / "annotations.json").open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    images = sorted(annotations["images"], key=lambda item: int(item["id"]))
    if max_images > 0:
        keep_ids = {int(image["id"]) for image in images[:max_images]}
        annotations = dict(annotations)
        annotations["images"] = [image for image in images if int(image["id"]) in keep_ids]
        annotations["annotations"] = [
            ann for ann in annotations.get("annotations", [])
            if int(ann["image_id"]) in keep_ids
        ]
    else:
        annotations["images"] = images
    return annotations


def image_path(data_dir: Path, file_name: str) -> Path:
    candidates = [data_dir / "images" / file_name, data_dir / file_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image {file_name!r} under {data_dir}")


def category_mapper(
    annotations: dict[str, Any],
    mode: str,
) -> tuple[dict[int, int], dict[int, str]]:
    categories = sorted(annotations["categories"], key=lambda item: int(item["id"]))
    id_to_name = {int(category["id"]): str(category["name"]) for category in categories}
    if mode == "identity":
        return defaultdict(lambda: None), id_to_name  # type: ignore[return-value]
    return {index: int(category["id"]) for index, category in enumerate(categories)}, id_to_name


def batched(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def run_predictions(
    model: YOLO,
    annotations: dict[str, Any],
    data_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | None,
    batch_size: int,
    class_map: dict[int, int],
    score_mode: str,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    images = list(annotations["images"])

    for batch in tqdm(list(batched(images, batch_size)), desc=f"imgsz={imgsz} conf={conf}"):
        pil_images = [
            Image.open(image_path(data_dir, str(image["file_name"]))).convert("RGB")
            for image in batch
        ]
        results = model.predict(
            pil_images,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        for image_info, result in zip(batch, results):
            image_id = int(image_info["id"])
            for box in result.boxes:
                cls = int(box.cls[0])
                category_id = class_map.get(cls, cls)
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                score = 1.0 if score_mode == "constant" else float(box.conf[0])
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": [x1, y1, width, height],
                        "score": score,
                    }
                )
    return predictions


def evaluate_coco(
    annotations: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, float]:
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


def summarize_predictions(predictions: list[dict[str, Any]], image_count: int) -> dict[str, Any]:
    by_image = Counter(int(pred["image_id"]) for pred in predictions)
    scores = [float(pred["score"]) for pred in predictions]
    invalid = [
        pred for pred in predictions
        if pred["bbox"][2] <= 0 or pred["bbox"][3] <= 0 or pred["bbox"][0] < 0 or pred["bbox"][1] < 0
    ]
    return {
        "detections": len(predictions),
        "images": image_count,
        "images_with_detections": len(by_image),
        "detections_per_image_mean": len(predictions) / max(1, image_count),
        "detections_per_detected_image_median": statistics.median(by_image.values()) if by_image else 0,
        "max_detections_single_image": max(by_image.values()) if by_image else 0,
        "invalid_boxes": len(invalid),
        "class_counts": Counter(int(pred["category_id"]) for pred in predictions).most_common(),
        "score_min": min(scores) if scores else None,
        "score_median": statistics.median(scores) if scores else None,
        "score_max": max(scores) if scores else None,
    }


def draw_overlays(
    annotations: dict[str, Any],
    predictions: list[dict[str, Any]],
    data_dir: Path,
    output_dir: Path,
    id_to_name: dict[int, str],
    count: int,
    min_conf: float,
) -> None:
    if count <= 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    preds_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)
    for pred in predictions:
        if float(pred["score"]) >= min_conf:
            preds_by_image[int(pred["image_id"])].append(pred)

    rendered = 0
    for image_info in annotations["images"]:
        image_id = int(image_info["id"])
        if rendered >= count:
            break
        if not anns_by_image.get(image_id) and not preds_by_image.get(image_id):
            continue

        image = Image.open(image_path(data_dir, str(image_info["file_name"]))).convert("RGB")
        draw = ImageDraw.Draw(image)

        for ann in anns_by_image.get(image_id, []):
            left, top, width, height = [float(value) for value in ann["bbox"]]
            category_id = int(ann["category_id"])
            label = f"GT {id_to_name.get(category_id, category_id)}"
            draw.rectangle([left, top, left + width, top + height], outline="lime", width=3)
            draw.text((left, max(0, top - 12)), label, fill="lime")

        for pred in preds_by_image.get(image_id, []):
            left, top, width, height = [float(value) for value in pred["bbox"]]
            category_id = int(pred["category_id"])
            score = float(pred["score"])
            label = f"P {id_to_name.get(category_id, category_id)} {score:.2f}"
            draw.rectangle([left, top, left + width, top + height], outline="red", width=2)
            draw.text((left, top + height + 2), label, fill="red")

        safe_name = Path(str(image_info["file_name"])).stem
        image.save(output_dir / f"{image_id}_{safe_name}.jpg", quality=92)
        rendered += 1


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.data_dir, args.max_images)
    class_map, id_to_name = category_mapper(annotations, args.class_id_mode)
    image_count = len(annotations["images"])
    model = YOLO(str(args.model))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for imgsz in parse_int_list(args.imgsz_list):
        for conf in parse_float_list(args.conf_list):
            for iou in parse_float_list(args.iou_list):
                run_name = (
                    f"imgsz{imgsz}_conf{conf:g}_iou{iou:g}_"
                    f"{args.class_id_mode}_{args.score_mode}"
                )
                predictions = run_predictions(
                    model=model,
                    annotations=annotations,
                    data_dir=args.data_dir,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=args.device,
                    batch_size=args.batch,
                    class_map=class_map,
                    score_mode=args.score_mode,
                )
                metrics = evaluate_coco(annotations, predictions)
                pred_summary = summarize_predictions(predictions, image_count)
                summary = {
                    "run": run_name,
                    "imgsz": imgsz,
                    "conf": conf,
                    "iou": iou,
                    "class_id_mode": args.class_id_mode,
                    "score_mode": args.score_mode,
                    **metrics,
                    **pred_summary,
                }
                summaries.append(summary)
                write_json(args.output_dir / f"{run_name}_predictions.json", predictions)
                write_json(args.output_dir / f"{run_name}_summary.json", summary)
                draw_overlays(
                    annotations=annotations,
                    predictions=predictions,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir / f"{run_name}_overlays",
                    id_to_name=id_to_name,
                    count=args.overlay_count,
                    min_conf=args.overlay_conf,
                )
                print(json.dumps(summary, indent=2))

    csv_path = args.output_dir / "sweep_summary.csv"
    if summaries:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
