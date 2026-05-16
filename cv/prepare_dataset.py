"""Prepare the TIL CV COCO dataset for Ultralytics YOLO training."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/jupyter/advanced/cv"),
        help="Directory containing annotations.json and images/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jupyter/advanced/cv_yolo"),
        help="Directory to write YOLO images/, labels/, and data.yaml.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of images reserved for validation.",
    )
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking them.",
    )
    return parser.parse_args()


def find_image(data_dir: Path, file_name: str) -> Path:
    candidates = [data_dir / "images" / file_name, data_dir / file_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image {file_name!r} in {data_dir}")


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def clip_bbox(
    bbox: list[float],
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float] | None:
    left, top, width, height = bbox
    right = min(max(left + width, 0.0), image_width)
    bottom = min(max(top + height, 0.0), image_height)
    left = min(max(left, 0.0), image_width)
    top = min(max(top, 0.0), image_height)
    width = right - left
    height = bottom - top
    if width <= 0.0 or height <= 0.0:
        return None
    return left, top, width, height


def yolo_line(
    annotation: dict[str, Any],
    category_to_index: dict[int, int],
    image_width: int,
    image_height: int,
) -> str | None:
    clipped = clip_bbox(
        [float(value) for value in annotation["bbox"]],
        float(image_width),
        float(image_height),
    )
    if clipped is None:
        return None
    left, top, width, height = clipped
    x_center = (left + width / 2.0) / image_width
    y_center = (top + height / 2.0) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    class_index = category_to_index[int(annotation["category_id"])]
    return (
        f"{class_index} {x_center:.8f} {y_center:.8f} "
        f"{norm_width:.8f} {norm_height:.8f}"
    )


def write_data_yaml(output_dir: Path, names: list[str]) -> None:
    quoted_names = ", ".join(json.dumps(name) for name in names)
    content = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(names)}\n"
        f"names: [{quoted_names}]\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def reset_output_dir(data_dir: Path, output_dir: Path) -> None:
    if data_dir.resolve() == output_dir.resolve():
        raise ValueError("--output-dir must be different from --data-dir")
    for child_name in ("images", "labels"):
        child = output_dir / child_name
        if child.exists():
            shutil.rmtree(child)
    data_yaml = output_dir / "data.yaml"
    if data_yaml.exists():
        data_yaml.unlink()


def main() -> None:
    args = parse_args()
    reset_output_dir(args.data_dir, args.output_dir)
    annotations_path = args.data_dir / "annotations.json"
    with annotations_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    categories = sorted(coco["categories"], key=lambda item: item["id"])
    names = [str(category["name"]) for category in categories]
    category_to_index = {
        int(category["id"]): index for index, category in enumerate(categories)
    }

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    images = list(coco["images"])
    random.Random(args.seed).shuffle(images)
    val_count = max(1, round(len(images) * args.val_fraction))
    val_ids = {int(image["id"]) for image in images[:val_count]}

    for split in ("train", "val"):
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_counts = {"train": 0, "val": 0}
    box_counts = {"train": 0, "val": 0}
    for image in images:
        image_id = int(image["id"])
        split = "val" if image_id in val_ids else "train"
        src = find_image(args.data_dir, str(image["file_name"]))
        relative_name = Path(str(image["file_name"]))
        dst = args.output_dir / "images" / split / relative_name
        link_or_copy(src, dst, args.copy_images)

        label_path = (args.output_dir / "labels" / split / relative_name).with_suffix(
            ".txt"
        )
        lines = []
        for annotation in annotations_by_image.get(image_id, []):
            line = yolo_line(
                annotation,
                category_to_index,
                int(image["width"]),
                int(image["height"]),
            )
            if line is not None:
                lines.append(line)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        image_counts[split] += 1
        box_counts[split] += len(lines)

    write_data_yaml(args.output_dir, names)
    print(f"Wrote {args.output_dir / 'data.yaml'}")
    print(
        "Images: "
        f"train={image_counts['train']} val={image_counts['val']}; "
        f"boxes: train={box_counts['train']} val={box_counts['val']}"
    )
    print(f"Classes: {len(names)}")


if __name__ == "__main__":
    main()
