"""Manages the CV model."""

import io
import os
from typing import Any

from PIL import Image

try:
    from ultralytics import YOLO
    import torch
    _ultralytics_available = True
except ImportError:
    _ultralytics_available = False


MODEL_PATH = os.getenv("CV_MODEL_PATH", "/workspace/best.pt")
CONF_THRESHOLD = float(os.getenv("CV_CONF_THRESHOLD", "0.001"))
IOU_THRESHOLD = float(os.getenv("CV_IOU_THRESHOLD", "0.6"))
# 1280 gives substantially better small-object detection on the advanced track.
# Override with CV_IMG_SIZE=640 if inference budget is tight.
IMG_SIZE = int(os.getenv("CV_IMG_SIZE", "1280"))
# Test-time augmentation (horizontal flip + multi-scale). Adds ~2x compute but
# typically gains 2-3 mAP points. Disable with CV_TTA=0 if too slow.
TTA = os.getenv("CV_TTA", "1") not in ("0", "false", "False", "no")


class CVManager:

    def __init__(self):
        if not _ultralytics_available:
            raise RuntimeError("ultralytics is not installed")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(MODEL_PATH)
        self.model.to(device)
        self.device = device

        # Warmup: run one dummy forward pass so that the first real inference
        # is not penalised by CUDA kernel compilation / memory allocation.
        dummy = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        self.model.predict(dummy, imgsz=IMG_SIZE, verbose=False)

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of dicts with "bbox" ([left, top, width, height]) and
            "category_id" (0-indexed).  Bounding-box values are pixel integers.
        """
        img = Image.open(io.BytesIO(image)).convert("RGB")
        results = self.model.predict(
            img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            augment=TTA,  # test-time augmentation
            device=self.device,
            verbose=False,
        )
        predictions = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                predictions.append({
                    # Round to integers; pixel coordinates should not be floats.
                    "bbox": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)],
                    "category_id": int(box.cls[0]),
                })
        return predictions
