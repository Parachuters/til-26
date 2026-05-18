# CV Solution Plan — Advanced Track

## Problem Summary

Detect and classify **18 object types** (aircraft, vehicles, ships, military equipment) in images. Advanced track has noisier images and smaller targets compared to Novice.

**Scoring:** mAP@.5:.05:.95 (COCO-style), averaged over all IoU thresholds from 0.5 to 0.95 in steps of 0.05.

**Interface:** POST `/cv` port `5002`
- Input: base64-encoded JPEG bytes per instance
- Output: list of `{"bbox": [left, top, width, height], "category_id": 0-17}` per image

---

## Recommended Approach: YOLOv11 Fine-tuned

### Why YOLOv11?
- State-of-the-art real-time object detector with strong mAP@.5:.95
- `ultralytics` library handles training, export, and inference cleanly
- Native support for COCO-format datasets
- Fast inference on GPU (~5ms/image for YOLOv11m)

### Model Size Selection

| Model | Params | mAP (COCO) | Speed (ms) | Recommendation |
|---|---|---|---|---|
| YOLOv11n | 2.6M | 39.5 | 1.5 | Too weak for small targets |
| YOLOv11m | 20.1M | 51.5 | 4.7 | **Good balance** |
| YOLOv11l | 25.3M | 53.4 | 6.5 | Best accuracy, still fast |
| YOLOv11x | 56.9M | 54.7 | 11.3 | Use only if time budget allows |

Start with `YOLOv11l`; fall back to `m` if inference is too slow.

---

## Implementation Plan

### 1. Dataset Preparation

Expected dataset structure (YOLO format):
```
dataset/
  images/train/  *.jpg
  images/val/    *.jpg
  labels/train/  *.txt   (cx cy w h class — normalized, center format)
  labels/val/    *.txt
data.yaml
```

`data.yaml`:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 18
names: [class0, class1, ..., class17]  # fill in actual class names
```

### 2. Training

```python
from ultralytics import YOLO

model = YOLO("yolo11l.pt")  # pretrained COCO weights
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=1280,
    batch=-1,          # auto-batch; use an explicit value if needed
    device=0,
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,    # good for small objects
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    patience=20,       # early stopping
    project="runs/cv",
    name="yolo11l_adv",
)
```

**Key augmentations for advanced track (noisy, small targets):**
- `copy_paste=0.1` — helps with small object detection
- `mosaic=1.0` — standard; trains on mixed multi-image tiles
- Higher `scale` variance to simulate size variation

### 3. Inference Code (`cv_manager.py`)

```python
import io
import numpy as np
from PIL import Image
from ultralytics import YOLO

class CVManager:
    def __init__(self):
        self.model = YOLO("best.pt")  # path inside container
        self.model.to("cuda")

    def cv(self, image: bytes) -> list[dict]:
        img = Image.open(io.BytesIO(image)).convert("RGB")
        results = self.model.predict(
            img,
            conf=0.001,
            iou=0.6,
            imgsz=1280,
            device=0,
            verbose=False,
        )
        predictions = []
        for result in results:
            for box in result.boxes:
                # YOLO outputs xyxy; convert to LTWH
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                predictions.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # left, top, w, h
                    "category_id": int(box.cls[0]),
                })
        return predictions
```

### 4. Bounding Box Format

**YOLO internal format:** `cx cy w h` (normalized, center-based)
**YOLO `xyxy` output:** absolute pixel coords `x1 y1 x2 y2`
**Required output (LTWH):** `left top width height`

Conversion: `[x1, y1, x2-x1, y2-y1]`

### 5. Confidence Threshold Tuning

Lower `conf` threshold → more true positives but also more FPs. For mAP scoring, **lower conf threshold is generally better** (mAP evaluates across all thresholds via PR curve). Set `conf=0.001` for evaluation-style inference:

```python
self.model.predict(img, conf=0.001, iou=0.6, ...)
```

Tune on local validation set to maximize mAP@.5:.95. The local
`test/test_cv.py` harness rewrites every returned detection score to `1.0`,
so threshold tuning is more important than in a normal COCO evaluation where
detections are ranked by confidence.

### 6. Multi-Scale Inference (TTA)

Test-time augmentation for small targets:
```python
results = self.model.predict(img, augment=True, ...)
```
Adds horizontal flip + multi-scale. Adds ~3x compute cost; use only if within time budget.

### 7. Model Export for Faster Inference

Export to TensorRT for production:
```python
model.export(format="engine", device=0, half=True)  # TensorRT fp16
```
Then load with `YOLO("best.engine")`.

---

## Handling Noisy / Small Targets (Advanced Track)

- Use `imgsz=1280` during training for better small-object resolution
- Add noise augmentation to training: Gaussian noise, JPEG compression artifacts
- Consider SAHI (Slicing Aided Hyper Inference) for very small objects:
  ```python
  from sahi import AutoDetectionModel
  from sahi.predict import get_sliced_prediction
  result = get_sliced_prediction(img, detection_model, slice_height=320, slice_width=320, overlap_ratio=0.2)
  ```

---

## Dockerfile Notes

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.11-py3
RUN pip install ultralytics pillow
COPY best.pt /app/best.pt
```

## requirements.txt

```
ultralytics>=8.3.0
pillow
sahi  # optional, for sliced inference
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Small targets missed | Use imgsz=1280, copy_paste aug, SAHI |
| Noisy images degrade detection | Add noise augmentation during training |
| Slow TensorRT build time | Pre-export `.engine` file before competition |
| Category ID off-by-one | Verify class index alignment against provided class list |

---

## Scoring Checklist

- [ ] Output bbox in LTWH format (not XYXY or normalized XYWH)
- [ ] `category_id` is 0-indexed and matches provided class mapping
- [ ] Tune `conf` threshold on local val set for best mAP@.5:.95
- [ ] Verify inference time well within 30-min budget
- [ ] Test on sample noisy advanced track images
