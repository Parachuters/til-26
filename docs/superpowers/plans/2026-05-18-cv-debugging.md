# CV Debugging Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:systematic-debugging before implementing fixes. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify why the current CV model scores poorly and isolate whether the root cause is inference output formatting, confidence/score handling, data conversion, model training quality, or train/eval distribution mismatch.

**Architecture:** Treat CV as a pipeline: COCO dataset -> YOLO conversion -> training artifacts -> FastAPI manager -> test harness -> COCO mAP. Debug one boundary at a time and avoid changing model code until there is evidence for the failing boundary.

**Tech Stack:** Python, Ultralytics YOLO, PIL, FastAPI, pycocotools COCOeval, TIL `til test cv`.

---

### Task 1: Establish a Reproducible Baseline

**Files:**
- Read: `cv/src/cv_manager.py`
- Read: `test/test_cv.py`
- Output artifact: `/home/jupyter/$TEAM_NAME/cv_results.json` on GCP Workbench

- [ ] **Step 1: Confirm exact model and runtime config**

Run on the GCP Workbench instance:

```bash
echo "TEAM_NAME=$TEAM_NAME"
echo "TEAM_TRACK=$TEAM_TRACK"
echo "CV_MODEL_PATH=${CV_MODEL_PATH:-/workspace/best.pt}"
echo "CV_CONF_THRESHOLD=${CV_CONF_THRESHOLD:-0.001}"
echo "CV_IOU_THRESHOLD=${CV_IOU_THRESHOLD:-0.6}"
echo "CV_IMG_SIZE=${CV_IMG_SIZE:-1280}"
echo "CV_TTA=${CV_TTA:-0}"
ls -lh cv/best.pt
```

Expected: `cv/best.pt` exists and is the intended trained model, not a stale or missing artifact.

- [ ] **Step 2: Run the official local test once without changing code**

```bash
til build cv
til test cv
```

Record:
- Final `mAP@.5:.05:.95`
- Runtime or timeout behavior
- Number of detections written to `/home/jupyter/$TEAM_NAME/cv_results.json`

- [ ] **Step 3: Inspect result volume and obvious anomalies**

```bash
python - <<'PY'
import json, os
from collections import Counter

path = f"/home/jupyter/{os.environ['TEAM_NAME']}/cv_results.json"
preds = json.load(open(path))
print("detections:", len(preds))
print("classes:", Counter(p["category_id"] for p in preds).most_common())
bad = [
    p for p in preds
    if p["bbox"][2] <= 0 or p["bbox"][3] <= 0 or p["bbox"][0] < 0 or p["bbox"][1] < 0
]
print("bad boxes:", len(bad))
print("first 5:", preds[:5])
PY
```

Expected: No negative or zero-size boxes. If detections are extremely high, suspect false positives amplified by `score: 1.0` in `test/test_cv.py`.

### Task 2: Verify Evaluation Contract Before Model Changes

**Files:**
- Read: `cv/README.md`
- Read: `cv/src/cv_manager.py`
- Read: `test/test_cv.py`

- [ ] **Step 1: Validate output format**

Confirm each detection returned by `CVManager.cv()` is:

```python
{
    "bbox": [left, top, width, height],
    "category_id": int_category_id,
}
```

No YOLO normalized center coordinates should leave the manager.

- [ ] **Step 2: Check class-id mapping**

Inspect the training data YAML:

```bash
cat /home/jupyter/$TEAM_TRACK/cv_yolo/data.yaml
python - <<'PY'
import json, os
ann = json.load(open(f"/home/jupyter/{os.environ['TEAM_TRACK']}/cv/annotations.json"))
print("COCO categories:", ann["categories"])
PY
```

Expected: YOLO class index `0..N-1` matches the challenge's expected `category_id`. If COCO category IDs are not already contiguous from zero, verify `prepare_dataset.py` intentionally maps category IDs to contiguous YOLO indices and that evaluation expects those same contiguous IDs.

- [ ] **Step 3: Check score handling risk**

`test/test_cv.py` currently writes every detection with:

```python
"score": 1.0
```

This means confidence ranking is discarded. For debugging only, create a temporary local copy of the evaluator that preserves YOLO confidence and compare AP. If AP improves when scores are preserved, the root cause is not just model quality; it is false positives plus unranked detections.

### Task 3: Add a Diagnostic Prediction Export

**Files:**
- Temporarily modify or copy: `cv/src/cv_manager.py`
- Temporarily modify or copy: `test/test_cv.py`

- [ ] **Step 1: Export raw confidence and xyxy for debugging**

Temporarily include extra fields in detections:

```python
"score": float(box.conf[0]),
"_xyxy": [float(x1), float(y1), float(x2), float(y2)]
```

Do not submit this unless the challenge server accepts extra fields; this is only for diagnosis.

- [ ] **Step 2: Run local eval with confidence preserved**

Temporarily change `test/test_cv.py` result serialization to:

```python
"score": float(detection.get("score", 1.0))
```

Run:

```bash
til build cv
til test cv
```

Expected: If AP jumps materially, tune confidence/NMS and consider returning fewer high-quality boxes for the official harness. If AP remains poor, continue to dataset/model checks.

### Task 4: Measure Precision/Recall Across Thresholds

**Files:**
- Modify temporarily: `cv/src/cv_manager.py` through environment variables only

- [ ] **Step 1: Sweep confidence thresholds**

Run the same model with:

```bash
for conf in 0.001 0.01 0.025 0.05 0.1 0.2 0.3 0.5; do
  CV_CONF_THRESHOLD=$conf CV_TTA=0 til build cv
  CV_CONF_THRESHOLD=$conf CV_TTA=0 til test cv
done
```

Record mAP and detection count for each threshold.

- [ ] **Step 2: Sweep image size only after confidence**

```bash
for size in 640 960 1280 1536; do
  CV_IMG_SIZE=$size CV_CONF_THRESHOLD=<best_conf_from_step_1> CV_TTA=0 til build cv
  CV_IMG_SIZE=$size CV_CONF_THRESHOLD=<best_conf_from_step_1> CV_TTA=0 til test cv
done
```

Expected: Small-object AP may improve with larger `imgsz`, but runtime may fail. Keep a speed/accuracy table.

- [ ] **Step 3: Test TTA only if runtime allows**

```bash
CV_TTA=1 CV_CONF_THRESHOLD=<best_conf> CV_IMG_SIZE=<best_size> til build cv
CV_TTA=1 CV_CONF_THRESHOLD=<best_conf> CV_IMG_SIZE=<best_size> til test cv
```

Expected: If TTA improves AP but breaks runtime, it is not a viable final fix.

### Task 5: Validate Dataset Conversion

**Files:**
- Read: `cv/prepare_dataset.py`
- Read: `/home/jupyter/$TEAM_TRACK/cv/annotations.json`
- Read: `/home/jupyter/$TEAM_TRACK/cv_yolo/labels/{train,val}`

- [ ] **Step 1: Compare COCO boxes to YOLO labels for random samples**

Pick 20 images, convert YOLO labels back to pixels, and compare against original COCO bboxes.

Expected:
- Coordinates match within rounding tolerance.
- Class IDs match expected contiguous category indices.
- Images with no annotations have empty label files.

- [ ] **Step 2: Render visual overlays**

Create a temporary script that draws:
- Ground-truth boxes in green
- Model predictions in red
- Class IDs and confidence values

Inspect at least:
- 10 true positives
- 10 false positives
- 10 false negatives
- Several very small target images

Expected: Visual inspection should classify failures as localization, classification, missed small objects, duplicate boxes, or domain/noise issue.

### Task 6: Validate Training Quality

**Files:**
- Read: `cv/train_yolo.py`
- Read: `runs/cv/yolo11l_adv/results.csv`
- Read: `runs/cv/yolo11l_adv/weights/best.pt`

- [ ] **Step 1: Inspect training curves**

```bash
python - <<'PY'
import pandas as pd
p = "runs/cv/yolo11l_adv/results.csv"
df = pd.read_csv(p)
print(df.tail(10).to_string())
print("best map50-95:", df["metrics/mAP50-95(B)"].max())
print("best map50:", df["metrics/mAP50(B)"].max())
PY
```

Expected: If validation mAP is also poor, focus on training recipe and data. If validation mAP is good but official `til test cv` is poor, focus on inference/evaluation mismatch.

- [ ] **Step 2: Run Ultralytics validation directly**

```bash
yolo detect val model=cv/best.pt data=/home/jupyter/$TEAM_TRACK/cv_yolo/data.yaml imgsz=1280 conf=0.001 iou=0.6
```

Expected: Direct validation should roughly align with local split performance. Large mismatch against `til test cv` points to conversion, class mapping, or serving output issues.

### Task 7: Form One Root-Cause Hypothesis and Test It

**Files:**
- Depends on findings from Tasks 1-6

- [ ] **Step 1: Write the hypothesis**

Use this exact format:

```text
I think CV AP is poor because <specific root cause>, based on <specific evidence>.
The smallest test is <one controlled change>.
Success means <measurable metric>.
```

- [ ] **Step 2: Test only that hypothesis**

Examples of valid one-variable tests:
- Raise `CV_CONF_THRESHOLD` only.
- Preserve confidence in debug evaluator only.
- Fix class mapping only.
- Retrain with adjusted augmentation only.
- Change `imgsz` only.

- [ ] **Step 3: Decide next action from evidence**

If the hypothesis is confirmed, implement the smallest production-safe fix. If it is rejected, record the result and return to the next most likely boundary.

### Task 8: Only Then Implement the Fix

**Files:**
- Likely modify: `cv/src/cv_manager.py`
- Possibly modify: `cv/prepare_dataset.py`
- Possibly modify: `cv/train_yolo.py`
- Rebuild: `cv/Dockerfile` image through `til build cv`

- [ ] **Step 1: Write a small regression check**

At minimum, keep a script or notebook cell that verifies:
- Returned boxes are LTWH pixel boxes.
- `category_id` is an integer in the expected range.
- Empty detections return `[]`.
- Detection count is not explosively high on validation samples.

- [ ] **Step 2: Implement one fix**

Do not combine threshold tuning, image-size changes, data fixes, and retraining in the same commit.

- [ ] **Step 3: Verify with official path**

```bash
til build cv
til test cv
```

Expected: mAP improves against the recorded baseline without unacceptable runtime.

- [ ] **Step 4: Commit the proven change**

```bash
git add cv
git commit -m "fix(cv): improve detection evaluation performance"
```

---

## Initial Suspects From Current Code

1. `CONF_THRESHOLD=0.001` may produce many false positives.
2. `test/test_cv.py` assigns `score: 1.0` to every detection, so COCOeval cannot rank predictions by confidence.
3. `CVManager.cv()` does not return confidence, which is allowed by the challenge output but weak for local diagnosis.
4. Category remapping in `prepare_dataset.py` must match what the evaluator expects.
5. If training validation mAP is good but served mAP is poor, focus on inference contract and evaluator behavior. If both are poor, focus on data quality, augmentations, model size, and training schedule.

