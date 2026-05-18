# Noise (Adversarial Noising) Solution Plan

## Problem Summary

Add adversarial noise to JPEG images to degrade **opponent teams' CV models** while staying within distortion constraints (SSIM and RMSE L2 norm limits). The public challenge specification says Qualifiers do not directly reward noise output, but it can matter in Finals by making opponents' CV inputs harder.

**Interface:** POST `/noise` port `5003`
- Input: base64-encoded JPEG bytes
- Output: base64-encoded adversarially-noised JPEG bytes

**Constraint:** Output must stay within SSIM and RMSE thresholds set by the organizers. Images that fail the fairness check are returned unchanged.

---

## Challenge Context

We do **not** have access to opponents' model weights or architecture. Attacks must be **transferable** across different model architectures. This means white-box attacks on a surrogate model whose perturbations transfer to black-box models.

---

## Recommended Strategy: Universal Adversarial Perturbations on Surrogate Model

### Approach Hierarchy (best → simplest)

1. **PGD attack on surrogate YOLO** — strongest transferable attack
2. **Universal perturbation** — precomputed, near-zero inference cost
3. **Frequency-domain noise** — heuristic, architecture-agnostic
4. **JPEG re-encoding with noise** — fallback, very simple

---

## Implementation Plan

### Option 1: PGD Attack on Surrogate YOLOv11 (Recommended)

Run PGD to minimize the detection confidence of a locally-trained YOLOv11 model. The perturbation transfers to other architectures.

```python
import torch
import torchvision.transforms as T
from ultralytics import YOLO

class NoiseManager:
    def __init__(self):
        self.surrogate = YOLO("best.pt")  # our own trained CV model
        self.surrogate.model.eval()
        self.eps = 8 / 255          # L-inf budget
        self.alpha = 2 / 255        # PGD step size
        self.steps = 10             # PGD iterations

    def noise(self, image: bytes) -> str:
        import io, base64
        from PIL import Image
        import numpy as np

        img = Image.open(io.BytesIO(image)).convert("RGB")
        img_tensor = T.ToTensor()(img).unsqueeze(0).cuda()  # [1, 3, H, W]
        
        delta = torch.zeros_like(img_tensor).uniform_(-self.eps, self.eps).cuda()
        delta.requires_grad_(True)

        for _ in range(self.steps):
            perturbed = (img_tensor + delta).clamp(0, 1)
            loss = self._detection_loss(perturbed)
            loss.backward()
            delta.data = (delta.data + self.alpha * delta.grad.sign()).clamp(-self.eps, self.eps)
            delta.grad.zero_()

        adv = ((img_tensor + delta).clamp(0, 1).squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
        buffered = io.BytesIO()
        Image.fromarray(adv).save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("ascii")

    def _detection_loss(self, img_tensor):
        # Minimize total detection confidence → maximize missed detections
        results = self.surrogate.model(img_tensor)
        # Sum of confidence scores (we want to minimize this → gradient descent)
        return -results[0][..., 4].sum()  # raw objectness scores
```

**Note:** The exact loss depends on YOLO version internals. An easier approach is to use the `ultralytics` training loss directly.

### Option 2: Universal Adversarial Perturbation (UAP)

Pre-compute a single perturbation that fools the surrogate model on average across many images. At inference time, just add it (no gradient computation needed → very fast).

```python
# Pre-training phase (done before competition):
uap = torch.zeros(1, 3, H, W)
for img_batch in training_images:
    # Update uap using fooling gradient
    ...
torch.save(uap, "uap.pt")

# Inference phase:
class NoiseManager:
    def __init__(self):
        self.uap = torch.load("uap.pt")

    def noise(self, image: bytes) -> str:
        img_tensor = load_image(image)
        adv = (img_tensor + self.uap).clamp(0, 1)
        return encode_jpeg(adv)
```

### Option 3: Frequency-Domain Noise (No Model Required)

Inject high-frequency noise in specific DCT bands — these tend to confuse CNNs while being visually imperceptible:

```python
import numpy as np
import cv2

def add_hf_noise(image_array: np.ndarray, strength: float = 5.0) -> np.ndarray:
    noise = np.random.randn(*image_array.shape).astype(np.float32) * strength
    # High-pass filter the noise
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    for c in range(3):
        noise[..., c] = cv2.filter2D(noise[..., c], -1, kernel)
    noised = np.clip(image_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noised
```

---

## Fairness Constraint Handling

The evaluation system checks SSIM and RMSE. Stay within limits by clamping perturbation magnitude:

```python
from skimage.metrics import structural_similarity as ssim
import numpy as np

def check_constraints(original: np.ndarray, noised: np.ndarray, ssim_min=0.9, rmse_max=10.0) -> bool:
    s = ssim(original, noised, channel_axis=2, data_range=255)
    rmse = np.sqrt(np.mean((original.astype(float) - noised.astype(float)) ** 2))
    return s >= ssim_min and rmse <= rmse_max
```

If constraints are violated, scale down the perturbation or fall back to a weaker attack.

---

## Speed Considerations

PGD with 10 steps is slow per image. Options:
- Use UAP (pre-computed, ~0 inference cost)
- Reduce steps to 5
- Use fp16 computation
- Cache perturbations for similar images

---

## requirements.txt

```
ultralytics>=8.3.0
torch>=2.1.0
torchvision
pillow
opencv-python-headless
scikit-image  # for SSIM check
numpy
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Attack doesn't transfer to opponent model | Use ensemble of surrogate architectures (YOLO + RT-DETR) |
| Perturbation exceeds SSIM/RMSE budget | Clamp delta to eps=4/255 L-inf; verify with fairness checker |
| PGD too slow for 30-min budget | Pre-compute UAP; or limit to 5 PGD steps |
| JPEG re-encoding destroys perturbation | Use high quality=95; or apply perturbation after initial decode |

---

## Scoring Checklist

- [ ] Verify output passes fairness checker (`test/noise_eval/fairness_checker.py`)
- [ ] Test that attacked images actually reduce detection mAP on surrogate model
- [ ] Measure inference time per image; ensure full test set fits in budget
- [ ] Fall back to frequency-domain noise if PGD is too slow
