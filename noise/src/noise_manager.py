"""Manages the noise model."""

import base64
import io
import os

import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
    _torch_available = True
except ImportError:
    _torch_available = False

try:
    from ultralytics import YOLO
    _ultralytics_available = True
except ImportError:
    _ultralytics_available = False

try:
    from skimage.metrics import structural_similarity as ssim
    _skimage_available = True
except ImportError:
    _skimage_available = False


MODEL_PATH = os.getenv("NOISE_MODEL_PATH", "/workspace/best.pt")
EPS = float(os.getenv("NOISE_EPS", str(8 / 255)))
ALPHA = float(os.getenv("NOISE_ALPHA", str(2 / 255)))
# MI-FGSM momentum decay factor (μ). 1.0 = full momentum accumulation.
MOMENTUM = float(os.getenv("NOISE_MOMENTUM", "1.0"))
ATTACK_STEPS = int(os.getenv("NOISE_ATTACK_STEPS", "10"))
SSIM_MIN = float(os.getenv("NOISE_SSIM_MIN", "0.85"))
RMSE_MAX = float(os.getenv("NOISE_RMSE_MAX", "12.0"))
HF_NOISE_STRENGTH = float(os.getenv("NOISE_HF_STRENGTH", "6.0"))


def _encode_jpeg(arr: np.ndarray, quality: int = 95) -> str:
    buffered = io.BytesIO()
    Image.fromarray(arr).save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("ascii")


def _hf_noise(image_array: np.ndarray, strength: float = HF_NOISE_STRENGTH) -> np.ndarray:
    try:
        import cv2
        noise = np.random.randn(*image_array.shape).astype(np.float32) * strength
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        for c in range(image_array.shape[2]):
            noise[..., c] = cv2.filter2D(noise[..., c], -1, kernel)
        return np.clip(image_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    except ImportError:
        rng = np.random.default_rng()
        noise = rng.normal(0, strength, image_array.shape).astype(np.float32)
        return np.clip(image_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _check_constraints(original: np.ndarray, noised: np.ndarray) -> bool:
    rmse = float(np.sqrt(np.mean((original.astype(np.float64) - noised.astype(np.float64)) ** 2)))
    if rmse > RMSE_MAX:
        return False
    if _skimage_available:
        s = ssim(original, noised, channel_axis=2, data_range=255)
        if s < SSIM_MIN:
            return False
    return True


class NoiseManager:

    def __init__(self):
        self.surrogate = None
        self.device = "cpu"
        if _torch_available and _ultralytics_available and os.path.exists(MODEL_PATH):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            surrogate = YOLO(MODEL_PATH)
            self.surrogate = surrogate.model
            self.surrogate.eval()
            for p in self.surrogate.parameters():
                p.requires_grad_(False)
            self.surrogate.to(self.device)

    def noise(self, image: bytes) -> str:
        """Performs adversarial noising on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A string containing the output image encoded in base64.
        """
        img = Image.open(io.BytesIO(image)).convert("RGB")
        original_arr = np.array(img)

        try:
            if self.surrogate is not None and _torch_available:
                noised_arr = self._mifgsm_attack(original_arr)
            else:
                noised_arr = _hf_noise(original_arr)

            if not _check_constraints(original_arr, noised_arr):
                noised_arr = self._scale_to_constraints(original_arr, noised_arr)

            return _encode_jpeg(noised_arr)
        except Exception as e:
            print(f"Noise attack failed: {e}, falling back to HF noise")
            try:
                noised_arr = _hf_noise(original_arr)
                if _check_constraints(original_arr, noised_arr):
                    return _encode_jpeg(noised_arr)
            except Exception:
                pass
            return base64.b64encode(image).decode("ascii")

    def _mifgsm_attack(self, image_array: np.ndarray) -> np.ndarray:
        """Momentum Iterative FGSM (MI-FGSM) adversarial attack.

        MI-FGSM accumulates a momentum vector over iterations which smooths
        the gradient direction and produces perturbations that transfer much
        better to unknown architectures compared to vanilla PGD.
        """
        img_tensor = T.ToTensor()(Image.fromarray(image_array)).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        delta = torch.empty_like(img_tensor).uniform_(-EPS, EPS)
        delta.requires_grad_(True)

        # Momentum accumulator (initialised to zero)
        momentum = torch.zeros_like(img_tensor)

        try:
            for _ in range(ATTACK_STEPS):
                if delta.grad is not None:
                    delta.grad.zero_()

                perturbed = (img_tensor + delta).clamp(0, 1)
                loss = self._detection_loss(perturbed)
                loss.backward()

                with torch.no_grad():
                    # Normalise gradient by its L1 norm (stabilises step size)
                    grad = delta.grad
                    grad_norm = grad.abs().mean() + 1e-8
                    grad = grad / grad_norm

                    # Momentum update: g_{t+1} = μ * g_t + ∇
                    momentum = MOMENTUM * momentum + grad

                    # Sign-based step
                    delta.data = (
                        delta.data + ALPHA * momentum.sign()
                    ).clamp(-EPS, EPS)

            with torch.no_grad():
                adv = (img_tensor + delta).clamp(0, 1)
                adv_arr = (
                    adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
        finally:
            # Release GPU tensors to prevent VRAM fragmentation across calls.
            del delta, momentum, img_tensor
            if _torch_available and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return adv_arr

    def _detection_loss(self, img_tensor: "torch.Tensor") -> "torch.Tensor":
        with torch.enable_grad():
            result = self.surrogate(img_tensor)
            if isinstance(result, (list, tuple)):
                preds = result[0]
            else:
                preds = result
            # YOLOv8/11 anchor-free: [batch, 4+nc, num_preds]
            # Minimise class confidence scores to suppress detections.
            if preds.dim() == 3:
                cls_scores = preds[:, 4:, :].sigmoid()
                return cls_scores.max(dim=1)[0].sum()
            # Fallback: minimise all output activations
            return preds.abs().sum()

    def _scale_to_constraints(
        self, original: np.ndarray, noised: np.ndarray
    ) -> np.ndarray:
        delta = noised.astype(np.float32) - original.astype(np.float32)
        for scale in [0.8, 0.6, 0.4, 0.2, 0.1]:
            candidate = np.clip(
                original.astype(np.float32) + delta * scale, 0, 255
            ).astype(np.uint8)
            if _check_constraints(original, candidate):
                return candidate
        return original
