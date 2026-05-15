# ASR Solution Plan — Advanced Track

## Problem Summary

Transcribe noisy, single-speaker audio to text. Advanced track covers **four languages**: English, Malay, Tamil, and Chinese in roughly equal proportions, with more background noise than Novice.

**Scoring:**
- English / Tamil / Malay: `max(0, 1 - WER)` (word-level, lowercase, punctuation removed)
- Chinese: `max(0, 1 - CER)` (character-level)
- Final score = average across all languages in test set

**Interface:** POST `/asr` port `5001`
- Input: base64-encoded WAV bytes per instance
- Output: plain string transcript per instance

---

## Recommended Approach: Faster-Whisper (large-v3)

### Why Whisper?
Whisper `large-v3` is pretrained on 680k hours of multilingual data including all four required languages. It handles noise robustly and produces transcripts in the correct script (Mandarin Chinese characters for ZH, Latin for EN/MS/TA).

### Why Faster-Whisper?
`faster-whisper` is a CTranslate2-based reimplementation of Whisper that is **4x faster** with lower GPU memory footprint. Speed is 25% of the score (deadline = 30 min), so this is critical.

---

## Implementation Plan

### 1. Model Selection

```
Primary:   openai/whisper-large-v3  (via faster-whisper)
Fallback:  openai/whisper-medium    (lower VRAM, quicker)
```

Download and cache at container build time (not at runtime) via `requirements.txt` + a `download_model.py` script called in `Dockerfile`.

### 2. Language Detection Strategy

Whisper auto-detects language. For the advanced track with 4 languages, let Whisper do its own language detection rather than hard-coding. This avoids misrouting errors.

Optionally, run `detect_language` on first 30 seconds and pass `language=` hint to `transcribe()` for speed, but only if confidence > 0.9.

### 3. Audio Preprocessing

```python
import io
import numpy as np
import soundfile as sf

def preprocess(audio_bytes: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    # Resample to 16 kHz if needed (Whisper expects 16k)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)
```

### 4. Inference Code (`asr_manager.py`)

```python
from faster_whisper import WhisperModel

class ASRManager:
    def __init__(self):
        self.model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",  # fp16 for speed on GPU
        )

    def asr(self, audio_bytes: bytes) -> str:
        audio = preprocess(audio_bytes)
        segments, info = self.model.transcribe(
            audio,
            beam_size=5,
            vad_filter=True,           # skip silent sections
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return "".join(seg.text for seg in segments).strip()
```

### 5. Post-processing

- **English / Malay / Tamil:** Strip leading/trailing whitespace; evaluation strips punctuation and lowercases, so no manual lowercasing needed (but it doesn't hurt).
- **Chinese:** Whisper outputs Traditional or Simplified depending on training data. If mismatches occur, use `opencc` to convert Traditional → Simplified (or vice versa).
- Do NOT strip Chinese characters or add spaces between them — CER is character-level.

### 6. Batching for Speed

`faster-whisper` processes one file at a time. For further speed, use **concurrent processing** in the server layer:

```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)
results = list(executor.map(manager.asr, audio_bytes_list))
```

Alternatively, `whisperx` supports true GPU batching (`batch_size=16`), which is faster if GPU VRAM allows.

### 7. WhisperX Alternative (if VRAM is sufficient)

`whisperx` adds batched inference and word-level alignment:
```python
import whisperx
model = whisperx.load_model("large-v3", device="cuda", compute_type="float16")
result = model.transcribe(audio, batch_size=16)
```
Trade-off: higher VRAM usage (~10 GB for large-v3 + alignment model).

---

## Dockerfile Notes

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
RUN pip install faster-whisper soundfile librosa
# Pre-download model weights to avoid cold-start penalty
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu')"
```

## requirements.txt

```
faster-whisper>=1.0.0
soundfile
librosa
opencc-python-reimplemented  # optional, for Chinese script conversion
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Tamil WER high (low-resource language) | Whisper large-v3 trained on Tamil; verify on local test data |
| Chinese script mismatch (Trad vs Simp) | Use `opencc` for post-processing if needed |
| Slow inference exceeds 30-min budget | Use `faster-whisper` fp16; enable VAD filter to skip silence |
| VRAM OOM | Use `compute_type="int8"` for smaller footprint |

---

## Scoring Checklist

- [ ] Verify WER computation matches JiWER with `lowercase=True`, `RemovePunctuation()` transforms
- [ ] Test all 4 languages with local advanced track samples
- [ ] Benchmark inference time: target < 25 min for full test set
- [ ] Confirm output is plain string (no extra whitespace or newlines)
