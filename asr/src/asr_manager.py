"""Manages the ASR model."""

from __future__ import annotations

import io
import os
import re

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in the runtime image.
    torch = None

try:
    from opencc import OpenCC
except ImportError:  # pragma: no cover - optional dependency for Chinese output.
    OpenCC = None


TARGET_SAMPLE_RATE = 16_000
ZH_LANGUAGE_CODES = {"zh", "zh-cn", "zh-tw", "chinese"}
TRUE_VALUES = {"1", "true", "yes", "on"}

# Fictional vocabulary primer — biases Whisper toward Clairos-world jargon so
# that out-of-vocabulary slang terms are transcribed more accurately.
CLAIROS_INITIAL_PROMPT = (
    "Clairos, Haven, the Cascade, megacorporation, cy, ty, "
    "cyberware, netrunner, datacore, warlord, syndicate, blacksite, "
    "exosuit, cryo, biomech, droneswarm, neurallink, splice,"
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in TRUE_VALUES


def preprocess(audio_bytes: bytes) -> np.ndarray:
    """Decodes WAV bytes into a mono 16 kHz float32 waveform.

    The Advanced ASR data is already mono 16 kHz PCM, so the common path avoids
    resampling. Extra normalization is intentionally conservative because the
    multilingual scorer rewards lexical fidelity over perceptual cleanliness.
    """

    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        import librosa

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio

    if _env_bool("ASR_REMOVE_DC_OFFSET", True):
        audio = audio - float(np.mean(audio))

    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-8:
        return np.zeros_like(audio, dtype=np.float32)

    if _env_bool("ASR_NORMALIZE_QUIET", True):
        quiet_peak_threshold = float(os.getenv("ASR_QUIET_PEAK_THRESHOLD", "0.06"))
        target_peak = float(os.getenv("ASR_TARGET_PEAK", "0.9"))
        if peak < quiet_peak_threshold:
            audio = audio * min(target_peak / peak, 20.0)

    if _env_bool("ASR_SOFT_LIMIT_CLIPPED", False):
        clipped_peak_threshold = float(os.getenv("ASR_CLIPPED_PEAK_THRESHOLD", "0.999"))
        if peak >= clipped_peak_threshold:
            drive = float(os.getenv("ASR_SOFT_LIMIT_DRIVE", "1.5"))
            audio = np.tanh(drive * audio) / np.tanh(drive)

    return np.asarray(audio, dtype=np.float32)


class ASRManager:
    """Loads Faster-Whisper once and reuses it for all requests."""

    def __init__(self):
        device = os.getenv("ASR_DEVICE")
        if device is None:
            device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

        compute_type = os.getenv("ASR_COMPUTE_TYPE")
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        model_name = os.getenv("ASR_MODEL_NAME", "large-v3")
        cpu_threads = int(os.getenv("ASR_CPU_THREADS", "4"))
        num_workers = int(os.getenv("ASR_NUM_WORKERS", "1"))
        download_root = os.getenv("HF_HOME")

        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            download_root=download_root,
        )
        self.beam_size = int(os.getenv("ASR_BEAM_SIZE", "3"))
        self.min_silence_duration_ms = int(
            os.getenv("ASR_MIN_SILENCE_DURATION_MS", "500")
        )
        # Threshold below which a segment is considered non-speech. Higher
        # values reduce hallucinations on noisy / silent sections.
        self.no_speech_threshold = float(os.getenv("ASR_NO_SPEECH_THRESHOLD", "0.7"))
        # Penalises repeated n-grams — reduces Whisper's looping hallucination.
        self.repetition_penalty = float(os.getenv("ASR_REPETITION_PENALTY", "1.1"))
        # Minimum language-detection probability before we pass a hard language
        # hint to avoid the overhead of re-detecting on the full audio.
        self.lang_conf_threshold = float(os.getenv("ASR_LANG_CONF_THRESHOLD", "0.90"))
        self.language_detection_mode = os.getenv(
            "ASR_LANGUAGE_DETECTION_MODE", "single"
        ).lower()
        self.cc = OpenCC("t2s") if OpenCC is not None else None

    def asr(self, audio_bytes: bytes) -> str:
        """Performs ASR transcription on an audio file.

        Args:
            audio_bytes: The audio file in bytes.

        Returns:
            A string containing the transcription of the audio.
        """

        audio = preprocess(audio_bytes)
        if audio.size == 0:
            return ""

        detected_lang = None
        if self.language_detection_mode in {"two_pass", "two-pass", "detect"}:
            # Optional speed/accuracy tradeoff. It costs an extra decode pass,
            # so the default keeps Faster-Whisper's internal auto-detection.
            try:
                _, info = self.model.transcribe(audio, beam_size=1, language=None)
                if info.language_probability >= self.lang_conf_threshold:
                    detected_lang = info.language
            except Exception:
                pass  # fall back to auto-detect

        # --- Full transcription pass ---
        segments, info = self.model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=detected_lang,  # None → auto-detect
            initial_prompt=CLAIROS_INITIAL_PROMPT,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": self.min_silence_duration_ms,
            },
            no_speech_threshold=self.no_speech_threshold,
            repetition_penalty=self.repetition_penalty,
            condition_on_previous_text=False,
        )

        transcript = "".join(segment.text for segment in segments).strip()
        transcript = self._postprocess_transcript(transcript, info.language)
        return transcript

    def _postprocess_transcript(self, transcript: str, language: str | None) -> str:
        """Normalizes whitespace and optionally converts Chinese to Simplified."""

        transcript = re.sub(r"\s+", " ", transcript).strip()
        if transcript and self.cc is not None and self._is_chinese(language):
            transcript = self.cc.convert(transcript)
        return transcript

    @staticmethod
    def _is_chinese(language: str | None) -> bool:
        if language is None:
            return False
        return language.lower() in ZH_LANGUAGE_CODES
