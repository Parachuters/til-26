import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_asr_manager_module(monkeypatch):
    state = {"init_args": None, "transcribe_kwargs": None}

    class _DummyWhisperModel:
        def __init__(self, *args, **kwargs):
            state["init_args"] = args
            state["init_kwargs"] = kwargs

        def transcribe(self, audio, **kwargs):
            state["transcribe_kwargs"] = kwargs
            return [types.SimpleNamespace(text=" hello")], types.SimpleNamespace(language="en")

    heavy_stubs = {
        "faster_whisper": types.SimpleNamespace(WhisperModel=_DummyWhisperModel),
        "soundfile": types.SimpleNamespace(),
        "torch": types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
        ),
        "opencc": types.SimpleNamespace(OpenCC=lambda mode: None),
    }
    previous = {name: sys.modules.get(name) for name in heavy_stubs}
    sys.modules.update(heavy_stubs)
    module_path = Path(__file__).resolve().parents[1] / "asr" / "src" / "asr_manager.py"
    spec = importlib.util.spec_from_file_location("asr_manager_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
        return module, state
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def test_default_asr_model_is_turbo_for_time_first(monkeypatch):
    monkeypatch.delenv("ASR_MODEL_NAME", raising=False)

    module, state = _load_asr_manager_module(monkeypatch)
    module.ASRManager()

    assert state["init_args"][0] == "large-v3-turbo"


def test_vad_filter_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("ASR_VAD_FILTER", "0")
    module, state = _load_asr_manager_module(monkeypatch)
    monkeypatch.setattr(module, "preprocess", lambda audio_bytes: np.ones(16000, dtype=np.float32))

    manager = module.ASRManager()
    transcript = manager.asr(b"audio")

    assert transcript == "hello"
    assert state["transcribe_kwargs"]["vad_filter"] is False
