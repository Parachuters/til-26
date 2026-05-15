"""Pre-downloads the Faster-Whisper model during image build."""

import os

from faster_whisper import WhisperModel


def main() -> None:
    model_name = os.getenv("ASR_MODEL_NAME", "large-v3")
    download_root = os.getenv("HF_HOME")

    WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        download_root=download_root,
    )


if __name__ == "__main__":
    main()
