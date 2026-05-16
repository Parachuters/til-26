"""Benchmark Faster-Whisper ASR decoding settings on a stratified GCP subset."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_predictions import score_many
from asr_analysis_common import (
    LANGUAGES,
    default_data_dir,
    load_env,
    markdown_table,
    read_jsonl,
    write_csv,
    write_json,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ASR_SRC = REPO_ROOT / "asr" / "src"


def parse_args() -> argparse.Namespace:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/asr"))
    parser.add_argument("--limit-per-language", type=int, default=25)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--beam-sizes", default="1,3")
    parser.add_argument("--no-speech-thresholds", default="0.6,0.7,0.8")
    parser.add_argument("--min-silence-ms", default="500")
    parser.add_argument("--repetition-penalties", default="1.1")
    parser.add_argument("--language-modes", default="single,two_pass")
    parser.add_argument("--model-name", default=os.getenv("ASR_MODEL_NAME", "large-v3"))
    parser.add_argument("--device", default=os.getenv("ASR_DEVICE", "cuda"))
    parser.add_argument(
        "--compute-type",
        default=os.getenv("ASR_COMPUTE_TYPE", "float16"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or default_data_dir()
    records = read_jsonl(data_dir / "asr.jsonl")
    samples = stratified_sample(records, args.limit_per_language, args.seed)
    combos = build_combos(args)

    print(f"Loaded {len(samples)} stratified samples from {data_dir}")
    print(f"Benchmarking {len(combos)} decoding combinations")

    if str(ASR_SRC) not in sys.path:
        sys.path.insert(0, str(ASR_SRC))
    from asr_manager import CLAIROS_INITIAL_PROMPT, preprocess
    from faster_whisper import WhisperModel

    try:
        from opencc import OpenCC
    except ImportError:  # pragma: no cover - optional local dependency.
        OpenCC = None

    model = WhisperModel(
        args.model_name,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=int(os.getenv("ASR_CPU_THREADS", "4")),
        num_workers=int(os.getenv("ASR_NUM_WORKERS", "1")),
        download_root=os.getenv("HF_HOME"),
    )
    cc = OpenCC("t2s") if OpenCC is not None else None

    results: list[dict[str, Any]] = []
    for combo_index, combo in enumerate(combos, start=1):
        print(f"[{combo_index}/{len(combos)}] {combo_label(combo)}")
        started = time.perf_counter()
        predictions = [
            transcribe_one(
                model,
                cc,
                preprocess,
                CLAIROS_INITIAL_PROMPT,
                data_dir,
                record,
                combo,
            )
            for record in samples
        ]
        elapsed = time.perf_counter() - started
        result = score_combo(samples, predictions, combo, elapsed)
        results.append(result)
        print(
            f"  macro={result['macro_score']:.4f} "
            f"mean_error={result['mean_error_rate']:.4f} "
            f"sec={elapsed:.1f}"
        )

    results.sort(key=lambda row: row["macro_score"], reverse=True)
    write_json(args.out_dir / "decoding_grid_results.json", results)
    write_csv(args.out_dir / "decoding_grid_results.csv", results, grid_fieldnames())
    write_text(args.out_dir / "decoding_grid_results.md", render_markdown(results))
    print(f"Wrote decoding grid results to {args.out_dir}")


def stratified_sample(
    records: list[dict[str, Any]],
    limit_per_language: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("language", "unknown")).lower()].append(record)

    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    for language in LANGUAGES:
        lang_records = list(grouped.get(language, []))
        rng.shuffle(lang_records)
        samples.extend(lang_records[:limit_per_language])
    rng.shuffle(samples)
    return samples


def build_combos(args: argparse.Namespace) -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for language_mode in split_strings(args.language_modes):
        for beam_size in split_ints(args.beam_sizes):
            for no_speech in split_floats(args.no_speech_thresholds):
                for min_silence in split_ints(args.min_silence_ms):
                    for repetition_penalty in split_floats(args.repetition_penalties):
                        combos.append(
                            {
                                "language_mode": language_mode,
                                "beam_size": beam_size,
                                "no_speech_threshold": no_speech,
                                "min_silence_duration_ms": min_silence,
                                "repetition_penalty": repetition_penalty,
                            }
                        )
    return combos


def transcribe_one(
    model: Any,
    cc: Any,
    preprocess_func: Any,
    initial_prompt: str,
    data_dir: Path,
    record: dict[str, Any],
    combo: dict[str, Any],
) -> str:
    with (data_dir / str(record["audio"])).open("rb") as handle:
        audio = preprocess_func(handle.read())
    if audio.size == 0:
        return ""

    detected_lang = None
    if combo["language_mode"] == "two_pass":
        try:
            _, info = model.transcribe(audio, beam_size=1, language=None)
            if info.language_probability >= float(os.getenv("ASR_LANG_CONF_THRESHOLD", "0.90")):
                detected_lang = info.language
        except Exception:
            detected_lang = None

    segments, info = model.transcribe(
        audio,
        beam_size=combo["beam_size"],
        language=detected_lang,
        initial_prompt=initial_prompt,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": combo["min_silence_duration_ms"],
        },
        no_speech_threshold=combo["no_speech_threshold"],
        repetition_penalty=combo["repetition_penalty"],
        condition_on_previous_text=False,
    )
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    transcript = " ".join(transcript.split())
    if transcript and cc is not None and (info.language or "").lower().startswith("zh"):
        transcript = cc.convert(transcript)
    return transcript


def score_combo(
    samples: list[dict[str, Any]],
    predictions: list[str],
    combo: dict[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    by_language: dict[str, dict[str, list[str]]] = {
        language: {"refs": [], "hyps": []} for language in LANGUAGES
    }
    for record, prediction in zip(samples, predictions):
        language = str(record.get("language", "unknown")).lower()
        if language in by_language:
            by_language[language]["refs"].append(str(record.get("transcript", "")))
            by_language[language]["hyps"].append(prediction)

    language_errors = {}
    for language, values in by_language.items():
        language_errors[language] = score_many(
            language, values["refs"], values["hyps"]
        )

    mean_error = float(np.mean(list(language_errors.values())))
    macro_score = max(0.0, 1.0 - mean_error)
    return {
        **combo,
        "sample_count": len(samples),
        "elapsed_sec": round(elapsed_sec, 3),
        "sec_per_sample": round(elapsed_sec / max(1, len(samples)), 3),
        "mean_error_rate": round(mean_error, 6),
        "macro_score": round(macro_score, 6),
        "english_error": language_errors["english"],
        "chinese_error": language_errors["chinese"],
        "malay_error": language_errors["malay"],
        "tamil_error": language_errors["tamil"],
    }


def render_markdown(results: list[dict[str, Any]]) -> str:
    top_rows = [
        [
            row["macro_score"],
            row["mean_error_rate"],
            row["language_mode"],
            row["beam_size"],
            row["no_speech_threshold"],
            row["min_silence_duration_ms"],
            row["repetition_penalty"],
            row["sec_per_sample"],
            row["english_error"],
            row["chinese_error"],
            row["malay_error"],
            row["tamil_error"],
        ]
        for row in results[:20]
    ]
    return "\n".join(
        [
            "# ASR Decoding Grid Results",
            "",
            "Top 20 configurations by macro score.",
            "",
            markdown_table(
                [
                    "macro",
                    "mean err",
                    "lang mode",
                    "beam",
                    "no speech",
                    "silence ms",
                    "rep penalty",
                    "sec/sample",
                    "en",
                    "zh",
                    "ms",
                    "ta",
                ],
                top_rows,
            ),
        ]
    )


def combo_label(combo: dict[str, Any]) -> str:
    return (
        f"mode={combo['language_mode']} beam={combo['beam_size']} "
        f"no_speech={combo['no_speech_threshold']} "
        f"silence={combo['min_silence_duration_ms']} "
        f"rep={combo['repetition_penalty']}"
    )


def split_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def split_ints(value: str) -> list[int]:
    return [int(item) for item in split_strings(value)]


def split_floats(value: str) -> list[float]:
    return [float(item) for item in split_strings(value)]


def grid_fieldnames() -> list[str]:
    return [
        "macro_score",
        "mean_error_rate",
        "language_mode",
        "beam_size",
        "no_speech_threshold",
        "min_silence_duration_ms",
        "repetition_penalty",
        "sample_count",
        "elapsed_sec",
        "sec_per_sample",
        "english_error",
        "chinese_error",
        "malay_error",
        "tamil_error",
    ]


if __name__ == "__main__":
    main()
