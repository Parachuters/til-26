"""Fine-tune Whisper with LoRA adapters on the ASR dataset.

This is an offline GCP training utility, not part of the competition server.
It saves adapter weights to an ignored model directory and compact metrics to
reports/asr for Git sync.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asr_analysis_common import (
    LANGUAGES,
    default_data_dir,
    load_env,
    markdown_table,
    read_jsonl,
    write_json,
    write_text,
)


LANGUAGE_NAMES = {
    "english": "english",
    "chinese": "chinese",
    "malay": "malay",
    "tamil": "tamil",
}


def parse_args() -> argparse.Namespace:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("asr/models/whisper-lora"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/asr"))
    parser.add_argument(
        "--model-name",
        default=os.getenv("ASR_HF_MODEL", "openai/whisper-large-v3"),
    )
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or default_data_dir()

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    records = read_jsonl(data_dir / "asr.jsonl")
    train_records, eval_records = stratified_split(records, args.eval_ratio, args.seed)
    train_records = cap_records(train_records, args.max_train_samples, args.seed)
    eval_records = cap_records(eval_records, args.max_eval_samples, args.seed)

    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = WhisperAsrDataset(data_dir, train_records, processor)
    eval_dataset = WhisperAsrDataset(data_dir, eval_records, processor)
    collator = WhisperDataCollator(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        predict_with_generate=False,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=processor.feature_extractor,
    )

    train_result = trainer.train()
    trainer.save_model(str(args.out_dir))
    processor.save_pretrained(str(args.out_dir / "processor"))
    metrics = {
        "data_dir": str(data_dir),
        "model_name": args.model_name,
        "output_dir": str(args.out_dir),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "languages": summarize_languages(train_records, eval_records),
        "train_metrics": train_result.metrics,
        "eval_metrics": trainer.evaluate(),
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": ["q_proj", "v_proj"],
        },
    }
    write_json(args.report_dir / "whisper_lora_training_metrics.json", metrics)
    write_text(args.report_dir / "whisper_lora_training_metrics.md", render_metrics(metrics))
    print(f"Saved LoRA adapter to {args.out_dir}")
    print(f"Wrote compact metrics to {args.report_dir}")


class WhisperAsrDataset:
    def __init__(
        self,
        data_dir: Path,
        records: list[dict[str, Any]],
        processor: Any,
    ) -> None:
        self.data_dir = data_dir
        self.records = records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import soundfile as sf

        record = self.records[index]
        audio, sample_rate = sf.read(
            str(self.data_dir / str(record["audio"])),
            dtype="float32",
        )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).input_features[0]

        language = LANGUAGE_NAMES.get(str(record["language"]).lower(), "english")
        self.processor.tokenizer.set_prefix_tokens(
            language=language,
            task="transcribe",
        )
        labels = self.processor.tokenizer(str(record["transcript"])).input_ids
        return {"input_features": input_features, "labels": labels}


@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )
        batch["labels"] = labels
        return batch


def stratified_split(
    records: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("language", "unknown")).lower()].append(record)

    rng = random.Random(seed)
    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    for language in LANGUAGES:
        lang_records = list(grouped.get(language, []))
        rng.shuffle(lang_records)
        eval_count = max(1, int(len(lang_records) * eval_ratio))
        eval_records.extend(lang_records[:eval_count])
        train_records.extend(lang_records[eval_count:])
    rng.shuffle(train_records)
    rng.shuffle(eval_records)
    return train_records, eval_records


def cap_records(
    records: list[dict[str, Any]],
    max_records: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_records <= 0 or len(records) <= max_records:
        return records
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    return records[:max_records]


def summarize_languages(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    return {
        language: {
            "train": sum(
                str(record.get("language", "")).lower() == language
                for record in train_records
            ),
            "eval": sum(
                str(record.get("language", "")).lower() == language
                for record in eval_records
            ),
        }
        for language in LANGUAGES
    }


def render_metrics(metrics: dict[str, Any]) -> str:
    rows = [
        [language, values["train"], values["eval"]]
        for language, values in metrics["languages"].items()
    ]
    return "\n".join(
        [
            "# Whisper LoRA Training Metrics",
            "",
            f"- Base model: `{metrics['model_name']}`",
            f"- Adapter output: `{metrics['output_dir']}`",
            f"- Train samples: {metrics['train_samples']}",
            f"- Eval samples: {metrics['eval_samples']}",
            "",
            "## Language Split",
            "",
            markdown_table(["language", "train", "eval"], rows),
            "",
            "## Metrics",
            "",
            f"- Train: `{metrics['train_metrics']}`",
            f"- Eval: `{metrics['eval_metrics']}`",
        ]
    )


if __name__ == "__main__":
    main()
