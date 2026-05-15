# DSTA BrainHack TIL-AI 2026 — Advanced Track Strategy Overview

## Score Weights

| Challenge | Weight | Speed Weight |
|---|---|---|
| ASR | 20% | 25% of ASR score |
| CV | 20% | 25% of CV score |
| NLP | 20% | 25% of NLP score |
| AE | 40% | 25% of AE score |

Speed score formula: `1 - min(t_elapsed, t_max) / t_max` where `t_max = 30 minutes`

**AE is worth 2× the others — prioritize it.**

---

## Challenge Summary & Recommended Models

| Challenge | Model / Approach | Key Files |
|---|---|---|
| ASR | faster-whisper large-v3 (fp16) | `asr/src/asr_manager.py` |
| CV | YOLOv11l fine-tuned, LTWH output | `cv/src/cv_manager.py` |
| Noise | PGD on surrogate YOLO / UAP | `noise/src/noise_manager.py` |
| NLP | FAISS + bge-m3 + Qwen2.5-7B-Instruct | `nlp/src/nlp_manager.py` |
| AE | Rule-based → PPO on til_environment | `ae/src/ae_manager.py` |

---

## Advanced Track Specifics

| Challenge | Advanced Difference |
|---|---|
| ASR | 4 languages (EN, MS, TA, ZH); more noise |
| CV | Noisier images; smaller targets |
| NLP | L1–L5 questions including unanswerable (L4/L5) |
| AE | Variable map layout per match |

---

## Priority Order & Time Allocation

1. **AE (highest weight)** — Start rule-based immediately, run RL training in background
2. **ASR** — fastest to get working (Whisper drop-in); verify 4-language support
3. **NLP** — needs careful L4/L5 handling; embedding + LLM pipeline
4. **CV** — requires training on provided dataset; most time-intensive
5. **Noise** — lowest risk/reward; do after CV model is trained

---

## Critical Constraints

- 30-minute inference budget for full test set
- Docker containerized; no internet access at inference time → pre-download all models
- All models must be GPU-optimized (fp16, TensorRT, or quantized)
- AE: never call `/reset` — detect `step == 0` internally
- NLP: first request is corpus loading, subsequent are questions

---

## Detailed Plans

- [ASR Solution](asr_solution.md)
- [CV Solution](cv_solution.md)
- [Noise/Adversarial Solution](noise_solution.md)
- [NLP Solution](nlp_solution.md)
- [AE Solution](ae_solution.md)
