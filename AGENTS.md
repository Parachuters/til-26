# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sources of Truth

Use these references as the canonical context for challenge rules, submission requirements, and implementation details:

- Project submission: https://tribegroup.notion.site/Project-Submission-33a5263ef45a80c3bad7d6006752cba4#33a5263ef45a81c9a4bce1c6c562a050
- BrainHack 2026 TIL AI Strategist's Handbook: https://tribegroup.notion.site/BrainHack-2026-TIL-AI-Strategist-s-Handbook-33a5263ef45a80429a9dc47c569e40c3
- TIL 2026 wiki: https://github.com/til-ai/til-26/wiki
- TIL 2026 repository: https://github.com/til-ai/til-26

## Commands

All commands below run on the GCP Workbench instance where the `til` CLI is pre-installed and training data lives at `/home/jupyter/{TEAM_TRACK}/`.

```bash
# Build a challenge container
til build asr
til build cv
til build noise
til build nlp
til build ae

# Test locally against training data (requires running container)
til test asr
til test cv
til test noise
til test nlp
til test ae

# Submit for evaluation
til submit asr
til submit nlp

# Run a single test script directly (container must already be running on correct port)
python test/test_asr.py
python test/test_cv.py
python test/test_nlp.py
python test/test_ae.py
python test/test_noise.py

# Initialize submodules after cloning
git submodule update --init

# Install dev dependencies
pip install -r requirements-dev.txt
```

Environment variables (`TEAM_NAME`, `TEAM_TRACK`) are loaded from a `.env` file at repo root. Test scripts expect data at `/home/jupyter/$TEAM_TRACK/<challenge>/`.

---

## Architecture

### Manager + Server Pattern

Each challenge follows the same two-file structure inside `<challenge>/src/`:

- **`*_manager.py`** — the only file you implement. Contains a single class (`ASRManager`, `CVManager`, etc.) instantiated once at startup. All inference logic lives here.
- **`*_server.py`** — FastAPI server; do not modify. It decodes base64 inputs, calls the manager, and returns predictions.

The server files handle base64 decode/encode and HTTP routing. Managers receive raw `bytes` (audio/images) or plain Python types (dicts, strings) and return Python types directly.

### Port Assignments

| Challenge | Port | Route |
|---|---|---|
| ASR | 5001 | `POST /asr` |
| CV | 5002 | `POST /cv` |
| Noise | 5003 | `POST /noise` |
| NLP | 5004 | `POST /nlp` |
| AE | 5005 | `POST /ae` |

All servers also expose `GET /health`.

### NLP Server — Async Corpus Loading

The NLP server has a special two-phase protocol. The first POST to `/nlp` contains a `"documents"` key and kicks off an async background load via `asyncio.to_thread`. Subsequent POSTs with `"poll": "true"` return the current load status (`"idle" | "loading" | "loaded" | "failed"`). Only after status is `"loaded"` will question POSTs (containing `"question"` keys) be processed. Implement `load_corpus` to be synchronous and blocking — the server handles the async wrapping.

### AE Environment

The `til-26-ae` submodule installs the `til_environment` package (a PettingZoo multi-agent environment). The test script (`test/test_ae.py`) runs 6 rounds of the game locally using `bomberman_env.basic_env`, with your agent at index 0 and random actions for other agents.

`/reset` is **not called** by the competition infrastructure. Detect round boundaries by checking `observation["step"] == 0` inside `AEManager.ae()`.

### Viewcone Shape Reference

- `agent_viewcone`: `[7, 5, 25]` — 7 cells deep, 5 cells wide, 25 feature channels, oriented to agent's facing direction
- `base_viewcone`: `[5, 5, 25]` — square view centred on the team's base

### Scoring Details

**ASR:** Chinese uses CER (character-level); all other languages use WER (word-level). The exact transforms applied before scoring are in `test/test_asr.py`: lowercase, dash→space substitution, remove punctuation, strip. Score = `max(0, 1 - mean_error_rate)` averaged over all 4 languages equally.

**NLP:** Scoring is in `test/test_nlp.py` via `AnswerEquivalenceEvaluator` (ModernBERT-based, threshold=0.9, max_length=512). Special cases:
- **L4** (no relevant docs exist): both `documents` and `answer` must be empty → score 1.0
- **L5** (false premise): needs at least 1 doc ID overlapping ground truth **and** empty `answer` → score 1.0; overlap without empty answer → 0.4
- Retrieval success (≥1 doc overlap) but wrong answer → 0.4 partial credit
- No doc overlap → 0.0

**AE:** `score = total_rewards / NUM_ROUNDS / MAX_SCORE` where `NUM_ROUNDS=6`, `MAX_SCORE=1000`.

### Dockerfiles

ASR/CV/NLP/Noise use `nvcr.io/nvidia/pytorch:25.11-py3` (GPU). AE uses `python:3.11-slim` (CPU-only, faster evaluation). Pre-download model weights inside the Dockerfile to avoid cold-start penalties — the container has no internet access during evaluation.

### Git Submodules

- `til-26-ae/` — `til_environment` package for AE training and testing. Installed via `-e ./til-26-ae` in `requirements-dev.txt`.
- `til-26-finals/` — pulled in for Semifinals/Finals. Do not modify either submodule.

### Advanced Track Specifics

- **ASR:** 4 languages (English, Malay, Tamil, Chinese) in equal proportions, more noise
- **CV:** Noisier images, smaller targets; output bbox in `[left, top, width, height]` pixel format (not normalized YOLO center format)
- **NLP:** Questions span L1–L5 including L4 (unanswerable, no source) and L5 (unanswerable, false premise)
- **AE:** Variable map layout per match (cannot memorize layout)
