# Testing the Solutions

This guide documents how to test each challenge solution step by step on the GCP Workbench instance, where the `til` CLI and training data are available.

## Prerequisites

1. Initialize the submodules:

```bash
git submodule update --init
```

2. Create and activate a Python environment:

```bash
conda create --name til-2026 python=3.10
conda activate til-2026
```

3. Install the local development dependencies:

```bash
pip install -r requirements-dev.txt
```

4. Create a `.env` file at the repo root with your team settings:

```dotenv
TEAM_NAME=your-team-name
TEAM_TRACK=advanced
```

Use `TEAM_TRACK=novice` only if you are intentionally testing the novice setup. The solution plans in `plans/` target the advanced track.

5. Confirm the training data exists under `/home/jupyter/$TEAM_TRACK/`.

## Quick Reference

| Challenge | Build | Full local test | Direct script | Port | Result |
|---|---|---|---|---|---|
| ASR | `til build asr` | `til test asr` | `python test/test_asr.py` | `5001` | `/home/jupyter/$TEAM_NAME/asr_results.json` |
| CV | `til build cv` | `til test cv` | `python test/test_cv.py` | `5002` | `/home/jupyter/$TEAM_NAME/cv_results.json` |
| Noise | `til build noise` | `til test noise` | `python test/test_noise.py` | `5003` | printed score only |
| NLP | `til build nlp` | `til test nlp` | `python test/test_nlp.py` | `5004` | `/home/jupyter/$TEAM_NAME/nlp_results.json` |
| AE | `til build ae` | `til test ae` | `python test/test_ae.py` | `5005` | printed total reward and score |

## Standard Testing Workflow

Use this workflow for any challenge:

1. Implement the logic in `<challenge>/src/*_manager.py`.
2. Build the container with `til build <challenge>`.
3. Run the full local evaluation with `til test <challenge>`.
4. If the full test fails, run the container manually and execute the matching `test/test_<challenge>.py` script to debug faster.
5. Inspect the printed metric and any saved results file.

The direct Python test scripts expect the model server to already be running on the correct port.

## Running a Server Manually

Use this when you want faster debug cycles than `til test`.

1. Build the image directly from the challenge directory:

```bash
cd asr
docker build -t your-team-asr:debug .
```

2. Start the container:

```bash
docker run --rm -p 5001:5001 your-team-asr:debug
```

3. In a second terminal, confirm the server is healthy:

```bash
curl http://localhost:5001/health
```

4. Run the matching test script from the repo root:

```bash
python test/test_asr.py
```

5. Stop the container, edit the manager, rebuild, and rerun.

Repeat the same pattern for the other challenges, changing the challenge name and port.

## ASR

Files involved:
- `asr/src/asr_manager.py`
- `asr/src/asr_server.py`
- `test/test_asr.py`

Steps:

1. Build the ASR container:

```bash
til build asr
```

2. Wait for the build to finish. The ASR image may take longer because model weights can be downloaded during the Docker build.

3. Run the local evaluation:

```bash
til test asr
```

4. Read the printed per-language error rates:
- English: WER
- Malay: WER
- Tamil: WER
- Chinese: CER

5. Read the final score printed as `1 - MER`.

6. Inspect the saved predictions:

```bash
cat /home/jupyter/$TEAM_NAME/asr_results.json
```

7. If you need to debug manually, run:

```bash
python test/test_asr.py
```

ASR-specific notes:
- The test script batches requests in groups of 4.
- Inputs are base64-encoded WAV files.
- The server must return one transcript string per input.

## CV

Files involved:
- `cv/src/cv_manager.py`
- `cv/src/cv_server.py`
- `test/test_cv.py`

Steps:

1. Build the CV container:

```bash
til build cv
```

2. Run the local evaluation:

```bash
til test cv
```

3. Read the printed `mAP@.5:.05:.95` score.

4. Inspect the saved detections:

```bash
cat /home/jupyter/$TEAM_NAME/cv_results.json
```

5. If needed, debug with the direct script:

```bash
python test/test_cv.py
```

CV-specific notes:
- The test script reads `/home/jupyter/$TEAM_TRACK/cv/annotations.json`.
- Predictions must be a list of detections for each input image.
- Each detection must use pixel coordinates in `[left, top, width, height]`.

## Noise

Files involved:
- `noise/src/noise_manager.py`
- `noise/src/noise_server.py`
- `test/test_noise.py`

Steps:

1. Build the noise container:

```bash
til build noise
```

2. Run the local evaluation:

```bash
til test noise
```

3. Read the printed `Noise Score (Pass Rate)`.

4. If needed, run the direct script:

```bash
python test/test_noise.py
```

Noise-specific notes:
- The noise test uses images from `/home/jupyter/$TEAM_TRACK/cv`.
- The direct test script limits itself to the first 500 images.
- The score is based on the fairness and image-quality evaluation pipeline in `test/noise_eval/`.
- The server must return one base64-encoded output image string per input image.

## NLP

Files involved:
- `nlp/src/nlp_manager.py`
- `nlp/src/nlp_server.py`
- `test/test_nlp.py`

Steps:

1. Build the NLP container:

```bash
til build nlp
```

2. Run the local evaluation:

```bash
til test nlp
```

3. Watch the startup logs carefully. The NLP test is two-phase:
- First it sends the full document corpus to `POST /nlp`.
- Then it polls the same endpoint until the server reports `loaded`.
- Only after that does it start the question-answer evaluation.

4. Read the printed `NLP RAG QA Accuracy` score.

5. Inspect the saved predictions:

```bash
cat /home/jupyter/$TEAM_NAME/nlp_results.json
```

6. If needed, debug with the direct script:

```bash
python test/test_nlp.py
```

NLP-specific notes:
- `load_corpus` in `NLPManager` should be synchronous and blocking.
- The server wraps corpus loading in `asyncio.to_thread` and exposes poll states.
- The direct test polls every 10 seconds for up to 30 retries.
- L4 and L5 scoring behavior is handled in `test/test_nlp.py`; test against those cases explicitly if your score looks unexpectedly low.

## AE

Files involved:
- `ae/src/ae_manager.py`
- `ae/src/ae_server.py`
- `test/test_ae.py`

Steps:

1. Build the AE container:

```bash
til build ae
```

2. Run the local evaluation:

```bash
til test ae
```

3. Read the printed `total rewards` and `score`.

4. If needed, debug with the direct script:

```bash
python test/test_ae.py
```

AE-specific notes:
- The test runs 6 rounds of the Bomberman environment.
- Your agent is always the agent at index 0.
- The other agents act randomly in the local test.
- The competition infrastructure does not call a reset endpoint.
- Your manager should detect a new round using `observation["step"] == 0` and reset internal state there.
- The current direct test script still sends a legacy empty `POST /ae` before each round; do not rely on that behavior for your real solution logic.

## What to Check When a Test Fails

1. Confirm `.env` has the correct `TEAM_NAME` and `TEAM_TRACK`.
2. Confirm the training data exists at `/home/jupyter/$TEAM_TRACK/<challenge>/`.
3. Confirm the container starts and `GET /health` returns `{"message": "health ok"}`.
4. Confirm your response format exactly matches the relevant challenge `README.md`.
5. Confirm all runtime dependencies are listed in `<challenge>/requirements.txt`, not just `requirements-dev.txt`.
6. Re-run the direct Python test script with the server running manually so you can inspect the failure faster.
