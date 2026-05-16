# Dockerfile for building the NLP image.

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HOME=/workspace/.cache/huggingface
ENV NLP_EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV NLP_RERANKER_MODEL=BAAI/bge-reranker-base
ENV NLP_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
WORKDIR /workspace

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models into the image (no internet at evaluation time)
COPY download_model.py .
RUN python download_model.py

# Block all network access at runtime — all models must be in the cache
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copies your source files.
COPY src .

CMD uvicorn nlp_server:app --port 5004 --host 0.0.0.0
