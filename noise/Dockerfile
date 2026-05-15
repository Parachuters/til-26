# Dockerfile for building the noise image.

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV NOISE_MODEL_PATH=/workspace/best.pt
WORKDIR /workspace

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trained CV model weights as the surrogate for adversarial attacks
COPY best.pt /workspace/best.pt

# Copies your source files.
COPY src .

CMD uvicorn noise_server:app --port 5003 --host 0.0.0.0
