# Dockerfile for building the CV image.

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV CV_MODEL_PATH=/workspace/best.pt
WORKDIR /workspace

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless

# Copy trained model weights (must be placed in cv/ before building)
COPY best.pt /workspace/best.pt

# Copies your source files.
COPY src .

CMD uvicorn cv_server:app --port 5002 --host 0.0.0.0
