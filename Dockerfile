FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

SHELL ["/bin/bash", "-c"]

# Install system packages
RUN apt update && apt install -y \
    python3.10 python3-pip python3.10-venv git curl ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && python3 -m pip install --upgrade pip \
    && apt clean

# Install uv (Python package manager from Astral)
RUN curl -Ls https://astral.sh/uv/install.sh | bash
RUN uv --version

# Install torch nightly (CUDA 12.1)
RUN uv pip install --system --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# Install Triton 3.4.0
RUN uv pip install --system triton==3.4.0

# Install vLLM GPT-OSS build
RUN uv pip install --system --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu121 \
    --index-strategy unsafe-best-match

CMD ["python3"]
