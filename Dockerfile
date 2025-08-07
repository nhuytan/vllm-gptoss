FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system packages
RUN apt update && apt install -y \
    python3.10 python3-pip python3.10-venv git curl ca-certificates \
    && ln -sf python3.10 /usr/bin/python3 \
    && pip3 install --upgrade pip \
    && apt clean

# Install torch nightly (CUDA 12.1)
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# Install Triton ≥ 3.4.0 and triton_kernels
RUN pip install triton==3.4.0

# Install vLLM GPT-OSS version (note: --pre required)
RUN pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --index-url https://download.pytorch.org/whl/nightly/cu121 \
    --index-strategy unsafe-best-match

# Default command
CMD ["vllm", "serve", "openai/gpt-oss-20b"]
