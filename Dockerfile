FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.cargo/bin:$PATH"

# Use bash to preserve PATH changes across RUN commands
SHELL ["/bin/bash", "-c"]

# Install system packages
RUN apt update && apt install -y \
    python3.10 python3-pip python3.10-venv git curl ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && python3 -m pip install --upgrade pip \
    && apt clean

# Install uv (Python package manager from Astral)
RUN curl -Ls https://astral.sh/uv/install.sh | bash && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc && \
    source ~/.bashrc && \
    uv --version

# Install torch nightly (CUDA 12.1)
RUN uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# Install Triton 3.4.0
RUN uv pip install triton==3.4.0

# Install vLLM GPT-OSS build using uv
RUN uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu121 \
    --index-strategy unsafe-best-match

# Default command
CMD ["python3"]
