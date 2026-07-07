# Training + eval environment. GPU access: run with `docker run --gpus all`
# (NVIDIA container runtime). The agent CLI must also be present — install the
# one you need below, or use MODE=native on a host that already has it.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl git ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Python deps, resolved from the project lockfile.
WORKDIR /opt/autoembed
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

# Install the agent CLI you intend to run, e.g.:
#   RUN npm i -g @anthropic-ai/claude-code     # claude
#   RUN npm i -g @openai/codex                 # codex
#   RUN npm i -g @google/gemini-cli            # gemini

WORKDIR /work
