# Training + eval environment. For enforced single-GPU access, run with
# `docker run --gpus device=<physical-index>`; run_task.sh does this in MODE=docker.
# (NVIDIA container runtime). Build one immutable image per agent CLI.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Python deps, resolved from the project lockfile.
WORKDIR /opt/autoembed
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

ARG AGENT_CLI=claude
ARG CLAUDE_CLI_VERSION=2.1.218
ARG CODEX_CLI_VERSION=0.145.0
ARG GEMINI_CLI_VERSION=0.53.0
RUN case "$AGENT_CLI" in \
      claude) npm install -g "@anthropic-ai/claude-code@$CLAUDE_CLI_VERSION" ;; \
      codex) npm install -g "@openai/codex@$CODEX_CLI_VERSION" ;; \
      gemini) npm install -g "@google/gemini-cli@$GEMINI_CLI_VERSION" ;; \
      none) true ;; \
      *) echo "unknown AGENT_CLI=$AGENT_CLI" >&2; exit 2 ;; \
    esac

WORKDIR /work
