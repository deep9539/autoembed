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
RUN case "$AGENT_CLI" in \
      claude) npm install -g @anthropic-ai/claude-code ;; \
      codex) npm install -g @openai/codex ;; \
      antigravity) mkdir -p /tmp/agy-install \
        && curl -fsSL https://antigravity.google/cli/install.sh \
        | HOME=/tmp/agy-install bash -s -- --dir /usr/local/bin ;; \
      none) true ;; \
      *) echo "unknown AGENT_CLI=$AGENT_CLI" >&2; exit 2 ;; \
    esac

WORKDIR /work
