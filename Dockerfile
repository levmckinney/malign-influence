# syntax=docker/dockerfile:1.9

# Multi-stage Dockerfile following uv best practices
# - Builder uses uv to resolve/install into a project virtualenv
# - Runtime is a minimal python:slim image containing only the venv and code
# - Devbox target adds OpenSSH for remote development (used by Kubernetes manifest)

############################
# Builder: install deps with uv
############################
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel AS builder

# Install uv
RUN pip install uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Some dependencies may be sourced from VCS; ensure git is available
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for optimal caching
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,readonly \
    --mount=type=bind,source=uv.lock,target=uv.lock,readonly \
    uv sync --frozen --no-install-project --no-dev

# Then add the rest of the source and install the project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


############################
# Devbox: SSH-enabled environment (used by Kubernetes dev box)
############################
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel AS devbox

# Install uv
RUN pip install uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server sudo git vim tmux less htop curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

# Create user `dev` with passwordless sudo
RUN useradd -m -u 1000 -s /bin/bash dev \
    && echo 'dev ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/dev \
    && chmod 0440 /etc/sudoers.d/dev

# SSH configuration: key-only auth, no root login
RUN sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin no/' /etc/ssh/sshd_config \
    && sed -i 's@^#\?AuthorizedKeysFile .*@AuthorizedKeysFile /home/dev/.ssh/authorized_keys@' /etc/ssh/sshd_config

USER dev
WORKDIR /workspace
