# syntax=docker/dockerfile:1.9

# Multi-stage Dockerfile following uv best practices
# - Builder uses uv to resolve/install into a project virtualenv
# - Runtime is a minimal python:slim image containing only the venv and code
# - Devbox target adds OpenSSH for remote development (used by Kubernetes manifest)

############################
# Builder: install deps with uv
############################
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

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
# Minimal runtime
############################
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

# Create non-root user
RUN groupadd -r app && useradd -r -d /app -g app app

WORKDIR /app

# Copy app with correct ownership
COPY --from=builder --chown=app:app /app /app

# Ensure the venv executables are on PATH
ENV PATH="/app/.venv/bin:$PATH"

USER app

# Default command kept generic; override in your orchestrator or `docker run`
CMD ["/bin/bash"]


############################
# Devbox: SSH-enabled environment (used by Kubernetes dev box)
############################
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS devbox

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server sudo git vim tmux less htop curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

# Ensure uv binaries are on PATH for all users
RUN ln -sf /uv /usr/local/bin/uv || true \
    && ln -sf /uvx /usr/local/bin/uvx || true \
    && command -v uv

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

# The Kubernetes manifest mounts authorized_keys at /home/dev/.ssh/authorized_keys
# and a writable PVC at /workspace. Default to running sshd in foreground.
USER root
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e"]