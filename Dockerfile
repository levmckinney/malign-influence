# Alternative Dockerfile with Python 3.12 support
# Uses NVIDIA's newer image with CUDA 12.6+ and Python 3.12
# Use this if CUDA 12.6 is acceptable (instead of 12.4)

# Use NVIDIA's official PyTorch image with Python 3.12
# Version 24.12 includes PyTorch 2.6, CUDA 12.6.3, and Python 3.12
FROM nvcr.io/nvidia/pytorch:24.12-py3 AS base

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock README.md ./

# Copy source code
COPY src/ ./src/

# Install UV package manager for better Python dependency management
RUN pip install uv

# Install project dependencies using UV
RUN uv pip install --system -r pyproject.toml

# Install the package in editable mode for development
RUN uv pip install --system -e .

# Development stage - includes additional dev tools and doesn't install the package
FROM nvcr.io/nvidia/pytorch:24.12-py3 AS dev

# Set working directory
WORKDIR /workspace

# Install system dependencies and development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    htop \
    tmux \
    less \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install uv

# Copy only dependency files first for better caching
COPY pyproject.toml uv.lock README.md ./

# Install project dependencies
RUN uv pip install --system -r pyproject.toml

# Set up a nice prompt for development
RUN echo 'export PS1="\[\e[32m\]malign-influence-dev\[\e[m\]:\[\e[34m\]\w\[\e[m\]$ "' >> ~/.bashrc

# Default command for dev container
CMD ["/bin/bash"]