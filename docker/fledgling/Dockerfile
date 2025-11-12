# Dockerfile for Fledgling Pipeline (Current Approach)
# Multi-stage build for efficient caching and smaller final image

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first for better layer caching
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for reproducibility
RUN pip install --no-cache-dir \
    jsonschema \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyterlab

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Create necessary directories
RUN mkdir -p /workspace/slm_swap/02_dataset/{structured,toolcall} \
    /workspace/slm_swap/04_ft \
    /workspace/slm_swap/05_eval \
    /workspace/slm_swap/logs \
    /workspace/slm_swap/models

# Copy the entire codebase
COPY . /workspace/

# Set up entrypoint script
COPY docker/fledgling/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose Jupyter port (optional for debugging)
EXPOSE 8888

# Default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
