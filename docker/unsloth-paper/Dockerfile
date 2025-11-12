# Dockerfile for FUNCTION_CALLING_FINETUNE Approach (Paper's Method)
# Based on official Unsloth Docker image with Hermes dataset support

FROM unsloth/unsloth:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Set working directory
WORKDIR /workspace

# Install additional dependencies
RUN pip install --no-cache-dir \
    jsonschema \
    python-dotenv \
    langfuse \
    openai \
    datasets \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn

# Create directory structure
RUN mkdir -p /workspace/paper_approach/datasets \
    /workspace/paper_approach/models \
    /workspace/paper_approach/adapters \
    /workspace/paper_approach/eval_results \
    /workspace/paper_approach/logs

# Copy paper-based training scripts
COPY docker/unsloth-paper/scripts/ /workspace/paper_approach/

# Set up entrypoint
COPY docker/unsloth-paper/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
