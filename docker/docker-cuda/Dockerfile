# Default use the NVIDIA official image with PyTorch 2.6.0
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html
#
# 24.12 is a CUDA 12.6 release: right for H100 (sm_90), and it has no kernels at
# all for Blackwell. For a B300 (sm_103) build, pass an NGC release whose CUDA is
# 12.9 or newer, and set TORCH_CUDA_ARCH_LIST so anything compiled here (flash
# attention, deepspeed ops) targets this arch instead of the builder's GPU:
#
#   docker build \
#     --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3 \
#     --build-arg CUDA_ARCH_LIST=10.3a \
#     -f docker/docker-cuda/Dockerfile .
#
# `bash scripts/gpu_profile.sh --print` on the target node prints both values.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FROM ${BASE_IMAGE}

# Compute capabilities to generate code for. Hopper by default; 10.3a for B300,
# 10.0a for B200. The `a` suffix is required by the FP8/NVFP4 CUTLASS kernels and
# is arch-exact: code built as sm_100a does NOT run on a B300.
ARG CUDA_ARCH_LIST=9.0
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}

# Define environments
ENV MAX_JOBS=4
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Define installation arguments
ARG INSTALL_BNB=false
ARG INSTALL_VLLM=false
ARG INSTALL_DEEPSPEED=false
ARG INSTALL_FLASHATTN=false
ARG INSTALL_LIGER_KERNEL=false
ARG INSTALL_HQQ=false
ARG INSTALL_EETQ=false
ARG PIP_INDEX=https://pypi.org/simple
ARG HTTP_PROXY=

# Set the working directory
WORKDIR /app

# Set http proxy
RUN if [ -n "$HTTP_PROXY" ]; then \
        echo "Configuring proxy..."; \
        export http_proxy=$HTTP_PROXY; \
        export https_proxy=$HTTP_PROXY; \
    fi

# Install the requirements
COPY requirements.txt /app
RUN pip config set global.index-url "$PIP_INDEX" && \
    pip config set global.extra-index-url "$PIP_INDEX" && \
    python -m pip install --upgrade pip && \
    if [ -n "$HTTP_PROXY" ]; then \
        python -m pip install --proxy=$HTTP_PROXY -r requirements.txt; \
    else \
        python -m pip install -r requirements.txt; \
    fi

# Copy the rest of the application into the image
COPY . /app

# Install the LLaMA Factory
RUN EXTRA_PACKAGES="metrics"; \
    if [ "$INSTALL_BNB" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},bitsandbytes"; \
    fi; \
    if [ "$INSTALL_VLLM" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},vllm"; \
    fi; \
    if [ "$INSTALL_DEEPSPEED" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},deepspeed"; \
    fi; \
    if [ "$INSTALL_LIGER_KERNEL" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},liger-kernel"; \
    fi; \
    if [ "$INSTALL_HQQ" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},hqq"; \
    fi; \
    if [ "$INSTALL_EETQ" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},eetq"; \
    fi; \
    if [ -n "$HTTP_PROXY" ]; then \
        pip install --proxy=$HTTP_PROXY -e ".[$EXTRA_PACKAGES]"; \
    else \
        pip install -e ".[$EXTRA_PACKAGES]"; \
    fi

# Rebuild flash attention
RUN pip uninstall -y transformer-engine flash-attn && \
    if [ "$INSTALL_FLASHATTN" == "true" ]; then \
        pip uninstall -y ninja && \
        if [ -n "$HTTP_PROXY" ]; then \
            pip install --proxy=$HTTP_PROXY ninja && \
            pip install --proxy=$HTTP_PROXY --no-cache-dir flash-attn --no-build-isolation; \
        else \
            pip install ninja && \
            pip install --no-cache-dir flash-attn --no-build-isolation; \
        fi; \
    fi


# Unset http proxy
RUN if [ -n "$HTTP_PROXY" ]; then \
        unset http_proxy; \
        unset https_proxy; \
    fi

# Set up volumes
VOLUME [ "/root/.cache/huggingface", "/root/.cache/modelscope", "/app/data", "/app/output" ]

# Expose port 7860 for the LLaMA Board
ENV GRADIO_SERVER_PORT 7860
EXPOSE 7860

# Expose port 8000 for the API service
ENV API_PORT 8000
EXPOSE 8000
