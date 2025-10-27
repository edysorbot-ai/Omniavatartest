# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Set environment variables to prevent Python buffering and ensure CUDA is available
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy the repository
COPY . /workspace/OmniAvatar/

# Set working directory to the project
WORKDIR /workspace/OmniAvatar

# Install PyTorch with CUDA support
RUN pip install --upgrade pip && \
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install requirements
RUN pip install -r requirements.txt

# Install flash attention for better performance (optional but recommended)
RUN pip install flash_attn --no-build-isolation

# Install Gradio if not in requirements
RUN pip install gradio

# Install huggingface-cli for model downloads
RUN pip install "huggingface_hub[cli]"

# Create directory for models
RUN mkdir -p /workspace/OmniAvatar/pretrained_models

# Download models at build time (optional - comment out if you want to download at runtime)
# This will make the image larger but container startup faster
# RUN huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./pretrained_models/Wan2.1-T2V-14B && \
#     huggingface-cli download OmniAvatar/OmniAvatar-14B --local-dir ./pretrained_models/OmniAvatar-14B && \
#     huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./pretrained_models/wav2vec2-base-960h

# For 1.3B model instead, use:
# RUN huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./pretrained_models/Wan2.1-T2V-1.3B && \
#     huggingface-cli download OmniAvatar/OmniAvatar-1.3B --local-dir ./pretrained_models/OmniAvatar-1.3B && \
#     huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./pretrained_models/wav2vec2-base-960h

# Expose Gradio default port
EXPOSE 7860

# Set environment variables for GPU optimization
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    CUDA_VISIBLE_DEVICES=0 \
    FORCE_CUDA=1

# Default command - adjust based on how the Gradio app is launched in the repository
# Common patterns:
# CMD ["python", "app.py"]
# CMD ["python", "demo.py"]
# CMD ["python", "scripts/gradio_demo.py"]
# CMD ["python", "-m", "gradio", "app.py"]

# Since we don't know the exact script name, using a generic command
# You should replace this with the actual command to run the Gradio interface
CMD ["python", "app.py"]