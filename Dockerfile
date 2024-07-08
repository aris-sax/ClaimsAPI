FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Clone and install Detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git /detectron2_repo
WORKDIR /detectron2_repo
RUN pip3 install -e .

# Set up the working directory
WORKDIR /app

# Copy your requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . .

# Set the entrypoint
CMD ["python3", "main.py"]
