FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV UPLOAD_FOLDER='uploads'
ENV EXTRACTED_IMAGES_FOLDER='extracted_images'

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install specific versions of Python packages
RUN pip3 install --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir pillow==9.5.0
RUN pip3 install --no-cache-dir detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html

# Install layoutparser and other dependencies
RUN pip3 install --no-cache-dir layoutparser

# Download the model during the build process
RUN python3 -c "import layoutparser as lp; lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config')"

# Set up the working directory
WORKDIR /app

# Copy your requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


# Copy the rest of your application
COPY . .

# Create necessary directories
RUN mkdir -p $UPLOAD_FOLDER $EXTRACTED_IMAGES_FOLDER

# Copy the .env file
COPY .env .env

# Set the entrypoint
CMD ["python3", "main.py"]
