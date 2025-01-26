# Dockerfile

# Use an Ubuntu base image with CUDA (if you want GPU support)
ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0
FROM docker.io/nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder

# Disable interactive prompts
ENV DEBIAN_FRONTEND=interactive
# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

RUN apt-get install -y \
    nvidia-container-toolkit \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc \
    nvidia-driver-535 \
    vim 

# Clone and build COLMAP from source
WORKDIR /home/runyi_yang/SGSLAM/

RUN git clone https://github.com/colmap/colmap.git -b release/3.11 && \
    cd colmap && \
    mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=86 && \
    ninja && \
    ninja install

# Expose a working directory for data
RUN mkdir /data
WORKDIR /data

