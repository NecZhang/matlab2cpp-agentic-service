# Dockerfile for C++ Testing Environment
# This container provides all dependencies needed for C++ compilation and testing
FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and C++ compilation tools
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    g++-13 \
    gcc-13 \
    cmake \
    make \
    pkg-config \
    git \
    wget \
    curl \
    \
    # C++ libraries and dependencies
    libeigen3-dev \
    libopencv-dev \
    libopencv-contrib-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libomp-dev \
    libgomp1 \
    \
    # Python and testing tools
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    \
    # Additional utilities
    valgrind \
    gdb \
    strace \
    \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Skip pip upgrade due to network timeout issues
# The system pip (24.0) is sufficient for our needs
# RUN python3 -m pip install --upgrade pip --break-system-packages

# Create testing directories
RUN mkdir -p /testing/projects /testing/results /testing/temp /testing/scripts

# Set working directory
WORKDIR /testing

# Copy Python requirements (will be created)
COPY requirements.txt /testing/
RUN pip3 install -r requirements.txt --break-system-packages

# Set environment variables for C++ compilation
ENV CXX=g++-13
ENV CC=gcc-13
ENV CXXFLAGS="-std=c++17 -O2 -Wall -Wextra"
ENV LDFLAGS="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -leigen3 -lgomp"

# Default command
CMD ["/bin/bash"]
