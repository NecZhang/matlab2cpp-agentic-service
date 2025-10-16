#!/bin/bash
# Build script for C++ testing Docker image

set -e

echo "🐳 Building C++ Testing Docker Image..."

# Build the Docker image
docker build -t matlab2cpp/testing:latest ./docker/testing/

echo "✅ Docker image built successfully!"
echo "Image name: matlab2cpp/testing:latest"

# Test the image
echo "🧪 Testing Docker image..."
docker run --rm matlab2cpp/testing:latest g++ --version
docker run --rm matlab2cpp/testing:latest pkg-config --cflags eigen3

echo "✅ Docker image test completed!"








