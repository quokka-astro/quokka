#!/bin/bash

set -e

QK_DIR="${1:-$HOME/quokka}"
if [[ $# -gt 0 ]]; then
    shift
fi

IMAGE_NAME="ghcr.io/quokka-astro/quokka-arm64-cuda:development"
CONTAINER_NAME="quokka-arm64-cuda-container"
build_dir="sims/regular-tests/builds/container/build-3D-cuda"

cd $QK_DIR/sims/regular-tests || exit 1

jobs=""
use_log=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -log)
            use_log=true
            shift
            ;;
        *)
            if [ -z "$jobs" ]; then
                jobs="$1"
            fi
            shift
            ;;
    esac
done


is_reconfig=false
# Check if $QK_DIR/$build_dir exists and remove it with confirmation
if [ -d "$QK_DIR/$build_dir" ]; then
    read -p "Directory $QK_DIR/$build_dir exists. Do you want to continue with the build (y) or remove it and do a rebuild (r)? [y/r] " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Will build upon existing build..."
    elif [[ "$response" =~ ^[Rr]$ ]]; then
        echo "Removing directory $QK_DIR/$build_dir..."
        rm -rf "$QK_DIR/$build_dir"
        is_reconfig=true
    else
        echo "Aborting..."
				exit 0
    fi
else
		is_reconfig=true
    echo "Directory $QK_DIR/$build_dir does not exist. Creating it..."
    mkdir -p "$QK_DIR/$build_dir"
fi

# Check if the Docker image exists
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}$"; then
    echo "Image $IMAGE_NAME not found locally. Attempting to pull..."
    if ! docker pull $IMAGE_NAME; then
        echo "Error: Failed to pull image $IMAGE_NAME. Please check if the image exists and you have access to it."
        exit 1
    fi
fi

# Check if the container exists and is running
if [ "$(docker ps -a -q -f name=^${CONTAINER_NAME}$)" ]; then
    # If the container is stopped, start it
    if [ "$(docker ps -q -f name=^${CONTAINER_NAME}$)" ]; then
        echo "Container $CONTAINER_NAME is already running."
    else
        echo "Starting container $CONTAINER_NAME..."
        docker start $CONTAINER_NAME
    fi
else
    echo "Container $CONTAINER_NAME does not exist. Creating and starting it from image $IMAGE_NAME..."
    docker run -d --name $CONTAINER_NAME \
        -v "$QK_DIR:/home/ubuntu/workspace" \
        --workdir /home/ubuntu/workspace \
        $IMAGE_NAME \
        tail -f /dev/null
fi

# build the test
if [ "$use_log" = true ]; then
    log_dir_rel="sims/regular-tests/test-logs/$(date +'%Y%m%d-%H%M%S')-cuda"
    log_dir="$QK_DIR/$log_dir_rel"
    if [ ! -d $log_dir ]; then
        mkdir -p $log_dir
    fi

    # Save git info
    git status >> $log_dir/git.log
    git log -3 >> $log_dir/git.log

    # Redirect output to build.log
    if [ "$is_reconfig" = true ]; then
        docker exec $CONTAINER_NAME bash -c "mkdir -p /home/ubuntu/workspace/$build_dir && cd /home/ubuntu/workspace/$build_dir && cmake /home/ubuntu/workspace -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_CXX_FLAGS='-Wno-deprecated-gpu-targets' -DAMReX_GPU_BACKEND=CUDA -DAMReX_MPI=OFF -DAMReX_SPACEDIM=3 -G Ninja && ninja -j8 $jobs" &> $log_dir/build.log
    else
        docker exec $CONTAINER_NAME bash -c "cd /home/ubuntu/workspace/$build_dir && ninja -j8 $jobs" &> $log_dir/build.log
    fi
else
    # Output to terminal
    if [ "$is_reconfig" = true ]; then
        docker exec $CONTAINER_NAME bash -c "mkdir -p /home/ubuntu/workspace/$build_dir && cd /home/ubuntu/workspace/$build_dir && cmake /home/ubuntu/workspace -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_CXX_FLAGS='-Wno-deprecated-gpu-targets' -DAMReX_GPU_BACKEND=CUDA -DAMReX_MPI=OFF -DAMReX_SPACEDIM=3 -G Ninja && ninja -j8 $jobs"
    else
        docker exec $CONTAINER_NAME bash -c "cd /home/ubuntu/workspace/$build_dir && ninja -j8 $jobs"
    fi
fi
