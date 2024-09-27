# Use the NVIDIA CUDA image as the base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables for NVIDIA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Update package lists and install necessary dependencies
RUN apt-get update -qq && apt-get upgrade -y -qq && \
    apt-get install -y -qq --no-install-recommends \
        build-essential \
	wget \
        git \
        cmake \
        gcc \
        g++ \
        libopenmpi-dev \
        libhdf5-mpi-dev \
        python3-defaults \
	ninja-build && \
    apt-get --yes -qq clean && \
    rm -rf /var/lib/apt/lists/*

# Install Clang 19.x
RUN wget -qO - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main" > /etc/apt/sources.list.d/llvm.list && \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        clang-19 \
        llvm-19 \
        libomp-19-dev \
        libclang-rt-19-dev \
        clangd-19 && \
    rm -rf /var/lib/apt/lists/*

# Install blosc2 (needed for ADIOS2)
# Commented out because python is not installed
# RUN pip3 install blosc2 --break-system-packages

# Install ADIOS2 (needed for OpenPMD)
RUN mkdir -p /tmp/build-adios2 && cd /tmp/build-adios2 && \
    wget -q https://github.com/ornladios/ADIOS2/archive/refs/tags/v2.10.1.tar.gz && \
    tar xzf v2.10.1.tar.gz && \
    mkdir adios2-build && cd adios2-build && \
    cmake ../ADIOS2-2.10.1 -DADIOS2_USE_Blosc2=ON -DADIOS2_USE_Fortran=OFF && \
    make -j$(nproc) && make install && \
    cd / && \
    rm -rf /tmp/build-adios2

# Create a new user 'ubuntu' and set the working directory
RUN useradd -ms /bin/bash ubuntu

# Set the working directory and user
WORKDIR /home/ubuntu
USER ubuntu

# Set the default command to bash
CMD [ "/bin/bash" ]
