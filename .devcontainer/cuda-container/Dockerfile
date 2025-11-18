# Use the NVIDIA CUDA image as the base image
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c

# Set environment variables for NVIDIA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Update package lists and install necessary dependencies
RUN apt-get update -qq && apt-get upgrade -y -qq && \
    apt-get install -y -qq --no-install-recommends \
    build-essential wget curl git ninja-build gcc g++ \
    python3-dev python3-numpy python3-matplotlib python3-pip \
    libhdf5-dev openmpi-bin libopenmpi-dev && \
    apt-get --yes -qq clean && \
    rm -rf /var/lib/apt/lists/*

# Install Clang 20.x
RUN mkdir -m 0755 -p /etc/apt/keyrings/ && \
 curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /etc/apt/keyrings/llvm-snapshot.gpg.key && \
 echo "deb [signed-by=/etc/apt/keyrings/llvm-snapshot.gpg.key] http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main" | tee /etc/apt/sources.list.d/llvm.list > /dev/null

RUN apt-get clean && apt-get update -y && \
 apt-get install -y --no-install-recommends clang-20 llvm-20 libomp-20-dev libclang-rt-20-dev clangd-20 && \
 rm -rf /var/lib/apt/lists/*

# Create symlink for clangd
RUN ln -s /usr/bin/clangd-20 /usr/bin/clangd

# Install the latest version of CMake
RUN test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null

RUN apt-get clean && apt-get update -y && \
    apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/*

# Install blosc2 (needed for ADIOS2)
RUN pip3 install blosc2 --break-system-packages

# Install ADIOS2 (needed for OpenPMD)
RUN mkdir -p /tmp/build-adios2 && cd /tmp/build-adios2 && \
    wget -q https://github.com/ornladios/ADIOS2/archive/refs/tags/v2.10.1.tar.gz && \
    tar xzf v2.10.1.tar.gz && \
    mkdir adios2-build && cd adios2-build && \
    cmake ../ADIOS2-2.10.1 -DADIOS2_USE_Blosc2=ON -DADIOS2_USE_Fortran=OFF && \
    make -j$(nproc) && make install && \
    cd / && \
    rm -rf /tmp/build-adios2

# Install Viskores (volume rendering and isosurfaces)
# NOTE: to build this, you need a Docker VM with a LARGE amount of RAM (at least 15 GB RAM, 4 GB swap)

RUN mkdir -p /tmp/build-viskores && cd /tmp/build-viskores && \
    wget -q https://github.com/Viskores/viskores/archive/refs/tags/v1.0.0.tar.gz && \
    tar xzf v1.0.0.tar.gz && \
    mkdir build && cd build && \
    cmake ../viskores-1.0.0 -DViskores_ENABLE_RENDERING=ON -DViskores_USE_64BIT_IDS=ON -DViskores_USE_DOUBLE_PRECISION=ON -DViskores_ENABLE_CUDA=ON -DViskores_CUDA_Architecture=70 && \
    make -j$(nproc) && make install && \
    cd / && \
    rm -rf /tmp/build-viskores

# Set the working directory and user
WORKDIR /home/ubuntu
USER ubuntu

# Set the default command to bash
CMD [ "/bin/bash" ]
