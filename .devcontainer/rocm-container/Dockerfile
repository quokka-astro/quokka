# Use the AMD ROCm image as the base image
FROM rocm/dev-ubuntu-24.04@sha256:4fbcf879d22aaca65b756c7c6c3586ee75090ae55953c4e44e7e88960fbb1e1c

# Set environment variables for CMake
ENV hip_DIR=/opt/rocm-6.4.0/lib/cmake/hip
ENV rocrand_DIR=/opt/rocm-6.4.0/lib/cmake/rocrand
ENV hiprand_DIR=/opt/rocm-6.4.0/lib/cmake/hiprand
ENV rocprim_DIR=/opt/rocm-6.4.0/lib/cmake/rocprim
ENV rocsparse_DIR=/opt/rocm-6.4.0/lib/cmake/rocsparse
ENV rocthrust_DIR=/opt/rocm-6.4.0/lib/cmake/rocthrust
ENV AMREX_AMD_ARCH=gfx90a
ENV CC=hipcc
ENV CXX=hipcc

# Update package lists and install necessary dependencies
RUN apt-get update -qq && apt-get upgrade -y -qq && \
    apt-get install -y -qq --no-install-recommends \
    rocrand-dev hiprand-dev rocprim-dev rocsparse-dev rocthrust-dev \
    build-essential wget curl git ninja-build gcc g++ \
    python3-dev python3-numpy python3-matplotlib python3-pip \
    libhdf5-dev libopenmpi-dev netbase && \
    apt-get --yes -qq clean && \
    rm -rf /var/lib/apt/lists/*

# Install Clang 19.x
RUN mkdir -m 0755 -p /etc/apt/keyrings/ && \
 curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /etc/apt/keyrings/llvm-snapshot.gpg.key && \
 echo "deb [signed-by=/etc/apt/keyrings/llvm-snapshot.gpg.key] http://apt.llvm.org/noble/ llvm-toolchain-noble-19 main" | tee /etc/apt/sources.list.d/llvm.list > /dev/null

RUN apt-get clean && apt-get update -y && \
 apt-get install -y --no-install-recommends clang-19 llvm-19 libomp-19-dev libclang-rt-19-dev clangd-19 && \
 rm -rf /var/lib/apt/lists/*

# Create symlink for clangd
RUN ln -s /usr/bin/clangd-19 /usr/bin/clangd

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

# Install Kokkos (needed for Viskores for ROCm)
RUN mkdir -p /tmp/build-kokkos && cd /tmp/build-kokkos && \
    wget -q https://github.com/kokkos/kokkos/releases/download/4.6.01/kokkos-4.6.01.tar.gz && \
    tar xzf kokkos-4.6.01.tar.gz && \
    mkdir build && cd build && \
    cmake ../kokkos-4.6.01 -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON && \
    make -j$(nproc) && make install && \
    cd / && \
    rm -rf /tmp/build-kokkos

# Install Viskores (volume rendering and isosurfaces)
# NOTE: to build this, you need a Docker VM with a LARGE amount of RAM (at least 15 GB RAM, 4 GB swap)

RUN mkdir -p /tmp/build-viskores && cd /tmp/build-viskores && \
    wget -q https://github.com/Viskores/viskores/archive/refs/tags/v1.0.0.tar.gz && \
    tar xzf v1.0.0.tar.gz && \
    mkdir build && cd build && \
    cmake ../viskores-1.0.0 -DViskores_ENABLE_RENDERING=ON -DViskores_USE_64BIT_IDS=ON -DViskores_USE_DOUBLE_PRECISION=ON -DViskores_ENABLE_KOKKOS=ON && \
    make -j$(nproc) && make install && \
    cd / && \
    rm -rf /tmp/build-viskores


# Set the working directory and user
WORKDIR /home/ubuntu
USER ubuntu

# Set the default command to bash
CMD [ "/bin/bash" ]
