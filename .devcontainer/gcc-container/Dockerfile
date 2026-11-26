FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-24.04@sha256:a7cddc16b93b4ccfab0d33c805460a826703211ec30e3c04761aec17d71275cb

RUN apt-get --yes -qq update \
 && apt-get --yes -qq upgrade \
 && apt-get --yes -qq install build-essential \
                      curl git cmake gcc g++ \
                      python3-dev python3-numpy python3-matplotlib python3-pip \
                      libopenmpi-dev \
                      libhdf5-mpi-dev \
 && apt-get --yes -qq clean \
 && rm -rf /var/lib/apt/lists/*

# install Clang 20.x
RUN mkdir -m 0755 -p /etc/apt/keyrings/ && \
 curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /etc/apt/keyrings/llvm-snapshot.gpg.key && \
 echo "deb [signed-by=/etc/apt/keyrings/llvm-snapshot.gpg.key] http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main" | tee /etc/apt/sources.list.d/llvm.list > /dev/null

RUN apt-get clean && apt-get update -y && \
 apt-get install -y --no-install-recommends clang-20 llvm-20 libomp-20-dev libclang-rt-20-dev clangd-20 && \
 rm -rf /var/lib/apt/lists/*
 
# Create symlink for clangd
RUN ln -s /usr/bin/clangd-20 /usr/bin/clangd

# Install Viskores (volume rendering and isosurfaces)
RUN mkdir -p /tmp/build-viskores && cd /tmp/build-viskores && \
    wget -q https://github.com/Viskores/viskores/archive/refs/tags/v1.0.0.tar.gz && \
    tar xzf v1.0.0.tar.gz && \
    mkdir build && cd build && \
    cmake ../viskores-1.0.0 -DViskores_ENABLE_RENDERING=ON -DViskores_USE_64BIT_IDS=ON -DViskores_USE_DOUBLE_PRECISION=ON && \
    make -j$(nproc) && make install && \
    cd / && \
    rm -rf /tmp/build-viskores

WORKDIR /home/vscode
USER vscode

CMD [ "/bin/bash" ]
