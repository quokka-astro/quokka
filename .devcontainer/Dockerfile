FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-24.04

RUN apt-get --yes -qq update \
 && apt-get --yes -qq upgrade \
 && apt-get --yes -qq install build-essential \
                      git cmake clangd gcc g++ \
                      python3-dev python3-numpy python3-matplotlib python3-pip \
                      sphinx-doc python3-sphinx-rtd-theme python3-sphinxcontrib.bibtex python3-sphinx-copybutton \
                      libopenmpi-dev \
                      libhdf5-mpi-dev \
                      paraview \
 && apt-get --yes -qq clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /home/ubuntu
ENV CMAKE_PREFIX_PATH="/opt/conduit:/opt/catalyst"

# build Conduit
RUN git clone --recursive https://github.com/llnl/conduit.git \
 && cd conduit \
 && mkdir -p /opt/conduit \
 && cmake -S src -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_MPI=ON -DCMAKE_INSTALL_PREFIX=/opt/conduit \
 && cmake --build build --parallel 4 \
 && cmake --install build

# build Catalyst
RUN git clone https://gitlab.kitware.com/paraview/catalyst.git \
 && cd catalyst \
 && mkdir -p /opt/catalyst \
 && cmake -B build -S . -DCATALYST_WITH_EXTERNAL_CONDUIT=ON -DCMAKE_INSTALL_PREFIX=/opt/catalyst \
 && cmake --build build --parallel 4 \
 && cmake --install build

USER ubuntu

# install esbonio for Sphinx VSCode support (no Ubuntu package for this)
RUN pip install esbonio --break-system-packages

# workaround Python babel bug
ENV TZ=UTC

CMD [ "/bin/bash" ]