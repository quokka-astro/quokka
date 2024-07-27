FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-24.04

RUN apt-get --yes -qq update \
 && apt-get --yes -qq upgrade \
 && apt-get --yes -qq install build-essential \
                      git cmake clangd gcc g++ \
                      python3-dev python3-numpy python3-matplotlib python3-pip \
                      python3-sphinxcontrib.bibtex sphinx-doc \
                      libopenmpi-dev \
                      libhdf5-mpi-dev \
 && apt-get --yes -qq clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /home/ubuntu
USER ubuntu

# install esbonio for Sphinx VSCode support (no Ubuntu package for this)
RUN pip install esbonio --break-system-packages

# workaround Python babel bug
ENV TZ=UTC

CMD [ "/bin/bash" ]