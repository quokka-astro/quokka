FROM registry.opensuse.org/opensuse/leap:15.6

RUN zypper --non-interactive refresh && \
    zypper --non-interactive update && \
    zypper clean --all

COPY rocm_so.conf rocm_so.conf

RUN if ! zypper repos -E | grep devel_languages_perl; then \
    zypper --non-interactive addrepo https://download.opensuse.org/repositories/devel:/languages:/perl/15.6/devel:languages:perl.repo; \
    fi && \
    echo "[ROCm-7.1.0]" | tee --append /etc/zypp/repos.d/rocm.repo && \
    echo "name=ROCm7.1.0" | tee --append /etc/zypp/repos.d/rocm.repo && \
    echo "baseurl=https://repo.radeon.com/rocm/zyp/7.1/main" | tee --append /etc/zypp/repos.d/rocm.repo && \
    echo "enabled=1" | tee --append /etc/zypp/repos.d/rocm.repo && \
    echo "gpgcheck=1" | tee --append /etc/zypp/repos.d/rocm.repo && \
    echo "gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" | tee --append /etc/zypp/repos.d/rocm.repo && \
    zypper --non-interactive --gpg-auto-import-keys refresh && \
    zypper --non-interactive install rocm7.1.0 && \
    zypper clean --all && \
    cat rocm_so.conf >> /etc/ld.so.conf.d/rocm.conf && \
    rm rocm_so.conf && \
    ldconfig

ENV PATH="/opt/rocm/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"

RUN zypper --non-interactive install wget tar gzip gcc gcc-fortran file gcc-c++ make && zypper clean --all && \
    if [ ! -z "$(gcc --version | grep -Eo -m 1 [0-9][0-9]+\.[0-9]+\.[0-9]+ | grep -m 1 [0-9])" ]; then \
    export FFLAGS=-fallow-argument-mismatch; \
    fi && \
    wget --progress=bar:force https://www.mpich.org/static/downloads/3.4.3/mpich-3.4.3.tar.gz && \
    tar xzf mpich-3.4.3.tar.gz && \
    cd mpich-3.4.3 && \
    ./configure --with-device=ch4:ofi --prefix=/opt/mpich/3.4.3 --with-hip=/opt/rocm/hip && \
    make install -j7 && \
    cd / && \
    rm -rf mpich-3.4.3/ mpich-3.4.3.tar.gz

ENV MPICH_VERSION="3.4.3" \
    MPICH_URL="https://www.mpich.org/static/downloads/3.4.3/mpich-3.4.3.tar.gz" \
    MPICH_DIR="/opt/mpich/3.4.3" \
    PATH="/opt/mpich/3.4.3/bin:$PATH" \
    LIBRARY_PATH="/opt/mpich/3.4.3/lib:$LIBRARY_PATH" \
    LD_LIBRARY_PATH="/opt/mpich/3.4.3/lib:$LD_LIBRARY_PATH" \
    MANPATH="/opt/mpich/3.4.3/share/man:$MANPATH"

# Install CMake
RUN zypper --non-interactive install cmake && zypper clean --all

# Install Python headers
RUN zypper --non-interactive install python3-devel && zypper clean --all

# Set environment variables for ROCm
ENV ROCM_PATH=/opt/rocm
ENV hip_DIR=/opt/rocm/lib/cmake/hip
ENV rocrand_DIR=/opt/rocm/lib/cmake/rocrand
ENV hiprand_DIR=/opt/rocm/lib/cmake/hiprand
ENV rocprim_DIR=/opt/rocm/lib/cmake/rocprim
ENV rocsparse_DIR=/opt/rocm/lib/cmake/rocsparse
ENV AMREX_AMD_ARCH=gfx90a
ENV CC=amdclang
ENV CXX=amdclang++

# Install blosc2 (needed for ADIOS2)
RUN pip3 install blosc2

# Install ADIOS2 (needed for OpenPMD)
RUN mkdir -p /tmp/build-adios2 && cd /tmp/build-adios2 && \
    wget -q https://github.com/ornladios/ADIOS2/archive/refs/tags/v2.11.0.tar.gz && \
    tar xzf v2.11.0.tar.gz && \
    mkdir adios2-build && cd adios2-build && \
    cmake ../ADIOS2-2.11.0 -DADIOS2_USE_Blosc2=ON -DADIOS2_USE_Fortran=OFF && \
    make -j$(nproc) && make install && \
    cd / && \
    rm -rf /tmp/build-adios2

# add packages needed for full C++20 support
RUN zypper --non-interactive install gcc13 gcc13-c++ libstdc++6-devel-gcc13 && zypper clean --all

# Install Kokkos (needed for Viskores for ROCm)
RUN mkdir -p /tmp/build-kokkos && cd /tmp/build-kokkos && \
    wget -q https://github.com/kokkos/kokkos/releases/download/5.0.0/kokkos-5.0.0.tar.gz && \
    tar xzf kokkos-5.0.0.tar.gz && \
    mkdir build && cd build && \
    cmake ../kokkos-5.0.0 -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON -DKokkos_ENABLE_ROCTHRUST=OFF -DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=OFF -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_EXTENSIONS=OFF -DBUILD_SHARED_LIBS=ON && \
    make -j$(nproc) && make install && \
    cd / && \
    rm -rf /tmp/build-kokkos

# Install Viskores (volume rendering and isosurfaces)
# NOTE: to build this, you need a Docker VM with a LARGE amount of RAM (at least 15 GB RAM, 4 GB swap)

RUN mkdir -p /tmp/build-viskores && cd /tmp/build-viskores && \
    wget -q https://github.com/Viskores/viskores/archive/refs/tags/v1.1.0.tar.gz && \
    tar xzf v1.1.0.tar.gz && \
    mkdir build && cd build && \
    cmake ../viskores-1.1.0 -DViskores_ENABLE_RENDERING=ON -DViskores_USE_64BIT_IDS=OFF -DViskores_USE_DOUBLE_PRECISION=ON -DViskores_ENABLE_KOKKOS=ON -DVTKm_ENABLE_KOKKOS_THRUST=OFF -DCMAKE_HIP_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC" && \
    make -j$(nproc) && make install && \
    cd / && \
    rm -rf /tmp/build-viskores

# add packages needed to download and build Quokka
RUN zypper --non-interactive install git gdb hdf5-devel python3-devel && zypper clean --all

# create user 'leap'
RUN useradd -ms /bin/bash leap
RUN usermod -a -G render,video leap
WORKDIR /home/leap
USER leap

# Set the default command to bash
CMD [ "/bin/bash" ]
