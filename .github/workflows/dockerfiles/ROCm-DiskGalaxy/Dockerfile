FROM --platform=linux/amd64 opensuse/leap:15.6 AS build

ARG ROCM_VERSION=7.14
ARG ROCM_ZYPP_PACKAGES=
ARG MPICH_VERSION=4.2.2
ARG MPICH_URL=https://www.mpich.org/static/downloads/4.2.2/mpich-4.2.2.tar.gz
ARG ADIOS2_VERSION=2.12.1
ARG AMREX_SPACEDIM=3
ARG AMREX_MPI=ON
ARG AMREX_AMD_ARCH=gfx90a

# Combine system updates, repo setup, and package installation
RUN set -euo pipefail && \
    echo "rpm.install.excludedocs = yes" >> /etc/zypp/zypp.conf && \
    zypper --non-interactive refresh && \
    zypper --non-interactive update && \
    if ! zypper repos -E | grep -q devel_languages_perl; then \
      zypper --non-interactive addrepo https://download.opensuse.org/repositories/devel:/languages:/perl/15.6/devel:languages:perl.repo; \
    fi && \
    printf "[rocm]\nname=ROCm %s\nbaseurl=https://repo.amd.com/rocm/packages-multi-arch/sles15/x86_64\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg\npriority=50\n" \
      "${ROCM_VERSION}" > /etc/zypp/repos.d/rocm.repo && \
    zypper --non-interactive --gpg-auto-import-keys refresh && \
    if [ -n "${ROCM_ZYPP_PACKAGES}" ]; then \
      zypper --non-interactive install --no-recommends ${ROCM_ZYPP_PACKAGES}; \
    else \
      zypper --non-interactive install --no-recommends \
        "amdrocm-core-devel${ROCM_VERSION}-${AMREX_AMD_ARCH}"; \
    fi && \
    zypper --non-interactive install --no-recommends \
      cmake git gcc13 gcc13-c++ libstdc++6-devel-gcc13 \
      make tar gzip wget patch hdf5-devel python3-devel && \
    ln -sf /usr/lib64/libefa.so.1 /usr/lib64/libefa.so && \
    zypper clean --all && \
    printf "%s\n" "/opt/rocm/core-${ROCM_VERSION}/lib" >> /etc/ld.so.conf.d/rocm.conf && \
    ldconfig

ENV ROCM_PATH="/opt/rocm/core-${ROCM_VERSION}"

ENV PATH="${ROCM_PATH}/lib/llvm/bin:${ROCM_PATH}/bin:${PATH}" \
    LD_LIBRARY_PATH="${ROCM_PATH}/lib" \
    AMREX_AMD_ARCH="${AMREX_AMD_ARCH}" \
    CC="amdclang" \
    CXX="amdclang++"

COPY .github/workflows/dockerfiles/ROCm-DiskGalaxy/mpich-4.2.x-rocm-hip-pointer-attr.patch /tmp/mpich.patch

# Combine MPICH download, patch, and build
RUN set -euo pipefail && \
    mkdir -p /tmp/build-mpich && \
    cd /tmp/build-mpich && \
    wget -q "${MPICH_URL}" -O "mpich-${MPICH_VERSION}.tar.gz" && \
    tar xzf "mpich-${MPICH_VERSION}.tar.gz" && \
    cd "mpich-${MPICH_VERSION}" && \
    patch -p1 < /tmp/mpich.patch && \
    ./configure --with-device=ch4:ofi --with-hip="${ROCM_PATH}" --disable-fortran --prefix="/opt/mpich/${MPICH_VERSION}" && \
    make -j"$(nproc)" && \
    make install && \
    cd .. && \
    rm -rf "mpich-${MPICH_VERSION}/" "mpich-${MPICH_VERSION}.tar.gz" /tmp/mpich.patch && \
    cd / && \
    rm -rf /tmp/build-mpich


ENV MPICH_VERSION="${MPICH_VERSION}" \
    MPICH_DIR="/opt/mpich/${MPICH_VERSION}" \
    PATH="/opt/mpich/${MPICH_VERSION}/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/mpich/${MPICH_VERSION}/lib:${LD_LIBRARY_PATH}"

# Build ADIOS2 for openPMD output. Quokka never passes device pointers to ADIOS2,
# so GPU support is not needed. Optional compressors stay off so that the runtime
# stage needs no extra libraries.
RUN set -euo pipefail && \
    mkdir -p /tmp/build-adios2 && \
    cd /tmp/build-adios2 && \
    wget -q "https://github.com/ornladios/ADIOS2/archive/refs/tags/v${ADIOS2_VERSION}.tar.gz" -O adios2.tar.gz && \
    tar xzf adios2.tar.gz && \
    cmake -S "ADIOS2-${ADIOS2_VERSION}" -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="/opt/adios2/${ADIOS2_VERSION}" \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_C_COMPILER="${CC}" \
      -DCMAKE_CXX_COMPILER="${CXX}" \
      -DADIOS2_USE_MPI=ON \
      -DADIOS2_USE_Fortran=OFF \
      -DADIOS2_USE_Python=OFF \
      -DADIOS2_USE_HDF5=OFF \
      -DADIOS2_USE_SST=OFF \
      -DADIOS2_USE_ZeroMQ=OFF \
      -DADIOS2_USE_CUDA=OFF \
      -DADIOS2_USE_Kokkos=OFF \
      -DADIOS2_USE_BZip2=OFF \
      -DADIOS2_USE_PNG=OFF \
      -DADIOS2_USE_ZFP=OFF \
      -DADIOS2_USE_SZ=OFF \
      -DADIOS2_USE_MGARD=OFF \
      -DADIOS2_USE_Blosc2=OFF \
      -DADIOS2_USE_Sodium=OFF \
      -DADIOS2_USE_Catalyst=OFF \
      -DADIOS2_USE_Campaign=OFF \
      -DADIOS2_USE_OpenSSL=OFF \
      -DADIOS2_BUILD_EXAMPLES=OFF \
      -DBUILD_TESTING=OFF && \
    cmake --build build -j"$(nproc)" && \
    cmake --install build && \
    cd / && \
    rm -rf /tmp/build-adios2

ENV ADIOS2_DIR="/opt/adios2/${ADIOS2_VERSION}" \
    LD_LIBRARY_PATH="/opt/adios2/${ADIOS2_VERSION}/lib:${LD_LIBRARY_PATH}"

COPY . /tmp/quokka-src
WORKDIR /tmp/quokka-src

# Build Quokka with openPMD output: DiskGalaxy, plus HydroBlast3D and the
# OpenPMDCheckpoint compare tool for scripts/slurm/frontier-openpmd-restart.submit.
RUN set -euo pipefail && \
    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DAMReX_SPACEDIM="${AMREX_SPACEDIM}" \
      -DAMReX_GPU_BACKEND=HIP \
      -DAMReX_MPI="${AMREX_MPI}" \
      -DAMReX_GPU_ARCH="${AMREX_AMD_ARCH}" \
      -DCMAKE_C_COMPILER="${CC}" \
      -DCMAKE_CXX_COMPILER="${CXX}" \
      -DQUOKKA_PYTHON=OFF \
      -DQUOKKA_OPENPMD=ON \
      -DADIOS2_DIR="${ADIOS2_DIR}/lib/cmake/adios2" && \
    cmake --build build -j"$(nproc)" --target DiskGalaxy HydroBlast3D OpenPMDCheckpoint && \
    test -x build/src/problems/DiskGalaxy/DiskGalaxy && \
    test -x build/src/problems/HydroBlast3D/HydroBlast3D && \
    test -x build/src/problems/ParticleSchema/OpenPMDCheckpoint

# Runtime stage (reduced image size)
FROM --platform=linux/amd64 opensuse/leap:15.6

ARG ROCM_VERSION=7.14
ARG MPICH_VERSION=4.2.2
ARG ADIOS2_VERSION=2.12.1

COPY --from=build /opt/rocm /opt/rocm
COPY --from=build /opt/mpich /opt/mpich
COPY --from=build /opt/adios2 /opt/adios2

# Install runtime deps, cleanup, and setup user
RUN set -euo pipefail && \
    zypper --non-interactive refresh && \
    zypper --non-interactive install --no-recommends libhdf5 libpciaccess0 libnuma1 libdrm2 libdrm_amdgpu1 && \
    zypper clean --all && \
    printf "%s\n" "/opt/rocm/core-${ROCM_VERSION}/lib" >> /etc/ld.so.conf.d/rocm.conf && \
    rm -rf "/opt/rocm/core-${ROCM_VERSION}/bin" "/opt/rocm/core-${ROCM_VERSION}/include" \
           "/opt/rocm/core-${ROCM_VERSION}/share" "/opt/rocm/core-${ROCM_VERSION}/lib/cmake" \
           "/opt/rocm/core-${ROCM_VERSION}/lib/llvm/bin" "/opt/rocm/core-${ROCM_VERSION}/lib/llvm/include" \
           "/opt/rocm/core-${ROCM_VERSION}/lib/llvm/share" "/opt/rocm/core-${ROCM_VERSION}/lib/llvm/lib/cmake" && \
    find /opt/rocm -type f -name "*.a" -delete && \
    find /opt/rocm -type f -name "*.o" -delete && \
    find /opt/mpich -type f -name "*.a" -delete && \
    find /opt/mpich -type f -name "*.o" -delete && \
    rm -rf /opt/adios2/*/include /opt/adios2/*/share /opt/adios2/*/lib/cmake /opt/adios2/*/bin && \
    ldconfig && \
    mkdir -p /opt/quokka/bin /opt/quokka/lib /opt/quokka/share/quokka/inputs /opt/quokka/share/quokka/extern/agora_data \
      /opt/quokka/share/quokka/extern/cooling && \
    useradd -ms /bin/bash leap && \
    usermod -a -G render,video leap

COPY --from=build /tmp/quokka-src/build/src/problems/DiskGalaxy/DiskGalaxy /opt/quokka/bin/DiskGalaxy
COPY --from=build /tmp/quokka-src/build/src/problems/HydroBlast3D/HydroBlast3D /opt/quokka/bin/HydroBlast3D
COPY --from=build /tmp/quokka-src/build/src/problems/ParticleSchema/OpenPMDCheckpoint /opt/quokka/bin/OpenPMDCheckpoint
COPY --from=build /tmp/quokka-src/build/lib/ /opt/quokka/lib/
COPY --from=build /tmp/quokka-src/inputs/DiskGalaxy.toml /opt/quokka/share/quokka/inputs/DiskGalaxy.toml
COPY --from=build /tmp/quokka-src/inputs/HydroBlast3D.toml /opt/quokka/share/quokka/inputs/HydroBlast3D.toml
COPY --from=build /tmp/quokka-src/extern/agora_data/ /opt/quokka/share/quokka/extern/agora_data/
COPY --from=build /tmp/quokka-src/extern/cooling/ /opt/quokka/share/quokka/extern/cooling/

ENV ROCM_PATH="/opt/rocm/core-${ROCM_VERSION}"

ENV MPICH_VERSION="${MPICH_VERSION}" \
    PATH="/opt/quokka/bin:/opt/mpich/${MPICH_VERSION}/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/quokka/lib:/opt/adios2/${ADIOS2_VERSION}/lib:/opt/mpich/${MPICH_VERSION}/lib:${ROCM_PATH}/lib"

# Point the bundled DiskGalaxy input at the bundled cooling table, and fail the
# build if a shared library of any binary is missing at runtime.
RUN set -euo pipefail && \
    sed -i 's|"\./CloudyData_UVB=HM2012_resampled.h5"|"/opt/quokka/share/quokka/extern/cooling/CloudyData_UVB=HM2012_resampled.h5"|' \
      /opt/quokka/share/quokka/inputs/DiskGalaxy.toml && \
    grep -q '"/opt/quokka/share/quokka/extern/cooling/CloudyData_UVB=HM2012_resampled.h5"' /opt/quokka/share/quokka/inputs/DiskGalaxy.toml && \
    chmod 0755 /opt/quokka/bin/* && \
    chown -R leap /opt/quokka && \
    if ldd /opt/quokka/bin/* | grep "not found"; then exit 1; fi

USER leap
WORKDIR /home/leap
