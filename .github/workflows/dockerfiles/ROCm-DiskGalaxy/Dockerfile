FROM --platform=linux/amd64 opensuse/leap:15.6 AS build

ARG ROCM_ZYPP_REPO_VERSION=6.4.4
ARG ROCM_ZYPP_PACKAGES=
ARG MPICH_VERSION=4.1.2
ARG MPICH_URL=https://www.mpich.org/static/downloads/4.1.2/mpich-4.1.2.tar.gz
ARG AMREX_SPACEDIM=3
ARG AMREX_MPI=ON
ARG AMREX_AMD_ARCH=gfx90a

RUN set -euo pipefail && \
    echo "rpm.install.excludedocs = yes" >> /etc/zypp/zypp.conf && \
    zypper --non-interactive refresh && \
    zypper --non-interactive update && \
    zypper clean --all

COPY . /tmp/quokka-src
WORKDIR /tmp/quokka-src

COPY .github/workflows/dockerfiles/ROCm-DiskGalaxy/rocm_so.conf /tmp/rocm_so.conf
#COPY rocm_so.conf /tmp/rocm_so.conf
RUN set -euo pipefail && \
    if ! zypper repos -E | grep -q devel_languages_perl; then \
      zypper --non-interactive addrepo https://download.opensuse.org/repositories/devel:/languages:/perl/15.6/devel:languages:perl.repo; \
    fi && \
    printf "[ROCm-%s]\nname=ROCm%s\nbaseurl=https://repo.radeon.com/rocm/zyp/%s/main\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key\n" \
      "${ROCM_ZYPP_REPO_VERSION}" "${ROCM_ZYPP_REPO_VERSION}" "${ROCM_ZYPP_REPO_VERSION}" > /etc/zypp/repos.d/rocm.repo && \
    zypper --non-interactive --gpg-auto-import-keys refresh && \
    if [ -n "${ROCM_ZYPP_PACKAGES}" ]; then \
      zypper --non-interactive install --no-recommends ${ROCM_ZYPP_PACKAGES}; \
    else \
      zypper --non-interactive install --no-recommends \
        rocm-llvm \
        hipcc \
        hip-runtime-amd \
        hip-devel \
        hsa-rocr \
        comgr \
        rocm-device-libs \
	rocrand-devel \
	hiprand-devel \
	rocprim-devel; \
    fi && \
    zypper --non-interactive install --no-recommends \
      cmake \
      git \
      gcc13 gcc13-c++ libstdc++6-devel-gcc13 \
      make tar gzip wget \
      patch \
      python3-devel && \
    ln -sf /usr/lib64/libefa.so.1 /usr/lib64/libefa.so && \
    zypper clean --all && \
    cat /tmp/rocm_so.conf >> /etc/ld.so.conf.d/rocm.conf && \
    rm /tmp/rocm_so.conf && \
    ldconfig

ENV PATH="/opt/rocm/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}" \
    ROCM_PATH="/opt/rocm" \
    AMREX_AMD_ARCH="${AMREX_AMD_ARCH}" \
    CC="amdclang" \
    CXX="amdclang++"

COPY .github/workflows/dockerfiles/ROCm-DiskGalaxy/mpich-4.1.2-rocm-6.4-hip-pointer-attr.patch /tmp/mpich-rocm.patch
#COPY mpich-4.1.2-rocm-6.4-hip-pointer-attr.patch /tmp/mpich-rocm.patch
RUN set -euo pipefail && \
    mkdir -p /tmp/build-mpich && \
    cd /tmp/build-mpich && \
    wget -q "${MPICH_URL}" -O "mpich-${MPICH_VERSION}.tar.gz" && \
    tar xzf "mpich-${MPICH_VERSION}.tar.gz" && \
    cd "mpich-${MPICH_VERSION}" && \
    patch -p1 < /tmp/mpich-rocm.patch && \
    ./configure --with-device=ch4:ofi --disable-fortran --prefix="/opt/mpich/${MPICH_VERSION}" && \
    make -j"$(nproc)" && \
    make install && \
    cd / && \
    rm -rf /tmp/build-mpich /tmp/mpich-rocm.patch

ENV MPICH_VERSION="${MPICH_VERSION}" \
    MPICH_DIR="/opt/mpich/${MPICH_VERSION}" \
    PATH="/opt/mpich/${MPICH_VERSION}/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/mpich/${MPICH_VERSION}/lib:${LD_LIBRARY_PATH}"

RUN set -euo pipefail && \
    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DAMReX_SPACEDIM="${AMREX_SPACEDIM}" \
      -DAMReX_GPU_BACKEND=HIP \
      -DAMReX_MPI="${AMREX_MPI}" \
      -DAMReX_GPU_ARCH="${AMREX_AMD_ARCH}" \
      -DCMAKE_C_COMPILER="${CC}" \
      -DCMAKE_CXX_COMPILER="${CXX}" && \
    cmake --build build -j"$(nproc)" --target DiskGalaxy && \
    test -x build/src/problems/DiskGalaxy/DiskGalaxy

FROM --platform=linux/amd64 opensuse/leap:15.6

ARG MPICH_VERSION=4.1.2

USER root
COPY --from=build /opt/rocm /opt/rocm
COPY --from=build /opt/mpich /opt/mpich
COPY .github/workflows/dockerfiles/ROCm-DiskGalaxy/rocm_so.conf /tmp/rocm_so.conf
#COPY rocm_so.conf /tmp/rocm_so.conf
RUN set -euo pipefail && \
    cat /tmp/rocm_so.conf >> /etc/ld.so.conf.d/rocm.conf && \
    rm /tmp/rocm_so.conf && \
    rm -rf \
      /opt/rocm/bin \
      /opt/rocm/include \
      /opt/rocm/share \
      /opt/rocm/lib/cmake \
      /opt/rocm/llvm/bin \
      /opt/rocm/llvm/include \
      /opt/rocm/llvm/share \
      /opt/rocm/llvm/lib/cmake && \
    find /opt/rocm -type f -name "*.a" -delete && \
    find /opt/rocm -type f -name "*.o" -delete && \
    find /opt/mpich -type f -name "*.a" -delete && \
    find /opt/mpich -type f -name "*.o" -delete && \
    ldconfig

RUN mkdir -p /opt/quokka/bin /opt/quokka/share/quokka/inputs /opt/quokka/share/quokka/extern/agora_data
COPY --from=build /tmp/quokka-src/build/src/problems/DiskGalaxy/DiskGalaxy /opt/quokka/bin/DiskGalaxy
COPY --from=build /tmp/quokka-src/inputs/DiskGalaxy.in /opt/quokka/share/quokka/inputs/DiskGalaxy.in
COPY --from=build /tmp/quokka-src/extern/agora_data/ /opt/quokka/share/quokka/extern/agora_data/
RUN chmod 0755 /opt/quokka/bin/DiskGalaxy

RUN useradd -ms /bin/bash leap && \
    usermod -a -G render,video leap && \
    chown -R leap /opt/quokka

ENV MPICH_VERSION="${MPICH_VERSION}" \
    PATH="/opt/quokka/bin:/opt/mpich/${MPICH_VERSION}/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/mpich/${MPICH_VERSION}/lib:${LD_LIBRARY_PATH}"
USER leap
WORKDIR /home/leap
