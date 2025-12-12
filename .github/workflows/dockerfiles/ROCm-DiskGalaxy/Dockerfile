ARG BASE_IMAGE=ghcr.io/quokka-astro/quokka-linux-amd64-rocm:development
FROM ${BASE_IMAGE} AS build

ARG QUOKKA_REPO=https://github.com/quokka-astro/quokka.git
ARG QUOKKA_REF=development

WORKDIR /tmp/quokka-src
COPY . .

RUN set -euo pipefail && \
    if [[ ! -f CMakeLists.txt ]]; then \
      echo "No CMakeLists.txt in build context; cloning ${QUOKKA_REPO}@${QUOKKA_REF}..." && \
      find . -mindepth 1 -maxdepth 1 -exec rm -rf {} + && \
      git clone --depth 1 --recurse-submodules --shallow-submodules --branch "${QUOKKA_REF}" "${QUOKKA_REPO}" . ; \
    fi && \
    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DAMReX_SPACEDIM=3 \
      -DAMReX_GPU_BACKEND=HIP \
      -DAMReX_MPI=ON \
      -DAMReX_GPU_ARCH="${AMREX_AMD_ARCH:-gfx90a}" && \
    cmake --build build -j"$(nproc)" --target DiskGalaxy && \
    test -x build/src/problems/DiskGalaxy/DiskGalaxy

FROM ${BASE_IMAGE}

USER root
RUN mkdir -p /opt/quokka/bin /opt/quokka/share/quokka/inputs /opt/quokka/share/quokka/extern/agora_data
COPY --from=build /tmp/quokka-src/build/src/problems/DiskGalaxy/DiskGalaxy /opt/quokka/bin/DiskGalaxy
COPY --from=build /tmp/quokka-src/inputs/DiskGalaxy.in /opt/quokka/share/quokka/inputs/DiskGalaxy.in
COPY --from=build /tmp/quokka-src/extern/agora_data/ /opt/quokka/share/quokka/extern/agora_data/
RUN chmod 0755 /opt/quokka/bin/DiskGalaxy && chown -R leap:leap /opt/quokka
ENV PATH="/opt/quokka/bin:${PATH}"
USER leap
WORKDIR /home/leap
