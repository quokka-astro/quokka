[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=BenWibking_TwoMomentRad&metric=alert_status&token=5049c56ffe08dcc83afd5ca4c8e0d951a2836652)](https://sonarcloud.io/dashboard?id=BenWibking_TwoMomentRad)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=BenWibking_TwoMomentRad&metric=bugs&token=5049c56ffe08dcc83afd5ca4c8e0d951a2836652)](https://sonarcloud.io/dashboard?id=BenWibking_TwoMomentRad)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=BenWibking_TwoMomentRad&metric=ncloc&token=5049c56ffe08dcc83afd5ca4c8e0d951a2836652)](https://sonarcloud.io/dashboard?id=BenWibking_TwoMomentRad)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9489/badge)](https://www.bestpractices.dev/projects/9489)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/quokka-astro/quokka/badge)](https://scorecard.dev/viewer/?uri=github.com/quokka-astro/quokka)
[![Cite](https://img.shields.io/badge/Cite-Quokka-blue)](https://www.tomwagg.com/software-citation-station/?auto-select=Quokka)
[![AMReX](https://amrex-codes.github.io/badges/powered%20by-AMReX-red.svg)](https://amrex-codes.github.io)
[![yt-project](https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet")](https://yt-project.org)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/quokka-astro/quokka)
[![Join Zulip Community](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://quokka-astro.zulipchat.com/join/3sahu2rgvkengbkzwkpndc5h/)

# QUOKKA
*Quadrilateral, Umbra-producing, Orthogonal, Kangaroo-conserving Kode for Astrophysics!*

**For detailed instructions on installing the code, please refer to the [Quokka Documentation](https://quokka-astro.github.io/quokka/index.html). You can start a [Discussion](https://github.com/BenWibking/quokka/discussions) for technical support, or open an [Issue](https://github.com/BenWibking/quokka/issues) for any bug reports.**

Quokka is a two-moment radiation hydrodynamics code that uses the piecewise-parabolic method, with AMR and subcycling in time. Runs on CPUs (MPI+vectorized) or NVIDIA GPUs (MPI+CUDA) with a single-source codebase. Written in C++20. (100% Fortran-free.)

![Quokka capabilities showcase graphic](extern/quokka_graphic.png)

This is a [a Kelvin-Helmholz instability simulated with Quokka](https://vimeo.com/714653592) on a 512x512 uniform grid:

![Animated GIF of KH Instability](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/1f468be6-6d7b-4d53-a02c-4dd8f3ad5154.gif?ClientID=vimeo-core-prod&Date=1653705774&Signature=9bea89d5c9657180391a9538a10fd4f8f7099025)

## Dependencies
* C++ compiler (with C++20 support)
* CMake 3.16+
* Python 3.8+
* MPI library with GPU-aware support (OpenMPI, MPICH, or Cray MPI)
* HDF5 1.10+ (serial version)
* CUDA 12.0+ (optional, for NVIDIA GPUs)
* ROCm 6.3+ (optional, for AMD GPUs)
* ADIOS2 2.9+ with GPU-aware support (optional, for writing terabyte-sized or larger outputs)

## Problems?
If you run into problems, please start a [Discussion](https://github.com/BenWibking/quokka/discussions) for technical support. If you discover a bug, please let us know by opening an [Issue](https://github.com/BenWibking/quokka/issues). You can also join our [Zulip community](https://quokka-astro.zulipchat.com/join/3sahu2rgvkengbkzwkpndc5h/).

## Acknowledgements
We thank Kandra Labs, Inc. (“Zulip”) for providing hosted [Zulip](https://zulip.com/) chat for our open-source community.
