# QUOKKA

**Quadrilateral, Umbra-producing, Orthogonal, Kangaroo-conserving Kode for Astrophysics!**

Quokka is a two-moment radiation hydrodynamics code that uses the piecewise-parabolic method, with AMR and subcycling in time. Runs on CPUs (MPI+vectorized) or NVIDIA GPUs (MPI+CUDA) with a single-source codebase. Written in C++17. (100% Fortran-free.)

!!! Note  
    The Quokka methods paper is now [available on arXiv](https://arxiv.org/abs/2110.01792).


We use the AMReX library [@AMReX_JOSS] to provide patch-based adaptive mesh functionality. We take advantage of the C++ loop abstractions in AMReX in order to run with high performance on either CPUs or GPUs.

Example simulation set-ups are included in the GitHub repository for many astrophysical problems of interest related to star formation and the interstellar medium.

## Contact

All communication takes place on the [Quokka GitHub repository](https://github.com/quokka-astro/quokka). You can start a [Discussion](https://github.com/quokka-astro/quokka/discussions) for technical support, or open an [Issue](https://github.com/quokka-astro/quokka/issues) for any bug reports.
