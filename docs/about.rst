.. About

About
=====

Quokka is a high-resolution shock capturing AMR radiation hydrodynamics code using the AMReX library :cite:`AMReX_JOSS` to provide patch-based adaptive mesh functionality. We take advantage of the C++ loop abstractions in AMReX in order to run with high performance on either CPUs, NVIDIA GPUs, or AMD GPUs.

Development methodology
-----------------------
The code is written in modern C++17, using MPI for distributed-memory
parallelism, and OpenMP for multithreading on CPUs, with the AMReX GPU
abstraction compiling as either native CUDA code or native HIP code when GPU support is enabled. 

We use a modern C++ development methodology, using CMake, CTest,
and Doxygen. We use :code:`clang-format` for automated code formatting, and :code:`clang-tidy` and SonarCloud for static analysis, in order to audit code adherence to the ISO C++ Core Guidelines and the MISRA C/C++ guidelines. We additionally ensure the code is free of memory corruption bugs using Clang's :code:`AddressSanitizer`.

There is an automated suite of test problems that can be run using CTest. Each test problem has a validated solution against which it is compared (usually in L1 norm) in order to pass.

Code development is managed using pull requests (PRs) on GitHub. 
In an effort to ensure long-term code maintainability,
all code must be written in C++17 following the Coding Guidelines, it must compile using Clang without warnings, all
tests must pass, and the static analyzers must show zero new bugs
before a pull request is merged with the main branch.

User assistance and bug reports are managed via Discussions and Issues
in the GitHub repository.


Numerical methods
-----------------

Hydrodynamics
~~~~~~~~~~~~~
The hydrodynamics solver is an unsplit method, using the
piecewise parabolic method :cite:`CW84` for reconstruction
in the primitive variables, the HLLC Riemann solver
:cite:`Toro2013` for flux computations, and a method-of-lines formulation for the time integration.

We use the method of :cite:`Miller_2002` to reduce the order of
reconstruction in zones where shocks are detected in order to suppress
spurious oscillations in strong shocks.

Radiation
~~~~~~~~~
The radiation hydrodynamics formulation is based on the mixed-frame
moment equations (e.g., :cite:`MihalasMihalas`). The radiation subsystem is coupled to the hydrodynamic subsystem
via operator splitting, with the hydrodynamic update computed first,
followed by the radiation update, with the latter update including
the source terms corresponding to the radiation four-force applied
to both the radiation and hydrodynamic variables. A method-of-lines formulation is also used for the time integration, with the time integration done by the same integrator chosen for the hydrodynamic subsystem.

The hyperbolic radiation subsystem is solved using an unsplit method, using PPM for reconstruction of the moment variables, with fluxes computed via the HLL Riemann solver, with the wavespeeds computed using the 'frozen Eddington factor' approximation :cite:`Balsara_1999`, which is more robust than using the eigenvalues of the M1 system :cite:`Skinner_2013` itself.

We reconstruct the energy density and the `reduced flux` :math:`f = F/cE`, in order to maintain the flux-limiting condition :math:`F \le cE` in discontinuous and near-discontinuous radiation flows.

To ensure the correct behavior of the advection terms in the asymptotic diffusion limit :cite:`Lowrie_2001`, we modify the Riemann solver according to :cite:`Skinner_2019`. We use
the Lorentz-factor local closure of :cite:`Levermore_1984` to compute 
the variable Eddington tensor.

The source terms corresponding to matter-radiation energy exchange are solved implicitly with the method of :cite:`Howell_2003` following
the hyperbolic subsystem update. The matter-radiation momentum update is likewise computed implicitly 
in order to maintain the correct behavior in the asymptotic diffusion limit :cite:`Skinner_2019`.
