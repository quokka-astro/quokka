.. Assertions and error checking

Assertions and error checking
==========================

AMReX assert macros
-----------------------

AMReX provides several assertion macros:

* `AMREX_ASSERT`: Works when `CMAKE_BUILD_TYPE=Debug`.
* `AMREX_ALWAYS_ASSERT`: Always works on CPU. **Works on GPU only if "-DNDEBUG" is NOT added to the compiler flags. Note that CMake adds "-DNDEBUG" by default when "CMAKE_BUILD_TYPE=Release".** (See this `GitHub discussion <https://github.com/AMReX-Codes/amrex/discussions/2648>`_ for details.)

Abort
-----------------------

Because the default CMake flags added in Release mode causes `AMREX_ALWAYS_ASSERT` not to function in GPU code, `amrex::Abort` is the best option to use if you want to abort a GPU kernel.

`amrex::Abort` requires additional GPU register usage, so it should be used sparingly. The best strategy for error handling is often to set a value in an array that indicates an iterative solve failed in a given cell. (This is what Castro does for its nuclear burning networks.)

For more details, see the `AMReX documentation on assertions and error checking <https://amrex-codes.github.io/amrex/docs_html/GPU.html#assertions-and-error-checking>`_.
