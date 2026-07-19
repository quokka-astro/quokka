# Assertions and error checking

## AMReX assert macros

AMReX provides several assertion macros:

-   `AMREX_ASSERT`: Enabled when either `-DAMREX_DEBUG` or `-DAMREX_USE_ASSERTION` is defined. When enabled, it calls `amrex::Assert()` which will abort unless both `-DNDEBUG` is defined AND `-DAMREX_USE_ASSERTION` is NOT defined. This means:
    - The macro itself is a no-op when neither `AMREX_DEBUG` nor `AMREX_USE_ASSERTION` is defined
    - When the macro is enabled and assertion fails, it will abort unless `NDEBUG` is defined without `AMREX_USE_ASSERTION`
-   `AMREX_ALWAYS_ASSERT`: Always calls `amrex::Assert()` regardless of build configuration. The actual abort behavior follows the same rules as `AMREX_ASSERT` - it will abort unless both `-DNDEBUG` is defined AND `-DAMREX_USE_ASSERTION` is NOT defined. **Note that CMake adds "-DNDEBUG" by default when "CMAKE_BUILD_TYPE=Release".** (See this [GitHub discussion](https://github.com/AMReX-Codes/amrex/discussions/2648) for details.)

## Abort

Because the default CMake flags added in Release mode causes ``AMREX_ALWAYS_ASSERT`` not to function in GPU code, ``amrex::Abort`` is the best option to use if you want to abort a GPU kernel.

``amrex::Abort`` requires additional GPU register usage, so it should be used sparingly. The best strategy for error handling is often to set a value in an array that indicates an iterative solve failed in a given cell. (This is what Castro does for its nuclear burning networks.)

For more details, see the [AMReX documentation on assertions and error checking](https://amrex-codes.github.io/amrex/docs_html/GPU.html#assertions-and-error-checking).
