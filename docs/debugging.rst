.. Debugging

Debugging
=====

General guidelines
-----------------------

A good general guide to debugging simulation codes is `provided by WarpX <https://warpx.readthedocs.io/en/latest/usage/workflows/debugging.html>`_ (although some details are WarpX-specific).

Common bugs
-----------------------

You might be reading this page because you have encountered a common type of bug:

* an `out-of-bounds array access <https://www.geeksforgeeks.org/accessing-array-bounds-ccpp/>`_

  This is when the code accesses an array with an index that corresponds to an element that doesn't actually exist. In C++, this causes the computer to access a memory location that is completely unrelated to the array that you intended to access. In a CPU code, this usually causes a silently incorrect result, but on GPU, this may actually cause the simulation to crash. However, if you are accessing an ``amrex::Array4`` object *and you have compiled in Debug mode*, then AMReX will issue an error message when this occurs. There is a significant performance cost to this error checking, so it does not occur when compiled in Release mode.

* accessing a host variable from the GPU

  The second most common type of bug encountered in Quokka is accessing a host variable (i.e., a variable that can only be accessed from code that runs on the CPU) from code running on the GPU (i.e., within a ``ParallelFor``). Sometimes the compiler will detect this situation and print an error message, but often this will only present an issue when actuallly running the code -- for instance, this can happen when the GPU code tries to dereference a pointer to an address in CPU memory. In that case, the only way to debug this error is to run Quokka under ``cuda-gdb`` (or, on AMD GPUs, ``rocgdb``).

How to debug on GPUs
-----------------------

The best way to debug on GPUs is to... not debug on GPUs. That is, it is always easier to instead debug the problem on a CPU-only run. GPU debugging is very painful and itself quite buggy. This is unfortunately true for all GPU vendors.

* Try to shrink the problem to a size that can run on a single node, or (even better) a single MPI rank / CPU core, but where it still exhibits the error that you are trying to debug.

* Build Quokka without GPU support but with ``-DCMAKE_BUILD_TYPE=Debug`` and re-run. If there are any array out-of-bounds errors, it will stop and report exactly which array is being accessed out-of-bounds and what the indices are. The only downside is that Quokka will run very slowly in this mode.

  * For more details, see the `AMReX debugging guide <https://amrex-codes.github.io/amrex/docs_html/Debugging.html>`_.

* Build Quokka without GPU support but with ``-DCMAKE_BUILD_TYPE=Release -DENABLE_ASAN=ON``. This turns on the AddressSanitizer, which checks for out-of-bounds array accesses and other memory bugs. This is faster than the previous method, but it produces less informative error messages (e.g., no array indices).

  * This method may produce a lot of messages about memory leaks, `which are not necessarily bugs <https://stackoverflow.com/a/654766>`_, and should not cause GPU crashes. These messages `can be disabled <https://stackoverflow.com/questions/51060801/how-to-suppress-leaksanitizer-report-when-running-under-fsanitize-address>`_ if you are looking for, e.g., out-of-bounds array accesses, which is a class of bug that can cause a GPU crash.

  * It is recommended to set these environment variables when you run it: ``ASAN_OPTIONS=abort_on_error=1:fast_unwind_on_malloc=1:detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=0``. Note that CTest appends its own options to this environment variable when running tests, so it is recommended to run the simulation manually (i.e., without ``make test``, ``ninja test``, or ``ctest``).

  * For more information, see this `guide to using AddressSanitizer in an HPC context <https://www.osc.edu/resources/getting_started/howto/howto_use_address_sanitizer>`_.

* On AMD GPUs, there is a `GPU-aware AddressSanitizer <https://rocm.docs.amd.com/en/latest/understand/using_gpu_sanitizer.html#compiling-for-address-sanitizer>`_. Currently, enabling this requires manually changing the compiler flags.

How to actually debug on GPUs
-----------------------

As an *absolute last resort* if it is impossible to reproduce the error you are seeing on a CPU-only run, then the best option is to:

* downsize the simulation to fit on a single GPU

* start the simulation on an NVIDIA GPU from within CUDA-GDB
  (see the `documentation <https://docs.nvidia.com/cuda/cuda-gdb/index.html>`_ and `slides <https://www.olcf.ornl.gov/wp-content/uploads/2021/06/cuda_training_series_cuda_debugging.pdf>`_).

* hope CUDA-GDB does not itself crash

* hope CUDA-GDB produces a useful error message that you can analyze

NVIDIA also provides the ``compute-sanitizer`` tool that is essentially the equivalent of AddressSanitizer (see the `documentation <https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>`_). Unfortunately, it does not work as reliably as AddressSanitizer, and may itself crash while attempting to debug a GPU program.

For AMD GPUs, you have to use the AMD-provided debugger ``rocgdb``. A tutorial its use is available `here <https://www.olcf.ornl.gov/wp-content/uploads/2021/04/rocgdb_hipmath_ornl_2021_v2.pdf>`_.

AMD also provides a GPU-aware AddressSanitizer that can be enabled when building Quokka. Currently, the compiler flags must be manually modified in order to enable this. For details, see its `documentation <https://rocm.docs.amd.com/en/latest/understand/using_gpu_sanitizer.html#compiling-for-address-sanitizer>`_.

GPU kernel asynchronicity
-----------------------

**By default, GPU kernels launch asynchronously, i.e., execution of CPU code continues before the kernel starts on the GPU. This can cause synchronization problems if there is an implicit assumption about the order of operations with respect to CPU and GPU code.**

The easiest way to debug this is to set the environment variables:

* ``CUDA_LAUNCH_BLOCKING=1`` on NVIDIA GPUs, or
* ``HIP_LAUNCH_BLOCKING=1`` on AMD GPUs.

This will cause the CPU to wait until the GPU kernel execution is complete before continuing past the call to ``ParallelFor``.

For more details, refer to the `AMReX GPU debugging guide <https://amrex-codes.github.io/amrex/docs_html/Debugging.html#basic-gpu-debugging>`_.

When all else fails: Debugging with ``printf``
-----------------------

If you have tried *all* of the above steps, then you have to resort to adding ``printf`` statements within the GPU code. Note that ``printf`` inside GPU code is different from the CPU-side ``printf`` function, as explained in the `NVIDIA documentation <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#formatted-output>`_.
