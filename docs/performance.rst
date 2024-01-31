.. Performance

Performance tips
=====

Prerequisites
-----------------------

You should:

* Understand what a `GPU kernel <https://en.wikipedia.org/wiki/Compute_kernel>`_ is. (For reference, consult these `notes <https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/kernel_sm>`_.)

* Understand what a `processor register <https://en.wikipedia.org/wiki/Processor_register>`_ is.

* Know that calling `amrex::ParallelFor` launches a GPU kernel (when GPU support is enabled at compile time).

GPU hardware characteristics
-----------------------

GPUs have hardware design features that make their performance characteristics significantly different from CPUs. In practice, two factors dominate GPU performance behavior:

* *Kernel launch latency:* this is a fundamental hardware characteristic of GPUs. It takes several microseconds (typically 3-10 microseconds, but it can vary depending on the compute kernel, the GPU hardware, the CPU hardware, and the driver) to launch a GPU kernel (i.e., to start running the code within an `amrex::ParallelFor` on the GPU). In practice, latency is generally longer for AMD and Intel GPUs.

* *Register pressure:* the number of registers per thread available for use by a given kernel is limited to the size of the GPU register file divided by the number of threads. If a kernel needs more registers than are available in the register file, the compiler will "spill" registers to memory, which will then make the kernel run very slowly. Alternatively, the number of concurrent threads can be reduced, which increases the number of registers available per thread.
  
  * For more details, see these `AMD website notes <https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/>`_ and OLCF `training materials <https://www.olcf.ornl.gov/wp-content/uploads/Intro_Register_pressure_ORNL_20220812_2083.pdf>`_.

MPI communication latency vs. bandwidth
-----------------------

A traditional rule of thumb for CPU-based MPI codes is that communication latency often limits performance when scaling to large number of CPU cores (or, equivalently, MPI ranks). We have found that this is *not* the case for Quokka when running on GPU nodes (by, e.g., adding additional dummy variables to the state arrays).

Communication performance appears to (virtually always) be *bandwidth limited*. There are two likely reasons for this:

* GPU node performance is about 10x faster than CPU node performance, whereas network bandwidth is only 2-4x larger on GPU nodes compared to CPU nodes. The network bandwidth to compute ratio is therefore *lower* on GPU nodes than on CPU nodes.
* GPU kernel launch latency (3-10 microseconds) is often larger than the minimum MPI message latency (i.e., the latency for small messages to travel between nodes) of 2-3 microseconds.

Guidelines
-----------------------

* Combine operations into fewer kernels in order to reduce the fraction of time lost to kernel launch latency
  
  * This can also be done by using the `MultiFab` version of `ParallelFor` that operates on all of the FABs at once, rather than launching a separate kernel for each FAB. This should not increase register pressure.
  
  * *However,* combining multiple kernels can increase register pressure, which can decrease performance. There is no real way to know a priori whether there will be a net performance gain or loss without trying it out. The strategy that yields the best performance may be different for GPUs from different vendors!

* Split operations into multiple kernels in order to decrease register pressure
  
  * *However,* this may increase the time lost due to kernel launch latency. This is an engineering trade-off that must be determined by performance measurements on the GPU hardware. This trade-off may be different on GPUs from different vendors!

* In order to decrease register pressure, avoid using `printf`, `assert`, and `amrex::Abort` in GPU code . All of these functions require using additional registers that could instead be allocated to the useful computations does in a kernel. This may require a significant code rewrite to handle errors in a different way. (You should *not* just ignore errors, e.g. in an iterative solver.)

* *Experts only:* Manually tune the number of GPU threads per block on a kernel-by-kernel basis. This can reduce register pressure by allowing each thread to use more registers. Note that this is an advanced optimization and should only be done with careful performance measurements done on multiple GPUs. The `AMReX documentation <https://amrex-codes.github.io/amrex/docs_html/GPU.html#gpu-block-size>`_ provides guidance on how to do this.
