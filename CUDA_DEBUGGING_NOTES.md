# CUDA Debugging Notes For Future Agents

Useful advice for future agents debugging this:

- Don't assume `amrex::Gpu::PinnedVector` is the bug. In this tree, `The_Pinned_Arena()` uses `cudaHostAlloc(..., cudaHostAllocMapped)` in [AMReX_Arena.cpp](/mnt/home/wibkingb/quokka/extern/amrex/Src/Base/AMReX_Arena.cpp:221), so device reads from that memory are a legitimate path.
- Don't use `-DCMAKE_BUILD_TYPE=Debug` as the first CUDA debugging move here. It turns on `-O0` and changes behavior enough that `DiskGalaxy` stopped reproducing quickly under `cuda-gdb`.
- Use `RelWithDebInfo` plus explicit CUDA line info instead. For this repo, that means configuring a fresh tree with `-DAMReX_GPU_BACKEND=CUDA -DCMAKE_BUILD_TYPE=RelWithDebInfo`; the AMReX summary should show `CUDA flags ... --generate-line-info`.
- Be explicit about the GPU backend. I accidentally configured a `RelWithDebInfo` tree with `AMReX_GPU_BACKEND=NONE`, which gave a CPU-only stack and wasted time.
- Run outside the sandbox on this machine. Inside the sandbox, OpenMPI emits `opal_ifinit: socket() failed with errno=1`, which pollutes both build and debugger runs.
- The user's `inputs/DiskGalaxy.toml` is dirty, so don't assume stock behavior. Treat input overrides as part of the bug.
- The highest-value result so far is that the fault is not in cooling-table startup. Under `cuda-gdb` with the original `3d` CUDA binary, the first useful stop was a CUDA exception in the 7-point Gauss quadrature path, reported at [gauss.hpp](/mnt/home/wibkingb/quokka/src/math/gauss.hpp:639).
- The symbolized host-side stack from the accidental CPU-only run points straight at the optional halo parser path: `halo_vphi_parser(x, y, z)` at [testDiskGalaxy.cpp](/mnt/home/wibkingb/quokka/src/problems/DiskGalaxy/testDiskGalaxy.cpp:361), feeding the halo azimuthal velocity terms at [378](/mnt/home/wibkingb/quokka/src/problems/DiskGalaxy/testDiskGalaxy.cpp:378), [403](/mnt/home/wibkingb/quokka/src/problems/DiskGalaxy/testDiskGalaxy.cpp:403), and the `quad_3d` call at [424](/mnt/home/wibkingb/quokka/src/problems/DiskGalaxy/testDiskGalaxy.cpp:424). That is the first place I would focus next.
- Practical next step: reproduce with a CUDA `RelWithDebInfo` build outside the sandbox, then check whether `disk_galaxy.halo_vphi_expr` is set and whether the AMReX parser executor is valid on device for that expression path.
