# GPU Debugging with Agents: A Real-Life Example

This note documents a CUDA debugging session on `DiskGalaxy`, including a plausible but incorrect hypothesis, the evidence that falsified it, the fix that resolved the problem, and a suggested playbook for future GPU debugging.

## Problem

`DiskGalaxy` crashed on CUDA during initialization with:

```text
CUDA error 700: an illegal memory access was encountered
```

The failure occurred before the main evolution loop, during `setInitialConditions()`.

## Investigation timeline

### 1. Start with the best reproduction, not the most convenient one

The first useful setup was:

- build with `AMReX_GPU_BACKEND=CUDA`
- use `RelWithDebInfo`
- make sure CUDA line info is enabled
- run outside the sandbox on this machine, because OpenMPI/PMIx fails noisily inside it

A CPU-only build was useful for context, but it also risked emphasizing CPU-only artifacts instead of the actual device failure. In the same spirit, it was important not to over-interpret `amrex::Gpu::PinnedVector` as the likely bug source. In this codebase, `The_Pinned_Arena()` uses `cudaHostAlloc(..., cudaHostAllocMapped)`, so device reads from that memory are a legitimate access path rather than immediate evidence of a host/device pointer bug.

`Debug` was also not the best first CUDA debugging configuration for this case. Here it changed behavior enough that `DiskGalaxy` stopped reproducing quickly under `cuda-gdb`, so `RelWithDebInfo` with line information was the better diagnostic build.

### 2. The first strong hypothesis was the halo parser

The optional `disk_galaxy.halo_vphi_expr` path looked suspicious for several reasons:

- the input file enabled it
- an earlier CPU-side stack pointed into the halo azimuthal velocity path in `testDiskGalaxy.cpp`
- the parser sat on the same path as the `quad_3d` calls used during initialization

This made the parser hypothesis plausible:

- maybe the runtime expression path was invalid on device
- maybe the parser executor lifetime was wrong
- maybe the expression itself triggered the crash

This was a plausible lead, but it remained a hypothesis.

### 3. The parser theory was plausible, but wrong

Two factors pushed the investigation too far toward the parser path:

- it gave too much weight to a CPU-side stack from a different build mode
- it leaned on evidence from `EB2::ParserIF`, which is not the parser path `DiskGalaxy` is using

A plausible theory is not a diagnosis. It must survive isolation tests.

### 4. Isolation tests killed the parser hypothesis

The following tests were run:

- replace `disk_galaxy.halo_vphi_expr` with `0.0`
- remove `disk_galaxy.halo_vphi_expr` from the input entirely

The CUDA failure still happened.

This established that:

- the optional halo parser path was not required to trigger the crash
- any parser-specific ownership concern was, at best, incidental

At that point the parser stopped being the leading theory.

### 5. The decisive tool was `compute-sanitizer`

The key command was:

```bash
compute-sanitizer --tool memcheck ../build/3d-rwdi-cuda/src/problems/DiskGalaxy/DiskGalaxy ../inputs/DiskGalaxy.toml
```

This produced the crucial diagnosis:

```text
Stack overflow
... in gauss.hpp:639
... launched from testDiskGalaxy.cpp:254
```

This changed the interpretation of the failure. It was not a generic illegal memory access from a bad pointer. It was a device stack overflow in the nested Gauss quadrature path used by `quad_3d`.

One additional configuration detail mattered here: be explicit about the GPU backend. An earlier accidental `RelWithDebInfo` configuration with `AMReX_GPU_BACKEND=NONE` produced a CPU-only stack and sent the investigation in the wrong direction.

### 6. The actual bug was recursive stack growth through nested device lambdas

`quad_3d` was implemented by nesting:

- `quad_3d`
- `quad_2d`
- `quad_1d`
- `gauss<double, 7>::integrate(...)`

Each level added more device lambdas and more call depth. This is acceptable on CPU, but on GPU, especially inside a large initialization kernel, it exhausted the per-thread device stack.

The fault site was not the parser. It was the nested tensor-product quadrature implementation.

### 7. The fix was to flatten the 3D quadrature

The working fix was:

- keep the 1D Gauss rule
- expose the Gauss abscissae and weights
- rewrite `quad_3d` as an explicit `7 x 7 x 7` tensor-product loop

This preserved the quadrature rule but removed the deep device call stack.

The quadrature rule remained unchanged, but the GPU execution shape changed.

### 8. Validation

After the quadrature change, with the parser code restored to its original state:

- `../build/3d-rwdi-cuda/src/problems/DiskGalaxy/DiskGalaxy ../inputs/DiskGalaxy.toml max_timesteps=0` completed initialization successfully
- `compute-sanitizer --tool memcheck ../build/3d-rwdi-cuda/src/problems/DiskGalaxy/DiskGalaxy ../inputs/DiskGalaxy.toml max_timesteps=0` finished with `ERROR SUMMARY: 0 errors`

The final evidence chain was:

- parser hypothesis tested and falsified
- stack-overflow diagnosis obtained from a device-aware tool
- quadrature implementation changed
- original input verified clean

## Lessons learned

### Plausible is not proven

The parser theory matched part of the evidence, but it was still wrong. The correct response to a plausible lead is not to trust it, but to test it quickly.

### Device-side tools matter

The host backtrace only established that something went wrong before a synchronization point. `compute-sanitizer` answered the more useful questions: what kind of device failure occurred, and where.

### CPU-only debugging can help, but it can also mislead

CPU builds are still valuable for:

- shrinking the problem
- getting symbolized stacks
- catching bounds errors under Debug or ASAN

However, once the CPU and GPU execution shapes differ meaningfully, CPU-side evidence must be treated carefully.

### GPU stack usage is a real failure mode

Not every CUDA 700 is a bad pointer or out-of-bounds array access. Deep call trees, nested lambdas, and heavily inlined tensor-product helper code can overflow the device stack.

## Suggested CUDA Debugging Playbook

The following order is recommended unless there is a strong reason to deviate from it.

### 1. Reproduce in the smallest realistic GPU configuration

- reduce to one MPI rank if possible
- reduce problem size while keeping the crash
- keep the same backend and major execution path

### 2. Build for diagnosis, not for maximum optimization

Prefer:

```bash
-DCMAKE_BUILD_TYPE=RelWithDebInfo
-DAMReX_GPU_BACKEND=CUDA
```

Make sure CUDA line info is present.

Do not assume `Debug` is the best first step for GPU crashes. It can perturb behavior enough to hide the bug or move it elsewhere.

Also verify that the configured build actually uses the intended GPU backend. A mistaken CPU-only configuration can produce a very polished but irrelevant stack trace.

### 3. Run outside any environment that breaks MPI or runtime setup

On this machine, sandboxed runs polluted the results with OpenMPI/PMIx failures. Remove environment noise before interpreting a crash.

### 4. Do one fast round of hypothesis-driven isolation

Disable suspicious optional features one at a time:

- runtime parser expressions
- optional source terms
- particle paths
- refinement logic

If the crash survives, drop that theory and move on.

### 5. Use a device-aware tool early

Prefer `compute-sanitizer` first for memory faults:

```bash
compute-sanitizer --tool memcheck <binary> <input>
```

Use `cuda-gdb` when you need:

- a stopped thread context
- manual breakpoints
- detailed stepping or device backtraces

### 6. Distinguish fault classes

Determine whether the failure is actually an illegal access, or one of the following:

- stack overflow
- race condition
- host/device lifetime bug
- asynchronous ordering bug
- invalid launch configuration

The fix depends on the class of failure.

### 7. Validate on the original input

Once a fix is found:

- test the smallest init-only case
- test the original input path
- rerun under `compute-sanitizer` if the original diagnosis came from it

### 8. Write down the false leads too

Future agents will move faster if they know:

- what looked plausible
- what was tested
- what was ruled out
- what tool finally broke the ambiguity

In practice, this information is often nearly as valuable as the code change itself.
