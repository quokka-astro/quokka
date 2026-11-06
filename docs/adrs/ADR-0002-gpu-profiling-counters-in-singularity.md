# ADR-0002: GPU Performance Counters for `ncu` in a Singularity Container
Date: 2026-11-05 • Status: Proposed (deferred; not implemented)

## Context

Nsight Compute (`ncu`) needs GPU hardware performance counters to show whether a kernel is limited by memory bandwidth, latency or occupancy. Many shared GPU hosts restrict these counters to administrators. When Quokka is built and run inside a Singularity container on such a host, `ncu` then fails with `ERR_NVGPUCTRPERM`.

We hit this while we optimized the photochemistry module of PR #2429 (Rosenbrock solver) on a Tesla V100. The host is `avatargpu` at ANU. Development agents run there inside a persistent Singularity instance that a launch script starts with `singularity instance start --nv ...`. Without counters, profiling is limited to:
- registers, stack and local-memory traffic, from `cuobjdump` and SASS;
- device-side instrumentation, for example solver step counts and step divergence within a warp;
- timing A/B variants against each other.

The cause has three parts:

1. **The driver.** The NVIDIA driver runs with `RmProfilingAdminOnly: 1` (see `/proc/driver/nvidia/params`). This is the default `NVreg_RestrictProfilingToAdminUsers=1`. With this setting, only processes that have `CAP_SYS_ADMIN` in the host's initial user namespace can read the counters.
2. **The container mode.** On this host the instance uses Singularity's setuid starter, not `--userns`, because `--nv` through `nvidia-container-cli` refuses a user namespace. In setuid mode, a capability added to the container is a real host capability. Under `--userns` it would be confined to the container's user namespace, and the driver would ignore it.
3. **Singularity's capability rules.** Singularity drops every capability for a non-root user unless root has granted it with `singularity capability add`. On this host nothing is granted: `capability.json` is empty.

The change touches no `ParmParse` options, no output format and no problem-file interface. It concerns only the development and profiling environment.

## Options

- **Option A: change the driver option.** Set `options nvidia NVreg_RestrictProfilingToAdminUsers=0`, for example in `/etc/modprobe.d/nvidia-profiling.conf`, and reload the `nvidia` kernel module or reboot.
  - Every user on the host can read GPU counters. The change gives no other privilege, and the container needs no change.
  - Risk: a minor side channel between GPU users.
  - Cost: root access and a GPU maintenance window, because all GPU jobs must stop for the module reload.
- **Option B: give `CAP_SYS_ADMIN` to the whole container instance.** Root grants the capability to the user once, and the launch script adds `--add-caps CAP_SYS_ADMIN` when it starts the instance (see "How to do Option B" below).
  - This is the most convenient option.
  - Every process in the instance gets near-root power on a shared host. That includes development agents that run without permission prompts.
- **Option C: give `CAP_SYS_ADMIN` only to one-off profiling runs.** Root grants the capability once, as in Option B. Then a person, not an agent, runs each profile in a separate short-lived container, for example `singularity exec --nv --add-caps CAP_SYS_ADMIN --bind <workspace>:<container workspace> <image>.sif ncu <args> <binary> <inputs>`. The build directories live in the shared workspace, so this container sees the same executables. The persistent instance and its processes are not affected.
- **Option D: no counters.** Profile with the counter-free methods listed in Context.

## Decision

Deferred. Option D stays in place for PR #2429. If counters become necessary, the preferred order is A, then C, then B:
- Option A gives no privilege beyond counter access.
- Option C keeps root-equivalent power away from long-running and autonomous processes.
- Option B is the most convenient, but it gives near-root power on a shared host to everything in the instance.

### How to do Option B

1. As root on the host, grant the capability to the user once:
   ```bash
   sudo singularity capability add --user <user> CAP_SYS_ADMIN
   ```
2. In the launch script, request the capability when the instance starts, only on the setuid path. For example, next to the existing `--userns` handling:
   ```bash
   if [[ "${USE_USERNS}" -eq 0 ]]; then
     start_args+=(--add-caps CAP_SYS_ADMIN)
   fi
   ```
3. Restart the instance. This stops every process in the instance.
4. Check from inside the instance:
   - `grep CapEff /proc/self/status` shows a mask that includes bit 21 (`CAP_SYS_ADMIN`);
   - a short `ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed <binary>` run prints values instead of `ERR_NVGPUCTRPERM`.
5. To undo: remove the lines from step 2, restart the instance, and run `sudo singularity capability drop --user <user> CAP_SYS_ADMIN`.

## Consequences

- With Option B, every process in the instance can mount file systems, change kernel parameters it is permitted to change, and use other admin system calls on the host. A mistake by any of them can then affect the whole host and its other users.
- Options A and C need no change to the container setup. Option A needs a GPU maintenance window; Option C needs a person to run each profile.
- Option D makes GPU tuning slower, because a bottleneck must be inferred from SASS, instrumentation and timing instead of measured. It does not affect correctness.
- None of the options affects Quokka's code, CI, accuracy or portability.

## Rollout & Testing

Nothing to roll out until an option is chosen. After Option A, B or C, verify with the `ncu` command in step 4 above, and record the counter-based profile of the photochemistry kernel in PR #2429 or a follow-up.

## Links

- PR #2429: photochemistry without Microphysics (Rosenbrock solver), where the need arose.
- NVIDIA, "ERR_NVGPUCTRPERM": https://developer.nvidia.com/ERR_NVGPUCTRPERM
- SingularityCE documentation, "Security options" (`--add-caps`, `singularity capability`).
