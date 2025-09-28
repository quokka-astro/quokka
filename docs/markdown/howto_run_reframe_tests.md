# Running ReFrame tests

ReFrame provides a portable way to build and execute Quokka performance and
scaling tests across different systems. This guide walks through the workflow
for running the bundled HydroBlast3D weak-scaling benchmark.

## Prerequisites

- A working build environment for Quokka (CMake, Ninja, MPI, GPU toolchain if
  required). See the [installation guide](installation.md) for general
  dependencies.
- Python 3.8 or newer with the `reframe-hpc` package. Install via pip if your
  system does not provide a module:

  ```bash
  python -m pip install --user reframe-hpc
  ```

- A ReFrame site configuration describing the partitions and programming
  environments that should be used for the test run. Site configuration files
  can live in `$HOME/.reframe/site.py` or you can point ReFrame to a custom file
  with `-C <config.py>`.

  A minimal configuration for a generic Slurm+CUDA system might look like:

  ```python
  from reframe.core.backends import getlauncher

  site_configuration = {
      'systems': [
          {
              'name': 'slurm-cuda',
              'descr': 'Generic GPU cluster',
              'hostnames': ['login'],
              'scheduler': 'slurm',
              'partitions': [
                  {
                      'name': 'gpu',
                      'launcher': 'srun',
                      'environs': ['cuda-nvhpc'],
                      'access': ['--partition=gpu'],
                      'resources': [
                          {
                              'name': 'gres',
                              'options': ['--gres=gpu:4'],
                          }
                      ],
                  }
              ],
          }
      ],
      'environments': [
          {
              'name': 'cuda-nvhpc',
              'modules': ['nvhpc', 'cuda'],
          }
      ],
  }
  ```

  Adjust the partition name, modules, and launcher to match the target machine.

## Running the HydroBlast3D weak-scaling test

1. Clone Quokka and move into the repository:

   ```bash
   git clone https://github.com/quokka-astro/quokka.git
   cd quokka
   ```

2. (Optional) Set the GPU backend if it cannot be inferred from the machine's
   features. Valid values are `CUDA` and `HIP`:

   ```bash
   export QUOKKA_GPU_BACKEND=CUDA
   ```

3. Launch ReFrame, pointing it at the tests under `scripts/reframe/` and pass in
   your site configuration. The example below runs the entire parameter sweep,
   keeps build and run output under `reframe_stage/` and `reframe_runs/`, and
   produces a performance summary once finished:

   ```bash
   reframe \
       -C path/to/site.py \
       -c scripts/reframe \
       -r --performance-report
   ```

   Use `-n <pattern>` to select a subset of scales (e.g. `-n n1_256`) or
   `--run-report` to export the results as JSON for later analysis.

4. ReFrame will configure and build Quokka inside `build/`, then submit the job
   to the scheduler for each scale defined in
   `scripts/reframe/hydroblast3d_weak_scaling.py`. Output logs are placed under
   `reframe_logs/`, while job output and error files are written to
   `reframe_runs/<test-name>/`.

## Troubleshooting tips

- If ReFrame reports that it cannot determine the GPU backend, set
  `QUOKKA_GPU_BACKEND` explicitly.
- Ensure the requested number of nodes and GPUs is available on the target
  system. The default configuration will request up to 512 nodes for the largest
  scale; use `-n` to restrict the run or edit `_SCALES_CONFIG` in the test file.
- If the build stage needs additional CMake flags (e.g., enabling MPI backends),
  modify the `config_opts` list in the
  `scripts/reframe/hydroblast3d_weak_scaling.py` test definition or override them
  via the `--override` option when invoking ReFrame.

For more information on ReFrame usage, consult the
[official documentation](https://reframe-hpc.readthedocs.io/).
