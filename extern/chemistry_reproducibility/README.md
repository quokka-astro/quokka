# Chemistry reproducibility references

The `ChemistryReproducibility` CI workflow (`.github/workflows/chemistry-reproducibility.yml`)
runs every CHEMISTRY and PHOTOCHEMISTRY test problem for 10 timesteps on one MPI rank (the
problems that are 64^3 under ctest run on a 16^3 grid) and requires the result to be bitwise
identical to the references in this directory. It is a guard against unintended changes to the
numerics: any change that moves a single bit of any of these problems fails it.

| File | Contents |
| --- | --- |
| `<case>.txt` | Fingerprint of one case: the exit status of the run and, for the plotfiles at steps 0 and 10, the SHA-256 of every file plus the SHA-256, min, max and sum of every variable on every level |
| `toolchain.txt` | Compiler, libc, architecture, CMake flags and Python packages the references were made with |
| `MANIFEST` | The commit the references fingerprint, its Microphysics and AMReX pins, the CI run and its CPU, and the exit status and run time of each case |
| `PopIII_zdrv.hdf5` | PopIII's initial velocity perturbations. `src/turbulence/perturbation.py` writes them when the references are made, and the check reuses this file rather than running the script, because numpy's float64 power gives different last bits on CPUs with and without AVX-512 |

The cases, and how each one is run, are listed in `scripts/bash/chemistry_reproducibility_test.sh`.
The fingerprints are written by `scripts/python/plotfile_hashes.py`.

## When the check fails

The job log shows a diff of the fingerprint against the reference. The `var` lines name the
level and variable that changed, and their min, max and sum show by how much. If the job also
prints a toolchain warning, the runner image has changed the compiler or libc, and the
references must be regenerated even though the code has not changed.

To run the check locally (it only passes on the toolchain in `toolchain.txt`):

```sh
scripts/bash/chemistry_reproducibility_test.sh --build-dir <build dir> [--cases "PopIII DTypeFront1D"]
```

## Regenerating the references

Do this only when a change to the results is intended, or when the toolchain changes.
References have to be made by the CI runner itself, because results differ between
compilers, libc versions and architectures.

1. Choose the source to fingerprint, `<source>`.
2. Make a commit on top of `<source>` that adds only the reproducibility CI files: the
   workflow, `scripts/bash/chemistry_reproducibility_test.sh`,
   `scripts/python/plotfile_hashes.py` and this directory. The generate job checks this and
   refuses anything else.
3. Push it to the branch `ci/chem-repro-generate`, force-pushing if the branch already exists.
   Only the generate job runs. It builds the cases, runs every case once, and uploads the
   `chemistry-reproducibility-references` artifact.
4. Copy the artifact's contents into this directory and commit. `*.hdf5` is git-ignored, so
   add `PopIII_zdrv.hdf5` with `git add -f`.
