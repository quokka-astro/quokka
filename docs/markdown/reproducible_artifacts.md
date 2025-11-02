# Reproducible build artifacts

This guide explains how Quokka’s reproducible builds work end-to-end: building devcontainer images, producing verified install tarballs, and downloading them locally. Follow these steps whenever you need to refresh or consume nightly artifacts.

## 1. Build the devcontainer images

We publish CUDA, ROCm, and GCC devcontainer images to GitHub Container Registry (GHCR). Rebuild them on demand:

1. Open the [GitHub Actions](https://github.com/quokka-astro/quokka/actions) tab.
2. Run **Build Devcontainer Images** (`.github/workflows/build-devcontainers.yml`) via *Run workflow*.
3. Confirm each matrix job pushes `ghcr.io/quokka-astro/quokka-linux-amd64-{cuda,rocm,gcc}:development`. The step summary lists digests for auditing.

These images pin compilers, toolchains, and dependencies so subsequent builds are deterministic.

## 2. Produce reproducible artifacts

With fresh images available, run **Build Reproducible Artifacts** (`.github/workflows/build-reproducible-artifacts.yml`):

1. Navigate to the workflow page and choose *Run workflow*.
2. Each job pulls the matching devcontainer, configures CMake in Release mode, and executes two clean builds/installations.
3. The workflow compares SHA-256 manifests from both runs; it fails if any file differs.
4. On success, it uploads an artifact (`quokka-{cuda,rocm,gcc}-reproducible-build`) containing:
   - `repro-*-install.tar`: the deterministic install tree.
   - `repro-*-build{1,2}.sha256`: file digests from both runs.
   - `repro-*-manifest.txt`: metadata (commit hash, image digest, tarball hash).
5. A GitHub Artifact Attestation is generated for each artifact, binding the output to this workflow run via Sigstore. The job summary shows the command to download it.

Artifacts currently follow Actions’ default retention (90 days). For long-term archival, promote them to GHCR packages or releases.

## 3. Download artifacts locally

Use `scripts/download_gha_artifact.py` to grab the latest artifact from a public workflow. First-time setup:

1. Create a GitHub Personal Access Token (PAT) with *Actions → Read* scope:
   - GitHub → Settings → Developer settings → Personal access tokens → Generate new token.
   - Assign a name, expiry, and repository access (this repo or all repos).
   - Grant **Actions: Read** permission and generate the token.
2. Export the token:

   ```bash
   export GITHUB_TOKEN=<your-token>
   ```

Download the artifact:

```bash
python3 scripts/download_gha_artifact.py \
  --repo quokka-astro/quokka \
  --workflow build-reproducible-artifacts.yml \
  --branch development \
  --artifact quokka-cuda-reproducible-build \
  --output ./artifacts
```

The script selects the newest successful run (unless `--run-id` is provided), downloads the artifact ZIP, and extracts it into `./artifacts/quokka-cuda-reproducible-build`. Repeat with `--artifact` set to `quokka-rocm-reproducible-build` or `quokka-gcc-reproducible-build` as needed.

## 4. Verify hashes and install

Inside the extracted artifact:

```bash
cd artifacts/quokka-cuda-reproducible-build
sha256sum --check repro-cuda-build1.sha256
sha256sum --check repro-cuda-build2.sha256
sha256sum repro-cuda-install.tar
```

Compare the reported hashes with the `repro-*-manifest.txt` values (or the GitHub summary). When satisfied, unpack the tarball into your desired prefix:

```bash
tar -xf repro-cuda-install.tar -C /path/to/install
```

To pull the OIDC-backed attestation that GitHub issued during the workflow, run:

```bash
gh attestation download \
  --repository quokka-astro/quokka \
  --workflow build-reproducible-artifacts.yml \
  --artifact quokka-cuda-reproducible-build \
  --run <run-id> \
  --output attestation-cuda.json
```

Replace `<run-id>` with the Actions run number (also printed in the job summary). You can then verify it with `gh attestation verify --predicate attestation-cuda.json`.

## 5. Tips and troubleshooting

- If the workflow fails, inspect the “Compare build manifests” step to see which file differs. Often this indicates a nondeterministic dependency; update the Dockerfile/CMake flags accordingly.
- Before running the artifact workflow, ensure the corresponding devcontainer image exists (step summary will reference the GHCR digest).
- Artifacts expire after 90 days. Consider promoting canonical releases to GHCR images and publishing tarballs alongside GitHub releases for long-term availability.
- To script downloads for a specific run, pass `--run-id <run-number>` to `download_gha_artifact.py`.

With these steps automated, everyone can reproduce Quokka binaries exactly from CI, verify their provenance, and install them locally without re-running the entire build toolchain.
