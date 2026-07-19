# Glossary

Quick reference for recurring terminology across the Quokka documentation and reviews.

- **PR (Pull Request):** a proposed code change waiting for review.
- **Revert:** a follow-up change that undoes a previous merge to restore a known-good state.
- **ADR (Architecture Decision Record):** a short document capturing an important technical decision, options considered, and the rationale.
- **CI (Continuous Integration):** automated builds and tests that run on every change.
- **GPU backend (CUDA/HIP):** the selected GPU programming platform chosen at compile time.
- **MPI:** a standard for running a program across multiple processes (often across nodes).
- **Unit test:** a small test for a single function or module.
- **Physics test (domain test):** a small end-to-end run that checks scientifically meaningful results.
- **Regression test:** a test that ensures behavior hasn’t unintentionally changed compared to a known baseline.
- **Benchmark:** a timing or resource-usage measurement to track performance.
- **Debug build / Release build:** debug enables extra checks and slower safety features; release is optimized for speed.
- **Stream (GPU):** an ordered queue of GPU work; operations in a stream execute in sequence.
- **Seed (random):** a number used to initialize a random number generator so results can be repeated.
- **Tolerance (absolute/relative):** acceptable numerical error bounds (e.g., max difference allowed).
- **Baseline:** a trusted reference result used for comparison.
