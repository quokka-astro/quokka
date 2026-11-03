# Vendored yaml-cpp

This directory is a trimmed, vendored copy of [yaml-cpp](https://github.com/jbeder/yaml-cpp),
replacing the former git submodule.

- Upstream commit: `1e0876c671268661deb2628040e3959e1e9d6e69` (`yaml-cpp-0.9.0-74-g1e0876c`, 2026-09-17)
- Every file kept from upstream is unmodified, except:
  - `CMakeLists.txt`: a minimal replacement that builds a static `yaml-cpp::yaml-cpp` target.
  - `src/fptostring.cpp`: `jkj::dragonbox::to_decimal` is replaced by a small `ToDecimal` helper built on
    C++17 `std::to_chars`, so `src/contrib/dragonbox.h` (4200 lines) is not vendored. The rest of the
    formatting logic is unchanged, and output was verified byte-identical to upstream on ~13M values.
    Reapply this change when updating.
- Removed from upstream: tests, docs, tools (`util/`), `src/contrib/` (graph builder, dragonbox), `Clone()` (`src/node.cpp`),
  install/packaging CMake, Bazel files, and repository metadata.
- Quokka needs both the parser (reading inputs) and the emitter (writing `metadata.yaml`),
  so both are kept.

To update, copy the same set of files from a newer upstream commit, add any new `src/*.cpp`
dependencies they require, and update the hash above.
