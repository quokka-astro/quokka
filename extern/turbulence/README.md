# TurbGen subset for Quokka

Vendored from https://github.com/chfeder/turbulence_generator at commit
`6a7eee7c3d1c09dd2d9366b225a8ae7a9f98dd73`.

Quokka uses the header-only generator in `TurbGen.h` through the AMReX adapter
in `plugins/AMReX/TurbGenEx.h`. The adjacent `CMakeLists.txt` exposes the
`turbulence::turbulence` interface target. Quokka's driving implementation lives
in `src/turbulence/TurbulentDriving.hpp` in the parent repository.

Only these build dependencies, the MIT license, and upstream citation metadata
are retained. Standalone generators, demos, legacy implementations, other code
plugins, Python tools, and media are omitted. The retained source files are
unchanged from the vendored revision.

See `LICENSE` for copyright and license terms and `CITATION.cff` for citations.
