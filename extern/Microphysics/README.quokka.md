# Microphysics subset vendored for Quokka

Source: https://github.com/AMReX-Astro/Microphysics.git
Upstream revision: `59c48535461d1310efcf3fbf4970e4c19e1eac64`
(the former Quokka submodule revision, also recorded in `UPSTREAM_REVISION`).

This directory contains ordinary files tracked by Quokka. It retains the
Microphysics dependencies of Quokka's gamma-law EOS, primordial chemistry,
and photoionization configurations, with Rosenbrock as the only ODE backend.
Photoionization network sources live in Quokka's `src/networks/photoionization`.

The subset was selected by compiler dependency tracing of Quokka's compilation
units, supplemented with the CMake parameter files, network descriptions,
templates, and Python generators used at configure time. GCEM's umbrella header
is modified to include only `pow`, `sqrt`, `exp`, and their transitive helpers
(18 headers total); unused GCEM headers are omitted. The retained
GCEM helper implementations are unchanged. The network generator no longer adds
auxiliary thermodynamics variables. Shared integration
code is adapted to remove obsolete VODE cleanup multipliers and runtime parameters;
comments now describe the Rosenbrock backend. Unused Nonaka, NSE, and nuclear-rate
parameters and the empty EOS override hook are removed. Disabled SDC, NSE, Nonaka,
auxiliary-thermodynamics, alternate nuclear-network, reaction-initialization, and
upstream conductivity-output branches are removed from the retained C++ sources.
Strang integration is unconditional. The active numerical operations are unchanged;
CPU/GPU, photochemistry/flux, auxiliary-species, and Jacobian-precision branches
remain available. Uncalled EOS/burn conversion and composition-derivative helpers,
empty network initialization hooks, unused charge balancing, and obsolete error
codes are removed. The burn state omits unused SDC storage and its diagnostic
output and initialization. Unused EOS structure variants, one-dimensional array
and Jacobian helpers, radiation-flux tolerance settings and storage, and the
unused diagnostic backup of `c_hat` are also removed. Flux evolution is retained;
Rosenbrock excludes flux from its error norm.

`CMakeLists.txt` fixes the backend to Rosenbrock and removes unused metal chemistry,
standalone AMReX fetching, upstream unit-test builds, unused thread/package helpers,
and the unused auxiliary-species counting script. Other EOS/network
implementations, VODE and other integrators, SDC/NSE paths, tests, documentation,
and development tools are omitted. Those upstream configurations are not supported by this subset.

Keep `license.txt`, `CITATION.md`, and GCEM's `util/gcem/LICENSE` and
`util/gcem/NOTICE.txt` with the sources. When updating, use the upstream revision,
retrace dependencies for all three configurations and the Rosenbrock tableaux, and
update `UPSTREAM_REVISION`; Quokka reads it for runtime build provenance.
