# Nested radiation solver accuracy tests

Configure this standalone CTest project from the repository root:

```sh
python3 -m pip install mpmath
cmake -S tests/radiation_coupling -B build/coupling-kernel
cmake --build build/coupling-kernel
ctest --test-dir build/coupling-kernel --output-on-failure
```

`mpmath` is used only by offline tests; the C++ kernel uses double precision.
The suite checks 79 coupled cases against independent high-precision gas-space
roots, 720 inner inversions, 10016 exact-rational certificate comparisons,
a width-return case, and 1400 bitwise reduction comparisons. The unit executable
also checks missing contracts, structural zeros, range failures, bad derivative
proposals, and iteration/tolerance limits. Assertions stay enabled in Release.
Reports are generated in the build directory.

The separate `NestedRadCoupling` AMReX problem tests both live source-update
entry points. It includes radiation source injection and RSLA/collision scaling
for manufactured heating, cooling, and equilibrium states. Its model coefficients
and bands are analytic; it does not certify the production Planck table.
