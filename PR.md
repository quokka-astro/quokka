## Fix HIP build failure on Setonix due to GCC 15 libstdc++ incompatibility

GCC 15's libstdc++ added hardened assertions to `std::array::operator[]` that call the host-only function `__glibcxx_assert_fail`. This causes a compilation error when hipcc compiles AMReX device code that uses `std::array`:

```
error: reference to __host__ function '__glibcxx_assert_fail' in __host__ __device__ function
```

Fix: pin hipcc to GCC 14's libstdc++ headers via `--gcc-install-dir=/usr/lib64/gcc/x86_64-suse-linux/14` in `CFLAGS`/`CXXFLAGS`.
