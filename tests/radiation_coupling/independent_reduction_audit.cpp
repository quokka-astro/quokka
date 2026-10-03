#include "multigroup_solver.hpp"
#include <random>
#include <vector>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
std::mt19937_64 rng(3402326);
template <std::size_t N> void check() {
  for (int trial = 0; trial < 100; ++trial) {
    mgsolve::Array<double, N> a;
    std::vector<double> ref(N);
    for (std::size_t i = 0; i < N; ++i)
      a[i] = ref[i] = (rng() % 5 == 0)
                          ? 0
                          : std::ldexp(1.0 + double(rng() % 65536) / 65536,
                                       int(rng() % 401) - 200);
    for (std::size_t n = N; n > 1;) {
      std::size_t m = 0;
      for (std::size_t i = 0; i < n; i += 2)
        ref[m++] = (i + 1 < n) ? ref[i] + ref[i + 1] : ref[i];
      n = m;
    }
    mgsolve::detail::Arithmetic arithmetic;
    double value = mgsolve::detail::sum(arithmetic, a);
    assert(arithmetic.ok);
    assert(mgsolve::detail::rank(value) ==
           mgsolve::detail::rank(N ? ref[0] : 0));
  }
}
int main() {
  check<0>();
  check<1>();
  check<2>();
  check<3>();
  check<5>();
  check<7>();
  check<15>();
  check<31>();
  check<63>();
  check<127>();
  check<255>();
  check<511>();
  check<1023>();
  check<1024>();
  std::cout << "1400 bitwise reduction comparisons passed\n";
}
