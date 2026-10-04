#include "../../src/radiation/nested/multigroup_solver.hpp"
#include <random>
#include <vector>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
template <std::size_t N> void check(std::mt19937_64 &rng) {
  for (int trial = 0; trial < 100; ++trial) {
    mgsolve::Array<double, N> a;
    std::vector<double> ref(N);
    for (std::size_t i = 0; i < N; ++i) {
      a[i] = ref[i] =
          (rng() % 5 == 0)
              ? 0
              : std::ldexp(1.0 + static_cast<double>(rng() % 65536) / 65536,
                           static_cast<int>(rng() % 401) - 200);
    }
    while (ref.size() > 1) {
      std::vector<double> next;
      next.reserve((ref.size() + 1) / 2);
      auto it = ref.cbegin();
      while (it != ref.cend()) {
        double value = *it++;
        if (it != ref.cend()) {
          value += *it++;
        }
        next.push_back(value);
      }
      ref = std::move(next);
    }
    mgsolve::detail::Arithmetic arithmetic;
    double const value = mgsolve::detail::sum(arithmetic, a);
    assert(arithmetic.ok);
    assert(mgsolve::detail::rank(value) ==
           mgsolve::detail::rank(N ? ref[0] : 0));
  }
}
auto main() -> int try {
  // A fixed seed makes the numerical audit reproducible.
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
  std::mt19937_64 rng(3402326);
  check<0>(rng);
  check<1>(rng);
  check<2>(rng);
  check<3>(rng);
  check<5>(rng);
  check<7>(rng);
  check<15>(rng);
  check<31>(rng);
  check<63>(rng);
  check<127>(rng);
  check<255>(rng);
  check<511>(rng);
  check<1023>(rng);
  check<1024>(rng);
  std::cout << "1400 bitwise reduction comparisons passed\n";
}

catch (std::exception const &e) {
  std::cerr << e.what() << "\n";
  return 1;
}
