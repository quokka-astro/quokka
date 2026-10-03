// Standalone audit of the arithmetic used to bound exp(b)-1.
#include "multigroup_solver.hpp"
#include <iostream>
int main() {
  std::uint64_t b;
  while (std::cin >> b)
    std::cout << mgsolve::detail::rank(mgsolve::detail::relative_bound(
                     mgsolve::detail::unrank(b)))
              << "\n";
}
