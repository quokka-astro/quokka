// Independent auditor's driver for the three inner collision charts.
#include "../../src/radiation/nested/multigroup_solver.hpp"
#include <cmath>

#include <iomanip>
#include <iostream>
auto main() -> int {
  double A = NAN;
  double D = NAN;
  double T = NAN;
  double x = NAN;
  int newton = 0;
  std::cout << std::setprecision(17);
  while (std::cin >> A >> D >> T >> x >> newton) {
    mgsolve::Options opt;
    opt.max_inner = 512;
    opt.max_bracket = 4096;
    opt.use_newton = (newton != 0);
    auto const r = mgsolve::detail::inner(A, D, T, x, opt);
    std::cout << mgsolve::status_name(r.status) << " " << r.t << " " << r.q
              << " " << r.balance << " " << r.iterations << "\n";
  }
}
