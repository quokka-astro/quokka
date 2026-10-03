// Pure-binary64 full-frequency Planck model that reaches width acceptance.
#include "../../src/radiation/nested/multigroup_solver.hpp"
#include <iomanip>
#include <iostream>
int main() {
  mgsolve::Problem<1> p;
  p.h = 0x1p-40;
  p.r[0] = 2;
  mgsolve::Contract<1> c;
  c.certified = c.root_in_domain = true;
  c.sensitivity[0] = 4;
  mgsolve::Options o;
  o.x_min = .5;
  o.x_max = 2;
  o.relative_tolerance = 2e-14;
  o.allow_residual = false;
  auto oracle = [](double x, mgsolve::GroupValues<1> &v) {
    v.alpha[0] = v.p[0] = 1;
    double x2 = x * x;
    v.B[0] = x2 * x2;
    return true;
  };
  auto r = mgsolve::solve(p, oracle, c, o);
  std::cout << std::setprecision(17) << mgsolve::status_name(r.status) << " "
            << int(r.stop) << " " << r.dust_temperature << " " << r.gas_energy
            << " " << r.radiation[0] << " " << r.bracket_lo << " "
            << r.bracket_hi << " " << r.certificate.dust_relative << " "
            << r.certificate.gas_relative << " "
            << r.certificate.group_relative[0] << "\n";
}
