#include "../../src/radiation/nested/multigroup_solver.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cfenv>
#include <iostream>
#include <limits>
using mgsolve::accepted;
namespace detail = mgsolve::detail;
using mgsolve::Array;
using mgsolve::Contract;
using mgsolve::GroupValues;
using mgsolve::infinity;
using mgsolve::min_normal;
using mgsolve::Options;
using mgsolve::Problem;
using mgsolve::solve;
using mgsolve::Status;
using mgsolve::status_name;
using mgsolve::Stop;
using mgsolve::u;
struct TwoGroups {
  bool derivatives = true;
  bool poison = false;
  auto operator()(double x, GroupValues<2> &v) const -> bool {
    const double x2 = x * x;
    const double x4 = x2 * x2;
    for (int g = 0; g < 2; ++g) {
      v.alpha[g] = g == 0 ? 1 : 4;
      v.p[g] = v.alpha[g];
      v.B[g] = (g == 0 ? .25 : .75) * x4;
      v.d_B[g] = 4 * v.B[g] / x;
    }
    v.derivatives = derivatives;
    if (poison) {
      v.d_B[0] = std::numeric_limits<double>::quiet_NaN();
    }
    return true;
  }
};
auto contract() -> Contract<2> {
  Contract<2> c;
  c.certified = true;
  c.root_in_domain = true;
  c.sensitivity[0] = c.sensitivity[1] = 4;
  return c;
}
auto main() -> int {
  assert(std::fegetround() == FE_TONEAREST);
  Problem<2> p;
  p.r[0] = .5;
  p.r[1] = 1.5;
  Options o;
  o.x_min = .001;
  o.x_max = 100;
  auto c = contract();
  TwoGroups const oracle;
  auto r = solve(p, oracle, c, o);
  if (!accepted(r.status)) {
    std::cerr << status_name(r.status) << "\n";
  }
  assert(accepted(r.status));
  assert(r.dust_temperature > 1 && r.gas_temperature > 1);
  assert(r.radiation[0] > 0 && r.radiation[1] > 0);
  assert(r.certificate.conditional);
  assert(r.certificate.gas_relative <= o.relative_tolerance);
  std::feclearexcept(FE_DIVBYZERO);
  auto const derivative_free =
      solve(p, TwoGroups{.derivatives = false, .poison = false}, c, o);
  assert(std::fetestexcept(FE_DIVBYZERO) == 0);
  Options bisection = o;
  bisection.use_newton = false;
  std::feclearexcept(FE_DIVBYZERO);
  auto const derivative_free_bisection =
      solve(p, TwoGroups{.derivatives = false, .poison = false}, c, bisection);
  assert(std::fetestexcept(FE_DIVBYZERO) == 0);
  assert(accepted(derivative_free_bisection.status));
  assert(std::abs(derivative_free_bisection.gas_energy - r.gas_energy) < 1e-12);
  auto const poisoned =
      solve(p, TwoGroups{.derivatives = true, .poison = true}, c, o);
  assert(accepted(derivative_free.status) && accepted(poisoned.status));
  assert(std::abs(derivative_free.gas_energy - r.gas_energy) < 1e-12);
  c.certified = false;
  assert(solve(p, TwoGroups{}, c, o).status == Status::missing_contract);
  o.allow_estimated = true;
  assert(solve(p, TwoGroups{}, c, o).status == Status::accepted_estimated);
  Options blocked = o;
  blocked.x_min = .99;
  blocked.x_max = 1.01;
  assert(solve(p, TwoGroups{}, c, blocked).status == Status::no_bracket);
  c = contract();
  o.allow_estimated = false;
  o.relative_tolerance = 1e-20;
  assert(solve(p, TwoGroups{}, c, o).status == Status::tolerance_unavailable);
  o.relative_tolerance = 1e-12;
  p.D = 0;
  assert(solve(p, TwoGroups{}, c, o).status == Status::invalid_input);
  p.D = 1;
  p.r[0] = -1;
  assert(solve(p, TwoGroups{}, c, o).status == Status::invalid_input);
  p.r[0] = .5;
  c.band_log_error = 9 * u;
  assert(solve(p, TwoGroups{}, c, o).status == Status::missing_contract);
  c = contract();
  Options huge_budget = o;
  huge_budget.max_outer = std::numeric_limits<int>::max();
  assert(solve(p, TwoGroups{}, c, huge_budget).status == Status::invalid_input);
  c.root_in_domain = false;
  assert(solve(p, TwoGroups{}, c, o).status == Status::missing_contract);
  c = contract();
  std::fesetround(FE_UPWARD);
  assert(solve(p, TwoGroups{}, c, o).status == Status::unsupported_model);
  std::fesetround(FE_TONEAREST);
  // Net zero does not imply unchanged radiation: a dyadic equal-weight
  // exchange.
  auto const netzero = [](double x, GroupValues<2> &v) {
    for (int g = 0; g < 2; ++g) {
      v.alpha[g] = 1;
      v.p[g] = 1;
      v.B[g] = x * x * x * x;
    }
    return true;
  };
  p.r[0] = .25;
  p.r[1] = 1.75;
  auto z = solve(p, netzero, c, o);
  assert(accepted(z.status));
  assert(z.gas_energy == 1 && z.dust_temperature == 1);
  assert(z.radiation[0] == .625 && z.radiation[1] == 1.375);
  // Every hidden underflow must fail, not silently become structural zero.
  auto const badzero = [](double, GroupValues<2> &v) {
    for (int g = 0; g < 2; ++g) {
      v.alpha[g] = 1;
      v.p[g] = 1;
      v.B[g] = 0;
    }
    return true;
  };
  assert(solve(p, badzero, c, o).status == Status::oracle_failure);
  auto const underflow = [](double, GroupValues<2> &v) {
    for (int g = 0; g < 2; ++g) {
      v.alpha[g] = 1;
      v.p[g] = min_normal;
      v.B[g] = min_normal;
    }
    return true;
  };
  assert(solve(p, underflow, c, o).status == Status::range_failure);
  // Width graph accepts exact subnormal Sterbenz gaps without allowing them
  // into positive relative-error primitive nodes.
  assert(detail::narrow(min_normal, std::nextafter(min_normal, infinity), 16));
  // Odd balanced tree includes each group exactly once, with no padding copies.
  detail::Arithmetic ar;
  Array<double, 5> terms;
  for (int i = 0; i < 5; ++i) {
    terms[i] = i + 1;
  }
  assert(detail::sum(ar, terms) == 15 && ar.ok);
  // Compare standalone inner solve against substitution, all three charts.
  for (double const x : {.0001, .1, .9, 1., 1.1, 4., 1e8}) {
    auto const inner = detail::inner(1., 10., 1., x, o);
    assert(accepted(inner.status));
    double const collision =
        inner.t - 1 + 10 * std::sqrt(inner.t) * (inner.t - x);
    assert(std::abs(collision) <=
           1e-12 * std::max(1., inner.t * std::sqrt(inner.t)));
  }
  // Exercise the genuine certified-width path with a tolerance tighter than
  // the residual certificate. Thin coupling leaves a narrow root near T.
  Problem<1> thin;
  thin.r[0] = 2;
  thin.h = 0x1p-40;
  Contract<1> tc;
  tc.certified = true;
  tc.root_in_domain = true;
  tc.sensitivity[0] = 4;
  auto const to = [](double x, GroupValues<1> &v) {
    v.alpha[0] = v.p[0] = 1;
    double const x2 = x * x;
    v.B[0] = x2 * x2;
    return true;
  };
  Options wo;
  wo.x_min = .5;
  wo.x_max = 2;
  wo.relative_tolerance = 2e-14;
  wo.allow_residual = false;
  auto const width = solve(thin, to, tc, wo);
  assert(accepted(width.status) && width.stop == Stop::width);
  assert(width.bracket_lo <= width.dust_temperature &&
         width.dust_temperature <= width.bracket_hi);
  // Direct absorption must not consume any iterative collision budget.
  Problem<1> absorption;
  absorption.r[0] = 3;
  Contract<1> ac;
  ac.certified = true;
  ac.root_in_domain = true;
  ac.emission_identically_zero = true;
  auto const ao = [](double, GroupValues<1> &v) {
    v.alpha[0] = 1;
    v.p_zero[0] = true;
    v.B_zero[0] = true;
    return true;
  };
  Options direct_opt;
  direct_opt.max_inner = 1;
  auto const direct = solve(absorption, ao, ac, direct_opt);
  assert(accepted(direct.status) && direct.stop == Stop::direct_absorption &&
         direct.inner_iterations == 0);
  Problem<0> empty;
  empty.A = 2;
  empty.T = 3;
  Contract<0> ec;
  ec.certified = true;
  ec.root_in_domain = true;
  auto const eo = [](double, GroupValues<0> &) { return true; };
  auto const er = solve(empty, eo, ec);
  assert(accepted(er.status) && er.stop == Stop::zero_exchange &&
         er.gas_energy == 6 && er.dust_temperature == 3);
  std::cout << "All unit tests passed\n";
}
