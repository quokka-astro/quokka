#include "../../src/radiation/nested/multigroup_solver.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct GroupInput {
  double r, alpha, p, b;
  int alpha_quarters, p_quarters, beta;
  double L;
};
struct Input {
  std::string name;
  int N = 0;
  double A{}, D{}, T{}, h{}, chi{}, xmin{}, xmax{}, tol{}, margin{};
  int certified{}, variable{}, residual{}, width{}, max_outer{}, max_inner{},
      max_bracket{}, newton{}, derivative_mode{}, root_in_domain{};
  std::vector<GroupInput> groups;
};

// libc++ formatted extraction sets failbit for representable subnormals.
// strtod preserves these deliberate invalid-input cases so the solver can
// reject them.
struct ReadDouble {
  // This extraction proxy deliberately writes to its caller-owned value.
  double &value; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};
auto operator>>(std::istream &stream, ReadDouble number) -> std::istream & {
  std::string token;
  if (!(stream >> token)) {
    return stream;
  }
  char *end = nullptr;
  number.value = std::strtod(token.c_str(), &end);
  if (end == token.c_str() || *end != '\0') {
    stream.setstate(std::ios::failbit);
  }
  return stream;
}

auto read(Input &i) -> bool {
  if (!(std::cin >> i.name)) {
    return false;
  }
  if (!(std::cin >> i.N >> ReadDouble{i.A} >> ReadDouble{i.D} >>
        ReadDouble{i.T} >> ReadDouble{i.h} >> ReadDouble{i.chi} >>
        ReadDouble{i.xmin} >> ReadDouble{i.xmax} >> ReadDouble{i.tol} >>
        i.margin >> i.certified >> i.variable >> i.residual >> i.width >>
        i.max_outer >> i.max_inner >> i.max_bracket >> i.newton >>
        i.derivative_mode >> i.root_in_domain)) {
    throw std::runtime_error("malformed scalar inputs");
  }
  i.groups.resize(i.N);
  for (auto &g : i.groups) {
    if (!(std::cin >> ReadDouble{g.r} >> ReadDouble{g.alpha} >>
          ReadDouble{g.p} >> ReadDouble{g.b} >> g.alpha_quarters >>
          g.p_quarters >> g.beta >> ReadDouble{g.L})) {
      throw std::runtime_error("malformed group inputs");
    }
  }
  return true;
}

// Test oracle only. Each accepted active primitive is finite normal. Quarter
// powers use correctly rounded IEEE sqrt rather than a general libm pow.
// Degree <=4 monomials require <=4 rounded multiplies, below the 8lambda
// budget.
auto normal_or_zero(double x) -> bool {
  return x == 0 || (x > 0 && std::isnormal(x));
}
auto quarter_power(double x, int q) -> double {
  if (q == 0) {
    return 1;
  }
  if (q == 1) {
    return std::sqrt(std::sqrt(x));
  }
  if (q == 2) {
    return std::sqrt(x);
  }
  return std::numeric_limits<double>::quiet_NaN();
}

template <std::size_t N> void run(Input const &input) {
  mgsolve::Problem<N> p;
  p.A = input.A;
  p.D = input.D;
  p.T = input.T;
  p.h = input.h;
  p.chi = input.chi;
  mgsolve::Options o;
  o.x_min = input.xmin;
  o.x_max = input.xmax;
  o.relative_tolerance = input.tol;
  o.max_outer = input.max_outer;
  o.max_inner = input.max_inner;
  o.max_bracket = input.max_bracket;
  o.use_newton = input.newton;
  o.allow_residual = input.residual;
  o.allow_width = input.width;
  mgsolve::Contract<N> c;
  c.root_in_domain = input.root_in_domain;
  c.certified = input.certified;
  c.variable_opacity = input.variable;
  c.margin = input.margin;
  c.band_log_error = 8 * mgsolve::u;
  c.alpha_log_error = c.p_log_error = input.variable ? 8 * mgsolve::u : 0;
  c.emission_identically_zero = true;
  for (std::size_t g = 0; g < N; ++g) {
    p.r[g] = input.groups[g].r;
    c.sensitivity[g] = input.groups[g].L;
    c.emission_identically_zero =
        c.emission_identically_zero &&
        (input.groups[g].p == 0 || input.groups[g].b == 0);
  }
  auto const oracle = [&](double x, mgsolve::GroupValues<N> &v) {
    if (x <= 0 || !std::isnormal(x)) {
      return false;
    }
    v.derivatives = input.derivative_mode != 0;
    for (std::size_t j = 0; j < N; ++j) {
      auto const &g = input.groups[j];
      v.alpha_zero[j] = g.alpha == 0;
      v.p_zero[j] = g.p == 0;
      v.B_zero[j] = g.b == 0;
      double const ap = quarter_power(x, g.alpha_quarters);
      double const pp = quarter_power(x, g.p_quarters);
      if (!normal_or_zero(ap) || !normal_or_zero(pp)) {
        return false;
      }
      v.alpha[j] = g.alpha == 0 ? 0 : g.alpha * ap;
      v.p[j] = g.p == 0 ? 0 : g.p * pp;
      double monomial = x;
      if (g.beta == 2 || g.beta == 4) {
        monomial = x * x;
      }
      if (!normal_or_zero(monomial) || monomial == 0) {
        return false;
      }
      if (g.beta == 4) {
        monomial = monomial * monomial;
      }
      if (!normal_or_zero(monomial) || monomial == 0) {
        return false;
      }
      if (g.beta != 1 && g.beta != 2 && g.beta != 4) {
        return false;
      }
      v.B[j] = g.b == 0 ? 0 : g.b * monomial;
      if (!normal_or_zero(v.alpha[j]) || !normal_or_zero(v.p[j]) ||
          !normal_or_zero(v.B[j])) {
        return false;
      }
      if ((v.alpha[j] == 0 && g.alpha != 0) || (v.p[j] == 0 && g.p != 0) ||
          (v.B[j] == 0 && g.b != 0)) {
        return false;
      }
      v.d_alpha[j] = (g.alpha_quarters / 4.0) * v.alpha[j] / x;
      v.d_p[j] = (g.p_quarters / 4.0) * v.p[j] / x;
      v.d_B[j] = g.beta * v.B[j] / x;
      if (input.derivative_mode == 2) {
        v.d_B[j] = std::numeric_limits<double>::quiet_NaN();
      }
      if (input.derivative_mode == 3) {
        v.d_B[j] = -1e100 * v.d_B[j];
      }
    }
    return true;
  };
  auto r = mgsolve::solve(p, oracle, c, o);
  auto const number = [](double x) {
    if (std::isfinite(x)) {
      std::cout << x;
    } else {
      std::cout << "null";
    }
  };
  std::cout << R"({"name":")" << input.name << R"(","status":")"
            << mgsolve::status_name(r.status) << R"(","accepted":)"
            << (mgsolve::accepted(r.status) ? "true" : "false")
            << ",\"stop\":" << int(r.stop) << ",\"dust_temperature\":";
  number(r.dust_temperature);
  std::cout << ",\"gas_temperature\":";
  number(r.gas_temperature);
  std::cout << ",\"gas_energy\":";
  number(r.gas_energy);
  std::cout << ",\"radiation\":[";
  for (std::size_t g = 0; g < N; ++g) {
    if (g) {
      std::cout << ',';
    }
    number(r.radiation[g]);
  }
  std::cout << R"(],"certificate":{"conditional":)"
            << (r.certificate.conditional ? "true" : "false")
            << ",\"dust_relative\":";
  number(r.certificate.dust_relative);
  std::cout << ",\"gas_relative\":";
  number(r.certificate.gas_relative);
  std::cout << ",\"group_relative\":[";
  for (std::size_t g = 0; g < N; ++g) {
    if (g) {
      std::cout << ',';
    }
    number(r.certificate.group_relative[g]);
  }
  std::cout << "]},\"outer_iterations\":" << r.outer_iterations
            << ",\"inner_iterations\":" << r.inner_iterations
            << ",\"oracle_calls\":" << r.oracle_calls << "}\n";
}

auto main() -> int {
  std::cout << std::setprecision(17);
  try {
    Input i;
    while (read(i)) {
      switch (i.N) {
      case 1:
        run<1>(i);
        break;
      case 2:
        run<2>(i);
        break;
      case 3:
        run<3>(i);
        break;
      case 4:
        run<4>(i);
        break;
      case 8:
        run<8>(i);
        break;
      case 16:
        run<16>(i);
        break;
      case 64:
        run<64>(i);
        break;
      case 1024:
        run<1024>(i);
        break;
      default:
        throw std::runtime_error("unsupported test group count");
      }
    }
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
