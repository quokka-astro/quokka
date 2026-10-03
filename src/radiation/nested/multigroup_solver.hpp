#ifndef MGSOLVE_MULTIGROUP_SOLVER_HPP
#define MGSOLVE_MULTIGROUP_SOLVER_HPP

// Nested binary64 kernel. See docs/markdown/radiation_solver_accuracy.md.
// The Rocq graph proof is not a C++ or compiler refinement proof.
// Build without fast-math, reassociation, reciprocal approximation or contraction.
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#ifndef MGSOLVE_HD
#if defined(__CUDACC__) || defined(__HIPCC__)
#define MGSOLVE_HD __host__ __device__
#else
#define MGSOLVE_HD
#endif
#endif

namespace mgsolve
{
constexpr double u = 0x1p-53;
constexpr double min_normal = 0x1p-1022;
constexpr double infinity = std::numeric_limits<double>::infinity();
// Strict upper bound on -log(1-u); avoids a libm log contract.
constexpr double lambda_upper = 0x1.0000000000001p-53;

template <class T, std::size_t N> struct Array {
	T data[N ? N : 1]{};
	MGSOLVE_HD T &operator[](std::size_t i) { return data[i]; }
	MGSOLVE_HD const T &operator[](std::size_t i) const { return data[i]; }
};

enum class Status {
	accepted_conditional,
	accepted_estimated,
	invalid_input,
	missing_contract,
	oracle_failure,
	range_failure,
	no_bracket,
	precision_limit,
	tolerance_unavailable,
	iteration_limit,
	unsupported_model
};
enum class Stop { none, residual, width, direct_absorption, zero_exchange };
MGSOLVE_HD inline bool accepted(Status s) { return s == Status::accepted_conditional || s == Status::accepted_estimated; }
inline const char *status_name(Status s)
{
	switch (s) {
#define MG_STATUS(name)                                                                                                                                        \
	case Status::name:                                                                                                                                     \
		return #name
		MG_STATUS(accepted_conditional);
		MG_STATUS(accepted_estimated);
		MG_STATUS(invalid_input);
		MG_STATUS(missing_contract);
		MG_STATUS(oracle_failure);
		MG_STATUS(range_failure);
		MG_STATUS(no_bracket);
		MG_STATUS(precision_limit);
		MG_STATUS(tolerance_unavailable);
		MG_STATUS(iteration_limit);
		MG_STATUS(unsupported_model);
#undef MG_STATUS
	}
	return "unknown";
}

template <std::size_t N> struct Problem {
	double A = 1, D = 1, T = 1, h = 1, chi = 1;
	Array<double, N> r{};
};
template <std::size_t N> struct GroupValues {
	Array<double, N> alpha{}, p{}, B{};
	Array<double, N> d_alpha{}, d_p{}, d_B{};
	// An exact-zero declaration means the specified function is IDENTICALLY
	// zero on the whole temperature domain. Otherwise it must be strictly
	// positive there. Never flag a value that merely underflowed to zero.
	Array<bool, N> alpha_zero{}, p_zero{}, B_zero{};
	bool derivatives = false;
};
template <std::size_t N> struct Contract {
	bool certified = false;
	bool variable_opacity = false;
	bool root_in_domain = false; // caller establishes existence in [x_min,x_max]
	// All statements refer to the WHOLE options temperature domain. Caller must
	// provide a proof/validated oracle, not infer these flags from sampled values.
	bool emission_identically_zero = false;
	double margin = 1;		// lower bound delta=min(1,m)-v > 0, <=1
	Array<double, N> sensitivity{}; // |d log E_g/d log x| <= sensitivity[g]
	double alpha_log_error = 0, p_log_error = 0;
	double band_log_error = 8 * u;
};
struct Options {
	double x_min = 0x1p-100, x_max = 0x1p100;
	int max_outer = 256, max_inner = 256, max_bracket = 2048;
	bool use_newton = true, allow_residual = true, allow_width = true;
	bool allow_estimated = false;
	double relative_tolerance = 1e-12;
};
template <std::size_t N> struct Certificate {
	bool conditional = false;
	double coordinate_log = infinity, gas_log = infinity;
	double dust_relative = infinity, gas_relative = infinity;
	Array<double, N> group_log{}, group_relative{};
	Array<bool, N> group_exact_zero{}; // group_relative=0 is an absolute-zero sentinel here
	int limiting_group = -1;
};
template <std::size_t N> struct Result {
	Status status = Status::invalid_input;
	Stop stop = Stop::none;
	double dust_temperature = 0, gas_temperature = 0, gas_energy = 0;
	Array<double, N> radiation{};
	Certificate<N> certificate{};
	double bracket_lo = 0, bracket_hi = 0, ratio = 0;
	std::int64_t outer_iterations = 0, inner_iterations = 0, oracle_calls = 0;
};

namespace detail
{
MGSOLVE_HD inline bool normal(double x) { return x >= min_normal && x <= std::numeric_limits<double>::max(); }
MGSOLVE_HD inline bool nonnegative(double x) { return x == 0 || normal(x); }
MGSOLVE_HD inline double min(double a, double b) { return a < b ? a : b; }
MGSOLVE_HD inline double max(double a, double b) { return a > b ? a : b; }
// Every actual graph node is checked. Structural-zero checks inspect operands,
// which is essential: an underflow-to-zero product is NOT an exact zero.
struct Arithmetic {
	bool ok = true;
	MGSOLVE_HD double check(double z, bool exact_zero = false)
	{
		if (!(normal(z) || (exact_zero && z == 0)))
			ok = false;
		return z;
	}
	MGSOLVE_HD double add(double a, double b) { return check(a + b, a == 0 && b == 0); }
	MGSOLVE_HD double mul(double a, double b) { return check(a * b, a == 0 || b == 0); }
	MGSOLVE_HD double div(double a, double b)
	{
		if (!normal(b)) {
			ok = false;
			return 0;
		}
		return check(a / b, a == 0);
	}
	MGSOLVE_HD double sub(double a, double b) { return check(a - b, a == b); }
	MGSOLVE_HD double sqrt(double a) { return check(::sqrt(a)); }
};
MGSOLVE_HD inline double up(double x) { return ::nextafter(x, infinity); }
MGSOLVE_HD inline double add_up(double a, double b) { return up(a + b); }
MGSOLVE_HD inline double mul_up(double a, double b) { return (a == 0 || b == 0) ? 0 : up(a * b); }
MGSOLVE_HD inline double div_up(double a, double b) { return a == 0 ? 0 : up(a / b); }
// exp(b)-1 <= b/(1-b), 0<=b<1. Scalar conservative bound arithmetic,
// not an interval root solver and no assumption about exp/expm1 accuracy.
MGSOLVE_HD inline double relative_bound(double b)
{
	if (b == 0)
		return 0;
	if (!(b > 0 && b < 0.5))
		return infinity;
	double lower_den = ::nextafter(1 - b, 0.0);
	return div_up(b, lower_den);
}
// Positive double bit ranks are monotone. memcpy avoids aliasing UB on hosts;
// device compilers lower this fixed-size copy. No floating graph relies on it.
MGSOLVE_HD inline std::uint64_t rank(double x)
{
	std::uint64_t bits;
	::memcpy(&bits, &x, sizeof(bits));
	return bits;
}
MGSOLVE_HD inline double unrank(std::uint64_t bits)
{
	double x;
	::memcpy(&x, &bits, sizeof(x));
	return x;
}
MGSOLVE_HD inline double midpoint(double lo, double hi)
{
	auto l = rank(lo), h = rank(hi);
	return unrank(l + (h - l) / 2);
}
MGSOLVE_HD inline bool adjacent(double lo, double hi) { return rank(hi) - rank(lo) <= 1; }
// Newton is only a proposal. Requiring the middle half of bit ranks guarantees
// shrinkage even when derivatives are wrong, nonfinite or discontinuous.
MGSOLVE_HD inline double safeguard(double lo, double hi, double proposal)
{
	auto l = rank(lo), h = rank(hi), width = h - l;
	if (normal(proposal)) {
		auto p = rank(proposal);
		if (p > l && p < h && p - l >= width / 4 && h - p >= width / 4)
			return proposal;
	}
	return midpoint(lo, hi);
}
MGSOLVE_HD inline bool narrow(double lo, double hi, double units)
{
	if (!(normal(lo) && normal(hi) && lo <= hi))
		return false;
	if (lo == hi)
		return true;
	// SafeDifference64: close binary64 endpoints subtract exactly (Sterbenz),
	// including subnormal gaps; far endpoints have a finite normal difference.
	double gap = hi - lo;
	if (!(gap > 0 && gap <= std::numeric_limits<double>::max()))
		return false;
	double width = gap / lo;
	return normal(width) && width <= units * u;
}
MGSOLVE_HD inline int guard(double ratio, double units) { return ratio < 1 - units * u ? -1 : (ratio > 1 + units * u ? 1 : 0); }

enum class Chart { heating, weak, strong };
struct Inner {
	Status status = Status::range_failure;
	double t = 0, q = 0, balance = 0;
	int iterations = 0;
};
struct InnerEval {
	bool ok = false;
	double t = 0, q = 0, ratio = 0, derivative = 0;
};
MGSOLVE_HD inline InnerEval eval_inner(double A, double D, double T, double x, double z, Chart chart)
{
	Arithmetic a;
	InnerEval e;
	double delta = 0, th = 0, numerator = 0, denominator = 0;
	if (chart == Chart::strong) {
		th = z;
		delta = a.sub(T, z);
		e.q = a.mul(A, delta);
		double sh = a.sqrt(z), ds = a.mul(D, sh), gh = a.div(e.q, ds);
		numerator = a.add(x, gh);
		denominator = z;
		e.derivative = -(x + gh * (1.5 + z / delta)) / (z * z); // proposal only
	} else {
		th = chart == Chart::heating ? a.add(T, z) : a.sub(T, z);
		delta = chart == Chart::heating ? a.sub(x, T) : a.sub(T, x);
		e.q = a.mul(A, z);
		double sh = a.sqrt(th), ds = a.mul(D, sh), gh = a.div(e.q, ds);
		numerator = a.add(z, gh);
		denominator = delta;
		e.derivative = (1 + (A / ds) * (1 + (chart == Chart::heating ? -0.5 : 0.5) * z / th)) / delta;
	}
	e.ratio = a.div(numerator, denominator);
	e.t = th;
	e.ok = a.ok;
	return e;
}
MGSOLVE_HD inline Inner inner(double A, double D, double T, double x, const Options &opt)
{
	Inner out;
	if (x == T) {
		out.status = Status::accepted_conditional;
		out.t = T;
		out.q = 0;
		return out;
	}
	Chart chart = Chart::heating;
	double hi = 0, lo = 0;
	InnerEval e;
	if (x < T) {
		double half = T * 0.5;
		if (!normal(half))
			return out;
		e = eval_inner(A, D, T, x, half, Chart::strong);
		++out.iterations;
		if (!e.ok)
			return out;
		int s = guard(e.ratio, 16);
		if (s == 0) {
			out.status = Status::accepted_conditional;
			out.t = e.t;
			out.q = e.q;
			out.balance = e.ratio;
			return out;
		}
		chart = s < 0 ? Chart::strong : Chart::weak;
		if (chart == Chart::strong) {
			lo = x;
			hi = half;
		} // exact analytic x < t* < T/2
	}
	if (chart != Chart::strong) {
		Arithmetic a;
		double delta = chart == Chart::heating ? a.sub(x, T) : a.sub(T, x);
		hi = a.mul(delta, 1 + 32 * u);
		if (chart == Chart::weak)
			hi = min(hi, T * 0.5);
		if (!a.ok || !normal(hi))
			return out;
		// Search for actual certified signs; a rounded analytic seed alone is not
		// a certified endpoint. The widened upper endpoint encloses the root.
		e = eval_inner(A, D, T, x, hi, chart);
		++out.iterations;
		if (!e.ok)
			return out;
		int sign = guard(e.ratio, 16);
		if (sign == 0) {
			out.status = Status::accepted_conditional;
			out.t = e.t;
			out.q = e.q;
			out.balance = e.ratio;
			return out;
		}
		if (sign < 0) {
			out.status = Status::no_bracket;
			return out;
		}
		lo = hi;
		bool found = false;
		for (int i = 0; i < opt.max_bracket; ++i) {
			lo *= 0.5;
			if (!normal(lo))
				return out;
			e = eval_inner(A, D, T, x, lo, chart);
			++out.iterations;
			if (!e.ok)
				return out;
			sign = guard(e.ratio, 16);
			if (sign == 0) {
				out.status = Status::accepted_conditional;
				out.t = e.t;
				out.q = e.q;
				out.balance = e.ratio;
				return out;
			}
			if (sign < 0) {
				found = true;
				break;
			}
			hi = lo;
		}
		if (!found) {
			out.status = Status::iteration_limit;
			return out;
		}
	}
	double z = midpoint(lo, hi);
	for (int i = 0; i < opt.max_inner; ++i) {
		e = eval_inner(A, D, T, x, z, chart);
		++out.iterations;
		if (!e.ok)
			return out;
		int sign = guard(e.ratio, 16);
		if (sign == 0 || narrow(lo, hi, 4)) {
			out.status = Status::accepted_conditional;
			out.t = e.t;
			out.q = e.q;
			out.balance = e.ratio;
			return out;
		}
		if (adjacent(lo, hi)) {
			out.status = Status::precision_limit;
			return out;
		}
		if ((sign < 0) != (chart == Chart::strong))
			lo = z;
		else
			hi = z;
		double proposal = z - (e.ratio - 1) / e.derivative;
		z = safeguard(lo, hi, opt.use_newton ? proposal : 0);
	}
	out.status = Status::iteration_limit;
	return out;
}

template <std::size_t N> MGSOLVE_HD double sum(Arithmetic &a, Array<double, N> values)
{
	std::size_t n = N;
	while (n > 1) {
		const std::size_t pairs = n / 2;
		for (std::size_t i = 0; i < pairs; ++i)
			values[i] = a.add(values[2 * i], values[2 * i + 1]);
		if (n % 2)
			values[pairs] = values[n - 1];
		n = pairs + n % 2;
	}
	return N ? values[0] : 0;
}
template <std::size_t N> struct Evaluation {
	Status status = Status::oracle_failure;
	double t = 0, q = 0, U = 0, M = 0, H = 0, ratio = 0, ratio_derivative = 0;
	Array<double, N> E{};
	int sign = 0, inner_iterations = 0;
};

template <std::size_t N, class Oracle>
MGSOLVE_HD Evaluation<N> evaluate(const Problem<N> &prob, Oracle &oracle, double x, const Options &opt, bool groups_only = false)
{
	Evaluation<N> e;
	GroupValues<N> v;
	if (!oracle(x, v))
		return e;
	Arithmetic a;
	Array<double, N> emissions{}, absorptions{};
	double dM = 0, dH = 0;
	for (std::size_t g = 0; g < N; ++g) {
		if (!nonnegative(v.alpha[g]) || !nonnegative(v.p[g]) || !nonnegative(v.B[g]) || (v.alpha[g] == 0 && !v.alpha_zero[g]) ||
		    (v.p[g] == 0 && !v.p_zero[g]) || (v.B[g] == 0 && !v.B_zero[g]) || (v.alpha[g] != 0 && v.alpha_zero[g]) || (v.p[g] != 0 && v.p_zero[g]) ||
		    (v.B[g] != 0 && v.B_zero[g])) {
			e.status = Status::oracle_failure;
			return e;
		}
		double tau = a.mul(prob.h, v.alpha[g]), den = a.add(1, tau), hp = a.mul(prob.h, v.p[g]);
		double w = a.div(hp, den), c = a.mul(prob.chi, w);
		emissions[g] = a.mul(c, v.B[g]);
		double fraction = a.div(tau, den), chi_fraction = a.mul(prob.chi, fraction);
		absorptions[g] = a.mul(chi_fraction, prob.r[g]);
		double emitted = a.mul(hp, v.B[g]), numerator = a.add(prob.r[g], emitted);
		e.E[g] = a.div(numerator, den);
		if (v.derivatives) {
			double taup = prob.h * v.d_alpha[g];
			dM += prob.chi * prob.h * (v.d_p[g] * v.B[g] + v.p[g] * v.d_B[g]) / den - emissions[g] * taup / den;
			dH += prob.chi * prob.r[g] * taup / (den * den);
		}
	}
	e.M = sum(a, emissions);
	e.H = sum(a, absorptions);
	if (!a.ok) {
		e.status = Status::range_failure;
		return e;
	}
	if (groups_only) {
		e.status = Status::accepted_conditional;
		return e;
	}
	Inner in = inner(prob.A, prob.D, prob.T, x, opt);
	e.inner_iterations = in.iterations;
	if (!accepted(in.status)) {
		e.status = in.status;
		return e;
	}
	e.t = in.t;
	e.q = in.q;
	e.U = a.mul(prob.A, e.t);
	if (x >= prob.T) {
		double numerator = a.add(e.q, e.M);
		if (e.H == 0) {
			e.sign = numerator == 0 ? 0 : 1;
			e.ratio = numerator == 0 ? 1 : infinity;
		} else
			e.ratio = a.div(numerator, e.H);
	} else {
		double den = a.add(e.q, e.H);
		if (den == 0) {
			e.sign = e.M == 0 ? 0 : 1;
			e.ratio = e.M == 0 ? 1 : infinity;
		} else
			e.ratio = a.div(e.M, den);
	}
	if (!a.ok) {
		e.status = Status::range_failure;
		return e;
	}
	if (e.ratio != infinity)
		e.sign = guard(e.ratio, 128);
	if (v.derivatives) {
		double t = e.t;
		double dt = 1 / (1 + prob.A / prob.D * (t + prob.T) / (2 * t * ::sqrt(t)));
		double dq = (x >= prob.T ? 1 : -1) * prob.A * dt;
		if (x >= prob.T && e.H > 0)
			e.ratio_derivative = (dq + dM - e.ratio * dH) / e.H;
		if (x < prob.T && e.q + e.H > 0)
			e.ratio_derivative = (dM - e.ratio * (dq + dH)) / (e.q + e.H);
	}
	e.status = Status::accepted_conditional;
	return e;
}

template <std::size_t N> MGSOLVE_HD Certificate<N> certificate(const Contract<N> &c, double coordinate, const Array<double, N> &energies, double gas_log = -1)
{
	Certificate<N> out;
	out.conditional = c.certified;
	out.coordinate_log = coordinate;
	out.gas_log = gas_log >= 0 ? gas_log : add_up(mul_up(53, lambda_upper), mul_up(2, coordinate));
	out.dust_relative = relative_bound(coordinate);
	out.gas_relative = relative_bound(out.gas_log);
	double worst = max(out.dust_relative, out.gas_relative);
	for (std::size_t g = 0; g < N; ++g) {
		out.group_exact_zero[g] = energies[g] == 0;
		out.group_log[g] = energies[g] == 0 ? 0 : add_up(mul_up(c.variable_opacity ? 30 : 14, lambda_upper), mul_up(c.sensitivity[g], coordinate));
		out.group_relative[g] = relative_bound(out.group_log[g]);
		if (out.group_relative[g] > worst) {
			worst = out.group_relative[g];
			out.limiting_group = static_cast<int>(g);
		}
	}
	return out;
}
template <std::size_t N> MGSOLVE_HD bool meets(const Certificate<N> &c, double tolerance)
{
	if (!(c.dust_relative <= tolerance && c.gas_relative <= tolerance))
		return false;
	for (std::size_t g = 0; g < N; ++g)
		if (!(c.group_relative[g] <= tolerance))
			return false;
	return true;
}
} // namespace detail

template <std::size_t N, class Oracle> MGSOLVE_HD Result<N> solve(const Problem<N> &p, Oracle oracle, const Contract<N> &c, const Options &opt = {})
{
	static_assert(N <= 1024, "The published specialization supports at most depth ten / 1024 groups");
	static_assert(sizeof(double) == 8 && std::numeric_limits<double>::digits == 53 && std::numeric_limits<double>::is_iec559, "IEEE binary64 is required");
	using namespace detail;
	Result<N> out;
#if defined(__FAST_MATH__)
	out.status = Status::unsupported_model;
	return out;
#endif
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
	if (std::fegetround() != FE_TONEAREST) {
		out.status = Status::unsupported_model;
		return out;
	}
#endif
	if (!(normal(p.A) && normal(p.D) && normal(p.T) && normal(p.h) && normal(p.chi) && normal(opt.x_min) && normal(opt.x_max) && opt.x_min <= opt.x_max &&
	      opt.relative_tolerance > 0 && opt.relative_tolerance < infinity && opt.max_outer > 0 && opt.max_inner > 0 && opt.max_bracket > 0 &&
	      opt.max_outer <= 65536 && opt.max_inner <= 65536 && opt.max_bracket <= 65536))
		return out;
	for (std::size_t g = 0; g < N; ++g)
		if (!nonnegative(p.r[g]))
			return out;
	if (!c.certified && !opt.allow_estimated) {
		out.status = Status::missing_contract;
		return out;
	}
	if (!(c.margin > 0 && c.margin <= 1 && normal(c.margin)) || (!c.variable_opacity && c.margin != 1)) {
		out.status = Status::missing_contract;
		return out;
	}
	if (c.certified) {
		if (!c.root_in_domain) {
			out.status = Status::missing_contract;
			return out;
		}
		// Budgets are declared in physical log-error units. A binary64 value equal
		// to 8u is <=8lambda, whereas rounded 8*lambda_upper is slightly too large.
		double coeff_limit = c.variable_opacity ? 8 * u : 0;
		if (!(c.alpha_log_error >= 0 && c.alpha_log_error <= coeff_limit && c.p_log_error >= 0 && c.p_log_error <= coeff_limit &&
		      c.band_log_error >= 0 && c.band_log_error <= 8 * u)) {
			out.status = Status::missing_contract;
			return out;
		}
	}
	for (std::size_t g = 0; g < N; ++g)
		if (!(c.sensitivity[g] >= 0 && c.sensitivity[g] < infinity)) {
			out.status = Status::missing_contract;
			return out;
		}
	double x = max(opt.x_min, min(p.T, opt.x_max));
	auto e = evaluate(p, oracle, x, opt, c.emission_identically_zero || N == 0);
	++out.oracle_calls;
	out.inner_iterations += e.inner_iterations;
	if (!accepted(e.status)) {
		out.status = e.status;
		return out;
	}
	// Constant zero-emission bypass is legal only from the global declaration,
	// never because a temperature-dependent callback happened to return zero.
	if (c.emission_identically_zero || N == 0) {
		if (c.variable_opacity || e.M != 0) {
			out.status = Status::unsupported_model;
			return out;
		}
		Arithmetic a;
		double t = p.T;
		if (e.H > 0) {
			double increment = a.div(e.H, p.A);
			t = a.add(p.T, increment);
			double ds = a.mul(p.D, a.sqrt(t));
			x = a.add(t, a.div(e.H, ds));
		} else
			x = t;
		if (!a.ok) {
			out.status = Status::range_failure;
			return out;
		}
		if (!(x >= opt.x_min && x <= opt.x_max)) {
			out.status = Status::no_bracket;
			return out;
		}
		auto final = evaluate(p, oracle, x, opt, true);
		++out.oracle_calls;
		out.inner_iterations += final.inner_iterations;
		if (!accepted(final.status)) {
			out.status = final.status;
			return out;
		}
		out.dust_temperature = x;
		out.gas_temperature = t;
		out.gas_energy = a.mul(p.A, t);
		out.radiation = final.E;
		if (!a.ok) {
			out.status = Status::range_failure;
			return out;
		}
		double bx = e.H > 0 ? mul_up(27.5, lambda_upper) : 0; // eta_H <=(5+10)lambda
		double gaslog = mul_up(e.H > 0 ? 18 : 1, lambda_upper);
		out.certificate = certificate(c, bx, final.E, gaslog);
		out.stop = e.H > 0 ? Stop::direct_absorption : Stop::zero_exchange;
		out.status = meets(out.certificate, opt.relative_tolerance) ? (c.certified ? Status::accepted_conditional : Status::accepted_estimated)
									    : Status::tolerance_unavailable;
		return out;
	}
	if (e.M == 0) {
		out.status = Status::missing_contract;
		return out;
	}
	double lo = 0, hi = 0;
	int first_sign = e.sign;
	bool bracket = false;
	for (int k = 0; k < opt.max_bracket + opt.max_outer; ++k) {
		out.dust_temperature = x;
		out.gas_temperature = e.t;
		out.gas_energy = e.U;
		out.radiation = e.E;
		out.ratio = e.ratio;
		out.bracket_lo = lo;
		out.bracket_hi = hi;
		if (e.sign == 0) {
			double bx = div_up(mul_up(c.variable_opacity ? 223 : 207, lambda_upper), c.margin);
			out.certificate = certificate(c, bx, e.E);
			if (opt.allow_residual && meets(out.certificate, opt.relative_tolerance)) {
				out.stop = Stop::residual;
				out.status = c.certified ? Status::accepted_conditional : Status::accepted_estimated;
				return out;
			}
			// Do not manufacture a sign from an ambiguous residual. If a genuine
			// bracket exists, width may already suffice; otherwise binary64 cannot
			// establish this requested certificate by this graph.
			if (!(bracket && opt.allow_width && narrow(lo, hi, 16))) {
				out.status = Status::tolerance_unavailable;
				return out;
			}
		}
		if (bracket && opt.allow_width && narrow(lo, hi, 16)) {
			out.certificate = certificate(c, mul_up(32, lambda_upper), e.E);
			if (meets(out.certificate, opt.relative_tolerance)) {
				out.stop = Stop::width;
				out.status = c.certified ? Status::accepted_conditional : Status::accepted_estimated;
				return out;
			}
			out.status = Status::tolerance_unavailable;
			return out;
		}
		if (e.sign < 0)
			lo = x;
		else if (e.sign > 0)
			hi = x;
		if (lo > 0 && hi > 0)
			bracket = true;
		if (bracket) {
			if (++out.outer_iterations > opt.max_outer) {
				out.status = Status::iteration_limit;
				return out;
			}
			if (adjacent(lo, hi)) {
				out.status = Status::precision_limit;
				return out;
			}
			double proposal = x - (e.ratio - 1) / e.ratio_derivative;
			x = safeguard(lo, hi, opt.use_newton ? proposal : 0);
		} else {
			if (k >= opt.max_bracket) {
				out.status = Status::iteration_limit;
				return out;
			}
			double next;
			if (first_sign < 0)
				next = x > opt.x_max * 0.5 ? opt.x_max : min(opt.x_max, x * 2);
			else
				next = max(opt.x_min, x * 0.5);
			if (next == x) {
				out.status = Status::no_bracket;
				return out;
			}
			x = next;
		}
		e = evaluate(p, oracle, x, opt);
		++out.oracle_calls;
		out.inner_iterations += e.inner_iterations;
		if (!accepted(e.status)) {
			out.status = e.status;
			return out;
		}
		if (e.M == 0) {
			out.status = Status::missing_contract;
			return out;
		}
	}
	out.status = Status::iteration_limit;
	return out;
}
} // namespace mgsolve
#endif
