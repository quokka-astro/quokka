#include <cmath>
#include <iostream>
#include <limits>

#include "math/ODEIntegrate.hpp"

auto main() -> int
{
	int failures = 0;
	const auto check = [&failures](bool passed, const char *message) {
		if (!passed) {
			std::cerr << message << '\n';
			++failures;
		}
	};
	const quokka::valarray<Real, 1> positive{1.};
	const quokka::valarray<Real, 1> negative{-1.};
	const quokka::valarray<Real, 1> error{0.1};
	const quokka::valarray<Real, 1> tolerance{0.1};
	check(std::abs(error_norm(positive, error, 0.1, tolerance) - error_norm(negative, error, 0.1, tolerance)) < 1.e-14,
	      "Error norm must be invariant under state sign changes");
	{
		quokka::valarray<Real, 1> y{1.};
		int calls = 0;
		const auto rhs = [&calls](Real, auto &state, auto &derivative) {
			++calls;
			if (calls == 1) {
				state[0] = 99.;
				return 1;
			}
			derivative[0] = 1.;
			return 0;
		};
		int steps = 0;
		rk_adaptive_integrate(rhs, 0., y, 1., 1.e-4, tolerance, steps);
		check(calls == 1 && steps == maxStepsODEIntegrate && y[0] == 1., "Initial RHS failure must return failure without changing state or retrying");
	}
	for (const Real initial : {0., -1., 1.}) {
		quokka::valarray<Real, 1> y{initial};
		const quokka::valarray<Real, 1> atol{1.e-10};
		const auto rhs = [](Real, auto &, auto &derivative) {
			derivative[0] = 1.;
			return 0;
		};
		int steps = 0;
		rk_adaptive_integrate(rhs, 0., y, 1., 1.e-4, atol, steps);
		check(steps > 0 && steps < maxStepsODEIntegrate && std::abs(y[0] - initial - 1.) < 1.e-12,
		      "Constant source must advance from zero and signed states");
	}
	{
		quokka::valarray<Real, 3> y{0., 2., 0.};
		const quokka::valarray<Real, 3> atol{1.e-10, 1.e-10, 1.e-10};
		const auto rhs = [](Real, auto &, auto &derivative) {
			derivative = {1., -1., 0.};
			return 0;
		};
		int steps = 0;
		rk_adaptive_integrate(rhs, 0., y, 1., 1.e-4, atol, steps);
		check(steps < maxStepsODEIntegrate && std::abs(y[0] - 1.) < 1.e-12 && std::abs(y[1] - 1.) < 1.e-12 && y[2] == 0.,
		      "Mixed zero states and stationary components must integrate");
	}
	{
		quokka::valarray<Real, 1> y{0.};
		const auto rhs = [](Real, auto &, auto &derivative) {
			derivative[0] = 0.;
			return 0;
		};
		int steps = 0;
		rk_adaptive_integrate(rhs, 0., y, 1., 1.e-4, tolerance, steps);
		check(steps > 0 && steps < maxStepsODEIntegrate && y[0] == 0., "Stationary zero solution must succeed");
	}
	{
		quokka::valarray<Real, 1> y{0.};
		const quokka::valarray<Real, 1> atol{1.e-30};
		const auto rhs = [](Real, auto &, auto &derivative) {
			derivative[0] = 1.;
			return 0;
		};
		const Real start = 1.e16;
		const Real end = std::nextafter(start, std::numeric_limits<Real>::infinity());
		int steps = 0;
		rk_adaptive_integrate(rhs, start, y, end, 1.e-4, atol, steps);
		check(steps < maxStepsODEIntegrate && y[0] == end - start, "Initial step must advance representable time");
	}
	for (const Real sign : {-1., 1.}) {
		quokka::valarray<Real, 1> y{sign};
		const quokka::valarray<Real, 1> atol{1.e-8};
		const auto rhs = [](Real, auto &state, auto &derivative) {
			derivative[0] = -state[0];
			return 0;
		};
		int steps = 0;
		rk_adaptive_integrate(rhs, 0., y, 1., 1.e-4, atol, steps);
		check(steps > 0 && steps < maxStepsODEIntegrate && std::abs(y[0] - sign * std::exp(-1.)) < 1.e-4,
		      "Signed exponential decay must meet the accuracy target");
	}
	{
		quokka::valarray<Real, 1> y{0.};
		const quokka::valarray<Real, 1> atol{1.e-8};
		const auto rhs = [](Real, auto &state, auto &derivative) {
			derivative[0] = 1. + state[0];
			return 0;
		};
		int steps = 0;
		rk_adaptive_integrate(rhs, 0., y, 1., 1.e-4, atol, steps);
		check(steps > 0 && steps < maxStepsODEIntegrate && std::abs(y[0] - std::expm1(1.)) < 2.e-4,
		      "Nonlinear growth from zero must meet the accuracy target");
	}
	return failures == 0 ? 0 : 1;
}
