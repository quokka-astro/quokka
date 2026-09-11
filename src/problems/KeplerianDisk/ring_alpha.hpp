// Copyright 2026 Quokka developers.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
#ifndef QUOKKA_RING_ALPHA_HPP_
#define QUOKKA_RING_ALPHA_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

namespace ring_alpha
{
// exp(-z) I_{1/4}(z): power series at small z, asymptotic expansion at large z.
// Scaling avoids overflow for cold, narrow rings (z = 2x/tau can exceed 1e5).
inline auto scaledBessel(double z) -> double
{
	if (z <= 50.) {
		double term = std::pow(0.5 * z, 0.25) / std::tgamma(1.25);
		double sum = term;
		for (int k = 1; k < 200; ++k) {
			term *= 0.25 * z * z / (k * (k + 0.25));
			sum += term;
			if (term <= 1.e-16 * sum) {
				break;
			}
		}
		return std::exp(-z) * sum;
	}
	double term = 1.;
	double sum = term;
	for (int k = 1; k <= 12; ++k) {
		term *= ((2. * k - 1.) * (2. * k - 1.) - 0.25) / (8. * z * k);
		sum += term;
	}
	return sum / std::sqrt(2. * std::numbers::pi * z);
}

// Equation (23), with the dimensional factor m/(pi R0^2) omitted.
inline auto kernel(double x, double tau) -> double
{
	if (x == 0.) {
		return std::exp(-1. / tau) / (std::pow(tau, 1.25) * std::tgamma(1.25));
	}
	return std::exp(-(x - 1.) * (x - 1.) / tau) * scaledBessel(2. * x / tau) / (tau * std::pow(x, 0.25));
}

// Area-average the continuous solution over the same radial bins as the data.
inline auto binMean(double lower, double upper, double radius, double mass, double tau) -> double
{
	constexpr std::array<double, 4> nodes{0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363};
	constexpr std::array<double, 4> weights{0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};
	const double midpoint = 0.5 * (lower + upper);
	const double halfWidth = 0.5 * (upper - lower);
	double sum = 0.;
	for (int q = 0; q < 4; ++q) {
		for (const double sign : {-1., 1.}) {
			const double r = midpoint + sign * halfWidth * nodes[q];
			sum += weights[q] * r * kernel(r / radius, tau);
		}
	}
	return mass / (std::numbers::pi * radius * radius) * sum / (lower + upper);
}

struct Fit {
	double tau = std::numeric_limits<double>::quiet_NaN();
	double relativeL2 = std::numeric_limits<double>::quiet_NaN();
	// 0: interior minimum; 1: bound reached; 2: nonfinite data; 3: no positive signal.
	int status = 2;
};

inline auto fit(std::vector<double> const &sigma, double dr, double radius, double mass) -> Fit
{
	if (!(std::isfinite(dr) && dr > 0. && std::isfinite(radius) && radius > 0. && std::isfinite(mass) && mass > 0.)) {
		return {};
	}
	double norm = 0.;
	double signal = 0.;
	for (double value : sigma) {
		if (!std::isfinite(value)) {
			return {};
		}
		norm += value * value;
		signal += value;
	}
	if (!std::isfinite(norm) || !std::isfinite(signal)) {
		return {};
	}
	if (!(norm > 0. && signal > 0.)) {
		return {NAN, NAN, 3};
	}
	const auto loss = [&](double logTau) {
		const double tau = std::exp(logTau);
		double sum = 0.;
		for (int i = 0; i < static_cast<int>(sigma.size()); ++i) {
			const double error = binMean(i * dr, (i + 1) * dr, radius, mass, tau) - sigma[i];
			sum += error * error;
		}
		return sum;
	};
	constexpr int intervals = 80;
	const double lower = std::log(1.e-5);
	const double upper = std::log(10.);
	const double step = (upper - lower) / intervals;
	int best = 0;
	double minimum = loss(lower);
	for (int i = 1; i <= intervals; ++i) {
		const double value = loss(lower + i * step);
		if (value < minimum) {
			minimum = value;
			best = i;
		}
	}
	if (best == 0 || best == intervals) {
		return {std::exp(lower + best * step), std::sqrt(minimum / norm), 1};
	}
	double a = lower + (best - 1) * step;
	double b = lower + (best + 1) * step;
	constexpr double golden = 0.6180339887498948482;
	double c = b - golden * (b - a);
	double d = a + golden * (b - a);
	double fc = loss(c);
	double fd = loss(d);
	for (int iteration = 0; iteration < 48; ++iteration) {
		if (fc < fd) {
			b = d;
			d = c;
			fd = fc;
			c = b - golden * (b - a);
			fc = loss(c);
		} else {
			a = c;
			c = d;
			fc = fd;
			d = a + golden * (b - a);
			fd = loss(d);
		}
	}
	const double optimum = 0.5 * (a + b);
	return {std::exp(optimum), std::sqrt(loss(optimum) / norm), 0};
}

inline auto viscosity(double tau, double radius, double time) -> double { return tau * radius * radius / (12. * time); }
inline auto alpha(double nu, double omega, double csSquared) -> double { return 1.5 * nu * omega / csSquared; }

// Trapezoidal time integration, clipped to the paper's averaging interval.
// Invalid samples break continuity; uncovered intervals are never filled with zero.
struct Average {
	double previousTime = NAN;
	double previousAlpha = NAN;
	double integral = 0.;
	double covered = 0.;
	void add(double orbits, double value)
	{
		if (std::isfinite(previousAlpha) && std::isfinite(value) && orbits > previousTime) {
			const double lo = std::max(0.09, previousTime);
			const double hi = std::min(0.9, orbits);
			if (hi > lo) {
				const double slope = (value - previousAlpha) / (orbits - previousTime);
				integral += (hi - lo) * (previousAlpha + slope * (0.5 * (lo + hi) - previousTime));
				covered += hi - lo;
			}
		}
		previousTime = orbits;
		previousAlpha = value;
	}
	[[nodiscard]] auto mean() const -> double { return (covered > 0.) ? integral / covered : NAN; }
};
} // namespace ring_alpha
#endif
