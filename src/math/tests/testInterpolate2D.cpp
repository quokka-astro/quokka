#include <array>
#include <cmath>
#include <iostream>

#include "Interpolate2D.hpp"

auto main() -> int
{
	// Nonlinear samples expose selection of the wrong interpolation interval.
	const std::array<double, 4> x{0., 1., 2., 3.};
	const std::array<double, 4> y{10., 11., 12., 13.};
	std::array<double, 16> values{};
	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			values[i + 4 * j] = x[i] * x[i] + y[j] * y[j];
		}
	}
	int failures = 0;
	for (const int xb : {0, 5, -5}) {
		for (const int yb : {0, 7, -7}) {
			const amrex::Table1D<const double> xv(x.data(), xb, xb + 4);
			const amrex::Table1D<const double> yv(y.data(), yb, yb + 4);
			const amrex::Table2D<const double> table(values.data(), {xb, yb}, {xb + 4, yb + 4});
			// Interior, both upper edges, corners, and clamping outside every edge.
			for (const auto &point : std::array<std::array<double, 3>, 10>{{{1.5, 10.5, 113.},
											{1.5, 13., 171.5},
											{3., 11.5, 141.5},
											{3., 13., 178.},
											{0., 10., 100.},
											{-1., 11.5, 132.5},
											{4., 11.5, 141.5},
											{1.5, 9., 102.5},
											{1.5, 14., 171.5},
											{4., 14., 178.}}}) {
				const double actual = interpolate2d(point[0], point[1], xv, yv, table);
				if (!std::isfinite(actual) || std::abs(actual - point[2]) > 1.e-12) {
					std::cerr << "origins " << xb << ',' << yb << " at " << point[0] << ',' << point[1] << ": got " << actual
						  << ", expected " << point[2] << '\n';
					++failures;
				}
			}
		}
	}
	return failures == 0 ? 0 : 1;
}
