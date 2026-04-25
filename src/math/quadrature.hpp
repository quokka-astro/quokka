#ifndef QUADRATURE_HPP_
#define QUADRATURE_HPP_

// system headers
#include <cmath>
#include <cstddef>

// library headers
#include <AMReX.H>
#include <AMReX_REAL.H>

#include "gauss.hpp"

AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto kernel_wendland_c2(const amrex::Real r) -> amrex::Real
{
	amrex::Real val = NAN;
	if (r > 1.0) {
		val = 0;
	} else {
		val = (21. / (2. * M_PI)) * std::pow((1.0 - r), 4) * (4.0 * r + 1.0);
	}
	return val;
}

template <typename F>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto quad_3d(F f, amrex::Real x0, amrex::Real x1, amrex::Real y0, amrex::Real y1, amrex::Real z0, amrex::Real z1)
    -> amrex::Real
{
	// Use an explicit tensor-product quadrature here to avoid the deep stack growth
	// from nesting device lambdas through quad_3d -> quad_2d -> quad_1d.
	using Gauss7 = quokka::math::quadrature::gauss<amrex::Real, 7>;
	auto const &abscissa = Gauss7::abscissa();
	auto const &weights = Gauss7::weights();

	const amrex::Real xAvg = 0.5 * (x0 + x1);
	const amrex::Real yAvg = 0.5 * (y0 + y1);
	const amrex::Real zAvg = 0.5 * (z0 + z1);
	const amrex::Real xScale = 0.5 * (x1 - x0);
	const amrex::Real yScale = 0.5 * (y1 - y0);
	const amrex::Real zScale = 0.5 * (z1 - z0);

	amrex::Real sum = 0.0;
	for (std::size_t iz = 0; iz < abscissa.size(); ++iz) {
		const amrex::Real zOffset = zScale * abscissa[iz];
		const amrex::Real zWeight = weights[iz];
		const int zCount = (iz == 0) ? 1 : 2;
		for (int zSign = 0; zSign < zCount; ++zSign) {
			const amrex::Real z = (zSign == 0) ? (zAvg + zOffset) : (zAvg - zOffset);
			for (std::size_t iy = 0; iy < abscissa.size(); ++iy) {
				const amrex::Real yOffset = yScale * abscissa[iy];
				const amrex::Real yWeight = weights[iy];
				const int yCount = (iy == 0) ? 1 : 2;
				for (int ySign = 0; ySign < yCount; ++ySign) {
					const amrex::Real y = (ySign == 0) ? (yAvg + yOffset) : (yAvg - yOffset);
					for (std::size_t ix = 0; ix < abscissa.size(); ++ix) {
						const amrex::Real xOffset = xScale * abscissa[ix];
						const amrex::Real xWeight = weights[ix];
						const int xCount = (ix == 0) ? 1 : 2;
						for (int xSign = 0; xSign < xCount; ++xSign) {
							const amrex::Real x = (xSign == 0) ? (xAvg + xOffset) : (xAvg - xOffset);
							sum += xWeight * yWeight * zWeight * f(x, y, z);
						}
					}
				}
			}
		}
	}

	return xScale * yScale * zScale * sum;
}

template <typename F> AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto quad_2d(F f, amrex::Real x0, amrex::Real x1, amrex::Real y0, amrex::Real y1) -> amrex::Real
{
	// integrate F over the rectangular domain [x0, y0] -> [x1, y1].
	auto integrand = [=] AMREX_GPU_DEVICE(amrex::Real y) { return quad_1d([=] AMREX_GPU_DEVICE(amrex::Real x) { return f(x, y); }, x0, x1); };
	return quad_1d(integrand, y0, y1);
}

template <typename F> AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto quad_1d(F f, amrex::Real x0, amrex::Real x1) -> amrex::Real
{
	// integrate F over the rectangular domain [x0] -> [x1].
	// use 7-point Gauss-Legendre quadrature
	return quokka::math::quadrature::gauss<double, 7>::integrate(f, x0, x1);
}

#endif // QUADRATURE_HPP_
