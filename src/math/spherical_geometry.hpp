#ifndef SPHERICAL_GEOMETRY_HPP_
#define SPHERICAL_GEOMETRY_HPP_

#include <cmath>

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "math/quadrature.hpp"

namespace quokka::math
{

namespace detail
{

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto minDistSqToInterval(amrex::Real const a0, amrex::Real const a1) -> amrex::Real
{
	if (a1 < 0.0) {
		return a1 * a1;
	}
	if (a0 > 0.0) {
		return a0 * a0;
	}
	return 0.0;
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto maxDistSqToInterval(amrex::Real const a0, amrex::Real const a1) -> amrex::Real
{
	const amrex::Real aa0 = std::abs(a0);
	const amrex::Real aa1 = std::abs(a1);
	const amrex::Real amax = (aa0 > aa1) ? aa0 : aa1;
	return amax * amax;
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto clampReal(amrex::Real const x, amrex::Real const lo, amrex::Real const hi) -> amrex::Real
{
	return (x < lo) ? lo : ((x > hi) ? hi : x);
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto normalizeAngle0To2Pi(amrex::Real phi) -> amrex::Real
{
	const amrex::Real two_pi = 2.0 * M_PI;
	while (phi < 0.0) {
		phi += two_pi;
	}
	while (phi >= two_pi) {
		phi -= two_pi;
	}
	return phi;
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto inClosedInterval(amrex::Real const x, amrex::Real const a, amrex::Real const b, amrex::Real const tol) -> bool
{
	return (x >= (a - tol)) && (x <= (b + tol));
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto sortSmallArray(amrex::Real *vals, int n) -> void
{
	for (int i = 1; i < n; ++i) {
		const amrex::Real key = vals[i];
		int j = i - 1;
		while (j >= 0 && vals[j] > key) {
			vals[j + 1] = vals[j];
			--j;
		}
		vals[j + 1] = key;
	}
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto appendAngleUnique(amrex::Real *angles, int &nangles, int max_angles, amrex::Real phi, amrex::Real tol) -> void
{
	phi = normalizeAngle0To2Pi(phi);
	for (int i = 0; i < nangles; ++i) {
		amrex::Real d = std::abs(phi - angles[i]);
		d = (d > M_PI) ? (2.0 * M_PI - d) : d;
		if (d <= tol) {
			return;
		}
	}
	if (nangles < max_angles) {
		angles[nangles] = phi;
		++nangles;
	}
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto appendZEventIfInRange(amrex::Real *zvals, int &nz, int max_nz, amrex::Real z, amrex::Real zlo, amrex::Real zhi,
								    amrex::Real tol) -> void
{
	if (z < (zlo - tol) || z > (zhi + tol)) {
		return;
	}
	z = clampReal(z, zlo, zhi);
	for (int i = 0; i < nz; ++i) {
		if (std::abs(z - zvals[i]) <= tol) {
			return;
		}
	}
	if (nz < max_nz) {
		zvals[nz] = z;
		++nz;
	}
}

AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto deltaPhiCircleRect(amrex::Real const rho, amrex::Real const x0, amrex::Real const x1, amrex::Real const y0,
							    amrex::Real const y1) -> amrex::Real
{
	const amrex::Real two_pi = 2.0 * M_PI;
	const amrex::Real tol = 1.0e-12 * (1.0 + rho);
	if (rho <= tol) {
		return ((0.0 >= x0 && 0.0 <= x1) && (0.0 >= y0 && 0.0 <= y1)) ? two_pi : 0.0;
	}

	// Early reject: circle of radius rho does not reach the rectangle.
	const amrex::Real dx = (0.0 < x0) ? x0 : ((0.0 > x1) ? x1 : 0.0);
	const amrex::Real dy = (0.0 < y0) ? y0 : ((0.0 > y1) ? y1 : 0.0);
	const amrex::Real dmin2 = dx * dx + dy * dy;
	const amrex::Real rho2 = rho * rho;
	if (rho2 < dmin2) {
		return 0.0;
	}

	// Early accept: the full circle lies inside the rectangle.
	if (x0 <= -rho && x1 >= rho && y0 <= -rho && y1 >= rho) {
		return two_pi;
	}

	constexpr int max_angles = 32;
	amrex::Real angles[max_angles];
	int nangles = 0;

	const amrex::Real x_edges[2] = {x0, x1};
	const amrex::Real y_edges[2] = {y0, y1};

	for (int ie = 0; ie < 2; ++ie) {
		const amrex::Real xe = x_edges[ie];
		if (std::abs(xe) <= rho + tol) {
			const amrex::Real ysq = rho * rho - xe * xe;
			const amrex::Real yabs = std::sqrt((ysq > 0.0) ? ysq : 0.0);
			const amrex::Real y_plus = yabs;
			const amrex::Real y_minus = -yabs;
			if (inClosedInterval(y_plus, y0, y1, tol)) {
				appendAngleUnique(angles, nangles, max_angles, std::atan2(y_plus, xe), 1.0e-12);
			}
			if (inClosedInterval(y_minus, y0, y1, tol)) {
				appendAngleUnique(angles, nangles, max_angles, std::atan2(y_minus, xe), 1.0e-12);
			}
		}
	}

	for (int ie = 0; ie < 2; ++ie) {
		const amrex::Real ye = y_edges[ie];
		if (std::abs(ye) <= rho + tol) {
			const amrex::Real xsq = rho * rho - ye * ye;
			const amrex::Real xabs = std::sqrt((xsq > 0.0) ? xsq : 0.0);
			const amrex::Real x_plus = xabs;
			const amrex::Real x_minus = -xabs;
			if (inClosedInterval(x_plus, x0, x1, tol)) {
				appendAngleUnique(angles, nangles, max_angles, std::atan2(ye, x_plus), 1.0e-12);
			}
			if (inClosedInterval(x_minus, x0, x1, tol)) {
				appendAngleUnique(angles, nangles, max_angles, std::atan2(ye, x_minus), 1.0e-12);
			}
		}
	}

	auto pointInside = [=] AMREX_GPU_DEVICE(amrex::Real const phi) -> bool {
		const amrex::Real x = rho * std::cos(phi);
		const amrex::Real y = rho * std::sin(phi);
		return inClosedInterval(x, x0, x1, tol) && inClosedInterval(y, y0, y1, tol);
	};

	if (nangles == 0) {
		return pointInside(0.123456789) ? two_pi : 0.0;
	}

	sortSmallArray(angles, nangles);

	amrex::Real dphi = 0.0;
	for (int i = 0; i < nangles; ++i) {
		const amrex::Real a = angles[i];
		const amrex::Real b = (i + 1 < nangles) ? angles[i + 1] : (angles[0] + two_pi);
		const amrex::Real len = b - a;
		if (len <= 0.0) {
			continue;
		}
		const amrex::Real mid = a + 0.5 * len;
		if (pointInside(mid)) {
			dphi += len;
		}
	}

	if (dphi < 0.0) {
		dphi = 0.0;
	}
	if (dphi > two_pi) {
		dphi = two_pi;
	}
	return dphi;
}

} // namespace detail

AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto sphericalSectionAreaInCell(amrex::Real const R, amrex::Real const x0, amrex::Real const x1, amrex::Real const y0,
								    amrex::Real const y1, amrex::Real const z0, amrex::Real const z1) -> amrex::Real
{
	const amrex::Real R2 = R * R;
	const amrex::Real r2_min = detail::minDistSqToInterval(x0, x1) + detail::minDistSqToInterval(y0, y1) + detail::minDistSqToInterval(z0, z1);
	const amrex::Real r2_max = detail::maxDistSqToInterval(x0, x1) + detail::maxDistSqToInterval(y0, y1) + detail::maxDistSqToInterval(z0, z1);
	if (R2 < r2_min || R2 > r2_max) {
		return 0.0;
	}

	const amrex::Real zlo = (z0 > -R) ? z0 : -R;
	const amrex::Real zhi = (z1 < R) ? z1 : R;
	if (zhi <= zlo) {
		return 0.0;
	}

	const amrex::Real tol_z = 1.0e-12 * (1.0 + R);
	constexpr int max_z_events = 32;
	amrex::Real z_events[max_z_events];
	int nz = 0;

	detail::appendZEventIfInRange(z_events, nz, max_z_events, zlo, zlo, zhi, tol_z);
	detail::appendZEventIfInRange(z_events, nz, max_z_events, zhi, zlo, zhi, tol_z);

	const amrex::Real edge_vals[4] = {x0, x1, y0, y1};
	for (int i = 0; i < 4; ++i) {
		const amrex::Real a = std::abs(edge_vals[i]);
		if (a <= R) {
			const amrex::Real zc = std::sqrt((R2 - a * a > 0.0) ? (R2 - a * a) : 0.0);
			detail::appendZEventIfInRange(z_events, nz, max_z_events, zc, zlo, zhi, tol_z);
			detail::appendZEventIfInRange(z_events, nz, max_z_events, -zc, zlo, zhi, tol_z);
		}
	}

	const amrex::Real x_edges[2] = {x0, x1};
	const amrex::Real y_edges[2] = {y0, y1};
	for (int ix = 0; ix < 2; ++ix) {
		for (int iy = 0; iy < 2; ++iy) {
			const amrex::Real c2 = x_edges[ix] * x_edges[ix] + y_edges[iy] * y_edges[iy];
			if (c2 <= R2) {
				const amrex::Real zc = std::sqrt((R2 - c2 > 0.0) ? (R2 - c2) : 0.0);
				detail::appendZEventIfInRange(z_events, nz, max_z_events, zc, zlo, zhi, tol_z);
				detail::appendZEventIfInRange(z_events, nz, max_z_events, -zc, zlo, zhi, tol_z);
			}
		}
	}

	detail::sortSmallArray(z_events, nz);

	amrex::Real area = 0.0;
	for (int i = 0; i + 1 < nz; ++i) {
		const amrex::Real za = z_events[i];
		const amrex::Real zb = z_events[i + 1];
		if (zb <= za) {
			continue;
		}
		area += quad_1d(
		    [=] AMREX_GPU_DEVICE(amrex::Real const z) -> amrex::Real {
			    const amrex::Real rho2 = R2 - z * z;
			    if (rho2 <= 0.0) {
				    return 0.0;
			    }
			    const amrex::Real rho = std::sqrt(rho2);
			    return R * detail::deltaPhiCircleRect(rho, x0, x1, y0, y1);
		    },
		    za, zb);
	}

	return area;
}

} // namespace quokka::math

#endif // SPHERICAL_GEOMETRY_HPP_
