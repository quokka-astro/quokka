#ifndef SPHERICAL_GEOMETRY_HPP_
#define SPHERICAL_GEOMETRY_HPP_

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace quokka::math
{

namespace detail
{

using Point = amrex::GpuArray<amrex::Real, 3>;
using Edge = amrex::GpuArray<int, 2>;

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

template <unsigned int MaxPts>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto addPointUnique(amrex::GpuArray<Point, MaxPts> &pts, int &npts, amrex::Real x, amrex::Real y, amrex::Real z,
							     amrex::Real tol) -> void
{
	const amrex::Real tol2 = tol * tol;
	for (int i = 0; i < npts; ++i) {
		const amrex::Real dx = pts[i][0] - x;
		const amrex::Real dy = pts[i][1] - y;
		const amrex::Real dz = pts[i][2] - z;
		if ((dx * dx + dy * dy + dz * dz) <= tol2) {
			return;
		}
	}
	if (npts < static_cast<int>(pts.size())) {
		pts[npts] = Point{x, y, z};
		++npts;
	}
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto planeBoxSectionArea(amrex::Real const x0, amrex::Real const x1, amrex::Real const y0, amrex::Real const y1,
								  amrex::Real const z0, amrex::Real const z1, amrex::Real const nx, amrex::Real const ny,
								  amrex::Real const nz, amrex::Real const d) -> amrex::Real
{
	// Plane: n·x = d. Compute exact area of intersection polygon with an axis-aligned box.
	const amrex::Real scale = (std::abs(x0) + std::abs(x1) + std::abs(y0) + std::abs(y1) + std::abs(z0) + std::abs(z1) + std::abs(d) + 1.0);
	const amrex::Real tol = 1.0e-12 * scale;

	const amrex::GpuArray<Point, 8> verts{
	    Point{x0, y0, z0}, Point{x1, y0, z0}, Point{x0, y1, z0}, Point{x1, y1, z0},
	    Point{x0, y0, z1}, Point{x1, y0, z1}, Point{x0, y1, z1}, Point{x1, y1, z1}};
	const amrex::GpuArray<Edge, 12> edges{
	    Edge{0, 1}, Edge{2, 3}, Edge{4, 5}, Edge{6, 7}, Edge{0, 2}, Edge{1, 3},
	    Edge{4, 6}, Edge{5, 7}, Edge{0, 4}, Edge{1, 5}, Edge{2, 6}, Edge{3, 7}};

	amrex::GpuArray<Point, 16> pts{};
	int npts = 0;

	for (auto const &edge : edges) {
		const int i0 = edge[0];
		const int i1 = edge[1];
		const amrex::Real p0x = verts[i0][0];
		const amrex::Real p0y = verts[i0][1];
		const amrex::Real p0z = verts[i0][2];
		const amrex::Real p1x = verts[i1][0];
		const amrex::Real p1y = verts[i1][1];
		const amrex::Real p1z = verts[i1][2];

		amrex::Real f0 = nx * p0x + ny * p0y + nz * p0z - d;
		amrex::Real f1 = nx * p1x + ny * p1y + nz * p1z - d;
		if (std::abs(f0) <= tol) {
			f0 = 0.0;
		}
		if (std::abs(f1) <= tol) {
			f1 = 0.0;
		}

		if (f0 == 0.0 && f1 == 0.0) {
			addPointUnique(pts, npts, p0x, p0y, p0z, tol);
			addPointUnique(pts, npts, p1x, p1y, p1z, tol);
			continue;
		}
		if (f0 == 0.0) {
			addPointUnique(pts, npts, p0x, p0y, p0z, tol);
			continue;
		}
		if (f1 == 0.0) {
			addPointUnique(pts, npts, p1x, p1y, p1z, tol);
			continue;
		}
		if ((f0 < 0.0 && f1 > 0.0) || (f0 > 0.0 && f1 < 0.0)) {
			const amrex::Real t = f0 / (f0 - f1);
			const amrex::Real x = p0x + t * (p1x - p0x);
			const amrex::Real y = p0y + t * (p1y - p0y);
			const amrex::Real z = p0z + t * (p1z - p0z);
			addPointUnique(pts, npts, x, y, z, tol);
		}
	}

	if (npts < 3) {
		return 0.0;
	}

	amrex::Real cx = 0.0;
	amrex::Real cy = 0.0;
	amrex::Real cz = 0.0;
	for (int i = 0; i < npts; ++i) {
		cx += pts[i][0];
		cy += pts[i][1];
		cz += pts[i][2];
	}
	cx /= static_cast<amrex::Real>(npts);
	cy /= static_cast<amrex::Real>(npts);
	cz /= static_cast<amrex::Real>(npts);

	// Build an orthonormal basis (e1,e2) spanning the plane.
	amrex::Real ax = 1.0;
	amrex::Real ay = 0.0;
	amrex::Real az = 0.0;
	if (std::abs(nx) > 0.9) {
		ax = 0.0;
		ay = 1.0;
		az = 0.0;
	}
	amrex::Real e1x = ny * az - nz * ay;
	amrex::Real e1y = nz * ax - nx * az;
	amrex::Real e1z = nx * ay - ny * ax;
	const amrex::Real e1norm = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
	if (e1norm <= 0.0) {
		return 0.0;
	}
	e1x /= e1norm;
	e1y /= e1norm;
	e1z /= e1norm;

	const amrex::Real e2x = ny * e1z - nz * e1y;
	const amrex::Real e2y = nz * e1x - nx * e1z;
	const amrex::Real e2z = nx * e1y - ny * e1x;

	amrex::GpuArray<amrex::Real, 16> u{};
	amrex::GpuArray<amrex::Real, 16> v{};
	amrex::GpuArray<amrex::Real, 16> ang{};
	for (int i = 0; i < npts; ++i) {
		const amrex::Real rx = pts[i][0] - cx;
		const amrex::Real ry = pts[i][1] - cy;
		const amrex::Real rz = pts[i][2] - cz;
		u[i] = rx * e1x + ry * e1y + rz * e1z;
		v[i] = rx * e2x + ry * e2y + rz * e2z;
		ang[i] = std::atan2(v[i], u[i]);
	}

	for (int i = 1; i < npts; ++i) {
		const amrex::Real key_ang = ang[i];
		const amrex::Real key_u = u[i];
		const amrex::Real key_v = v[i];
		int j = i - 1;
		while (j >= 0 && ang[j] > key_ang) {
			ang[j + 1] = ang[j];
			u[j + 1] = u[j];
			v[j + 1] = v[j];
			--j;
		}
		ang[j + 1] = key_ang;
		u[j + 1] = key_u;
		v[j + 1] = key_v;
	}

	amrex::Real area2 = 0.0;
	for (int i = 0; i < npts; ++i) {
		const int j = (i + 1 < npts) ? (i + 1) : 0;
		area2 += u[i] * v[j] - v[i] * u[j];
	}
	return 0.5 * std::abs(area2);
}

} // namespace detail

AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto sphericalSectionAreaInCell(amrex::Real const R, amrex::Real const x0, amrex::Real const x1, amrex::Real const y0,
								    amrex::Real const y1, amrex::Real const z0, amrex::Real const z1) -> amrex::Real
{
	// Approximate the spherical section area by the exact area of the tangent
	// plane section through the cell, using the sphere normal at the cell center.
	const amrex::Real R2 = R * R;
	const amrex::Real r2_min = detail::minDistSqToInterval(x0, x1) + detail::minDistSqToInterval(y0, y1) + detail::minDistSqToInterval(z0, z1);
	const amrex::Real r2_max = detail::maxDistSqToInterval(x0, x1) + detail::maxDistSqToInterval(y0, y1) + detail::maxDistSqToInterval(z0, z1);
	if (R2 < r2_min || R2 > r2_max) {
		return 0.0;
	}

	const amrex::Real dx = x1 - x0;
	const amrex::Real dy = y1 - y0;
	const amrex::Real dz = z1 - z0;
	const amrex::Real vol = dx * dy * dz;
	if (vol <= 0.0) {
		return 0.0;
	}

	const amrex::Real xc = 0.5 * (x0 + x1);
	const amrex::Real yc = 0.5 * (y0 + y1);
	const amrex::Real zc = 0.5 * (z0 + z1);
	const amrex::Real rc = std::sqrt(xc * xc + yc * yc + zc * zc);
	if (rc <= 0.0) {
		return 0.0;
	}

	const amrex::Real nx = xc / rc;
	const amrex::Real ny = yc / rc;
	const amrex::Real nz = zc / rc;
	return detail::planeBoxSectionArea(x0, x1, y0, y1, z0, z1, nx, ny, nz, R);
}

} // namespace quokka::math

#endif // SPHERICAL_GEOMETRY_HPP_
