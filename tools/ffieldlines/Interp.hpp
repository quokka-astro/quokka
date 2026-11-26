#ifndef FFIELDLINES_INTERP_HPP_
#define FFIELDLINES_INTERP_HPP_
/// \file Interp.hpp
/// \brief Device-safe trilinear sampling of cell-centered fields.

#include <cmath>

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include "FieldData.hpp"

namespace ffieldlines
{

using Vec3 = amrex::GpuArray<amrex::Real, 3>;

/// Interpolated values of all FieldData components at one point.
struct FieldSample {
	Vec3 b{};
	amrex::Real rho{};
	Vec3 m{};
	amrex::Real temp{};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto Dot(Vec3 const &a, Vec3 const &b) -> amrex::Real { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

/// Cell index containing position `x`, using the same convention as AMReX particle redistribution.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto CellIndex(Vec3 const &x, Vec3 const &plo, Vec3 const &dxi, amrex::IntVect const &domLo) -> amrex::IntVect
{
	return {static_cast<int>(std::floor((x[0] - plo[0]) * dxi[0])) + domLo[0], static_cast<int>(std::floor((x[1] - plo[1]) * dxi[1])) + domLo[1],
		static_cast<int>(std::floor((x[2] - plo[2]) * dxi[2])) + domLo[2]};
}

/// Trilinear interpolation of all components of `a` (cell-centered, NComp
/// components) at position `x`. Returns false when the 8-cell stencil is not
/// contained in `a` (including ghost cells).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto SampleFields(amrex::Array4<amrex::Real const> const &a, Vec3 const &plo, Vec3 const &dxi,
							   amrex::IntVect const &domLo, Vec3 const &x, FieldSample &out) -> bool
{
	using namespace amrex::literals;

	amrex::GpuArray<int, 3> i0{};
	Vec3 w{};
	for (int d = 0; d < 3; ++d) {
		const amrex::Real xi = (x[d] - plo[d]) * dxi[d] - 0.5_rt;
		if (!(std::abs(xi) < 1.0e9_rt)) {
			return false; // non-finite or absurdly far away
		}
		const amrex::Real fl = std::floor(xi);
		i0[d] = static_cast<int>(fl) + domLo[d];
		w[d] = xi - fl;
	}
	if (!a.contains(i0[0], i0[1], i0[2]) || !a.contains(i0[0] + 1, i0[1] + 1, i0[2] + 1)) {
		return false;
	}

	amrex::GpuArray<amrex::Real, NComp> v{};
	for (int dk = 0; dk < 2; ++dk) {
		const amrex::Real wk = (dk == 0) ? (1.0_rt - w[2]) : w[2];
		for (int dj = 0; dj < 2; ++dj) {
			const amrex::Real wj = (dj == 0) ? (1.0_rt - w[1]) : w[1];
			for (int di = 0; di < 2; ++di) {
				const amrex::Real wi = (di == 0) ? (1.0_rt - w[0]) : w[0];
				const amrex::Real weight = wi * wj * wk;
				for (int n = 0; n < NComp; ++n) {
					v[n] += weight * a(i0[0] + di, i0[1] + dj, i0[2] + dk, n);
				}
			}
		}
	}
	out.b = {v[Bx], v[By], v[Bz]};
	out.rho = v[Rho];
	out.m = {v[Mx], v[My], v[Mz]};
	out.temp = v[Temp];
	return true;
}

} // namespace ffieldlines

#endif // FFIELDLINES_INTERP_HPP_
