// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef GRACKLE_NT_TABLE_HPP_
#define GRACKLE_NT_TABLE_HPP_

/// \file GrackleNTTable.hpp
/// \brief The Grackle cooling table on its native (nH, T) grid, for the photochemistry network.
///
/// The table (written by extern/cooling/make_grackle_nT_table.py) holds two outputs on a log-log grid: the net cooling function
/// Lambda_net(nH, T) in erg cm^3 s^-1, which can be negative, and the CIE electron fraction xe_CIE(nH, T) per H nucleus. The lookup
/// interpolates bilinearly in (log nH, log T), clamps its inputs to the table range, and returns the derivatives of the interpolant.

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "util/DataTable.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <string>

namespace quokka::ISMCooling
{

using amrex::Real;

/// The table's handles for use in kernels, with the attributes of the file.
struct GrackleNTGpuConst {
	quokka::DataTableGpuConst<2, 2> table; ///< outputs: 0 Lambda_net, 1 xe_CIE
	Real zmet;			       ///< metallicity of the table, in units of Z'
	Real H_mass_fraction;		       ///< hydrogen mass fraction assumed by the table
};

/// The scalar attributes stored in the table file.
struct GrackleNTAttributes {
	Real zmet;
	Real H_mass_fraction;
};

/// The result of a lookup. The derivatives are those of the interpolant, with respect to T and nH (not to their logarithms).
struct GrackleNTValues {
	Real Lambda;	 ///< Lambda_net [erg cm^3 s^-1]
	Real dLambda_dT; ///< d Lambda_net / dT
	Real xe;	 ///< xe_CIE, clamped to >= 0
	Real dxe_dT;	 ///< d xe / dT, zero where xe is clamped
	Real dxe_dnH;	 ///< d xe / d nH, zero where xe is clamped
};

/// The managed-memory registry of the table, so that device code can reach it without passing it through every call.
struct GrackleNTRegistry {
	GrackleNTGpuConst host;
	GrackleNTGpuConst device;
};

extern AMREX_GPU_MANAGED GrackleNTRegistry *g_grackle_nt_registry; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

/// Read the table (group tab1) from an HDF5 file, with `log` spacing in both inputs and `linear` interpolation of both outputs.
/// Aborts, naming the file, if it is missing or does not have two log-spaced inputs and two outputs.
auto ReadGrackleNTTable(std::string const &path) -> std::unique_ptr<quokka::DataTable<2, 2>>;

/// Read the attributes zmet and H_mass_fraction of the table file.
auto ReadGrackleNTAttributes(std::string const &path) -> GrackleNTAttributes;

/// Register (or re-register) the table handles. Allocates the registry on the first call. The DataTable they refer to must outlive their use.
void RegisterGrackleNTTable(GrackleNTGpuConst host, GrackleNTGpuConst device);

/// The registered table: the device handles in device code, the host handles in host code. Null if nothing is registered.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto GetGrackleNTTable() -> GrackleNTGpuConst const *
{
	if (g_grackle_nt_registry == nullptr) {
		return nullptr;
	}
	AMREX_IF_ON_DEVICE((return &g_grackle_nt_registry->device;))
	AMREX_IF_ON_HOST((return &g_grackle_nt_registry->host;))
}

/// Look up Lambda_net and xe_CIE at (nH, T), with their derivatives.
///
/// nH [cm^-3] and T [K] are clamped to the table range, so the result is finite for any positive input. Inside the range, the T and nH derivatives
/// are centred differences of the interpolant in ln T and ln nH, with a relative step of 1e-3, one-sided close to the edges. Outside the range
/// the clamped value is constant in that coordinate, so its derivative is 0 (dLambda_dT and dxe_dT for T, dxe_dnH for nH).
/// xe is clamped to >= 0 (the table is slightly negative, down to about -0.008, in the cold dense corner), and its derivatives are 0 where it is clamped.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto LookupGrackleNT(Real nH, Real T, GrackleNTGpuConst const &t) -> GrackleNTValues
{
	constexpr Real step = 1.0e-3;
	constexpr Real tiny = std::numeric_limits<Real>::min();
	const auto &tab = t.table;

	const Real lnnH_lo = tab.coord_min[0];
	const Real lnnH_hi = tab.coord_max[0];
	const Real lnT_lo = tab.coord_min[1];
	const Real lnT_hi = tab.coord_max[1];
	const Real lnnH_in = std::log(amrex::max(nH, tiny));
	const Real lnT_in = std::log(amrex::max(T, tiny));
	const bool nH_clamped = (lnnH_in < lnnH_lo) || (lnnH_in > lnnH_hi);
	const bool T_clamped = (lnT_in < lnT_lo) || (lnT_in > lnT_hi);
	const Real lnnH = amrex::min(amrex::max(lnnH_in, lnnH_lo), lnnH_hi);
	const Real lnT = amrex::min(amrex::max(lnT_in, lnT_lo), lnT_hi);

	// the stencil in ln T and ln nH, shortened (one-sided) at the edges
	const Real lnT_p = amrex::min(lnT + step, lnT_hi);
	const Real lnT_m = amrex::max(lnT - step, lnT_lo);
	const Real lnnH_p = amrex::min(lnnH + step, lnnH_hi);
	const Real lnnH_m = amrex::max(lnnH - step, lnnH_lo);

	const Real nH_c = std::exp(lnnH);
	const Real T_c = std::exp(lnT);
	const auto centre = tab.interpolate({nH_c, T_c});
	const auto T_plus = tab.interpolate({nH_c, std::exp(lnT_p)});
	const auto T_minus = tab.interpolate({nH_c, std::exp(lnT_m)});
	const auto nH_plus = tab.interpolate({std::exp(lnnH_p), T_c});
	const auto nH_minus = tab.interpolate({std::exp(lnnH_m), T_c});

	const Real inv_dT = 1. / ((lnT_p - lnT_m) * T_c);
	const Real inv_dnH = 1. / ((lnnH_p - lnnH_m) * nH_c);

	GrackleNTValues v{};
	v.Lambda = centre[0];
	v.dLambda_dT = T_clamped ? 0. : (T_plus[0] - T_minus[0]) * inv_dT;
	if (centre[1] > 0.) {
		v.xe = centre[1];
		v.dxe_dT = T_clamped ? 0. : (T_plus[1] - T_minus[1]) * inv_dT;
		v.dxe_dnH = nH_clamped ? 0. : (nH_plus[1] - nH_minus[1]) * inv_dnH;
	}
	return v;
}

} // namespace quokka::ISMCooling

#endif // GRACKLE_NT_TABLE_HPP_
