// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef GOW_TABLES_HPP_
#define GOW_TABLES_HPP_
/// \file tables.hpp
/// \brief The table lookups of the GOW network, which the generated network.hpp calls: CO rotational cooling (Omukai
/// et al. 2010 fits as modified by Gong et al. 2017), the gas-dust coupling psi_gd, and the line shielding of C
/// photoionization and CO photodissociation. Written by hand; regenerate.sh keeps this file.
///
/// Each lookup takes and returns physical values. The *_partial_N functions are the derivatives with respect to the
/// lookup's argument N, which the analytic Jacobian calls. The shielding lookups have none: the column densities are
/// constant during a burn. The tables are read from gow_tables.hdf5 (src/networks/GOW_config/make_tables.py) by
/// LoadTables(), and clamp to their edges outside their range.

#include <array>

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "util/DataTable.hpp"

namespace quokka::networks::GOW
{

/// output indices of the "co" table
enum CoOutput : int { co_llte = 0, co_nhalf = 1, co_alpha = 2 }; // NOLINT(cppcoreguidelines-use-enum-class)
/// output indices of the shielding tables: shielding by H2, and self-shielding
enum ShieldingOutput : int { shielding_by_h2 = 0, shielding_self = 1 }; // NOLINT(cppcoreguidelines-use-enum-class)

/// The table views, for one memory space.
struct Tables {
	quokka::DataTableGpuConst<1, 1> l0;	   ///< L0(T) [erg cm^3 s^-1]
	quokka::DataTableGpuConst<2, 3> co;	   ///< L_LTE, n_1/2, alpha (T, N_eff [cm^-2 / (km/s)])
	quokka::DataTableGpuConst<2, 1> dust;	   ///< psi_gd (n_H [cm^-3], T)
	quokka::DataTableGpuConst<1, 2> c_shield;  ///< shielding of C photoionization by H2 and by C (N [cm^-2])
	quokka::DataTableGpuConst<1, 2> co_shield; ///< shielding of CO photodissociation by H2 and by CO (N [cm^-2])
};

/// The views in host and in device memory.
struct TableRegistry {
	Tables host;
	Tables device;
};

/// set by LoadTables(); defined in tables.cpp
extern AMREX_GPU_MANAGED TableRegistry *table_registry; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

/// Read the tables from the file network.tables_file (default: gow_tables.hdf5 in the source tree).
void LoadTables();

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto GetTables() -> Tables const &
{
	AMREX_ASSERT(table_registry != nullptr);
	AMREX_IF_ON_DEVICE((return table_registry->device;))
	AMREX_IF_ON_HOST((return table_registry->host;))
}

// CO cooling

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto l0_co_interp1d(const amrex::Real T) -> amrex::Real
{
	return GetTables().l0.interpolate_single(std::array<amrex::Real, 1>{T}, 0);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto l0_co_interp1d_partial_0(const amrex::Real T) -> amrex::Real
{
	return GetTables().l0.partial_derivative(std::array<amrex::Real, 1>{T}, 0, 0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto llte_co_interp2d(const amrex::Real T, const amrex::Real Neff) -> amrex::Real
{
	return GetTables().co.interpolate_single(std::array<amrex::Real, 2>{T, Neff}, co_llte);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto llte_co_interp2d_partial_0(const amrex::Real T, const amrex::Real Neff) -> amrex::Real
{
	return GetTables().co.partial_derivative(std::array<amrex::Real, 2>{T, Neff}, 0, co_llte);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto llte_co_interp2d_partial_1(const amrex::Real T, const amrex::Real Neff) -> amrex::Real
{
	return GetTables().co.partial_derivative(std::array<amrex::Real, 2>{T, Neff}, 1, co_llte);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto nhalf_co_interp2d(const amrex::Real T, const amrex::Real Neff) -> amrex::Real
{
	return GetTables().co.interpolate_single(std::array<amrex::Real, 2>{T, Neff}, co_nhalf);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto nhalf_co_interp2d_partial_0(const amrex::Real T, const amrex::Real Neff) -> amrex::Real
{
	return GetTables().co.partial_derivative(std::array<amrex::Real, 2>{T, Neff}, 0, co_nhalf);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto nhalf_co_interp2d_partial_1(const amrex::Real T, const amrex::Real Neff) -> amrex::Real
{
	return GetTables().co.partial_derivative(std::array<amrex::Real, 2>{T, Neff}, 1, co_nhalf);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto alpha_co_interp2d(const amrex::Real T, const amrex::Real Neff) -> amrex::Real
{
	return GetTables().co.interpolate_single(std::array<amrex::Real, 2>{T, Neff}, co_alpha);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto alpha_co_interp2d_partial_0(const amrex::Real T, const amrex::Real Neff) -> amrex::Real
{
	return GetTables().co.partial_derivative(std::array<amrex::Real, 2>{T, Neff}, 0, co_alpha);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto alpha_co_interp2d_partial_1(const amrex::Real T, const amrex::Real Neff) -> amrex::Real
{
	return GetTables().co.partial_derivative(std::array<amrex::Real, 2>{T, Neff}, 1, co_alpha);
}

// gas-dust coupling

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto psigd_coll_interp2d(const amrex::Real nH, const amrex::Real T) -> amrex::Real
{
	return GetTables().dust.interpolate_single(std::array<amrex::Real, 2>{nH, T}, 0);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto psigd_coll_interp2d_partial_0(const amrex::Real nH, const amrex::Real T) -> amrex::Real
{
	return GetTables().dust.partial_derivative(std::array<amrex::Real, 2>{nH, T}, 0, 0);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto psigd_coll_interp2d_partial_1(const amrex::Real nH, const amrex::Real T) -> amrex::Real
{
	return GetTables().dust.partial_derivative(std::array<amrex::Real, 2>{nH, T}, 1, 0);
}

// line shielding; reactions 8 (C photoionization) and 10 (CO photodissociation)

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interp_8_shielding_H2(const amrex::Real N_H2) -> amrex::Real
{
	return GetTables().c_shield.interpolate_single(std::array<amrex::Real, 1>{N_H2}, shielding_by_h2);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interp_8_shielding_self(const amrex::Real N_C) -> amrex::Real
{
	return GetTables().c_shield.interpolate_single(std::array<amrex::Real, 1>{N_C}, shielding_self);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interp_10_shielding_H2(const amrex::Real N_H2) -> amrex::Real
{
	return GetTables().co_shield.interpolate_single(std::array<amrex::Real, 1>{N_H2}, shielding_by_h2);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interp_10_shielding_self(const amrex::Real N_CO) -> amrex::Real
{
	return GetTables().co_shield.interpolate_single(std::array<amrex::Real, 1>{N_CO}, shielding_self);
}

} // namespace quokka::networks::GOW

#endif // GOW_TABLES_HPP_
