// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef PHOTOCHEMISTRY_HOT_GAS_HPP_ // NOLINT
#define PHOTOCHEMISTRY_HOT_GAS_HPP_

/// \file photochemistry_hot_gas.hpp
/// \brief The hot-gas regime of the ISM photochemistry cooling (SPEC §5.1), built with PHOTOCHEM_ISM_COOLING.
///
/// Once per hydro step and level, before the radiation substeps, BuildHotMask labels each cell hot if its photochemistry-EOS temperature is at or
/// above hot_gas_threshold_temperature, and ComputeHotGasCooling sets each hot cell fully ionized and integrates its internal energy over the hydro
/// step with the Grackle (nH, T) table rate, photoelectric heating and cosmic-ray heating. computePhotoChemistry then skips the burn in hot cells and
/// attenuates their chemical bands by dust only.

#include <array>
#include <cstdio>
#include <limits>

#include "AMReX.H"
#include "AMReX_Array4.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MFIter.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_iMultiFab.H"

#include "cooling/GrackleNTTable.hpp"
#include "cooling/ISMCooling.hpp"
#include "cooling/ResampledCooling.hpp"
#include "hydro/EOS.hpp"
#include "math/ODEIntegrate.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"

#include "actual_eos_data.H"
#include "actual_rhs.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network_properties.H"

namespace quokka::photochemistry
{

/// Cells at or above this temperature [K] are hot (SPEC §5.1).
constexpr amrex::Real hot_gas_threshold_temperature = 2.0e4;

namespace detail
{
static_assert(NumSpec == 3, "the hot-gas step assumes the pure-H network of species H, H+ and e-");

/// The gas internal energy density [erg cm^-3] of the cell, as computePhotoChemistry takes it: total minus kinetic minus magnetic energy.
/// Device-only, like ComputeCellCenteredMagneticEnergy.
template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto CellInternalEnergy(amrex::Array4<const amrex::Real> const &state, int i, int j, int k,
							    std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const &cons_fc) -> amrex::Real
{
	const amrex::Real rho = state(i, j, k, RadSystem<problem_t>::gasDensity_index);
	const amrex::Real xmom = state(i, j, k, RadSystem<problem_t>::x1GasMomentum_index);
	const amrex::Real ymom = state(i, j, k, RadSystem<problem_t>::x2GasMomentum_index);
	const amrex::Real zmom = state(i, j, k, RadSystem<problem_t>::x3GasMomentum_index);
	const amrex::Real Ener = state(i, j, k, RadSystem<problem_t>::gasEnergy_index);
	const amrex::Real Emag = ComputeCellCenteredMagneticEnergy<problem_t>(i, j, k, cons_fc);
	return ::quokka::EOS<problem_t>::ComputeEintFromEgas(rho, xmom, ymom, zmom, Ener, Emag);
}

/// The temperature [K] of the photochemistry EOS for density rho, internal energy density Eint and species number densities xn [cm^-3].
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto PhotochemistryTemperature(amrex::Real rho, amrex::Real Eint, amrex::GpuArray<amrex::Real, NumSpec> const &xn)
    -> amrex::Real
{
	burn_t s;
	s.rho = rho;
	s.e = Eint / rho;
	for (int nn = 0; nn < NumSpec; ++nn) {
		s.xn[nn] = xn[nn];
	}
	eos(eos_input_re, s);
	return s.T;
}

/// The cell's species number densities [cm^-3], from its mass scalars.
template <typename problem_t>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto CellNumberDensities(amrex::Array4<const amrex::Real> const &state, int i, int j, int k)
    -> amrex::GpuArray<amrex::Real, NumSpec>
{
	amrex::GpuArray<amrex::Real, NumSpec> xn{};
	for (int nn = 0; nn < NumSpec; ++nn) {
		xn[nn] = state(i, j, k, RadSystem<problem_t>::scalar0_index + nn) / spmasses[nn];
	}
	return xn;
}

/// The right-hand side dE/dt = -n_H^2 Lambda_net(n_H, T) + G_PE + G_CR [erg cm^-3 s^-1] of a fully ionized hot cell at fixed density, for
/// rk_adaptive_integrate. T comes from the photochemistry EOS at each call. The photoelectric heating uses n_e^max with n_HI = 0 and n_H+ = n_H; the
/// cosmic-ray heating, proportional to n_HI, is zero. Each heating term is off if its network switch (pe_heating_switch, cr_heating_switch) is 0,
/// as in cold cells.
struct HotGasCoolingRHS {
	amrex::Real rho;			  ///< gas density [g cm^-3]
	amrex::Real nH;				  ///< hydrogen number density [cm^-3]
	amrex::GpuArray<amrex::Real, NumSpec> xn; ///< species number densities, fully ionized [cm^-3]
	amrex::Real G0;				  ///< FUV field [Habing units]
	amrex::Real Zp;				  ///< metallicity [Z_sun]
	amrex::Real zetaHI;			  ///< cosmic-ray ionization rate of HI [s^-1]
	bool pe_heating;			  ///< network.pe_heating_switch != 0
	bool cr_heating;			  ///< network.cr_heating_switch != 0

	/// The internal energy density [erg cm^-3] at temperature T.
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto InternalEnergy(amrex::Real T) const -> amrex::Real
	{
		burn_t s;
		s.rho = rho;
		s.T = T;
		for (int nn = 0; nn < NumSpec; ++nn) {
			s.xn[nn] = xn[nn];
		}
		eos(eos_input_rt, s);
		return s.e * rho;
	}

	/// The net heating rate [erg cm^-3 s^-1] at temperature T.
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto NetHeating(amrex::Real T) const -> amrex::Real
	{
		namespace ISM = quokka::ISMCooling;
		constexpr amrex::Real nHI = 0.0;
		constexpr amrex::Real yHe = 0.0;	  // SPEC §5.0
		constexpr amrex::Real ne_floor = 1.0e-30; // that of get_ism_energy_terms
		const auto nt = ISM::LookupGrackleNT(nH, T, *ISM::GetGrackleNTTable());
		const amrex::Real ne = amrex::max(ISM::ElectronDensityMax(nHI, nH, T, nt.xe, nt.dxe_dT, nt.dxe_dnH, Zp, zetaHI,
									  get_recombination_coefficient(T), dRecombination_coefficient_dT(T))
						      .ne,
						  ne_floor);
		const amrex::Real pe = pe_heating ? ISM::PEHeating(nHI, nH, ne, T, G0, Zp).val : 0.0;
		const amrex::Real cr = cr_heating ? ISM::CRHeating(nHI, zetaHI, yHe).val : 0.0;
		return -nH * nH * nt.Lambda + pe + cr;
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(amrex::Real /*t*/, quokka::valarray<amrex::Real, 1> &y,
								 quokka::valarray<amrex::Real, 1> &ydot) const -> int
	{
		if (!(y[0] > 0.0)) {
			return 1; // no temperature: the integrator retries with a smaller step
		}
		ydot[0] = NetHeating(PhotochemistryTemperature(rho, y[0], xn));
		return 0;
	}
};
} // namespace detail

/// Label each cell hot (1) or cold (0). A cell is hot if rho >= min_density_allowed and its temperature, from the photochemistry EOS with the cell's
/// species, is at least hot_gas_threshold_temperature.
template <typename problem_t>
void BuildHotMask(amrex::MultiFab const &state, std::array<amrex::MultiFab const *, AMREX_SPACEDIM> const &fc, amrex::iMultiFab &hotMask,
		  amrex::Real min_density_allowed)
{
	const BL_PROFILE("PhotoChemistry::BuildHotMask()");
	for (amrex::MFIter iter(state); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &s = state.const_array(iter);
		auto const &mask = hotMask.array(iter);
		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				cons_fc[dir] = fc[dir]->const_array(iter);
			}
		}
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real rho = s(i, j, k, RadSystem<problem_t>::gasDensity_index);
			if (rho < min_density_allowed) {
				mask(i, j, k) = 0;
				return;
			}
			const amrex::Real Eint = detail::CellInternalEnergy<problem_t>(s, i, j, k, cons_fc);
			const amrex::Real T = detail::PhotochemistryTemperature(rho, Eint, detail::CellNumberDensities<problem_t>(s, i, j, k));
			mask(i, j, k) = (T >= hot_gas_threshold_temperature) ? 1 : 0;
		});
	}
}

/// The hot-cell step over one hydro step (SPEC §5.1). In each cell with hotMask = 1:
///  1. move all H to H+ and set n_e = n_H+, keeping the gas internal energy;
///  2. integrate dE_int/dt = -n_H^2 Lambda_net(n_H, T) + G_PE + G_CR over dt_hydro at fixed density with rk_adaptive_integrate, T coming from the
///     photochemistry EOS at each evaluation and the Grackle rate applying also below the threshold. Each heating term is gated by its network
///     switch. If network.energy_switch is 0, the integration is skipped and E_int is left unchanged;
///  3. clamp E_int to the energy at tempFloor (if tempFloor > 0).
/// The change of E_int is added to the internal and total gas energy. Aborts, printing the cell, if the integration does not finish in
/// maxStepsODEIntegrate steps: the step runs after hydro, so there is no retry.
/// \param G0Rate The FUV field G0 [Habing units] per cell.
template <typename problem_t>
void ComputeHotGasCooling(amrex::MultiFab &state, std::array<amrex::MultiFab const *, AMREX_SPACEDIM> const &fc, amrex::iMultiFab const &hotMask,
			  amrex::Real dt_hydro, quokka::ResampledCooling::ExternalHeatingRate const &G0Rate, amrex::Real tempFloor)
{
	const BL_PROFILE("PhotoChemistry::ComputeHotGasCooling()");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(quokka::ISMCooling::GetGrackleNTTable() != nullptr,
					 "ComputeHotGasCooling: the Grackle (nH, T) table is not registered (ism_cooling.grackle_nT_table)");

	constexpr amrex::Real rtol = 1.0e-4;	   // as ResampledCooling::computeCooling
	constexpr amrex::Real reltol_floor = 0.01; // absolute tolerance, relative to the energy at tempFloor
	const amrex::Real Zp = network_rp::ism_metallicity;
	const amrex::Real zetaHI = network_rp::cr_ionization_rate_HI;
	const bool pe_heating = (network_rp::pe_heating_switch != 0);
	const bool cr_heating = (network_rp::cr_heating_switch != 0);
	const bool energy_on = (network_rp::energy_switch != 0);

	for (amrex::MFIter iter(state); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &s = state.array(iter);
		auto const &mask = hotMask.const_array(iter);
		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				cons_fc[dir] = fc[dir]->const_array(iter);
			}
		}
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			if (mask(i, j, k) == 0) {
				return;
			}
			const amrex::Real rho = s(i, j, k, RadSystem<problem_t>::gasDensity_index);

			// 1. fully ionized, conserving the number of H nuclei; E_int is not touched
			const auto xn_old = detail::CellNumberDensities<problem_t>(s, i, j, k);
			const amrex::Real nH = xn_old[Species::H] + xn_old[Species::H_p];
			amrex::GpuArray<amrex::Real, NumSpec> xn{};
			xn[Species::H] = 0.0;
			xn[Species::H_p] = nH;
			xn[Species::e] = nH;
			for (int nn = 0; nn < NumSpec; ++nn) {
				s(i, j, k, RadSystem<problem_t>::scalar0_index + nn) = xn[nn] * spmasses[nn];
			}

			// 2. integrate the internal energy over the hydro step (skipped if network.energy_switch = 0)
			if (!energy_on) {
				return;
			}
			const amrex::Real Eint = detail::CellInternalEnergy<problem_t>(s, i, j, k, cons_fc);
			const detail::HotGasCoolingRHS rhs{rho, nH, xn, G0Rate(i, j, k), Zp, zetaHI, pe_heating, cr_heating};
			const amrex::Real Eint_floor = (tempFloor > 0.0) ? rhs.InternalEnergy(tempFloor) : 0.0;
			quokka::valarray<amrex::Real, 1> y = {Eint};
			quokka::valarray<amrex::Real, 1> const abstol = {reltol_floor * amrex::max(Eint_floor, std::numeric_limits<amrex::Real>::min())};
			int nsteps = 0;
			rk_adaptive_integrate(rhs, 0, y, dt_hydro, rtol, abstol, nsteps);
			if (nsteps >= maxStepsODEIntegrate) {
				printf("ComputeHotGasCooling: max substeps exceeded! rho = %.17e, n_H = %.17e, Eint = %.17e, T = %.17e, G0 = %.17e, dt = "
				       "%.17e\n", // NOLINT
				       rho, nH, Eint, detail::PhotochemistryTemperature(rho, Eint, xn), rhs.G0, dt_hydro);
				amrex::Abort("ComputeHotGasCooling: the hot-cell cooling integration failed.");
			}

			// 3. temperature floor
			const amrex::Real Eint_new = (tempFloor > 0.0) ? amrex::max(y[0], Eint_floor) : y[0];
			const amrex::Real dEint = Eint_new - Eint;
			s(i, j, k, RadSystem<problem_t>::gasInternalEnergy_index) += dEint;
			s(i, j, k, RadSystem<problem_t>::gasEnergy_index) += dEint;
		});
	}
}

} // namespace quokka::photochemistry

#endif // PHOTOCHEMISTRY_HOT_GAS_HPP_
