// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testISMCoolingSlab.cpp
/// \brief A beam of ionizing photons through a uniform, fully ionized slab of the ISM photochemistry network, with dust absorption of ionizing
/// photons and no gas-dust thermal coupling (ISM_Traits::enable_dust_gas_thermal_coupling_model = false). The absorbed energy leaves the
/// simulation and the gas only gets the photon momentum. At t = 5 L / c_hat the test checks that
///  - the photon density follows N(x) / N(0) = exp(-kappa rho x),
///  - the gas internal energy is unchanged, and
///  - the gas momentum equals the integral of rho kappa F / c over the volume and time.

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "QuokkaSimulation.hpp"
#include "cooling/GrackleNTTable.hpp"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include <algorithm>
#include <cmath>
#include <string>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct ISMCoolingSlab {};

constexpr double c_hat = C::c_light;
constexpr double Erad_floor_ = 1.0e-60; // erg cm^-3

template <> struct quokka::EOS_Traits<ISMCoolingSlab> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<ISMCoolingSlab> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int numMassScalars = NumSpec;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr int nGroups = NumThermalBands + NumChemBands;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<ISMCoolingSlab> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr int beta_order = 1;
	static constexpr auto ChemBandsPowerLawIndex() { return ChemBandsPowerLawIndex_; }
	static constexpr auto ChemBands() { return ChemBandsHeader(); }
};

// Dust absorbs ionizing photons, but there are no thermal groups and no gas-dust coupling: the absorbed energy leaves the simulation.
template <> struct ISM_Traits<ISMCoolingSlab> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = false;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = false;
	static constexpr bool thermal_band_photochemistry = false;
	static constexpr bool dust_chemical_band_absorption = true;
};

static_assert(NumThermalBands == 0, "ISMCoolingSlab has no thermal radiation groups");

template <> struct SimulationData<ISMCoolingSlab> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real n_H{};
	amrex::Real n_HI_init{};
	amrex::Real flux_ion{};
	amrex::Real eps_ion{};
	amrex::Real kappa_rho{};       ///< dust_kappa * rho [cm^-1]
	amrex::Real Eint_init{};       ///< initial gas internal energy density [erg cm^-3]
	amrex::Real momentum_budget{}; ///< integral of rho kappa F / c over the volume and time, per unit cross-section [g cm s^-1 cm^-2]
};

template <> void QuokkaSimulation<ISMCoolingSlab>::preCalculateInitialConditions()
{
	// initialize microphysics routines
	init_extern_parameters();

	amrex::ParmParse const pp("slab");
	userData_.small_temp = 1e-2;
	userData_.small_dens = 1e-60;
	userData_.temperature = 1.0e4;
	userData_.n_H = 1.0_rt;
	userData_.n_HI_init = 1.0e-12_rt;
	userData_.flux_ion = 1.0e3_rt;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("n_H", userData_.n_H);
	pp.query("n_HI_init", userData_.n_HI_init);
	pp.query("flux_ion", userData_.flux_ion);

	userData_.eps_ion = RadSystem<ISMCoolingSlab>::GetChemBandQuanta(0);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
}

// The chemical band couples to the gas only through photochemistry, so its Planck and flux-mean opacities are zero.
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ISMCoolingSlab>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ISMCoolingSlab>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <>
void RadSystem<ISMCoolingSlab>::AddRadSource(array_t &radEnergy, array_t &reducedFlux, const amrex::Box &indexRange,
					     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real /*time*/)
{
	// The beam enters at x = prob_lo, through the first cell, moving in +x.
	amrex::ParmParse const pp("slab");
	amrex::Real flux_ion = 0.0_rt;
	pp.query("flux_ion", flux_ion);

	const amrex::Real eps_ion = RadSystem<ISMCoolingSlab>::GetChemBandQuanta(0);
	const amrex::Real src_ionizing = flux_ion * eps_ion / dx[0];
	const amrex::Real x_edge = prob_lo[0] + dx[0];
	const amrex::Real x_lo = prob_lo[0];
	const amrex::Real cell_length = dx[0];

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const amrex::Real x = x_lo + (static_cast<amrex::Real>(i) + 0.5_rt) * cell_length;
		const bool in_source = x < x_edge;
		radEnergy(i, j, k, 0) = in_source ? src_ionizing : 0.0_rt;
		reducedFlux(i, j, k, 0) = in_source ? 1.0_rt : 0.0_rt;
		reducedFlux(i, j, k, 1) = 0.0_rt;
		reducedFlux(i, j, k, 2) = 0.0_rt;
	});
}

template <> void QuokkaSimulation<ISMCoolingSlab>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	burn_t state;
	std::array<Real, NumSpec> numdens = {-1.0};
	numdens[Species::H] = userData_.n_HI_init;
	numdens[Species::H_p] = userData_.n_H - userData_.n_HI_init;
	numdens[Species::e] = numdens[Species::H_p];

	state.T = userData_.temperature;
	Real rhotot = 0.0_rt;
	for (int n = 0; n < NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * spmasses[n];
	}
	state.rho = rhotot;

	eos(eos_input_rt, state);
	const auto Egas0 = state.e * rhotot;
	userData_.Eint_init = Egas0;
	userData_.kappa_rho = network_rp::dust_kappa * rhotot;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, RadSystem<ISMCoolingSlab>::radEnergy_index) = Erad_floor_;
		state_cc(i, j, k, RadSystem<ISMCoolingSlab>::x1RadFlux_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<ISMCoolingSlab>::x2RadFlux_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<ISMCoolingSlab>::x3RadFlux_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<ISMCoolingSlab>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<ISMCoolingSlab>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<ISMCoolingSlab>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<ISMCoolingSlab>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<ISMCoolingSlab>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<ISMCoolingSlab>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<ISMCoolingSlab>::scalar0_index + nn) = state.xn[nn] * spmasses[nn];
		}
	});
}

template <> void QuokkaSimulation<ISMCoolingSlab>::computeAfterTimestep()
{
	// the momentum the dust absorbs from the beam: integral of kappa rho F / c over the volume and the step, per unit cross-section
	const amrex::Real kappa = network_rp::dust_kappa;
	const amrex::Real transverse_area = (geom[0].ProbHi(1) - geom[0].ProbLo(1)) * (geom[0].ProbHi(2) - geom[0].ProbLo(2));
	const amrex::Real absorbed = computeVolumeIntegral(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state,
				 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const & /*state_fc*/) noexcept -> amrex::Real {
		    return kappa * state(i, j, k, HydroSystem<ISMCoolingSlab>::density_index) * state(i, j, k, RadSystem<ISMCoolingSlab>::x1RadFlux_index) /
			   C::c_light;
	    });
	userData_.momentum_budget += dt_[0] * absorbed / transverse_area;
}

auto problem_main() -> int
{
	const double CFL_number = 0.3;

	// nothing in the simulation registers the Grackle table yet, so the problem does; the network reads it only when energy_switch != 0
	amrex::ParmParse const pp_cooling("ism_cooling");
	std::string table_path;
	pp_cooling.get("grackle_nT_table", table_path);
	const auto table_data = quokka::ISMCooling::ReadGrackleNTTable(table_path);
	const auto table_attrs = quokka::ISMCooling::ReadGrackleNTAttributes(table_path);
	quokka::ISMCooling::RegisterGrackleNTTable({table_data->const_tables_host(), table_attrs.zmet, table_attrs.H_mass_fraction},
						   {table_data->const_tables(), table_attrs.zmet, table_attrs.H_mass_fraction});

	QuokkaSimulation<ISMCoolingSlab> sim;

	sim.setInitialConditions();
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.plotfileInterval_ = -1;

	sim.evolve();

	int status = 0;

	amrex::MultiFab const &state_mf = sim.state_new_cc_[0];
	const amrex::Real dx = sim.geom[0].CellSize(0);
	const amrex::Real x_lo = sim.geom[0].ProbLo(0);
	const amrex::Real kappa_rho = sim.userData_.kappa_rho;
	const amrex::Real Eint_init = sim.userData_.Eint_init;
	const amrex::Real transverse_area = (sim.geom[0].ProbHi(1) - sim.geom[0].ProbLo(1)) * (sim.geom[0].ProbHi(2) - sim.geom[0].ProbLo(2));
	amrex::Print() << "optical depth kappa rho L = " << kappa_rho * (sim.geom[0].ProbHi(0) - x_lo) << '\n';

	// E(x) / E(first cell) against exp(-kappa rho (x - x_first)), the gas internal energy and the gas momentum
	amrex::Real E_first = 0.0;
	amrex::Real momentum = 0.0;
	amrex::Real max_Eint_error = 0.0;
	for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		amrex::FArrayBox host_fab(box, state_mf.nComp(), amrex::The_Pinned_Arena());
		static_cast<void>(state_mf[mfi].template copyToMem<amrex::RunOn::Device>(box, 0, state_mf.nComp(), host_fab.dataPtr()));
		amrex::Gpu::synchronize();
		const auto state = host_fab.const_array();
		amrex::LoopOnCpu(box, [&](int i, int j, int k) noexcept {
			if (i == 0) {
				E_first = std::max(E_first, state(i, j, k, RadSystem<ISMCoolingSlab>::radEnergy_index));
			}
			momentum += state(i, j, k, HydroSystem<ISMCoolingSlab>::x1Momentum_index) * dx;
			max_Eint_error =
			    std::max(max_Eint_error, std::abs((state(i, j, k, RadSystem<ISMCoolingSlab>::gasInternalEnergy_index) / Eint_init) - 1.0));
		});
	}
	amrex::ParallelDescriptor::ReduceRealMax(E_first);
	amrex::ParallelDescriptor::ReduceRealMax(max_Eint_error);
	amrex::ParallelDescriptor::ReduceRealSum(momentum);

	amrex::Real max_profile_error = 0.0;
	for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		amrex::FArrayBox host_fab(box, state_mf.nComp(), amrex::The_Pinned_Arena());
		static_cast<void>(state_mf[mfi].template copyToMem<amrex::RunOn::Device>(box, 0, state_mf.nComp(), host_fab.dataPtr()));
		amrex::Gpu::synchronize();
		const auto state = host_fab.const_array();
		amrex::LoopOnCpu(box, [&](int i, int j, int k) noexcept {
			const amrex::Real ratio = state(i, j, k, RadSystem<ISMCoolingSlab>::radEnergy_index) / E_first;
			const amrex::Real expected = std::exp(-kappa_rho * static_cast<amrex::Real>(i) * dx);
			max_profile_error = std::max(max_profile_error, std::abs(ratio - expected));
		});
	}
	amrex::ParallelDescriptor::ReduceRealMax(max_profile_error);

	const amrex::Real momentum_ratio = (momentum / transverse_area) / sim.userData_.momentum_budget;

	const amrex::Real profile_tol = 2.0e-2;
	const amrex::Real Eint_tol = 1.0e-12;
	const amrex::Real momentum_lo = 0.95;
	const amrex::Real momentum_hi = 1.005;

	amrex::Print() << "Max |N(x)/N(0) - exp(-kappa rho x)| = " << max_profile_error << " (tolerance " << profile_tol << ")\n";
	amrex::Print() << "Max relative change of the gas internal energy = " << max_Eint_error << " (tolerance " << Eint_tol << ")\n";
	amrex::Print() << "Gas momentum / integral of rho kappa F / c dt = " << momentum_ratio << " (allowed [" << momentum_lo << ", " << momentum_hi << "])\n";

	if (!(max_profile_error <= profile_tol)) {
		amrex::Print() << "Test failed: the photon profile does not follow exp(-kappa rho x).\n";
		status = 1;
	}
	if (!(max_Eint_error <= Eint_tol)) {
		amrex::Print() << "Test failed: the gas internal energy changed.\n";
		status = 1;
	}
	if (!(momentum_ratio >= momentum_lo && momentum_ratio <= momentum_hi)) {
		amrex::Print() << "Test failed: the gas momentum does not match the absorbed photon momentum.\n";
		status = 1;
	}

	amrex::Print() << "Finished." << '\n';
	return status;
}
