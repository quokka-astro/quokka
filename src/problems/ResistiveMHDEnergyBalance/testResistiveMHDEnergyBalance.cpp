//==============================================================================
// Copyright 2026 Ben Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testResistiveMHDEnergyBalance.cpp
/// \brief Diagnostic for the resistive-MHD total-energy flux eta * J x B.

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <tuple>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Gpu.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "hydro/mhd_system.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"

struct ResistiveMHDEnergyBalance {
};

template <> struct quokka::EOS_Traits<ResistiveMHDEnergyBalance> {
	static constexpr double gamma = 2.0;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<ResistiveMHDEnergyBalance> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1;
	static constexpr bool is_mhd_enabled = true;
	static constexpr ResistivityModel resistivity_model = ResistivityModel::constant;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr double gamma_gas = quokka::EOS_Traits<ResistiveMHDEnergyBalance>::gamma;
constexpr double density0 = 1.0;
constexpr double total_pressure0 = 1.0;
constexpr double field_amplitude = 0.5;
constexpr double mode_number = 1.0;
constexpr double domain_length = 1.0;
constexpr double wavenumber = 2.0 * M_PI * mode_number / domain_length;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto initialB_y(const amrex::Real x) -> amrex::Real { return field_amplitude * std::sin(wavenumber * x); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto initialPressure(const amrex::Real x) -> amrex::Real
{
	const amrex::Real by = initialB_y(x);
	return total_pressure0 - 0.5 * by * by;
}

template <> void QuokkaSimulation<ResistiveMHDEnergyBalance>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp_cc = Physics_Indices<ResistiveMHDEnergyBalance>::nvarTotal_cc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0;
		}

		const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
		const amrex::Real pressure = initialPressure(x);
		const amrex::Real eint = pressure / (gamma_gas - 1.0);
		const amrex::Real by = initialB_y(x);
		const amrex::Real emag = 0.5 * by * by;

		state_cc(i, j, k, HydroSystem<ResistiveMHDEnergyBalance>::density_index) = density0;
		state_cc(i, j, k, HydroSystem<ResistiveMHDEnergyBalance>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<ResistiveMHDEnergyBalance>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<ResistiveMHDEnergyBalance>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<ResistiveMHDEnergyBalance>::internalEnergy_index) = eint;
		state_cc(i, j, k, HydroSystem<ResistiveMHDEnergyBalance>::energy_index) = eint + emag;
	});
}

template <> void QuokkaSimulation<ResistiveMHDEnergyBalance>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const int ncomp_fc = Physics_Indices<ResistiveMHDEnergyBalance>::nvarPerDim_fc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0;
		}

		if (dir == quokka::direction::y) {
			const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
			state_fc(i, j, k, MHDSystem<ResistiveMHDEnergyBalance>::bfield_index) = initialB_y(x);
		}
	});
}

auto extractPressureProfile(QuokkaSimulation<ResistiveMHDEnergyBalance> &sim) -> std::tuple<amrex::Vector<amrex::Real>, amrex::Gpu::HostVector<amrex::Real>>
{
	amrex::MultiFab pressure(sim.state_new_cc_[0].boxArray(), sim.state_new_cc_[0].DistributionMap(), 1, 0);
	auto const &pressure_arrs = pressure.arrays();
	auto const &state = sim.state_new_cc_[0].const_arrays();
	auto const &fcx = sim.state_new_fc_[0][0].const_arrays();
	auto const &fcy = sim.state_new_fc_[0][1].const_arrays();
	auto const &fcz = sim.state_new_fc_[0][2].const_arrays();

	amrex::ParallelFor(pressure, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const state_fc = {fcx[bx], fcy[bx], fcz[bx]};
		pressure_arrs[bx](i, j, k) = HydroSystem<ResistiveMHDEnergyBalance>::ComputePressure(state[bx], i, j, k, &state_fc);
	});
	amrex::Gpu::streamSynchronize();

	auto [position, values] = fextract(pressure, sim.Geom(0), 0, 0.5, true);
	return {std::move(position), std::move(values[0])};
}

auto writeDiagnosticCsv(amrex::Vector<amrex::Real> const &position, amrex::Gpu::HostVector<amrex::Real> const &pressure, const amrex::Real time,
			const amrex::Real eta) -> std::tuple<amrex::Real, amrex::Real>
{
	amrex::Real l2_correct = 0.0;
	amrex::Real l2_missing_flux = 0.0;
	amrex::Real l2_norm = 0.0;

	std::ofstream file("resistive_mhd_energy_balance.csv");
	file << std::setprecision(17);
	file << "x,pressure_initial,pressure_final,heating_rate,heating_rate_correct,heating_rate_missing_flux\n";

	for (int i = 0; i < static_cast<int>(position.size()); ++i) {
		const amrex::Real x = position[i];
		const amrex::Real phase = wavenumber * x;
		const amrex::Real pressure_initial = initialPressure(x);
		const amrex::Real heating_rate = (pressure[i] - pressure_initial) / time;
		const amrex::Real heating_rate_correct = eta * field_amplitude * field_amplitude * wavenumber * wavenumber * std::cos(phase) * std::cos(phase);
		const amrex::Real heating_rate_missing_flux =
		    eta * field_amplitude * field_amplitude * wavenumber * wavenumber * std::sin(phase) * std::sin(phase);

		const amrex::Real err_correct = heating_rate - heating_rate_correct;
		const amrex::Real err_missing_flux = heating_rate - heating_rate_missing_flux;
		l2_correct += err_correct * err_correct;
		l2_missing_flux += err_missing_flux * err_missing_flux;
		l2_norm += heating_rate_correct * heating_rate_correct;

		file << x << ',' << pressure_initial << ',' << pressure[i] << ',' << heating_rate << ',' << heating_rate_correct << ','
		     << heating_rate_missing_flux << '\n';
	}

	const amrex::Real norm = std::sqrt(std::max(l2_norm, std::numeric_limits<amrex::Real>::min()));
	return {std::sqrt(l2_correct) / norm, std::sqrt(l2_missing_flux) / norm};
}

auto problem_main() -> int
{
	QuokkaSimulation<ResistiveMHDEnergyBalance> sim;
	sim.setInitialConditions();
	sim.evolve();

	const amrex::Real time = sim.tNew_[0];
	const amrex::Real eta = sim.mhdResistivity_;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(time > 0.0, "ResistiveMHDEnergyBalance requires a positive final time.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(eta > 0.0, "ResistiveMHDEnergyBalance requires mhd.resistivity > 0.");

	auto [position, pressure] = extractPressureProfile(sim);

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto [rel_l2_correct, rel_l2_missing_flux] = writeDiagnosticCsv(position, pressure, time, eta);
		amrex::Print() << "Resistive MHD energy-balance diagnostic at t = " << time << "\n";
		amrex::Print() << "relative L2 error vs correct Ohmic heating profile = " << rel_l2_correct << "\n";
		amrex::Print() << "relative L2 error vs missing-flux profile = " << rel_l2_missing_flux << "\n";

		if (rel_l2_correct > rel_l2_missing_flux) {
			amrex::Print() << "test failed as expected: pressure follows the missing resistive energy-flux profile.\n";
			status = 1;
		} else {
			amrex::Print() << "test passed: pressure is closer to the correct Ohmic heating profile.\n";
		}
	}
	amrex::ParallelDescriptor::Bcast(&status, 1, amrex::ParallelDescriptor::IOProcessorNumber());
	return status;
}
