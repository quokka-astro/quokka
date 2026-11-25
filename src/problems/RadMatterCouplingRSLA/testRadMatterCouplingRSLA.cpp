// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testRadMatterCouplingRSLA.cpp
/// \brief Defines a test problem for radiation-matter coupling.
///

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/OneZoneHarness.hpp"

struct CouplingProblem {}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double chat_over_c = 0.1;
constexpr double c_rsla = chat_over_c * C::c_light;

// Su & Olson (1997) test problem
constexpr double eps_SuOlson = 1.0;
constexpr double a_rad = C::a_rad; // cgs
constexpr double alpha_SuOlson = 4.0 * a_rad / eps_SuOlson;

template <> struct SimulationData<CouplingProblem> : quokka::onezone::Recorder {};

template <> struct quokka::EOS_Traits<CouplingProblem> {
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<CouplingProblem> {
	static constexpr double c_hat_over_c = chat_over_c;
	static constexpr double Erad_floor = 0.;
	static constexpr int beta_order = 1;
};

template <> struct Physics_Traits<CouplingProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CouplingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 1.0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CouplingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 1.0;
}

[[maybe_unused]] static constexpr int nmscalars_ = Physics_Traits<CouplingProblem>::numMassScalars;
template <>
AMREX_GPU_HOST_DEVICE auto
quokka::EOS<CouplingProblem>::ComputeTgasFromEint([[maybe_unused]] const double rho, const double Egas,
						  [[maybe_unused]] quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars) -> double
{
	return std::pow(4.0 * Egas / alpha_SuOlson, 1. / 4.);
}

template <>
AMREX_GPU_HOST_DEVICE auto
quokka::EOS<CouplingProblem>::ComputeEintFromTgas([[maybe_unused]] const double rho, const double Tgas,
						  [[maybe_unused]] quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars) -> double
{
	return (alpha_SuOlson / 4.0) * std::pow(Tgas, 4);
}

template <>
AMREX_GPU_HOST_DEVICE auto
quokka::EOS<CouplingProblem>::ComputeEintTempDerivative(const double /*rho*/, const double Tgas,
							quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/) -> double
{
	// This is also known as the heat capacity, i.e.
	// 		\del E_g / \del T = \rho c_v,
	// for normal materials.

	// However, for this problem, this must be of the form \alpha T^3
	// in order to obtain an exact solution to the problem.
	// The input parameter is the *temperature*, not Egas itself.

	return alpha_SuOlson * std::pow(Tgas, 3);
}

constexpr double Erad0 = 1.0e12; // erg cm^-3
constexpr double Egas0 = 1.0e2;	 // erg cm^-3
constexpr double rho0 = 1.0e-7;	 // g cm^-3

template <> void QuokkaSimulation<CouplingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	quokka::onezone::Cell<CouplingProblem> cell{};
	cell[RadSystem<CouplingProblem>::radEnergy_index] = Erad0;
	cell[RadSystem<CouplingProblem>::gasEnergy_index] = Egas0;
	cell[RadSystem<CouplingProblem>::gasInternalEnergy_index] = Egas0;
	cell[RadSystem<CouplingProblem>::gasDensity_index] = rho0;
	quokka::onezone::FillUniform<CouplingProblem>(grid_elem, cell);
}

template <> void QuokkaSimulation<CouplingProblem>::computeAfterTimestep() { userData_.record(state_new_cc_[0], Geom(0), tNew_[0]); }

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 1.0;
	const double max_time = 1.0e-2; // s
	const int max_timesteps = 1e6;

	QuokkaSimulation<CouplingProblem> sim;

	sim.cflNumber_ = CFL_number;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.stopTime_ = max_time;
	sim.plotfileInterval_ = -1;

	sim.userData_.addObservable("Tgas", [](quokka::onezone::CellValues const &cell) {
		const amrex::Real rho = cell[RadSystem<CouplingProblem>::gasDensity_index];
		static_assert(!Physics_Traits<CouplingProblem>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
		const amrex::Real Eint = quokka::EOS<CouplingProblem>::ComputeEintFromEgas(
		    rho, cell[RadSystem<CouplingProblem>::x1GasMomentum_index], cell[RadSystem<CouplingProblem>::x2GasMomentum_index],
		    cell[RadSystem<CouplingProblem>::x3GasMomentum_index], cell[RadSystem<CouplingProblem>::gasEnergy_index], 0.0);
		return quokka::EOS<CouplingProblem>::ComputeTgasFromEint(rho, Eint);
	});

	// Gas temperature as a function of time, with the reduced speed of light
	const double initial_Tgas = quokka::EOS<CouplingProblem>::ComputeTgasFromEint(rho0, Egas0);
	const auto kappa = RadSystem<CouplingProblem>::ComputePlanckOpacity(rho0, initial_Tgas);
	const auto Tgas_rsla_exact = [=](double t) {
		const double c = RadSystem<CouplingProblem>::c_light_;
		const double T0_4 = std::pow(initial_Tgas, 4);
		const double E0_rsla = ((c / c_rsla) * Erad0 + Egas0) / (a_rad + (c_rsla / c) * alpha_SuOlson / 4.0);
		const double T4_rsla =
		    (T0_4 - (c_rsla / c) * E0_rsla) * std::exp(-(4. / alpha_SuOlson) * (a_rad + (c_rsla / c) * alpha_SuOlson / 4.0) * kappa * rho0 * c * t) +
		    (c_rsla / c) * E0_rsla;
		return std::pow(T4_rsla, 1. / 4.);
	};

	// When using C::a_rad as radiation_constant_cgs_, the relative error goes up to 3e-5, so the tolerance is 5e-5
	const std::vector<quokka::onezone::Check> checks = {
	    {.name = "Tgas relative L1 error",
	     .comparisons = {{.observable = "Tgas", .reference = quokka::onezone::AnalyticReference(Tgas_rsla_exact)}},
	     .tolerance = 5e-5}};
	const std::vector<quokka::onezone::Figure> figures = {
	    {.filename = "radcoupling_rsla.pdf",
	     .curves = {{.observable = "Tgas", .label = "gas temperature", .reference = quokka::onezone::AnalyticReference(Tgas_rsla_exact)}},
	     .xlabel = "time t (s)",
	     .ylabel = "temperature T (K)",
	     .log_x = true,
	     .log_y = true}};
	return quokka::onezone::Run(sim, checks, {}, "", figures);
}
