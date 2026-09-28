//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testRadCouplingUnitTests.cpp
/// \brief Single-cell tests of the bracketed matter-radiation coupling solver in radiation_coupling.hpp.
///
/// The cells reproduce the sweeps of hydro3d.jl (docs/coupling-new-method.md section 5.2 and
/// docs/coupling-new-method-dust.md section 6) in code units: c = chat = a_rad = k_B = 1, rho = 1 and
/// mu = 1.5 so that c_V = 1, four groups spaced logarithmically between 0.1 and 20, and a temperature floor of
/// 1e-10. hydro3d's dust sweep varies rho at fixed c_V, which an ideal gas cannot do, so kappa_0 is varied over
/// the same four decades instead. Every case runs inside an amrex::ParallelFor, so the same code is exercised on
/// the host (CPU build) and on the device (GPU build).

#include "AMReX.H"
#include "AMReX_Gpu.H"
#include "AMReX_GpuContainers.H"
#include "eos.H"
#include "extern_parameters.H"
#include "hydro/EOS.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "util/valarray.hpp"
#include <array>
#include <cmath>
#include <format>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

constexpr double a_rad = 1.0;
constexpr double mu = 1.5; // c_V = rho k_B / ((gamma - 1) mu) = 1 at rho = 1
constexpr double Tfloor = 1.0e-10;
constexpr double Efloor_group = 1.0e-30;
// exp(range(log 0.1, log 20; length = 5)): hydro3d's RadiationGroups(0.1, 20, 4)
constexpr amrex::GpuArray<double, 5> edges4 = {0.1, 0.37606030930863937, 1.4142135623730951, 5.318295896944989, 20.0};

// The opacity law of the cells being solved, kappa = kappa0 * T^expo, set on the host before each kernel launch.
AMREX_GPU_MANAGED double sweep_kappa0 = 1.0; // NOLINT
AMREX_GPU_MANAGED double sweep_expo = 0.0;   // NOLINT

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto sweep_kappa(double T) -> double { return sweep_kappa0 * std::pow(amrex::max(T, 0.0), sweep_expo); }

// Four groups: the gas and dust sweeps.
struct Sweep4 {};
// One group: the grey dust sweep.
struct Sweep1 {};

} // namespace

#define SWEEP_TRAITS(P, NG)                                                                                                                                    \
	template <> struct quokka::EOS_Traits<P> {                                                                                                             \
		static constexpr double mean_molecular_weight = mu;                                                                                            \
		static constexpr double gamma = 5. / 3.;                                                                                                       \
	};                                                                                                                                                     \
	template <> struct Physics_Traits<P> : DefaultPhysicsTraits {                                                                                          \
		static constexpr bool is_hydro_enabled = false;                                                                                                \
		static constexpr bool is_radiation_enabled = true;                                                                                             \
		static constexpr int nGroups = NG;                                                                                                             \
		static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;                                                                               \
		static constexpr double boltzmann_constant = 1.0;                                                                                              \
		static constexpr double gravitational_constant = 1.0;                                                                                          \
		static constexpr double c_light = 1.0;                                                                                                         \
		static constexpr double radiation_constant = a_rad;                                                                                            \
	};

SWEEP_TRAITS(Sweep4, 4)
SWEEP_TRAITS(Sweep1, 1)

template <> struct RadSystem_Traits<Sweep4> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = 4 * Efloor_group;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = 1.0;
	static constexpr amrex::GpuArray<double, 5> radBoundaries = edges4;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
};
template <> struct RadSystem_Traits<Sweep1> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = Efloor_group;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = 1.0;
	static constexpr OpacityModel opacity_model = OpacityModel::single_group;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<Sweep4>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/,
												      const double /*rho*/, const double Tgas)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		exponents_and_values[1][i] = sweep_kappa(Tgas);
	}
	return exponents_and_values;
}
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<Sweep1>::ComputePlanckOpacity(const double /*rho*/, const double Tgas) -> amrex::Real
{
	return sweep_kappa(Tgas);
}
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<Sweep1>::ComputeEnergyMeanOpacity(const double /*rho*/, const double Tgas) -> amrex::Real
{
	return sweep_kappa(Tgas);
}
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<Sweep1>::ComputeFluxMeanOpacity(const double /*rho*/, const double Tgas) -> amrex::Real
{
	return sweep_kappa(Tgas);
}

namespace
{

/// One cell in code units. K is the collisional gas-dust coefficient (dtK = dt * K); 0 means no dust coupling.
template <typename P> auto make_cell(double Tgas, double Trad, double dt, double K) -> CouplingCell<P>
{
	constexpr int NG = Physics_Traits<P>::nGroups;
	CouplingCell<P> cell{};
	cell.rho = 1.0;
	cell.dt = dt;
	cell.tau_scale = dt * RadSystem<P>::c_hat_;
	cell.dtK = dt * K;
	cell.Tfloor = Tfloor;
	cell.Emin = quokka::EOS<P>::ComputeEintFromTgas(cell.rho, Tfloor, cell.massScalars);
	cell.Egas0 = quokka::EOS<P>::ComputeEintFromTgas(cell.rho, Tgas, cell.massScalars);
	cell.rad_boundaries = RadSystem<P>::radBoundaries_;
	if constexpr (NG > 1) {
		for (int g = 0; g < NG; ++g) {
			cell.rad_boundary_ratios[g] = cell.rad_boundaries[g + 1] / cell.rad_boundaries[g];
		}
		const auto frac = RadSystem<P>::ComputePlanckEnergyFractions(cell.rad_boundaries, std::max(Trad, Tfloor));
		for (int g = 0; g < NG; ++g) {
			cell.Erad0[g] = a_rad * std::pow(Trad, 4) * frac[g] + 1.0e-20;
		}
	} else {
		cell.Erad0[0] = a_rad * std::pow(Trad, 4) + 1.0e-20;
	}
	cell.Src.fillin(0.0);
	cell.work.fillin(0.0);
	return cell;
}

template <typename P>
auto total_energy(CouplingCell<P> const & /*cell*/, double Egas, quokka::valarray<double, Physics_Traits<P>::nGroups> const &Erad) -> double
{
	return Egas + (RadSystem<P>::c_light_ / RadSystem<P>::c_hat_) * sum(Erad);
}

template <typename P> auto conserved_total(CouplingCell<P> const &cell) -> double { return total_energy(cell, cell.Egas0, cell.Erad0 + cell.Src); }

/// Solve every cell of a batch inside one kernel, with dust (SolveDustCoupling) or without (SolveGasCoupling).
template <typename P, bool with_dust> auto solve_cells(std::vector<CouplingCell<P>> const &cells, double tol) -> std::vector<CouplingSolution<P>>
{
	const int n = static_cast<int>(cells.size());
	amrex::Gpu::DeviceVector<CouplingCell<P>> d_cells(n);
	amrex::Gpu::DeviceVector<CouplingSolution<P>> d_sols(n);
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, cells.begin(), cells.end(), d_cells.begin());
	CouplingCell<P> const *cell_ptr = d_cells.data();
	CouplingSolution<P> *sol_ptr = d_sols.data();
	amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE(int i) noexcept {
		// force capture unconditionally: nvcc rejects a captured variable first used inside an if constexpr branch
		CouplingCell<P> const *const cell_ptr_capture = cell_ptr;
		CouplingSolution<P> *const sol_ptr_capture = sol_ptr;
		const double tol_capture = tol;
		if constexpr (with_dust) {
			sol_ptr_capture[i] = RadSystem<P>::SolveDustCoupling(cell_ptr_capture[i], tol_capture);
		} else {
			sol_ptr_capture[i] = RadSystem<P>::SolveGasCoupling(cell_ptr_capture[i], tol_capture);
		}
	});
	std::vector<CouplingSolution<P>> sols(n);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_sols.begin(), d_sols.end(), sols.begin());
	return sols;
}

/// The closed-form state at the gas floor, computed by RadSystem<P>::GasCouplingState the same way solve_cells exercises
/// SolveGasCoupling: the reference the floor-clamp branch of SolveGasCoupling must reproduce exactly.
template <typename P> auto floor_state(CouplingCell<P> const &cell) -> CouplingSolution<P>
{
	std::vector<CouplingCell<P>> cells{cell};
	amrex::Gpu::DeviceVector<CouplingCell<P>> d_cells(1);
	amrex::Gpu::DeviceVector<CouplingSolution<P>> d_sols(1);
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, cells.begin(), cells.end(), d_cells.begin());
	CouplingCell<P> const *cell_ptr = d_cells.data();
	CouplingSolution<P> *sol_ptr = d_sols.data();
	amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int i) noexcept { sol_ptr[i] = RadSystem<P>::GasCouplingState(cell_ptr[i], cell_ptr[i].Emin); });
	std::vector<CouplingSolution<P>> sols(1);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_sols.begin(), d_sols.end(), sols.begin());
	return sols[0];
}

auto check(bool ok, std::string const &what) -> int
{
	std::cout << std::format("{:<78}{}\n", what, ok ? "ok" : "FAIL");
	return ok ? 0 : 1;
}

// hydro3d docs/coupling-new-method.md section 5.2: 8 states x 10 optical depths x 3 opacity laws, dt = 1e-3, tol = 1e-9
constexpr std::array<std::pair<double, double>, 8> gas_states{
    {{1.3, 0.5}, {0.5, 1.3}, {1.0, 0.3}, {10.0, 1.0}, {1.0, 10.0}, {1.0, 1.0001}, {3.0, 0.1}, {0.1, 3.0}}};
constexpr std::array<double, 10> gas_taus{1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8, 1e9, 1e10, 1e12};
constexpr std::array<double, 3> gas_laws{0.0, -3.5, 2.0};
constexpr double gas_dt = 1.0e-3;

auto TestGasSweep() -> int
{
	int status = 0;
	int nfail = 0;
	double worst_energy = 0.0;
	double worst_vs_ref = 0.0;
	long nevals_sum = 0;
	int nevals_max = 0;
	int ncells = 0;
	for (const double expo : gas_laws) {
		for (const double tau : gas_taus) {
			sweep_kappa0 = tau / gas_dt; // rho = 1, chat = 1: tau = dt rho kappa0 at T = 1
			sweep_expo = expo;
			std::vector<CouplingCell<Sweep4>> cells;
			for (const auto &[Tg, Tr] : gas_states) {
				cells.push_back(make_cell<Sweep4>(Tg, Tr, gas_dt, 0.0));
			}
			const auto sols = solve_cells<Sweep4, false>(cells, 1.0e-9);
			const auto refs = solve_cells<Sweep4, false>(cells, 1.0e-13);
			for (std::size_t i = 0; i < cells.size(); ++i) {
				const double Etot = conserved_total(cells[i]);
				nfail += sols[i].converged ? 0 : 1;
				worst_energy = std::max(worst_energy, std::abs(total_energy(cells[i], sols[i].Egas, sols[i].Erad) - Etot) / Etot);
				worst_vs_ref = std::max(worst_vs_ref, std::abs(sols[i].Egas - refs[i].Egas) / Etot);
				nevals_sum += sols[i].nevals;
				nevals_max = std::max(nevals_max, sols[i].nevals);
				++ncells;
			}
		}
	}
	const double nevals_mean = static_cast<double>(nevals_sum) / ncells;
	std::cout << std::format("gas sweep: {} cells, {} failures, energy error {:.2e}, vs 1e-13 solve {:.2e}, evaluations mean {:.1f} max {}\n", ncells,
				 nfail, worst_energy, worst_vs_ref, nevals_mean, nevals_max);
	status |= check(ncells == 240, "gas sweep has 240 cells");
	status |= check(nfail == 0, "gas sweep: every cell converged");
	// Convergence is on E_gas, so the conservation error is dG/dE_gas times the tolerance: up to thousands of times
	// 1e-9 in a radiation-dominated cell (hydro3d section 4.3). The measured value is printed above.
	status |= check(worst_energy <= 1.0e-8, "gas sweep: energy conserved to 1e-8 of the cell's energy");
	status |= check(worst_vs_ref <= 2.0e-9, "gas sweep: within 2e-9 of a 1e-13 solve");
	status |= check(nevals_mean <= 16.0, "gas sweep: mean evaluations per cell at most 16 (hydro3d: 12)");

	// Three roots (hydro3d section 3.3): kappa = 10 T^2, gas at 0.1, radiation at 3; G changes sign at 0.110, 1.17 and
	// 2.52 and the error-controlled ODE ends at 0.109. Marching from the old state must return the cold root.
	sweep_kappa0 = 10.0;
	sweep_expo = 2.0;
	{
		const auto sols = solve_cells<Sweep4, false>({make_cell<Sweep4>(0.1, 3.0, gas_dt, 0.0)}, 1.0e-9);
		std::cout << std::format("three-root gas cell: Egas = {:.6f}\n", sols[0].Egas);
		status |= check(sols[0].converged && sols[0].Egas < 0.2, "three-root gas cell returns the root connected to the old state");
	}

	// A transparent group keeps its source exactly: kappa = 0 everywhere, source in group 1.
	sweep_kappa0 = 0.0;
	sweep_expo = 0.0;
	{
		auto cell = make_cell<Sweep4>(1.0, 0.5, gas_dt, 0.0);
		cell.Src[1] = 0.25;
		const auto sols = solve_cells<Sweep4, false>({cell}, 1.0e-9);
		status |= check(sols[0].converged && (sols[0].Erad[1] == cell.Erad0[1] + 0.25) && (std::abs(sols[0].Egas - cell.Egas0) <= 1e-9 * cell.Egas0),
				"transparent group keeps its source and the gas is untouched");
	}

	// A work term that by itself exceeds the gas energy breaks the bracket (hydro3d section 3.1): G(E_min) > 0, so the
	// downward march reaches the floor without a sign change. The root lies below the admissible range, so the gas is
	// clamped to the floor and the groups take the closed form at T_floor (the floor rule of design-coupling-rewrite.md
	// section 2.4), not kept at the old, unconverged state.
	sweep_kappa0 = 1.0e3;
	sweep_expo = 0.0;
	{
		auto cell = make_cell<Sweep4>(1.0, 0.5, gas_dt, 0.0);
		for (int g = 0; g < 4; ++g) {
			cell.work[g] = 2.0 * cell.Egas0;
		}
		const auto sols = solve_cells<Sweep4, false>({cell}, 1.0e-9);
		const auto floor_sol = floor_state<Sweep4>(cell);
		bool erad_matches_floor = true;
		for (int g = 0; g < 4; ++g) {
			erad_matches_floor = erad_matches_floor && (sols[0].Erad[g] == floor_sol.Erad[g]);
		}
		const double Etot = conserved_total(cell);
		const double residual_check = total_energy(cell, sols[0].Egas, sols[0].Erad) - conserved_total(cell);
		status |= check(sols[0].converged && (std::abs(sols[0].Egas - cell.Emin) <= 1.0e-12 * cell.Emin) && erad_matches_floor &&
				    (sols[0].residual > 0.0) && (std::abs(residual_check - sols[0].residual) <= 1.0e-9 * Etot),
				"root below the floor: clamped to the floor state, converged");
	}

	// The reviewer's hot-cell case: a transparent cell whose root lies just below the floor must be clamped, not kept
	// hot. The work term totals Egas0 + 1e-12 across the four groups, so G(E_min) = E_min + 1e-12 > 0 while Egas0 = 1 is
	// far above E_min = 1e-10 (the old code returned this cell converged with Egas == Egas0, i.e. no exchange at all).
	sweep_kappa0 = 0.0;
	sweep_expo = 0.0;
	{
		auto cell = make_cell<Sweep4>(1.0, 0.5, gas_dt, 0.0);
		for (int g = 0; g < 4; ++g) {
			cell.work[g] = (cell.Egas0 + 1.0e-12) / 4.0;
		}
		const auto sols = solve_cells<Sweep4, false>({cell}, 1.0e-9);
		bool erad_matches_transparent = true;
		for (int g = 0; g < 4; ++g) {
			erad_matches_transparent = erad_matches_transparent && (sols[0].Erad[g] == cell.Erad0[g] + cell.work[g]);
		}
		status |= check(sols[0].converged && (std::abs(sols[0].Egas - cell.Emin) <= 1.0e-12 * cell.Emin) && erad_matches_transparent &&
				    (std::abs(sols[0].residual - (cell.Emin + 1.0e-12)) <= 1.0e-9),
				"hot cell with the root just below the floor is clamped, not kept hot");
	}

	// The same hot cell with a zero temperature floor (every CONSTANTS-unit problem): the march floors at round-off of the
	// initial gas energy instead of halving towards E = 0 until its budget runs out.
	{
		auto cell = make_cell<Sweep4>(1.0, 0.5, gas_dt, 0.0);
		cell.Tfloor = 0.0;
		cell.Emin = 0.0;
		for (int g = 0; g < 4; ++g) {
			cell.work[g] = (cell.Egas0 + 1.0e-12) / 4.0;
		}
		const auto sols = solve_cells<Sweep4, false>({cell}, 1.0e-9);
		bool erad_matches_transparent = true;
		for (int g = 0; g < 4; ++g) {
			erad_matches_transparent = erad_matches_transparent && (sols[0].Erad[g] == cell.Erad0[g] + cell.work[g]);
		}
		constexpr double eps = std::numeric_limits<double>::epsilon();
		status |= check(sols[0].converged && (sols[0].Egas > 0.0) && (sols[0].Egas <= 16.0 * eps * 1.0) && erad_matches_transparent,
				"hot cell, zero temperature floor: clamped at round-off of Egas0");
	}

	// A temperature floor of zero (every CONSTANTS-unit problem): the downward march must stay finite and find its root.
	sweep_kappa0 = 1.0e6;
	{
		auto cell = make_cell<Sweep4>(3.0, 0.1, gas_dt, 0.0);
		cell.Tfloor = 0.0;
		cell.Emin = 0.0;
		const auto sols = solve_cells<Sweep4, false>({cell}, 1.0e-9);
		const double Etot = conserved_total(cell);
		status |=
		    check(sols[0].converged && std::isfinite(sols[0].Egas) && std::abs(total_energy(cell, sols[0].Egas, sols[0].Erad) - Etot) <= 1e-8 * Etot,
			  "zero temperature floor: converged and conserved");
	}
	return status;
}

// hydro3d docs/coupling-new-method-dust.md section 6, with kappa0 in place of rho (see the file comment): N_G in {1, 4},
// two opacity laws, Tgas in {0.01, 1, 10}, Trad in {0, 1, 5}, kappa0 in {1e-4, 1, 1e4, 1e8}, K in {0, 1, 1e8}; dt = 1.
template <typename P>
auto dust_sweep(int &ncells, int &nfail, double &worst_energy, double &worst_state, long &nevals_sum, int &nevals_max, std::string &worst_cell) -> void
{
	for (const double expo : {0.0, 2.0}) {
		for (const double kappa0 : {1e-4, 1.0, 1e4, 1e8}) {
			sweep_kappa0 = kappa0;
			sweep_expo = expo;
			std::vector<CouplingCell<P>> cells;
			std::vector<std::array<double, 3>> params; // (Tg, Tr, K) of each cell
			for (const double Tg : {0.01, 1.0, 10.0}) {
				for (const double Tr : {0.0, 1.0, 5.0}) {
					for (const double K : {0.0, 1.0, 1e8}) {
						cells.push_back(make_cell<P>(Tg, Tr, 1.0, K));
						params.push_back({Tg, Tr, K});
					}
				}
			}
			const auto sols = solve_cells<P, true>(cells, 1.0e-9);
			const auto refs = solve_cells<P, true>(cells, 1.0e-12);
			for (std::size_t i = 0; i < cells.size(); ++i) {
				const double Etot = conserved_total(cells[i]);
				nfail += sols[i].converged ? 0 : 1;
				worst_energy = std::max(worst_energy, std::abs(total_energy(cells[i], sols[i].Egas, sols[i].Erad) / Etot - 1.0));
				const double state_err = std::abs(sols[i].Egas - refs[i].Egas) / Etot;
				if (state_err > worst_state) {
					worst_state = state_err;
					worst_cell = std::format("N_G {} expo {} kappa0 {:.0e} Tg {} Tr {} K {:.0e}: Egas {:.16e} vs ref {:.16e} ({} evals)",
								 Physics_Traits<P>::nGroups, expo, kappa0, params[i][0], params[i][1], params[i][2],
								 sols[i].Egas, refs[i].Egas, sols[i].nevals);
				}
				nevals_sum += sols[i].nevals;
				nevals_max = std::max(nevals_max, sols[i].nevals);
				++ncells;
			}
		}
	}
}

auto TestDustSweep() -> int
{
	int status = 0;
	int ncells = 0;
	int nfail = 0;
	double worst_energy = 0.0;
	double worst_state = 0.0;
	long nevals_sum = 0;
	int nevals_max = 0;
	std::string worst_cell;
	dust_sweep<Sweep4>(ncells, nfail, worst_energy, worst_state, nevals_sum, nevals_max, worst_cell);
	dust_sweep<Sweep1>(ncells, nfail, worst_energy, worst_state, nevals_sum, nevals_max, worst_cell);
	std::cout << std::format("dust sweep: {} cells, {} failures, energy error {:.2e}, vs 1e-12 solve {:.2e}, evaluations mean {:.1f} max {}\n", ncells,
				 nfail, worst_energy, worst_state, static_cast<double>(nevals_sum) / ncells, nevals_max);
	std::cout << std::format("dust sweep: largest state error at {}\n", worst_cell);
	status |= check(ncells == 432, "dust sweep has 432 cells");
	status |= check(nfail == 0, "dust sweep: every cell converged");
	status |= check(worst_energy < 1.0e-13, "dust sweep: energy conserved to round-off");
	status |= check(worst_state <= 2.0e-9, "dust sweep: state within 2e-9 of a 1e-12 solve");

	// K -> infinity locks the dust to the gas: the dust-free solve with the same opacity law. K = 0 leaves the gas alone.
	for (const double expo : {0.0, -1.5}) {
		sweep_kappa0 = 1.0;
		sweep_expo = expo;
		for (const auto &[Tg, Tr] : {std::pair{1.0, 0.1}, std::pair{0.3, 2.0}}) {
			{
				const auto cell = make_cell<Sweep4>(Tg, Tr, 1.0, 1.0e12);
				const auto dust = solve_cells<Sweep4, true>({cell}, 1.0e-9);
				const auto gas = solve_cells<Sweep4, false>({cell}, 1.0e-9);
				const double Etot = conserved_total(cell);
				status |= check(dust[0].converged && std::abs(dust[0].Egas - gas[0].Egas) <= 1.0e-8 * Etot &&
						    std::abs(dust[0].T_d / dust[0].T_gas - 1.0) <= 1.0e-8,
						std::format("K = 1e12 recovers the dust-free solve (expo {}, Tg {}, Tr {})", expo, Tg, Tr));
			}
			{
				const auto cell = make_cell<Sweep1>(Tg, Tr, 1.0, 1.0e12);
				const auto dust = solve_cells<Sweep1, true>({cell}, 1.0e-9);
				const auto gas = solve_cells<Sweep1, false>({cell}, 1.0e-9);
				const double Etot = conserved_total(cell);
				status |= check(dust[0].converged && std::abs(dust[0].Egas - gas[0].Egas) <= 1.0e-8 * Etot,
						std::format("K = 1e12, one group (expo {}, Tg {})", expo, Tg));
			}
			{
				const auto cell = make_cell<Sweep4>(Tg, Tr, 1.0, 0.0);
				const auto dust = solve_cells<Sweep4, true>({cell}, 1.0e-9);
				const double Etot = conserved_total(cell);
				status |= check(dust[0].converged && std::abs(dust[0].Egas - cell.Egas0) <= 1.0e-12 * Etot,
						std::format("K = 0 leaves the gas unchanged (expo {}, Tg {})", expo, Tg));
			}
		}
	}

	// Three roots of the dust balance itself (hydro3d dust section 3): kappa = T_d^2, gas at 0.01, radiation at 1,
	// K = 1, dt = 1; roots at T_d = 0.0114, 0.134 and 0.738. From the gas temperature the march returns the cold one.
	sweep_kappa0 = 1.0;
	sweep_expo = 2.0;
	{
		const auto sols = solve_cells<Sweep4, true>({make_cell<Sweep4>(0.01, 1.0, 1.0, 1.0)}, 1.0e-9);
		std::cout << std::format("three-root dust cell: T_d = {:.6f}\n", sols[0].T_d);
		status |=
		    check(sols[0].converged && (0.0113 < sols[0].T_d) && (sols[0].T_d < 0.0115), "three-root dust cell returns the root connected to cold gas");
	}
	return status;
}

} // namespace

auto problem_main() -> int
{
	// the Microphysics EOS behind quokka::EOS must be initialised before any energy-temperature conversion
	init_extern_parameters();
	amrex::Real small_temp = Tfloor; // eos_init takes non-const references
	amrex::Real small_dens = 1.0e-100;
	eos_init(small_temp, small_dens);
	// every pinned number below assumes c_V = 1 at rho = 1; fail loudly if the EOS convention differs
	AMREX_ALWAYS_ASSERT(std::abs(make_cell<Sweep4>(1.0, 0.5, 1.0, 0.0).Egas0 - 1.0) < 1e-12);
	const int gas_status = TestGasSweep();
	const int dust_status = TestDustSweep();
	const int status = (gas_status == 0 && dust_status == 0) ? 0 : 1;
	std::cout << (status == 0 ? "RadCouplingUnitTests: all tests passed.\n" : "RadCouplingUnitTests: FAILED.\n");
	return status;
}
