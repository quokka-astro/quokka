/// \file testRadCouplingUnitTests.cpp
/// \brief Single-cell tests of the bracketed matter-radiation coupling solver in radiation_coupling.hpp.
///
/// The cells reproduce the sweeps of hydro3d.jl (docs/coupling-new-method.md section 5.2 and
/// docs/coupling-new-method-dust.md section 6) in code units: c = chat = a_rad = k_B = 1, rho = 1 and
/// mu = 1.5 so that c_V = 1, four groups spaced logarithmically between 0.1 and 20, and a temperature floor of
/// 1e-10. hydro3d's dust sweep varies rho at fixed c_V, which an ideal gas cannot do, so kappa_0 is varied over
/// the same four decades instead. A third sweep solves the two-temperature system in CGS at the conditions of a
/// dusty D-type ionization front (the DTypeFront1D setup of PR #2305). Every case runs inside an amrex::ParallelFor,
/// so the same code is exercised on the host (CPU build) and on the device (GPU build).

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
#include <cstdint>
#include <format>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

namespace
{

constexpr double a_rad = 1.0;
constexpr double mu = 1.5; // c_V = rho k_B / ((gamma - 1) mu) = 1 at rho = 1
constexpr double Tfloor = 1.0e-10;
constexpr double Efloor_group = 1.0e-30;
// exp(range(log 0.1, log 20; length = 5)): hydro3d's RadiationGroups(0.1, 20, 4)
constexpr amrex::GpuArray<double, 5> edges4 = {0.1, 0.37606030930863937, std::numbers::sqrt2, 5.318295896944989, 20.0};

// The opacity law of the cells being solved, kappa = kappa0 * T^expo, set on the host before each kernel launch.
AMREX_GPU_MANAGED double sweep_kappa0 = 1.0; // NOLINT
AMREX_GPU_MANAGED double sweep_expo = 0.0;   // NOLINT

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto sweep_kappa(double T) -> double { return sweep_kappa0 * std::pow(amrex::max(T, 0.0), sweep_expo); }

// Four groups: the gas and dust sweeps.
struct Sweep4 {};
// One group: the grey dust sweep.
struct Sweep1 {};
// CGS, IR and optical groups of a dusty D-type front: the hard dust sweep.
struct SweepDType {};
// CGS, one grey group: a weakly coupled, radiation-dominated two-temperature cell.
struct GreyCGS {};
// CGS, six half-decade groups from 0.01 eV to 10 eV with kappa_nu = kappa_0 (nu / nu_0)^p (T / T_0)^beta: the round-off
// bound sweep.
struct PowerLawCGS {};

// The opacity law of PowerLawCGS, set on the host before each kernel launch: kappa_0 in cm^2 g^-1 at nu_0 = 1 eV and
// T_0 = 100 K, the frequency exponent p and the temperature exponent beta.
AMREX_GPU_MANAGED double pl_kappa0 = 1.0; // NOLINT
AMREX_GPU_MANAGED double pl_expo = 0.0;	  // NOLINT
AMREX_GPU_MANAGED double pl_beta = 0.0;	  // NOLINT
constexpr double pl_nu0 = 1.0;		  // eV
constexpr double pl_T0 = 100.0;		  // K
constexpr int pl_ngroups = 6;
constexpr amrex::GpuArray<double, pl_ngroups + 1> pl_edges = {1.0e-2, 3.1622776601683795e-2, 1.0e-1, 3.1622776601683795e-1, 1.0, 3.1622776601683795, 10.0};
constexpr double pl_Erad_floor = 1.0e-30; // erg cm^-3
constexpr double pl_Tfloor = 1.0;	  // K

// The group opacities of SweepDType (IR, optical) in cm^2 g^-1, set on the host before each kernel launch.
AMREX_GPU_MANAGED double dtype_kappa_ir = 1.0e-2;		// NOLINT
AMREX_GPU_MANAGED double dtype_kappa_opt = 1.0e3;		// NOLINT
constexpr double dtype_Tfloor = 10.0;				// K
constexpr double dtype_Erad_floor = 1.0e-10 * 13.6 * C::ev2erg; // erg cm^-3, shared by the two groups

} // namespace

template <> struct quokka::EOS_Traits<Sweep4> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};
template <> struct Physics_Traits<Sweep4> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 4;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = a_rad;
};
template <> struct quokka::EOS_Traits<Sweep1> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};
template <> struct Physics_Traits<Sweep1> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = a_rad;
};

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

template <> struct quokka::EOS_Traits<SweepDType> {
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double gamma = 5. / 3.;
};
template <> struct Physics_Traits<SweepDType> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 2;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};
// The thermal bands of the D-type front: IR below 0.41 eV, optical up to the Lyman edge; chat = c / 1000.
template <> struct RadSystem_Traits<SweepDType> {
	static constexpr double c_hat_over_c = 1.0e-3;
	static constexpr double Erad_floor = dtype_Erad_floor;
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr amrex::GpuArray<double, 3> radBoundaries = {1.0e-6, 0.413567, 13.6};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
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
template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<SweepDType>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							    const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	exponents_and_values[1][0] = dtype_kappa_ir;
	exponents_and_values[1][1] = dtype_kappa_opt;
	return exponents_and_values;
}

template <> struct quokka::EOS_Traits<GreyCGS> {
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double gamma = 5. / 3.;
};
template <> struct Physics_Traits<GreyCGS> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};
template <> struct RadSystem_Traits<GreyCGS> {
	static constexpr double c_hat_over_c = 1.0e-3;
	static constexpr double Erad_floor = 1.0e-40;
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr OpacityModel opacity_model = OpacityModel::single_group;
};
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<GreyCGS>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real { return 1.0e3; }
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<GreyCGS>::ComputeEnergyMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 1.0e3;
}

template <> struct quokka::EOS_Traits<PowerLawCGS> {
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double gamma = 5. / 3.;
};
template <> struct Physics_Traits<PowerLawCGS> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = pl_ngroups;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};
template <> struct RadSystem_Traits<PowerLawCGS> {
	static constexpr double c_hat_over_c = 1.0e-3;
	static constexpr double Erad_floor = pl_Erad_floor;
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr amrex::GpuArray<double, pl_ngroups + 1> radBoundaries = pl_edges;
	static constexpr OpacityModel opacity_model = OpacityModel::PPL_opacity_fixed_slope_spectrum;
};
// kappa_nu = kappa_0 (nu / nu_0)^p (T / T_0)^beta in every group: the exponent p and the value at the group's lower edge.
template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<PowerLawCGS>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries,
													   const double /*rho*/, const double Tgas)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	const double kappa_T = pl_kappa0 * std::pow(amrex::max(Tgas, 1.0e-10) / pl_T0, pl_beta);
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = pl_expo;
		exponents_and_values[1][i] = kappa_T * std::pow(rad_boundaries[i] / pl_nu0, pl_expo);
	}
	return exponents_and_values;
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

/// One SweepDType cell in CGS: hydrogen number density n_H [cm^-3], gas temperature [K], IR and optical group energies
/// [erg cm^-3] (held at the floor at least), step dt [s], collisional coefficient k_gd [erg cm^3 s^-1 K^-3/2], and a
/// lagged work term on the optical group equal to w times its energy.
auto make_dtype_cell(double n_H, double Tgas, double E_ir, double E_opt, double dt, double k_gd, double w) -> CouplingCell<SweepDType>
{
	using P = SweepDType;
	CouplingCell<P> cell{};
	cell.rho = n_H * C::m_u;
	cell.dt = dt;
	cell.tau_scale = dt * RadSystem<P>::c_hat_;
	cell.dtK = dt * k_gd * n_H * n_H;
	cell.Tfloor = dtype_Tfloor;
	cell.Emin = quokka::EOS<P>::ComputeEintFromTgas(cell.rho, dtype_Tfloor, cell.massScalars);
	cell.Egas0 = quokka::EOS<P>::ComputeEintFromTgas(cell.rho, Tgas, cell.massScalars);
	cell.rad_boundaries = RadSystem<P>::radBoundaries_;
	for (int g = 0; g < 2; ++g) {
		cell.rad_boundary_ratios[g] = cell.rad_boundaries[g + 1] / cell.rad_boundaries[g];
	}
	cell.Erad0[0] = std::max(E_ir, RadSystem<P>::Erad_floor_);
	cell.Erad0[1] = std::max(E_opt, RadSystem<P>::Erad_floor_);
	cell.Src.fillin(0.0);
	cell.work.fillin(0.0);
	cell.work[1] = w * cell.Erad0[1];
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

int n_failed = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// Print one check line and count a failure; the test functions return whether any check failed so far.
void check(bool ok, std::string const &what)
{
	std::cout << std::format("{:<78}{}\n", what, ok ? "ok" : "FAIL");
	if (!ok) {
		++n_failed;
	}
}

// hydro3d docs/coupling-new-method.md section 5.2: 8 states x 10 optical depths x 3 opacity laws, dt = 1e-3, tol = 1e-9
constexpr std::array<std::pair<double, double>, 8> gas_states{
    {{1.3, 0.5}, {0.5, 1.3}, {1.0, 0.3}, {10.0, 1.0}, {1.0, 10.0}, {1.0, 1.0001}, {3.0, 0.1}, {0.1, 3.0}}};
constexpr std::array<double, 10> gas_taus{1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8, 1e9, 1e10, 1e12};
constexpr std::array<double, 3> gas_laws{0.0, -3.5, 2.0};
constexpr double gas_dt = 1.0e-3;

auto TestGasSweep() -> int
{
	int nfail = 0;
	double worst_energy = 0.0;
	double worst_vs_ref = 0.0;
	std::int64_t nevals_sum = 0;
	int nevals_max = 0;
	int ncells = 0;
	for (const double expo : gas_laws) {
		for (const double tau : gas_taus) {
			sweep_kappa0 = tau / gas_dt; // rho = 1, chat = 1: tau = dt rho kappa0 at T = 1
			sweep_expo = expo;
			std::vector<CouplingCell<Sweep4>> cells;
			cells.reserve(gas_states.size());
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
	check(ncells == 240, "gas sweep has 240 cells");
	check(nfail == 0, "gas sweep: every cell converged");
	// Convergence is on E_gas, so the conservation error is dG/dE_gas times the tolerance: up to thousands of times
	// 1e-9 in a radiation-dominated cell (hydro3d section 4.3). The measured value is printed above.
	check(worst_energy <= 1.0e-8, "gas sweep: energy conserved to 1e-8 of the cell's energy");
	check(worst_vs_ref <= 2.0e-9, "gas sweep: within 2e-9 of a 1e-13 solve");
	check(nevals_mean <= 16.0, "gas sweep: mean evaluations per cell at most 16 (hydro3d: 12)");

	// Three roots (hydro3d section 3.3): kappa = 10 T^2, gas at 0.1, radiation at 3; G changes sign at 0.110, 1.17 and
	// 2.52 and the error-controlled ODE ends at 0.109. Marching from the old state must return the cold root.
	sweep_kappa0 = 10.0;
	sweep_expo = 2.0;
	{
		const auto sols = solve_cells<Sweep4, false>({make_cell<Sweep4>(0.1, 3.0, gas_dt, 0.0)}, 1.0e-9);
		std::cout << std::format("three-root gas cell: Egas = {:.6f}\n", sols[0].Egas);
		check(sols[0].converged && sols[0].Egas < 0.2, "three-root gas cell returns the root connected to the old state");
	}

	// A transparent group keeps its source exactly: kappa = 0 everywhere, source in group 1.
	sweep_kappa0 = 0.0;
	sweep_expo = 0.0;
	{
		auto cell = make_cell<Sweep4>(1.0, 0.5, gas_dt, 0.0);
		cell.Src[1] = 0.25;
		const auto sols = solve_cells<Sweep4, false>({cell}, 1.0e-9);
		check(sols[0].converged && (sols[0].Erad[1] == cell.Erad0[1] + 0.25) && (std::abs(sols[0].Egas - cell.Egas0) <= 1e-9 * cell.Egas0),
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
		check(sols[0].converged && (std::abs(sols[0].Egas - cell.Emin) <= 1.0e-12 * cell.Emin) && erad_matches_floor && (sols[0].residual > 0.0) &&
			  (std::abs(residual_check - sols[0].residual) <= 1.0e-9 * Etot),
		      "root below the floor: clamped to the floor state, converged");
	}

	// A hot transparent cell whose root lies just below the floor must be clamped, not kept
	// hot. The work term totals Egas0 + 1e-12 across the four groups, so G(E_min) = E_min + 1e-12 > 0 while Egas0 = 1 is
	// far above E_min = 1e-10; keeping Egas == Egas0 would mean no exchange at all.
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
		check(sols[0].converged && (std::abs(sols[0].Egas - cell.Emin) <= 1.0e-12 * cell.Emin) && erad_matches_transparent &&
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
		check(sols[0].converged && (sols[0].Egas > 0.0) && (sols[0].Egas <= 16.0 * eps * 1.0) && erad_matches_transparent,
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
		check(sols[0].converged && std::isfinite(sols[0].Egas) && std::abs(total_energy(cell, sols[0].Egas, sols[0].Erad) - Etot) <= 1e-8 * Etot,
		      "zero temperature floor: converged and conserved");
	}
	return (n_failed > 0) ? 1 : 0;
}

// hydro3d docs/coupling-new-method-dust.md section 6, with kappa0 in place of rho (see the file comment): N_G in {1, 4},
// two opacity laws, Tgas in {0.01, 1, 10}, Trad in {0, 1, 5}, kappa0 in {1e-4, 1, 1e4, 1e8}, K in {0, 1, 1e8}; dt = 1.
template <typename P>
auto dust_sweep(int &ncells, int &nfail, double &worst_energy, double &worst_state, std::int64_t &nevals_sum, int &nevals_max, std::string &worst_cell) -> void
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
	int ncells = 0;
	int nfail = 0;
	double worst_energy = 0.0;
	double worst_state = 0.0;
	std::int64_t nevals_sum = 0;
	int nevals_max = 0;
	std::string worst_cell;
	dust_sweep<Sweep4>(ncells, nfail, worst_energy, worst_state, nevals_sum, nevals_max, worst_cell);
	dust_sweep<Sweep1>(ncells, nfail, worst_energy, worst_state, nevals_sum, nevals_max, worst_cell);
	std::cout << std::format("dust sweep: {} cells, {} failures, energy error {:.2e}, vs 1e-12 solve {:.2e}, evaluations mean {:.1f} max {}\n", ncells,
				 nfail, worst_energy, worst_state, static_cast<double>(nevals_sum) / ncells, nevals_max);
	std::cout << std::format("dust sweep: largest state error at {}\n", worst_cell);
	check(ncells == 432, "dust sweep has 432 cells");
	check(nfail == 0, "dust sweep: every cell converged");
	check(worst_energy < 1.0e-13, "dust sweep: energy conserved to round-off");
	check(worst_state <= 2.0e-9, "dust sweep: state within 2e-9 of a 1e-12 solve");

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
				check(dust[0].converged && std::abs(dust[0].Egas - gas[0].Egas) <= 1.0e-8 * Etot &&
					  std::abs(dust[0].T_d / dust[0].T_gas - 1.0) <= 1.0e-8,
				      std::format("K = 1e12 recovers the dust-free solve (expo {}, Tg {}, Tr {})", expo, Tg, Tr));
			}
			{
				const auto cell = make_cell<Sweep1>(Tg, Tr, 1.0, 1.0e12);
				const auto dust = solve_cells<Sweep1, true>({cell}, 1.0e-9);
				const auto gas = solve_cells<Sweep1, false>({cell}, 1.0e-9);
				const double Etot = conserved_total(cell);
				check(dust[0].converged && std::abs(dust[0].Egas - gas[0].Egas) <= 1.0e-8 * Etot,
				      std::format("K = 1e12, one group (expo {}, Tg {})", expo, Tg));
			}
			{
				const auto cell = make_cell<Sweep4>(Tg, Tr, 1.0, 0.0);
				const auto dust = solve_cells<Sweep4, true>({cell}, 1.0e-9);
				const double Etot = conserved_total(cell);
				check(dust[0].converged && std::abs(dust[0].Egas - cell.Egas0) <= 1.0e-12 * Etot,
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
		check(sols[0].converged && (0.0113 < sols[0].T_d) && (sols[0].T_d < 0.0115), "three-root dust cell returns the root connected to cold gas");
	}
	return (n_failed > 0) ? 1 : 0;
}

// The two-temperature system at the conditions of a dusty D-type ionization front in CGS (the DTypeFront1D setup of
// PR #2305, and the stronger optical source and IR opacity of the DTypeFront1D shipped here): chat = c / 1000, so the
// radiation can hold far more energy than the gas; the gas at the 10 K floor or photoheated; IR and optical
// radiation from the floor to near the source; the gas-dust coupling from zero (the PR #2305 input) through the default
// coefficient to 1e8 times it; radiation substeps and hydro-sized steps; and a lagged work term of +-half the optical
// energy, which in a radiation-dominated cell charges the gas many times its own energy.
auto TestDTypeDustSweep() -> int
{
	using P = SweepDType;
	std::vector<CouplingCell<P>> cells_all;
	std::vector<std::string> labels;
	int ncells = 0;
	int nfail = 0;
	int nfail_ref = 0;
	int nfloored = 0;
	int nbad = 0;
	double worst_energy = 0.0;  // cells above the gas floor: relative change of the conserved total
	double worst_created = 0.0; // cells at the gas floor: energy created, relative to the floor energy it can come from
	double worst_state = 0.0;   // |Egas - Egas_ref| / Etot
	double worst_Td = 0.0;	    // |T_d / T_d_ref - 1|
	std::string worst_cell;
	std::int64_t nevals_sum = 0;
	int nevals_max = 0;
	for (const double kappa_ir : {1.0e-2, 10.0}) {
		dtype_kappa_ir = kappa_ir;
		dtype_kappa_opt = 1.0e3;
		std::vector<CouplingCell<P>> cells;
		std::vector<std::string> params;
		for (const double n_H : {40.0, 1.0e2, 1.0e4}) {
			for (const double Tg : {10.0, 42.0, 1.0e3, 8.0e3}) {
				for (const double E_opt : {0.0, 1.0e-13, 1.0e-11, 1.0e-9}) {
					for (const double E_ir : {0.0, 1.0e-12, 1.0e-9}) {
						for (const double k_gd : {0.0, 2.5e-34, 2.5e-30, 2.5e-26}) {
							for (const double dt : {1.6e9, 1.6e11}) {
								for (const double w : {0.0, 0.5, -0.5}) {
									cells.push_back(make_dtype_cell(n_H, Tg, E_ir, E_opt, dt, k_gd, w));
									params.push_back(std::format("kappa_ir {:.0e} n_H {:.0e} Tg {:.0f} E_opt {:.0e} E_ir "
												     "{:.0e} k_gd {:.1e} dt {:.1e} w {}",
												     kappa_ir, n_H, Tg, E_opt, E_ir, k_gd, dt, w));
								}
							}
						}
					}
				}
			}
		}
		const auto sols = solve_cells<P, true>(cells, 1.0e-9);
		const auto refs = solve_cells<P, true>(cells, 1.0e-12);
		for (std::size_t i = 0; i < cells.size(); ++i) {
			const auto &cell = cells[i];
			const auto &sol = sols[i];
			const double Etot0 = conserved_total(cell);
			const double Etot1 = total_energy(cell, sol.Egas, sol.Erad);
			nfail += sol.converged ? 0 : 1;
			nfail_ref += refs[i].converged ? 0 : 1;
			const bool finite = std::isfinite(sol.Egas) && std::isfinite(sol.T_d) && std::isfinite(sum(sol.Erad));
			const bool admissible = finite && (sol.T_d >= 0.0) && (sol.Egas >= cell.Emin * (1.0 - 1.0e-12)) &&
						(min(sol.Erad) >= RadSystem<P>::Erad_floor_ * (1.0 - 1.0e-12));
			nbad += admissible ? 0 : 1;
			// The floors only ever add energy, and only as much as the floor state holds: the gas at E_min and the
			// groups at Erad_floor. Anywhere else the step conserves the total to round-off.
			const bool floored = sol.Egas <= cell.Emin * (1.0 + 1.0e-12);
			if (floored) {
				++nfloored;
				const double floor_energy = cell.Emin + (RadSystem<P>::c_light_ / RadSystem<P>::c_hat_) * dtype_Erad_floor;
				worst_created = std::max(worst_created, (Etot1 - Etot0) / floor_energy);
				worst_energy = std::max(worst_energy, std::max(0.0, (Etot0 - Etot1) / Etot0)); // never removes energy
			} else {
				worst_energy = std::max(worst_energy, std::abs(Etot1 / Etot0 - 1.0));
			}
			const double dE = std::abs(sol.Egas - refs[i].Egas);
			worst_Td = std::max(worst_Td, std::abs(sol.T_d / refs[i].T_d - 1.0));
			if (dE / Etot0 > worst_state) {
				worst_state = dE / Etot0;
				worst_cell = std::format("{}: Egas {:.16e} vs ref {:.16e}, Etot {:.3e} ({} evals)", params[i], sol.Egas, refs[i].Egas, Etot0,
							 sol.nevals);
			}
			nevals_sum += sol.nevals;
			nevals_max = std::max(nevals_max, sol.nevals);
			++ncells;
		}
	}
	std::cout << std::format("D-type dust sweep: {} cells, {} failures ({} at 1e-12), {} inadmissible, {} at the gas floor\n", ncells, nfail, nfail_ref,
				 nbad, nfloored);
	std::cout << std::format("D-type dust sweep: vs 1e-12 solve {:.2e} of Etot in Egas and {:.2e} in T_d; energy error {:.2e}, floor energy created "
				 "{:.2e} of the floor state; evaluations mean {:.1f} max {}\n",
				 worst_state, worst_Td, worst_energy, worst_created, static_cast<double>(nevals_sum) / ncells, nevals_max);
	std::cout << std::format("D-type dust sweep: largest state error at {}\n", worst_cell);
	check(ncells == 6912, "D-type dust sweep has 6912 cells");
	check(nfail == 0 && nfail_ref == 0, "D-type dust sweep: every cell converged, at 1e-9 and at 1e-12");
	check(nbad == 0, "D-type dust sweep: every state finite and above the floors");
	check(worst_state <= 2.0e-9 && worst_Td <= 2.0e-9, "D-type dust sweep: state within 2e-9 of a 1e-12 solve");
	check(worst_energy < 1.0e-8, "D-type dust sweep: energy conserved to 1e-8 off the gas floor");
	check(worst_created <= 1.0 + 1.0e-12, "D-type dust sweep: at the gas floor, at most the floor energy is created");

	// Radiation at its floor and no gas-dust coupling: the dust emits the floor whatever T_d is, so H stays positive
	// down to T_d = 0 and the dust floor rule must return the state unchanged, converged.
	{
		const auto cell = make_dtype_cell(1.0e2, 42.0, 0.0, 0.0, 1.6e9, 0.0, 0.0);
		const auto sols = solve_cells<P, true>({cell}, 1.0e-9);
		check(sols[0].converged && std::abs(sols[0].Egas / cell.Egas0 - 1.0) <= 1.0e-12 && std::abs(sols[0].Erad[0] / cell.Erad0[0] - 1.0) <= 1.0e-12 &&
			  std::abs(sols[0].Erad[1] / cell.Erad0[1] - 1.0) <= 1.0e-12,
		      "radiation at the floor, no gas-dust coupling: converged, unchanged");
	}
	return (n_failed > 0) ? 1 : 0;
}

// A weakly coupled, radiation-dominated cell (a reviewer's counterexample): n = 0.03 cm^-3 of 10 K gas in 1459 K
// radiation, kappa = 1e3 cm^2 g^-1, dt = 1e12 s, chat = c / 1000 and the default gas-dust coefficient. The radiation holds
// 5.5e17 times the gas energy, so energy conservation cannot resolve the gas energy: at the returned T_d it gives
// 9.2 K. The gas equation can, because the collisional heating is only 1.7e-5 of the gas energy, and the heating must
// be evaluated at the gas's own temperature. The reference solves c_V (T - T0) = dt K sqrt(T) (T_d - T) at the returned
// T_d by Newton's method on the host; the exact root is 10.00016594 K.
auto TestWeakCouplingCell() -> int
{
	using P = GreyCGS;
	const double n = 0.03;
	const double T0 = 10.0;
	CouplingCell<P> cell{};
	cell.rho = n * C::m_u;
	cell.dt = 1.0e12;
	cell.tau_scale = cell.dt * RadSystem<P>::c_hat_;
	cell.dtK = cell.dt * 2.5e-34 * n * n;
	cell.Tfloor = 1.0e-3;
	cell.Emin = quokka::EOS<P>::ComputeEintFromTgas(cell.rho, cell.Tfloor, cell.massScalars);
	cell.Egas0 = quokka::EOS<P>::ComputeEintFromTgas(cell.rho, T0, cell.massScalars);
	cell.Erad0[0] = C::a_rad * std::pow(1459.0, 4);
	cell.Src.fillin(0.0);
	cell.work.fillin(0.0);
	const double c_v = cell.Egas0 / T0; // ideal gas
	for (const double tol : {1.0e-9, 1.0e-14}) {
		const auto sol = solve_cells<P, true>({cell}, tol);
		double T = T0;
		for (int k = 0; k < 50; ++k) {
			const double F = c_v * (T - T0) - cell.dtK * std::sqrt(T) * (sol[0].T_d - T);
			const double dF = c_v - cell.dtK * (0.5 * (sol[0].T_d - T) / std::sqrt(T) - std::sqrt(T));
			T -= F / dF;
		}
		const double increment_error = std::abs((sol[0].T_gas - T0) / (T - T0) - 1.0);
		std::cout << std::format("weakly coupled cell (tol {:.0e}): T_gas {:.10f} K (reference {:.10f} K), T_d {:.6f} K, increment error {:.2e}\n", tol,
					 sol[0].T_gas, T, sol[0].T_d, increment_error);
		check(sol[0].converged && std::abs(sol[0].T_gas / T - 1.0) <= 1.0e-14 && increment_error <= 1.0e-8 && std::abs(T - 10.00016594) <= 1.0e-8,
		      std::format("weakly coupled, radiation-dominated cell: gas heating exact (tol {:.0e})", tol));
	}
	return (n_failed > 0) ? 1 : 0;
}

// The group energies of an optically thick group with negligible emission: E_g = rad0 / (1 + tau), which rad0 + Delta_g
// would lose to cancellation (a relative error of 2e-5 at tau = 1e12, and zero instead of 1e-16 at tau = 1e16).
auto TestThickGroupEnergy() -> int
{
	for (const double tau : {1.0e12, 1.0e16}) {
		CouplingCell<Sweep1> cell{};
		cell.tau_scale = tau;
		cell.Erad0[0] = 1.0;
		cell.Src.fillin(0.0);
		cell.work.fillin(0.0);
		CouplingCoefficients<Sweep1> coef{};
		coef.emission[0] = 1.0e-30;
		coef.absorption[0] = 1.0;
		amrex::Gpu::DeviceScalar<double> d_E(0.0);
		double *E_ptr = d_E.dataPtr();
		amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int) noexcept { *E_ptr = RadSystem<Sweep1>::GroupEnergies(cell, coef)[0]; });
		const double E = d_E.dataValue();
		const double exact = (1.0 + tau * 1.0e-30) / (1.0 + tau);
		check(std::abs(E / exact - 1.0) <= 1.0e-15, std::format("optically thick group keeps its remaining energy (tau {:.0e})", tau));
	}
	return (n_failed > 0) ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// Round-off bounds for power-law opacities, kappa_nu = kappa_0 (nu / nu_0)^p (T / T_0)^beta with -4 < p <= 2 and
// -3 <= beta <= 2.
//
// The analysis is in docs/markdown/radiation_integrator.md ("Round-off error of the returned state"). In short: the
// solve returns the root of a scalar residual (G in E_gas, or H in T_d) whose floating-point evaluation carries an
// absolute round-off of a few eps times the gross energy traffic of the step, and the root inherits that divided by the
// slope of the residual. The group energies then follow in closed form, so their error is a few eps plus the
// temperature error times the logarithmic sensitivity of E_g to the matter temperature, which for kappa_nu ~ nu^p is
// bounded by the local exponents of the group emission and absorption. The bound is also the condition number of the
// step itself: perturbing one input by one ulp moves the exact root by the same amount.
//
// This test measures both sides of that statement. The reference solution of every cell is found by bisection in
// long double on the host, using the code's own double-precision coupling coefficients (ComputeCouplingCoefficients,
// evaluated on the device at the trial temperature rounded to double), so it isolates the arithmetic of the solve from
// the model behind the coefficients and is exact to one ulp of the trial temperature. The bound is evaluated at the
// reference root from the measured temperature derivatives of the coefficients, and the ratio of the observed error to
// the bound is reported per p.
// ---------------------------------------------------------------------------------------------------------------------

using LD = long double;
constexpr double eps_double = std::numeric_limits<double>::epsilon();
constexpr int pl_NG = pl_ngroups;

/// One PowerLawCGS cell: hydrogen number density n_H [cm^-3], gas temperature [K], radiation temperature [K] (the groups
/// carry a Planck spectrum at T_rad), step dt [s], and the collisional coefficient k_gd [erg cm^3 s^-1 K^-3/2].
auto make_pl_cell(double n_H, double Tgas, double Trad, double dt, double k_gd) -> CouplingCell<PowerLawCGS>
{
	using P = PowerLawCGS;
	CouplingCell<P> cell{};
	cell.rho = n_H * C::m_u;
	cell.dt = dt;
	cell.tau_scale = dt * RadSystem<P>::c_hat_;
	cell.dtK = dt * k_gd * n_H * n_H;
	cell.Tfloor = pl_Tfloor;
	cell.Emin = quokka::EOS<P>::ComputeEintFromTgas(cell.rho, pl_Tfloor, cell.massScalars);
	cell.Egas0 = quokka::EOS<P>::ComputeEintFromTgas(cell.rho, Tgas, cell.massScalars);
	cell.rad_boundaries = RadSystem<P>::radBoundaries_;
	for (int g = 0; g < pl_NG; ++g) {
		cell.rad_boundary_ratios[g] = cell.rad_boundaries[g + 1] / cell.rad_boundaries[g];
	}
	const auto frac = RadSystem<P>::ComputePlanckEnergyFractions(cell.rad_boundaries, Trad);
	for (int g = 0; g < pl_NG; ++g) {
		cell.Erad0[g] = std::max(RadSystem<P>::radiation_constant_ * std::pow(Trad, 4) * frac[g], pl_Erad_floor);
	}
	cell.Src.fillin(0.0);
	cell.work.fillin(0.0);
	return cell;
}

/// The coupling coefficients of every cell at its own trial temperature, evaluated on the device by the code's
/// ComputeCouplingCoefficients.
template <typename P> auto coefficients_at(std::vector<CouplingCell<P>> const &cells, std::vector<double> const &T) -> std::vector<CouplingCoefficients<P>>
{
	const int n = static_cast<int>(cells.size());
	amrex::Gpu::DeviceVector<CouplingCell<P>> d_cells(n);
	amrex::Gpu::DeviceVector<double> d_T(n);
	amrex::Gpu::DeviceVector<CouplingCoefficients<P>> d_coef(n);
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, cells.begin(), cells.end(), d_cells.begin());
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, T.begin(), T.end(), d_T.begin());
	CouplingCell<P> const *cell_ptr = d_cells.data();
	double const *T_ptr = d_T.data();
	CouplingCoefficients<P> *coef_ptr = d_coef.data();
	amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE(int i) noexcept { coef_ptr[i] = RadSystem<P>::ComputeCouplingCoefficients(cell_ptr[i], T_ptr[i]); });
	std::vector<CouplingCoefficients<P>> coef(n);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_coef.begin(), d_coef.end(), coef.begin());
	return coef;
}

/// The state of a PowerLawCGS cell at a trial unknown, in long double, from double coefficients.
struct RefEval {
	LD residual{};
	LD Egas{};
	LD T{};
	LD Tcoef{};  // the temperature the coefficients are evaluated at: T without dust, T_d with it
	LD T_cons{}; // with dust: the gas temperature from conservation, before the gas equation replaces it
	std::array<LD, pl_NG> Erad{};
};

template <bool with_dust> auto ref_eval(CouplingCell<PowerLawCGS> const &cell, CouplingCoefficients<PowerLawCGS> const &coef, LD x, LD cV) -> RefEval
{
	using P = PowerLawCGS;
	const LD cscale = static_cast<LD>(RadSystem<P>::c_light_) / static_cast<LD>(RadSystem<P>::c_hat_);
	const LD tau = cell.tau_scale;
	LD gas0 = cell.Egas0;
	LD dust_gain = 0.0L;
	RefEval r{};
	for (int g = 0; g < pl_NG; ++g) {
		const LD rad0 = static_cast<LD>(cell.Erad0[g]) + static_cast<LD>(cell.Src[g]) + static_cast<LD>(cell.work[g]);
		gas0 -= cscale * static_cast<LD>(cell.work[g]);
		const LD eps = coef.emission[g];
		const LD alpha = coef.absorption[g];
		dust_gain += cscale * tau * (eps - alpha * rad0) / (1.0L + tau * alpha);
		r.Erad[g] = (rad0 + tau * eps) / (1.0L + tau * alpha);
	}
	auto T_of = [&](LD E) { return (E > static_cast<LD>(cell.Emin)) ? E / cV : static_cast<LD>(cell.Tfloor); };
	if constexpr (with_dust) {
		r.Egas = gas0 - dust_gain;
		r.T = T_of(r.Egas);
		r.Tcoef = x;
		r.residual = dust_gain - static_cast<LD>(cell.dtK) * std::sqrt(r.T) * (r.T - x);
	} else {
		r.Egas = x;
		r.T = T_of(x);
		r.Tcoef = r.T;
		r.residual = (x - gas0) + dust_gain;
	}
	return r;
}

/// The reference root of every cell: the march of bracket_root_of_increasing from x0 (never below xmin), then bisection
/// to 1e-18 relative, all in long double and in lockstep across the batch so that the device evaluates the coefficients of
/// every cell once per step. found == false marks a cell whose march reached xmin with a positive residual (the floor
/// rule of the solver) or exhausted its budget.
struct RefRoot {
	LD x{};
	RefEval state{};
	CouplingCoefficients<PowerLawCGS> coef{};
	bool found{};
};

template <bool with_dust>
auto reference_roots(std::vector<CouplingCell<PowerLawCGS>> const &cells, std::vector<LD> const &x0, std::vector<LD> const &xmin, std::vector<LD> const &cV)
    -> std::vector<RefRoot>
{
	const int n = static_cast<int>(cells.size());
	struct Walker {
		int phase{0}; // 0 marching, 1 bisecting, 2 done
		int steps{0};
		LD x{}, f{};
		LD lo{}, hi{}, flo{}, fhi{};
		bool have_f{false};
		bool found{false};
	};
	std::vector<Walker> w(n);
	for (int i = 0; i < n; ++i) {
		w[i].x = x0[i];
	}
	auto trial_T = [&](int i) -> double {
		if constexpr (with_dust) {
			return static_cast<double>(w[i].x);
		} else {
			return static_cast<double>((w[i].x > static_cast<LD>(cells[i].Emin)) ? w[i].x / cV[i] : static_cast<LD>(cells[i].Tfloor));
		}
	};
	std::vector<double> T(n);
	for (int iter = 0; iter < 600; ++iter) {
		bool any = false;
		for (int i = 0; i < n; ++i) {
			T[i] = trial_T(i);
			any = any || (w[i].phase < 2);
		}
		if (!any) {
			break;
		}
		const auto coef = coefficients_at<PowerLawCGS>(cells, T);
		for (int i = 0; i < n; ++i) {
			auto &s = w[i];
			if (s.phase == 2) {
				continue;
			}
			const LD f = ref_eval<with_dust>(cells[i], coef[i], s.x, cV[i]).residual;
			if (s.phase == 0) {
				++s.steps;
				if (f == 0.0L) {
					s.lo = s.hi = s.x;
					s.flo = s.fhi = f;
					s.found = true;
					s.phase = 2;
					continue;
				}
				if (s.have_f && ((f > 0.0L) != (s.f > 0.0L))) {
					// sign change between the previous point and this one
					if (s.x < s.lo) {
						s.lo = s.x;
						s.flo = f;
					} else {
						s.hi = s.x;
						s.fhi = f;
					}
					s.phase = 1;
					s.x = s.lo / 2 + s.hi / 2;
					continue;
				}
				if (!s.have_f || (f < 0.0L) == (s.f < 0.0L)) {
					s.f = f;
					s.have_f = true;
					s.lo = s.hi = s.x;
					s.flo = s.fhi = f;
					if (f < 0.0L) {
						s.x = 2 * s.x;
					} else {
						if (s.x == xmin[i] || s.steps > 300) {
							s.phase = 2; // the floor rule, or no bracket: not a root
							continue;
						}
						s.x = std::max(s.x / 2, xmin[i]);
					}
				}
				continue;
			}
			// bisection
			if ((f > 0.0L) == (s.fhi > 0.0L)) {
				s.hi = s.x;
				s.fhi = f;
			} else {
				s.lo = s.x;
				s.flo = f;
			}
			const LD mid = s.lo / 2 + s.hi / 2;
			if ((s.hi - s.lo) <= 1.0e-18L * std::abs(s.hi) || mid == s.lo || mid == s.hi) {
				s.found = true;
				s.phase = 2;
				continue;
			}
			s.x = mid;
		}
	}
	// the final estimate: the chord crossing of the last bracket, as the solver does
	std::vector<RefRoot> roots(n);
	for (int i = 0; i < n; ++i) {
		auto &s = w[i];
		if (s.found && s.hi > s.lo && s.flo != s.fhi) {
			s.x = (s.lo * std::abs(s.fhi) + s.hi * std::abs(s.flo)) / (std::abs(s.flo) + std::abs(s.fhi));
		} else if (s.found) {
			s.x = s.lo;
		}
		roots[i].x = s.x;
		roots[i].found = s.found;
		T[i] = trial_T(i);
	}
	const auto coef = coefficients_at<PowerLawCGS>(cells, T);
	for (int i = 0; i < n; ++i) {
		roots[i].coef = coef[i];
		roots[i].state = ref_eval<with_dust>(cells[i], coef[i], roots[i].x, cV[i]);
		if constexpr (with_dust) {
			// At the root the gas energy can be written from conservation, gas0 - (c/chat) sum_g Delta_g, or from the gas
			// equation, E = gas0 - dt K sqrt(T) (T - T_d). In a radiation-dominated cell the first cancels terms up to
			// 1e15 times the gas energy, which even long double (eps = 5e-20) cannot resolve, so the reference takes the
			// gas equation, solved by Newton's method in T from the conservation value, whenever that converges.
			auto &st = roots[i].state;
			st.T_cons = st.T;
			if (roots[i].found && st.Egas > static_cast<LD>(cells[i].Emin)) {
				const LD cscale = static_cast<LD>(RadSystem<PowerLawCGS>::c_light_) / static_cast<LD>(RadSystem<PowerLawCGS>::c_hat_);
				LD gas0 = cells[i].Egas0;
				for (int g = 0; g < pl_NG; ++g) {
					gas0 -= cscale * static_cast<LD>(cells[i].work[g]);
				}
				const LD dtK = cells[i].dtK;
				const LD Td = roots[i].x;
				// F(T) = c_V (T - T0) + dt K sqrt(T) (T - T_d) increases with T only where T >= T_d / 3 or the coupling is
				// weak, so Newton is tried from the conservation value, then from T_d, then from the start-of-step T.
				LD Tn = NAN;
				bool settled = false;
				for (const LD start : {st.T, Td, static_cast<LD>(cells[i].Egas0) / cV[i]}) {
					Tn = start;
					for (int k = 0; k < 100 && !settled; ++k) {
						const LD sq = std::sqrt(Tn);
						const LD F = cV[i] * Tn - gas0 + dtK * sq * (Tn - Td);
						const LD dF = cV[i] + dtK * (1.5L * sq - 0.5L * Td / sq);
						if (!(dF > 0.0L) || !(Tn > 0.0L)) {
							break;
						}
						const LD step = F / dF;
						Tn -= step;
						settled = std::abs(step) <= 1.0e-18L * Tn;
					}
					if (settled) {
						break;
					}
				}
				if (settled && Tn * cV[i] > static_cast<LD>(cells[i].Emin)) {
					st.T = Tn;
					st.Egas = Tn * cV[i];
				}
			}
		}
	}
	return roots;
}

/// The round-off bound of one cell at its reference root, from the temperature derivatives of the coefficients measured
/// by central differences on the device. All quantities on the gas side (multiplied by c / chat).
struct RoundoffBound {
	double cond_T{};		     // (|gas0| + R) / (E_gas |1 + S|): the condition number of the gas energy without dust
	double slope_ratio{};		     // the residual's slope over the sum of the magnitudes of its terms: near zero only at a fold
	double bound_T{};		     // relative bound on T_gas (= E_gas for an ideal gas)
	double bound_Td{};		     // relative bound on T_d (dust only)
	std::array<double, pl_NG> bound_E{}; // relative bound on each group energy
	double sens_E_max{};		     // the largest |d ln E_g / d ln T_m| among the groups kept
};

template <bool with_dust>
auto roundoff_bounds(std::vector<CouplingCell<PowerLawCGS>> const &cells, std::vector<RefRoot> const &roots, std::vector<LD> const &cV)
    -> std::vector<RoundoffBound>
{
	using P = PowerLawCGS;
	const int n = static_cast<int>(cells.size());
	const double cscale = RadSystem<P>::c_light_ / RadSystem<P>::c_hat_;
	const double h = 1.0e-4;
	std::vector<double> Tp(n);
	std::vector<double> Tm(n);
	for (int i = 0; i < n; ++i) {
		const double Tc = static_cast<double>(roots[i].state.Tcoef);
		Tp[i] = Tc * (1.0 + h);
		Tm[i] = Tc * (1.0 - h);
	}
	const auto cp = coefficients_at<P>(cells, Tp);
	const auto cm = coefficients_at<P>(cells, Tm);
	std::vector<RoundoffBound> out(n);
	for (int i = 0; i < n; ++i) {
		auto const &cell = cells[i];
		auto const &st = roots[i].state;
		auto const &coef = roots[i].coef;
		const double Tc = static_cast<double>(st.Tcoef);
		const double T = static_cast<double>(st.T);
		const double Egas = static_cast<double>(st.Egas);
		const double tau = cell.tau_scale;
		double gas0 = cell.Egas0;
		double R = 0.0;		 // gross exchange: emitted plus absorbed over the step, gas side
		double dDelta = 0.0;	 // d(sum_g Delta_g)/dT_m, gas side
		double dDelta_abs = 0.0; // the same with every group's term taken positive
		std::array<double, pl_NG> sens{};
		std::array<double, pl_NG> coef_round{};
		for (int g = 0; g < pl_NG; ++g) {
			const double rad0 = cell.Erad0[g] + cell.Src[g] + cell.work[g];
			gas0 -= cscale * cell.work[g];
			const double eps = coef.emission[g];
			const double alpha = coef.absorption[g];
			const double deps = (cp[i].emission[g] - cm[i].emission[g]) / (2 * h * Tc);
			const double dalpha = (cp[i].absorption[g] - cm[i].absorption[g]) / (2 * h * Tc);
			// The group emission is rho kappa_P,g a T^4 f_g with f_g the Planck fraction of the group, computed as the
			// difference of the cumulative Planck integrals at the group's two edges: its absolute round-off is eps
			// times rho kappa_P,g a T^4 Y_g, with Y_g the cumulative fraction below the upper edge, not eps times the
			// group's own emission. That is the round-off of the coefficient the solve is handed, so it enters both the
			// residual and the group energy. (rho kappa_P,g a T^4 Y_g = rho kappa_P,g times the cumulative 4 pi B_k / c.)
			double planck_cum = 0.0;
			for (int k = 0; k <= g; ++k) {
				planck_cum += coef.fourPiBoverC[k];
			}
			const double eps_full = cell.rho * coef.opacity.kappaP[g] * planck_cum;
			const double absorbed = cscale * tau * alpha * rad0 / (1.0 + tau * alpha);
			const double emitted_full = cscale * tau * std::max(eps, eps_full) / (1.0 + tau * alpha);
			R += absorbed + emitted_full;
			const double dDelta_g =
			    cscale * tau * (deps * (1.0 + tau * alpha) - dalpha * (rad0 + tau * eps)) / ((1.0 + tau * alpha) * (1.0 + tau * alpha));
			dDelta += dDelta_g;
			dDelta_abs += std::abs(dDelta_g);
			// d ln E_g / d ln T_m for E_g = (rad0 + tau eps) / (1 + tau alpha)
			sens[g] = std::abs(tau * deps * Tc / (rad0 + tau * eps)) + std::abs(tau * dalpha * Tc / (1.0 + tau * alpha));
			// the coefficient round-off reaching E_g directly: tau delta(eps_g) / (1 + tau alpha) relative to E_g
			const double Eg = (rad0 + tau * eps) / (1.0 + tau * alpha);
			coef_round[g] = eps_double * tau * std::max(eps, eps_full) / ((1.0 + tau * alpha) * Eg);
		}
		const double c_v = static_cast<double>(cV[i]);
		RoundoffBound b{};
		if constexpr (!with_dust) {
			const double S = dDelta / c_v;
			b.slope_ratio = (1.0 + S) / (1.0 + dDelta_abs / c_v);
			b.cond_T = (std::abs(gas0) + R) / (Egas * std::abs(1.0 + S));
			b.bound_T = eps_double * (1.0 + b.cond_T);
			b.bound_Td = b.bound_T;
		} else {
			const double Td = Tc;
			const double Lp = cell.dtK * std::sqrt(T);				     // d(collisional term)/dT_d, up to sign
			const double LT = cell.dtK * (1.5 * std::sqrt(T) - 0.5 * Td / std::sqrt(T)); // d(collisional term)/dT at fixed T_d
			const double Hp = dDelta * (1.0 + LT / c_v) + Lp;
			const double dE_round = eps_double * (std::abs(gas0) + R);
			const double dH = eps_double * (R + Lp * (T + Td)) + std::abs(LT) * dE_round / c_v;
			b.slope_ratio = Hp / (dDelta_abs * (1.0 + std::abs(LT) / c_v) + Lp);
			b.bound_Td = dH / (std::abs(Hp) * Td);
			// the gas energy: from conservation, or from the gas equation solved at its own temperature
			const double bound_cons = (dE_round + std::abs(dDelta) * Td * b.bound_Td) / Egas;
			const double gas_form_slope = 1.0 + LT / c_v;
			double bound_gas = std::numeric_limits<double>::infinity();
			if (gas_form_slope > 0.5) {
				bound_gas = (eps_double * (std::abs(gas0) + Lp * (T + Td)) + Lp * Td * b.bound_Td) / (gas_form_slope * Egas);
			}
			b.bound_T = std::min(bound_cons, bound_gas);
			b.cond_T = bound_cons / eps_double;
		}
		for (int g = 0; g < pl_NG; ++g) {
			b.bound_E[g] = 4.0 * eps_double + coef_round[g] + sens[g] * b.bound_Td;
			b.sens_E_max = std::max(b.sens_E_max, sens[g]);
		}
		out[i] = b;
	}
	return out;
}

/// The sweep over p and beta: for each (p, beta) and each optical-depth scale, 84 cells (3 densities, 4 gas temperatures,
/// 7 radiation temperatures) solved without dust and with four dust couplings, each compared with its long-double
/// reference.
auto TestRoundoffBounds() -> int
{
	using P = PowerLawCGS;
	constexpr double dt = 1.0e9; // s
	constexpr std::array<double, 7> expos{-3.9, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0};
	constexpr std::array<double, 5> betas{-3.0, -1.5, 0.0, 1.0, 2.0};
	constexpr std::array<double, 5> tau0s{1.0e-4, 1.0e-1, 1.0e2, 1.0e5, 1.0e8}; // rho kappa_0 chat dt at n_H = 100
	constexpr std::array<double, 3> densities{1.0e-2, 1.0e2, 1.0e4};
	constexpr std::array<double, 4> Tgases{10.0, 1.0e2, 1.0e3, 1.0e4};
	constexpr std::array<double, 7> Tratios{0.1, 0.5, 1.0, 1.001, 2.0, 10.0, 100.0};
	constexpr std::array<double, 4> kgds{0.0, 2.5e-34, 2.5e-30, 2.5e-26};
	constexpr double tol = 1.0e-11;
	// the bound constants asserted below; the measured maxima are printed
	constexpr double C_T = 4.0;
	constexpr double C_E = 8.0;
	constexpr double C_Td = 16.0;

	struct Stats {
		int cells{0}, checked{0}, floored{0}, unconverged{0}, fold{0}, branch{0};
		double max_err_T{0}, max_err_E{0}, max_err_Td{0};
		double max_ratio_T{0}, max_ratio_E{0}, max_ratio_Td{0};
		double max_cond{0}, max_sens{0};
		std::string worst_T, worst_E;
	};
	std::array<std::array<Stats, betas.size()>, expos.size()> stats_gas{};
	std::array<std::array<Stats, betas.size()>, expos.size()> stats_dust{};

	auto accumulate = [&](Stats &s, bool with_dust, std::vector<CouplingCell<P>> const &cells, std::vector<CouplingSolution<P>> const &sols,
			      std::vector<RefRoot> const &roots, std::vector<RoundoffBound> const &bounds, std::vector<std::string> const &labels) {
		for (std::size_t i = 0; i < cells.size(); ++i) {
			++s.cells;
			if (!sols[i].converged) {
				++s.unconverged;
				continue;
			}
			// cells the solver or the reference clamped at a floor carry no root to compare with
			if (!roots[i].found || sols[i].Egas <= cells[i].Emin * (1.0 + 1.0e-10) ||
			    roots[i].state.Egas <= static_cast<LD>(cells[i].Emin) * (1.0L + 1.0e-10L)) {
				++s.floored;
				continue;
			}
			// near a fold of the residual (its slope cancels to under a tenth of the magnitude of its terms) the branch
			// itself is decided by rounding; the bound is formally valid but the comparison is not meaningful there
			if (std::abs(bounds[i].slope_ratio) < 0.1) {
				++s.fold;
				continue;
			}
			const double T_ref = static_cast<double>(roots[i].state.T);
			const double err_T = std::abs(sols[i].T_gas / T_ref - 1.0);
			if (err_T > 1.0e-3) {
				++s.branch; // the solver and the reference followed different roots
				continue;
			}
			++s.checked;
			double err_E = 0.0;
			double ratio_E = 0.0;
			for (int g = 0; g < pl_NG; ++g) {
				const double E_ref = static_cast<double>(roots[i].state.Erad[g]);
				if (E_ref <= 2.0 * pl_Erad_floor) {
					continue; // a group at the floor is set by ApplyEnergyFloors, not by the solve
				}
				const double e = std::abs(sols[i].Erad[g] / E_ref - 1.0);
				err_E = std::max(err_E, e);
				ratio_E = std::max(ratio_E, e / bounds[i].bound_E[g]);
			}
			const double ratio_T = err_T / bounds[i].bound_T;
			s.max_err_T = std::max(s.max_err_T, err_T);
			s.max_err_E = std::max(s.max_err_E, err_E);
			s.max_cond = std::max(s.max_cond, bounds[i].cond_T);
			s.max_sens = std::max(s.max_sens, bounds[i].sens_E_max);
			if (ratio_T > s.max_ratio_T) {
				s.max_ratio_T = ratio_T;
				s.worst_T = std::format("{} err_T {:.2e} bound {:.2e} cond {:.1e}", labels[i], err_T, bounds[i].bound_T, bounds[i].cond_T);
			}
			if (ratio_E > s.max_ratio_E) {
				s.max_ratio_E = ratio_E;
				s.worst_E = std::format("{} err_E {:.2e} sens {:.1f}", labels[i], err_E, bounds[i].sens_E_max);
			}
			if (with_dust) {
				const double Td_ref = static_cast<double>(roots[i].x);
				const double err_Td = std::abs(sols[i].T_d / Td_ref - 1.0);
				s.max_err_Td = std::max(s.max_err_Td, err_Td);
				s.max_ratio_Td = std::max(s.max_ratio_Td, err_Td / bounds[i].bound_Td);
			}
		}
	};

	for (std::size_t ip = 0; ip < expos.size(); ++ip) {
		for (std::size_t ib = 0; ib < betas.size(); ++ib) {
			pl_expo = expos[ip];
			pl_beta = betas[ib];
			for (const double tau0 : tau0s) {
				pl_kappa0 = tau0 / (1.0e2 * C::m_u * RadSystem<P>::c_hat_ * dt);
				std::vector<CouplingCell<P>> cells_gas;
				std::vector<CouplingCell<P>> cells_dust;
				std::vector<std::string> labels_gas;
				std::vector<std::string> labels_dust;
				for (const double n_H : densities) {
					for (const double Tg : Tgases) {
						for (const double ratio : Tratios) {
							cells_gas.push_back(make_pl_cell(n_H, Tg, ratio * Tg, dt, 0.0));
							labels_gas.push_back(std::format("p {} beta {} tau0 {:.0e} n_H {:.0e} Tg {:.0f} Tr/Tg {}", expos[ip],
											 betas[ib], tau0, n_H, Tg, ratio));
							for (const double k_gd : kgds) {
								cells_dust.push_back(make_pl_cell(n_H, Tg, ratio * Tg, dt, k_gd));
								labels_dust.push_back(std::format("{} k_gd {:.1e}", labels_gas.back(), k_gd));
							}
						}
					}
				}
				auto cV_of = [](std::vector<CouplingCell<P>> const &cells) {
					std::vector<LD> cV(cells.size());
					for (std::size_t i = 0; i < cells.size(); ++i) {
						cV[i] = static_cast<LD>(quokka::EOS<P>::ComputeEintFromTgas(cells[i].rho, 1.0, cells[i].massScalars));
					}
					return cV;
				};
				// without dust: the unknown is E_gas, marched from Egas0 and never below the solver's march floor
				{
					const auto cV = cV_of(cells_gas);
					std::vector<LD> x0(cells_gas.size());
					std::vector<LD> xmin(cells_gas.size());
					for (std::size_t i = 0; i < cells_gas.size(); ++i) {
						xmin[i] = std::max(static_cast<LD>(cells_gas[i].Emin),
								   16.0L * static_cast<LD>(eps_double) * static_cast<LD>(cells_gas[i].Egas0));
						x0[i] = std::max(static_cast<LD>(cells_gas[i].Egas0), xmin[i]);
					}
					const auto sols = solve_cells<P, false>(cells_gas, tol);
					const auto roots = reference_roots<false>(cells_gas, x0, xmin, cV);
					const auto bounds = roundoff_bounds<false>(cells_gas, roots, cV);
					accumulate(stats_gas[ip][ib], false, cells_gas, sols, roots, bounds, labels_gas);
				}
				// with dust: the unknown is T_d, marched from the start-of-step gas temperature down to T_d^min
				{
					const auto cV = cV_of(cells_dust);
					std::vector<LD> x0(cells_dust.size());
					std::vector<LD> xmin(cells_dust.size());
					const LD T_emit_floor =
					    std::pow(static_cast<LD>(pl_Erad_floor) / static_cast<LD>(RadSystem<P>::radiation_constant_), 0.25L);
					for (std::size_t i = 0; i < cells_dust.size(); ++i) {
						const LD T0 = std::max(static_cast<LD>(cells_dust[i].Egas0) / cV[i], static_cast<LD>(cells_dust[i].Tfloor));
						x0[i] = T0;
						xmin[i] = std::min(std::max(T_emit_floor, 1.0e-10L * T0), T0);
					}
					const auto sols = solve_cells<P, true>(cells_dust, tol);
					const auto roots = reference_roots<true>(cells_dust, x0, xmin, cV);
					const auto bounds = roundoff_bounds<true>(cells_dust, roots, cV);
					accumulate(stats_dust[ip][ib], true, cells_dust, sols, roots, bounds, labels_dust);
				}
			}
		}
	}

	auto report = [&](std::string const &name, std::array<std::array<Stats, betas.size()>, expos.size()> const &stats, bool with_dust) {
		std::cout << std::format("round-off bounds, {} (kappa_nu ~ nu^p T^beta, {} groups, chat = c/1000, reference in long double):\n", name, pl_NG);
		std::cout << std::format("{:>5} {:>5} {:>6} {:>7} {:>7} {:>5} {:>6} {:>6} {:>9} {:>8} {:>9} {:>8} {:>8} {:>7}{}\n", "p", "beta", "cells",
					 "checked", "floored", "fold", "branch", "unconv", "err_T", "ratio_T", "err_E", "ratio_E", "cond_T", "sens_E",
					 with_dust ? "    err_Td ratio_Td" : "");
		Stats worst_T_all{};
		Stats worst_E_all{};
		for (std::size_t ip = 0; ip < expos.size(); ++ip) {
			for (std::size_t ib = 0; ib < betas.size(); ++ib) {
				auto const &s = stats[ip][ib];
				std::cout << std::format(
				    "{:>5} {:>5} {:>6} {:>7} {:>7} {:>5} {:>6} {:>6} {:>9.2e} {:>8.2f} {:>9.2e} {:>8.2f} {:>8.1e} {:>7.1f}", expos[ip],
				    betas[ib], s.cells, s.checked, s.floored, s.fold, s.branch, s.unconverged, s.max_err_T, s.max_ratio_T, s.max_err_E,
				    s.max_ratio_E, s.max_cond, s.max_sens);
				if (with_dust) {
					std::cout << std::format(" {:>9.2e} {:>8.2f}", s.max_err_Td, s.max_ratio_Td);
				}
				std::cout << "\n";
				if (s.max_ratio_T > worst_T_all.max_ratio_T) {
					worst_T_all = s;
				}
				if (s.max_ratio_E > worst_E_all.max_ratio_E) {
					worst_E_all = s;
				}
			}
		}
		std::cout << std::format("  worst T ratio at {}\n  worst E ratio at {}\n", worst_T_all.worst_T, worst_E_all.worst_E);
	};
	report("one-temperature model", stats_gas, false);
	report("two-temperature model", stats_dust, true);

	int checked_gas = 0;
	int checked_dust = 0;
	int unconverged = 0;
	double worst_T = 0.0;
	double worst_E = 0.0;
	double worst_Td = 0.0;
	for (std::size_t ip = 0; ip < expos.size(); ++ip) {
		for (std::size_t ib = 0; ib < betas.size(); ++ib) {
			checked_gas += stats_gas[ip][ib].checked;
			checked_dust += stats_dust[ip][ib].checked;
			unconverged += stats_gas[ip][ib].unconverged + stats_dust[ip][ib].unconverged;
			worst_T = std::max({worst_T, stats_gas[ip][ib].max_ratio_T, stats_dust[ip][ib].max_ratio_T});
			worst_E = std::max({worst_E, stats_gas[ip][ib].max_ratio_E, stats_dust[ip][ib].max_ratio_E});
			worst_Td = std::max(worst_Td, stats_dust[ip][ib].max_ratio_Td);
		}
	}
	check(unconverged == 0, "round-off sweep: every cell converged");
	check(checked_gas >= 10000 && checked_dust >= 30000,
	      std::format("round-off sweep: {} + {} cells compared with the reference", checked_gas, checked_dust));
	check(worst_T <= C_T, std::format("round-off sweep: gas temperature error within {} times its bound (max ratio {:.2f})", C_T, worst_T));
	check(worst_E <= C_E, std::format("round-off sweep: group energy error within {} times its bound (max ratio {:.2f})", C_E, worst_E));
	check(worst_Td <= C_Td, std::format("round-off sweep: dust temperature error within {} times its bound (max ratio {:.2f})", C_Td, worst_Td));
	return (n_failed > 0) ? 1 : 0;
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
	const int dtype_status = TestDTypeDustSweep();
	const int thick_status = TestThickGroupEnergy();
	const int weak_status = TestWeakCouplingCell();
	const int roundoff_status = TestRoundoffBounds();
	const int status = (gas_status == 0 && dust_status == 0 && dtype_status == 0 && thick_status == 0 && weak_status == 0 && roundoff_status == 0) ? 0 : 1;
	std::cout << (status == 0 ? "RadCouplingUnitTests: all tests passed.\n" : "RadCouplingUnitTests: FAILED.\n");
	return (n_failed > 0) ? 1 : 0;
}
