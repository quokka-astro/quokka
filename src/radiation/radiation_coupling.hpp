// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef RADIATION_COUPLING_HPP_ // NOLINT
#define RADIATION_COUPLING_HPP_
/// \file radiation_coupling.hpp
/// \brief One backward-Euler step of the matter-radiation energy exchange, reduced to one scalar equation.
///
/// The N_G group equations of the step are linear in the group energies at a fixed matter temperature, so they are
/// solved in closed form (GroupEnergies) and the whole step reduces to one equation: without dust, the statement that the
/// cell's total energy is unchanged, G(E_gas) = 0 (GasCouplingState); with dust, the gas equation in the dust
/// temperature, H(T_d) = 0 (DustCouplingState), with the gas energy following from conservation. Both are increasing in
/// their unknown except where a steep opacity law makes them non-monotone, so the bracket is built by marching outward
/// from the old state (which selects the root continuously connected to it) and the root is found with Brent's method to
/// a relative tolerance on the unknown. The method, its derivation and its measurements are hydro3d.jl's
/// docs/coupling-new-method.md and docs/coupling-new-method-dust.md; the Quokka specifics are in the PR's design note.

#include <cmath>
#include <limits>

#include "math/bracketing_root_finding.hpp"
#include "math/root_finding.hpp"
#include "radiation/radiation_system.hpp" // IWYU pragma: keep

// Gas temperature from the internal energy, held at the floor below it: a trial energy handed over by the bracket march
// or by the dust balance may lie below the floor, and the Planck function must still be evaluable there.
template <typename problem_t> AMREX_GPU_DEVICE auto RadSystem<problem_t>::TgasOf(CouplingCell<problem_t> const &cell, double const Egas) -> double
{
	if (!(Egas > cell.Emin)) {
		return cell.Tfloor;
	}
	return ::quokka::EOS<problem_t>::ComputeTgasFromEint(cell.rho, Egas, cell.massScalars);
}

// The energy the step conserves: E_gas^0 + (c/chat) sum_g (E_g^0 + S_g). The lagged work term cancels out of it (it is
// subtracted from the gas and added to the radiation, both counted here at c/chat).
template <typename problem_t> AMREX_GPU_DEVICE auto RadSystem<problem_t>::TotalEnergy(CouplingCell<problem_t> const &cell) -> double
{
	return cell.Egas0 + (c_light_ / c_hat_) * sum(cell.Erad0 + cell.Src);
}

// emission_g = rho kappa_P,g 4 pi B_g / c and absorption_g = rho kappa_E,g at the matter temperature T: the single
// source of truth for the coupling physics. Bands that do not emit (chemical bands, and every band under
// dust_absorption_only) get zero emission from the thermal-radiation hooks and keep their absorption.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeCouplingCoefficients(CouplingCell<problem_t> const &cell, double const T) -> CouplingCoefficients<problem_t>
{
	CouplingCoefficients<problem_t> coef{};
	if constexpr (nGroups_ == 1) {
		// A single chemical (ionizing) band emits no blackbody radiation (nGroupsThermal_ == 0).
		coef.fourPiBoverC[0] = (nGroupsThermal_ == 0) ? 0.0 : ComputeThermalRadiationSingleGroup(T);
		coef.opacity.kappaP[0] = ComputePlanckOpacity(cell.rho, T);
		coef.opacity.kappaE[0] = ComputeEnergyMeanOpacity(cell.rho, T);
	} else {
		coef.fourPiBoverC = ComputeThermalRadiationMultiGroup(T, cell.rad_boundaries);
		// PPL_opacity_full_spectrum fits alpha_E to the spectrum the groups start the step with and alpha_P to the
		// Planck spectrum at T; the other models ignore the last argument.
		coef.opacity = ComputeModelDependentKappaEAndKappaP(T, cell.rho, cell.rad_boundaries, cell.rad_boundary_ratios, coef.fourPiBoverC,
								    cell.Erad0 + cell.Src + cell.work);
	}
	for (int g = 0; g < nGroups_; ++g) {
		AMREX_ASSERT(coef.opacity.kappaP[g] >= 0.0);
		AMREX_ASSERT(coef.opacity.kappaE[g] >= 0.0);
		coef.emission[g] = cell.rho * coef.opacity.kappaP[g] * coef.fourPiBoverC[g];
		coef.absorption[g] = cell.rho * coef.opacity.kappaE[g];
	}
	return coef;
}

// The group block of the step, solved in closed form:
//     E_g = (rad0_g + tau_scale emission_g) / (1 + tau_scale absorption_g),   rad0_g = E_g^0 + S_g + W_g .
// A convex combination of where the group started and the Planck value at the matter temperature, weighted by the
// optical depth of the step: all of the stiffness is in that one weight and is handled exactly. A transparent group
// (zero opacity) keeps rad0_g, source included.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::GroupEnergies(CouplingCell<problem_t> const &cell, CouplingCoefficients<problem_t> const &coef)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> Erad{};
	for (int g = 0; g < nGroups_; ++g) {
		const double rad0 = cell.Erad0[g] + cell.Src[g] + cell.work[g];
		Erad[g] = (rad0 + cell.tau_scale * coef.emission[g]) / (1.0 + cell.tau_scale * coef.absorption[g]);
	}
	return Erad;
}

// The state at a trial gas energy, and G there: the new total energy minus the old. Emission and absorption each move
// energy between the gas and a group, and both sides are counted, so they cancel out of G; what is left is the
// conservation error of the step, which is what the solve drives to zero.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::GasCouplingState(CouplingCell<problem_t> const &cell, double const Egas) -> CouplingSolution<problem_t>
{
	CouplingSolution<problem_t> sol{};
	sol.Egas = Egas;
	sol.T_gas = TgasOf(cell, Egas);
	sol.T_d = sol.T_gas;
	sol.Erad = GroupEnergies(cell, ComputeCouplingCoefficients(cell, sol.T_gas));
	sol.residual = Egas + (c_light_ / c_hat_) * sum(sol.Erad) - TotalEnergy(cell);
	return sol;
}

// Project the solved state onto the admissible set by moving energy between the gas and the radiation rather than
// creating it: lift every group to the floor and charge the gas, and if the gas then cannot pay, take the shortfall back
// from every group in proportion to what it holds above the floor, so the spectrum keeps its shape.
template <typename problem_t>
AMREX_GPU_DEVICE void RadSystem<problem_t>::ApplyEnergyFloors(CouplingCell<problem_t> const &cell, CouplingSolution<problem_t> &sol)
{
	const double cscale = c_light_ / c_hat_;
	double paid = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		const double lifted = amrex::max(sol.Erad[g], Erad_floor_);
		paid += cscale * (lifted - sol.Erad[g]);
		sol.Erad[g] = lifted;
	}
	sol.Egas -= paid;
	sol.T_gas = TgasOf(cell, sol.Egas);
	if (sol.Egas >= cell.Emin) {
		return;
	}
	double headroom = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		headroom += cscale * (sol.Erad[g] - Erad_floor_);
	}
	const double frac = (headroom > 0.0) ? amrex::min((cell.Emin - sol.Egas) / headroom, 1.0) : 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		sol.Erad[g] -= frac * (sol.Erad[g] - Erad_floor_);
	}
	sol.Egas = amrex::max(sol.Egas + frac * headroom, cell.Emin);
	sol.T_gas = TgasOf(cell, sol.Egas);
}

// The solve without dust: bracket G by marching from the old gas energy, hand it to Brent, and take the midpoint of the
// final bracket. The tolerance is relative on the unknown, |hi - lo| <= tol min(|lo|, |hi|), which for an ideal gas is a
// relative tolerance on the gas temperature. The conservation error that follows is about dG/dE_gas times tol E_gas; it
// is not tested per cell (RadCouplingUnitTests measures it over hydro3d's sweep).
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveGasCoupling(CouplingCell<problem_t> const &cell, double const tol) -> CouplingSolution<problem_t>
{
	int nevals = 0;
	auto G = [&](double const Egas) {
		++nevals;
		return GasCouplingState(cell, Egas).residual;
	};

	AMREX_ASSERT(cell.Egas0 > 0.0);
	const auto br = quokka::math::bracket_root_of_increasing(G, amrex::max(cell.Egas0, cell.Emin), cell.Emin);
	if (!br.found) {
		// reported unconverged rather than guessed at, with one exception: the march stopped at the gas energy
		// floor (br.lo == cell.Emin) because the equilibrium value lies below it, and G there is already smaller
		// than tol times the cell's energy scale. Clamping to the floor is the best an inadmissible root allows,
		// and the conservation error it leaves is below the solve's own tolerance (a transparent, radiation-
		// dominated cell whose gas has relaxed to the EOS floor: RadStreamingFluxSource).
		CouplingSolution<problem_t> sol{};
		sol.Egas = cell.Egas0;
		sol.T_gas = TgasOf(cell, cell.Egas0);
		sol.T_d = sol.T_gas;
		sol.Erad = cell.Erad0 + cell.Src;
		sol.residual = br.flo;
		sol.nevals = br.nevals;
		const double atol = std::abs(TotalEnergy(cell)) * amrex::max(tol, 4 * std::numeric_limits<double>::epsilon());
		sol.converged = (br.lo == cell.Emin) && (amrex::max(std::abs(br.flo), std::abs(br.fhi)) <= atol);
		ApplyEnergyFloors(cell, sol);
		sol.T_d = sol.T_gas;
		return sol;
	}

	quokka::math::eps_tolerance<double> tolerance(tol);
	int iters = max_root_iterations_;
	const auto [lo, hi] = quokka::math::brent_solve(G, br.lo, br.hi, br.flo, br.fhi, tolerance, iters);
	auto sol = GasCouplingState(cell, lo / 2 + hi / 2);
	sol.converged = tolerance(lo, hi);
	sol.nevals = nevals;
	ApplyEnergyFloors(cell, sol);
	sol.T_d = sol.T_gas; // without dust the radiation couples at the gas temperature
	return sol;
}

// The state at a trial dust temperature, and H there. The dust holds no energy and the radiation couples to it alone, so
// given T_d everything else is closed form: the group block at T_d (GroupEnergies), the gas energy from conservation,
// and what is left is the gas equation,
//     H = (gas0 - E_gas) - dt K sqrt(T) (T - T_d) ,   gas0 = E_gas^0 - (c/chat) sum_g W_g ,
// zero where the dust is in balance, written with the sign that makes it increase with T_d (hotter dust radiates more
// and is fed less). Total energy is conserved to round-off at every trial T_d. At K = 0 the gas is untouched and H = 0 is
// radiative equilibrium of the dust; as K -> infinity, T_d -> T and the dust-free step is recovered.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::DustCouplingState(CouplingCell<problem_t> const &cell, double const T_d) -> CouplingSolution<problem_t>
{
	const double cscale = c_light_ / c_hat_;
	CouplingSolution<problem_t> sol{};
	sol.T_d = T_d;
	sol.Erad = GroupEnergies(cell, ComputeCouplingCoefficients(cell, T_d));
	sol.Egas = TotalEnergy(cell) - cscale * sum(sol.Erad);
	sol.T_gas = TgasOf(cell, sol.Egas);
	const double gas0 = cell.Egas0 - cscale * sum(cell.work);
	const double T = sol.T_gas;
	sol.residual = (gas0 - sol.Egas) - cell.dtK * std::sqrt(T) * (T - T_d);
	return sol;
}

// The solve with dust: bracket H by marching from the gas temperature at the start of the step (Quokka's initial guess
// for the dust temperature, which where the dust balance has several roots selects the one connected to dust as warm as
// the gas), hand it to Brent, and take the midpoint of the final bracket. The convergence test is on the state
// (hydro3d.jl, docs/coupling-new-method-dust.md sections 4-5). H is not the test: its collision term multiplies the
// round-off in T - T_d by dt K sqrt(T), which reaches 1e8 of the cell's energy in a strongly coupled cell. The bracket
// width in T_d is not the test either: at small K the gas energy follows from conservation, and its error is the T_d
// error times (c/chat) sum_g E_g / E_gas, which is large where the radiation holds most of the energy (DTypeFront1D).
// So the test is the gas energy across the final bracket, relative to itself: |E_gas(lo) - E_gas(hi)| <= atol,
// atol = max(tol |E_gas|, 4 eps E_tot), which is a relative tolerance on T_gas. It is floored by the round-off of the
// conservation subtraction E_gas = E_tot - (c/chat) sum_g E_g, which cannot resolve E_gas better than eps E_tot, and by
// the resolution of T_d: once the T_d bracket is a few ulp wide it cannot shrink further, the gas-energy width across it
// is round-off, and the cell is converged. Otherwise, where the test fails, the T_d tolerance is set from the slope
// dE_gas/dT_d measured across the bracket and the bracket re-solved, at most three times.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveDustCoupling(CouplingCell<problem_t> const &cell, double const tol) -> CouplingSolution<problem_t>
{
	int nevals = 0;
	auto H = [&](double const T_d) {
		++nevals;
		return DustCouplingState(cell, T_d).residual;
	};

	const double T0 = amrex::max(TgasOf(cell, cell.Egas0), cell.Tfloor);
	AMREX_ASSERT(T0 > 0.0);
	// The march is not stopped at the gas temperature floor: the dust holds no energy and is not floored, and where the
	// radiation field is near Erad_floor its radiative-equilibrium temperature can lie far below T_floor, where the floor
	// emission alone already makes H positive (DTypeFront1D: T_floor = 10 K). The old Newton solver did not floor T_d either.
	const auto br = quokka::math::bracket_root_of_increasing(H, T0, 0.0);
	if (!br.found) {
		CouplingSolution<problem_t> sol{};
		sol.Egas = cell.Egas0;
		sol.T_gas = TgasOf(cell, cell.Egas0);
		sol.T_d = T0;
		sol.Erad = cell.Erad0 + cell.Src;
		sol.residual = br.flo;
		sol.nevals = br.nevals;
		sol.converged = false;
		ApplyEnergyFloors(cell, sol);
		return sol;
	}

	const double atol_floor = 4 * std::numeric_limits<double>::epsilon() * std::abs(TotalEnergy(cell));
	int iters = max_root_iterations_;
	auto [lo, hi] = quokka::math::brent_solve(H, br.lo, br.hi, br.flo, br.fhi, quokka::math::eps_tolerance<double>(tol), iters);

	// the spread of the gas energy across the final bracket; two evaluations, counted
	auto gas_width = [&](double const a, double const b) {
		if (a == b) {
			return 0.0;
		}
		nevals += 2;
		return std::abs(DustCouplingState(cell, a).Egas - DustCouplingState(cell, b).Egas);
	};
	constexpr double eps_mach = std::numeric_limits<double>::epsilon();
	auto mid = DustCouplingState(cell, lo / 2 + hi / 2);
	double atol = amrex::max(tol * std::abs(mid.Egas), atol_floor);
	double width = gas_width(lo, hi);
	auto at_fp_limit = [&]() { return (hi - lo) <= 4 * eps_mach * std::abs(lo / 2 + hi / 2); };
	const int max_solves = 4; // the first solve and at most three repeats
	for (int nsolve = 1; (width > atol) && !at_fp_limit() && (nsolve < max_solves); ++nsolve) {
		// the T_d half-width that would bring the gas-energy width to atol, from the slope measured across the bracket;
		// the factor 4: 2 for the half- versus full-width of the bracket, 2 as margin for the slope changing across it
		const double slope = width / (hi - lo);
		const double eps = amrex::max(atol / (4.0 * slope) / std::abs(lo / 2 + hi / 2), 4 * eps_mach);
		iters = max_root_iterations_;
		const auto bracket = quokka::math::brent_solve(H, lo, hi, quokka::math::eps_tolerance<double>(eps), iters);
		lo = bracket.first;
		hi = bracket.second;
		mid = DustCouplingState(cell, lo / 2 + hi / 2);
		atol = amrex::max(tol * std::abs(mid.Egas), atol_floor);
		width = gas_width(lo, hi);
	}

	auto sol = mid;
	sol.converged = (width <= atol) || at_fp_limit();
	sol.nevals = nevals;
	ApplyEnergyFloors(cell, sol);
	return sol;
}

#endif // RADIATION_COUPLING_HPP_
