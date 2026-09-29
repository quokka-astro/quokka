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
/// docs/coupling-new-method.md and docs/coupling-new-method-dust.md; the Quokka version is described in
/// docs/markdown/radiation_integrator.md ("Matter-radiation coupling solve").

#include <cmath>
#include <limits>

#include "math/bracketing_root_finding.hpp"
#include "math/root_finding.hpp"
#include "radiation/radiation_system.hpp" // IWYU pragma: keep // NOLINT(misc-header-include-cycle)

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

// The energy each group gains from the matter over the step, in the closed form of the backward-Euler group block:
//     Delta_g = tau_scale (emission_g - absorption_g rad0_g) / (1 + tau_scale absorption_g),   rad0_g = E_g^0 + S_g + W_g ,
// so that E_g = rad0_g + Delta_g = (rad0_g + tau_scale emission_g) / (1 + tau_scale absorption_g), a convex combination of
// where the group started and the Planck value at the matter temperature, weighted by the optical depth of the step. The
// exchange is formed as a difference of rates, not of energies: the gas energy then follows from gas0 - (c/chat) sum Delta_g
// without subtracting the radiation energy from the cell's total, which in a radiation-dominated cell would leave the gas
// energy with the round-off of a number 1e5 times larger than itself (DTypeFront1D's mirror-symmetry check measures that).
// A transparent group (zero opacity) has Delta_g = 0 exactly and keeps rad0_g, source included.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::GroupExchange(CouplingCell<problem_t> const &cell, CouplingCoefficients<problem_t> const &coef)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> exchange{};
	for (int g = 0; g < nGroups_; ++g) {
		const double rad0 = cell.Erad0[g] + cell.Src[g] + cell.work[g];
		exchange[g] = cell.tau_scale * (coef.emission[g] - coef.absorption[g] * rad0) / (1.0 + cell.tau_scale * coef.absorption[g]);
	}
	return exchange;
}

// The group energies implied by the exchange: E_g = rad0_g + Delta_g.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::GroupEnergies(CouplingCell<problem_t> const &cell, quokka::valarray<double, nGroups_> const &exchange)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> Erad{};
	for (int g = 0; g < nGroups_; ++g) {
		Erad[g] = cell.Erad0[g] + cell.Src[g] + cell.work[g] + exchange[g];
	}
	return Erad;
}

// The state at a trial gas energy, and G there: the new total energy minus the old. Emission and absorption each move
// energy between the gas and a group, and both sides are counted, so they cancel out of G; what is left is the
// conservation error of the step, which is what the solve drives to zero.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::GasCouplingState(CouplingCell<problem_t> const &cell, double const Egas) -> CouplingSolution<problem_t>
{
	const double cscale = c_light_ / c_hat_;
	CouplingSolution<problem_t> sol{};
	sol.Egas = Egas;
	sol.T_gas = TgasOf(cell, Egas);
	sol.T_d = sol.T_gas;
	const auto exchange = GroupExchange(cell, ComputeCouplingCoefficients(cell, sol.T_gas));
	sol.Erad = GroupEnergies(cell, exchange);
	// G = (E_gas - gas0) + (c/chat) sum_g Delta_g: the new total energy minus the old, written through the exchange
	const double gas0 = cell.Egas0 - cscale * sum(cell.work);
	sol.residual = (Egas - gas0) + cscale * sum(exchange);
	return sol;
}

// Project the solved state onto the admissible set by moving energy between the gas and the radiation rather than
// creating it: lift every group to the floor and charge the gas, and if the gas then cannot pay, take the shortfall back
// from every group in proportion to what it holds above the floor, so the spectrum keeps its shape.
template <typename problem_t>
AMREX_GPU_DEVICE void RadSystem<problem_t>::ApplyEnergyFloors(CouplingCell<problem_t> const &cell, CouplingSolution<problem_t> &sol)
{
	const double cscale = c_light_ / c_hat_;
	const double erad_floor = Erad_floor_; // local copy: nvcc cannot address a static constexpr member in device code
	double paid = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		const double lifted = amrex::max(sol.Erad[g], erad_floor);
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
		headroom += cscale * (sol.Erad[g] - erad_floor);
	}
	const double frac = (headroom > 0.0) ? amrex::min((cell.Emin - sol.Egas) / headroom, 1.0) : 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		sol.Erad[g] -= frac * (sol.Erad[g] - erad_floor);
	}
	sol.Egas = amrex::max(sol.Egas + frac * headroom, cell.Emin);
	sol.T_gas = TgasOf(cell, sol.Egas);
}

// The solve without dust: bracket G by marching from the old gas energy, hand it to Brent, and take the state where the
// chord through the ends of the final bracket crosses zero. The tolerance is relative on the unknown, |hi - lo| <= tol min(|lo|, |hi|), which for an ideal gas is a
// relative tolerance on the gas temperature. The conservation error that follows is at most about dG/dE_gas times
// tol E_gas (much less at the chord crossing); it
// is not tested per cell (RadCouplingUnitTests measures it over a sweep of 240 cells taken from hydro3d.jl).
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveGasCoupling(CouplingCell<problem_t> const &cell, double const tol) -> CouplingSolution<problem_t>
{
	int nevals = 0;
	auto G = [&](double const Egas) {
		++nevals;
		return GasCouplingState(cell, Egas).residual;
	};

	AMREX_ASSERT(cell.Egas0 > 0.0);
	// a relative floor at round-off of the initial gas energy, so that a zero temperature floor (every CONSTANTS-unit
	// problem) still ends the march; probing exactly E = 0 would evaluate the Planck function at T = 0
	const double Emin_march = amrex::max(cell.Emin, 16.0 * std::numeric_limits<double>::epsilon() * cell.Egas0);
	const auto br = quokka::math::bracket_root_of_increasing(G, amrex::max(cell.Egas0, Emin_march), Emin_march);
	if (!br.found) {
		// br.flo > 0: the march reached the floor with a positive residual, the only way it gets there legitimately
		// (a non-finite residual also stops the march, and is reported unconverged below)
		if ((br.lo == Emin_march) && (br.flo > 0.0)) {
			// the march reached the gas floor without a sign change, so the root lies below the admissible range:
			// the gas is clamped to the floor and the groups take the closed form at T_floor; the energy
			// G(Emin) > 0 is created by the temperature floor, as any floor does, and there is no threshold on it
			// (a threshold would be a residual test in disguise). Routine in a transparent, radiation-dominated
			// cell whose gas has relaxed to the floor and whose lagged work term is slightly positive
			// (RadStreamingFluxSource).
			auto sol = GasCouplingState(cell, Emin_march);
			sol.converged = true;
			sol.nevals = br.nevals + 1;
			ApplyEnergyFloors(cell, sol);
			sol.T_d = sol.T_gas;
			return sol;
		}
		// the march exhausted its budget going upward without a sign change, or met a non-finite residual: reported
		// unconverged rather than guessed at, and the driver aborts as it does for a failed iteration.
		CouplingSolution<problem_t> sol{};
		sol.Egas = cell.Egas0;
		sol.T_gas = TgasOf(cell, cell.Egas0);
		sol.T_d = sol.T_gas;
		sol.Erad = cell.Erad0 + cell.Src;
		sol.residual = br.flo;
		sol.nevals = br.nevals;
		sol.converged = false;
		ApplyEnergyFloors(cell, sol);
		sol.T_d = sol.T_gas;
		return sol;
	}

	quokka::math::eps_tolerance<double> tolerance(tol);
	int iters = max_root_iterations_;
	const auto bracket = quokka::math::brent_solve_bracket(G, br.lo, br.hi, br.flo, br.fhi, tolerance, iters);
	// The state is taken where the chord through the ends of Brent's final bracket crosses zero, not at its midpoint.
	// Brent stops anywhere inside the tolerance window, so two cells whose inputs differ by an ulp can stop at different
	// points of it; the chord crossing is within second order in the bracket width of the root, so the state depends on
	// the bracket only through terms of order tol^2, i.e. round-off. DTypeFront1D's mirror-symmetry check measures this.
	auto sol = GasCouplingState(cell, quokka::math::secant_point(bracket));
	sol.converged = tolerance(bracket.lo, bracket.hi);
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
	const auto exchange = GroupExchange(cell, ComputeCouplingCoefficients(cell, T_d));
	sol.Erad = GroupEnergies(cell, exchange);
	// the gas pays what the radiation gains: E_gas = gas0 - (c/chat) sum_g Delta_g, formed from the exchange rather than
	// from the cell's total energy, so that its round-off is that of the exchange and not of the radiation energy
	const double gas0 = cell.Egas0 - cscale * sum(cell.work);
	const double dust_gain = cscale * sum(exchange);
	sol.Egas = gas0 - dust_gain;
	sol.T_gas = TgasOf(cell, sol.Egas);
	const double T = sol.T_gas;
	sol.residual = dust_gain - cell.dtK * std::sqrt(T) * (T - T_d);
	return sol;
}

// The solve with dust: bracket H by marching from the gas temperature at the start of the step (Quokka's initial guess
// for the dust temperature, which where the dust balance has several roots selects the one connected to dust as warm as
// the gas), hand it to Brent, and take the state at the chord crossing of the final bracket. The convergence test is on the state
// (hydro3d.jl, docs/coupling-new-method-dust.md sections 4-5). H is not the test: its collision term multiplies the
// round-off in T - T_d by dt K sqrt(T), which reaches 1e8 of the cell's energy in a strongly coupled cell. The bracket
// width in T_d is not the test either: at small K the gas energy follows from conservation, and its error is the T_d
// error times (c/chat) sum_g E_g / E_gas, which is large where the radiation holds most of the energy (DTypeFront1D).
// So the test is the gas energy across the final bracket, relative to itself: |E_gas(lo) - E_gas(hi)| <= atol,
// atol = max(tol |E_gas|, 4 eps (|gas0| + (c/chat) sum_g |Delta_g|)), which is a relative tolerance on T_gas. It is
// floored by the round-off of E_gas = gas0 - (c/chat) sum_g Delta_g, which cannot resolve E_gas better than eps times
// the sizes of the terms it adds, and by the resolution of T_d: once the T_d bracket is a few ulp wide it cannot shrink further, the gas-energy width across it
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
	// emission alone already makes H positive (DTypeFront1D: T_floor = 10 K).
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

	// the round-off floor of the gas energy: it is gas0 minus (c/chat) times the exchange, so its round-off is that of those
	// two terms, not of the cell's total energy (which can be 1e5 times larger in a radiation-dominated cell)
	const double cscale = c_light_ / c_hat_;
	const double gas0 = cell.Egas0 - cscale * sum(cell.work);
	auto atol_floor_of = [&](CouplingSolution<problem_t> const &state) {
		double exchange_abs = 0.0;
		for (int g = 0; g < nGroups_; ++g) {
			exchange_abs += std::abs(state.Erad[g] - (cell.Erad0[g] + cell.Src[g] + cell.work[g]));
		}
		return 4 * std::numeric_limits<double>::epsilon() * (std::abs(gas0) + cscale * exchange_abs);
	};
	int iters = max_root_iterations_;
	// plain locals, not a structured binding: nvcc rejects a device lambda first-capturing a structured binding by reference
	auto bracket = quokka::math::brent_solve_bracket(H, br.lo, br.hi, br.flo, br.fhi, quokka::math::eps_tolerance<double>(tol), iters);
	double lo = bracket.lo;
	double hi = bracket.hi;

	// the spread of the gas energy across the final bracket; two evaluations, counted
	auto gas_width = [&](double const a, double const b) {
		if (a == b) {
			return 0.0;
		}
		nevals += 2;
		return std::abs(DustCouplingState(cell, a).Egas - DustCouplingState(cell, b).Egas);
	};
	constexpr double eps_mach = std::numeric_limits<double>::epsilon();
	// the state at the chord crossing of the final bracket, not its midpoint: see SolveGasCoupling
	auto mid = DustCouplingState(cell, quokka::math::secant_point(bracket));
	double atol = amrex::max(tol * std::abs(mid.Egas), atol_floor_of(mid));
	double width = gas_width(lo, hi);
	auto at_fp_limit = [&]() { return (hi - lo) <= 4 * eps_mach * std::abs(lo / 2 + hi / 2); };
	const int max_solves = 4; // the first solve and at most three repeats
	for (int nsolve = 1; (width > atol) && !at_fp_limit() && (nsolve < max_solves); ++nsolve) {
		// the T_d half-width that would bring the gas-energy width to atol, from the slope measured across the bracket;
		// the factor 4: 2 for the half- versus full-width of the bracket, 2 as margin for the slope changing across it
		const double slope = width / (hi - lo);
		const double eps = amrex::max(atol / (4.0 * slope) / std::abs(lo / 2 + hi / 2), 4 * eps_mach);
		iters = max_root_iterations_;
		bracket = quokka::math::brent_solve_bracket(H, lo, hi, H(lo), H(hi), quokka::math::eps_tolerance<double>(eps), iters);
		lo = bracket.lo;
		hi = bracket.hi;
		mid = DustCouplingState(cell, quokka::math::secant_point(bracket));
		atol = amrex::max(tol * std::abs(mid.Egas), atol_floor_of(mid));
		width = gas_width(lo, hi);
	}

	auto sol = mid;
	sol.converged = (width <= atol) || at_fp_limit();

	// The gas energy at the root can be written two ways that agree up to the residual H: from conservation,
	// gas0 - (c/chat) sum_g Delta_g, or from the gas equation, gas0 - dt K sqrt(T) (T - T_d). Their round-off differs.
	// The first cancels the group exchanges, which in a radiation-dominated cell are 1e5 times the gas energy; the
	// second carries only the collisional transfer, but that transfer is evaluated at the conservation-form temperature,
	// whose error it amplifies by dt K sqrt(T) / c_V (up to 1e8 when the dust is locked to the gas). Take the form with
	// the smaller estimated error. At K = 0 this leaves the gas energy exactly gas0, as the physics says: the gas
	// exchanges nothing with the dust, and mirror cells stay mirror images (DTypeFront1D's symmetry check).
	{
		double exchange_abs = 0.0;
		for (int g = 0; g < nGroups_; ++g) {
			exchange_abs += std::abs(sol.Erad[g] - (cell.Erad0[g] + cell.Src[g] + cell.work[g]));
		}
		const double T_c = sol.T_gas;
		const double c_v = ::quokka::EOS<problem_t>::ComputeEintTempDerivative(cell.rho, T_c, cell.massScalars);
		const double err_cons = eps_mach * (std::abs(gas0) + cscale * exchange_abs);
		const double err_gas = eps_mach * std::abs(gas0) + 1.5 * cell.dtK * std::sqrt(T_c) * (err_cons / c_v);
		if (err_gas < err_cons) {
			sol.Egas = gas0 - cell.dtK * std::sqrt(T_c) * (T_c - sol.T_d);
			sol.T_gas = TgasOf(cell, sol.Egas);
		}
	}
	sol.nevals = nevals;
	ApplyEnergyFloors(cell, sol);
	return sol;
}

#endif // RADIATION_COUPLING_HPP_
