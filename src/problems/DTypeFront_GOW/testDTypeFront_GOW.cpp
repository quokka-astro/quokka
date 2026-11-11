// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testDTypeFront_GOW.cpp
/// \brief A source of far-UV photons in a uniform atomic cloud, with the jaff-generated GOW network (Gong,
/// Ostriker & Wolfire 2017; 18 species, 2 FUV bands, 23 equations). A timing problem for a large network: it has no
/// pass/fail check, and prints the state of the gas at the end.
///
/// The setup follows DTypeFront3D: a source at the centre of the box, reflecting boundaries, and a uniform cloud.
/// The source emits in the bands 6-11.2 eV and 11.2-13.6 eV, which dissociate H2 and CO and ionize C.
///
/// The line shielding of H2, CO and C uses local column densities, as in XRByte's turbulent chemistry box (branch
/// turb-chem-box of XRByte/quokka): N_X = n_X L, with L the Jeans length at a temperature capped at 40 K
/// (Safranek-Shrader et al. 2017), computed from the cell's state before each burn and held constant during it.

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_Reduce.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "microphysics/burn_state.hpp"
#include "microphysics/microphysics_traits.hpp"
#include "networks/GOW/network.hpp"
#include "physics_info.hpp"
#include "radiation/photochemistry.hpp"
#include "radiation/radiation_system.hpp"

struct DTypeFront_GOW {};

constexpr double c_hat = C::c_light / 1000.0;

/// the temperature cap [K] of the Jeans length that sets the shielding columns
constexpr double T_shield_cap = 40.0;

template <> struct quokka::EOS_Traits<DTypeFront_GOW> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Microphysics_Traits<DTypeFront_GOW> : DefaultMicrophysicsTraits {
	using Network = quokka::networks::GOW::Network;
	// Rodas5P, as in XRByte's turbulent chemistry box; ROS2S, Rodas3P and Rodas4P are the other tableaus. The CMake
	// variable DTypeFront_GOW_TABLEAU selects one for the timing experiments.
#ifndef QUOKKA_GOW_TABLEAU
#define QUOKKA_GOW_TABLEAU Rodas5P
#endif
	using Integrator = quokka::microphysics::Rosenbrock<quokka::microphysics::QUOKKA_GOW_TABLEAU>;
	static constexpr bool photochemistry = true;
	// No finite-difference fallback: the analytic Jacobian was NaN below ~13 K (the H2-e- cooling fit overflowed), which
	// the network's cooling_H2 now guards; a non-finite Jacobian fails the burn and (DTypeFront_GOW_BURN_STATS) prints it.
	static constexpr bool numerical_jacobian_fallback = false;
	// Partial pivoting in the LU factorization of the integrator matrix (DTypeFront_GOW_PIVOT=OFF turns it off, for the
	// timing experiments: the pivot-row swaps were 7% of the burn kernel on the MI250X).
#ifdef QUOKKA_GOW_NO_PIVOT
	static constexpr bool linalg_do_pivoting = false;
#endif
	// The network's shared evaluation of the right-hand side and the Jacobian (DTypeFront_GOW_SHARED_RHS_JAC=OFF uses
	// actual_rhs and actual_jac instead, for the timing experiments).
#ifdef QUOKKA_GOW_NO_SHARED_RHS_JAC
	static constexpr bool shared_rhs_and_jac = false;
#endif

	/// Local shielding columns N_X = n_X L, with L the Jeans length of the cell's gas at min(T, T_shield_cap), and the
	/// H-nucleus column N_H = n_H L, which sets the cloud depth of the CO cooling.
	template <typename BurnT> AMREX_GPU_HOST_DEVICE static void ColumnDensities(BurnT &state)
	{
		using Net = quokka::networks::GOW::Network;
		amrex::Real n_tot = 0.0;
		for (int n = 0; n < Net::NumSpec; ++n) {
			n_tot += state.xn[n];
		}
		const amrex::Real n_H_nuclei = state.xn[Net::h] + state.xn[Net::hj] + 2.0 * (state.xn[Net::h2] + state.xn[Net::h2j]) +
					       3.0 * state.xn[Net::h3j] + state.xn[Net::hcoj] + state.xn[Net::chx] + state.xn[Net::ohx];
		const amrex::Real T_shield = (state.T < T_shield_cap) ? state.T : T_shield_cap;
		// the isothermal sound speed of the mixture, squared
		const amrex::Real cs2 = C::k_B * T_shield * n_tot / state.rho;
		const amrex::Real L = std::sqrt(M_PI * cs2 / (C::Gconst * state.rho));
		state.ncol[Net::ncol_H2] = state.xn[Net::h2] * L;
		state.ncol[Net::ncol_CO] = state.xn[Net::co] * L;
		state.ncol[Net::ncol_C] = state.xn[Net::c] * L;
		state.ncol[Net::ncol_H] = n_H_nuclei * L;
	}
};

using Network = Microphysics_Traits<DTypeFront_GOW>::Network;

template <> struct Physics_Traits<DTypeFront_GOW> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int numMassScalars = Network::NumSpec;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr int nGroups = Network::NumChemBands;
};

template <> struct RadSystem_Traits<DTypeFront_GOW> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = C::a_rad * 1.0e-8;
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr amrex::GpuArray<double, Physics_Traits<DTypeFront_GOW>::nGroups + 1> radBoundaries{
	    Network::ChemBands().arr[0], Network::ChemBands().arr[1], Network::ChemBands().arr[2]};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr auto ChemBandsPowerLawIndex() { return Network::ChemBandsPowerLawIndex; }
	static constexpr auto ChemBands() { return Network::ChemBands(); }
};

template <> struct SimulationData<DTypeFront_GOW> {
	amrex::Real temperature{};			     ///< initial gas temperature [K]
	amrex::Real n_H{};				     ///< H-nucleus number density [cm^-3]
	std::array<amrex::Real, Network::NumSpec> numdens{}; ///< initial number densities [cm^-3]
};

namespace
{

/// The initial abundances per H nucleus, as in XRByte's turbulent chemistry box: atomic, with carbon as C+, silicon
/// as Si, and the gow17 element totals.
auto initial_abundances() -> std::array<amrex::Real, Network::NumSpec>
{
	constexpr amrex::Real xC_tot = 1.6e-4;
	constexpr amrex::Real xO_tot = 3.2e-4;
	constexpr amrex::Real xSi_tot = 1.7e-6;
	constexpr amrex::Real xHe_tot = 0.1;
	constexpr amrex::Real seed = 1.0e-40;

	std::array<amrex::Real, Network::NumSpec> x{};
	x.fill(seed);
	x[Network::hj] = 1.0e-4;
	x[Network::h2] = 1.0e-10;
	x[Network::hej] = 1.450654e-08;
	x[Network::co] = 1.0e-7;
	x[Network::h3j] = 2.681411e-07;
	// the element totals fix the carriers of H, He, C, O and Si, and charge neutrality the electrons
	x[Network::h] = 1.0 - (x[Network::hj] + 2.0 * x[Network::h2] + 2.0 * x[Network::h2j] + x[Network::hcoj] + x[Network::chx] + x[Network::ohx] +
			       3.0 * x[Network::h3j]);
	x[Network::he] = xHe_tot - x[Network::hej];
	x[Network::cj] = xC_tot - (x[Network::c] + x[Network::co] + x[Network::hcoj] + x[Network::chx]);
	x[Network::o] = xO_tot - (x[Network::co] + x[Network::hcoj] + x[Network::ohx] + x[Network::oj]);
	x[Network::si] = xSi_tot - x[Network::sij];
	x[Network::ek] =
	    x[Network::hj] + x[Network::h2j] + x[Network::hej] + x[Network::cj] + x[Network::hcoj] + x[Network::sij] + x[Network::h3j] + x[Network::oj];
	return x;
}

AMREX_GPU_HOST_DEVICE auto wendland_c2(amrex::Real r) -> amrex::Real
{
	if (r > 1.0) {
		return 0.0;
	}
	return (21. / (2. * M_PI)) * std::pow((1.0 - r), 4) * (4.0 * r + 1.0);
}

void print_microphysics_integrator()
{
	amrex::Print() << "DTypeFront_GOW microphysics integrator: Rosenbrock (Rosenbrock tableau "
		       << Microphysics_Traits<DTypeFront_GOW>::Integrator::tableau::name << ")\n";
}

} // namespace

template <>
void RadSystem<DTypeFront_GOW>::AddRadSource(array_t &radEnergy, array_t &reducedFlux, const amrex::Box &indexRange,
					     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	// photon emission rate [s^-1] of the source in each band
	amrex::ParmParse const pp("gow");
	amrex::GpuArray<amrex::Real, nGroups_> luminosity{};
	for (int g = 0; g < nGroups_; ++g) {
		amrex::Real Q = 0.0_rt;
		pp.get(("Q_band" + std::to_string(g)).c_str(), Q);
		luminosity[g] = Q * RadSystem<DTypeFront_GOW>::GetChemBandQuanta(g);
	}

	constexpr int N = 2;
	constexpr amrex::Real inv_N = 1.0 / static_cast<amrex::Real>(N);
	constexpr auto cutoff_r2 = static_cast<amrex::Real>(N * N);

	const amrex::Real x0 = 0.5_rt * (prob_lo[0] + prob_hi[0]);
	const amrex::Real y0 = 0.5_rt * (prob_lo[1] + prob_hi[1]);
	const amrex::Real z0 = 0.5_rt * (prob_lo[2] + prob_hi[2]);
	const amrex::Real inv_volume = 1.0 / (dx[0] * dx[1] * dx[2]);

	const int src_i = static_cast<int>(amrex::Math::floor((x0 - prob_lo[0]) / dx[0]));
	const int src_j = static_cast<int>(amrex::Math::floor((y0 - prob_lo[1]) / dx[1]));
	const int src_k = static_cast<int>(amrex::Math::floor((z0 - prob_lo[2]) / dx[2]));
	const amrex::Real frac_x = (x0 - prob_lo[0]) / dx[0] - static_cast<amrex::Real>(src_i);
	const amrex::Real frac_y = (y0 - prob_lo[1]) / dx[1] - static_cast<amrex::Real>(src_j);
	const amrex::Real frac_z = (z0 - prob_lo[2]) / dx[2] - static_cast<amrex::Real>(src_k);

	constexpr int stencil_width = 2 * N + 1;
	amrex::Real norm_sum = 0.0_rt;
	for (int kk = 0; kk < stencil_width; ++kk) {
		const amrex::Real dz = static_cast<amrex::Real>(kk - N) + 0.5 - frac_z;
		for (int jj = 0; jj < stencil_width; ++jj) {
			const amrex::Real dy = static_cast<amrex::Real>(jj - N) + 0.5 - frac_y;
			for (int ii = 0; ii < stencil_width; ++ii) {
				const amrex::Real di = static_cast<amrex::Real>(ii - N) + 0.5 - frac_x;
				const amrex::Real r2 = di * di + dy * dy + dz * dz;
				if (r2 <= cutoff_r2) {
					norm_sum += wendland_c2(std::sqrt(r2) * inv_N);
				}
			}
		}
	}
	const amrex::Real inv_norm = 1.0_rt / norm_sum;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const amrex::Real di = static_cast<amrex::Real>(i - src_i) + 0.5 - frac_x;
		const amrex::Real dj = static_cast<amrex::Real>(j - src_j) + 0.5 - frac_y;
		const amrex::Real dk = static_cast<amrex::Real>(k - src_k) + 0.5 - frac_z;
		const amrex::Real r2 = di * di + dj * dj + dk * dk;
		const amrex::Real weight = (r2 <= cutoff_r2) ? wendland_c2(std::sqrt(r2) * inv_N) * inv_norm * inv_volume : 0.0_rt;
		for (int g = 0; g < nGroups_; ++g) {
			radEnergy(i, j, k, g) = luminosity[g] * weight;
			reducedFlux(i, j, k, 3 * g + 0) = 0.0_rt;
			reducedFlux(i, j, k, 3 * g + 1) = 0.0_rt;
			reducedFlux(i, j, k, 3 * g + 2) = 0.0_rt;
		}
	});
}

// The bands couple to the gas only through photochemistry.
template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<DTypeFront_GOW>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
								const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	return amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>{};
}

template <> void QuokkaSimulation<DTypeFront_GOW>::preCalculateInitialConditions()
{
	amrex::ParmParse const pp("gow");
	// XRByte's turbulent chemistry box
	userData_.temperature = 100.0;
	userData_.n_H = 30.0;
	pp.query("temperature", userData_.temperature);
	pp.query("n_H", userData_.n_H);

	const auto x = initial_abundances();
	for (int n = 0; n < Network::NumSpec; ++n) {
		userData_.numdens[n] = x[n] * userData_.n_H;
	}
}

template <> void QuokkaSimulation<DTypeFront_GOW>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	quokka::microphysics::BurnState<DTypeFront_GOW> state;
	state.T = userData_.temperature;
	Real rhotot = 0.0_rt;
	for (int n = 0; n < Network::NumSpec; ++n) {
		state.xn[n] = userData_.numdens[n];
		rhotot += state.xn[n] * Network::SpeciesMass()[n];
	}
	state.rho = rhotot;

	// call the EOS to set the initial internal energy e
	quokka::EOS<DTypeFront_GOW>::Solve(quokka::EOS<DTypeFront_GOW>::Input::RhoT, state);
	const auto Egas0 = state.e * rhotot;
	const auto xn = state.xn;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<DTypeFront_GOW>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<DTypeFront_GOW>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = 1.e-99_rt;
			state_cc(i, j, k, RadSystem<DTypeFront_GOW>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFront_GOW>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFront_GOW>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
		}
		state_cc(i, j, k, RadSystem<DTypeFront_GOW>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFront_GOW>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<DTypeFront_GOW>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFront_GOW>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFront_GOW>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFront_GOW>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < Network::NumSpec; ++nn) {
			// the scalars carry partial densities
			state_cc(i, j, k, HydroSystem<DTypeFront_GOW>::scalar0_index + nn) = xn[nn] * Network::SpeciesMass()[nn];
		}
	});
}

namespace
{
/// Prints the range of the gas temperature and the domain-wide molecular fractions of H and C.
void print_gas_state(amrex::MultiFab const &state_mf)
{
	amrex::ReduceOps<amrex::ReduceOpMin, amrex::ReduceOpMax, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
		auto const &s = state[box_no];
		std::array<amrex::Real, Network::NumSpec> n{};
		for (int nn = 0; nn < Network::NumSpec; ++nn) {
			n[nn] = s(i, j, k, HydroSystem<DTypeFront_GOW>::scalar0_index + nn) / Network::SpeciesMass()[nn];
		}
		const amrex::Real rho = s(i, j, k, HydroSystem<DTypeFront_GOW>::density_index);
		const amrex::Real Eint = s(i, j, k, HydroSystem<DTypeFront_GOW>::internalEnergy_index);
		const auto massScalars = RadSystem<DTypeFront_GOW>::ComputeMassScalars(s, i, j, k);
		const amrex::Real T = quokka::EOS<DTypeFront_GOW>::ComputeTgasFromEint(rho, Eint, massScalars);
		const amrex::Real n_H_nuclei = n[Network::h] + n[Network::hj] + 2.0 * (n[Network::h2] + n[Network::h2j]) + 3.0 * n[Network::h3j] +
					       n[Network::hcoj] + n[Network::chx] + n[Network::ohx];
		const amrex::Real n_C_nuclei = n[Network::c] + n[Network::cj] + n[Network::co] + n[Network::hcoj] + n[Network::chx];
		return amrex::makeTuple(T, T, n_H_nuclei, 2.0 * n[Network::h2], n_C_nuclei, n[Network::co]);
	});

	auto const values = reduce_data.value(reduce_op);
	amrex::Real T_min = amrex::get<0>(values);
	amrex::Real T_max = amrex::get<1>(values);
	std::array<amrex::Real, 4> sums{amrex::get<2>(values), amrex::get<3>(values), amrex::get<4>(values), amrex::get<5>(values)};
	amrex::ParallelAllReduce::Min(T_min, amrex::ParallelContext::CommunicatorSub());
	amrex::ParallelAllReduce::Max(T_max, amrex::ParallelContext::CommunicatorSub());
	amrex::ParallelAllReduce::Sum(sums.data(), static_cast<int>(sums.size()), amrex::ParallelContext::CommunicatorSub());

	amrex::Print() << "Gas temperature: " << T_min << " K to " << T_max << " K\n";
	amrex::Print() << "Fraction of H in H2: " << sums[1] / sums[0] << "\n";
	amrex::Print() << "Fraction of C in CO: " << sums[3] / sums[2] << "\n";
}

/// GOW experiment diagnostic: with gow.probe = 1, evaluate the network's right-hand side and analytic Jacobian on the
/// host at the state gow.probe_T, probe_rho, probe_rn0, probe_rn1 and probe_xn (the NumSpec number densities), and
/// print the non-finite entries. Nothing is simulated.
auto probe_jacobian() -> int
{
	using BurnT = quokka::microphysics::BurnState<DTypeFront_GOW>;
	using System = quokka::microphysics::ODESystem<DTypeFront_GOW, BurnT>;

	Network::ReadParameters();
	quokka::eos_floors::ReadParameters(); // the EOS clamps rho and T to its floors

	const amrex::ParmParse pp("gow");
	BurnT state;
	pp.get("probe_T", state.T);
	pp.get("probe_rho", state.rho);
	std::vector<double> xn;
	pp.getarr("probe_xn", xn);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(xn.size() == Network::NumSpec, "gow.probe_xn needs NumSpec entries");
	for (int n = 0; n < Network::NumSpec; ++n) {
		state.xn[n] = xn[n];
	}
	double rn0 = 0.0;
	double rn1 = 0.0;
	pp.get("probe_rn0", rn0);
	pp.get("probe_rn1", rn1);
	state.rn[0] = rn0;
	state.rn[System::vars_per_group] = rn1;
	state.rn[1] = 1.0;
	state.rn[System::vars_per_group + 1] = 1.0;
	state.c_hat = c_hat;
	quokka::EOS<DTypeFront_GOW>::Solve(quokka::EOS<DTypeFront_GOW>::Input::RhoT, state);
	Microphysics_Traits<DTypeFront_GOW>::ColumnDensities(state);

	amrex::Print() << "probe: T " << state.T << " rho " << state.rho << " e " << state.e << " c_hat " << state.c_hat << '\n';
	for (int n = 0; n < Network::NumColumns; ++n) {
		amrex::Print() << "probe: ncol[" << n << "] = " << state.ncol[n] << '\n';
	}

	amrex::GpuArray<amrex::Real, System::neqs> ydot{};
	Network::actual_rhs(state, ydot);
	for (int n = 0; n < System::neqs; ++n) {
		if (!std::isfinite(ydot[n])) {
			amrex::Print() << "probe: rhs[" << n << "] = " << ydot[n] << '\n';
		}
	}

	amrex::Array2D<amrex::Real, 0, System::neqs - 1, 0, System::neqs - 1> jac{};
	Network::actual_jac(state, jac);
	int n_bad = 0;
	for (int j = 0; j < System::neqs; ++j) {
		for (int i = 0; i < System::neqs; ++i) {
			if (!std::isfinite(jac(i, j))) {
				amrex::Print() << "probe: jac(" << i << "," << j << ") = " << jac(i, j) << '\n';
				++n_bad;
			}
		}
	}
	amrex::Print() << "probe: " << n_bad << " non-finite Jacobian entries\n";
	return 0;
}
} // namespace

auto problem_main() -> int
{
	const double CFL_number = 0.3;
	const double dt_max = 1e99;

	{
		const amrex::ParmParse pp("gow");
		int probe = 0;
		pp.query("probe", probe);
		if (probe != 0) {
			return probe_jacobian();
		}
	}

	QuokkaSimulation<DTypeFront_GOW> sim;
	print_microphysics_integrator();

	sim.setInitialConditions();
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;

	sim.evolve();

	print_gas_state(sim.state_new_cc_[0]);

	amrex::Print() << "Finished." << '\n';
	return 0;
}
