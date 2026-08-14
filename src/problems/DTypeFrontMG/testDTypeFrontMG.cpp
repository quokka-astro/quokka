//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testDTypeFrontMG.cpp
/// \brief Defines a 3D spherical D-type ionization-front test problem with radiation pressure.
///
/// A point source at the origin emits ionizing and optical photons into a uniform, initially
/// neutral hydrogen medium (only the octant x,y,z > 0 is simulated, so the source luminosities
/// are divided by 8). With photochemistry, photoheating, and radiation pressure enabled, the gas
/// ionizes, heats, and drives a spherical D-type front. Depending on the ratio of the radiation
/// and gas-pressure characteristic radii (zeta = r_ch / r_S), the front is compared against
/// either the gas-pressure Spitzer solution or the radiation-pressure-augmented Krumholz &
/// Matzner (2009) solution.

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include "radiation/radiation_dust_system.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct DTypeFrontMG {
};

// reduced speed of light (same choice as DTypeFront1D)
constexpr double c_hat = C::c_light / 1000.0;

constexpr int IRBand = 0;
constexpr int OpticalBand = 1;
constexpr int IonizingBand = 2;

constexpr double nu_ion_lo = 3.29e15;
constexpr double nu_ion_hi = 1.50e16;
constexpr double nu_IR_lo = 3.0e12;
constexpr double nu_IR_optical = 3.29e14;

constexpr double E_photon = 0.5 * (nu_IR_lo + nu_IR_optical) * C::hplanck;
constexpr double epsilon_1 = 0.5 * (nu_ion_lo + nu_ion_hi) * C::hplanck;
constexpr double epsilon_2 = 0.5 * (nu_IR_optical + nu_ion_lo) * C::hplanck;
constexpr double Erad_floor_ = 1.0e-10 * E_photon;

template <> struct quokka::EOS_Traits<DTypeFrontMG> {
	static constexpr double mean_molecular_weight = C::m_p;
	static constexpr double gamma = 5. / 3.;
};

template <> struct ISM_Traits<DTypeFrontMG> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr bool enable_photoelectric_heating = false;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
};

template <> struct Physics_Traits<DTypeFrontMG> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = NumSpec;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// 3 radiation groups: group 0 = IR, group 1 = optical (both thermal), group 2 = ionizing (the
	// chemistry band). Chemistry bands must be the last groups; see radiation_system.hpp.
	static constexpr int nGroups = 3;
};

template <> struct RadSystem_Traits<DTypeFrontMG> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::hplanck;
	// Group frequency boundaries [Hz]: group 0 = IR [3e12, 3.29e14], group 1 = optical [3.29e14,
	// 3.29e15], group 2 = ionizing [3.29e15, 1.5e16]. The last group coincides with the chemistry
	// band (ChemBands below).
	static constexpr amrex::GpuArray<double, Physics_Traits<DTypeFrontMG>::nGroups + 1> radBoundaries{nu_IR_lo, nu_IR_optical, nu_ion_lo, nu_ion_hi};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr auto ChemBands() { return ChemBandsHeader_; }
};

AMREX_GPU_MANAGED double kappa_IR = 0.0;      // NOLINT
AMREX_GPU_MANAGED double kappa_optical = 0.0; // NOLINT

template <> struct SimulationData<DTypeFrontMG> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real primary_species_1{};
	amrex::Real primary_species_2{};
	amrex::Real primary_species_3{};
	amrex::Real Q_ion{};
	amrex::Real Q_optical{};
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> r_effective_vec_;
	amrex::Vector<amrex::Real> r_analytical_vec_;
	amrex::Vector<amrex::Real> r_spitzer_vec_;
	amrex::Vector<amrex::Real> r_KM09_vec_;
	amrex::Vector<amrex::Real> r_KM09_old_vec_;
	amrex::Real zeta_{};
	std::ofstream output_file_;
};

namespace
{

// Effective ionized radius, computed from the octant-integrated ionized volume (V = sum_cells (1
// - x_HI) * dx^3) assuming a spherical ionized region: r_eff = [3 * (8V) / (4 pi)]^(1/3), where
// the factor of 8 accounts for the full sphere (only one octant is simulated).
auto compute_effective_radius(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		const amrex::Real n_HI = state[box_no](i, j, k, HydroSystem<DTypeFrontMG>::scalar0_index + 1) / spmasses[1];
		const amrex::Real n_HII = state[box_no](i, j, k, HydroSystem<DTypeFrontMG>::scalar0_index + 2) / spmasses[2];
		const amrex::Real denom = n_HI + n_HII;
		if (denom <= 0.0_rt) {
			return 0.0_rt;
		}
		const amrex::Real x_HI = n_HI / denom;
		return cell_volume * (1.0_rt - x_HI);
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real total_ionized_volume = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Sum(total_ionized_volume, amrex::ParallelContext::CommunicatorSub());
	return std::cbrt((3.0_rt * 8.0_rt * total_ionized_volume) / (4.0_rt * M_PI));
}

auto lambda_rec(double T) -> double
{
	if (T < 100.0) {
		return 0.0;
	}
	return 6.1e-10 * 1.380649e-16 * T * std::pow(T, -0.89);
}

auto lambda_ion_ff(double T) -> double { return 1.4e-27 * std::sqrt(T) + 1.0e-19 * std::exp(-118348.0 / T); }

auto lambda_KI(double T) -> double { return 2.0e-26 * (1.0e7 * std::exp(-118400.0 / (T + 1.0e3)) + 1.4e-2 * std::sqrt(T) * std::exp(-92.0 / T)); }

auto net_energy_ionized(double T, double n_e) -> double
{
	const double alpha_B = 2.6e-13 * std::pow(T / 1.0e4, -0.7);
	const double epsilon = 6.4e-12;
	// alpha_B * n_e^2 = n_gamma
	const double photoheating = alpha_B * n_e * n_e * epsilon;
	const double recombination_cooling = n_e * n_e * lambda_rec(T);
	const double ion_ff_cooling = n_e * n_e * lambda_ion_ff(T);
	// Assume KI heating and cooling are negligible in the cavity since the neutral fraction is low.
	const double KI_heating = 0.0;
	const double KI_cooling = 0.0;
	return photoheating - recombination_cooling - ion_ff_cooling + KI_heating - KI_cooling;
}

auto net_energy_neutral(double T, double n_HI) -> double
{
	const double photoheating = 0.0;
	const double KI_heating = n_HI * 2e-26;
	const double KI_cooling = n_HI * n_HI * lambda_KI(T);
	const double recombination_cooling = 0.0;
	const double ion_ff_cooling = 0.0;
	return photoheating + KI_heating - recombination_cooling - KI_cooling - ion_ff_cooling;
}

auto compute_equilibrium_temperature_neutral(double n_HI) -> double
{
	double T_lo = 1;
	double T_hi = 1000;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(net_energy_neutral(T_lo, n_HI) > 0.0 && net_energy_neutral(T_hi, n_HI) < 0.0,
					 "compute_equilibrium_temperature_neutral: brackets do not straddle a root");
	int const max_iter = 10000;
	for (int iter = 0; iter < max_iter; ++iter) {
		const double T_mid = 0.5 * (T_lo + T_hi);
		if (net_energy_neutral(T_mid, n_HI) > 0.0) {
			T_lo = T_mid;
		} else {
			T_hi = T_mid;
		}
		if ((T_hi - T_lo) < 1e-2) {
			break;
		}
	}
	return 0.5 * (T_lo + T_hi);
}

auto compute_equilibrium_temperature_ionized(double n_e) -> double
{
	double T_lo = 1000.0;
	double T_hi = 1.0e5;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(net_energy_ionized(T_lo, n_e) > 0.0 && net_energy_ionized(T_hi, n_e) < 0.0,
					 "compute_equilibrium_temperature_ionized: brackets do not straddle a root");
	int const max_iter = 10000;
	for (int iter = 0; iter < max_iter; ++iter) {
		const double T_mid = 0.5 * (T_lo + T_hi);
		if (net_energy_ionized(T_mid, n_e) > 0.0) {
			T_lo = T_mid;
		} else {
			T_hi = T_mid;
		}
		if ((T_hi - T_lo) < 1.0) {
			break;
		}
	}
	return 0.5 * (T_lo + T_hi);
}

} // namespace

// Point source at the origin injects ionizing and optical luminosity into the single grid cell
// containing it. radEnergy is a luminosity volume density [erg s^-1 cm^-3]; dividing the
// per-octant luminosity by the cell volume gives the volumetric injection rate.
template <>
void RadSystem<DTypeFrontMG>::SetRadEnergySource(array_t &radEnergy, const amrex::Box &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real /*time*/)
{
	amrex::ParmParse const pp("stromgen");
	amrex::Real Q_ion = 1.0e49_rt;
	pp.query("Q_ion", Q_ion);
	amrex::Real Q_optical = 1.0e48_rt;
	pp.query("Q_optical", Q_optical);

	// only 1/8 of the source is in this octant
	const amrex::Real eps_optical = 0.5_rt * (nu_IR_optical + nu_ion_lo) * C::hplanck; // mean photon energy of the optical band [erg]
	const amrex::Real L_star_ion = 0.125_rt * Q_ion * RadSystem<DTypeFrontMG>::GetChemBandQuanta(0);
	const amrex::Real L_star_optical = 0.125_rt * Q_optical * eps_optical;
	const amrex::Real x0 = 0.0_rt;
	const amrex::Real y0 = 0.0_rt;
	const amrex::Real z0 = 0.0_rt;
	const amrex::Real volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	const amrex::Real inv_volume = 1.0 / volume;

	const int src_i = static_cast<int>(amrex::Math::floor((x0 - prob_lo[0]) / dx[0]));
	const int src_j = static_cast<int>(amrex::Math::floor((y0 - prob_lo[1]) / dx[1]));
	const int src_k = static_cast<int>(amrex::Math::floor((z0 - prob_lo[2]) / dx[2]));

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		radEnergy(i, j, k, IRBand) = 0.0_rt;
		if (i == src_i && j == src_j && k == src_k) {
			radEnergy(i, j, k, OpticalBand) = L_star_optical * inv_volume;
			radEnergy(i, j, k, IonizingBand) = L_star_ion * inv_volume;
		} else {
			radEnergy(i, j, k, OpticalBand) = 0.0_rt;
			radEnergy(i, j, k, IonizingBand) = 0.0_rt;
		}
	});
}

template <> void QuokkaSimulation<DTypeFrontMG>::preCalculateInitialConditions()
{
	// initialize microphysics routines
	init_extern_parameters();

	// parmparse species and temperature
	amrex::ParmParse const pp("stromgen");
	userData_.small_temp = 1e-2;
	userData_.small_dens = 1e-60;
	userData_.temperature = 1.0e4;
	userData_.primary_species_1 = 0.0e0_rt;
	userData_.primary_species_2 = 1.0e2_rt;
	userData_.primary_species_3 = 0.0e0_rt;
	userData_.Q_ion = 1.0e49_rt;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("primary_species_1", userData_.primary_species_1);
	pp.query("primary_species_2", userData_.primary_species_2);
	pp.query("primary_species_3", userData_.primary_species_3);
	pp.query("Q_ion", userData_.Q_ion);
	pp.query("Q_optical", userData_.Q_optical);
	pp.query("kappa_IR", kappa_IR);
	pp.query("kappa_optical", kappa_optical);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::string const filename = "dtype_front_radii.csv";
		userData_.output_file_.open(filename);
		userData_.output_file_ << "time,r_effective,r_analytical\n";
	}
}

template <>
// Hydrogen nucleus number density, n_H = n_HI + n_HII. Must be specialized: the base implementation
// returns rho / mean_molecular_weight, which for this problem (mean_molecular_weight = 1.0) is off by
// 1/m_p ~ 6e23.
AMREX_GPU_HOST_DEVICE auto RadSystem<DTypeFrontMG>::ComputeNumberDensityH(double /*rho*/, amrex::GpuArray<Real, nmscalars_> const &massScalars) -> double
{
	return (massScalars[1] + massScalars[2]) / C::m_p;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<DTypeFrontMG>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/,
											 const double /*rho*/, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int g = 0; g < nGroups_ + 1; ++g) {
		exponents_and_values[0][g] = 0.0;
	}
	exponents_and_values[1][IRBand] = kappa_IR;
	exponents_and_values[1][OpticalBand] = kappa_optical;
	exponents_and_values[1][IonizingBand] = 0.0_rt;
	return exponents_and_values;
}

template <> void QuokkaSimulation<DTypeFrontMG>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	burn_t state;
	std::array<Real, NumSpec> numdens = {-1.0};
	numdens[0] = userData_.primary_species_1;
	numdens[1] = userData_.primary_species_2;
	numdens[2] = userData_.primary_species_3;

	state.T = userData_.temperature;
	// find the density in g/cm^3
	Real rhotot = 0.0_rt;
	for (int n = 0; n < NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * spmasses[n]; // spmasses contains the masses of all species, defined in EOS
	}
	state.rho = rhotot;

	// call the EOS to set initial internal energy e
	eos(eos_input_rt, state);
	const auto Egas0 = state.e * rhotot;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<DTypeFrontMG>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<DTypeFrontMG>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad_floor_;
			state_cc(i, j, k, RadSystem<DTypeFrontMG>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFrontMG>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFrontMG>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
		}
		state_cc(i, j, k, RadSystem<DTypeFrontMG>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFrontMG>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<DTypeFrontMG>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFrontMG>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFrontMG>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFrontMG>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<DTypeFrontMG>::scalar0_index + nn) =
			    state.xn[nn] * spmasses[nn]; // scalar indices carry partial densities instead of number densities
		}
	});
}

template <> void QuokkaSimulation<DTypeFrontMG>::computeAfterTimestep()
{
	const int lev = 0;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const amrex::Real r_effective = compute_effective_radius(state_new_cc_[lev], dx);
	const amrex::Real t = tNew_[lev];
	userData_.r_effective_vec_.push_back(r_effective);
	userData_.t_vec_.push_back(t);

	// TODO(analytic solution): placeholder gas-pressure (Spitzer) / radiation-pressure (Krumholz &
	// Matzner 2009) reference solution, evaluated at the equilibrium ionized-region temperature.
	const amrex::Real n_e = userData_.primary_species_2;
	const amrex::Real T_eq = compute_equilibrium_temperature_ionized(n_e);
	const amrex::Real alpha_B = 2.6e-13 * std::pow(T_eq / 1.0e4, -0.7);
	const amrex::Real mu = 0.5;
	const amrex::Real c_i = std::sqrt(C::k_B * T_eq / (mu * C::m_p));
	const amrex::Real Q_ion = userData_.Q_ion;
	const amrex::Real Q_optical = userData_.Q_optical;
	const amrex::Real rho =
	    userData_.primary_species_1 * spmasses[0] + userData_.primary_species_2 * spmasses[1] + userData_.primary_species_3 * spmasses[2];
	const amrex::Real epsilon_1 = 0.5_rt * (nu_ion_lo + nu_ion_hi) * C::hplanck;
	const amrex::Real epsilon_2 = 0.5_rt * (nu_IR_optical + nu_ion_lo) * C::hplanck;

	const amrex::Real r_ch_old = Q_ion * epsilon_1 * epsilon_1 * alpha_B / (12.0_rt * M_PI * C::k_B * C::k_B * T_eq * T_eq * C::c_light * C::c_light);
	const amrex::Real t_ch_old = std::sqrt(4 * M_PI * rho * r_ch_old * r_ch_old * r_ch_old * r_ch_old * C::c_light / (3.0_rt * Q_ion * epsilon_1));

	const amrex::Real r_ch = std::pow(Q_ion * epsilon_1 + Q_optical * epsilon_2, 2.0_rt) / Q_ion *
				 (alpha_B / (12.0_rt * M_PI * C::c_light * C::c_light * C::k_B * C::k_B * T_eq * T_eq));
	const amrex::Real t_ch = std::sqrt(4 * M_PI * rho * r_ch * r_ch * r_ch * r_ch * C::c_light / (3.0_rt * (Q_ion * epsilon_1 + Q_optical * epsilon_2)));

	const amrex::Real r_s = std::pow((3.0_rt * userData_.Q_ion) / (4.0_rt * M_PI * alpha_B * n_e * n_e), 1.0_rt / 3.0_rt);
	const amrex::Real t_s = r_s / c_i;

	// Spitzer gas-pressure D-type solution
	const amrex::Real r_spitzer = r_s * std::pow(1.0_rt + 7.0_rt * t / (4.0_rt * t_s), 4.0_rt / 7.0_rt);

	// Krumholz & Matzner (2009) solution
	const amrex::Real tau = t / t_ch;
	const amrex::Real x_rad = std::pow(2.0_rt * tau * tau, 1.0_rt / 4.0_rt);
	const amrex::Real x_gas = std::pow(49.0_rt / (36.0_rt) * tau * tau, 2.0_rt / 7.0_rt);
	const amrex::Real x = std::pow(std::pow(x_rad, 7.0_rt / 2.0_rt) + std::pow(x_gas, 7.0_rt / 2.0_rt), 2.0_rt / 7.0_rt);
	const amrex::Real r_KM09 = r_ch * x;
	const amrex::Real zeta = r_ch / r_s;
	const amrex::Real r_analytical = (zeta > 1.0_rt) ? r_KM09 : r_spitzer;
	userData_.zeta_ = zeta;

	userData_.r_analytical_vec_.push_back(r_analytical);

	const amrex::Real tau_old = t / t_ch_old;
	const amrex::Real x_rad_old = std::pow(2.0_rt * tau_old * tau_old, 1.0_rt / 4.0_rt);
	const amrex::Real x_gas_old = std::pow(49.0_rt / (36.0_rt) * tau_old * tau_old, 2.0_rt / 7.0_rt);
	const amrex::Real x_old = std::pow(std::pow(x_rad_old, 7.0_rt / 2.0_rt) + std::pow(x_gas_old, 7.0_rt / 2.0_rt), 2.0_rt / 7.0_rt);
	const amrex::Real r_KM09_old = r_ch_old * x_old;

	// Store all three analytical solutions unconditionally (not gated by zeta) so they can all be
	// plotted for comparison; r_analytical above (regime-selected) remains the one used for the
	// pass/fail check.
	userData_.r_spitzer_vec_.push_back(r_spitzer);
	userData_.r_KM09_vec_.push_back(r_KM09);
	userData_.r_KM09_old_vec_.push_back(r_KM09_old);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_ << t << ',' << r_effective << ',' << r_analytical << '\n';
	}
}

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 0.3;
	const double dt_max = 1e99;

	// Problem initialization
	QuokkaSimulation<DTypeFrontMG> sim;

	// initialize
	sim.setInitialConditions();
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;
	sim.plotfileInterval_ = -1;

	int status = 0;

	sim.evolve();

	// Check 1: effective radius vs analytical radius at end of simulation.
	{
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = sim.geom[0].CellSizeArray();
		const amrex::Real cell_size = dx[0];
		const amrex::Real tol_cells = 3.0_rt * std::sqrt(3.0_rt) / 2.0_rt;
		const amrex::Real tol_percent = 10.0_rt;
		const bool gas_regime = sim.userData_.zeta_ < 1.0_rt;

		if (!sim.userData_.r_effective_vec_.empty()) {
			const amrex::Real r_analytical = sim.userData_.r_analytical_vec_.back();
			const amrex::Real r_effective = sim.userData_.r_effective_vec_.back();
			const amrex::Real delta_over_dx = (r_effective - r_analytical) / cell_size;
			const amrex::Real percent_diff = std::abs(r_effective - r_analytical) / r_analytical * 100.0_rt;

			const bool failed = gas_regime ? ((delta_over_dx < -tol_cells) || (delta_over_dx > tol_cells)) : (percent_diff > tol_percent);

			if (failed) {
				amrex::Print() << "Test FAILED: radius check at end of simulation.\n";
				amrex::Print() << "Analytical radius: " << r_analytical << '\n';
				amrex::Print() << "Effective radius: " << r_effective << '\n';
				if (gas_regime) {
					amrex::Print() << "(r_effective - r_analytical) / dx = " << delta_over_dx << '\n';
					amrex::Print() << "Tolerance: " << tol_cells << " cell sizes\n";
				} else {
					amrex::Print() << "Difference: " << percent_diff << " percent\n";
					amrex::Print() << "Tolerance: " << tol_percent << " percent\n";
				}
				status = 1;
			} else if (gas_regime) {
				amrex::Print() << "Test passed: D-type front effective radius matches analytical radius within " << tol_cells
					       << " cell sizes at end of simulation.\n";
			} else {
				amrex::Print() << "Test passed: D-type front effective radius matches analytical radius within " << tol_percent
					       << " percent at end of simulation.\n";
			}
		}
	}

	// Check 2: temperature in cavity and neutral region at end of simulation
	{
		// primary_species_2 is the initial n_HI (species index 1), which equals n_e in the fully ionized cavity
		const double ne_eq = sim.userData_.primary_species_2;
		const double n_HI_init = sim.userData_.primary_species_2; // in neutral region all hydrogen remains as HI
		const double T_ion_eq = compute_equilibrium_temperature_ionized(ne_eq);
		const double T_neu_eq = compute_equilibrium_temperature_neutral(n_HI_init);

		amrex::MultiFab const &state_mf = sim.state_new_cc_[0];

		// Collect temperatures per region: cavity (1% < x_HII < 99%), neutral (x_HI > 99.99%)
		std::vector<double> cavity_temps;
		std::vector<double> neutral_temps;

		for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
			const amrex::Box &box = mfi.validbox();

			// In GPU builds, MultiFab data resides on device; copy to pinned host memory before CPU access.
			amrex::FArrayBox host_fab(box, state_mf.nComp(), amrex::The_Pinned_Arena());
			static_cast<void>(state_mf[mfi].template copyToMem<amrex::RunOn::Device>(box, 0, state_mf.nComp(), host_fab.dataPtr()));
			amrex::Gpu::synchronize();

			const auto state = host_fab.const_array();

			amrex::LoopOnCpu(box, [&](int i, int j, int k) noexcept {
				const amrex::Real rho = state(i, j, k, HydroSystem<DTypeFrontMG>::density_index);
				const amrex::Real Eint = state(i, j, k, RadSystem<DTypeFrontMG>::gasInternalEnergy_index);
				const amrex::Real n_HI_cell = state(i, j, k, HydroSystem<DTypeFrontMG>::scalar0_index + 1) / spmasses[1];
				const amrex::Real n_HII_cell = state(i, j, k, HydroSystem<DTypeFrontMG>::scalar0_index + 2) / spmasses[2];
				const amrex::Real denom = n_HI_cell + n_HII_cell;
				if (denom <= 0.0_rt) {
					return;
				}
				const amrex::Real x_HII = n_HII_cell / denom;
				const amrex::Real x_HI = n_HI_cell / denom;

				burn_t bstate;
				for (int nn = 0; nn < NumSpec; ++nn) {
					bstate.xn[nn] = state(i, j, k, HydroSystem<DTypeFrontMG>::scalar0_index + nn) / spmasses[nn];
				}
				bstate.rho = rho;
				bstate.e = Eint / rho;
				bstate.T = 1.0e4; // initial guess
				eos(eos_input_re, bstate);
				const double T_cell = bstate.T;

				if (x_HII > 0.01_rt && x_HII < 0.99_rt) {
					cavity_temps.push_back(T_cell);
				}
				if (x_HI > 0.9999_rt) {
					neutral_temps.push_back(T_cell);
				}
			});
		}

		// Gather all temperatures to IOProcessor, compute median, check within 5%
		auto compute_median_and_check = [&](std::vector<double> &local_temps, double T_analytical, const char *region_name) {
			const int num_local = static_cast<int>(local_temps.size());
			auto num_local_vec = amrex::ParallelDescriptor::Gather(num_local, amrex::ParallelDescriptor::IOProcessorNumber());

			amrex::Vector<int> recvcnt;
			amrex::Vector<int> disp;
			std::vector<double> all_temps;
			if (amrex::ParallelDescriptor::IOProcessor()) {
				recvcnt.resize(num_local_vec.size());
				disp.resize(num_local_vec.size());
				int ntot = 0;
				disp[0] = 0;
				for (int r = 0, n = static_cast<int>(num_local_vec.size()); r < n; ++r) {
					recvcnt[r] = num_local_vec[r];
					ntot += num_local_vec[r];
					if (r + 1 < n) {
						disp[r + 1] = disp[r] + num_local_vec[r];
					}
				}
				all_temps.resize(ntot);
			} else {
				recvcnt.resize(1);
				disp.resize(1);
				all_temps.resize(1);
			}

			static double static_val = 0.0;
			const double *send_ptr = local_temps.empty() ? &static_val : local_temps.data();
			double *recv_ptr = all_temps.empty() ? &static_val : all_temps.data();
			amrex::ParallelDescriptor::Gatherv(send_ptr, num_local, recv_ptr, recvcnt, disp, amrex::ParallelDescriptor::IOProcessorNumber());

			if (amrex::ParallelDescriptor::IOProcessor()) {
				const int ntot = static_cast<int>(all_temps.size());
				if (ntot == 0) {
					amrex::Print() << "Warning: no " << region_name << " cells found.\n";
					return;
				}
				std::sort(all_temps.begin(), all_temps.end());
				const double T_median = (ntot % 2 == 0) ? 0.5 * (all_temps[ntot / 2 - 1] + all_temps[ntot / 2]) : all_temps[ntot / 2];
				const double rel_err = std::abs(T_median - T_analytical) / T_analytical;
				if (rel_err > 0.05) {
					amrex::Print()
					    << "Test FAILED: " << region_name << " median temperature " << T_median << " K differs from analytical equilibrium "
					    << T_analytical << " K by " << 100.0 * rel_err << "% (tolerance: 5%).\n";
					status = 1;
				} else {
					amrex::Print() << "Test passed: " << region_name << " median temperature " << T_median
						       << " K is within 5% of analytical equilibrium " << T_analytical << " K (" << ntot << " cells).\n";
				}
			}
		};

		compute_median_and_check(cavity_temps, T_ion_eq, "cavity");
		compute_median_and_check(neutral_temps, T_neu_eq, "neutral");
	}

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr amrex::Real seconds_per_Myr = 3.15576e13;
		constexpr amrex::Real cm_per_pc = 3.085677581491367e18;

		std::vector<amrex::Real> t_Myr(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> r_effective_pc(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> r_spitzer_pc(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> r_KM09_pc(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> r_KM09_old_pc(sim.userData_.t_vec_.size());
		for (int i = 0; i < static_cast<int>(sim.userData_.t_vec_.size()); ++i) {
			t_Myr[i] = sim.userData_.t_vec_[i] / seconds_per_Myr;
			r_effective_pc[i] = sim.userData_.r_effective_vec_[i] / cm_per_pc;
			r_spitzer_pc[i] = sim.userData_.r_spitzer_vec_[i] / cm_per_pc;
			r_KM09_pc[i] = sim.userData_.r_KM09_vec_[i] / cm_per_pc;
			r_KM09_old_pc[i] = sim.userData_.r_KM09_old_vec_[i] / cm_per_pc;
		}
		// Plot radii vs time
		matplotlibcpp::clf();
		std::map<std::string, std::string> numerical_args;
		numerical_args["label"] = "numerical";
		numerical_args["color"] = "C0";
		std::map<std::string, std::string> spitzer_args;
		spitzer_args["label"] = "Spitzer";
		spitzer_args["color"] = "k";
		spitzer_args["linestyle"] = "--";
		std::map<std::string, std::string> km09_args;
		km09_args["label"] = "Krumholz & Matzner (2009)";
		km09_args["color"] = "C3";
		km09_args["linestyle"] = "--";
		std::map<std::string, std::string> km09_old_args;
		km09_old_args["label"] = "Krumholz & Matzner (2009), old";
		km09_old_args["color"] = "C2";
		km09_old_args["linestyle"] = ":";

		matplotlibcpp::plot(t_Myr, r_effective_pc, numerical_args);
		matplotlibcpp::plot(t_Myr, r_spitzer_pc, spitzer_args);
		matplotlibcpp::plot(t_Myr, r_KM09_pc, km09_args);
		matplotlibcpp::plot(t_Myr, r_KM09_old_pc, km09_old_args);
		matplotlibcpp::xlabel("time (Myr)");
		matplotlibcpp::ylabel("radius (pc)");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dtype_front_radii.pdf");

		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = sim.geom[0].CellSizeArray();
		const amrex::Real cell_size = dx[0];
		std::vector<amrex::Real> delta_over_dx_vec(sim.userData_.t_vec_.size());
		for (int i = 0; i < static_cast<int>(sim.userData_.t_vec_.size()); ++i) {
			delta_over_dx_vec[i] = (sim.userData_.r_effective_vec_[i] - sim.userData_.r_analytical_vec_[i]) / cell_size;
		}

		matplotlibcpp::clf();
		std::map<std::string, std::string> diff_args;
		diff_args["label"] = "(r_effective - r_analytical) / dx";
		diff_args["color"] = "C1";
		matplotlibcpp::plot(t_Myr, delta_over_dx_vec, diff_args);
		matplotlibcpp::xlabel("time (Myr)");
		matplotlibcpp::ylabel("delta r / dx");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dtype_front_radii_difference.pdf");
	}
#endif

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return status;
}
