//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testDTypeFrontMG.cpp
/// \brief Defines a test problem for the static Stromgren sphere with no temperature dependence.
///

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <math/quadrature.hpp>
#include <string>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct DTypeFrontMG {
};

constexpr double c_hat = C::c_light / 1000.0;

constexpr int IRBand = 0;
constexpr int OpticalBand = 1;
constexpr int IonizingBand = 2;

constexpr double eV_per_Hz = C::hplanck / C::ev2erg;
constexpr double nu_ion_lo = 3.29e15;
constexpr double nu_ion_hi = 1.50e16;
constexpr double nu_IR_lo = 3.0e12;
constexpr double nu_IR_optical = 3.29e14;

template <> struct quokka::EOS_Traits<DTypeFrontMG> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DTypeFrontMG> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = NumSpec;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = NumThermalBands + NumChemBands;
};

template <> struct RadSystem_Traits<DTypeFrontMG> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = C::a_rad * 1.0e-8;
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr int NumThermalBands = ::NumThermalBands;
	static constexpr amrex::GpuArray<double, NumThermalBands + 1> thermalRadBoundaries{nu_IR_lo * eV_per_Hz, nu_IR_optical * eV_per_Hz,
											   nu_ion_lo * eV_per_Hz};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr int NumChemBands = ::NumChemBands;
	static constexpr amrex::GpuArray<double, NumChemBands + 1> chemRadBoundaries{nu_ion_lo * eV_per_Hz, nu_ion_hi * eV_per_Hz};
};

template <> struct ISM_Traits<DTypeFrontMG> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = false;
	static constexpr bool enable_photoelectric_heating = false;
	static constexpr double gas_dust_coupling_threshold = 1.0e-2;
};

template <> struct SimulationData<DTypeFrontMG> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real primary_species_1{};
	amrex::Real primary_species_2{};
	amrex::Real primary_species_3{};
	amrex::Real Q{};
	int recombination_switch{};
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> r_effective_vec_;
	amrex::Vector<amrex::Real> r_analytical_vec_;
	amrex::Real r_analytical_last_t{};
	amrex::Real r_analytical_last_R{};
	std::ofstream output_file_;
};

namespace
{

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

AMREX_GPU_HOST_DEVICE auto wendland_c2(amrex::Real r) -> amrex::Real
{
	if (r > 1.0) {
		return 0.0;
	}
	return (21. / (2. * M_PI)) * std::pow((1.0 - r), 4) * (4.0 * r + 1.0);
}

template <>
void RadSystem<DTypeFrontMG>::SetRadEnergySource(array_t &radEnergy, const amrex::Box &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real /*time*/)
{
	amrex::ParmParse const pp("stromgen");
	amrex::Real Q = 1.0e49_rt;
	pp.query("Q", Q);
	amrex::Real optical_to_ionizing_fraction = 0.1_rt;
	pp.query("optical_to_ionizing_fraction", optical_to_ionizing_fraction);

	constexpr int N = 2;
	constexpr amrex::Real inv_N = 1.0 / static_cast<amrex::Real>(N);
	constexpr auto cutoff_r2 = static_cast<amrex::Real>(N * N);

	const amrex::Real L_star_ion = Q * RadSystem<DTypeFrontMG>::GetChemBandQuanta(0);
	const amrex::Real L_star_optical = optical_to_ionizing_fraction * L_star_ion;
	const amrex::Real x0 = 0.0_rt;
	const amrex::Real y0 = 0.0_rt;
	const amrex::Real z0 = 0.0_rt;
	const amrex::Real volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	const amrex::Real inv_volume = 1.0 / volume;

	const int src_i = static_cast<int>(amrex::Math::floor((x0 - prob_lo[0]) / dx[0]));
	const int src_j = static_cast<int>(amrex::Math::floor((y0 - prob_lo[1]) / dx[1]));
	const int src_k = static_cast<int>(amrex::Math::floor((z0 - prob_lo[2]) / dx[2]));
	const amrex::Real frac_x = (x0 - prob_lo[0]) / dx[0] - static_cast<amrex::Real>(src_i);
	const amrex::Real frac_y = (y0 - prob_lo[1]) / dx[1] - static_cast<amrex::Real>(src_j);
	const amrex::Real frac_z = (z0 - prob_lo[2]) / dx[2] - static_cast<amrex::Real>(src_k);

	constexpr int stencil_width = 2 * N + 1;
	const int nz_loop = (AMREX_SPACEDIM >= 3) ? stencil_width : 1;
	const int ny_loop = (AMREX_SPACEDIM >= 2) ? stencil_width : 1;
	amrex::Real norm_sum = 0.0_rt;
	for (int kk = 0; kk < nz_loop; ++kk) {
		const amrex::Real dz = (AMREX_SPACEDIM >= 3) ? static_cast<amrex::Real>(kk - N) + 0.5 - frac_z : 0.0;
		for (int jj = 0; jj < ny_loop; ++jj) {
			const amrex::Real dy = (AMREX_SPACEDIM >= 2) ? static_cast<amrex::Real>(jj - N) + 0.5 - frac_y : 0.0;
			for (int ii = 0; ii < stencil_width; ++ii) {
				const amrex::Real di = static_cast<amrex::Real>(ii - N) + 0.5 - frac_x;
				const amrex::Real r2 = AMREX_D_TERM(di * di, +dy * dy, +dz * dz);
				if (r2 <= cutoff_r2) {
					norm_sum += wendland_c2(std::sqrt(r2) * inv_N);
				}
			}
		}
	}

	const amrex::Real inv_norm = 1.0_rt / norm_sum;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const amrex::Real di = static_cast<amrex::Real>(i - src_i) + 0.5 - frac_x;
		const amrex::Real dj = (AMREX_SPACEDIM >= 2) ? static_cast<amrex::Real>(j - src_j) + 0.5 - frac_y : 0.0;
		const amrex::Real dk = (AMREX_SPACEDIM >= 3) ? static_cast<amrex::Real>(k - src_k) + 0.5 - frac_z : 0.0;
		const amrex::Real r2 = AMREX_D_TERM(di * di, +dj * dj, +dk * dk);
		radEnergy(i, j, k, IRBand) = 0.0_rt;
		if (r2 <= cutoff_r2) {
			const amrex::Real shape = wendland_c2(std::sqrt(r2) * inv_N) * inv_norm * inv_volume;
			radEnergy(i, j, k, OpticalBand) = L_star_optical * shape;
			radEnergy(i, j, k, IonizingBand) = L_star_ion * shape;
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
	userData_.Q = 1.0e49_rt;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("primary_species_1", userData_.primary_species_1);
	pp.query("primary_species_2", userData_.primary_species_2);
	pp.query("primary_species_3", userData_.primary_species_3);
	pp.query("Q", userData_.Q);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
	userData_.r_analytical_last_t = 0.0_rt;
	userData_.r_analytical_last_R = 0.0_rt;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::string const filename = "dtype_front_radii.csv";
		userData_.output_file_.open(filename);
		userData_.output_file_ << "time,r_effective,r_analytical\n";
	}
}

constexpr double kappa_IR = 1.0e-2;
constexpr double kappa_optical = 3.0e2;

template <>
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
		exponents_and_values[1][g] = 0.0;
	}
	exponents_and_values[1][IRBand] = kappa_IR;
	exponents_and_values[1][OpticalBand] = kappa_optical;
	return exponents_and_values;
}

template <> void QuokkaSimulation<DTypeFrontMG>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	burn_t state;
	std::array<Real, NumSpec> numdens = {-1.0};
	for (int n = 1; n <= NumSpec; ++n) {
		switch (n) {
			case 1:
				numdens[n - 1] = userData_.primary_species_1;
				break;
			case 2:
				numdens[n - 1] = userData_.primary_species_2;
				break;
			case 3:
				numdens[n - 1] = userData_.primary_species_3;
				break;
			default:
				amrex::Abort("Cannot initialize number density for chem specie");
				break;
		}
	}

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
			state_cc(i, j, k, RadSystem<DTypeFrontMG>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) =
			    RadSystem_Traits<DTypeFrontMG>::Erad_floor;
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
	userData_.r_effective_vec_.push_back(r_effective);
	userData_.t_vec_.push_back(tNew_[lev]);

	const amrex::Real n_e = userData_.primary_species_2;
	const double T_eq = compute_equilibrium_temperature_ionized(static_cast<double>(n_e));
	const double alpha_B = 2.6e-13 * std::pow(T_eq / 1.0e4, -0.7);
	const double mu = 0.5;
	const double c_i = std::sqrt(C::k_B * T_eq / (mu * C::m_p));
	const amrex::Real r_s = std::pow((3.0_rt * userData_.Q) / (4.0_rt * M_PI * alpha_B * n_e * n_e), 1.0_rt / 3.0_rt);
	const amrex::Real t_s = r_s / static_cast<amrex::Real>(c_i);

	const amrex::Real t = tNew_[lev];
	amrex::Real r_analytical = 0.0_rt;
	if (t_s > 0.0_rt) {
		r_analytical = r_s * std::pow(1.0_rt + 7.0_rt * t / (4.0_rt * t_s), 4.0_rt / 7.0_rt);
	}
	userData_.r_analytical_vec_.push_back(r_analytical);

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

	// Check 1: effective radius vs analytical radius at end of simulation
	{
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = sim.geom[0].CellSizeArray();
		const amrex::Real cell_size = dx[0];
		const amrex::Real tol_cells = 3.0_rt * std::sqrt(3.0_rt) / 2.0_rt;

		if (!sim.userData_.r_effective_vec_.empty()) {
			const amrex::Real r_analytical = sim.userData_.r_analytical_vec_.back();
			const amrex::Real r_effective = sim.userData_.r_effective_vec_.back();
			const amrex::Real delta_over_dx = (r_effective - r_analytical) / cell_size;
			if ((delta_over_dx < -tol_cells) || (delta_over_dx > tol_cells)) {
				amrex::Print() << "Test FAILED: radius check at end of simulation.\n";
				amrex::Print() << "Analytical radius: " << r_analytical << '\n';
				amrex::Print() << "Effective radius: " << r_effective << '\n';
				amrex::Print() << "(r_effective - r_analytical) / dx = " << delta_over_dx << '\n';
				amrex::Print() << "Tolerance: " << tol_cells << " cell sizes\n";
				status = 1;
			} else {
				amrex::Print() << "Test passed: D-type front effective radius matches analytical radius within " << tol_cells
					       << " cell sizes at end of simulation.\n";
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
		// Plot radii vs time
		matplotlibcpp::clf();
		std::map<std::string, std::string> numerical_args;
		numerical_args["label"] = "numerical";
		numerical_args["color"] = "C0";
		std::map<std::string, std::string> analytical_args;
		analytical_args["label"] = "analytical";
		analytical_args["color"] = "k";
		analytical_args["linestyle"] = "--";

		matplotlibcpp::plot(sim.userData_.t_vec_, sim.userData_.r_effective_vec_, numerical_args);
		matplotlibcpp::plot(sim.userData_.t_vec_, sim.userData_.r_analytical_vec_, analytical_args);
		matplotlibcpp::xlabel("time (s)");
		matplotlibcpp::ylabel("radius (cm)");
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
		matplotlibcpp::plot(sim.userData_.t_vec_, delta_over_dx_vec, diff_args);
		matplotlibcpp::xlabel("time (s)");
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
