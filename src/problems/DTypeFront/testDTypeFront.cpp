//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testDTypeFront.cpp
/// \brief Defines a test problem for the static Stromgren sphere with no temperature dependence.
///

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct DTypeFront {
};

constexpr double c_hat = C::c_light / 10.0;
constexpr double sigma_star_coeff = 1.5 / 16.0;
constexpr double r_trunc_coeff = 2.5;

template <> struct quokka::EOS_Traits<DTypeFront> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DTypeFront> {
	static constexpr bool is_self_gravity_enabled = false;
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = NumSpec;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gravitational_constant = C::Gconst;
	static constexpr double c_light = C::c_light;
	static constexpr double radiation_constant = C::a_rad;
};

template <> struct RadSystem_Traits<DTypeFront> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = 1e-99;
	static constexpr int beta_order = 0;
	AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<double, NumChemBands + 1> ChemBands() { return ChemBandsHeader_; }
};

template <>
void RadSystem<DTypeFront>::SetRadEnergySource(array_t &radEnergy, const amrex::Box &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	amrex::ParmParse pp("stromgen");
	amrex::Real Q = 1.0e49_rt;
	pp.query("Q", Q);

	amrex::ParmParse pp2("amr");
	int n = 16;
	pp2.query("n_cell", n);

	const amrex::Real sigma_star = sigma_star_coeff * (prob_hi[0] - prob_lo[0]);
	const amrex::Real r_trunc = r_trunc_coeff * sigma_star;
	const amrex::Real L_star = Q * RadSystem<DTypeFront>::GetChemBandQuanta(0) / 8.0_rt;
	const amrex::Real x0 = 0.0_rt;
	const amrex::Real y0 = 0.0_rt;
	const amrex::Real z0 = 0.0_rt;
	amrex::Real sum = 0.0_rt;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			for (int k = 0; k < n; ++k) {
				amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
				amrex::Real const y = prob_lo[1] + (j + 0.5) * dx[1];
				amrex::Real const z = prob_lo[2] + (k + 0.5) * dx[2];
				amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));
				if (r <= r_trunc) {
					sum += std::exp(-(r * r) / (2.0 * sigma_star * sigma_star)) * dx[0] * dx[1] * dx[2] /
					       (std::pow(2.0 * M_PI * sigma_star * sigma_star, 1.5));
				}
			}
		}
	}
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + 0.5) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + 0.5) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));
		if (r <= r_trunc) {
			amrex::Real w_i = std::exp(-(r * r) / (2.0 * sigma_star * sigma_star)) / (std::pow(2.0 * M_PI * sigma_star * sigma_star, 1.5));
			amrex::Real val = L_star * w_i / sum;
			radEnergy(i, j, k) = val;
		} else {
			radEnergy(i, j, k) = 0.0_rt;
		}
	});
}

template <> struct SimulationData<DTypeFront> {
	amrex::Real small_temp;
	amrex::Real small_dens;
	amrex::Real temperature;
	amrex::Real primary_species_1;
	amrex::Real primary_species_2;
	amrex::Real primary_species_3;
	amrex::Real Q;
	amrex::Real tend;
	int recombination_switch;
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> r50_vec_;
	amrex::Vector<amrex::Real> r16_vec_;
	amrex::Vector<amrex::Real> r84_vec_;
	amrex::Vector<amrex::Real> r_analytical_vec_;
	amrex::Real r_analytical_last_t;
	amrex::Real r_analytical_last_R;
	std::ofstream output_file_;
};

template <> void QuokkaSimulation<DTypeFront>::preCalculateInitialConditions()
{
	// initialize microphysics routines
	init_extern_parameters();

	// parmparse species and temperature
	amrex::ParmParse const pp("stromgen");
	userData_.small_temp = 1e-2;
	userData_.small_dens = 1e-60;
	userData_.temperature = 1.0e4;
	userData_.tend = 1000.0_rt;
	userData_.primary_species_1 = 0.0e0_rt;
	userData_.primary_species_2 = 1.0e2_rt;
	userData_.primary_species_3 = 0.0e0_rt;
	userData_.Q = 1.0e49_rt;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("tend", userData_.tend);
	pp.query("primary_species_1", userData_.primary_species_1);
	pp.query("primary_species_2", userData_.primary_species_2);
	pp.query("primary_species_3", userData_.primary_species_3);
	pp.query("Q", userData_.Q);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DTypeFront>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DTypeFront>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> void QuokkaSimulation<DTypeFront>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
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
		for (int g = 0; g < Physics_Traits<DTypeFront>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<DTypeFront>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = 1.e-99_rt;
			state_cc(i, j, k, RadSystem<DTypeFront>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFront>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFront>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
		}
		state_cc(i, j, k, RadSystem<DTypeFront>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFront>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<DTypeFront>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFront>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFront>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFront>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<DTypeFront>::scalar0_index + nn) =
			    state.xn[nn] * spmasses[nn]; // scalar indices carry partial densities instead of number densities
		}
	});
}

template <> void QuokkaSimulation<DTypeFront>::computeAfterTimestep() {}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
	const double CFL_number = 0.3;
	const double dt_max = 1e99;
	const int max_timesteps = 5000000;

	// Problem initialization
	QuokkaSimulation<DTypeFront> sim;

	// initialize
	sim.setInitialConditions();
	sim.stopTime_ = sim.userData_.tend;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	int status = 0;

	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return status;
}
