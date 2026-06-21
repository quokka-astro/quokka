//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testStromgrenSphere.cpp
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
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include <cmath>
#include <map>
#include <string>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct StromgrenSphere {
};

constexpr double c_hat = C::c_light / 100.0;
constexpr double sigma_star_coeff = 1.5 / 16.0;
constexpr double r_trunc_coeff = 2.5;

template <> struct quokka::EOS_Traits<StromgrenSphere> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<StromgrenSphere> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = NumSpec;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
};

template <> struct RadSystem_Traits<StromgrenSphere> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = 1e-99;
	static constexpr int beta_order = 0;
	static constexpr auto ChemBands() { return ChemBandsHeader_; }
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
		const amrex::Real n_HI = state[box_no](i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + 1) / spmasses[1];
		const amrex::Real n_HII = state[box_no](i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + 2) / spmasses[2];
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

auto integrate_radius(amrex::Real dt_target, amrex::Real Q, amrex::Real alpha_B, amrex::Real n_HI0, amrex::Real c_light, amrex::Real R0, amrex::Real r_s_est)
    -> amrex::Real
{
	if (dt_target <= 0.0_rt) {
		return R0;
	}

	auto rhs = [&](amrex::Real R) -> amrex::Real {
		const amrex::Real num = Q - (4.0_rt * M_PI * R * R * R * alpha_B * n_HI0 * n_HI0) / 3.0_rt;
		const amrex::Real den = Q / c_light + 4.0_rt * M_PI * R * R * n_HI0;
		return num / den;
	};

	int N = 256;
	const int max_iters = 10;
	const amrex::Real tol = 1e-6_rt * std::max(r_s_est, 1.0_rt);
	amrex::Real R_prev = R0;

	for (int iter = 0; iter < max_iters; ++iter) {
		const amrex::Real dt = dt_target / static_cast<amrex::Real>(N);
		amrex::Real R = R0;

		for (int step = 0; step < N; ++step) {
			const amrex::Real k1 = rhs(R);
			const amrex::Real k2 = rhs(R + 0.5_rt * dt * k1);
			const amrex::Real k3 = rhs(R + 0.5_rt * dt * k2);
			const amrex::Real k4 = rhs(R + dt * k3);
			R += (dt / 6.0_rt) * (k1 + 2.0_rt * k2 + 2.0_rt * k3 + k4);
			R = std::max(R, 0.0_rt);
		}

		if (iter > 0 && std::abs(R - R_prev) < tol) {
			return R;
		}
		R_prev = R;
		N *= 2;
	}

	amrex::Abort("integrate_radius failed to converge within max_iters for dt=" + std::to_string(dt_target));
	return R_prev;
}

} // namespace

template <>
void RadSystem<StromgrenSphere>::SetRadEnergySource(array_t &radEnergy, const amrex::Box &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
						    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	amrex::ParmParse const pp("stromgen");
	amrex::Real Q = 1.0e49_rt;
	pp.query("Q", Q);

	amrex::ParmParse const pp2("amr");
	int n = 16;
	pp2.query("n_cell", n);

	const amrex::Real sigma_star = sigma_star_coeff * (prob_hi[0] - prob_lo[0]);
	const amrex::Real r_trunc = r_trunc_coeff * sigma_star;
	const amrex::Real L_star = Q * RadSystem<StromgrenSphere>::GetChemBandQuanta(0) / 8.0_rt;
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
			amrex::Real const w_i = std::exp(-(r * r) / (2.0 * sigma_star * sigma_star)) / (std::pow(2.0 * M_PI * sigma_star * sigma_star, 1.5));
			amrex::Real const val = L_star * w_i / sum;
			radEnergy(i, j, k) = val;
		} else {
			radEnergy(i, j, k) = 0.0_rt;
		}
	});
}

template <> struct SimulationData<StromgrenSphere> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real primary_species_1{};
	amrex::Real primary_species_2{};
	amrex::Real primary_species_3{};
	amrex::Real Q{};
	amrex::Real tend{};
	int recombination_switch{};
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> r_effective_vec_;
	amrex::Vector<amrex::Real> r_analytical_vec_;
	amrex::Real r_analytical_last_t{};
	amrex::Real r_analytical_last_R{};
	std::ofstream output_file_;
};

template <> void QuokkaSimulation<StromgrenSphere>::preCalculateInitialConditions()
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
	userData_.recombination_switch = 0;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("tend", userData_.tend);
	pp.query("primary_species_1", userData_.primary_species_1);
	pp.query("primary_species_2", userData_.primary_species_2);
	pp.query("primary_species_3", userData_.primary_species_3);
	pp.query("Q", userData_.Q);

	amrex::ParmParse const pp2("network");
	pp2.query("recombination_switch", userData_.recombination_switch);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
	userData_.r_analytical_last_t = 0.0_rt;
	userData_.r_analytical_last_R = 0.0_rt;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::string const filename = "stromgren_sphere_radii.csv";
		userData_.output_file_.open(filename);
		userData_.output_file_ << "time,r_effective,r_analytical\n";
	}
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StromgrenSphere>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StromgrenSphere>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> void QuokkaSimulation<StromgrenSphere>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
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
		for (int g = 0; g < Physics_Traits<StromgrenSphere>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<StromgrenSphere>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = 1.e-99_rt;
			state_cc(i, j, k, RadSystem<StromgrenSphere>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<StromgrenSphere>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<StromgrenSphere>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
		}
		state_cc(i, j, k, RadSystem<StromgrenSphere>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StromgrenSphere>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<StromgrenSphere>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StromgrenSphere>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<StromgrenSphere>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<StromgrenSphere>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + nn) =
			    state.xn[nn] * spmasses[nn]; // scalar indices carry partial densities instead of number densities
		}
	});
}

template <> void QuokkaSimulation<StromgrenSphere>::computeAfterTimestep()
{
	const int lev = 0;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const amrex::Real r_effective = compute_effective_radius(state_new_cc_[lev], dx);
	userData_.r_effective_vec_.push_back(r_effective);
	userData_.t_vec_.push_back(tNew_[lev]);

	const amrex::Real n_HI0 = userData_.primary_species_2;
	const amrex::Real alpha_B = 2.6e-13;
	const amrex::Real r_s = std::pow((3.0_rt * userData_.Q) / (4.0_rt * M_PI * alpha_B * n_HI0 * n_HI0), 1.0_rt / 3.0_rt);

	amrex::Real dt = tNew_[lev] - userData_.r_analytical_last_t;
	if (dt < 0.0_rt) {
		userData_.r_analytical_last_t = 0.0_rt;
		userData_.r_analytical_last_R = 0.0_rt;
		dt = tNew_[lev];
	}

	const amrex::Real r_new = integrate_radius(dt, userData_.Q, alpha_B, n_HI0, C::c_light, userData_.r_analytical_last_R, r_s);
	userData_.r_analytical_last_t = tNew_[lev];
	userData_.r_analytical_last_R = r_new;
	userData_.r_analytical_vec_.push_back(r_new);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_ << tNew_[lev] << ',' << r_effective << ',' << r_new << '\n';
	}
}

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 0.3;
	const double dt_max = 1e99;
	const int max_timesteps = 5000000;

	// Problem initialization
	QuokkaSimulation<StromgrenSphere> sim;

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

	if (amrex::ParallelDescriptor::IOProcessor()) {
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = sim.geom[0].CellSizeArray();
		const amrex::Real cell_size = dx[0];
		const amrex::Real bound = std::sqrt(3.0_rt);
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim.geom[0].ProbLoArray();
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim.geom[0].ProbHiArray();
		const amrex::Real sigma_star = sigma_star_coeff * (prob_hi[0] - prob_lo[0]);
		const amrex::Real r_trunc = r_trunc_coeff * sigma_star;

		for (int i = 0; i < sim.userData_.t_vec_.size(); ++i) {
			const amrex::Real r_analytical = sim.userData_.r_analytical_vec_[i];
			if (r_analytical <= r_trunc) {
				continue;
			}
			const amrex::Real r_effective = sim.userData_.r_effective_vec_[i];
			const amrex::Real delta_over_dx = (r_effective - r_analytical) / cell_size;
			if ((delta_over_dx < -bound) || (delta_over_dx > bound)) {
				amrex::Print() << "Test failed at t = " << sim.userData_.t_vec_[i] << '\n';
				amrex::Print() << "Analytical radius: " << r_analytical << '\n';
				amrex::Print() << "Effective radius: " << r_effective << '\n';
				amrex::Print() << "(r_effective - r_analytical) / dx = " << delta_over_dx << '\n';
				amrex::Print() << "Expected range: [" << -bound << ", " << bound << "]" << '\n';
				status = 1;
			}
		}

		if (status == 0) {
			amrex::Print()
			    << "Test passed: Effective Stromgren radius matches the analytical radius within one cell diagonal whenever r_analytical > r_trunc."
			    << '\n';
		}
	}

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		matplotlibcpp::clf();
		std::map<std::string, std::string> numerical_args;
		std::map<std::string, std::string> analytical_args;
		numerical_args["label"] = "numerical";
		numerical_args["color"] = "C0";
		analytical_args["label"] = "analytical";
		analytical_args["color"] = "k";
		analytical_args["linestyle"] = "--";

		matplotlibcpp::plot(sim.userData_.t_vec_, sim.userData_.r_effective_vec_, numerical_args);
		matplotlibcpp::plot(sim.userData_.t_vec_, sim.userData_.r_analytical_vec_, analytical_args);
		matplotlibcpp::xlabel("time");
		matplotlibcpp::ylabel("radius");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./stromgren_sphere_rsla_radii.pdf");

		// Plot normalized difference
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = sim.geom[0].CellSizeArray();
		const amrex::Real cell_size = dx[0];
		std::vector<amrex::Real> delta_over_dx_vec(sim.userData_.t_vec_.size());
		for (int i = 0; i < sim.userData_.t_vec_.size(); ++i) {
			delta_over_dx_vec[i] = std::abs(sim.userData_.r_effective_vec_[i] - sim.userData_.r_analytical_vec_[i]) / cell_size;
		}

		matplotlibcpp::clf();
		std::map<std::string, std::string> diff_args;
		diff_args["label"] = "(r_effective - r_analytical) / dx";
		diff_args["color"] = "C1";
		matplotlibcpp::plot(sim.userData_.t_vec_, delta_over_dx_vec, diff_args);
		matplotlibcpp::xlabel("time");
		matplotlibcpp::ylabel("delta r / dx");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./stromgren_sphere_rsla_radii_difference.pdf");
	}
#endif

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return status;
}
