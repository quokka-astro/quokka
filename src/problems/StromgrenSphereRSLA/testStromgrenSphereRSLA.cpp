// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testStromgrenSphere.cpp
/// \brief Defines a test problem for the static Stromgren sphere with no temperature dependence.
///

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"
#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "radiation/photochemistry.hpp"
#include "radiation/radiation_system.hpp"
#include "util/fundamental_constants.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ios>
#include <map>
#include <string>
#include <utility>

#include "microphysics/burn_state.hpp"
#include "microphysics/microphysics_traits.hpp"
#include "networks/photoionization_H_caseB_handwritten/network.hpp"

struct StromgrenSphere {};

template <> struct Microphysics_Traits<StromgrenSphere> : DefaultMicrophysicsTraits {
	using Network = quokka::networks::PhotoionizationHCaseBHandwritten;
	static constexpr bool photochemistry = true;
};

using Network = Microphysics_Traits<StromgrenSphere>::Network;
using amrex::literals::operator""_rt;

constexpr double c_hat = C::c_light / static_cast<double>(C_HAT_DIVISOR);
constexpr int kernel_radius_cells = 2; // support radius of the source's Wendland C2 kernel, as in DTypeFront_JAFF

template <> struct quokka::EOS_Traits<StromgrenSphere> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<StromgrenSphere> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = Network::NumSpec;	     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
};

template <> struct RadSystem_Traits<StromgrenSphere> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = 1e-99;
	static constexpr int beta_order = 0;
	static constexpr auto ChemBandsPowerLawIndex() { return Network::ChemBandsPowerLawIndex(); }
	static constexpr auto ChemBands() { return Network::ChemBands(); }
};
namespace
{

// Sphere-equivalent radii of the ionized volume, int (1 - x_HI) dV, and of the region with x_HI < 1/2 (the whole sphere is in the domain).
// The first exceeds the Stromgren radius by the partially ionized front width; the second tracks the front itself.
auto compute_effective_radii(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> std::pair<amrex::Real, amrex::Real>
{
	amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data,
		       [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
			       const amrex::Real n_HI = state[box_no](i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + static_cast<int>(Network::H)) /
							Network::SpeciesMass()[Network::H];
			       const amrex::Real n_HII = state[box_no](i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + static_cast<int>(Network::H_p)) /
							 Network::SpeciesMass()[Network::H_p];
			       const amrex::Real denom = n_HI + n_HII;
			       if (denom <= 0.0_rt) {
				       return {0.0_rt, 0.0_rt};
			       }
			       const amrex::Real x_HI = n_HI / denom;
			       return {cell_volume * (1.0_rt - x_HI), (x_HI < 0.5_rt) ? cell_volume : 0.0_rt};
		       });

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real total_ionized_volume = amrex::get<0>(hv);
	amrex::Real half_ionized_volume = amrex::get<1>(hv);
	amrex::ParallelAllReduce::Sum(total_ionized_volume, amrex::ParallelContext::CommunicatorSub());
	amrex::ParallelAllReduce::Sum(half_ionized_volume, amrex::ParallelContext::CommunicatorSub());
	return {std::cbrt((3.0_rt * total_ionized_volume) / (4.0_rt * M_PI)), std::cbrt((3.0_rt * half_ionized_volume) / (4.0_rt * M_PI))};
}

AMREX_GPU_HOST_DEVICE auto wendland_c2(amrex::Real r) -> amrex::Real
{
	if (r > 1.0) {
		return 0.0;
	}
	return (21. / (2. * M_PI)) * std::pow((1.0 - r), 4) * (4.0 * r + 1.0);
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
void RadSystem<StromgrenSphere>::AddRadSource(array_t &radEnergy, array_t & /*reducedFluxSource*/, const amrex::Box &indexRange,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real /*time*/)
{
	amrex::ParmParse const pp("stromgen");
	amrex::Real Q = 1.0e49_rt;
	pp.query("Q", Q);

	// Wendland C2 kernel of radius N cells around the source at the origin (the centre of the domain), normalised over its
	// discrete stencil, as in DTypeFront_JAFF
	constexpr int N = kernel_radius_cells;
	constexpr amrex::Real inv_N = 1.0 / static_cast<amrex::Real>(N);
	constexpr auto cutoff_r2 = static_cast<amrex::Real>(N * N);

	const amrex::Real L_star = Q * RadSystem<StromgrenSphere>::GetChemBandQuanta(0);
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
		if (r2 <= cutoff_r2) {
			radEnergy(i, j, k) = L_star * wendland_c2(std::sqrt(r2) * inv_N) * inv_norm * inv_volume;
		} else {
			radEnergy(i, j, k) = 0.0_rt;
		}
	});
}

template <> struct SimulationData<StromgrenSphere> {
	amrex::Real temperature{};
	amrex::Real n_e_init{};
	amrex::Real n_HI_init{};
	amrex::Real n_HII_init{};
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
	// parmparse species and temperature
	amrex::ParmParse const pp("stromgen");
	userData_.temperature = 1.0e4;
	userData_.tend = 1000.0_rt;
	userData_.n_e_init = 0.0e0_rt;
	userData_.n_HI_init = 1.0e2_rt;
	userData_.n_HII_init = 0.0e0_rt;
	userData_.Q = 1.0e49_rt;
	userData_.recombination_switch = 0;
	pp.query("temperature", userData_.temperature);
	pp.query("tend", userData_.tend);
	pp.query("n_e_init", userData_.n_e_init);
	pp.query("n_HI_init", userData_.n_HI_init);
	pp.query("n_HII_init", userData_.n_HII_init);
	pp.query("Q", userData_.Q);

	amrex::ParmParse const pp2("network");
	pp2.query("recombination_switch", userData_.recombination_switch);

	userData_.r_analytical_last_t = 0.0_rt;
	userData_.r_analytical_last_R = 0.0_rt;
	std::string restart_chkfile;
	amrex::ParmParse const pp_global;
	pp_global.query("restartfile", restart_chkfile);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::string const filename = "stromgren_sphere_radii.csv";
		if (restart_chkfile.empty()) {
			userData_.output_file_.open(filename);
			userData_.output_file_ << "time,r_effective,r_analytical,r_half\n";
		} else {
			userData_.output_file_.open(filename, std::ios::app);
		}
		userData_.output_file_ << std::setprecision(17);
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

	quokka::microphysics::BurnState<Network::NumSpec> state;
	std::array<Real, Network::NumSpec> numdens = {-1.0};
	numdens[Network::e] = userData_.n_e_init;
	numdens[Network::H] = userData_.n_HI_init;
	numdens[Network::H_p] = userData_.n_HII_init;

	state.T = userData_.temperature;
	// find the density in g/cm^3
	Real rhotot = 0.0_rt;
	for (int n = 0; n < Network::NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * Network::SpeciesMass()[n];
	}
	state.rho = rhotot;

	// call the EOS to set initial internal energy e
	state.e = quokka::EOSMultigamma<StromgrenSphere>::EnergyFromTemperature(state.T, state.xn);
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
		for (int nn = 0; nn < Network::NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + nn) =
			    state.xn[nn] * Network::SpeciesMass()[nn]; // scalar indices carry partial densities instead of number densities
		}
	});
}

template <> void QuokkaSimulation<StromgrenSphere>::computeAfterTimestep()
{
	const int lev = 0;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	auto const [r_effective, r_half] = compute_effective_radii(state_new_cc_[lev], dx);
	userData_.r_effective_vec_.push_back(r_effective);
	userData_.t_vec_.push_back(tNew_[lev]);

	const amrex::Real n_HI0 = userData_.n_HI_init;
	const amrex::Real alpha_B = 2.63e-13;
	const amrex::Real r_s = std::pow((3.0_rt * userData_.Q) / (4.0_rt * M_PI * alpha_B * n_HI0 * n_HI0), 1.0_rt / 3.0_rt);

	if (tNew_[lev] < userData_.r_analytical_last_t) {
		userData_.r_analytical_last_t = 0.0_rt;
		userData_.r_analytical_last_R = 0.0_rt;
	}

	// After a restart the ODE starts again from t = 0: catch up over intervals doubling from one timestep, so each call
	// stays resolved. Otherwise this is a single call over the step just taken.
	const amrex::Real first_interval = (dt_[lev] > 0.0_rt) ? dt_[lev] : tNew_[lev];
	amrex::Real r_new = userData_.r_analytical_last_R;
	amrex::Real t_ode = userData_.r_analytical_last_t;
	while (t_ode < tNew_[lev]) {
		const amrex::Real t_next = std::min(tNew_[lev], std::max(2.0_rt * t_ode, first_interval));
		r_new = integrate_radius(t_next - t_ode, userData_.Q, alpha_B, n_HI0, C::c_light, r_new, r_s);
		t_ode = t_next;
	}
	userData_.r_analytical_last_t = tNew_[lev];
	userData_.r_analytical_last_R = r_new;
	userData_.r_analytical_vec_.push_back(r_new);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_ << tNew_[lev] << ',' << r_effective << ',' << r_new << ',' << r_half << '\n';
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

	// preCalculateInitialConditions() (parameters, EOS, network, output file) only runs when initialising from scratch
	std::string restart_chkfile;
	amrex::ParmParse const pp;
	pp.query("restartfile", restart_chkfile);
	if (!restart_chkfile.empty()) {
		sim.preCalculateInitialConditions();
	}

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
		const amrex::Real r_kernel = kernel_radius_cells * cell_size;

		for (int i = 0; i < sim.userData_.t_vec_.size(); ++i) {
			const amrex::Real r_analytical = sim.userData_.r_analytical_vec_[i];
			if (r_analytical <= r_kernel) {
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
			amrex::Print() << "Test passed: Effective Stromgren radius matches the analytical radius within one cell diagonal whenever "
					  "r_analytical > r_kernel."
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
