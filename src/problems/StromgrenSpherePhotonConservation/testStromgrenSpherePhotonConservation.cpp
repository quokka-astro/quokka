//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testStromgrenSpherePhotonConservation.cpp
/// \brief Photon conservation across a moving ionization front: the Stromgren sphere setup with c_hat = c and no recombination,
/// in the photon-limited regime where the front moves at a sizeable fraction of c and absorption dominates the photon budget.
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
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ios>
#include <map>
#include <string>
#include <utility>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct StromgrenSphere {};

constexpr double c_hat = C::c_light;
constexpr double sigma_star_coeff = 1.5 / 16.0;
constexpr double r_trunc_coeff = 2.5;
constexpr double Erad_init = 1.0e-99;
constexpr double photon_err_tol = 1.0e-12;

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
	static constexpr auto ChemBandsPowerLawIndex() { return ChemBandsPowerLawIndex_; }
	static constexpr auto ChemBands() { return ChemBandsHeader(); }
};
namespace
{

// Front radius of a point source in a uniform medium without recombination: the photons emitted, Q t, are those absorbed
// in the ionized sphere plus those still in flight inside it, (4 pi / 3) n R^3 + Q R / c = Q t. This is the real root of
// R^3 + p R - q = 0, written as q / (u^2 - u v + v^2) with Cardano's u, v (u + v = R, u^3 + v^3 = q) to avoid cancellation.
auto analytic_radius(amrex::Real t, amrex::Real Q, amrex::Real n_H, amrex::Real c_light) -> amrex::Real
{
	const amrex::Real p = 3.0_rt * Q / (4.0_rt * M_PI * n_H * c_light);
	const amrex::Real q = 3.0_rt * Q * t / (4.0_rt * M_PI * n_H);
	const amrex::Real u = std::cbrt(0.5_rt * q + std::sqrt(0.25_rt * q * q + p * p * p / 27.0_rt));
	const amrex::Real v = -p / (3.0_rt * u);
	return q / (u * u - u * v + v * v);
}

} // namespace

template <>
void RadSystem<StromgrenSphere>::AddRadSource(array_t &radEnergy, array_t & /*reducedFluxSource*/, const amrex::Box &indexRange,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	amrex::ParmParse const pp("stromgen");
	amrex::Real Q = 1.0e49_rt;
	pp.query("Q", Q);

	const amrex::Real sigma_star = sigma_star_coeff * (prob_hi[0] - prob_lo[0]);
	const amrex::Real r_trunc = r_trunc_coeff * sigma_star;
	const amrex::Real L_star = Q * RadSystem<StromgrenSphere>::GetChemBandQuanta(0) / 8.0_rt;
	const amrex::Real x0 = 0.0_rt;
	const amrex::Real y0 = 0.0_rt;
	const amrex::Real z0 = 0.0_rt;
	// the normalisation depends only on the geometry: an O(n^3) host loop per box per call is prohibitive at high resolution
	static const amrex::Real sum_cached = [&]() {
		amrex::ParmParse const pp2("amr");
		int n = 16;
		pp2.query("n_cell", n);
		// Neumaier-compensated: a plain sum over ~1e5 cells is off by ~1e-13, which would show up as a photon conservation error
		amrex::Real s = 0.0_rt;
		amrex::Real comp = 0.0_rt;
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				for (int k = 0; k < n; ++k) {
					amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
					amrex::Real const y = prob_lo[1] + (j + 0.5) * dx[1];
					amrex::Real const z = prob_lo[2] + (k + 0.5) * dx[2];
					amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));
					if (r <= r_trunc) {
						amrex::Real const term = std::exp(-(r * r) / (2.0 * sigma_star * sigma_star)) * dx[0] * dx[1] * dx[2] /
									 (std::pow(2.0 * M_PI * sigma_star * sigma_star, 1.5));
						amrex::Real const t = s + term;
						comp += (std::abs(s) >= std::abs(term)) ? (s - t) + term : (term - t) + s;
						s = t;
					}
				}
			}
		}
		return s + comp;
	}();
	const amrex::Real sum = sum_cached; // a local copy, so the device lambda captures it by value
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
	amrex::Real n_e_init{};
	amrex::Real n_HI_init{};
	amrex::Real n_HII_init{};
	amrex::Real Q{};
	amrex::Real tend{};
	amrex::Real N_gamma0{};
	amrex::Real N_e0{};
	amrex::Real max_photon_err{};
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> r_effective_vec_;
	amrex::Vector<amrex::Real> r_analytical_vec_;
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
	userData_.n_e_init = 0.0e0_rt;
	userData_.n_HI_init = 1.0e2_rt;
	userData_.n_HII_init = 0.0e0_rt;
	userData_.Q = 1.0e49_rt;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("tend", userData_.tend);
	pp.query("n_e_init", userData_.n_e_init);
	pp.query("n_HI_init", userData_.n_HI_init);
	pp.query("n_HII_init", userData_.n_HII_init);
	pp.query("Q", userData_.Q);

	// the budget below assumes each absorbed photon creates exactly one electron and nothing else creates or removes them
	if (network_rp::recombination_switch != 0 || network_rp::collisional_ionization_switch != 0) {
		amrex::Abort("StromgrenSpherePhotonConservation needs network.recombination_switch = 0 and network.collisional_ionization_switch = 0");
	}

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();

	// Domain totals at t = 0 from the initial conditions, so a restart has the same baseline as the original run
	const auto prob_lo = geom[0].ProbLoArray();
	const auto prob_hi = geom[0].ProbHiArray();
	const amrex::Real volume = (prob_hi[0] - prob_lo[0]) * (prob_hi[1] - prob_lo[1]) * (prob_hi[2] - prob_lo[2]);
	userData_.N_gamma0 = Erad_init / RadSystem<StromgrenSphere>::GetChemBandQuanta(0) * volume;
	userData_.N_e0 = userData_.n_e_init * volume;
	userData_.max_photon_err = 0.0_rt;

	std::string restart_chkfile;
	amrex::ParmParse const pp_global;
	pp_global.query("restartfile", restart_chkfile);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::string const filename = "stromgren_sphere_photon_conservation.csv";
		if (restart_chkfile.empty()) {
			userData_.output_file_.open(filename);
			userData_.output_file_ << "time,r_effective,r_analytical,r_half,N_injected,N_inflight,N_absorbed,rel_err\n";
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

	burn_t state;
	std::array<Real, NumSpec> numdens = {-1.0};
	numdens[Species::e] = userData_.n_e_init;
	numdens[Species::H] = userData_.n_HI_init;
	numdens[Species::H_p] = userData_.n_HII_init;

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
			state_cc(i, j, k, RadSystem<StromgrenSphere>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad_init;
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
	const amrex::Real quanta = RadSystem<StromgrenSphere>::GetChemBandQuanta(0);
	const amrex::Real N_gamma =
	    computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state,
						       std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const & /*state_fc*/) noexcept {
		    return state(i, j, k, RadSystem<StromgrenSphere>::radEnergy_index) / quanta;
	    });
	const amrex::Real N_e =
	    computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state,
						       std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const & /*state_fc*/) noexcept {
		    return state(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + static_cast<int>(Species::e)) / spmasses[Species::e];
	    });

	// Sphere-equivalent radii (8x the octant) of the ionized volume, int (1 - x_HI) dV, and of the region with x_HI < 1/2
	const amrex::Real V_ionized =
	    computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state,
						       std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const & /*state_fc*/) noexcept {
		    const amrex::Real n_HI = state(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + static_cast<int>(Species::H)) / spmasses[Species::H];
		    const amrex::Real n_HII =
			state(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + static_cast<int>(Species::H_p)) / spmasses[Species::H_p];
		    return n_HII / (n_HI + n_HII);
	    });
	const amrex::Real V_half =
	    computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state,
						       std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const & /*state_fc*/) noexcept {
		    const amrex::Real n_HI = state(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + static_cast<int>(Species::H)) / spmasses[Species::H];
		    const amrex::Real n_HII =
			state(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + static_cast<int>(Species::H_p)) / spmasses[Species::H_p];
		    return (n_HI < n_HII) ? 1.0_rt : 0.0_rt;
	    });
	const amrex::Real r_effective = std::cbrt(3.0_rt * 8.0_rt * V_ionized / (4.0_rt * M_PI));
	const amrex::Real r_half = std::cbrt(3.0_rt * 8.0_rt * V_half / (4.0_rt * M_PI));
	const amrex::Real r_analytical = analytic_radius(tNew_[0], userData_.Q, userData_.n_HI_init, C::c_light);

	const amrex::Real N_injected = userData_.Q * tNew_[0] / 8.0_rt;
	const amrex::Real N_inflight = N_gamma - userData_.N_gamma0;
	const amrex::Real N_absorbed = N_e - userData_.N_e0;
	const amrex::Real rel_err = (N_injected - (N_inflight + N_absorbed)) / N_injected;
	userData_.max_photon_err = std::max(userData_.max_photon_err, std::abs(rel_err));

	userData_.t_vec_.push_back(tNew_[0]);
	userData_.r_effective_vec_.push_back(r_effective);
	userData_.r_analytical_vec_.push_back(r_analytical);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_ << tNew_[0] << ',' << r_effective << ',' << r_analytical << ',' << r_half << ',' << N_injected << ',' << N_inflight
				       << ',' << N_absorbed << ',' << rel_err << '\n';
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

	sim.evolve();

	int status = 0;
	if (sim.userData_.max_photon_err > photon_err_tol) {
		amrex::Print() << "Test failed: maximum relative photon conservation error is " << sim.userData_.max_photon_err << " (tolerance "
			       << photon_err_tol << ")\n";
		status = 1;
	} else {
		amrex::Print() << "Test passed: maximum relative photon conservation error is " << sim.userData_.max_photon_err << " (tolerance "
			       << photon_err_tol << ")\n";
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
		matplotlibcpp::save("./stromgren_sphere_photon_conservation_radii.pdf");
	}
#endif

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return status;
}
