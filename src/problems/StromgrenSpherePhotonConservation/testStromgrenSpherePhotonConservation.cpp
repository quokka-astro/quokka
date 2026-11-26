// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

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

constexpr double c_hat = C::c_light;
constexpr int kernel_radius_cells = 2; // support radius of the source's Wendland C2 kernel, as in DTypeFront_JAFF
constexpr double Erad_init = 1.0e-99;
constexpr double photon_err_tol = 1.0e-12;

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

AMREX_GPU_HOST_DEVICE auto wendland_c2(amrex::Real r) -> amrex::Real
{
	if (r > 1.0) {
		return 0.0;
	}
	return (21. / (2. * M_PI)) * std::pow((1.0 - r), 4) * (4.0 * r + 1.0);
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
	// parmparse species and temperature
	amrex::ParmParse const pp("stromgen");
	userData_.temperature = 1.0e4;
	userData_.tend = 1000.0_rt;
	userData_.n_e_init = 0.0e0_rt;
	userData_.n_HI_init = 1.0e2_rt;
	userData_.n_HII_init = 0.0e0_rt;
	userData_.Q = 1.0e49_rt;
	pp.query("temperature", userData_.temperature);
	pp.query("tend", userData_.tend);
	pp.query("n_e_init", userData_.n_e_init);
	pp.query("n_HI_init", userData_.n_HI_init);
	pp.query("n_HII_init", userData_.n_HII_init);
	pp.query("Q", userData_.Q);

	// the budget below assumes each absorbed photon creates exactly one electron and nothing else creates or removes them
	if (Network::Parameters().recombination_switch != 0 || Network::Parameters().collisional_ionization_switch != 0) {
		amrex::Abort("StromgrenSpherePhotonConservation needs network.recombination_switch = 0 and network.collisional_ionization_switch = 0");
	}

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
		for (int nn = 0; nn < Network::NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + nn) =
			    state.xn[nn] * Network::SpeciesMass()[nn]; // scalar indices carry partial densities instead of number densities
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
		    return state(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + static_cast<int>(Network::e)) / Network::SpeciesMass()[Network::e];
	    });

	// Sphere-equivalent radii of the ionized volume, int (1 - x_HI) dV, and of the region with x_HI < 1/2 (the whole sphere is in the domain)
	const amrex::Real V_ionized =
	    computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state,
						       std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const & /*state_fc*/) noexcept {
		    const amrex::Real n_HI =
			state(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + static_cast<int>(Network::H)) / Network::SpeciesMass()[Network::H];
		    const amrex::Real n_HII =
			state(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + static_cast<int>(Network::H_p)) / Network::SpeciesMass()[Network::H_p];
		    return n_HII / (n_HI + n_HII);
	    });
	const amrex::Real V_half =
	    computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state,
						       std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const & /*state_fc*/) noexcept {
		    const amrex::Real n_HI =
			state(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + static_cast<int>(Network::H)) / Network::SpeciesMass()[Network::H];
		    const amrex::Real n_HII =
			state(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + static_cast<int>(Network::H_p)) / Network::SpeciesMass()[Network::H_p];
		    return (n_HI < n_HII) ? 1.0_rt : 0.0_rt;
	    });
	const amrex::Real r_effective = std::cbrt(3.0_rt * V_ionized / (4.0_rt * M_PI));
	const amrex::Real r_half = std::cbrt(3.0_rt * V_half / (4.0_rt * M_PI));
	const amrex::Real r_analytical = analytic_radius(tNew_[0], userData_.Q, userData_.n_HI_init, C::c_light);

	const amrex::Real N_injected = userData_.Q * tNew_[0];
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
