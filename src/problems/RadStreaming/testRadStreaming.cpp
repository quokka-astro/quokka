//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testRadStreaming.cpp
/// \brief Defines a test problem for radiation in the free-streaming regime.
///

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include "AMReX.H"
#include "AMReX_ParmParse.H"
#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"
#include "util/valarray.hpp"
#include <algorithm>
#include <format>

struct StreamingProblem {
};

constexpr double initial_Erad = 1.0e-5;
constexpr double initial_Egas = 1.0e-5;
constexpr double c = 1.0;	   // speed of light
constexpr double chat = 0.2;	   // reduced speed of light
constexpr double kappa0 = 1.0e-10; // opacity
constexpr double rho = 1.0;

// Flux-source variant, selected with problem.flux_source = 1 in the input file. Instead of letting the
// beam enter through the left Dirichlet boundary, it is injected in the interior slab [beam_lo, beam_hi)
// by AddRadSource, which sets the reduced flux to f = (1, 0, 0) so that the injected radiation is fully
// beamed (|F| = c E) along +x. In steady state the beam leaving the slab has
// Erad = beam_S * (beam_hi - beam_lo) / c, so beam_S is chosen to make that equal to 1.
constexpr double beam_lo = 0.1;
constexpr double beam_hi = 0.2;
constexpr double beam_width = beam_hi - beam_lo;
constexpr double beam_S = c / beam_width; // erg s^-1 cm^-3
// managed so that setCustomBoundaryConditions can read it on the device
AMREX_GPU_MANAGED int use_flux_source = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables); set in problem_main()

template <> struct quokka::EOS_Traits<StreamingProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<StreamingProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = c;
	static constexpr double radiation_constant = 1.0;
};

template <> struct RadSystem_Traits<StreamingProblem> {
	static constexpr double c_hat_over_c = chat / c;
	static constexpr double Erad_floor = initial_Erad;
	static constexpr int beta_order = 0;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <>
void RadSystem<StreamingProblem>::AddRadSource(array_t &radEnergySource, array_t &reducedFluxSource, amrex::Box const &indexRange,
					       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real /*time*/)
{
	if (use_flux_source == 0) {
		return; // the default variant injects the beam through the left boundary instead
	}

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
		if ((x >= beam_lo) && (x < beam_hi)) {
			radEnergySource(i, j, k, 0) = beam_S;
			// a reduced flux of unit magnitude injects fully beamed (|F| = c E) radiation along +x
			reducedFluxSource(i, j, k, 0) = 1.0;
		}
	});
}

template <> void QuokkaSimulation<StreamingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = initial_Erad;
	const auto Egas0 = initial_Egas;

	// calculate radEnergyFractions
	quokka::valarray<amrex::Real, Physics_Traits<StreamingProblem>::nGroups> radEnergyFractions{};
	for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
		radEnergyFractions[g] = 1.0 / Physics_Traits<StreamingProblem>::nGroups;
	}

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<StreamingProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) =
			    Erad0 * radEnergyFractions[g];
			state_cc(i, j, k, RadSystem<StreamingProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<StreamingProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<StreamingProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
		}
		state_cc(i, j, k, RadSystem<StreamingProblem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StreamingProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<StreamingProblem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x3GasMomentum_index) = 0.;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<StreamingProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
							     int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
							     const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	// Number of variables (use Physics_Indices which correctly accounts for enabled physics)
	constexpr int nvar = Physics_Indices<StreamingProblem>::nvarTotal_cc;

	// Prepare left boundary values: streaming inflow, or the ambient state when the beam is instead
	// injected in the interior by AddRadSource, so that the source is the only radiation input
	amrex::GpuArray<amrex::Real, nvar> low_bdr_cells{};
	{
		const double Erad = (use_flux_source != 0) ? initial_Erad : 1.0;
		const double Frad = (use_flux_source != 0) ? 0.0 : c * Erad;
		for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
			const double radEnergyFraction = 1.0 / Physics_Traits<StreamingProblem>::nGroups;
			low_bdr_cells[RadSystem<StreamingProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g] = Erad * radEnergyFraction;
			low_bdr_cells[RadSystem<StreamingProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = Frad * radEnergyFraction;
			low_bdr_cells[RadSystem<StreamingProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = 0;
			low_bdr_cells[RadSystem<StreamingProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = 0;
		}
		low_bdr_cells[RadSystem<StreamingProblem>::gasEnergy_index] = initial_Egas;
		low_bdr_cells[RadSystem<StreamingProblem>::gasDensity_index] = rho;
		low_bdr_cells[RadSystem<StreamingProblem>::gasInternalEnergy_index] = initial_Egas;
		low_bdr_cells[RadSystem<StreamingProblem>::x1GasMomentum_index] = 0.;
		low_bdr_cells[RadSystem<StreamingProblem>::x2GasMomentum_index] = 0.;
		low_bdr_cells[RadSystem<StreamingProblem>::x3GasMomentum_index] = 0.;
	}

	// Prepare right boundary values (constant)
	amrex::GpuArray<amrex::Real, nvar> high_bdr_cells{};
	{
		const double Erad = initial_Erad;
		for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
			const double radEnergyFraction = 1.0 / Physics_Traits<StreamingProblem>::nGroups;
			high_bdr_cells[RadSystem<StreamingProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g] = Erad * radEnergyFraction;
			high_bdr_cells[RadSystem<StreamingProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = 0;
			high_bdr_cells[RadSystem<StreamingProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = 0;
			high_bdr_cells[RadSystem<StreamingProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = 0;
		}
		high_bdr_cells[RadSystem<StreamingProblem>::gasEnergy_index] = initial_Egas;
		high_bdr_cells[RadSystem<StreamingProblem>::gasDensity_index] = rho;
		high_bdr_cells[RadSystem<StreamingProblem>::gasInternalEnergy_index] = initial_Egas;
		high_bdr_cells[RadSystem<StreamingProblem>::x1GasMomentum_index] = 0.;
		high_bdr_cells[RadSystem<StreamingProblem>::x2GasMomentum_index] = 0.;
		high_bdr_cells[RadSystem<StreamingProblem>::x3GasMomentum_index] = 0.;
	}

	// Apply boundary conditions using helper functions (direction 0 = x-axis)
	setConstantDirichletBCLo<0>(iv, consVar, geom, low_bdr_cells);
	setConstantDirichletBCHi<0>(iv, consVar, geom, high_bdr_cells);
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
	const double CFL_number = 0.8;
	const double dt_max = 1e-2;
	const double tmax = 1.0;
	const int max_timesteps = 5000;

	// Select the injection mode: boundary inflow (default) or the reducedFluxSource hook
	{
		amrex::ParmParse const pp("problem");
		pp.query("flux_source", use_flux_source);
	}

	// Boundary conditions
	constexpr int nvars = RadSystem<StreamingProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);  // Dirichlet x1
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate x1
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<StreamingProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = tmax;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	// compute error norm
	std::vector<double> erad(nx);
	std::vector<double> erad_exact(nx);
	std::vector<double> xs(nx);
	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		if (use_flux_source != 0) {
			// Radiation injected in [beam_lo, beam_hi) with |F| = c E free-streams in +x at speed
			// chat, giving a trapezoid: a ramp 0 -> 1 across the source slab, a plateau at 1 out to
			// beam_lo + chat * t, and a ramp 1 -> 0 down to the front at beam_hi + chat * t.
			erad_exact.at(i) = std::clamp(std::min((x - beam_lo) / beam_width, (beam_hi + chat * tmax - x) / beam_width), 0.0, 1.0);
		} else {
			erad_exact.at(i) = (x <= chat * tmax) ? 1.0 : 0.0;
		}
		double erad_sim = 0.0;
		for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
			erad_sim += values.at(RadSystem<StreamingProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g)[i];
		}
		erad.at(i) = erad_sim;
	}

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm += std::abs(erad[i] - erad_exact[i]);
		sol_norm += std::abs(erad_exact[i]);
	}

	const double rel_err_norm = err_norm / sol_norm;
	const double rel_err_tol = 0.05;
	int status = 1;
	if (rel_err_norm < rel_err_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm = " << rel_err_norm << '\n';

	if (use_flux_source != 0) {
		// The source-injected radiation must be fully beamed, i.e. the reduced flux F / (c E) must be 1
		// inside the beam. Check it on the plateau, away from the ramps and the front.
		double max_dev = 0.;
		for (int i = 0; i < nx; ++i) {
			const amrex::Real x = position[i];
			if ((x < beam_hi) || (x > beam_lo + chat * tmax)) {
				continue;
			}
			const double erad_i = values.at(RadSystem<StreamingProblem>::radEnergy_index)[i];
			const double frad_i = values.at(RadSystem<StreamingProblem>::x1RadFlux_index)[i];
			max_dev = std::max(max_dev, std::abs(frad_i / (c * erad_i) - 1.0));
		}
		// if the flux source were ignored the injected radiation would be isotropic and this deviation
		// would be of order unity, so this is a sharp check despite the loose tolerance
		const double max_dev_tol = 1.0e-4;
		amrex::Print() << "Max deviation of the reduced flux F / (c E) from 1 = " << max_dev << '\n';
		if (max_dev > max_dev_tol) {
			status = 1;
		}
	}

#ifdef HAVE_PYTHON
	// Plot results
	matplotlibcpp::clf();
	matplotlibcpp::ylim(0.0, 1.1);

	std::map<std::string, std::string> erad_args;
	std::map<std::string, std::string> erad_exact_args;
	erad_args["label"] = "numerical solution";
	erad_exact_args["label"] = "exact solution";
	erad_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, erad, erad_args);
	matplotlibcpp::plot(xs, erad_exact, erad_exact_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(std::format("t = {:.4f}", sim.tNew_[0]));
	matplotlibcpp::save((use_flux_source != 0) ? "./radiation_streaming_flux_source.pdf" : "./radiation_streaming.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return status;
}
