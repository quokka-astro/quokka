/// \file test_radhydro_uniform_advecting.cpp
/// \brief Defines a test problem for radiation advection in a uniform medium with grey radiation.
///

#include "test_radhydro_uniform_advecting.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "physics_info.hpp"

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double kappa0 = 100.; // cm^2 g^-1
constexpr double T0 = 1.0e7;	// K (temperature)
constexpr double rho0 = 1.2;	// g cm^-3 (matter density)
constexpr double a_rad = C::a_rad;
constexpr double c = C::c_light; // speed of light (cgs)
constexpr double chat = c;
constexpr double mu = 2.33 * C::m_u;
constexpr double k_B = C::k_B;
constexpr double max_time = 1e-5;
// constexpr double v0 = 0.;      // non-advecting pulse
constexpr double v0 = 1.0e7; // advecting pulse

template <> struct quokka::EOS_Traits<PulseProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<PulseProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = chat;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = 0.0;
	static constexpr bool compute_v_over_c_terms = true;
};

template <> struct Physics_Traits<PulseProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappaPVec{};
	for (int i = 0; i < nGroups_; ++i) {
		kappaPVec[i] = kappa0;
	}
	return kappaPVec;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<PulseProblem>::ComputeEddingtonFactor(double /*f*/) -> double
{
	return (1. / 3.); // Eddington approximation
}

template <> void RadhydroSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double Erad = a_rad * std::pow(T0, 4);
	const double Egas = quokka::EOS<PulseProblem>::ComputeEintFromTgas(rho0, T0);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index) = (1. + 4. / 3. * (v0 * v0) / (c * c)) * Erad;
		state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index) = Erad;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1RadFlux_index) = 4. / 3. * v0 * Erad;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasEnergy_index) = Egas + 0.5 * rho0 * v0 * v0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1GasMomentum_index) = v0 * rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// This problem is a *linear* radiation diffusion problem, i.e.
	// parameters are chosen such that the radiation and gas temperatures
	// should be near equilibrium, and the opacity is chosen to go as
	// T^3, such that the radiation diffusion equation becomes linear in T.

	// This makes this problem a stringent test of the asymptotic-
	// preserving property of the computational method, since the
	// optical depth per cell at the peak of the temperature profile is
	// of order 10^5.

	// Problem parameters
	const int max_timesteps = 0;
	const double CFL_number = 0.8;

	const double max_dt = 1e-3;

	// Boundary conditions
	constexpr int nvars = RadSystem<PulseProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	RadhydroSimulation<PulseProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = max_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

    amrex::Print() << "i,, rho" << std::endl;
	for (int i = 0; i < nx; ++i) {
		double rho = values.at(RadSystem<PulseProblem>::gasDensity_index)[i];
        amrex::Print() << i << ",, " << rho << std::endl;
	}

	// Cleanup and exit
	return 0;
}
