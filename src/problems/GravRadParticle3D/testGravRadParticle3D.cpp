/// \file testGravRadParticle3D.cpp
/// \brief Defines a 3D test problem for radiating particles with gravity.
///

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <format>
#include <limits>

#include "AMReX.H"
#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Box.H"
#include "AMReX_Vector.H"
#include "QuokkaSimulation.hpp"
#include "hydro/EOS.hpp"
#include "io/projection.hpp"
#include "particles/PhysicsParticles.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"

struct ParticleProblem {
};

constexpr int nGroups_ = 1;

constexpr double erad_floor = 1.0e-15;
constexpr double initial_Erad = erad_floor;
constexpr double initial_Egas = 1.0e-5;
constexpr double c = 100.0;	   // speed of light
constexpr double chat = 2.0;	   // reduced speed of light
constexpr double kappa0 = 1.0e-20; // opacity
constexpr double rho0 = 1.0e-8;
constexpr double m_H = C::m_p + C::m_e;

const double lum1 = 1.0;

template <> struct quokka::EOS_Traits<ParticleProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Particle_Traits<ParticleProblem> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC | ParticleSwitch::Rad | ParticleSwitch::CICRad;
};

template <> struct Physics_Traits<ParticleProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	// face-centred
	static constexpr int nGroups = nGroups_; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = c;
	static constexpr double radiation_constant = 1.0;
};

template <> struct RadSystem_Traits<ParticleProblem> {
	static constexpr double c_hat_over_c = chat / c;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 0;
};

template <> void QuokkaSimulation<ParticleProblem>::createInitialCICRadParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 6 + nGroups_; // mass vx vy vz birth_time death_time lum1
	CICRadParticles->SetVerbose(1);
	CICRadParticles->InitFromAsciiFile("../inputs/GravRadParticles3D.txt", nreal_extra, nullptr);
}

template <> void QuokkaSimulation<ParticleProblem>::createInitialCICParticles()
{
	// read particles from ASCII file - same as CICRadParticles but only mass and velocity components
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->SetVerbose(1);
	CICParticles->InitFromAsciiFile("../inputs/GravRadParticles3D_cic_only.txt", nreal_extra, nullptr);
}

template <> void QuokkaSimulation<ParticleProblem>::createInitialRadParticles()
{
	// read particles from ASCII file - same as CICRadParticles but only birth_time, death_time, and luminosity
	const int nreal_extra = 2 + nGroups_; // birth_time death_time lum1
	RadParticles->SetVerbose(1);
	RadParticles->InitFromAsciiFile("../inputs/GravRadParticles3D_rad_only.txt", nreal_extra, nullptr);
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> void QuokkaSimulation<ParticleProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = initial_Erad;
	const auto Egas0 = initial_Egas;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, RadSystem<ParticleProblem>::radEnergy_index) = Erad0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<ParticleProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<ParticleProblem>::x3GasMomentum_index) = 0.;
	});
}

namespace
{
auto checkGasDensityProjection(const QuokkaSimulation<ParticleProblem> &sim) -> int
{
	const amrex::Vector<const amrex::MultiFab *> state_mfs = amrex::GetVecOfConstPtrs(sim.state_new_cc_);
	amrex::Vector<amrex::IntVect> max_grid_size(sim.finestLevel() + 1);
	for (int lev = 0; lev <= sim.finestLevel(); ++lev) {
		max_grid_size[lev] = sim.maxGridSize(lev);
	}

	constexpr std::array dirs = {amrex::Direction::x, amrex::Direction::y, amrex::Direction::z};
	const amrex::Box &domain = sim.Geom(0).Domain();

	for (const auto dir : dirs) {
		const auto projections = quokka::diagnostics::ComputePlaneProjectionFromMultiFab(
		    state_mfs, sim.finestLevel(), sim.Geom(), sim.refRatio(), max_grid_size, dir, RadSystem<ParticleProblem>::gasDensity_index);
		const amrex::MultiFab &projection = projections.front();
		const amrex::Real projection_min = projection.min(0);
		const amrex::Real projection_max = projection.max(0);
		const amrex::Real projection_sum = projection.sum(0);

		const int dir_index = static_cast<int>(dir);
		amrex::Long n_plane_cells = 1;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (idim != dir_index) {
				n_plane_cells *= static_cast<amrex::Long>(domain.length(idim));
			}
		}

		const amrex::Real expected_projection = rho0 * (sim.Geom(0).ProbHi(dir_index) - sim.Geom(0).ProbLo(dir_index));
		const amrex::Real expected_projection_sum = expected_projection * static_cast<amrex::Real>(n_plane_cells);
		const amrex::Real projection_tol = 64.0 * std::numeric_limits<amrex::Real>::epsilon();
		const amrex::Real projection_sum_tol = projection_tol * static_cast<amrex::Real>(n_plane_cells);
		const char dir_name = "xyz"[dir_index];

		amrex::Print() << "Projected gasDensity statistics along " << dir_name << ": min=" << projection_min << " max=" << projection_max
			       << " sum=" << projection_sum << "\n";
		amrex::Print() << "Expected gasDensity projection along " << dir_name << ": value=" << expected_projection << " sum=" << expected_projection_sum
			       << "\n";

		if (std::abs(projection_min - expected_projection) > projection_tol || std::abs(projection_max - expected_projection) > projection_tol ||
		    std::abs(projection_sum - expected_projection_sum) > projection_sum_tol) {
			amrex::Print() << std::format(
			    "Projection check FAILED along {}: expected a uniform gasDensity projection with value {:.17g} and sum {:.17g}, but got "
			    "min={:.17g}, max={:.17g}, sum={:.17g}.\n",
			    dir_name, expected_projection, expected_projection_sum, projection_min, projection_max, projection_sum);
			return 1;
		}
	}

	return 0;
}
} // namespace

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;

	// Problem initialization
	QuokkaSimulation<ParticleProblem> sim;

	sim.radiationReconstructionOrder_ = 3; // PPM

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	int status = 0;
	const int projection_status = checkGasDensityProjection(sim);

	// compute total radiation energy
	const double total_Erad_over_vol = sim.state_new_cc_[0].sum(RadSystem<ParticleProblem>::radEnergy_index);
	const double dx = sim.Geom(0).CellSize(0);
	const double dy = sim.Geom(0).CellSize(1);
	const double dz = sim.Geom(0).CellSize(2);
	const double dvol = dx * dy * dz;
	const double total_Erad = total_Erad_over_vol * dvol;
	const double t_sim = sim.tNew_[0];
	const double t_alive = std::min(0.5, t_sim);		     // particles only live for 0.5 time units
	double total_Erad_exact = 2.0 * lum1 * t_alive * (chat / c); // two particles with luminosity lum1
	total_Erad_exact *= 2.0;				     // two particle system (Rad + CICRad)
	const auto total_num_of_cells = sim.Geom(0).Domain().volume();
	total_Erad_exact += static_cast<double>(total_num_of_cells) * dvol * initial_Erad;

	const double rel_err = std::abs(total_Erad - total_Erad_exact) / total_Erad_exact;

	// Compute exact location of the CICRad particles
	// The particles are originally at (-0.5, 0) and (0.5, 0) and they move with
	// velocity 1/sqrt(2) in the y/-y direction. The problem is designed such that
	// the particles will move in a circle with radius 0.5
	const double velocity = 0.5;
	const double radius = 1.0;
	const double theta = velocity * t_sim / radius;
	const double exact_x = radius * std::cos(theta);
	const double exact_y = radius * std::sin(theta);
	const double exact_z = 0.0;

	// Exact location of the CIC particles
	const double exact_x_cic = 0.0;
	const double exact_y_cic = 0.0;
	const double exact_z_cic = 0.0;

	// Exact location of the Rad particles
	const double exact_x_rad = 0.3;
	const double exact_y_rad = 0.0;
	const double exact_z_rad = 0.0;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// Test CICRad particles
		[[maybe_unused]] const auto [ids1, positions_cicrad, int1] =
		    sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CICRad)->getParticleDataAtAllLevels();
		double position_error_cicrad = 0.0;
		double position_norm_cicrad = 0.0;

		// Test CIC particles
		[[maybe_unused]] const auto [ids2, positions_cic, int2] =
		    sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->getParticleDataAtAllLevels();
		double position_error_cic = 0.0;
		const double position_norm_cic = 1.0; // set to 1.0 since the particles are exactly at the origin

		// Test Rad particles
		[[maybe_unused]] const auto [ids3, positions_rad, int3] =
		    sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Rad)->getParticleDataAtAllLevels();
		double position_error_rad = 0.0;
		double position_norm_rad = 0.0;

		// Test both particle types against exact solution
		for (const auto &position : positions_cicrad) {
			if (position[0] * exact_x > 0.0) {
				position_error_cicrad += std::abs(position[0] - exact_x);
				position_error_cicrad += std::abs(position[1] - exact_y);
				position_error_cicrad += std::abs(position[2] - exact_z);
			} else {
				position_error_cicrad += std::abs(position[0] - (-exact_x));
				position_error_cicrad += std::abs(position[1] - (-exact_y));
				position_error_cicrad += std::abs(position[2] - (-exact_z));
			}
			position_norm_cicrad += std::abs(exact_x);
			position_norm_cicrad += std::abs(exact_y);
			position_norm_cicrad += std::abs(exact_z);
		}

		for (const auto &position : positions_cic) {
			if (position[0] * exact_x > 0.0) {
				position_error_cic += std::abs(position[0] - exact_x_cic);
				position_error_cic += std::abs(position[1] - exact_y_cic);
				position_error_cic += std::abs(position[2] - exact_z_cic);
			} else {
				position_error_cic += std::abs(position[0] - (-exact_x_cic));
				position_error_cic += std::abs(position[1] - (-exact_y_cic));
				position_error_cic += std::abs(position[2] - (-exact_z_cic));
			}
		}

		for (const auto &position : positions_rad) {
			if (position[0] * exact_x_rad > 0.0) {
				position_error_rad += std::abs(position[0] - exact_x_rad);
				position_error_rad += std::abs(position[1] - exact_y_rad);
				position_error_rad += std::abs(position[2] - exact_z_rad);
			} else {
				position_error_rad += std::abs(position[0] - (-exact_x_rad));
				position_error_rad += std::abs(position[1] - (-exact_y_rad));
				position_error_rad += std::abs(position[2] - (-exact_z_rad));
			}
			position_norm_rad += std::abs(exact_x_rad);
			position_norm_rad += std::abs(exact_y_rad);
			position_norm_rad += std::abs(exact_z_rad);
		}

		const double rel_position_error_cicrad = position_error_cicrad / position_norm_cicrad;
		const double rel_position_error_cic = position_error_cic / position_norm_cic;
		const double rel_position_error_rad = position_error_rad / position_norm_rad;

		const double rel_err_tol = 1.0e-7;
		const double rel_position_error_tol = t_sim < 1.0 ? 2.0e-4 : 2.0e-3;
		status = 1;
		if (projection_status == 0 && rel_err < rel_err_tol && rel_position_error_cicrad < rel_position_error_tol &&
		    rel_position_error_cic < rel_position_error_tol && rel_position_error_rad < rel_position_error_tol) {
			status = 0;
			amrex::Print() << "Relative error within tolerance.\n";
		}

		amrex::Print() << "Exact positions of the CICRad particles should be: " << exact_x << ", " << exact_y << ", " << exact_z << "\n";
		amrex::Print() << "Real positions are: \n";
		for (const auto &position : positions_cicrad) {
			amrex::Print() << position[0] << ", " << position[1] << ", " << position[2] << "\n";
		}
		amrex::Print() << "Exact positions of the CIC particles should be: " << exact_x_cic << ", " << exact_y_cic << ", " << exact_z_cic << "\n";
		amrex::Print() << "Real positions are: \n";
		for (const auto &position : positions_cic) {
			amrex::Print() << position[0] << ", " << position[1] << ", " << position[2] << "\n";
		}
		amrex::Print() << "Exact positions of the Rad particles should be: " << exact_x_rad << ", " << exact_y_rad << ", " << exact_z_rad << "\n";
		amrex::Print() << "Real positions are: \n";
		for (const auto &position : positions_rad) {
			amrex::Print() << position[0] << ", " << position[1] << ", " << position[2] << "\n";
		}
		amrex::Print() << "Relative L1 norm on radiation energy = " << rel_err << "\n";
		amrex::Print() << "Relative L1 norm on CICRad particle positions = " << rel_position_error_cicrad << "\n";
		amrex::Print() << "Relative L1 norm on CIC particle positions = " << rel_position_error_cic << "\n";
		amrex::Print() << "Relative L1 norm on Rad particle positions = " << rel_position_error_rad << "\n";

		// Cleanup and exit
		amrex::Print() << "Finished."
			       << "\n";
	}

	return status;
}
