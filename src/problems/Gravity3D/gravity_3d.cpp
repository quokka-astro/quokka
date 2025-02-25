/// \file gravity_3d.cpp
/// \brief Defines a test problem for self-gravity in 3D.
///

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "AMReX_REAL.H"
#include "AMReX_ccse-mpi.H"
#include "QuokkaSimulation.hpp"
#include "gravity_3d.hpp"
#include "hydro/hydro_system.hpp"
#include <algorithm>

struct BinaryOrbit {
};

template <> struct quokka::EOS_Traits<BinaryOrbit> {
	static constexpr double gamma = 1.0;	     // isothermal
	static constexpr double cs_isothermal = 3.0; //
	static constexpr double mean_molecular_weight = 1.0;
};

template <> struct HydroSystem_Traits<BinaryOrbit> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<BinaryOrbit> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = 1;			     // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <> void QuokkaSimulation<BinaryOrbit>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		double const rho = 1.0e-5; // g cm^{-3}
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::energy_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::internalEnergy_index) = 0;
	});
}

template <> void QuokkaSimulation<BinaryOrbit>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) {}

template <> void QuokkaSimulation<BinaryOrbit>::createInitialCICParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->SetVerbose(1);
	CICParticles->InitFromAsciiFile("Gravity3D.txt", nreal_extra, nullptr);
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<BinaryOrbit>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<BinaryOrbit>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<BinaryOrbit>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<BinaryOrbit>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	// Problem initialization
	QuokkaSimulation<BinaryOrbit> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity
	// sim.initDt_ = 1.0e-2;	 // s
	sim.do_cic_particles = 1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// exact solution
	const double theta = 0.5 * sim.tNew_[0];
	const double exact_x = 1.0 * std::cos(theta);
	const double exact_y = 1.0 * std::sin(theta);
	const double exact_z = 0.0;

	double position_error = 0.0;
	double position_norm = 0.0;

	int status = 0; // Initialize to success

	auto particle_data = sim.particleRegister_.getParticleDescriptor("CIC_particles")->getParticleData(0);

	if (amrex::ParallelDescriptor::IOProcessor()) {

		// assume the first particle is in the first plane quadrant
		for (const auto &data : particle_data) {
			// First 3 elements are positions (x,y,z)
			if (data[0] * exact_x > 0.0) {
				position_error += std::abs(data[0] - exact_x);
				position_error += std::abs(data[1] - exact_y);
				position_error += std::abs(data[2] - exact_z);
			} else {
				position_error += std::abs(data[0] - (-exact_x));
				position_error += std::abs(data[1] - (-exact_y));
				position_error += std::abs(data[2] - (-exact_z));
			}
			position_norm += std::abs(data[0]);
			position_norm += std::abs(data[1]);
			position_norm += std::abs(data[2]);
		}

		amrex::Print() << "Particle positions and data are: \n";
		for (const auto &data : particle_data) {
			// Print positions
			amrex::Print() << "Position: " << data[0] << ", " << data[1] << ", " << data[2];
			// Print additional data (mass, velocities)
			amrex::Print() << " | Mass: " << data[3];
			amrex::Print() << " | Velocities: " << data[4] << ", " << data[5] << ", " << data[6] << "\n";
		}
		amrex::Print() << "Exact positions are: \n" << exact_x << ", " << exact_y << ", " << exact_z << "\n";

		// compute relative error
		const double relative_error = position_error / position_norm;

		amrex::Print() << "Position error: " << position_error << "\n";
		amrex::Print() << "Position norm: " << position_norm << "\n";
		amrex::Print() << "Relative error: " << relative_error << "\n";

		const double max_err_tol = sim.tNew_[0] < 1.0 ? 0.001 : 0.05; // max error tol in cell widths
		status = 1;
		if (relative_error < max_err_tol) {
			status = 0;
			amrex::Print() << "Relative error within tolerance.\n";
		}
	}

	return status;
}
