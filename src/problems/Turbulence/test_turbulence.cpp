#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "util/BC.hpp"

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"
#include "AMReX_iMultiFab.H"
#include <cmath>

struct BasicTurbulence {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct Physics_Traits<BasicTurbulence> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr bool is_self_gravity_enabled = false;

	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 1;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct quokka::EOS_Traits<BasicTurbulence> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0; // dimensionless
						     // static constexpr double mean_molecular_weight = C::m_u;
						     // static constexpr double boltzmann_constant = C::k_B;
};

template <> struct HydroSystem_Traits<BasicTurbulence> {
	static constexpr bool reconstruct_eint = false;
};

template <> void QuokkaSimulation<BasicTurbulence>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const rho = 1.0;
		amrex::Real const xmom = 0.0;
		amrex::Real const ymom = 0.0;
		amrex::Real const zmom = 0.0;
		amrex::Real const Eint = 0.0; // P0 / (gamma - 1.0);
		amrex::Real const Egas = Eint;
		amrex::Real const scalar_density = 0.0;

		state_cc(i, j, k, HydroSystem<BasicTurbulence>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<BasicTurbulence>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<BasicTurbulence>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<BasicTurbulence>::x3Momentum_index) = zmom;
		state_cc(i, j, k, HydroSystem<BasicTurbulence>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<BasicTurbulence>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<BasicTurbulence>::scalar0_index) = scalar_density;
	});
}

template <> void QuokkaSimulation<BasicTurbulence>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.2; // gradient refinement threshold
	const amrex::Real rho_min = 0.1;       // minimum density for refinement

	const auto state = state_new_cc_[lev].const_arrays();
	const auto tag = tags.arrays();

	amrex::ParallelFor(state_new_cc_[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const int n = HydroSystem<BasicTurbulence>::density_index;
		amrex::Real const rho = state[bx](i, j, k, n);
		amrex::Real const rho_xplus = state[bx](i + 1, j, k, n);
		amrex::Real const rho_xminus = state[bx](i - 1, j, k, n);
		amrex::Real const rho_yplus = state[bx](i, j + 1, k, n);
		amrex::Real const rho_yminus = state[bx](i, j - 1, k, n);
		amrex::Real const rho_zplus = state[bx](i, j, k + 1, n);
		amrex::Real const rho_zminus = state[bx](i, j, k - 1, n);

		amrex::Real const del_x = 0.5 * (rho_xplus - rho_xminus);
		amrex::Real const del_y = 0.5 * (rho_yplus - rho_yminus);
		amrex::Real const del_z = 0.5 * (rho_zplus - rho_zminus);

		amrex::Real const gradient_indicator = std::sqrt(del_x * del_x + del_y * del_y + del_z * del_z) / rho;

		if ((gradient_indicator > eta_threshold) && (rho > rho_min)) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
	amrex::Gpu::streamSynchronize();
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<BasicTurbulence>(quokka::BCType::int_dir,  // x: reflecting
						  quokka::BCType::int_dir,  // y: reflecting
						  quokka::BCType::int_dir); // z: reflecting

	QuokkaSimulation<BasicTurbulence> sim(BCs_cc);

	sim.setInitialConditions();

	// Main time loop
	sim.evolve();
	return 0;
}
