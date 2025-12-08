#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "util/BC.hpp"

#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include <cmath>

struct TurbulentBox {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct Physics_Traits<TurbulentBox> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 1;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct quokka::EOS_Traits<TurbulentBox> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0; // dimensionless
};

template <> struct HydroSystem_Traits<TurbulentBox> {
	static constexpr bool reconstruct_eint = false;
};

template <> void QuokkaSimulation<TurbulentBox>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<TurbulentBox>::density_index) = 1.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::energy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::internalEnergy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::scalar0_index) = 1.0;
	});
}

template <> void QuokkaSimulation<TurbulentBox>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.2; // gradient refinement threshold
	const amrex::Real rho_min = 0.1;       // minimum density for refinement

	const auto state = state_new_cc_[lev].const_arrays();
	const auto tag = tags.arrays();

	amrex::ParallelFor(state_new_cc_[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const int n = HydroSystem<TurbulentBox>::density_index;
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
	auto BCs_cc = quokka::BC<TurbulentBox>(quokka::BCType::int_dir,	 // x: periodic
					       quokka::BCType::int_dir,	 // y: periodic
					       quokka::BCType::int_dir); // z: periodic

	QuokkaSimulation<TurbulentBox> sim(BCs_cc);

	sim.setInitialConditions();

	// Main time loop
	sim.evolve();
	return 0;
}
