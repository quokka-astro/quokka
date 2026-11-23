//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testWindCloud.cpp
/// \brief Defines a wind-cloud problem with radiative cooling.
///
#include "AMReX.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"

/** Wind-cloud test problem: a hot wind flowing along z past a cold, dense spherical cloud. */

using amrex::Real;

const double Twind = 3.e6;
const double Tcloud  = 1.e4;
const double rho_cloud = 0.006 * C::m_p; // g/cm^3
const double Mach = 4.0; // Mach number of the wind
const double R0 = 545 * C::parsec; // radius of the cloud
AMREX_GPU_MANAGED Real v_wind = NAN; // wind speed (z direction), set in problem_main()

struct WindCloudProblem {
};

template <> struct quokka::EOS_Traits<WindCloudProblem> {
	static constexpr double gamma = 5./3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<WindCloudProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<WindCloudProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
};

template <> void QuokkaSimulation<WindCloudProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const amrex::Array4<double> &state_cc = grid_elem.array_;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
		const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
		const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

		amrex::Real rho;	  // g/cm^3
		amrex::Real T;
		amrex::Real vz;
		double R = std::sqrt((x)*(x) + (y)*(y) + (z-R0)*(z-R0));
		if(R < R0){
			T = Tcloud;
			rho = rho_cloud; // g/cm^3
			vz = 0.0; // cloud is stationary
		}
		else{
			T = Twind;
			rho = rho_cloud * Tcloud / Twind; // g/cm^3
			vz = ::v_wind; // set in problem_main(), so it stays consistent with the inflow BC
		}
		const amrex::Real Eint = quokka::EOS<WindCloudProblem>::ComputeEintFromTgas(rho, T);

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}
		state_cc(i, j, k, HydroSystem<WindCloudProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<WindCloudProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<WindCloudProblem>::energy_index) = Eint + 0.5 * (rho * vz * vz);
		state_cc(i, j, k, HydroSystem<WindCloudProblem>::internalEnergy_index) = Eint;
	});
}


template <> void QuokkaSimulation<WindCloudProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	auto const tag = tags.arrays();

	// density-based refinement: tag cloud material
	auto const &state = state_new_cc_[lev].const_arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
	    amrex::Real const rho = state[bx](i, j, k, HydroSystem<WindCloudProblem>::density_index);

		if(rho > 0.01 * rho_cloud){
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
	amrex::Gpu::streamSynchronize();
}


// Implement User-defined diode BC
template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<WindCloudProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
                             amrex::GeometryData const & /*geom*/, const Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
                             int /*orig_comp*/)
{
    auto [i, j, k] = iv.dim3();

    const double rho_edge = rho_cloud * Tcloud / Twind; // g/cm^3
    const double x3Mom_edge = rho_edge * ::v_wind;
    const double eint_edge = quokka::EOS<WindCloudProblem>::ComputeEintFromTgas(rho_edge, Twind);
    const double etot_edge = eint_edge + 0.5 * (x3Mom_edge * x3Mom_edge) / rho_edge;

    consVar(i, j, k, HydroSystem<WindCloudProblem>::density_index) = rho_edge;
    consVar(i, j, k, HydroSystem<WindCloudProblem>::x1Momentum_index) = 0.0;
    consVar(i, j, k, HydroSystem<WindCloudProblem>::x2Momentum_index) = 0.0;
    consVar(i, j, k, HydroSystem<WindCloudProblem>::x3Momentum_index) = x3Mom_edge;
    consVar(i, j, k, HydroSystem<WindCloudProblem>::energy_index) = etot_edge;
    consVar(i, j, k, HydroSystem<WindCloudProblem>::internalEnergy_index) = eint_edge;
}


auto problem_main() -> int
{
	// boundary conditions
	constexpr int ncomp_cc = Physics_Indices<WindCloudProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);

	for (int n = 0; n < ncomp_cc; ++n) {
	for (int i = 0; i < AMREX_SPACEDIM; ++i) {
		// diode boundary conditions
		if (i == 2) {
			BCs_cc[n].setLo(i, amrex::BCType::ext_dir); // inflow
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		} else {
			BCs_cc[n].setLo(i, amrex::BCType::foextrap); // outflow
			BCs_cc[n].setHi(i, amrex::BCType::foextrap); // outflow
		}
	}
	}

	// Problem initialization
	QuokkaSimulation<WindCloudProblem> sim(BCs_cc);

	// compute wind speed (pressure equilibrium with the cloud sets the wind density)
	const Real rho_wind = rho_cloud * Tcloud / Twind; // g/cm^3
	const Real P_wind = rho_wind * Twind * C::k_B / C::m_u;
	const Real cs_wind = quokka::EOS<WindCloudProblem>::ComputeSoundSpeed(rho_wind, P_wind);
	::v_wind = Mach * cs_wind;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;
}
