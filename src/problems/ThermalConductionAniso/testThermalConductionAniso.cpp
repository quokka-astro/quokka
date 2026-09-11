//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testThermalConductionConstant.cpp
/// \brief Defines a test problem for constant-conductivity thermal conduction (kappa = const).
///
#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "hydro/hydro_system.hpp"
#include <cmath>

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"

template <> struct quokka::EOS_Traits<ThermalConductionAnisoProblem> {
	static constexpr double gamma = 2.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<ThermalConductionAnisoProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ThermalConductionAnisoProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

// vector potential psi = -r (only a z-component, independent of z), so that the discrete curl
// gives Bx = dpsi/dy = -y/rad and By = -dpsi/dx = x/rad exactly.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeMagneticVectorPotential_z(amrex::Real x1, amrex::Real x2) -> amrex::Real
{
	return -std::sqrt(x1 * x1 + x2 * x2);
}

template <> void QuokkaSimulation<ThermalConductionAnisoProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Real rho = 1.0;	    // g/cm^3
	
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + i * dx[0];
		const amrex::Real y = prob_lo[1] + j * dx[1];
		const amrex::Real rad = std::sqrt(x * x + y * y);
		const amrex::Real theta = std::atan2(y, x);
		amrex::Real temp = 10.0;
		if(rad > 0.5 & rad < 0.7 & theta > 11.* M_PI/12.0 & theta < 13.* M_PI/12.0) {
			temp = 12.0;
		}
		const amrex::Real Eint = quokka::EOS<ThermalConductionAnisoProblem>::ComputeEintFromTgas(rho, temp);

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}

		state_cc(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::energy_index) = Eint;
		state_cc(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::internalEnergy_index) = Eint;
	});
}

template <> void QuokkaSimulation<ThermalConductionAnisoProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<ThermalConductionAnisoProblem>::nvarPerDim_fc;
	// loop over the grid and set the initial condition: a purely azimuthal unit field derived
	// from the vector potential psi = -rad, via the same discrete curl used for constrained
	// transport (Bx = dpsi/dy, By = -dpsi/dx), so the resulting face field is exactly
	// divergence-free on the mesh.
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0; // fill unused quantities with zeros
		}
		const amrex::Real x1_L = prob_lo[0] + i * dx[0];
		const amrex::Real x2_L = prob_lo[1] + j * dx[1];

		amrex::Real bval = 0.0;
		if (dir == quokka::direction::x) {
			bval = (computeMagneticVectorPotential_z(x1_L, x2_L + dx[1]) - computeMagneticVectorPotential_z(x1_L, x2_L)) / dx[1];
		} else if (dir == quokka::direction::y) {
			bval = -(computeMagneticVectorPotential_z(x1_L + dx[0], x2_L) - computeMagneticVectorPotential_z(x1_L, x2_L)) / dx[0];
		}
		// dir == z: Bz = 0 (psi is independent of z), already zero-filled above
		state_fc(i, j, k, MHDSystem<ThermalConductionAnisoProblem>::bfield_index) = bval;
	});
}

template <>
void QuokkaSimulation<ThermalConductionAnisoProblem>::ComputeDerivedVar(int /*lev*/, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
									   amrex::MultiFab const &state_cc,
									   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) const
{
	if (dname == "temperature") {
		const int ncomp = ncomp_cc_in;
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_cc.const_array(iter);
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
			    AMREX_D_DECL(state_fc[0].const_array(iter), state_fc[1].const_array(iter), state_fc[2].const_array(iter))};
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::density_index);
				Real const Eint = HydroSystem<ThermalConductionAnisoProblem>::ComputeInternalEnergy(state, i, j, k, &cons_fc);
				Real const Tgas = quokka::EOS<ThermalConductionAnisoProblem>::ComputeTgasFromEint(rho, Eint);
				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

auto problem_main() -> int
{
	// Single-resolution run of the ring conduction test (no AMR, no reference solution).
	constexpr int nx = 64;
	constexpr double max_time = 200.0;

	amrex::ParmParse pp("amr");
	pp.add("max_level", 0);
#if AMREX_SPACEDIM == 3
	amrex::Vector<int> const ncells = {nx, nx, 8};
	pp.add("blocking_factor_z", 8);
#else
	amrex::Vector<int> const ncells = {nx, nx};
#endif
	pp.addarr("n_cell", ncells);

	// Set domain bounds using AMReX parameter system
	amrex::ParmParse pp_geom("geometry");
	amrex::Vector<double> const prob_lo = {-1.0, -1.0, -1.0};
	amrex::Vector<double> const prob_hi = {1.0, 1.0, 1.0};
	amrex::Vector<int> const is_periodic = {0, 0, 0};
	pp_geom.addarr("prob_lo", prob_lo);
	pp_geom.addarr("prob_hi", prob_hi);
	pp_geom.addarr("is_periodic", is_periodic);

	// Setup boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionAnisoProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			BCs_cc[n].setLo(dir, amrex::BCType::foextrap);
			BCs_cc[n].setHi(dir, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	QuokkaSimulation<ThermalConductionAnisoProblem> sim(BCs_cc);

	sim.cflNumber_ = 0.3;
	sim.stopTime_ = max_time;

	// set initial conditions
	sim.setInitialConditions();

	sim.evolve();

	amrex::Print() << "\n✓ Thermal conduction (anisotropic) ring test completed\n";
	return 0;
}
