//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testThermalConductionAniso.cpp
/// \brief Defines a test problem for anisotropic conduction
/// Problem NAME: Hot wedge
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

struct ThermalConductionAnisoProblem {};

template <> struct quokka::EOS_Traits<ThermalConductionAnisoProblem> {
	static constexpr double gamma = 2.0;
	// C::m_u here gives a dimensionless mu = mean_molecular_weight / C::m_u = 1 in the EOS call,
	// independent of the (dimensionless) unit_system -- not a CGS mass in grams.
	static constexpr double mean_molecular_weight = 1.0;
};

template <> struct HydroSystem_Traits<ThermalConductionAnisoProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ThermalConductionAnisoProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
	// dimensionless problem: rho, length, and time carry no fixed physical scale, but
	// boltzmann_constant is kept at its physical CGS value so that Tgas is genuinely in kelvin.
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeMagneticVectorPotential_z(amrex::Real x1, amrex::Real x2) -> amrex::Real
{
	constexpr amrex::Real r_core = 0.1;
	constexpr amrex::Real B0 = 1.0e-3;
	const amrex::Real rad = std::sqrt(x1 * x1 + x2 * x2);
	if (rad < r_core) {
		// Quadratic ("solid-body rotation") core: C^1-matched to -min(r,1) at r=r_core, so
		// B = |dpsi/dr| ramps linearly from 0 at r=0 up to 1 at r=r_core, instead of jumping
		// straight to |B|=1 with an undefined direction at the origin. Since psi is now a plain
		// polynomial in x1,x2 near r=0 (no sqrt), the discrete curl is essentially exact there --
		// no more curvature blowup for a finite-difference stencil to trip over.
		return B0 * (-0.5 * rad * rad / r_core - 0.5 * r_core);
	}
	return -B0 * std::min(rad, 1.0);
}


template <> void QuokkaSimulation<ThermalConductionAnisoProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Real rho_bg = 1.0;	    // dimensionless background density
	const amrex::Real T_bg = 10.0;
	const amrex::Real T_wedge = 12.0;
	// true: wedge density = rho_bg * T_bg / T_wedge, so P = rho*T is uniform (no initial pressure jump).
	// false: uniform density, so the hot wedge starts overpressured and drives a flow.
	constexpr bool isobaric_wedge = false;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
		const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
		const amrex::Real rad = std::sqrt(x * x + y * y);
		amrex::Real theta = std::atan2(y, x);
		if (theta < 0.0){ theta += 2.0 * M_PI; }
		amrex::Real temp = T_bg;
		amrex::Real rho = rho_bg;
		if(rad > 0.5 & rad < 0.7 & theta > 11.* M_PI/12.0 & theta < 13.* M_PI/12.0) {
			temp = T_wedge;
			if (isobaric_wedge) {
				rho = rho_bg * T_bg / T_wedge; // P = rho*T (mu = 1) uniform
			}
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
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0; // fill unused quantities with zeros
		}
		// x1_L, x2_L are already the corner (nodal) position appropriate to `dir`: for dir==x,
		// i is nodal so x1_L is the exact x-face position; for dir==y, j is nodal so x2_L is
		// the exact y-face position. The other (cell-centered) index gives the lower corner of
		// that cell. No half-cell offset is needed -- see BxFace/ByFace in testDustyOrszagTang.cpp
		// for the same curl-from-vector-potential pattern.
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
	constexpr double max_time = 200.0;

	// Setup boundary conditions
	auto BCs_cc = quokka::BC<ThermalConductionAnisoProblem>(quokka::BCType::reflecting);
	const int nvars_fc = Physics_Indices<ThermalConductionAnisoProblem>::nvarTotal_fc;
	const int nvars_per_dim_fc = Physics_Indices<ThermalConductionAnisoProblem>::nvarPerDim_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		int const component_dir = (nvars_per_dim_fc > 0) ? (icomp / nvars_per_dim_fc) : 0;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			int const bc_type = (component_dir == idim) ? amrex::BCType::reflect_even : amrex::BCType::reflect_odd;
			BCs_fc[icomp].setLo(idim, bc_type);
			BCs_fc[icomp].setHi(idim, bc_type);
		}
	}

	QuokkaSimulation<ThermalConductionAnisoProblem> sim(BCs_cc, BCs_fc);

	sim.cflNumber_ = 0.3;
	sim.stopTime_ = max_time;

	// set initial conditions
	sim.setInitialConditions();

	sim.evolve();

	amrex::Print() << "\n✓ Thermal conduction (anisotropic) ring test completed\n";
	return 0;
}
