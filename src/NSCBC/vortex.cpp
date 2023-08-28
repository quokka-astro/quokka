//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file vortex.cpp
/// \brief Implements a subsonic vortex flow problem with Navier-Stokes
///        Characteristic Boundary Conditions (NSCBC).
///
#include <random>
#include <tuple>
#include <vector>

#include "AMReX.H"
#include "AMReX_Array.H"
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
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"

#include "ArrayUtil.hpp"
#include "EOS.hpp"
#include "HydroState.hpp"
#include "NSCBC_inflow.hpp"
#include "NSCBC_outflow.hpp"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "fundamental_constants.H"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "valarray.hpp"
#include "vortex.hpp"

using amrex::Real;

struct Vortex {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct quokka::EOS_Traits<Vortex> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = 28.96 * C::m_u; // air
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<Vortex> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 1; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
};

// global variables needed for Dirichlet boundary condition and initial conditions
namespace
{
Real G_vortex = NAN;							     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real T_ref = NAN;					     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real P_ref = NAN;					     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real u_inflow = NAN;					     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real v_inflow = NAN;					     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real w_inflow = NAN;					     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED GpuArray<Real, HydroSystem<Vortex>::nscalars_> s_inflow{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
};									     // namespace

template <> void RadhydroSimulation<Vortex>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;

	constexpr Real gamma = quokka::EOS_Traits<Vortex>::gamma;
	constexpr Real R = quokka::EOS_Traits<Vortex>::boltzmann_constant / quokka::EOS_Traits<Vortex>::mean_molecular_weight;
	const Real c = std::sqrt(gamma * R * T_ref);

	const Real G = ::G_vortex;
	const Real R_v = 0.1 * (prob_hi[0] - prob_lo[0]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];

		Real const P = P_ref + -0.5 * std::pow(G, 2) * P_ref * gamma * std::exp((-std::pow(x, 2) - std::pow(y, 2)) / std::pow(R_v, 2)) /
					   (std::pow(R_v, 2) * std::pow(c, 2));

		Real const rho = P / (R * T_ref);
		Real const u = u_inflow + -G * y * std::exp(-1.0 / 2.0 * (std::pow(x, 2) + std::pow(y, 2)) / std::pow(R_v, 2)) / std::pow(R_v, 2);
		Real const v = v_inflow + G * x * std::exp(-1.0 / 2.0 * (std::pow(x, 2) + std::pow(y, 2)) / std::pow(R_v, 2)) / std::pow(R_v, 2);
		Real const w = w_inflow;

		Real const xmom = rho * u;
		Real const ymom = rho * v;
		Real const zmom = rho * w;
		Real const Eint = quokka::EOS<Vortex>::ComputeEintFromPres(rho, P);
		Real const Egas = RadSystem<Vortex>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);
		Real const scalar = ::s_inflow[0];

		state_cc(i, j, k, HydroSystem<Vortex>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<Vortex>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<Vortex>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<Vortex>::x3Momentum_index) = zmom;
		state_cc(i, j, k, HydroSystem<Vortex>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<Vortex>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<Vortex>::scalar0_index) = scalar;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<Vortex>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
											    int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
											    const Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
											    int /*orig_comp*/)
{
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	const auto &domain_lo = box.loVect3d();
	const auto &domain_hi = box.hiVect3d();
	const int ilo = domain_lo[0];
	const int ihi = domain_hi[0];

	if (i < ilo) {
		NSCBC::setInflowX1Lower<Vortex>(iv, consVar, geom, ::T_ref, ::u_inflow, ::v_inflow, ::w_inflow, ::s_inflow);
	} else if (i > ihi) {
		NSCBC::setOutflowBoundary<Vortex, FluxDir::X1, NSCBC::BoundarySide::Upper>(iv, consVar, geom, ::P_ref);
	}
}

auto problem_main() -> int
{
	// Problem initialization
	constexpr int ncomp_cc = Physics_Indices<Vortex>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir); // NSCBC inflow
		BCs_cc[n].setHi(0, amrex::BCType::ext_dir); // NSCBC outflow

		if constexpr (AMREX_SPACEDIM >= 2) {
			BCs_cc[n].setLo(1, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(1, amrex::BCType::int_dir);
		} else if (AMREX_SPACEDIM == 3) {
			BCs_cc[n].setLo(2, amrex::BCType::int_dir);
			BCs_cc[n].setHi(2, amrex::BCType::int_dir);
		}
	}

	RadhydroSimulation<Vortex> sim(BCs_cc);

	amrex::ParmParse const pp("vortex");
	// initial condition parameters
	pp.query("strength", ::G_vortex); // vortex strength
	pp.query("Tgas0", ::T_ref);	  // initial temperature [K]
	pp.query("P0", ::P_ref);	  // initial pressure [erg cm^-3]
	// boundary condition parameters
	pp.query("u_inflow", ::u_inflow);    // inflow velocity along x-axis [cm/s]
	pp.query("v_inflow", ::v_inflow);    // transverse inflow velocity (v_y) [cm/s]
	pp.query("w_inflow", ::w_inflow);    // transverse inflow velocity (v_z) [cm/s]
	pp.query("s_inflow", ::s_inflow[0]); // inflow passive scalar [dimensionless]

	// Set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	const int status = 0;
	return status;
}
