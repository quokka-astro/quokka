//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testThermalConduction.cpp
/// \brief Defines a test problem for thermal conduction.
///
#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <fstream>

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"
#include "util/richardson.hpp"

/** Thermal conduction test problem
The initial condition for the test problem for running a wind-cloud problem. */



using amrex::Real;

constexpr double seconds_in_year = 3.1536e7;

const double Twind = 3.e6;
const double Tcloud  = 1.e4;
const double rho_cloud = 0.006 * C::m_p; // g/cm^3
AMREX_GPU_MANAGED double Mach = 4.0; // Mach number of the wind; overridden via ParmParse in problem_main()
const double R0 = 545 * C::parsec; // radius of the cloud
const double TracerPerVolume = 1.e3; // tracer content per volume

// frame-tracking globals (set inside problem_main() / computeAfterTimestep())
bool do_frame_shift = true;			      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real v_wind = NAN;		      // wind speed (z direction)
AMREX_GPU_MANAGED Real cloud_crushing_time = NAN;    // t_cc, estimated from R0 and v_wind
AMREX_GPU_MANAGED Real delta_vz = 0;		      // cumulative center-of-mass frame velocity offset

struct ThermalConductionProblem {
};

template <> struct quokka::EOS_Traits<ThermalConductionProblem> {
	static constexpr double gamma = 5./3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<ThermalConductionProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ThermalConductionProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 2; // cloud tracer + wind tracer
};

template <> void QuokkaSimulation<ThermalConductionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// initialize a ThermalConduction test problem using parameters from

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
		amrex::Real cs_wind;
		amrex::Real cloudTracer;
		amrex::Real windTracer;
		const amrex::Real cellVolume = dx[0] * dx[1] * dx[2];
		double R = std::sqrt((x)*(x) + (y)*(y) + (z-R0)*(z-R0));
		if(R < R0){
			T = Tcloud;
			rho = rho_cloud; // g/cm^3
			vz = 0.0; // cloud is stationary
			cloudTracer = TracerPerVolume; // 1/vol, so each cell contributes TracerPerCell regardless of resolution
			windTracer = 1.e-6 *  TracerPerVolume; ; // outside the wind
		}
		else{
			T = Twind;
			cloudTracer = 1.e-6 *  TracerPerVolume ; // outside the cloud
			windTracer = TracerPerVolume; // 1/vol, so each cell contributes TracerPerCell regardless of resolution
			rho = rho_cloud * Tcloud / Twind; // g/cm^3
			amrex::Real pressure = rho * T * C::k_B / C::m_u;
			cs_wind = quokka::EOS<ThermalConductionProblem>::ComputeSoundSpeed(rho, pressure);
			vz = ::v_wind; // set in problem_main(), so it stays consistent with the frame-shift BC
		}
		const amrex::Real Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, T);
		/*-------------------------------------------------*/

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}
		if(i==0 & j==0 & k==0 ){
			amrex::Print() << "Initial conditions at the center of the domain: " << std::endl;
			amrex::Print() << "Density: " << rho << std::endl;
			amrex::Print() << "Temperature: " << T << std::endl;
			amrex::Print() << "Internal Energy: " << Eint << std::endl;
			amrex::Print() << "cs: " << cs_wind << ", vz:" << vz << std::endl;
		}
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = Eint + 0.5 * (rho * vz * vz);
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::scalar0_index) = cloudTracer; // 1/vol
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::scalar0_index + 1) = windTracer; // 1/vol
	});
}


// template <> void QuokkaSimulation<ThermalConductionProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
// {
// 	const amrex::Array4<double> &state_fc = grid_elem.array_;
// 	const amrex::Box &indexRange = grid_elem.indexRange_;
// 	const quokka::direction dir = grid_elem.dir_;

// 	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
// 		constexpr double bx = 0.0;
// 		constexpr double by = 0.0;
// 		constexpr double bz = 1.;

// 		if (dir == quokka::direction::x) {
// 			state_fc(i, j, k, Physics_Indices<ThermalConductionProblem>::mhdFirstIndex) = bx;
// 		} else if (dir == quokka::direction::y) {
// 			state_fc(i, j, k, Physics_Indices<ThermalConductionProblem>::mhdFirstIndex) = by;
// 		} else if (dir == quokka::direction::z) {
// 			state_fc(i, j, k, Physics_Indices<ThermalConductionProblem>::mhdFirstIndex) = bz;
// 		}
// 	});
// }


template <> void QuokkaSimulation<ThermalConductionProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tracer-based refinement: tag cells that are less than 50% cloud AND less than 50% wind,
	// i.e. cells in the cloud-wind mixing/interface region
	const auto dx = geom[lev].CellSizeArray();
	const amrex::Real cellVolume = dx[0] * dx[1] * dx[2];
	const amrex::Real refine_threshold = 0.5 * TracerPerVolume;

	auto const &state = state_new_cc_[lev].const_arrays();
	auto const tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Real const cloudTracer = state[bx](i, j, k, HydroSystem<ThermalConductionProblem>::scalar0_index);
		amrex::Real const windTracer = state[bx](i, j, k, HydroSystem<ThermalConductionProblem>::scalar0_index + 1);
		if (cloudTracer < refine_threshold && windTracer < refine_threshold) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
	amrex::Gpu::streamSynchronize();
}


template <> void QuokkaSimulation<ThermalConductionProblem>::computeAfterTimestep()
{
	const Real dt_coarse = dt_[0];
	const Real time = tNew_[0];

	// perform Galilean transformation (velocity shift to center-of-mass frame)
	// N.B. the wind flows along z here, so we track/shift the z-momentum (cf. testShockCloud.cpp,
	// which tracks x-momentum). t_cc is only used for diagnostics below, not to gate the shift,
	// since (unlike ShockCloud) the wind is already interacting with the cloud from t=0.
	if (::do_frame_shift) {

		// N.B. must weight by the cloud tracer, since the wind also carries momentum!
		int const nc = 1; // number of components in temporary MF
		int const ng = 0; // number of ghost cells in temporary MF
		amrex::MultiFab temp_mf(boxArray(0), DistributionMap(0), nc, ng);

		// compute z-momentum weighted by cloud tracer
		amrex::MultiFab::Copy(temp_mf, state_new_cc_[0], HydroSystem<ThermalConductionProblem>::x3Momentum_index, 0, nc, ng);
		amrex::MultiFab::Multiply(temp_mf, state_new_cc_[0], HydroSystem<ThermalConductionProblem>::scalar0_index, 0, nc, ng);
		const Real zmom = temp_mf.sum(0);

		// compute cloud mass (weighted by cloud tracer) within simulation box
		amrex::MultiFab::Copy(temp_mf, state_new_cc_[0], HydroSystem<ThermalConductionProblem>::density_index, 0, nc, ng);
		amrex::MultiFab::Multiply(temp_mf, state_new_cc_[0], HydroSystem<ThermalConductionProblem>::scalar0_index, 0, nc, ng);
		const Real cloud_mass = temp_mf.sum(0);

		// compute center-of-mass velocity of the cloud
		const Real vz_cm = zmom / cloud_mass;

		// save cumulative position, velocity offsets in simulationMetadata_
		const Real delta_x_prev = simulationMetadata_["delta_x"].as<Real>();
		const Real delta_vz_prev = simulationMetadata_["delta_vz"].as<Real>();
		const Real delta_x = delta_x_prev + dt_coarse * delta_vz_prev;
		const Real delta_vz = delta_vz_prev + vz_cm;
		simulationMetadata_["delta_x"] = delta_x;
		simulationMetadata_["delta_vz"] = delta_vz;
		::delta_vz = delta_vz;

		amrex::Print() << "[Cloud Tracking] Delta z = " << (delta_x / C::parsec) << " pc,"
			       << " Delta vz = " << (delta_vz / 1.0e5) << " km/s,"
			       << " Inflow velocity = " << ((::v_wind - delta_vz) / 1.0e5) << " km/s,"
			       << " t/t_cc = " << (time / ::cloud_crushing_time) << "\n";

		// If we are moving faster than the wind, we should abort the simulation.
		// (otherwise, the boundary conditions become inconsistent.)
		AMREX_ALWAYS_ASSERT(delta_vz < ::v_wind);

		// subtract center-of-mass z-velocity on each level
		// N.B. must update both z-momentum *and* energy!
		for (int lev = 0; lev <= finest_level; ++lev) {
			auto const &mf = state_new_cc_[lev];
			auto const &state = state_new_cc_[lev].arrays();
			amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) noexcept {
				Real const rho = state[box](i, j, k, HydroSystem<ThermalConductionProblem>::density_index);
				Real const xmom = state[box](i, j, k, HydroSystem<ThermalConductionProblem>::x1Momentum_index);
				Real const ymom = state[box](i, j, k, HydroSystem<ThermalConductionProblem>::x2Momentum_index);
				Real const zmom = state[box](i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index);
				Real const E = state[box](i, j, k, HydroSystem<ThermalConductionProblem>::energy_index);
				Real const KE = 0.5 * (xmom * xmom + ymom * ymom + zmom * zmom) / rho;
				Real const Eint = E - KE;
				Real const new_zmom = zmom - rho * vz_cm;
				Real const new_KE = 0.5 * (xmom * xmom + ymom * ymom + new_zmom * new_zmom) / rho;

				state[box](i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index) = new_zmom;
				state[box](i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = Eint + new_KE;
			});
		}
		amrex::Gpu::streamSynchronizeAll();
	}
}


// Implement User-defined diode BC
template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<ThermalConductionProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
                             amrex::GeometryData const &geom, const Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
                             int /*orig_comp*/)
{
    auto [i, j, k] = iv.dim3();
    amrex::Box const &box = geom.Domain();
    const auto &domain_lo = box.loVect3d();
    const auto &domain_hi = box.hiVect3d();
    const int klo = domain_lo[2];
    const int khi = domain_hi[2];
    double rho_edge = NAN;
    double x1Mom_edge = NAN;
    double x2Mom_edge = NAN;
    double x3Mom_edge = NAN;
    double etot_edge = NAN;
    double eint_edge = NAN;


    const double cellVolume = geom.CellSize(0) * geom.CellSize(1) * geom.CellSize(2);

    // N.B. subtract the accumulated center-of-mass frame velocity offset (::delta_vz), so the
    // injected wind stays consistent with the shifted frame (cf. testShockCloud.cpp's use of ::delta_vx).
    rho_edge = rho_cloud * Tcloud / Twind; // g/cm^3
    const double vz_edge = ::v_wind - ::delta_vz;
    x3Mom_edge = rho_edge * vz_edge;
    eint_edge = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho_edge, Twind);
    etot_edge = eint_edge + 0.5 * (x3Mom_edge * x3Mom_edge) / rho_edge;
    
    consVar(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho_edge;
    consVar(i, j, k, HydroSystem<ThermalConductionProblem>::x1Momentum_index) = 0.0;
    consVar(i, j, k, HydroSystem<ThermalConductionProblem>::x2Momentum_index) = 0.0;
    consVar(i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index) = x3Mom_edge;
    consVar(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = etot_edge;
    consVar(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = eint_edge;
    consVar(i, j, k, HydroSystem<ThermalConductionProblem>::scalar0_index) = 0.0; // wind boundary carries no cloud tracer
    consVar(i, j, k, HydroSystem<ThermalConductionProblem>::scalar0_index + 1) = TracerPerVolume; // wind boundary carries wind tracer
}


auto problem_main() -> int
{
	// read problem-specific parameters
	amrex::ParmParse const pp("windcloud");
	pp.query("mach", ::Mach);

	// do frame shifting to follow cloud center-of-mass?
	amrex::ParmParse const pp_global; // top-level, unprefixed
	int do_frame_shift = 1;
	pp_global.query("do_frame_shift", do_frame_shift);
	::do_frame_shift = do_frame_shift == 1;

	// compute wind speed (pressure equilibrium with the cloud sets the wind density)
	const Real rho_wind = rho_cloud * Tcloud / Twind; // g/cm^3
	const Real P_wind = rho_wind * Twind * C::k_B / C::m_u;
	const Real cs_wind = quokka::EOS<ThermalConductionProblem>::ComputeSoundSpeed(rho_wind, P_wind);
	::v_wind = ::Mach * cs_wind;
	amrex::Print() << "rho_wind = " << rho_wind << " g/cm^3" << std::endl;
	amrex::Print() << "v_wind = " << (::v_wind / 1.0e5) << " km/s" << std::endl;

	// estimate cloud-crushing time: t_cc = sqrt(chi) * R_cloud / v_wind, chi = rho_cloud / rho_wind
	const Real chi = rho_cloud / rho_wind;
	::cloud_crushing_time = std::sqrt(chi) * R0 / ::v_wind;
	amrex::Print() << "t_cc = " << (::cloud_crushing_time / (1.0e6 * seconds_in_year)) << " Myr" << std::endl;

	// boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);

	for (int n = 0; n < ncomp_cc; ++n) {
	for (int i = 0; i < AMREX_SPACEDIM; ++i) {
		// diode boundary conditions
		if (i == 2) {
			BCs_cc[n].setLo(i, amrex::BCType::ext_dir); // inflow
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		} else {
			BCs_cc[n].setLo(i, amrex::BCType::foextrap); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::foextrap); // periodic
		}
	}
	} 

	// Problem initialization
	QuokkaSimulation<ThermalConductionProblem> sim(BCs_cc);

	// set metadata used by computeAfterTimestep() for center-of-mass frame tracking
	sim.simulationMetadata_["delta_x"] = 0._rt;
	sim.simulationMetadata_["delta_vz"] = 0._rt;
	sim.simulationMetadata_["t_cc"] = ::cloud_crushing_time;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;
}
