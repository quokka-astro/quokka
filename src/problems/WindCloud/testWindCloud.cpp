//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testWindCloud.cpp
/// \brief Defines a wind-cloud problem with Spitzer thermal conduction.
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
#include <string>

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
const double Tracer = 1.; // tracer content per volume

// frame-tracking globals (set inside problem_main() / computeAfterTimestep())
bool do_frame_shift = true;			      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real v_wind = NAN;		      // wind speed (z direction)
AMREX_GPU_MANAGED Real cloud_crushing_time = NAN;    // t_cc, estimated from R0 and v_wind
AMREX_GPU_MANAGED Real delta_vz = 0;		      // cumulative center-of-mass frame velocity offset

// Anisotropic conduction runs along the magnetic field, so MHD (and the magnetic field inputs
// windcloud.plasmaBeta and windcloud.field_orientation) is enabled only when conduction is anisotropic.
constexpr bool anisotropic_conduction = false;

// magnetic field (set inside problem_main(); only used when anisotropic_conduction == true)
// The field is uniform, with strength set by the plasma beta of the wind, beta = P_wind / (B0^2 / 2),
// and oriented either parallel (along z) or perpendicular (along x) to the wind direction.
enum class FieldOrientation { parallel, perpendicular };
AMREX_GPU_MANAGED Real plasma_beta = NAN;					    // plasma beta of the wind
AMREX_GPU_MANAGED Real B0 = 0.0;						    // field strength, in code units (E_mag = B0^2 / 2)
AMREX_GPU_MANAGED FieldOrientation field_orientation = FieldOrientation::parallel; // field direction relative to the wind

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
	static constexpr bool is_mhd_enabled = anisotropic_conduction;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 2; // cloud tracer + wind tracer
	static constexpr ConductionModel conduction_model = ConductionModel::spitzer; // kappa = prefactor * T^2.5
	static constexpr ConductionGeometry conduction_geometry =
	    anisotropic_conduction ? ConductionGeometry::anisotropic : ConductionGeometry::isotropic; // anisotropic: conduction along/across B
};

template <> void QuokkaSimulation<WindCloudProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// initialize a WindCloud problem using parameters from

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
			cloudTracer = Tracer; // dimensionless concentration, independent of rho
			windTracer  = 0.0; // outside the wind
		}
		else{
			T = Twind;
			rho = rho_cloud * Tcloud / Twind; // g/cm^3
			cloudTracer = 0.0 ; // outside the cloud
			windTracer = Tracer; // dimensionless concentration, independent of rho
			amrex::Real pressure = rho * T * C::k_B / C::m_u;
			cs_wind = quokka::EOS<WindCloudProblem>::ComputeSoundSpeed(rho, pressure);
			vz = ::v_wind; // set in problem_main(), so it stays consistent with the frame-shift BC
		}
		const amrex::Real Eint = quokka::EOS<WindCloudProblem>::ComputeEintFromTgas(rho, T);
		const amrex::Real Emag = 0.5 * ::B0 * ::B0; // uniform field
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
		state_cc(i, j, k, HydroSystem<WindCloudProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<WindCloudProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<WindCloudProblem>::energy_index) = Eint + 0.5 * (rho * vz * vz) + Emag;
		state_cc(i, j, k, HydroSystem<WindCloudProblem>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<WindCloudProblem>::scalar0_index) = rho * cloudTracer; // 1/vol
		state_cc(i, j, k, HydroSystem<WindCloudProblem>::scalar0_index + 1) = rho * windTracer; // 1/vol
	});
}


template <> void QuokkaSimulation<WindCloudProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	// uniform field: along z (the wind direction) if parallel, along x if perpendicular
	const bool is_parallel = (::field_orientation == FieldOrientation::parallel);
	const amrex::Real bx = is_parallel ? 0.0 : ::B0;
	const amrex::Real by = 0.0;
	const amrex::Real bz = is_parallel ? ::B0 : 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<WindCloudProblem>::mhdFirstIndex) = bx;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<WindCloudProblem>::mhdFirstIndex) = by;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<WindCloudProblem>::mhdFirstIndex) = bz;
		}
	});
}


template <> void QuokkaSimulation<WindCloudProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tracer-based refinement: tag cells that are less than 50% cloud AND less than 50% wind,
	// i.e. cells in the cloud-wind mixing/interface region
	const amrex::Real refine_threshold = 0.5 * Tracer;

	auto const &state = state_new_cc_[lev].const_arrays();
	auto const tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// the scalars are mass-weighted (rho * concentration), so divide by rho to recover the concentration
		amrex::Real const rho = state[bx](i, j, k, HydroSystem<WindCloudProblem>::density_index);
		amrex::Real const cloudTracer = state[bx](i, j, k, HydroSystem<WindCloudProblem>::scalar0_index) / rho;
		amrex::Real const windTracer = state[bx](i, j, k, HydroSystem<WindCloudProblem>::scalar0_index + 1) / rho;
		if (cloudTracer < refine_threshold && windTracer < refine_threshold) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
	amrex::Gpu::streamSynchronize();
}


template <>
void QuokkaSimulation<WindCloudProblem>::ComputeDerivedVar(int /*lev*/, std::string const &dname, amrex::MultiFab &mf, const int ncomp_in,
								   amrex::MultiFab const &state_cc,
								   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) const
{
	// compute derived variables and save in 'mf'
	if (dname == "temperature") {
		const int ncomp = ncomp_in;
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_cc.const_array(iter);
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
			if constexpr (Physics_Traits<WindCloudProblem>::is_mhd_enabled) {
				cons_fc = {AMREX_D_DECL(state_fc[0].const_array(iter), state_fc[1].const_array(iter), state_fc[2].const_array(iter))};
			}
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<WindCloudProblem>::density_index);
				// subtracts kinetic and magnetic energy from the total energy
				Real const Eint = HydroSystem<WindCloudProblem>::ComputeInternalEnergy(state, i, j, k, &cons_fc);
				output(i, j, k, ncomp) = quokka::EOS<WindCloudProblem>::ComputeTgasFromEint(rho, Eint);
			});
		}
	}
}

template <> void QuokkaSimulation<WindCloudProblem>::computeAfterTimestep()
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
		amrex::MultiFab::Copy(temp_mf, state_new_cc_[0], HydroSystem<WindCloudProblem>::x3Momentum_index, 0, nc, ng);
		amrex::MultiFab::Multiply(temp_mf, state_new_cc_[0], HydroSystem<WindCloudProblem>::scalar0_index, 0, nc, ng);
		const Real zmom = temp_mf.sum(0);

		// compute cloud mass (weighted by cloud tracer) within simulation box
		amrex::MultiFab::Copy(temp_mf, state_new_cc_[0], HydroSystem<WindCloudProblem>::density_index, 0, nc, ng);
		amrex::MultiFab::Multiply(temp_mf, state_new_cc_[0], HydroSystem<WindCloudProblem>::scalar0_index, 0, nc, ng);
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
				Real const rho = state[box](i, j, k, HydroSystem<WindCloudProblem>::density_index);
				Real const xmom = state[box](i, j, k, HydroSystem<WindCloudProblem>::x1Momentum_index);
				Real const ymom = state[box](i, j, k, HydroSystem<WindCloudProblem>::x2Momentum_index);
				Real const zmom = state[box](i, j, k, HydroSystem<WindCloudProblem>::x3Momentum_index);
				Real const E = state[box](i, j, k, HydroSystem<WindCloudProblem>::energy_index);
				Real const KE = 0.5 * (xmom * xmom + ymom * ymom + zmom * zmom) / rho;
				Real const Eint = E - KE; // N.B. includes magnetic energy, which the frame shift leaves unchanged
				Real const new_zmom = zmom - rho * vz_cm;
				Real const new_KE = 0.5 * (xmom * xmom + ymom * ymom + new_zmom * new_zmom) / rho;

				state[box](i, j, k, HydroSystem<WindCloudProblem>::x3Momentum_index) = new_zmom;
				state[box](i, j, k, HydroSystem<WindCloudProblem>::energy_index) = Eint + new_KE;
			});
		}
		amrex::Gpu::streamSynchronizeAll();
	}
}


// Implement User-defined diode BC
template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<WindCloudProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
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
    eint_edge = quokka::EOS<WindCloudProblem>::ComputeEintFromTgas(rho_edge, Twind);
    etot_edge = eint_edge + 0.5 * (x3Mom_edge * x3Mom_edge) / rho_edge + 0.5 * ::B0 * ::B0; // uniform field
    
    consVar(i, j, k, HydroSystem<WindCloudProblem>::density_index) = rho_edge;
    consVar(i, j, k, HydroSystem<WindCloudProblem>::x1Momentum_index) = 0.0;
    consVar(i, j, k, HydroSystem<WindCloudProblem>::x2Momentum_index) = 0.0;
    consVar(i, j, k, HydroSystem<WindCloudProblem>::x3Momentum_index) = x3Mom_edge;
    consVar(i, j, k, HydroSystem<WindCloudProblem>::energy_index) = etot_edge;
    consVar(i, j, k, HydroSystem<WindCloudProblem>::internalEnergy_index) = eint_edge;
    consVar(i, j, k, HydroSystem<WindCloudProblem>::scalar0_index) = 0.0; // wind boundary carries no cloud tracer
    consVar(i, j, k, HydroSystem<WindCloudProblem>::scalar0_index + 1) = rho * Tracer; // wind boundary carries wind tracer
}


auto problem_main() -> int
{
	// read problem-specific parameters
	amrex::ParmParse const pp("windcloud");
	pp.query("mach", ::Mach);

	// magnetic field: plasma beta of the wind, and orientation relative to the wind ("parallel" or "perpendicular")
	// (only read for anisotropic conduction, which needs MHD)
	std::string field_orientation;
	if constexpr (anisotropic_conduction) {
		pp.get("plasmaBeta", ::plasma_beta);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(::plasma_beta > 0.0, "windcloud.plasmaBeta must be > 0.");
		pp.get("field_orientation", field_orientation);
		if (field_orientation == "parallel") {
			::field_orientation = FieldOrientation::parallel;
		} else if (field_orientation == "perpendicular") {
			::field_orientation = FieldOrientation::perpendicular;
		} else {
			amrex::Abort("windcloud.field_orientation must be \"parallel\" or \"perpendicular\".");
		}
	} else {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!pp.contains("plasmaBeta") && !pp.contains("field_orientation"),
						 "windcloud.plasmaBeta / windcloud.field_orientation are set, but conduction is isotropic (no magnetic field). "
						 "Set anisotropic_conduction = true in testWindCloud.cpp to use them.");
	}

	// do frame shifting to follow cloud center-of-mass?
	amrex::ParmParse const pp_global; // top-level, unprefixed
	int do_frame_shift = 1;
	pp_global.query("do_frame_shift", do_frame_shift);
	::do_frame_shift = do_frame_shift == 1;

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
			BCs_cc[n].setLo(i, amrex::BCType::foextrap); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::foextrap); // periodic
		}
	}
	} 

	// face-centred (magnetic field) boundary conditions; empty when MHD is disabled
	// TODO (av): placeholder; set proper inflow/outflow BCs for the field
	constexpr int ncomp_fc = Physics_Indices<WindCloudProblem>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(ncomp_fc);
	for (int n = 0; n < ncomp_fc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_fc[n].setLo(i, amrex::BCType::foextrap);
			BCs_fc[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	QuokkaSimulation<WindCloudProblem> sim(BCs_cc, BCs_fc);

	// compute wind speed (pressure equilibrium with the cloud sets the wind density)
	const Real rho_wind = rho_cloud * Tcloud / Twind; // g/cm^3
	const Real P_wind = rho_wind * Twind * C::k_B / C::m_u;
	const Real cs_wind = quokka::EOS<WindCloudProblem>::ComputeSoundSpeed(rho_wind, P_wind);
	::v_wind = ::Mach * cs_wind;
	amrex::Print() << "rho_wind = " << rho_wind << " g/cm^3" << std::endl;
	amrex::Print() << "v_wind = " << (::v_wind / 1.0e5) << " km/s" << std::endl;

	// field strength from the plasma beta of the wind: beta = P_wind / (B0^2 / 2)
	if constexpr (anisotropic_conduction) {
		::B0 = std::sqrt(2.0 * P_wind / ::plasma_beta);
		amrex::Print() << "plasma beta = " << ::plasma_beta << ", B0 = " << ::B0 << " (code units) = " << (::B0 * std::sqrt(4.0 * M_PI) * 1.0e6)
			       << " uG, " << field_orientation << " to the wind" << std::endl;
	} else {
		amrex::Print() << "no magnetic field (isotropic conduction)" << std::endl;
	}

	// estimate cloud-crushing time: t_cc = sqrt(chi) * R_cloud / v_wind, chi = rho_cloud / rho_wind
	const Real chi = rho_cloud / rho_wind;
	::cloud_crushing_time = std::sqrt(chi) * R0 / ::v_wind;
	amrex::Print() << "t_cc = " << (::cloud_crushing_time / (1.0e6 * seconds_in_year)) << " Myr" << std::endl;

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
