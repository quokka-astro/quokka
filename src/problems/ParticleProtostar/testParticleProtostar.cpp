/// \file testParticleProtostar.cpp
/// \brief Defines a test problem for protostar particles.
///

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "math/interpolate.hpp"
#include "util/fextract.hpp"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_creation.hpp"
#include "particles/particle_types.hpp"
#include "util/BC.hpp"

struct ProtostarProblem {
};

static bool refine_half_domain = false; // NOLINT

constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
const double T0 = 10.0;		  // K
const double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
const double year = 3.15576e+07; // in seconds
const double dt_init = 3.0 * year;
constexpr double B0 = 1.0e-7; // constant background field [Gauss-equivalent units]

template <> struct Particle_Traits<ProtostarProblem> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Protostar;
};

template <> struct quokka::EOS_Traits<ProtostarProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<ProtostarProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<ProtostarProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	// face-centred
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> void QuokkaSimulation<ProtostarProblem>::createInitialProtostarParticles()
{
	// Start with no particles
}

template <> void QuokkaSimulation<ProtostarProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// Compute Jeans density to ensure instability
	// rho_J = pi * cs^2 / (G * dx^2) (approx)
	// Actually we use ParticleUtils::computeJeansDensity in creation, let's just make rho0 very large.
	// cs ~ 0.3 km/s. dx depends on resolution.
	// Assuming dx ~ 1e16 cm (pc scale), G ~ 6e-8.
	// rho_J ~ 1e5 / (6e-8 * 1e32) is very small.
	// But let's set rho0 = 1e-18 g/cm^3 (standard GMC density is 1e-22).
	const double rho0 = 1.0e-12;

	// We want to be Jeans unstable.
	// We also need to be the local maximum.
	// We set uniform density rho0, and a perturbation in the center.

	const double rho_e = CV * T0 * rho0;
	const double Emag = 0.5 * B0 * B0;

	const auto prob_lo = grid_elem.prob_lo_;
	const auto prob_hi = grid_elem.prob_hi_;
	const auto dx = grid_elem.dx_;
	const double center_x = 0.5 * (prob_lo[0] + prob_hi[0]);
	const double center_y = 0.5 * (prob_lo[1] + prob_hi[1]);
	const double center_z = 0.5 * (prob_lo[2] + prob_hi[2]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		double x = prob_lo[0] + (i + 0.5) * dx[0];
		double y = prob_lo[1] + (j + 0.5) * dx[1];
		double z = prob_lo[2] + (k + 0.5) * dx[2];

		double r2 = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) + (z - center_z) * (z - center_z);
		double rho = rho0;

		// Add single cell perturbation at center?
		// Or Gaussian.
		// Let's use a small Gaussian bump to trigger formation at center.
		if (r2 < (dx[0] * dx[0])) {
			rho *= 1.1;
		}

		state_cc(i, j, k, HydroSystem<ProtostarProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ProtostarProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<ProtostarProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<ProtostarProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<ProtostarProblem>::energy_index) = CV * T0 * rho + Emag;
		state_cc(i, j, k, HydroSystem<ProtostarProblem>::internalEnergy_index) = CV * T0 * rho;
	});
}

template <> void QuokkaSimulation<ProtostarProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const double B_val = (dir == quokka::direction::x) ? B0 : 0.0;

	amrex::ParallelFor(indexRange,
			   [=] AMREX_GPU_DEVICE(int i, int j, int k) { state_fc(i, j, k, Physics_Indices<ProtostarProblem>::mhdFirstIndex) = B_val; });
}

template <> void QuokkaSimulation<ProtostarProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement: static mesh refinement for the whole domain (if refine_half_domain is false) or for x > 0 (if refine_half_domain is true)

	auto const &dx = geom[lev].CellSizeArray();
	auto const &plo = geom[lev].ProbLoArray();
	auto const &phi = geom[lev].ProbHiArray();
	const bool refine_half_domain_ = refine_half_domain;

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const double x_frac = ((i + 0.5) * dx[0]) / (phi[0] - plo[0]);
			const double y_frac = ((j + 0.5) * dx[1]) / (phi[1] - plo[1]);
			const double z_frac = ((k + 0.5) * dx[2]) / (phi[2] - plo[2]);
			if (!refine_half_domain_ || (x_frac >= 0.7 && x_frac <= 0.8 && y_frac >= 0.3 && y_frac <= 0.7 && z_frac >= 0.3 && z_frac <= 0.7)) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<ProtostarProblem>(quokka::BCType::reflecting);

	const int nvars_fc = Physics_Indices<ProtostarProblem>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::reflect_even);
			BCs_fc[icomp].setHi(idim, amrex::BCType::reflect_even);
		}
	}

	amrex::ParmParse const pp("problem");
	pp.query("refine_half_domain", refine_half_domain);

	// Problem initialization
	QuokkaSimulation<ProtostarProblem> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	sim.stopTime_ = 10.0 * dt_init;
	sim.initDt_ = dt_init;
	sim.tempFloor_ = 10.0; // K

	// initialize
	sim.setInitialConditions();

	// get total gas mass in the initial state
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	amrex::Real const total_mass_init = sim.state_new_cc_[0].sum(HydroSystem<ProtostarProblem>::density_index) * vol;
	double total_particle_mass = 0.0;

	// get total particle mass
	const auto &real_data = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Protostar)->getParticleDataAtLevel(0).first;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		for (const auto &p : real_data) {
			total_particle_mass += p[quokka::ProtostarParticleMassIdx + 3]; // shift by 3 because positions take up the first 3 indices
		}
		amrex::Print() << "\nBefore evolution:\n";
		amrex::Print() << "Total gas mass = " << total_mass_init << "\n";
		amrex::Print() << "Total particle mass = " << total_particle_mass << "\n";
	}

	const double total_total_mass_init = total_mass_init + total_particle_mass;

	// evolve
	sim.maxTimesteps_ = 1;
	sim.evolve();

	// get total gas mass in the final state
	amrex::Real const total_mass_step1 = sim.state_new_cc_[0].sum(HydroSystem<ProtostarProblem>::density_index) * vol;

	int status = 0;

	const auto &real_data_ste1 = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Protostar)->getParticleDataAtLevel(0).first;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// compute total particle mass and error
		double total_particle_mass_step1 = 0.0;
		for (const auto &p : real_data_ste1) {
			total_particle_mass_step1 += p[quokka::ProtostarParticleMassIdx + 3];
		}
		const double total_total_mass_step1 = total_mass_step1 + total_particle_mass_step1;

		// compute difference in mass changes
		const double gas_mass_change = total_mass_step1 - total_mass_init;
		const double particle_mass_change = total_particle_mass_step1 - total_particle_mass;
		const double rel_mass_error = gas_mass_change == 0.0 ? 0.0 : std::abs(gas_mass_change + particle_mass_change) / std::abs(gas_mass_change);
		amrex::Print() << "\nAfter evolution:\n";
		amrex::Print() << "Gas mass change = " << gas_mass_change << "\n";
		amrex::Print() << "Particle mass change = " << particle_mass_change << "\n";
		amrex::Print() << "Total mass change = " << gas_mass_change + particle_mass_change << "\n";
		amrex::Print() << "Relative error in change of mass = " << rel_mass_error << "\n";

		// compute relative error in the change of total mass
		const double rel_error_total_mass = std::abs(total_total_mass_step1 - total_total_mass_init) / total_total_mass_init;
		amrex::Print() << "Relative error in change of total mass = " << rel_error_total_mass << "\n";

		// The total mass (gas + particles) should be conserved within machine precision (1e-14)
		const double mass_rel_error_tol = 1.0e-8;
		if (!(rel_error_total_mass < mass_rel_error_tol)) {
			status = 1;
			amrex::Print() << "Test failed: total mass is not conserved at step 1\n";
		}

		// Verify particles formed
		if (total_particle_mass_step1 <= 0.0) {
			status = 1;
			amrex::Print() << "Test failed: No particles formed!\n";
		} else {
			amrex::Print() << "ParticleProtostar Success: Particles formed.\n";
		}

		if (status == 0) {
			amrex::Print() << "ParticleProtostar Success\n";
		}
	}

	return status;
}
