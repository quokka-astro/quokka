/// \file testParticleStarEvolution.cpp
/// \brief Validates the toy stellar-evolution model (R(M), L(M, mdot)) for a Star particle
///        accreting from a uniform medium via the grid Bondi accretion module.

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "particles/stellar_models.hpp"
#include "util/BC.hpp"

#include <cmath>
#include <numeric>
#include <vector>

using amrex::Real;

struct StarEvolutionProblem {
};

// Ambient medium (matches ParticleAccretion: cold, dense, isothermal)
constexpr double T0 = 10.0;            // K
constexpr double mu = 2.33 * C::m_p;   // mean molecular weight
constexpr double cs0 = 1.882195750e4;  // sqrt(k_B T0 / mu) cm/s for T0=10 K, mu=2.33 m_p
constexpr double B0 = 1.0e-7;          // tiny background field (MHD enabled to mirror ParticleAccretion)

double rho0 = C::m_p;                  // NOLINT background density (n_H ~ 1)
double M0_in_Msun = 1.0;               // NOLINT initial particle mass
double t_end_over_t_b = 30.0;          // NOLINT run length in Bondi times

template <> struct Particle_Traits<StarEvolutionProblem> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Star;
};

template <> struct quokka::EOS_Traits<StarEvolutionProblem> {
	static constexpr double gamma = 1.0; // isothermal
	static constexpr double cs_isothermal = cs0;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<StarEvolutionProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<StarEvolutionProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1;
	static constexpr bool is_self_gravity_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // one luminosity slot
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct SimulationData<StarEvolutionProblem> {
	std::vector<Real> time;
	std::vector<Real> mass;
	std::vector<Real> mdot;
	std::vector<Real> radius;
	std::vector<Real> lum;
};

// Place a single Star particle of mass M0 at the domain center (cell-center of the origin cell).
template <> void QuokkaSimulation<StarEvolutionProblem>::createInitialStarParticles()
{
	const int lev = 0;
	using ContainerType = quokka::StarParticleContainer<StarEvolutionProblem>;

	// Update NextID for the particle we will create
	const amrex::Long pid = ContainerType::ParticleType::NextID();
	ContainerType::ParticleType::NextID(pid + 1);

	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const int cpu_id = amrex::ParallelDescriptor::MyProc();

	// Create a particle tile at level 0
	for (amrex::MFIter mfi = StarParticles->MakeMFIter(lev); mfi.isValid(); ++mfi) {
		auto &particle_tile = StarParticles->DefineAndReturnParticleTile(lev, mfi);
		auto &aos = particle_tile.GetArrayOfStructs();
		const int old_size = aos.size();
		constexpr int max_new_particles = 1;
		aos.resize(old_size + max_new_particles);

		auto *pdata = aos.data() + old_size;

		amrex::ParallelFor(max_new_particles, [=] AMREX_GPU_DEVICE(int i) {
			auto &p = pdata[i]; // NOLINT
			p.id() = pid + i;
			p.cpu() = cpu_id;
			p.pos(0) = 0.5 * dx[0];
			p.pos(1) = 0.5 * dx[1];
			p.pos(2) = 0.5 * dx[2];

			constexpr int NReal = quokka::StarParticleRealComps<StarEvolutionProblem>;
			for (int j = 0; j < NReal; ++j) {
				p.rdata(j) = 0.0;
			}
			p.rdata(quokka::StarParticleMassIdx) = M0_in_Msun * C::M_solar;
			p.rdata(quokka::StarParticleBirthTimeIdx) = 0.0;
		});
		break; // Only create on the first grid
	}
	amrex::Gpu::streamSynchronize();
	StarParticles->Redistribute();
}

template <> void QuokkaSimulation<StarEvolutionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double rho_bg = rho0;
	const double Eint = rho_bg / mu * C::k_B * T0; // arbitrary for isothermal EOS
	const double Emag = 0.5 * B0 * B0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<StarEvolutionProblem>::density_index) = rho_bg;
		state_cc(i, j, k, HydroSystem<StarEvolutionProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<StarEvolutionProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<StarEvolutionProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<StarEvolutionProblem>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<StarEvolutionProblem>::energy_index) = Eint + Emag;
	});
}

template <> void QuokkaSimulation<StarEvolutionProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const double B_val = (dir == quokka::direction::x) ? B0 : 0.0;

	amrex::ParallelFor(indexRange,
			   [=] AMREX_GPU_DEVICE(int i, int j, int k) { state_fc(i, j, k, Physics_Indices<StarEvolutionProblem>::mhdFirstIndex) = B_val; });
}

// Record the particle's (t, M, mdot, R, L) after every coarse step.
template <> void QuokkaSimulation<StarEvolutionProblem>::computeAfterTimestep()
{
	const int finest_level = finestLevel();
	const auto &real_data = particleRegister_.getParticleDescriptor(quokka::ParticleType::Star)->getParticleDataAtLevel(finest_level).first;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr int off = AMREX_SPACEDIM; // 3 position components precede rdata
		if (!real_data.empty()) {
			const auto &p = real_data[0];
			userData_.time.push_back(tNew_[0]);
			userData_.mass.push_back(p[off + quokka::StarParticleMassIdx]);
			userData_.mdot.push_back(p[off + quokka::StarParticleMdotIdx]);
			userData_.radius.push_back(p[off + quokka::StarParticleRadiusIdx]);
			userData_.lum.push_back(p[off + quokka::StarParticleLumIdx]);
		}
	}
}

auto problem_main() -> int
{
	amrex::ParmParse const pp("problem");
	pp.query("M0_in_Msun", M0_in_Msun);
	pp.query("rho0", rho0);
	pp.query("t_end_over_t_b", t_end_over_t_b);

	const double M0_g = M0_in_Msun * C::M_solar;
	const double r_B = C::Gconst * M0_g / (cs0 * cs0);
	const double t_B = r_B / cs0;

	QuokkaSimulation<StarEvolutionProblem> sim;
	sim.reconstructionOrder_ = 3;
	sim.cflNumber_ = 0.3;
	sim.tempFloor_ = 10.0;
	sim.stopTime_ = t_end_over_t_b * t_B;

	sim.setInitialConditions();
	sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Star)->setForceFinestLevel(true);

	sim.evolve();

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		using Model = quokka::ToyStellarModel;
		const auto &t = sim.userData_.time;
		const auto &M = sim.userData_.mass;
		const auto &mdot = sim.userData_.mdot;
		const auto &R = sim.userData_.radius;
		const auto &L = sim.userData_.lum;
		const int n = static_cast<int>(t.size());

		amrex::Print() << "\n=== Stellar-evolution validation (" << n << " samples) ===\n";
		amrex::Print() << "r_B = " << r_B << " cm, t_B = " << t_B << " s\n";

		const double tol = 2.0e-2; // 2% absorbs the one-step lag (see plan)
		int n_checked = 0;
		for (int i = 0; i < n; ++i) {
			if (!(R[i] > 0.0) || !(M[i] > 0.0)) {
				continue; // skip pre-activation samples
			}
			const double R_pred = Model::radius(M[i]);
			const double L_pred = Model::luminosityStar(M[i]) + Model::luminosityAcc(M[i], mdot[i], R[i]);

			const double R_err = std::abs(R[i] - R_pred) / R_pred;
			const double L_err = (L_pred > 0.0) ? std::abs(L[i] - L_pred) / L_pred : std::abs(L[i]);

			if (R_err > tol) {
				status += 1;
				amrex::Print() << "  FAIL[" << i << "] radius: sim=" << R[i] << " pred=" << R_pred << " rel_err=" << R_err << "\n";
			}
			if (L_err > tol) {
				status += 1;
				amrex::Print() << "  FAIL[" << i << "] lum: sim=" << L[i] << " pred=" << L_pred << " rel_err=" << L_err << "\n";
			}
			++n_checked;
		}
		amrex::Print() << "Checked " << n_checked << " active samples; tolerance = " << tol << "\n";
		if (n_checked == 0) {
			status += 1;
			amrex::Print() << "  FAIL: no active samples (particle never activated / never accreted)\n";
		}

		// Informational: linear mass growth and Bondi rate (coarse grid -> not asserted).
		if (n >= 2) {
			const double mdot_fit = (M[n - 1] - M[0]) / (t[n - 1] - t[0]);
			const double lambda = std::exp(1.5) / 4.0;
			const double Mdot_bondi = 4.0 * M_PI * rho0 * r_B * r_B * lambda * cs0;
			amrex::Print() << "Mean dM/dt = " << mdot_fit << " g/s; analytic Bondi (hydro) ~ " << Mdot_bondi << " g/s\n";
			amrex::Print() << "Mass growth over run: " << (M[n - 1] / M[0] - 1.0) * 100.0 << " %\n";
		}

		amrex::Print() << (status == 0 ? "\n=== All stellar-evolution checks passed ===\n"
					     : "\n=== Test FAILED (status=" + std::to_string(status) + ") ===\n");
	}

	amrex::ParallelDescriptor::Bcast(&status, 1, amrex::ParallelDescriptor::IOProcessorNumber());
	return status;
}
