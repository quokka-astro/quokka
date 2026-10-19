/// \file testParticleStar.cpp
/// \brief Test problem for Star particles with the deuterium-burning protostellar model: formation from a
/// Jeans-unstable cell, accretion with mass conservation, and the stellar evolution checked against reference
/// tracks from the ORION2 protostellar model (see README.md in this directory).
///

#include <array>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/deuterium_stellar_model.hpp"
#include "particles/particle_types.hpp"
#include "util/BC.hpp"

struct ParticleStar {
};

constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
const double rho0 = 1.0 * C::m_p; // g cm^-3
const double T0 = 10.0;		  // K
const double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
const double cs = std::sqrt(gamma_ * C::k_B * T0 / mu); // NOLINT
constexpr double B0 = 1.0e-7;				// uniform background field

using StellarModel = quokka::DeuteriumBurningStellarModel;

template <> struct Particle_Traits<ParticleStar> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Star;
	using stellar_model = StellarModel;
};

template <> struct quokka::EOS_Traits<ParticleStar> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<ParticleStar> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<ParticleStar> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr int n_groups = Physics_Traits<ParticleStar>::nGroups;
constexpr int mdeut_idx = quokka::StarParticleFixedComps + n_groups; // first model extra
constexpr int n_poly_idx = mdeut_idx + 1;

template <> void QuokkaSimulation<ParticleStar>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const auto prob_lo = geom[0].ProbLoArray();
	const auto dx = geom[0].CellSizeArray();

	const double jeans_density = quokka::ParticleUtils::computeJeansDensity(cs, dx[0]);
	const double Emag = 0.5 * B0 * B0;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + (i * dx[0]);
		const double y = prob_lo[1] + (j * dx[1]);
		const double z = prob_lo[2] + (k * dx[2]);
		double rho = rho0;
		const double sf_cell_loc = 0.1 * dx[0]; // The cell right next to the origin
		if (x <= sf_cell_loc && x + dx[0] > sf_cell_loc && y <= sf_cell_loc && y + dx[1] > sf_cell_loc && z <= sf_cell_loc && z + dx[2] > sf_cell_loc) {
			// this is the cell with peak density
			rho = 1.2 * jeans_density;
		} else if (x - 2 * dx[0] <= sf_cell_loc && x - dx[0] > sf_cell_loc && y <= sf_cell_loc && y + dx[1] > sf_cell_loc && z <= sf_cell_loc &&
			   z + dx[2] > sf_cell_loc) {
			// this is the cell that is 2 cells left of the peak density cell
			rho = 1.1 * jeans_density;
		}
		const double rho_e = CV * T0 * rho;
		state_cc(i, j, k, HydroSystem<ParticleStar>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ParticleStar>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<ParticleStar>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<ParticleStar>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<ParticleStar>::energy_index) = rho_e + Emag;
		state_cc(i, j, k, HydroSystem<ParticleStar>::internalEnergy_index) = rho_e;
	});
}

template <> void QuokkaSimulation<ParticleStar>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const double B_val = (dir == quokka::direction::x) ? B0 : 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) { state_fc(i, j, k, Physics_Indices<ParticleStar>::mhdFirstIndex) = B_val; });
}

// Evolve the stellar model on the host with the constant-accretion-rate steps of an ORION2 reference run
// (m0 = 1e-3 M_sun, dm = 1e-4 M_sun per step) and compare with the stored reference track.
auto compareWithReferenceTrack(std::string const &filename, double mdot_Msun_per_yr) -> int
{
	std::ifstream input(filename);
	if (!input) {
		amrex::Print() << "FAIL: cannot open reference track " << filename << "\n";
		return 1;
	}

	const double dm = 1.0e-4 * C::M_solar;
	const double mdot = mdot_Msun_per_yr * C::M_solar / quokka::yr_in_s;
	const double dt = dm / mdot;
	std::array<amrex::Real, quokka::StarParticleFixedComps + n_groups + StellarModel::nExtraReal> rdata{};
	std::array<int, StellarModel::nExtraInt> idata{};
	rdata[quokka::StarParticleMassIdx] = 1.0e-3 * C::M_solar;
	rdata[quokka::StarParticleMdotIdx] = mdot;

	int n_rows = 0;
	int n_state_mismatch = 0;
	bool phase_jump = false;
	double max_err_r = 0.0;
	double max_err_L = 0.0;
	double max_err_n = 0.0;
	double max_err_mdeut = 0.0;
	long step = 0;
	std::string line;
	while (std::getline(input, line)) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		std::istringstream row(line);
		long ref_step = 0;
		double m_ref = NAN;
		double r_ref = NAN;
		double n_ref = NAN;
		double mdeut_ref = NAN;
		int state_ref = -1;
		double L_ref = NAN;
		row >> ref_step >> m_ref >> r_ref >> n_ref >> mdeut_ref >> state_ref >> L_ref;
		for (; step < ref_step; ++step) {
			rdata[quokka::StarParticleMassIdx] += dm;
			StellarModel::evolve(rdata.data(), idata.data(), n_groups, dt);
		}
		++n_rows;

		// Transitions may land a few steps apart, so a row next to a transition can be one phase off.
		if (idata[0] != state_ref) {
			++n_state_mismatch;
			phase_jump = phase_jump || (std::abs(idata[0] - state_ref) > 1);
			continue;
		}
		max_err_r = std::max(max_err_r, std::abs(rdata[quokka::StarParticleRadiusIdx] / r_ref - 1.0));
		max_err_L = std::max(max_err_L, std::abs(rdata[quokka::StarParticleLumIdx] / L_ref - 1.0));
		max_err_n = std::max(max_err_n, std::abs(rdata[n_poly_idx] - n_ref));
		max_err_mdeut = std::max(max_err_mdeut, std::abs(rdata[mdeut_idx] - mdeut_ref) / m_ref);
	}

	// Relative tolerance for r and L: Quokka sub-cycles the radius update, so it differs from the reference
	// (itself ~0.3% from its converged solution) by up to ~1.5% just after the model starts at 0.01 M_sun.
	const double tol_rL = 2.0e-2;
	const bool passed = (n_rows > 0) && (n_state_mismatch <= 4) && !phase_jump && (max_err_r < tol_rL) && (max_err_L < tol_rL) && (max_err_n < 1.0e-10) &&
			    (max_err_mdeut < 1.0e-2);
	amrex::Print() << "Reference track " << filename << ": " << n_rows << " rows, " << n_state_mismatch
		       << " rows one phase apart, max |r/r_ref - 1| = " << max_err_r << ", max |L/L_ref - 1| = " << max_err_L
		       << ", max |n - n_ref| = " << max_err_n << ", max |mdeut - mdeut_ref| / m = " << max_err_mdeut << (passed ? "  PASS" : "  FAIL")
		       << "\n";
	return passed ? 0 : 1;
}

auto problem_main() -> int
{
	int status = 0;

	// Stellar evolution against the ORION2 reference model
	status += compareWithReferenceTrack("../src/problems/ParticleStar/orion2_track_mdot1e-5.dat", 1.0e-5);
	status += compareWithReferenceTrack("../src/problems/ParticleStar/orion2_track_mdot1e-4.dat", 1.0e-4);

	QuokkaSimulation<ParticleStar> sim;

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	sim.stopTime_ = 1.0e7 * quokka::yr_in_s;

	sim.setInitialConditions();

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	amrex::Real const m_tot_init = sim.state_new_cc_[0].sum(HydroSystem<ParticleStar>::density_index) * vol;

	// getParticleDataAtAllLevels returns [pos_x, pos_y, pos_z, rdata...] per particle
	auto star_data = [&sim]() {
		const auto [ids, real_data, int_data] = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Star)->getParticleDataAtAllLevels();
		return std::make_pair(real_data, int_data);
	};
	auto total_mass = [&](auto const &real_data) {
		const double m_stars = std::accumulate(real_data.begin(), real_data.end(), 0.0, [](double sum, const auto &particle) {
			return sum + particle[AMREX_SPACEDIM + quokka::StarParticleMassIdx];
		});
		return sim.state_new_cc_[0].sum(HydroSystem<ParticleStar>::density_index) * vol + m_stars;
	};

	// Formation: one Star particle forms in the first step, conserving mass
	sim.maxTimesteps_ = 1;
	sim.evolve();
	const auto [real_data_step1, int_data_step1] = star_data();
	if (real_data_step1.size() != 1) {
		amrex::Print() << "FAIL: expected 1 Star particle after the formation step, got " << real_data_step1.size() << "\n";
		return 1;
	}
	const double rel_err_step1 = std::abs(total_mass(real_data_step1) / m_tot_init - 1.0);
	amrex::Print() << "Formation step: relative change of total mass = " << rel_err_step1 << "\n";
	if (!(rel_err_step1 < 1.0e-14)) {
		amrex::Print() << "FAIL: mass not conserved in the formation step\n";
		status += 1;
	}

	// Accretion: total mass conserved
	sim.maxTimesteps_ = 20;
	sim.evolve();
	const auto [real_data, int_data] = star_data();
	const double rel_err_final = std::abs(total_mass(real_data) / m_tot_init - 1.0);
	amrex::Print() << "After " << sim.maxTimesteps_ << " steps: relative change of total mass = " << rel_err_final << "\n";
	if (!(rel_err_final < 1.0e-13)) {
		amrex::Print() << "FAIL: mass not conserved during accretion\n";
		status += 1;
	}

	// Stellar properties of the accreting star
	if (amrex::ParallelDescriptor::IOProcessor()) {
		for (size_t p = 0; p < real_data.size(); ++p) {
			const auto &rdata = real_data[p];
			const double mass = rdata[AMREX_SPACEDIM + quokka::StarParticleMassIdx];
			const double mdot = rdata[AMREX_SPACEDIM + quokka::StarParticleMdotIdx];
			const double radius = rdata[AMREX_SPACEDIM + quokka::StarParticleRadiusIdx];
			const double lum = rdata[AMREX_SPACEDIM + quokka::StarParticleLumIdx];
			const double n_poly = rdata[AMREX_SPACEDIM + n_poly_idx];
			const int state = int_data[p][0];
			amrex::Print() << "Star " << p << ": m = " << mass / C::M_solar << " M_sun, mdot = " << mdot * quokka::yr_in_s / C::M_solar
				       << " M_sun/yr, r = " << radius / C::R_solar << " R_sun, n = " << n_poly << ", L = " << lum / quokka::StellarPhysics::L_solar
				       << " L_sun, burn_state = " << state << "\n";

			if (mass >= quokka::StellarPhysics::M_rad_min && mdot > 0.0 && state == StellarModel::PreCollapse) {
				amrex::Print() << "FAIL: the stellar model did not start although m >= 0.01 M_sun and mdot > 0\n";
				status += 1;
			}
			if (state != StellarModel::PreCollapse) {
				if (!(n_poly >= 1.5 && n_poly <= 3.0)) {
					amrex::Print() << "FAIL: polytropic index outside [1.5, 3]\n";
					status += 1;
				}
				if (!(lum > 0.0 && std::isfinite(lum) && radius > 0.0 && std::isfinite(radius))) {
					amrex::Print() << "FAIL: luminosity and radius must be positive and finite\n";
					status += 1;
				}
				// The radius is evolved, not reset to the initial value each step
				if (radius == quokka::StellarPhysics::radInit(mdot)) {
					amrex::Print() << "FAIL: radius equals its initial value from the accretion rate\n";
					status += 1;
				}
			}
		}
	}

	if (status == 0) {
		amrex::Print() << "All ParticleStar checks passed\n";
	}
	return status;
}
