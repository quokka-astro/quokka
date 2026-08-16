/// \file testStromgrenVolumeFeedback.cpp
/// \brief Validates the Strömgren-volume photoionization feedback module against the analytic
///        Strömgren radius, and checks that the ionizing photon rate is assigned at birth.
///
/// A single star particle sits at the centre of a uniform, pure-hydrogen medium. The module marks
/// the surrounding gas ionized until the particle's ionizing photon budget is consumed by
/// recombinations. In a uniform medium the resulting volume must equal the classical Strömgren
/// volume, so the equivalent radius of the ionized region is compared with
///
///   R_St = (3 Q / (4 pi alpha_B n_H^2))^(1/3).
///
/// The particle mass is chosen so that the Vacca, Garmany & Shull (1996) fitting formula in
/// ToyStellarModel yields exactly the target Q. That way the test exercises the real birth-assignment
/// path rather than the Q override, while still knowing Q exactly.
///
/// The run is deliberately only a few steps long: with hydro enabled the gas would eventually expand,
/// but the analytic comparison is only valid while the density is still uniform.

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "particles/particle_photoionization.hpp"
#include "particles/particle_types.hpp"
#include "particles/stellar_models.hpp"

#include <cmath>
#include <string>

using amrex::Real;

struct StromgrenVolumeProblem {
};

// Target ionizing photon rate. The particle mass below is solved from this.
constexpr double Q_target = 1.0e49; // 1/s
// Case-B recombination coefficient at 1e4 K. Must match stromgren.alpha_B in the inputs file.
constexpr double alpha_B = 2.59e-13; // cm^3/s
// Uniform ambient hydrogen number density. Must be consistent with stromgren.hydrogen_mass_fraction=1.
constexpr double n_H = 1.0e3;	     // 1/cm^3
constexpr double rho0 = n_H * C::m_p; // g/cm^3, pure hydrogen
constexpr double T0 = 100.0;	     // K, cold neutral ambient medium

// Mass whose Vacca, Garmany & Shull rate equals Q_target:
//   Q(m) = Q_ion_coeff * (m / M_sun)^Q_ion_exponent  =>  m = M_sun * (Q_target / Q_ion_coeff)^(1/exponent)
AMREX_GPU_MANAGED double M_star = 0.0; // NOLINT, set in problem_main

// Optional non-uniform ambient medium. The density varies linearly across the box as
//   rho = rho0 * (1 + density_gradient * (x - x_centre) / (L_x / 2)),
// and the source is displaced from the centre by source_offset_frac of the box width. In this
// configuration the analytic Strömgren radius no longer applies, but photon conservation still
// must hold exactly, which is what the test checks.
AMREX_GPU_MANAGED double density_gradient = 0.0;	// NOLINT
AMREX_GPU_MANAGED double source_offset_frac = 0.0;	// NOLINT

template <> struct Particle_Traits<StromgrenVolumeProblem> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Star;
};

template <> struct quokka::EOS_Traits<StromgrenVolumeProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_p; // pure atomic hydrogen
};

template <> struct HydroSystem_Traits<StromgrenVolumeProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<StromgrenVolumeProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = false; // the module does not support MHD
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 1; // slot 0 receives x_ion
	static constexpr int nGroups = 1;	    // one (unused) luminosity slot
};

//! Place a single star particle at the domain centre.
template <> void QuokkaSimulation<StromgrenVolumeProblem>::createInitialStarParticles()
{
	// InitFromAsciiFile handles MPI correctly: only rank 0 reads the file, so exactly one particle
	// is created regardless of the number of ranks.
	constexpr int nreal_extra = 5; // mass, vx, vy, vz, birth_time
	const std::string star_file = "../inputs/star.txt";
	StarParticles->InitFromAsciiFile(star_file, nreal_extra, nullptr);

	const int lev = 0;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();
	constexpr int q_ion_idx = quokka::photoionization::ionizingPhotonRateIndex<StromgrenVolumeProblem>();

	for (auto &kv : StarParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_array = ikv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();
			if (np == 0) {
				continue;
			}
			auto *pdata = particle_array().data();
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
				p.pos(0) = (0.5 * (prob_lo[0] + prob_hi[0])) + (source_offset_frac * (prob_hi[0] - prob_lo[0]));
				p.pos(1) = 0.5 * (prob_lo[1] + prob_hi[1]);
				p.pos(2) = 0.5 * (prob_lo[2] + prob_hi[2]);
				p.rdata(quokka::StarParticleMassIdx) = M_star;
				p.rdata(quokka::StarParticleBirthTimeIdx) = 0.0;
				// InitFromAsciiFile only fills the components present in the file, leaving the rest
				// uninitialized. The ionizing photon rate must start at zero, because that is the
				// "not yet assigned" marker the stellar model looks for when it sets Q at birth.
				p.rdata(quokka::StarParticleMdotIdx) = 0.0;
				p.rdata(quokka::StarParticleRadiusIdx) = 0.0;
				for (int g = 0; g < Physics_Traits<StromgrenVolumeProblem>::nGroups; ++g) {
					p.rdata(quokka::StarParticleLumIdx + g) = 0.0;
				}
				p.rdata(q_ion_idx) = 0.0;
			});
		}
	}
	amrex::Gpu::streamSynchronize();
	StarParticles->Redistribute();
}

template <> void QuokkaSimulation<StromgrenVolumeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;

	const double x_centre = 0.5 * (prob_lo[0] + prob_hi[0]);
	const double half_width = 0.5 * (prob_hi[0] - prob_lo[0]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((static_cast<double>(i) + 0.5) * dx[0]);
		const double rho = rho0 * (1.0 + (density_gradient * (x - x_centre) / half_width));
		const double Eint = quokka::EOS<StromgrenVolumeProblem>::ComputeEintFromTgas(rho, T0);

		state_cc(i, j, k, HydroSystem<StromgrenVolumeProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<StromgrenVolumeProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<StromgrenVolumeProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<StromgrenVolumeProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<StromgrenVolumeProblem>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<StromgrenVolumeProblem>::energy_index) = Eint;
		state_cc(i, j, k, HydroSystem<StromgrenVolumeProblem>::scalar0_index) = 0.0;
	});
}

auto problem_main() -> int
{
	{
		amrex::ParmParse const pp("problem");
		pp.query("density_gradient", density_gradient);
		pp.query("source_offset_frac", source_offset_frac);
	}

	// Solve for the stellar mass that yields exactly Q_target under the toy model's Q(m) law.
	using Model = quokka::ToyStellarModel;
	M_star = C::M_solar * std::pow(Q_target / Model::Q_ion_coeff, 1.0 / Model::Q_ion_exponent);

	const int ncomp_cc = Physics_Indices<StromgrenVolumeProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::foextrap);
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	QuokkaSimulation<StromgrenVolumeProblem> sim(BCs_cc);
	sim.setInitialConditions();
	sim.evolve();

	// --- Validation ---
	int status = 0;

	// The module writes x_ion into passive scalar 0 at the end of every step, so the field is fresh.
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = sim.Geom(0).CellSizeArray();
	const double cell_volume = dx[0] * dx[1] * dx[2];
	const double x_ion_sum = sim.state_new_cc_[0].sum(HydroSystem<StromgrenVolumeProblem>::scalar0_index);
	const double V_ion = x_ion_sum * cell_volume;

	// Total recombination rate inside the flagged region. In a uniform medium this is redundant with
	// the volume check, but with a density gradient it is the only exact statement available: every
	// ionizing photon must be balanced by a recombination somewhere inside the ionized region.
	const double alpha_B_local = alpha_B;
	double recomb_density_sum = amrex::ReduceSum(
	    sim.state_new_cc_[0], 0, [=] AMREX_GPU_HOST_DEVICE(amrex::Box const &bx, amrex::Array4<const amrex::Real> const &arr) -> amrex::Real {
		    amrex::Real local_sum = 0.0;
		    amrex::Loop(bx, [&](int i, int j, int k) noexcept {
			    const amrex::Real rho = arr(i, j, k, HydroSystem<StromgrenVolumeProblem>::density_index);
			    const amrex::Real x_ion = arr(i, j, k, HydroSystem<StromgrenVolumeProblem>::scalar0_index);
			    const amrex::Real n_H_cell = rho / C::m_p;
			    local_sum += x_ion * alpha_B_local * n_H_cell * n_H_cell;
		    });
		    return local_sum;
	    });
	amrex::ParallelDescriptor::ReduceRealSum(recomb_density_sum);
	const double recomb_total = recomb_density_sum * cell_volume;

	const double R_eff = std::cbrt(3.0 * V_ion / (4.0 * M_PI));
	const double R_St = std::cbrt(3.0 * Q_target / (4.0 * M_PI * alpha_B * n_H * n_H));
	const double dx_max = std::max({dx[0], dx[1], dx[2]});

	// Check that Q was assigned at birth from the stellar mass, independently of the volume check.
	const double Q_from_model = Model::ionizingPhotonRate(M_star);
	const double Q_err = std::abs(Q_from_model - Q_target) / Q_target;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "\n=== Stromgren-volume photoionization validation ===\n";
		amrex::Print() << "M_star      = " << M_star / C::M_solar << " M_sun\n";
		amrex::Print() << "Q(M_star)   = " << Q_from_model << " 1/s (target " << Q_target << ")\n";
		amrex::Print() << "R_St        = " << R_St << " cm (" << R_St / dx_max << " cells)\n";
		amrex::Print() << "R_eff       = " << R_eff << " cm (" << R_eff / dx_max << " cells)\n";
		amrex::Print() << "|R_eff - R_St| = " << std::abs(R_eff - R_St) / dx_max << " cells\n";

		if (Q_err > 1.0e-10) {
			status += 1;
			amrex::Print() << "  FAIL: Q(m) does not reproduce the target rate, rel_err=" << Q_err << "\n";
		}
		if (V_ion <= 0.0) {
			status += 1;
			amrex::Print() << "  FAIL: no gas was ionized (x_ion is zero everywhere)\n";
		} else if (density_gradient == 0.0) {
			// Uniform medium: the ionized volume must match the analytic Strömgren volume.
			if (std::abs(R_eff - R_St) > dx_max) {
				status += 1;
				amrex::Print() << "  FAIL: ionized volume radius differs from the analytic Stromgren radius by more than one cell\n";
			}
		} else {
			// Non-uniform medium: the analytic radius no longer applies, but every ionizing photon
			// must still be accounted for by a recombination inside the flagged region.
			const double recomb_err = std::abs(recomb_total - Q_target) / Q_target;
			amrex::Print() << "Total recombination rate in ionized region = " << recomb_total << " 1/s (Q = " << Q_target << ")\n";
			if (recomb_err > 1.0e-6) {
				status += 1;
				amrex::Print() << "  FAIL: photons are not conserved, rel_err=" << recomb_err << "\n";
			}
		}

		amrex::Print() << (status == 0 ? "\n=== All Stromgren-volume checks passed ===\n"
					       : "\n=== Test FAILED (status=" + std::to_string(status) + ") ===\n");
	}

	amrex::ParallelDescriptor::Bcast(&status, 1, amrex::ParallelDescriptor::IOProcessorNumber());
	return status;
}
