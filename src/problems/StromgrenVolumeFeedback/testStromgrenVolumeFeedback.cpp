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
/// The particle is a 30 M_sun star, one of the published anchor points of the Martins, Schaerer &
/// Hillier (2005) calibration used by ToyStellarModel, for which log Q0 = 49.0. The test therefore
/// exercises the real birth-assignment path rather than the Q override, while still knowing Q. It
/// also checks the calibration directly at all three anchors and at the low-mass cutoff.
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

#include <array>
#include <cmath>
#include <string>

using amrex::Real;

struct StromgrenVolumeProblem {
};

// Mass of the ionizing source. 30 M_sun is a Martins et al. (2005) anchor point (log Q0 = 49.0),
// so the photon rate is known independently of the code under test.
constexpr double M_star_in_Msun = 30.0;
// Case-B recombination coefficient at 1e4 K. Must match stromgren.alpha_B in the inputs file.
constexpr double alpha_B = 2.59e-13; // cm^3/s
// Uniform ambient hydrogen number density. Must be consistent with stromgren.hydrogen_mass_fraction=1.
constexpr double n_H = 1.0e3;	      // 1/cm^3
constexpr double rho0 = n_H * C::m_p; // g/cm^3, pure hydrogen
constexpr double T0 = 100.0;	      // K, cold neutral ambient medium

AMREX_GPU_MANAGED double M_star = 0.0; // NOLINT, set in problem_main

// Optional non-uniform ambient medium. The density varies linearly across the box as
//   rho = rho0 * (1 + density_gradient * (x - x_centre) / (L_x / 2)),
// and the source is displaced from the centre by source_offset_frac of the box width. In this
// configuration the analytic Strömgren radius no longer applies, but photon conservation still
// must hold exactly, which is what the test checks.
AMREX_GPU_MANAGED double density_gradient = 0.0;   // NOLINT
AMREX_GPU_MANAGED double source_offset_frac = 0.0; // NOLINT

// Particle file to read. A file containing several co-located particles exercises the overlapping
// -source path, where each source must consume only the recombinations the earlier ones left unpaid.
std::string star_file = "../inputs/star.txt"; // NOLINT
int n_sources = 1;			      // NOLINT, number of particles in star_file

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
		pp.query("star_file", star_file);
		pp.query("n_sources", n_sources);
	}

	using Model = quokka::ToyStellarModel;
	M_star = M_star_in_Msun * C::M_solar;
	// The cubic Q(m) calibration has no closed-form inverse, so the source rate is taken from the
	// model at the chosen mass rather than the mass being solved from a target rate.
	const double Q_target = Model::ionizingPhotonRate(M_star);

	// The module may be given an explicit Q that overrides the stellar model. The analytic radius
	// must then be built from that rate instead of the one implied by the particle mass.
	double Q_override = -1.0;
	{
		amrex::ParmParse const pp_s("stromgren");
		pp_s.query("Q_ion", Q_override);
	}
	const double Q_per_source = (Q_override > 0.0) ? Q_override : Q_target;
	const double Q_total = static_cast<double>(n_sources) * Q_per_source;

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

	// The module writes the ionized fraction into passive scalar 0 at the end of every step, so the
	// field is fresh. Passive scalars are stored as conserved densities, so the slot holds
	// rho * x_ion and must be divided by rho to recover the fraction.
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = sim.Geom(0).CellSizeArray();
	const double cell_volume = dx[0] * dx[1] * dx[2];
	double x_ion_sum = amrex::ReduceSum(sim.state_new_cc_[0], 0,
					    [=] AMREX_GPU_HOST_DEVICE(amrex::Box const &bx, amrex::Array4<const amrex::Real> const &arr) -> amrex::Real {
						    amrex::Real local_sum = 0.0;
						    amrex::Loop(bx, [&](int i, int j, int k) noexcept {
							    const amrex::Real rho = arr(i, j, k, HydroSystem<StromgrenVolumeProblem>::density_index);
							    local_sum += arr(i, j, k, HydroSystem<StromgrenVolumeProblem>::scalar0_index) / rho;
						    });
						    return local_sum;
					    });
	amrex::ParallelDescriptor::ReduceRealSum(x_ion_sum);
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
			    const amrex::Real x_ion = arr(i, j, k, HydroSystem<StromgrenVolumeProblem>::scalar0_index) / rho;
			    const amrex::Real n_H_cell = rho / C::m_p;
			    local_sum += x_ion * alpha_B_local * n_H_cell * n_H_cell;
		    });
		    return local_sum;
	    });
	amrex::ParallelDescriptor::ReduceRealSum(recomb_density_sum);
	const double recomb_total = recomb_density_sum * cell_volume;

	const double R_eff = std::cbrt(3.0 * V_ion / (4.0 * M_PI));
	const double R_St = std::cbrt(3.0 * Q_total / (4.0 * M_PI * alpha_B * n_H * n_H));
	const double dx_max = std::max({dx[0], dx[1], dx[2]});

	const double Q_from_model = Model::ionizingPhotonRate(M_star);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "\n=== Stromgren-volume photoionization validation ===\n";
		amrex::Print() << "M_star      = " << M_star / C::M_solar << " M_sun\n";
		amrex::Print() << "Q(M_star)   = " << Q_from_model << " 1/s (target " << Q_target << ")\n";
		amrex::Print() << "sources     = " << n_sources << ", Q_total = " << Q_total << " 1/s\n";
		amrex::Print() << "R_St        = " << R_St << " cm (" << R_St / dx_max << " cells)\n";
		amrex::Print() << "R_eff       = " << R_eff << " cm (" << R_eff / dx_max << " cells)\n";
		amrex::Print() << "|R_eff - R_St| = " << std::abs(R_eff - R_St) / dx_max << " cells\n";

		// Validate the Martins, Schaerer & Hillier (2005) calibration at its published anchor points,
		// independently of anything the feedback module does with the result.
		struct Anchor {
			double mass_in_Msun;
			double log_Q;
		};
		const std::array<Anchor, 3> anchors = {{{20.0, 48.5}, {30.0, 49.0}, {50.0, 49.5}}};
		for (Anchor const &a : anchors) {
			const double log_Q_model = std::log10(Model::ionizingPhotonRate(a.mass_in_Msun * C::M_solar));
			const double dex_err = std::abs(log_Q_model - a.log_Q);
			amrex::Print() << "  anchor M = " << a.mass_in_Msun << " M_sun: log Q0 = " << log_Q_model << " (expected " << a.log_Q << ")\n";
			if (dex_err > 1.0e-3) {
				status += 1;
				amrex::Print() << "  FAIL: Q(m) misses the Martins+2005 anchor by " << dex_err << " dex\n";
			}
		}

		// Below the fit range the cubic must not be extrapolated: a B star contributes nothing.
		if (Model::ionizingPhotonRate(10.0 * C::M_solar) != 0.0) {
			status += 1;
			amrex::Print() << "  FAIL: a 10 M_sun star was credited with ionizing photons\n";
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
			// When the region is smaller than a cell, a one-cell tolerance is vacuous. This is the
			// subgrid regime the module exists for, so require relative agreement instead.
			if (R_St < dx_max) {
				const double R_rel_err = std::abs(R_eff - R_St) / R_St;
				amrex::Print() << "Subgrid regime (R_St < dx): relative radius error = " << R_rel_err << "\n";
				if (R_rel_err > 1.0e-3) {
					status += 1;
					amrex::Print() << "  FAIL: subgrid ionized volume is wrong, rel_err=" << R_rel_err << "\n";
				}
			}
			// Photon conservation must hold here too, and it is the check that catches a second
			// source being charged for recombinations an earlier source already paid for.
			const double recomb_err = std::abs(recomb_total - Q_total) / Q_total;
			amrex::Print() << "Total recombination rate in ionized region = " << recomb_total << " 1/s (Q_total = " << Q_total << ")\n";
			if (recomb_err > 1.0e-6) {
				status += 1;
				amrex::Print() << "  FAIL: photons are not conserved, rel_err=" << recomb_err << "\n";
			}
		} else {
			// Non-uniform medium: the analytic radius no longer applies, but every ionizing photon
			// must still be accounted for by a recombination inside the flagged region.
			const double recomb_err = std::abs(recomb_total - Q_total) / Q_total;
			amrex::Print() << "Total recombination rate in ionized region = " << recomb_total << " 1/s (Q = " << Q_total << ")\n";
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
