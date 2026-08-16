#ifndef PARTICLE_PHOTOIONIZATION_HPP_
#define PARTICLE_PHOTOIONIZATION_HPP_
//==============================================================================
// Quokka -- two-moment radiation hydrodynamics on AMR grids
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file particle_photoionization.hpp
/// \brief Strömgren-volume photoionization feedback from star particles.
///
/// This is the cheap subgrid alternative to solving the ionizing radiative transfer with
/// the M1 solver (see src/radiation/photochemistry.hpp). No photons are transported. For
/// each star particle we find the volume whose recombinations exactly consume the particle's
/// ionizing photon budget, mark the gas inside as ionized, and hold it at a fixed temperature.
/// The resulting overpressure drives the H II region expansion.
///
/// The method follows the Strömgren-volume technique of Kessel-Deynet & Burkert (2000) and
/// Dale et al. (2007b), in the form used for stellar feedback by Hopkins et al. (2018,
/// FIRE-2, appendix E), and relies on the on-the-spot approximation (hence the case-B
/// recombination coefficient).
///
/// Implementation note: FIRE-2 sorts nearby cells by distance and walks outward consuming the
/// photon budget. Because cells are consumed in order of increasing distance, the consumed set
/// is always a distance-prefix, which is exactly a ball. The walk is therefore equivalent to
/// finding the single radius R_St at which the enclosed recombination rate equals Q. That
/// equivalence is exact for an arbitrary, non-uniform density field. We exploit it to replace
/// the cross-rank cell sort with a radial-bin histogram plus one small MPI reduction per source.

#include <algorithm>
#include <cmath>

#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"

#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "particles/star_particle_indices.H"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"

#if AMREX_SPACEDIM == 3

namespace quokka::photoionization
{

// Number of reals stored per ionizing source in the gathered source list: x, y, z, Q.
constexpr int nSourceComps = 4;

//! Runtime parameters for the Strömgren-volume photoionization module, read from the
//! "stromgren" ParmParse prefix.
struct Parameters {
	//! Master switch. When false, the module is a no-op.
	bool enabled = false;
	//! Temperature imposed on fully ionized gas (K). Note that Quokka's EOS uses a fixed mean
	//! molecular weight, so ionized gas keeps its neutral mu and the overpressure is
	//! underestimated by roughly mu_neutral / mu_ionized ~ 2. Compensate by setting an
	//! effective T_HII of about 2e4 K rather than 1e4 K.
	amrex::Real T_HII = 1.0e4;
	//! Case-B recombination coefficient (cm^3/s). Constant because T_HII is fixed; the default
	//! is the standard value at 1e4 K.
	amrex::Real alpha_B = 2.59e-13;
	//! Hydrogen mass fraction, used to convert mass density to hydrogen number density.
	amrex::Real hydrogen_mass_fraction = 1.0;
	//! Cap on the search radius, in units of the largest cell width. Bounds the cost and the
	//! histogram size. A photon budget not exhausted within this radius is discarded.
	amrex::Real R_max_cells = 32.0;
	//! Passive scalar slot receiving x_ion for plotfile output. Negative disables the output.
	int x_ion_scalar_index = -1;
	//! When positive, overrides the per-particle ionizing photon rate from the stellar model.
	//! Used by the test problem, which needs an exact Q to compare against the analytic radius.
	amrex::Real Q_ion_override = -1.0;
};

//! Read the module parameters from the inputs file.
[[nodiscard]] inline auto readParameters() -> Parameters
{
	Parameters par;
	amrex::ParmParse const pp("stromgren");
	pp.query("enabled", par.enabled);
	pp.query("T_HII", par.T_HII);
	pp.query("alpha_B", par.alpha_B);
	pp.query("hydrogen_mass_fraction", par.hydrogen_mass_fraction);
	pp.query("R_max_cells", par.R_max_cells);
	pp.query("x_ion_scalar_index", par.x_ion_scalar_index);
	pp.query("Q_ion", par.Q_ion_override);
	return par;
}

//! Particle component index holding the birth-assigned ionizing photon rate. The stellar model's
//! extra reals follow the fixed scalars and the nGroups luminosity slots.
template <typename problem_t> [[nodiscard]] constexpr auto ionizingPhotonRateIndex() -> int
{
	return StarParticleLumIdx + Physics_Traits<problem_t>::nGroups + Particle_Traits<problem_t>::stellar_model::QIonExtraOffset;
}

//! Append this rank's star particles at level lev to the source list as (x, y, z, Q) tuples.
template <typename problem_t>
void collectSources(StarParticleContainer<problem_t> *container, amrex::Vector<amrex::Real> &sources, int lev, Parameters const &par)
{
	if (container == nullptr) {
		return;
	}
	constexpr int q_idx = ionizingPhotonRateIndex<problem_t>();
	using ParticleType = typename StarParticleContainer<problem_t>::ParticleType;

	for (StarParticleIterator<problem_t> pti(*container, lev); pti.isValid(); ++pti) {
		auto const &aos = pti.GetArrayOfStructs();
		const int np = static_cast<int>(aos.numParticles());
		if (np == 0) {
			continue;
		}
		// Particle data may live on the device; copy to the host to build the source list.
		amrex::Vector<ParticleType> host_particles(np);
		amrex::Gpu::copy(amrex::Gpu::deviceToHost, aos.begin(), aos.end(), host_particles.begin());

		for (int n = 0; n < np; ++n) {
			ParticleType const &p = host_particles[n];
			if (p.id() <= 0) { // invalidated particle
				continue;
			}
			const amrex::Real Q = (par.Q_ion_override > 0.0) ? par.Q_ion_override : p.rdata(q_idx);
			if (Q <= 0.0) {
				continue;
			}
			sources.push_back(p.pos(0));
			sources.push_back(p.pos(1));
			sources.push_back(p.pos(2));
			sources.push_back(Q);
		}
	}
}

//! Combine the per-rank source lists into a single list held identically on every rank, then order
//! it deterministically by descending Q (position breaks ties). Every rank must iterate the sources
//! in the same order, because each source costs one collective reduction.
inline void gatherAndOrderSources(amrex::Vector<amrex::Real> &sources)
{
	const int nprocs = amrex::ParallelDescriptor::NProcs();
	const int myproc = amrex::ParallelDescriptor::MyProc();

	// Learn every rank's source count by summing a vector in which each rank fills only its own
	// slot. This avoids needing a variable-length all-gather.
	amrex::Vector<int> counts(nprocs, 0);
	counts[myproc] = static_cast<int>(sources.size()) / nSourceComps;
	amrex::ParallelDescriptor::ReduceIntSum(counts.dataPtr(), nprocs);

	int total = 0;
	int my_offset = 0;
	for (int r = 0; r < nprocs; ++r) {
		if (r == myproc) {
			my_offset = total;
		}
		total += counts[r];
	}
	if (total == 0) {
		sources.clear();
		return;
	}

	// Same trick for the payload: write into this rank's slice and sum.
	amrex::Vector<amrex::Real> global(static_cast<std::size_t>(total) * nSourceComps, 0.0);
	std::copy(sources.begin(), sources.end(), global.begin() + (static_cast<std::size_t>(my_offset) * nSourceComps));
	amrex::ParallelDescriptor::ReduceRealSum(global.dataPtr(), static_cast<int>(global.size()));

	// Sort by descending Q; ties broken by position so the order is reproducible.
	amrex::Vector<int> order(total);
	for (int s = 0; s < total; ++s) {
		order[s] = s;
	}
	std::sort(order.begin(), order.end(), [&global](int a, int b) {
		const amrex::Real *pa = &global[static_cast<std::size_t>(a) * nSourceComps];
		const amrex::Real *pb = &global[static_cast<std::size_t>(b) * nSourceComps];
		if (pa[3] != pb[3]) {
			return pa[3] > pb[3];
		}
		for (int d = 0; d < 3; ++d) {
			if (pa[d] != pb[d]) {
				return pa[d] < pb[d];
			}
		}
		return false;
	});

	sources.resize(static_cast<std::size_t>(total) * nSourceComps);
	for (int s = 0; s < total; ++s) {
		for (int c = 0; c < nSourceComps; ++c) {
			sources[(static_cast<std::size_t>(s) * nSourceComps) + c] = global[(static_cast<std::size_t>(order[s]) * nSourceComps) + c];
		}
	}
}

//! Apply Strömgren-volume photoionization feedback to the gas.
//!
//! For each source in turn, accumulate the recombination rate of the still-neutral gas into radial
//! bins, reduce across ranks, and walk the cumulative sum outward until the photon budget Q is
//! spent. Cells inside the termination bin become fully ionized; the termination bin itself takes a
//! uniform fractional ionization. Cells already ionized by a brighter source are skipped and are not
//! charged against this source's budget, so photons are never spent twice.
//!
//! Once every source has been processed, ionized gas is heated toward T_HII.
template <typename problem_t>
void applyStromgrenFeedback(amrex::MultiFab &state, amrex::Vector<amrex::Real> const &sources, amrex::Geometry const &geom, Parameters const &par)
{
	const BL_PROFILE("quokka::photoionization::applyStromgrenFeedback()");

	const int n_src = static_cast<int>(sources.size()) / nSourceComps;
	if (!par.enabled || n_src == 0) {
		return;
	}

	// MHD is not supported: the internal energy below would need the magnetic contribution removed.
	// This must stay a runtime check -- a static_assert would fire when the header is instantiated
	// for an MHD problem even if the module is switched off.
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Physics_Indices<problem_t>::nvarTotal_fc == 0,
					 "Stromgren-volume photoionization feedback does not support MHD (face-centred variables) in this version.");

	using Hydro = HydroSystem<problem_t>;
	const auto dx = geom.CellSizeArray();
	const auto plo = geom.ProbLoArray();
	const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	const amrex::Real dr = 0.5 * std::min({dx[0], dx[1], dx[2]});
	const amrex::Real R_max = par.R_max_cells * std::max({dx[0], dx[1], dx[2]});
	const int n_bins = std::max(1, static_cast<int>(std::ceil(R_max / dr)));

	// Ionized fraction, rebuilt from scratch every step. It is never advected: it is a diagnostic
	// derived from the current density field and the current photon budgets.
	amrex::MultiFab xion(state.boxArray(), state.DistributionMap(), 1, 0);
	xion.setVal(0.0);

	const amrex::Real nH_per_rho = par.hydrogen_mass_fraction / C::m_p;
	const amrex::Real alpha_B = par.alpha_B;
	int n_unbounded = 0;

	for (int s = 0; s < n_src; ++s) {
		const amrex::Real src_x = sources[(static_cast<std::size_t>(s) * nSourceComps) + 0];
		const amrex::Real src_y = sources[(static_cast<std::size_t>(s) * nSourceComps) + 1];
		const amrex::Real src_z = sources[(static_cast<std::size_t>(s) * nSourceComps) + 2];
		const amrex::Real Q = sources[(static_cast<std::size_t>(s) * nSourceComps) + 3];

		// --- Pass 1: bin the remaining (un-ionized) recombination rate by distance ---
		amrex::Gpu::DeviceVector<amrex::Real> d_bins(n_bins, 0.0);
		amrex::Real *p_bins = d_bins.data();

		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.tilebox();
			auto const &cons = state.const_array(mfi);
			auto const &xi = xion.const_array(mfi);

			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real x = plo[0] + ((static_cast<amrex::Real>(i) + 0.5) * dx[0]) - src_x;
				const amrex::Real y = plo[1] + ((static_cast<amrex::Real>(j) + 0.5) * dx[1]) - src_y;
				const amrex::Real z = plo[2] + ((static_cast<amrex::Real>(k) + 0.5) * dx[2]) - src_z;
				const amrex::Real r = std::sqrt((x * x) + (y * y) + (z * z));
				if (r >= R_max) {
					return;
				}
				const amrex::Real neutral_frac = 1.0 - xi(i, j, k);
				if (neutral_frac <= 0.0) {
					return;
				}
				// Recombination rate assuming the gas inside the region is fully ionized
				// (n_e = n_H+ = n_H). This is the standard Strömgren closure; it removes what
				// would otherwise be a circular dependence of x_ion on itself.
				const amrex::Real n_H = cons(i, j, k, Hydro::density_index) * nH_per_rho;
				const amrex::Real capacity = neutral_frac * alpha_B * n_H * n_H * cell_volume;
				if (capacity <= 0.0) {
					return;
				}
				const int bin = amrex::min(static_cast<int>(r / dr), n_bins - 1);
				amrex::HostDevice::Atomic::Add(&p_bins[bin], capacity);
			});
		}

		amrex::Vector<amrex::Real> bins(n_bins, 0.0);
		amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_bins.begin(), d_bins.end(), bins.begin());
		amrex::ParallelDescriptor::ReduceRealSum(bins.dataPtr(), n_bins);

		// --- Walk the cumulative sum outward until the photon budget is spent ---
		amrex::Real remaining = Q;
		int bin_star = n_bins; // first bin not fully ionized
		amrex::Real frac_star = 0.0;
		for (int b = 0; b < n_bins; ++b) {
			if (remaining >= bins[b]) {
				remaining -= bins[b];
			} else {
				bin_star = b;
				frac_star = remaining / bins[b];
				remaining = 0.0;
				break;
			}
		}
		if (bin_star == n_bins && remaining > 0.0) {
			// Budget survives to R_max. As in FIRE-1, the remainder is discarded: there is no
			// long-range transfer step to hand it to.
			++n_unbounded;
		}

		// --- Pass 2: write the ionized fraction ---
		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.tilebox();
			auto const &xi = xion.array(mfi);

			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real x = plo[0] + ((static_cast<amrex::Real>(i) + 0.5) * dx[0]) - src_x;
				const amrex::Real y = plo[1] + ((static_cast<amrex::Real>(j) + 0.5) * dx[1]) - src_y;
				const amrex::Real z = plo[2] + ((static_cast<amrex::Real>(k) + 0.5) * dx[2]) - src_z;
				const amrex::Real r = std::sqrt((x * x) + (y * y) + (z * z));
				if (r >= R_max) {
					return;
				}
				const int bin = amrex::min(static_cast<int>(r / dr), n_bins - 1);
				const amrex::Real neutral_frac = 1.0 - xi(i, j, k);
				if (bin < bin_star) {
					xi(i, j, k) = 1.0;
				} else if (bin == bin_star) {
					xi(i, j, k) += frac_star * neutral_frac;
				}
			});
		}
	}

	if (n_unbounded > 0 && amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "[STROMGREN] Warning: " << n_unbounded
			       << " source(s) did not exhaust their ionizing photon budget within R_max = " << par.R_max_cells
			       << " cells; the remainder was discarded.\n";
	}

	// --- Heat the ionized gas toward T_HII ---
	const amrex::Real T_HII = par.T_HII;
	const int x_ion_scalar_index = par.x_ion_scalar_index;
	constexpr int nscalars = Hydro::nscalars_;

	for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.tilebox();
		auto const &cons = state.array(mfi);
		auto const &xi = xion.const_array(mfi);

		amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real x_ion = xi(i, j, k);

			if (x_ion > 0.0) {
				const amrex::Real rho = cons(i, j, k, Hydro::density_index);
				const amrex::Real px = cons(i, j, k, Hydro::x1Momentum_index);
				const amrex::Real py = cons(i, j, k, Hydro::x2Momentum_index);
				const amrex::Real pz = cons(i, j, k, Hydro::x3Momentum_index);
				const amrex::Real Ekin = 0.5 * ((px * px) + (py * py) + (pz * pz)) / rho;
				const amrex::Real Eint_old = cons(i, j, k, Hydro::energy_index) - Ekin;

				const amrex::GpuArray<amrex::Real, Hydro::nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
				const amrex::Real Eint_target = quokka::EOS<problem_t>::ComputeEintFromTgas(rho, T_HII, massScalars);

				// Heat only: gas already hotter than T_HII (shocked or supernova-heated) is left
				// alone rather than being artificially cooled. Reapplying this every step is what
				// makes the H II region hold its temperature, so no separate cooling switch is needed.
				const amrex::Real dEint = x_ion * amrex::max(0.0, Eint_target - Eint_old);
				cons(i, j, k, Hydro::energy_index) += dEint;
				cons(i, j, k, Hydro::internalEnergy_index) += dEint;
			}

			if constexpr (nscalars > 0) {
				if (x_ion_scalar_index >= 0 && x_ion_scalar_index < nscalars) {
					cons(i, j, k, Hydro::scalar0_index + x_ion_scalar_index) = x_ion;
				}
			}
		});
	}
}

} // namespace quokka::photoionization

#endif // AMREX_SPACEDIM == 3

#endif // PARTICLE_PHOTOIONIZATION_HPP_
