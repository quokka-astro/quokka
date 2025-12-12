//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testHydroWave.cpp
/// \brief Defines a test problem for a linear hydro wave.
///

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include <complex>
#include <fmt/format.h>
#include <fstream>
#include <valarray>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_GpuAtomic.H"
#include "AMReX_MFIter.H"
#include "AMReX_Math.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_ParticleMesh.H"
#include "AMReX_REAL.H"
#include "AMReX_Random.H"
#include "AMReX_Reduce.H"

#include "hydro/EOS.hpp"
#include "physics_info.hpp"

struct WaveProblem {
};

namespace
{
amrex::Real wave_amplitude = 1.0e-6;
int tracer_multiplier = 1;
amrex::Real wave_error_tol = 1.0e-8;
} // namespace

template <> struct quokka::EOS_Traits<WaveProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<WaveProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"

class WaveSimulation : public QuokkaSimulation<WaveProblem>
{
      public:
	using QuokkaSimulation<WaveProblem>::QuokkaSimulation;

	auto tracerContainer() -> amrex::AmrTracerParticleContainer * { return TracerPC.get(); }

	[[nodiscard]] auto suppressOutput() const -> bool { return suppress_output != 0; }
};

constexpr double rho0 = 1.0;					    // background density
constexpr double P0 = 1.0 / quokka::EOS_Traits<WaveProblem>::gamma; // background pressure
constexpr double v0 = 0.;					    // background velocity

AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real amplitude)
{
	const amrex::Real x_L = prob_lo[0] + (i + static_cast<amrex::Real>(0.0)) * dx[0];
	const amrex::Real x_R = prob_lo[0] + (i + static_cast<amrex::Real>(1.0)) * dx[0];
	const amrex::Real A = amplitude;

	const quokka::valarray<double, 3> R = {1.0, -1.0, 1.5}; // right eigenvector of sound wave
	const quokka::valarray<double, 3> U_0 = {rho0, rho0 * v0, P0 / (quokka::EOS_Traits<WaveProblem>::gamma - 1.0) + 0.5 * rho0 * std::pow(v0, 2)};
	const quokka::valarray<double, 3> dU = (A * R / (2.0 * M_PI * dx[0])) * (std::cos(2.0 * M_PI * x_L) - std::cos(2.0 * M_PI * x_R));

	double const rho = U_0[0] + dU[0];
	double const xmom = U_0[1] + dU[1];
	double const Etot = U_0[2] + dU[2];
	double const Eint = Etot - 0.5 * (xmom * xmom) / rho;

	state(i, j, k, HydroSystem<WaveProblem>::density_index) = rho;
	state(i, j, k, HydroSystem<WaveProblem>::x1Momentum_index) = xmom;
	state(i, j, k, HydroSystem<WaveProblem>::x2Momentum_index) = 0;
	state(i, j, k, HydroSystem<WaveProblem>::x3Momentum_index) = 0;
	state(i, j, k, HydroSystem<WaveProblem>::energy_index) = Etot;
	state(i, j, k, HydroSystem<WaveProblem>::internalEnergy_index) = Eint;
}

template <> void QuokkaSimulation<WaveProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<WaveProblem>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused components with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, wave_amplitude);
	});
}

auto problem_main() -> int
{
	// Based on the ATHENA test page:
	// https://www.astro.princeton.edu/~jstone/Athena/tests/linear-waves/linear-waves.html

	// Problem parameters
	// const int nx = 100;
	// const double Lx = 1.0;
	const double CFL_number = 0.1;
	double max_time = 1.0;
	int max_timesteps = 20000;

	// Problem initialization
	auto BCs_cc = quokka::BC<WaveProblem>(quokka::BCType::int_dir); // periodic

	WaveSimulation sim(BCs_cc);

	amrex::ParmParse hydro_pp("hydro");
	hydro_pp.query("wave_amp", wave_amplitude);
	hydro_pp.query("wave_err_tol", wave_error_tol);
	hydro_pp.query("tracer_multiplier", tracer_multiplier);
	amrex::ParmParse pp;
	pp.query("stop_time", max_time);
	pp.query("max_timesteps", max_timesteps);
	amrex::InitRandom(1234, amrex::ParallelDescriptor::NProcs());

	sim.cflNumber_ = CFL_number;
	pp.query("cfl", sim.cflNumber_);
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;
	// sim.plotfileInterval_ = -1; //moved to .in file

	// set initial conditions
	sim.setInitialConditions();
	amrex::Real initial_mass_centroid = 0.0;
	if (sim.do_tracers != 0) {
		const int lev = 0;
		const auto &geom = sim.geom[lev];
		const auto domain = geom.Domain();
		const auto prob_lo = geom.ProbLoArray();
		const auto dx = geom.CellSizeArray();
		amrex::Real cv = dx[0];
#if (AMREX_SPACEDIM >= 2)
		cv *= dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
		cv *= dx[2];
#endif
		amrex::Real total_mass = 0.0;
		amrex::Real moment = 0.0;
		for (amrex::MFIter mfi(sim.state_new_cc_[lev]); mfi.isValid(); ++mfi) {
			auto const arr = sim.state_new_cc_[lev].const_array(mfi);
			const amrex::Box &bx = mfi.validbox();
			for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
#if (AMREX_SPACEDIM >= 2)
				for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
#if (AMREX_SPACEDIM == 3)
					for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
						const amrex::Real rho = arr(i, j, k, HydroSystem<WaveProblem>::density_index);
						const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + static_cast<amrex::Real>(0.5)) * dx[0];
						const amrex::Real mass = rho * cv;
						total_mass += mass;
						moment += mass * x;
					}
#else
					const amrex::Real rho = arr(i, j, 0, HydroSystem<WaveProblem>::density_index);
					const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + static_cast<amrex::Real>(0.5)) * dx[0];
					const amrex::Real mass = rho * cv;
					total_mass += mass;
					moment += mass * x;
#endif
				}
#else
				const amrex::Real rho = arr(i, 0, 0, HydroSystem<WaveProblem>::density_index);
				const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + static_cast<amrex::Real>(0.5)) * dx[0];
				const amrex::Real mass = rho * cv;
				total_mass += mass;
				moment += mass * x;
#endif
			}
		}
		amrex::ParallelDescriptor::ReduceRealSum(&total_mass, 1);
		amrex::ParallelDescriptor::ReduceRealSum(&moment, 1);
		if (total_mass > 0.0) {
			initial_mass_centroid = moment / total_mass;
		}
		if (tracer_multiplier > 0) {
			auto *tracer_pc = sim.tracerContainer();
			AMREX_ALWAYS_ASSERT(tracer_pc != nullptr);
			tracer_pc->clearParticles();

			struct CellSeed {
				amrex::IntVect iv;
				int grid;
				int tile;
				amrex::Real frac;
				int count;
			};

			const amrex::Long num_cells = domain.numPts();
			const amrex::Real rho_mean = total_mass / (static_cast<amrex::Real>(num_cells) * cv);
			AMREX_ALWAYS_ASSERT(rho_mean > 0.0);

			std::vector<CellSeed> seeds;
			seeds.reserve(static_cast<std::size_t>(num_cells));

			for (amrex::MFIter mfi(sim.state_new_cc_[lev]); mfi.isValid(); ++mfi) {
				const auto arr = sim.state_new_cc_[lev].const_array(mfi);
				const amrex::Box &bx = mfi.validbox();
				for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
#if (AMREX_SPACEDIM >= 2)
					for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
#if (AMREX_SPACEDIM == 3)
						for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
							const amrex::Real rho = arr(i, j, k, HydroSystem<WaveProblem>::density_index);
							const amrex::Real expected = tracer_multiplier * (rho / rho_mean);
							int base = static_cast<int>(amrex::Math::floor(expected));
							amrex::Real frac = expected - static_cast<amrex::Real>(base);
							seeds.push_back(
							    CellSeed{amrex::IntVect(AMREX_D_DECL(i, j, k)), mfi.index(), mfi.LocalTileIndex(), frac, base});
						}
#else
						const amrex::Real rho = arr(i, j, 0, HydroSystem<WaveProblem>::density_index);
						const amrex::Real expected = tracer_multiplier * (rho / rho_mean);
						int base = static_cast<int>(amrex::Math::floor(expected));
						amrex::Real frac = expected - static_cast<amrex::Real>(base);
						seeds.push_back(CellSeed{amrex::IntVect(AMREX_D_DECL(i, j, 0)), mfi.index(), mfi.LocalTileIndex(), frac, base});
#endif
					}
#else
					const amrex::Real rho = arr(i, 0, 0, HydroSystem<WaveProblem>::density_index);
					const amrex::Real expected = tracer_multiplier * (rho / rho_mean);
					int base = static_cast<int>(amrex::Math::floor(expected));
					amrex::Real frac = expected - static_cast<amrex::Real>(base);
					seeds.push_back(CellSeed{amrex::IntVect(AMREX_D_DECL(i, 0, 0)), mfi.index(), mfi.LocalTileIndex(), frac, base});
#endif
				}
			}

			amrex::Long total_base = 0;
			for (auto const &s : seeds) {
				total_base += static_cast<amrex::Long>(s.count);
			}
			const amrex::Long desired_total = static_cast<amrex::Long>(tracer_multiplier) * num_cells;
			amrex::Long deficit = desired_total - total_base;
			if (deficit > 0) {
				std::sort(seeds.begin(), seeds.end(), [](CellSeed const &a, CellSeed const &b) { return a.frac > b.frac; });
				for (amrex::Long n = 0; n < deficit && n < static_cast<amrex::Long>(seeds.size()); ++n) {
					seeds[static_cast<std::size_t>(n)].count += 1;
				}
			} else if (deficit < 0) {
				deficit = -deficit;
				std::sort(seeds.begin(), seeds.end(), [](CellSeed const &a, CellSeed const &b) { return a.frac < b.frac; });
				for (auto &s : seeds) {
					if (deficit == 0) {
						break;
					}
					if (s.count > 0) {
						s.count -= 1;
						--deficit;
					}
				}
			}

			auto pid = amrex::AmrTracerParticleContainer::ParticleType::NextID();
			amrex::Long total_added = 0;
			for (amrex::MFIter mfi(tracer_pc->MakeMFIter(lev)); mfi.isValid(); ++mfi) {
				const int grid = mfi.index();
				const int tile = mfi.LocalTileIndex();
				int add_here = 0;
				for (auto const &s : seeds) {
					if (s.grid == grid && s.tile == tile) {
						add_here += s.count;
					}
				}
				if (add_here == 0) {
					continue;
				}

				auto &particle_tile = tracer_pc->DefineAndReturnParticleTile(lev, mfi);
				auto &aos = particle_tile.GetArrayOfStructs();
				int const old_size = aos.size();
				aos.resize(old_size + add_here);

				int p_index = old_size;
				const int cpu = amrex::ParallelDescriptor::MyProc();
				for (auto const &s : seeds) {
					if (s.grid != grid || s.tile != tile || s.count == 0) {
						continue;
					}
					const int i = s.iv[0];
#if (AMREX_SPACEDIM >= 2)
					const int j = s.iv[1];
#else
					constexpr int j = 0;
#endif
#if (AMREX_SPACEDIM == 3)
					const int k = s.iv[2];
#else
					constexpr int k = 0;
#endif
					const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + static_cast<amrex::Real>(0.5)) * dx[0];
#if (AMREX_SPACEDIM >= 2)
					const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + static_cast<amrex::Real>(0.5)) * dx[1];
#else
					constexpr amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
					const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + static_cast<amrex::Real>(0.5)) * dx[2];
#else
					constexpr amrex::Real z = 0.0;
#endif
					for (int n = 0; n < s.count; ++n) {
						auto &p = aos[p_index++];
						p.id() = pid++;
						p.cpu() = cpu;
						p.pos(0) = x;
#if (AMREX_SPACEDIM >= 2)
						p.pos(1) = y;
#endif
#if (AMREX_SPACEDIM == 3)
						p.pos(2) = z;
#endif
					}
				}
				total_added += add_here;
			}
			amrex::AmrTracerParticleContainer::ParticleType::NextID(pid);
			amrex::ParallelDescriptor::ReduceLongSum(total_added);
			if (amrex::ParallelDescriptor::IOProcessor()) {
				if (total_added != desired_total) {
					amrex::Print() << "[Tracer init] expected " << desired_total << " tracers, got " << total_added << " (multiplier "
						       << tracer_multiplier << ")\n";
				}
			}
			tracer_pc->Redistribute();
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(total_added > 0, "Tracer seeding failed: no particles inserted");
		}
	}
	auto [pos_exact, val_exact] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);

	// Main time loop
	sim.evolve();

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);
	int const nx = static_cast<int>(position.size());
	std::vector<double> const xs = position;
	std::vector<double> tracer_profile;
	amrex::Real tracer_mean = 0.0;
	bool have_tracer_profile = false;
	amrex::Real total_mass = 0.0;
	amrex::Real total_tracers = 0.0;

	// compute error norm
	amrex::Real err_sq = 0.;
	for (int n = 0; n < QuokkaSimulation<WaveProblem>::nvars_; ++n) {
		if (n == HydroSystem<WaveProblem>::internalEnergy_index) {
			continue;
		}
		amrex::Real dU_k = 0.;
		for (int i = 0; i < nx; ++i) {
			// Δ Uk = ∑i |Uk,in - Uk,i0| / Nx
			const amrex::Real U_k0 = val_exact.at(n)[i];
			const amrex::Real U_k1 = values.at(n)[i];
			dU_k += std::abs(U_k1 - U_k0) / static_cast<double>(nx);
		}
		// ε = || Δ U || = [&sum_k (Δ Uk)2]^{1/2}
		err_sq += dU_k * dU_k;
	}
	const amrex::Real epsilon = std::sqrt(err_sq);
	amrex::Print() << "rms of component-wise L1 error norms = " << epsilon << '\n';

	if (sim.do_tracers != 0) {
		const int lev = 0;
		const auto dx = sim.geom[lev].CellSizeArray();
		amrex::Real cell_vol = dx[0];
#if (AMREX_SPACEDIM >= 2)
		cell_vol *= dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
		cell_vol *= dx[2];
#endif

		// ensure particles are on the owning grids before analysis
		auto *tracer_pc = sim.tracerContainer();
		AMREX_ALWAYS_ASSERT(tracer_pc != nullptr);
		tracer_pc->Redistribute(lev);

		amrex::MultiFab tracer_counts(sim.boxArray(lev), sim.DistributionMap(lev), 1, 0);
		amrex::ParticleToMesh(*tracer_pc, tracer_counts, lev,
				      [] AMREX_GPU_DEVICE(auto const &p, amrex::Array4<amrex::Real> const &fab, auto const &plo, auto const &dxi) noexcept {
					      const int i = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]));
#if (AMREX_SPACEDIM >= 2)
					      const int j = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]));
#else
					      const int j = 0;
#endif
#if (AMREX_SPACEDIM == 3)
					      const int k = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]));
#else
					      const int k = 0;
#endif
					      amrex::Gpu::Atomic::Add(&fab(i, j, k, 0), 1.0);
				      });
		amrex::MFInfo host_info;
		host_info.SetArena(amrex::The_Pinned_Arena());
		amrex::MultiFab tracer_counts_host(sim.boxArray(lev), sim.DistributionMap(lev), 1, 0, host_info);
		amrex::MultiFab::Copy(tracer_counts_host, tracer_counts, 0, 0, 1, 0);

		total_tracers = tracer_counts_host.sum(0);
		total_mass = sim.state_new_cc_[lev].sum(HydroSystem<WaveProblem>::density_index) * cell_vol;

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(total_tracers > 0.0, "Tracer count should be positive when do_tracers != 0");
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(total_mass > 0.0, "Total mass must be positive");

		const amrex::Real inv_total_mass = 1.0 / total_mass;
		const amrex::Real inv_total_tracers = 1.0 / total_tracers;

		amrex::Real sum_mass_sq = 0.0;
		amrex::Real l1_error = 0.0;
		for (amrex::MFIter mfi(sim.state_new_cc_[lev]); mfi.isValid(); ++mfi) {
			const auto state_arr = sim.state_new_cc_[lev].const_array(mfi);
			const auto tracer_arr = tracer_counts_host.const_array(mfi);
			const amrex::Box &bx = mfi.validbox();
			amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
			amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_ops);
			reduce_ops.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real, amrex::Real> {
				amrex::Real const mass = state_arr(i, j, k, HydroSystem<WaveProblem>::density_index) * cell_vol;
				amrex::Real const tracer = tracer_arr(i, j, k, 0);
				amrex::Real const pmass = mass * inv_total_mass;
				amrex::Real const ptracer = tracer * inv_total_tracers;
				amrex::Real const l1 = amrex::Math::abs(ptracer - pmass);
				return {mass * mass, l1};
			});
			auto const reduce_vals = reduce_data.value();
			sum_mass_sq += amrex::get<0>(reduce_vals);
			l1_error += amrex::get<1>(reduce_vals);
		}
		const amrex::Real mass_prob_sq_sum = sum_mass_sq * (inv_total_mass * inv_total_mass);
		// binomial sampling variance, scaled by number of time steps (random walk)
		amrex::Real sigma = std::sqrt((1.0 - mass_prob_sq_sum) * inv_total_tracers);
		sigma *= std::sqrt(static_cast<amrex::Real>(sim.istep[0]));
		// allow occasional large deviations (4-sigma)
		const amrex::Real sigma_limit = 4.0 * sigma + 1.0e-14;
		amrex::Print() << "Tracer-mass L1 error = " << l1_error << " after " << sim.istep[0] << " steps (limit = " << sigma_limit << ")\n";
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(l1_error < sigma_limit, "Monte Carlo tracer distribution deviates beyond expected noise floor");

		// build 1D tracer profile summed over transverse directions
		const int nx_line = sim.geom[lev].Domain().length(0);
		tracer_profile.assign(nx_line, 0.0);
		for (amrex::MFIter mfi(tracer_counts_host); mfi.isValid(); ++mfi) {
			const auto tracer_arr = tracer_counts_host.const_array(mfi);
			const amrex::Box &bx = mfi.validbox();
			for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
#if (AMREX_SPACEDIM >= 2)
				for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
#if (AMREX_SPACEDIM == 3)
					for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
						tracer_profile.at(i) += tracer_arr(i, j, k, 0);
					}
#else
					tracer_profile.at(i) += tracer_arr(i, j, 0, 0);
#endif
				}
#else
				tracer_profile.at(i) += tracer_arr(i, 0, 0, 0);
#endif
			}
		}
		tracer_mean = total_tracers / static_cast<amrex::Real>(nx_line);
		amrex::ParallelDescriptor::ReduceRealSum(tracer_profile.data(), static_cast<int>(tracer_profile.size()));
		have_tracer_profile = true;

		// compute tracer and mass centroids for a simple phase diagnostic
		const auto prob_lo = sim.geom[lev].ProbLoArray();
		amrex::Real mass_moment = 0.0;
		amrex::Real tracer_moment = 0.0;
		amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> moment_ops;
		for (amrex::MFIter mfi(sim.state_new_cc_[lev]); mfi.isValid(); ++mfi) {
			const auto state_arr = sim.state_new_cc_[lev].const_array(mfi);
			const auto tracer_arr = tracer_counts.const_array(mfi);
			const amrex::Box &bx = mfi.validbox();
			amrex::ReduceData<amrex::Real, amrex::Real> moment_data(moment_ops);
			moment_ops.eval(bx, moment_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real, amrex::Real> {
				const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + static_cast<amrex::Real>(0.5)) * dx[0];
				const amrex::Real mass = state_arr(i, j, k, HydroSystem<WaveProblem>::density_index) * cell_vol;
				const amrex::Real tracer = tracer_arr(i, j, k, 0);
				return {mass * x, tracer * x};
			});
			auto const reduce_vals = moment_data.value();
			mass_moment += amrex::get<0>(reduce_vals);
			tracer_moment += amrex::get<1>(reduce_vals);
		}
		amrex::ParallelDescriptor::ReduceRealSum(&mass_moment, 1);
		amrex::ParallelDescriptor::ReduceRealSum(&tracer_moment, 1);
		amrex::Real const mass_centroid = mass_moment / total_mass;
		amrex::Real const tracer_centroid = tracer_moment / total_tracers;
		amrex::Real centroid_delta = tracer_centroid - mass_centroid;
		amrex::Real const Lx = sim.geom[lev].ProbLength(0);
		centroid_delta -= Lx * std::round(centroid_delta / Lx); // wrap into domain
		amrex::Print() << "Mass centroid: initial " << initial_mass_centroid << ", final " << mass_centroid << "; tracer centroid " << tracer_centroid
			       << ", delta = " << centroid_delta << " (" << centroid_delta / dx[0] << " cells)\n";
	}

#ifdef HAVE_PYTHON
	// plot results; always plot tracer overlay when available so we can inspect particle–mass agreement
	if (amrex::ParallelDescriptor::IOProcessor() && (have_tracer_profile || !sim.suppressOutput())) {
		// extract values
		std::vector<double> d(nx);
		std::vector<double> vx(nx);
		std::vector<double> P(nx);

		for (int i = 0; i < nx; ++i) {
			amrex::Real const rho = values.at(HydroSystem<WaveProblem>::density_index)[i];
			amrex::Real const xmom = values.at(HydroSystem<WaveProblem>::x1Momentum_index)[i];
			amrex::Real const Egas = values.at(HydroSystem<WaveProblem>::energy_index)[i];

			amrex::Real const xvel = xmom / rho;
			amrex::Real const Eint = Egas - xmom * xmom / (2.0 * rho);
			amrex::Real const pressure = Eint * (quokka::EOS_Traits<WaveProblem>::gamma - 1.);

			d.at(i) = (rho - rho0) / wave_amplitude;
			vx.at(i) = (xvel - v0) / wave_amplitude;
			P.at(i) = (pressure - P0) / wave_amplitude;
		}

		std::vector<double> density_exact(nx);
		std::vector<double> velocity_exact(nx);
		std::vector<double> pressure_exact(nx);

		for (int i = 0; i < nx; ++i) {
			amrex::Real const rho = val_exact.at(HydroSystem<WaveProblem>::density_index)[i];
			amrex::Real const xmom = val_exact.at(HydroSystem<WaveProblem>::x1Momentum_index)[i];
			amrex::Real const Egas = val_exact.at(HydroSystem<WaveProblem>::energy_index)[i];

			amrex::Real const xvel = xmom / rho;
			amrex::Real const Eint = Egas - xmom * xmom / (2.0 * rho);
			amrex::Real const pressure = Eint * (quokka::EOS_Traits<WaveProblem>::gamma - 1.);

			density_exact.at(i) = (rho - rho0) / wave_amplitude;
			velocity_exact.at(i) = (xvel - v0) / wave_amplitude;
			pressure_exact.at(i) = (pressure - P0) / wave_amplitude;
		}

		// Plot results
		amrex::Real const t = sim.tNew_[0];

		std::map<std::string, std::string> d_args;
		std::map<std::string, std::string> dinit_args;
		std::map<std::string, std::string> const dexact_args;
		d_args["label"] = "density";
		dinit_args["label"] = "density (initial)";

		matplotlibcpp::clf();
		matplotlibcpp::plot(xs, d, d_args);
		matplotlibcpp::plot(xs, density_exact, dinit_args);
		if (have_tracer_profile) {
			AMREX_ALWAYS_ASSERT(static_cast<int>(tracer_profile.size()) == nx);
			std::vector<double> tracer_rel(nx);
			for (int i = 0; i < nx; ++i) {
				tracer_rel[i] = (tracer_profile[i] / tracer_mean - 1.0) / wave_amplitude;
			}
			std::map<std::string, std::string> tracer_args;
			tracer_args["label"] = "tracer (deposited)";
			tracer_args["linestyle"] = "--";
			matplotlibcpp::plot(xs, tracer_rel, tracer_args);
		}
		matplotlibcpp::legend();
		matplotlibcpp::title(fmt::format("t = {:.4f}", t));
		matplotlibcpp::save(fmt::format("./density_{:.4f}.pdf", t));

		std::map<std::string, std::string> P_args;
		std::map<std::string, std::string> Pinit_args;
		std::map<std::string, std::string> const Pexact_args;
		P_args["label"] = "pressure";
		Pinit_args["label"] = "pressure (initial)";

		matplotlibcpp::clf();
		matplotlibcpp::plot(xs, P, P_args);
		matplotlibcpp::plot(xs, pressure_exact, Pinit_args);
		matplotlibcpp::legend();
		matplotlibcpp::title(fmt::format("t = {:.4f}", t));
		matplotlibcpp::save(fmt::format("./pressure_{:.4f}.pdf", t));

		std::map<std::string, std::string> v_args;
		std::map<std::string, std::string> vinit_args;
		std::map<std::string, std::string> const vexact_args;
		v_args["label"] = "velocity";
		vinit_args["label"] = "velocity (initial)";

		matplotlibcpp::clf();
		matplotlibcpp::plot(xs, vx, v_args);
		matplotlibcpp::plot(xs, velocity_exact, vinit_args);
		matplotlibcpp::legend();
		matplotlibcpp::title(fmt::format("t = {:.4f}", t));
		matplotlibcpp::save(fmt::format("./velocity_{:.4f}.pdf", t));
	}
#endif

	if (amrex::ParallelDescriptor::IOProcessor() && have_tracer_profile) {
		const auto &geom = sim.geom[0];
		const auto domain = geom.Domain();
		int n_transverse = 1;
#if (AMREX_SPACEDIM >= 2)
		n_transverse *= domain.length(1);
#endif
#if (AMREX_SPACEDIM == 3)
		n_transverse *= domain.length(2);
#endif
		const auto dx = geom.CellSizeArray();
		amrex::Real cv = dx[0];
#if (AMREX_SPACEDIM >= 2)
		cv *= dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
		cv *= dx[2];
#endif
		const amrex::Real mean_density = total_mass / (static_cast<amrex::Real>(domain.numPts()) * cv);
		const amrex::Real tracer_mean_per_cell = tracer_mean / static_cast<amrex::Real>(n_transverse);
		const amrex::Real Lx = geom.ProbLength(0);
		const amrex::Real kfund = 2.0 * M_PI / Lx;

		std::ofstream csv("hydro_wave_tracer_profile.csv");
		csv << "x,rho,tracer_per_cell,tracer_fraction,mass_fraction,relative_error\n";
		std::complex<double> mass_mode(0.0, 0.0);
		std::complex<double> tracer_mode(0.0, 0.0);
		for (int i = 0; i < nx; ++i) {
			const amrex::Real rho = values.at(HydroSystem<WaveProblem>::density_index)[i];
			const amrex::Real tracer_col = tracer_profile[i];
			const amrex::Real tracer_per_cell = tracer_col / static_cast<amrex::Real>(n_transverse);
			const amrex::Real tracer_frac = tracer_col / total_tracers;
			const amrex::Real mass_col = rho * cv * static_cast<amrex::Real>(n_transverse);
			const amrex::Real mass_frac = mass_col / total_mass;
			const amrex::Real rel_err = (tracer_frac / mass_frac) - 1.0;
			const amrex::Real x = xs[i];
			const std::complex<double> phase_factor(0.0, -static_cast<double>(kfund * x));
			mass_mode += std::exp(phase_factor) * static_cast<double>(mass_frac);
			tracer_mode += std::exp(phase_factor) * static_cast<double>(tracer_frac);
			csv << xs[i] << ',' << rho << ',' << tracer_per_cell << ',' << tracer_frac << ',' << mass_frac << ',' << rel_err << '\n';
		}
		csv.close();

		auto const mass_phase = std::arg(mass_mode);
		auto const tracer_phase = std::arg(tracer_mode);
		auto const delta_phase = std::remainder(tracer_phase - mass_phase, 2.0 * M_PI);
		auto const delta_cells = delta_phase / (2.0 * M_PI) * static_cast<amrex::Real>(nx);
		auto const amp_ratio = std::abs(tracer_mode) / std::abs(mass_mode);
		amrex::Print() << "Fundamental mode: mass phase " << mass_phase << " rad, tracer phase " << tracer_phase << " rad, delta " << delta_phase
			       << " rad (" << delta_cells << " cells), amplitude ratio tracer/mass = " << amp_ratio << '\n';
	}

	const double err_tol = wave_error_tol; // defaults tuned for convergence tests; override via inputs
	int status = 0;
	if (epsilon > err_tol) {
		status = 1;
	}

	return status;
}
