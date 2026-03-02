#ifndef PARTICLE_PHOTOIONIZATION_HPP_
#define PARTICLE_PHOTOIONIZATION_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <limits>

#include "AMReX_Array.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Math.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"

#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "util/DataTable.hpp"

namespace quokka::photoionization
{

amrex::Real constexpr mass_to_table_units = 1.0 / C::M_solar;
amrex::Real constexpr age_to_table_units = 1.0 / 3.15576e7;
amrex::Real constexpr mH = 1.67e-24;
amrex::Real constexpr mean_particle_mass_mu = 1.27;
//amrex::Real constexpr init_rsrc = 3.0e17;
amrex::Real constexpr alphaB = 2.6e-13;
amrex::Real constexpr sigma_HI = 6.3e-18; // cm^2
bool constexpr table_axes_are_mass_age = true;
constexpr int max_stage_retries = 6;

#if AMREX_SPACEDIM == 3
template <typename problem_t, quokka::OutOfBounds oob_policy>
void FillNGammaFromStromgrenVolumes(quokka::StochasticStellarPopParticleContainer<problem_t> *stellar_particles, int lev, amrex::Real time,
				    amrex::BoxArray const &ba_lev, amrex::DistributionMapping const &dm_lev, amrex::Geometry const &geom_lev,
				    amrex::MultiFab const &state_cc, amrex::MultiFab &n_gamma_cc, quokka::DataTableGpuConst<2, 1, oob_policy> const &qh0_table,
				    int const max_pseudo_iters = 20, amrex::Real const residual_tol = 1.0e-3, int const log_every = 0,
				    bool const abort_on_max_iters = false)
{
	if (stellar_particles == nullptr) {
		return;
	}

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state_cc.boxArray() == n_gamma_cc.boxArray(), "state_cc and n_gamma_cc must have the same BoxArray.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state_cc.DistributionMap() == n_gamma_cc.DistributionMap(),
					 "state_cc and n_gamma_cc must have the same DistributionMap.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_gamma_cc.nComp() == 1, "n_gamma_cc must have exactly one component.");

	auto const dx = geom_lev.CellSizeArray();
	auto const p_lo = geom_lev.ProbLoArray();
	auto const p_hi = geom_lev.ProbHiArray();
	auto const dxi = geom_lev.InvCellSizeArray();
	amrex::Real const cell_volume = dx[0] * dx[1] * dx[2];
	amrex::Real const n_to_rho = mean_particle_mass_mu * mH;

	amrex::MultiFab source_q(ba_lev, dm_lev, 1, 1);
	source_q.setVal(0.0);

	auto const domain = geom_lev.Domain();
	auto const dom_lo = amrex::lbound(domain);
	auto const dom_hi = amrex::ubound(domain);
	auto const is_per = geom_lev.periodicity().intVect();

	// Deposit ionizing photon luminosity (photons/s) from individually sampled stars only.
	// We exclude LowMassComposite, SNRemnant, and Removed particles.
	for (quokka::StochasticStellarPopParticleIterator<problem_t> pti(*stellar_particles, lev); pti.isValid(); ++pti) {
		auto &particles = pti.GetArrayOfStructs();
		auto *pData = particles().data();
		auto const np = pti.numParticles();
		auto const src = source_q.array(pti);
		auto const box = amrex::grow(pti.validbox(), 1);
		auto const lo = amrex::lbound(box);
		auto const hi = amrex::ubound(box);

		amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) noexcept {
			auto const &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			int const stage = p.idata(quokka::StochasticStellarPopParticleStageIdx);
			bool const is_individual_ionizing_star = (stage == static_cast<int>(quokka::StellarEvolutionStage::HighMassNonExploding)) ||
								 (stage == static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor));
			if (!is_individual_ionizing_star) {
				return;
			}

			amrex::Real const age = time - p.rdata(quokka::StochasticStellarPopParticleBirthTimeIdx);
			if (age <= 0.0) {
				return;
			}

			amrex::Real const zams_mass = p.rdata(quokka::StochasticStellarPopParticleMassAtBirthIdx);
			if (zams_mass <= 0.0) {
				return;
			}

			amrex::Real const mass_coord = zams_mass * mass_to_table_units;
			amrex::Real const age_coord = age * age_to_table_units;
			std::array<amrex::Real, 2> point{};
			if (table_axes_are_mass_age) {
				point = {mass_coord, age_coord};
			} else {
				point = {age_coord, mass_coord};
			}

			amrex::Real const S = qh0_table.interpolate_single(point, 0);
			if (!(S > 0.0) || !std::isfinite(S)) {
				return;
			}

			// CIC deposit in cell-centered index space to avoid half-cell bias for sources on faces.
			amrex::Real const x_idx = ((p.pos(0) - p_lo[0]) * dxi[0]) - 0.5;
			amrex::Real const y_idx = ((p.pos(1) - p_lo[1]) * dxi[1]) - 0.5;
			amrex::Real const z_idx = ((p.pos(2) - p_lo[2]) * dxi[2]) - 0.5;

			int const i0 = static_cast<int>(amrex::Math::floor(x_idx));
			int const j0 = static_cast<int>(amrex::Math::floor(y_idx));
			int const k0 = static_cast<int>(amrex::Math::floor(z_idx));

			amrex::Real const fx = x_idx - static_cast<amrex::Real>(i0);
			amrex::Real const fy = y_idx - static_cast<amrex::Real>(j0);
			amrex::Real const fz = z_idx - static_cast<amrex::Real>(k0);

			int const nx = dom_hi.x - dom_lo.x + 1;
			int const ny = dom_hi.y - dom_lo.y + 1;
			int const nz = dom_hi.z - dom_lo.z + 1;

			for (int kk = 0; kk <= 1; ++kk) {
				int kz = k0 + kk;
				if (is_per[2] != 0) {
					while (kz < lo.z) {
						kz += nz;
					}
					while (kz > hi.z) {
						kz -= nz;
					}
				}
				amrex::Real const wz = (kk == 0) ? (1.0 - fz) : fz;

				for (int jj = 0; jj <= 1; ++jj) {
					int jy = j0 + jj;
					if (is_per[1] != 0) {
						while (jy < lo.y) {
							jy += ny;
						}
						while (jy > hi.y) {
							jy -= ny;
						}
					}
					amrex::Real const wy = (jj == 0) ? (1.0 - fy) : fy;

					for (int ii = 0; ii <= 1; ++ii) {
						int ix = i0 + ii;
						if (is_per[0] != 0) {
							while (ix < lo.x) {
								ix += nx;
							}
							while (ix > hi.x) {
								ix -= nx;
							}
						}

						if ((ix < lo.x) || (ix > hi.x) || (jy < lo.y) || (jy > hi.y) || (kz < lo.z) || (kz > hi.z)) {
							continue;
						}

						amrex::Real const wx = (ii == 0) ? (1.0 - fx) : fx;
						amrex::Real const w = wx * wy * wz;
						amrex::Gpu::Atomic::AddNoRet(&src(ix, jy, kz, 0), w * S);
					}
				}
			}
		});
	}
	source_q.SumBoundary(geom_lev.periodicity());

	amrex::MultiFab phi(ba_lev, dm_lev, 1, 1);
	amrex::MultiFab phi_new(ba_lev, dm_lev, 1, 1);
	amrex::MultiFab explicit_rhs(ba_lev, dm_lev, 1, 0);
	amrex::MultiFab reaction_rate(ba_lev, dm_lev, 1, 0);
	amrex::MultiFab transport_rhs(ba_lev, dm_lev, 1, 0);
	amrex::MultiFab source_rhs(ba_lev, dm_lev, 1, 0);
	amrex::MultiFab reaction_sink(ba_lev, dm_lev, 1, 0);
	amrex::MultiFab residual(ba_lev, dm_lev, 1, 0);
	phi.setVal(0.0);
	phi_new.setVal(0.0);
	explicit_rhs.setVal(0.0);
	reaction_rate.setVal(0.0);
	transport_rhs.setVal(0.0);
	source_rhs.setVal(0.0);
	reaction_sink.setVal(0.0);
	residual.setVal(0.0);

	auto const state = state_cc.const_arrays();
	auto const source_arr = source_q.const_arrays();

	//
	// Pseudo-time FLD solve for photon number density:
	//   dn_gamma/dtau = div(D_FLD grad(n_gamma)) + q_vol - c*sigma_HI*n_HI*n_gamma,
	// with D_FLD = c*lambda(R)/kappa_F.
	//
	// We use a constant opacity floor for transport corresponding to mean free path
	// equal to the simulation box size: kappa_F = 1 / Lbox.
	// The flux limiter lambda(R) is the Levermore & Pomraning (1981) limiter.
	//
	// Local ionization balance (used for n_HI):
	//   c*sigma_HI*n_gamma*(1-x) = alphaB*n_H*x^2.
	//
	// We use a lowest-order IMEX time discretization:
	//   forward Euler for the explicit flux divergence and source terms,
	//   and backward Euler for the reaction term.
	// We iterate in pseudo-time until the residual of the steady-state PDE
	//  is reduced by a user-specified factor, or until max iter is reached.
	// We also enforce positivity of the solution with timestep retries.
	//

	amrex::Real const Lbox = std::min({p_hi[0] - p_lo[0], p_hi[1] - p_lo[1], p_hi[2] - p_lo[2]});
	amrex::Real const kappa_ref = 1.0 / Lbox;
	amrex::Real const lambda_lp_max = 1.0 / 3.0;
	amrex::Real const D_max = C::c_light * lambda_lp_max / kappa_ref;
	amrex::Real const sum_inv_dx2 = (1.0 / (dx[0] * dx[0])) + (1.0 / (dx[1] * dx[1])) + (1.0 / (dx[2] * dx[2]));
	amrex::Real const dtau_explicit_max = 1.0 / (2.0 * D_max * sum_inv_dx2);
	amrex::Real const diffusion_cfl = 1.0;
	amrex::Real const dtau = diffusion_cfl * dtau_explicit_max;
	int const min_pseudo_iters = 4;
	amrex::Real const eps_phi = 1.0e-30;

	auto compute_explicit_and_reaction = [&](amrex::MultiFab &phi_in, amrex::MultiFab &explicit_out, amrex::MultiFab &k_out, amrex::MultiFab &transport_out,
						 amrex::MultiFab &source_out, amrex::MultiFab &reaction_out) {
		phi_in.FillBoundary(geom_lev.periodicity());
		auto const phi_arr = phi_in.const_arrays();
		auto explicit_arr = explicit_out.arrays();
		auto k_arr = k_out.arrays();
		auto transport_arr = transport_out.arrays();
		auto source_term_arr = source_out.arrays();
		auto reaction_term_arr = reaction_out.arrays();

		amrex::ParallelFor(explicit_out, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
			amrex::Real const phi_c = amrex::max(phi_arr[nbx](i, j, k, 0), 0.0);
			amrex::Real const phi_ip = amrex::max(phi_arr[nbx](i + 1, j, k, 0), 0.0);
			amrex::Real const phi_im = amrex::max(phi_arr[nbx](i - 1, j, k, 0), 0.0);
			amrex::Real const phi_jp = amrex::max(phi_arr[nbx](i, j + 1, k, 0), 0.0);
			amrex::Real const phi_jm = amrex::max(phi_arr[nbx](i, j - 1, k, 0), 0.0);
			amrex::Real const phi_kp = amrex::max(phi_arr[nbx](i, j, k + 1, 0), 0.0);
			amrex::Real const phi_km = amrex::max(phi_arr[nbx](i, j, k - 1, 0), 0.0);

			amrex::Real const gxp = (phi_ip - phi_c) / dx[0];
			amrex::Real const gxm = (phi_c - phi_im) / dx[0];
			amrex::Real const gyp = (phi_jp - phi_c) / dx[1];
			amrex::Real const gym = (phi_c - phi_jm) / dx[1];
			amrex::Real const gzp = (phi_kp - phi_c) / dx[2];
			amrex::Real const gzm = (phi_c - phi_km) / dx[2];

			amrex::Real const rho_c = state[nbx](i, j, k, HydroSystem<problem_t>::density_index);
			amrex::Real const n_c = (rho_c > 0.0) ? (rho_c / n_to_rho) : 0.0;

			auto ion_frac_eq = [=] AMREX_GPU_DEVICE(amrex::Real nH_local, amrex::Real ng_local) noexcept {
				if ((nH_local <= 0.0) || (ng_local <= 0.0)) {
					return amrex::Real(0.0);
				}
				amrex::Real const a = alphaB * nH_local;
				amrex::Real const b = C::c_light * sigma_HI * ng_local;
				amrex::Real const disc = b * b + 4.0 * a * b;
				amrex::Real const x = (-b + std::sqrt(disc)) / (2.0 * a);
				return amrex::min<amrex::Real>(1.0, amrex::max<amrex::Real>(0.0, x));
			};

			amrex::Real const x_c = ion_frac_eq(n_c, phi_c);

			amrex::Real const nHI_c = n_c * (1.0 - x_c);

			amrex::Real const phi_xp = 0.5 * (phi_c + phi_ip);
			amrex::Real const phi_xm = 0.5 * (phi_c + phi_im);
			amrex::Real const phi_yp = 0.5 * (phi_c + phi_jp);
			amrex::Real const phi_ym = 0.5 * (phi_c + phi_jm);
			amrex::Real const phi_zp = 0.5 * (phi_c + phi_kp);
			amrex::Real const phi_zm = 0.5 * (phi_c + phi_km);

			amrex::Real const kappaxp = kappa_ref;
			amrex::Real const kappaxm = kappa_ref;
			amrex::Real const kappayp = kappa_ref;
			amrex::Real const kappaym = kappa_ref;
			amrex::Real const kappazp = kappa_ref;
			amrex::Real const kappazm = kappa_ref;

			amrex::Real const Rxp = std::abs(gxp) / amrex::max(kappaxp * phi_xp, eps_phi);
			amrex::Real const Rxm = std::abs(gxm) / amrex::max(kappaxm * phi_xm, eps_phi);
			amrex::Real const Ryp = std::abs(gyp) / amrex::max(kappayp * phi_yp, eps_phi);
			amrex::Real const Rym = std::abs(gym) / amrex::max(kappaym * phi_ym, eps_phi);
			amrex::Real const Rzp = std::abs(gzp) / amrex::max(kappazp * phi_zp, eps_phi);
			amrex::Real const Rzm = std::abs(gzm) / amrex::max(kappazm * phi_zm, eps_phi);

			amrex::Real const lambdaxp = (2.0 + Rxp) / (6.0 + 3.0 * Rxp + Rxp * Rxp);
			amrex::Real const lambdaxm = (2.0 + Rxm) / (6.0 + 3.0 * Rxm + Rxm * Rxm);
			amrex::Real const lambdayp = (2.0 + Ryp) / (6.0 + 3.0 * Ryp + Ryp * Ryp);
			amrex::Real const lambdaym = (2.0 + Rym) / (6.0 + 3.0 * Rym + Rym * Rym);
			amrex::Real const lambdazp = (2.0 + Rzp) / (6.0 + 3.0 * Rzp + Rzp * Rzp);
			amrex::Real const lambdazm = (2.0 + Rzm) / (6.0 + 3.0 * Rzm + Rzm * Rzm);

			amrex::Real const Fxp = -(C::c_light * lambdaxp / kappaxp) * gxp;
			amrex::Real const Fxm = -(C::c_light * lambdaxm / kappaxm) * gxm;
			amrex::Real const Fyp = -(C::c_light * lambdayp / kappayp) * gyp;
			amrex::Real const Fym = -(C::c_light * lambdaym / kappaym) * gym;
			amrex::Real const Fzp = -(C::c_light * lambdazp / kappazp) * gzp;
			amrex::Real const Fzm = -(C::c_light * lambdazm / kappazm) * gzm;

			amrex::Real const divF = (Fxp - Fxm) / dx[0] + (Fyp - Fym) / dx[1] + (Fzp - Fzm) / dx[2];

			amrex::Real const k_reac = C::c_light * sigma_HI * nHI_c;
			amrex::Real const qsrc = source_arr[nbx](i, j, k, 0) / cell_volume;

			amrex::Real const transport_term = -divF;
			amrex::Real const source_term = qsrc;
			amrex::Real const reaction_term = k_reac * phi_c;

			explicit_arr[nbx](i, j, k, 0) = transport_term + source_term;
			k_arr[nbx](i, j, k, 0) = k_reac;
			transport_arr[nbx](i, j, k, 0) = transport_term;
			source_term_arr[nbx](i, j, k, 0) = source_term;
			reaction_term_arr[nbx](i, j, k, 0) = reaction_term;
		});
	};

	amrex::Real const source_scale = amrex::max(source_q.norm0(0, 0, false) / cell_volume, 1.0e-60);

	int iter = 0;
	bool converged = false;
	amrex::Real final_rel_resid = std::numeric_limits<amrex::Real>::infinity();
	amrex::Real prev_rel_resid = std::numeric_limits<amrex::Real>::quiet_NaN();
	for (; iter < max_pseudo_iters; ++iter) {
		compute_explicit_and_reaction(phi, explicit_rhs, reaction_rate, transport_rhs, source_rhs, reaction_sink);

		amrex::ParallelFor(residual, [phi_arr = phi.const_arrays(), E0_arr = explicit_rhs.const_arrays(), k0_arr = reaction_rate.const_arrays(),
					      residual_arr = residual.arrays()] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
			residual_arr[nbx](i, j, k, 0) = E0_arr[nbx](i, j, k, 0) - k0_arr[nbx](i, j, k, 0) * phi_arr[nbx](i, j, k, 0);
		});

		amrex::Real const rel_resid = residual.norm0(0, 0, false) / source_scale;
		final_rel_resid = rel_resid;
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(rel_resid), "Stromgren pseudo-time solve produced non-finite residual.");

		if ((log_every > 0) && ((iter == 0) || (((iter + 1) % log_every) == 0))) {
			amrex::Real const phi_min = phi.min(0, 0, false);
			amrex::Real const phi_max = phi.max(0, 0, false);
			amrex::Real const explicit_inf = explicit_rhs.norm0(0, 0, false);
			amrex::Real const transport_inf = transport_rhs.norm0(0, 0, false);
			amrex::Real const source_inf = source_rhs.norm0(0, 0, false);
			amrex::Real const reaction_inf = reaction_sink.norm0(0, 0, false);
			amrex::Real const ratio = std::isfinite(prev_rel_resid) ? (rel_resid / amrex::max(prev_rel_resid, 1.0e-300)) : 1.0;
			amrex::Print() << std::format(
			    "[iter {:7d}] residual={:.6e} ratio={:.3e} phi_min={:.6e} phi_max={:.6e} |E|={:.6e} |T|={:.6e} |Q|={:.6e} |R|={:.6e}\n",
			    iter + 1, rel_resid, ratio, phi_min, phi_max, explicit_inf, transport_inf, source_inf, reaction_inf);
		}
		prev_rel_resid = rel_resid;
		if ((iter >= min_pseudo_iters) && (rel_resid < residual_tol)) {
			converged = true;
			break;
		}

		amrex::Real dt_try = amrex::max(dtau, 1.0e-60);
		bool accepted = false;

		for (int retry = 0; retry < max_stage_retries; ++retry) {
			// Lowest-order IMEX update: forward Euler for explicit terms + backward Euler for reaction.
			amrex::ParallelFor(phi_new, [phi_arr = phi.const_arrays(), E0_arr = explicit_rhs.const_arrays(), k0_arr = reaction_rate.const_arrays(),
						     phi1_arr = phi_new.arrays(), dt_try] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
				amrex::Real const numer = phi_arr[nbx](i, j, k, 0) + dt_try * E0_arr[nbx](i, j, k, 0);
				amrex::Real const denom = amrex::max(1.0 + dt_try * k0_arr[nbx](i, j, k, 0), 1.0e-60);
				phi1_arr[nbx](i, j, k, 0) = numer / denom;
			});

			amrex::Real const phi1_min = phi_new.min(0, 0, false);
			if (!(phi1_min >= 0.0)) {
				dt_try *= 0.5;
				continue;
			}

			accepted = true;
			break;
		}

		if (!accepted) {
			if ((log_every > 0) && ((iter == 0) || (((iter + 1) % log_every) == 0))) {
				amrex::Print() << "[iter " << std::setw(7) << (iter + 1) << "] step rejected after positivity retries\n";
			}
			continue;
		}

		amrex::MultiFab::Copy(phi, phi_new, 0, 0, 1, 1);
	}
	if (abort_on_max_iters && !converged) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
		    false,
		    std::format("Stromgren pseudo-time solve reached max iterations ({}) without satisfying residual tolerance (residual_tol={}, final_rel_resid={}).",
				max_pseudo_iters, residual_tol, final_rel_resid)
			.c_str());
	}

	auto const phi_arr = phi.const_arrays();
	auto n_gamma = n_gamma_cc.arrays();
	amrex::ParallelFor(
	    n_gamma_cc, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept { n_gamma[nbx](i, j, k, 0) = amrex::max(phi_arr[nbx](i, j, k, 0), 0.0); });
}
#endif

} // namespace quokka::photoionization

#endif // PARTICLE_PHOTOIONIZATION_HPP_
