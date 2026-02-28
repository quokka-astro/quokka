#ifndef PARTICLE_PHOTOIONIZATION_HPP_
#define PARTICLE_PHOTOIONIZATION_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "AMReX_Array.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
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

#if AMREX_SPACEDIM == 3
template <typename problem_t, quokka::OutOfBounds oob_policy>
void FillNGammaFromStromgrenVolumes(quokka::StochasticStellarPopParticleContainer<problem_t> *stellar_particles, int lev, amrex::Real time,
				    amrex::BoxArray const &ba_lev, amrex::DistributionMapping const &dm_lev, amrex::Geometry const &geom_lev,
				    amrex::MultiFab const &state_cc, amrex::MultiFab &n_gamma_cc, quokka::DataTableGpuConst<2, 1, oob_policy> const &qh0_table,
				    amrex::Real const mass_to_table_units = 1.0 / C::M_solar, amrex::Real const age_to_table_units = 1.0 / 3.15576e7,
				    bool const table_axes_are_mass_age = true, amrex::Real const alphaB = 2.6e-13,
				    amrex::Real const mean_particle_mass_mu = 1.27, amrex::Real const mH = 1.67e-24, int const max_pseudo_iters = 20,
				    int const log_every = 0, amrex::Real const init_rsrc = 3.0e17, amrex::Real const residual_tol = 1.0e-3)
{
	if (stellar_particles == nullptr) {
		return;
	}

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state_cc.boxArray() == n_gamma_cc.boxArray(), "state_cc and n_gamma_cc must have the same BoxArray.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state_cc.DistributionMap() == n_gamma_cc.DistributionMap(),
					 "state_cc and n_gamma_cc must have the same DistributionMap.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_gamma_cc.nComp() >= 1, "n_gamma_cc must have at least one component.");

	auto const dx = geom_lev.CellSizeArray();
	auto const plo = geom_lev.ProbLoArray();
	auto const phi_hi = geom_lev.ProbHiArray();
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
			amrex::Real const x_idx = ((p.pos(0) - plo[0]) * dxi[0]) - 0.5;
			amrex::Real const y_idx = ((p.pos(1) - plo[1]) * dxi[1]) - 0.5;
			amrex::Real const z_idx = ((p.pos(2) - plo[2]) * dxi[2]) - 0.5;

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
					while (kz < dom_lo.z) {
						kz += nz;
					}
					while (kz > dom_hi.z) {
						kz -= nz;
					}
				}
				amrex::Real const wz = (kk == 0) ? (1.0 - fz) : fz;

				for (int jj = 0; jj <= 1; ++jj) {
					int jy = j0 + jj;
					if (is_per[1] != 0) {
						while (jy < dom_lo.y) {
							jy += ny;
						}
						while (jy > dom_hi.y) {
							jy -= ny;
						}
					}
					amrex::Real const wy = (jj == 0) ? (1.0 - fy) : fy;

					for (int ii = 0; ii <= 1; ++ii) {
						int ix = i0 + ii;
						if (is_per[0] != 0) {
							while (ix < dom_lo.x) {
								ix += nx;
							}
							while (ix > dom_hi.x) {
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

	// Build source-centered Gaussian initial guess, matching Python initialization:
	// phi0 = alphaB * n_H^2 * exp(-(r/r_src)^2)
	amrex::Real wsum = 0.0;
	amrex::Real xsum = 0.0;
	amrex::Real ysum = 0.0;
	amrex::Real zsum = 0.0;
	for (amrex::MFIter mfi(source_q, false); mfi.isValid(); ++mfi) {
		auto const box = mfi.validbox();
		auto const src = source_q.const_array(mfi);
		for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
			for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
				for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
					amrex::Real const w = src(i, j, k, 0);
					if (w <= 0.0) {
						continue;
					}
					amrex::Real const x = plo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
					amrex::Real const y = plo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
					amrex::Real const z = plo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
					wsum += w;
					xsum += w * x;
					ysum += w * y;
					zsum += w * z;
				}
			}
		}
	}
	amrex::ParallelDescriptor::ReduceRealSum(wsum);
	amrex::ParallelDescriptor::ReduceRealSum(xsum);
	amrex::ParallelDescriptor::ReduceRealSum(ysum);
	amrex::ParallelDescriptor::ReduceRealSum(zsum);
	amrex::Real const src_x = (wsum > 0.0) ? (xsum / wsum) : (0.5 * (plo[0] + phi_hi[0]));
	amrex::Real const src_y = (wsum > 0.0) ? (ysum / wsum) : (0.5 * (plo[1] + phi_hi[1]));
	amrex::Real const src_z = (wsum > 0.0) ? (zsum / wsum) : (0.5 * (plo[2] + phi_hi[2]));
	amrex::Real const init_rsrc_safe = amrex::max(init_rsrc, 1.0e-60);

	amrex::MultiFab phi(ba_lev, dm_lev, 1, 1);
	amrex::MultiFab phi_stage(ba_lev, dm_lev, 1, 1);
	amrex::MultiFab phi_new(ba_lev, dm_lev, 1, 1);
	amrex::MultiFab explicit_rhs(ba_lev, dm_lev, 1, 0);
	amrex::MultiFab explicit_rhs_stage(ba_lev, dm_lev, 1, 0);
	amrex::MultiFab reaction_rate(ba_lev, dm_lev, 1, 0);
	amrex::MultiFab reaction_rate_stage(ba_lev, dm_lev, 1, 0);
	amrex::MultiFab residual(ba_lev, dm_lev, 1, 0);
	phi.setVal(0.0);
	phi_stage.setVal(0.0);
	phi_new.setVal(0.0);
	explicit_rhs.setVal(0.0);
	explicit_rhs_stage.setVal(0.0);
	reaction_rate.setVal(0.0);
	reaction_rate_stage.setVal(0.0);
	residual.setVal(0.0);

	auto const state = state_cc.const_arrays();
	auto const source_arr = source_q.const_arrays();

	// Pseudo-time FLD solve for photon number density (as a volumetric rate proxy):
	//   dn_gamma/dtau = div(D_FLD grad(n_gamma)) + q_vol - c*sigma_HI*n_HI*n_gamma,
	// with D_FLD = c*lambda(R)/kappa_F.
	// We use a constant opacity floor for transport corresponding to mean free path
	// equal to the simulation box size: kappa_F = 1 / Lbox.
	// Local ionization balance (used for n_HI):
	//   c*sigma_HI*n_gamma*(1-x) = alphaB*n_H*x^2.
	amrex::Real const dx_min = amrex::min(dx[0], amrex::min(dx[1], dx[2]));
	constexpr amrex::Real sigma_HI = 6.3e-18; // cm^2
	amrex::Real const Lbox = amrex::min(phi_hi[0] - plo[0], amrex::min(phi_hi[1] - plo[1], phi_hi[2] - plo[2]));
	amrex::Real const kappa_ref = 1.0 / amrex::max(Lbox, 1.0e-60);
	amrex::Real const cfl = 0.8;
	amrex::Real const dtau = cfl * dx_min / C::c_light;
	int const min_pseudo_iters = 4;
	amrex::Real const eps_phi = 1.0e-30;
	amrex::ParallelFor(phi, [=, phi_arr = phi.arrays()] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
		amrex::Real const x = plo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
		amrex::Real const y = plo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
		amrex::Real const z = plo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
		amrex::Real const r2 = (x - src_x) * (x - src_x) + (y - src_y) * (y - src_y) + (z - src_z) * (z - src_z);
		amrex::Real const rho = state[nbx](i, j, k, HydroSystem<problem_t>::density_index);
		amrex::Real const n = (rho > 0.0) ? (rho / n_to_rho) : 0.0;
		amrex::Real const cap = alphaB * n * n;
		phi_arr[nbx](i, j, k, 0) = cap * std::exp(-r2 / (init_rsrc_safe * init_rsrc_safe));
	});
	auto compute_explicit_and_reaction = [&](amrex::MultiFab &phi_in, amrex::MultiFab &explicit_out, amrex::MultiFab &k_out) {
		phi_in.FillBoundary(geom_lev.periodicity());
		auto const phi_arr = phi_in.const_arrays();
		auto explicit_arr = explicit_out.arrays();
		auto k_arr = k_out.arrays();

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

			explicit_arr[nbx](i, j, k, 0) = -divF + qsrc;
			k_arr[nbx](i, j, k, 0) = k_reac;
		});
	};

	amrex::Real const source_scale = amrex::max(source_q.norm0(0, 0, false) / cell_volume, 1.0e-60);

	int iter = 0;
	for (; iter < max_pseudo_iters; ++iter) {
		compute_explicit_and_reaction(phi, explicit_rhs, reaction_rate);
		amrex::ParallelFor(residual, [phi_arr = phi.const_arrays(), E0_arr = explicit_rhs.const_arrays(), k0_arr = reaction_rate.const_arrays(),
					      residual_arr = residual.arrays()] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
			residual_arr[nbx](i, j, k, 0) = E0_arr[nbx](i, j, k, 0) - k0_arr[nbx](i, j, k, 0) * phi_arr[nbx](i, j, k, 0);
		});
		amrex::Real const rel_resid = residual.norm0(0, 0, false) / source_scale;
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(rel_resid), "Stromgren pseudo-time solve produced non-finite residual.");
		if ((log_every > 0) && ((iter == 0) || (((iter + 1) % log_every) == 0))) {
			std::ostringstream oss;
			oss << "[iter " << std::setw(7) << (iter + 1) << "] residual=" << std::scientific << std::setprecision(6) << rel_resid << "\n";
			amrex::Print() << oss.str();
		}
		if ((iter >= min_pseudo_iters) && (rel_resid < residual_tol)) {
			break;
		}

		constexpr int max_stage_retries = 12;
		amrex::Real dt_try = amrex::max(dtau, 1.0e-60);
		bool accepted = false;
		for (int retry = 0; retry < max_stage_retries; ++retry) {
			amrex::ParallelFor(phi_stage,
					   [phi_arr = phi.const_arrays(), E0_arr = explicit_rhs.const_arrays(), k0_arr = reaction_rate.const_arrays(),
					    phi1_arr = phi_stage.arrays(), dt_try] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
						   amrex::Real const numer = phi_arr[nbx](i, j, k, 0) + dt_try * E0_arr[nbx](i, j, k, 0);
						   amrex::Real const denom = amrex::max(1.0 + dt_try * k0_arr[nbx](i, j, k, 0), 1.0e-60);
						   phi1_arr[nbx](i, j, k, 0) = numer / denom;
					   });
			amrex::Real const phi1_min = phi_stage.min(0, 0, false);
			if (!(phi1_min >= 0.0)) {
				dt_try *= 0.5;
				continue;
			}

			compute_explicit_and_reaction(phi_stage, explicit_rhs_stage, reaction_rate_stage);

			amrex::ParallelFor(phi_new, [phi_arr = phi.const_arrays(), E0_arr = explicit_rhs.const_arrays(),
						     E1_arr = explicit_rhs_stage.const_arrays(), k0_arr = reaction_rate.const_arrays(),
						     k1_arr = reaction_rate_stage.const_arrays(), phi2_arr = phi_new.arrays(),
						     dt_try] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
				amrex::Real const phi0 = phi_arr[nbx](i, j, k, 0);
				amrex::Real const numer =
				    phi0 + 0.5 * dt_try * (E0_arr[nbx](i, j, k, 0) + E1_arr[nbx](i, j, k, 0) - k0_arr[nbx](i, j, k, 0) * phi0);
				amrex::Real const denom = amrex::max(1.0 + 0.5 * dt_try * k1_arr[nbx](i, j, k, 0), 1.0e-60);
				phi2_arr[nbx](i, j, k, 0) = numer / denom;
			});
			amrex::Real const phi2_min = phi_new.min(0, 0, false);
			if (!(phi2_min >= 0.0)) {
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

	auto const phi_arr = phi.const_arrays();
	auto n_gamma = n_gamma_cc.arrays();
	amrex::ParallelFor(
	    n_gamma_cc, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept { n_gamma[nbx](i, j, k, 0) = amrex::max(phi_arr[nbx](i, j, k, 0), 0.0); });
}
#endif

} // namespace quokka::photoionization

#endif // PARTICLE_PHOTOIONIZATION_HPP_
