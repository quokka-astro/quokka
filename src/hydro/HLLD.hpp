#ifndef HLLD_HPP_ // NOLINT
#define HLLD_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include <AMReX.H>
#include <AMReX_REAL.H>
#include <algorithm>

#include "hydro/HydroState.hpp"
#include "util/ArrayView.hpp"
#include "util/valarray.hpp"

namespace quokka::Riemann
{
constexpr double DELTA = 1.0e-4;

// HLLD solver following Miyoshi and Kusano (2005), hereafter MK5.
template <typename problem_t, int N_scalars, int N_mscalars, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto HLLD(quokka::HydroState<N_scalars, N_mscalars> const &sL, quokka::HydroState<N_scalars, N_mscalars> const &sR,
					      const double gamma, const double bx, const double perp_v_jump)
    -> std::tuple<quokka::valarray<double, fluxdim>, double, double>
{
	//--- Step 1. Compute L/R states

	// initialize left and right conserved states
	ConsHydro1D<N_scalars> u_L{};
	ConsHydro1D<N_scalars> u_R{};
	// initialize temporary container to store flux across interface
	quokka::valarray<double, fluxdim> F_x = {};
	// initialize fluxes at left and right side of the interface
	ConsHydro1D<N_scalars> f_L{};
	ConsHydro1D<N_scalars> f_R{};
	// initialise signal speeds (left to right)
	std::array<double, 5> spds{};
	// initialise four intermediate conserved states
	ConsHydro1D<N_scalars> u_star_L{};
	ConsHydro1D<N_scalars> u_dstar_L{};
	ConsHydro1D<N_scalars> u_dstar_R{};
	ConsHydro1D<N_scalars> u_star_R{};

	// frequently used term
	const double bx_sq = SQUARE(bx);

	// compute L/R states for select conserved variables
	// (group transverse vector components for floating-point associativity symmetry)
	// magnetic pressure
	const double pb_L = 0.5 * (bx_sq + (SQUARE(sL.by) + SQUARE(sL.bz)));
	const double pb_R = 0.5 * (bx_sq + (SQUARE(sR.by) + SQUARE(sR.bz)));
	// kinetic energy
	const double ke_L = 0.5 * sL.rho * (SQUARE(sL.u) + (SQUARE(sL.v) + SQUARE(sL.w)));
	const double ke_R = 0.5 * sR.rho * (SQUARE(sR.u) + (SQUARE(sR.v) + SQUARE(sR.w)));
	// set left conserved states
	u_L.rho = sL.rho;
	u_L.mx = sL.u * u_L.rho;
	u_L.my = sL.v * u_L.rho;
	u_L.mz = sL.w * u_L.rho;
	u_L.E = ke_L + pb_L + sL.P / (gamma - 1.0); // TODO(neco): generalise EOS
	u_L.Eint = sL.Eint;
	u_L.by = sL.by;
	u_L.bz = sL.bz;
	// set right conserved states
	u_R.rho = sR.rho;
	u_R.mx = sR.u * u_R.rho;
	u_R.my = sR.v * u_R.rho;
	u_R.mz = sR.w * u_R.rho;
	u_R.E = ke_R + pb_R + sR.P / (gamma - 1.0);
	u_R.Eint = sR.Eint;
	u_R.by = sR.by;
	u_R.bz = sR.bz;

	for (int n = 0; n < N_scalars; ++n) {
		u_L.scalar[n] = sL.scalar[n];
		u_R.scalar[n] = sR.scalar[n];
	}

	//--- Step 2. Compute L & R wave speeds according to MK5, eqn. (67)

	const double fspd_L = FastMagnetoSonicSpeed(gamma, sL, bx);
	const double fspd_R = FastMagnetoSonicSpeed(gamma, sR, bx);
	spds[0] = std::min(sL.u - fspd_L, sR.u - fspd_R);
	spds[4] = std::max(sL.u + fspd_L, sR.u + fspd_R);
	const double fspd_m = -std::min(0.0, spds[0]);
	const double fspd_p = std::max(0.0, spds[4]);

	//--- Step 3. Compute L/R fluxes

	// total pressure
	const double ptot_L = sL.P + pb_L;
	const double ptot_R = sR.P + pb_R;
	// fluxes on the left side of the interface
	f_L.rho = u_L.mx;
	f_L.mx = u_L.mx * sL.u + ptot_L - bx_sq;
	f_L.my = u_L.my * sL.u - bx * u_L.by;
	f_L.mz = u_L.mz * sL.u - bx * u_L.bz;
	f_L.E = sL.u * (u_L.E + ptot_L - bx_sq) - bx * (sL.v * u_L.by + sL.w * u_L.bz);
	f_L.Eint = u_L.Eint * sL.u;
	f_L.by = u_L.by * sL.u - bx * sL.v;
	f_L.bz = u_L.bz * sL.u - bx * sL.w;
	// fluxes on the right side of the interface
	f_R.rho = u_R.mx;
	f_R.mx = u_R.mx * sR.u + ptot_R - bx_sq;
	f_R.my = u_R.my * sR.u - bx * u_R.by;
	f_R.mz = u_R.mz * sR.u - bx * u_R.bz;
	f_R.E = sR.u * (u_R.E + ptot_R - bx_sq) - bx * (sR.v * u_R.by + sR.w * u_R.bz);
	f_R.Eint = u_R.Eint * sR.u;
	f_R.by = u_R.by * sR.u - bx * sR.v;
	f_R.bz = u_R.bz * sR.u - bx * sR.w;

	// passive scalar fluxes right and left
	for (int n = 0; n < N_scalars; ++n) {
		f_L.scalar[n] = u_L.scalar[n] * sL.u;
		f_R.scalar[n] = u_R.scalar[n] * sR.u;
	}

	//--- Step 4. Compute middle and Alfven wave speeds

	// MK5: S_i - u_i (for i=L or R)
	const double siui_L = spds[0] - sL.u;
	const double siui_R = spds[4] - sR.u;
	// carbuncle detector
	const double max_spd = std::max(fspd_L, fspd_R);
	const double para_v_jump = sL.u - sR.u; // negative -> compression
	double theta = 1.0;
	// tp := shock anisotropy, clamped to [0, 1], with theta = tp^4
	const double denom_tp = std::max(1e-14, max_spd - std::min(perp_v_jump, 0.0));
	double tp = (max_spd - std::min(para_v_jump, 0.0)) / denom_tp;
	tp = std::clamp(tp, 0.0, 1.0);
	theta = SQUARE(SQUARE(tp));
	// modified middle speed S_M from MK5 eqn (38)
	const double sm_denom = (siui_R * u_R.rho - siui_L * u_L.rho);
	spds[2] = (siui_R * u_R.mx - siui_L * u_L.mx + theta * (ptot_L - ptot_R)) / sm_denom;
	// S_i - S_M (for i=L or R)
	const double sism_L = spds[0] - spds[2];
	
	const double sism_R = spds[4] - spds[2];
	const double sism_inv_L = 1.0 / sism_L;
	const double sism_inv_R = 1.0 / sism_R;
	// MK5: rho_i from eqn (43)
	u_star_L.rho = u_L.rho * siui_L * sism_inv_L;
	u_star_R.rho = u_R.rho * siui_R * sism_inv_R;
	u_star_L.Eint = u_L.Eint * siui_L * sism_inv_L;
	u_star_R.Eint = u_R.Eint * siui_R * sism_inv_R;
	for (int n = 0; n < N_scalars; ++n) {
		u_star_L.scalar[n] = u_L.scalar[n] * siui_L * sism_inv_L;
		u_star_R.scalar[n] = u_R.scalar[n] * siui_R * sism_inv_R;
	}

	const double u_star_rho_inv_L = 1.0 / u_star_L.rho;
	const double u_star_rho_inv_R = 1.0 / u_star_R.rho;
	const double rho_sqrt_L = std::sqrt(u_star_L.rho);
	const double rho_sqrt_R = std::sqrt(u_star_R.rho);
	// MK5: eqn (51)
	spds[1] = spds[2] - std::abs(bx) / rho_sqrt_L;
	spds[3] = spds[2] + std::abs(bx) / rho_sqrt_R;

	//--- Step 5. Compute intermediate states

	// compute total pressure
	// MK5: eqn (41) can be calculated (more explicitly) via eqn (23)
	const double ptot_star_L = ptot_L + u_L.rho * siui_L * (spds[2] - sL.u);
	const double ptot_star_R = ptot_R + u_R.rho * siui_R * (spds[2] - sR.u);
	const double ptot_star = 0.5 * (ptot_star_L + ptot_star_R);

	// MK5: u_L^(star, dstar) from, eqn (39)
	u_star_L.mx = u_star_L.rho * spds[2];
	if (std::abs(u_L.rho * siui_L * sism_L - bx_sq) < DELTA * ptot_star) {
		// degenerate case
		u_star_L.my = u_star_L.rho * sL.v;
		u_star_L.mz = u_star_L.rho * sL.w;
		u_star_L.by = u_L.by;
		u_star_L.bz = u_L.bz;
	} else {
		// MK5: eqns (44) and (46)
		double tmp = bx * (siui_L - sism_L) / (u_L.rho * siui_L * sism_L - bx_sq);
		u_star_L.my = u_star_L.rho * (sL.v - u_L.by * tmp);
		u_star_L.mz = u_star_L.rho * (sL.w - u_L.bz * tmp);
		// MK5: eqns (45) and (47)
		tmp = (u_L.rho * SQUARE(siui_L) - bx_sq) / (u_L.rho * siui_L * sism_L - bx_sq);
		u_star_L.by = u_L.by * tmp;
		u_star_L.bz = u_L.bz * tmp;
	}
	// vec(v_L^star) dot vec(b_L^star)
	// group transverse momenta-components for floating-point associativity
	double vb_star_L = (u_star_L.mx * bx + (u_star_L.my * u_star_L.by + u_star_L.mz * u_star_L.bz)) * u_star_rho_inv_L;
	// MK5: eqn (48)
	u_star_L.E = (siui_L * u_L.E - ptot_L * sL.u + ptot_star * spds[2] + bx * (sL.u * bx + (sL.v * u_L.by + sL.w * u_L.bz) - vb_star_L)) * sism_inv_L;

	// MK5: u_R^(star, dstar) from, eqn (39)
	u_star_R.mx = u_star_R.rho * spds[2];
	if (std::abs(u_R.rho * siui_R * sism_R - bx_sq) < DELTA * ptot_star) {
		// degenerate case
		u_star_R.my = u_star_R.rho * sR.v;
		u_star_R.mz = u_star_R.rho * sR.w;
		u_star_R.by = u_R.by;
		u_star_R.bz = u_R.bz;
	} else {
		// MK5: eqns (44) and (46)
		double tmp = bx * (siui_R - sism_R) / (u_R.rho * siui_R * sism_R - bx_sq);
		u_star_R.my = u_star_R.rho * (sR.v - u_R.by * tmp);
		u_star_R.mz = u_star_R.rho * (sR.w - u_R.bz * tmp);
		// MK5: eqns (45) and (47)
		tmp = (u_R.rho * SQUARE(siui_R) - bx_sq) / (u_R.rho * siui_R * sism_R - bx_sq);
		u_star_R.by = u_R.by * tmp;
		u_star_R.bz = u_R.bz * tmp;
	}
	// vec(v_R^star) dot vec(b_R^star)
	// group transverse momenta-components for floating-point associativity
	const double vb_star_R = (u_star_R.mx * bx + (u_star_R.my * u_star_R.by + u_star_R.mz * u_star_R.bz)) * u_star_rho_inv_R;
	// MK5: eqn (48)
	u_star_R.E = (siui_R * u_R.E - ptot_R * sR.u + ptot_star * spds[2] + bx * (sR.u * bx + (sR.v * u_R.by + sR.w * u_R.bz) - vb_star_R)) * sism_inv_R;

	// if Bx is near zero, then u_i^dstar = u_i^star
	if (0.5 * bx_sq < DELTA * ptot_star) {
		u_dstar_L = u_star_L;
		u_dstar_R = u_star_R;
	} else {
		const double rho_sum_inv = 1.0 / (rho_sqrt_L + rho_sqrt_R);
		const double bx_sign = (bx > 0.0 ? 1.0 : -1.0);
		u_dstar_L.rho = u_star_L.rho;
		u_dstar_R.rho = u_star_R.rho;
		u_dstar_L.mx = u_star_L.mx;
		u_dstar_R.mx = u_star_R.mx;
		u_dstar_L.Eint = u_star_L.Eint;
		u_dstar_R.Eint = u_star_R.Eint;
		for (int n = 0; n < N_scalars; ++n) {
			u_dstar_L.scalar[n] = u_star_L.scalar[n];
			u_dstar_R.scalar[n] = u_star_R.scalar[n];
		}

		// MK5: eqn (59)
		double tmp = rho_sum_inv * (rho_sqrt_L * (u_star_L.my * u_star_rho_inv_L) + rho_sqrt_R * (u_star_R.my * u_star_rho_inv_R) +
					    bx_sign * (u_star_R.by - u_star_L.by));
		u_dstar_L.my = u_dstar_L.rho * tmp;
		u_dstar_R.my = u_dstar_R.rho * tmp;
		// MK5: eqn (60)
		tmp = rho_sum_inv *
		      (rho_sqrt_L * (u_star_L.mz * u_star_rho_inv_L) + rho_sqrt_R * (u_star_R.mz * u_star_rho_inv_R) + bx_sign * (u_star_R.bz - u_star_L.bz));
		u_dstar_L.mz = u_dstar_L.rho * tmp;
		u_dstar_R.mz = u_dstar_R.rho * tmp;
		// MK5: eqn (61)
		tmp = rho_sum_inv * (rho_sqrt_L * u_star_R.by + rho_sqrt_R * u_star_L.by +
				     bx_sign * rho_sqrt_L * rho_sqrt_R * ((u_star_R.my * u_star_rho_inv_R) - (u_star_L.my * u_star_rho_inv_L)));
		u_dstar_L.by = tmp;
		u_dstar_R.by = tmp;
		// MK5: eqn (62)
		tmp = rho_sum_inv * (rho_sqrt_L * u_star_R.bz + rho_sqrt_R * u_star_L.bz +
				     bx_sign * rho_sqrt_L * rho_sqrt_R * ((u_star_R.mz * u_star_rho_inv_R) - (u_star_L.mz * u_star_rho_inv_L)));
		u_dstar_L.bz = tmp;
		u_dstar_R.bz = tmp;
		// MK5: eqn (63)
		tmp = spds[2] * bx + (u_dstar_L.my * u_dstar_L.by + u_dstar_L.mz * u_dstar_L.bz) / u_dstar_L.rho;
		u_dstar_L.E = u_star_L.E - rho_sqrt_L * bx_sign * (vb_star_L - tmp);
		u_dstar_R.E = u_star_R.E + rho_sqrt_R * bx_sign * (vb_star_R - tmp);
	}

	// Convert to arrays for simplified math

	quokka::valarray<double, fluxdim> U_L_array = {u_L.rho, u_L.mx, u_L.my, u_L.mz, u_L.E, u_L.Eint};
	quokka::valarray<double, fluxdim> U_R_array = {u_R.rho, u_R.mx, u_R.my, u_R.mz, u_R.E, u_R.Eint};
	for (int n = 0; n < N_scalars; ++n) {
		const int nstart = fluxdim - N_scalars;
		U_L_array[nstart + n] = u_L.scalar[n];
		U_R_array[nstart + n] = u_R.scalar[n];
	}

	quokka::valarray<double, fluxdim> U_star_L_array = {u_star_L.rho, u_star_L.mx, u_star_L.my, u_star_L.mz, u_star_L.E, u_star_L.Eint};
	quokka::valarray<double, fluxdim> U_star_R_array = {u_star_R.rho, u_star_R.mx, u_star_R.my, u_star_R.mz, u_star_R.E, u_star_R.Eint};
	for (int n = 0; n < N_scalars; ++n) {
		const int nstart = fluxdim - N_scalars;
		U_star_L_array[nstart + n] = u_star_L.scalar[n];
		U_star_R_array[nstart + n] = u_star_R.scalar[n];
	}

	quokka::valarray<double, fluxdim> U_dstar_L_array = {u_dstar_L.rho, u_dstar_L.mx, u_dstar_L.my, u_dstar_L.mz, u_dstar_L.E, u_dstar_L.Eint};
	quokka::valarray<double, fluxdim> U_dstar_R_array = {u_dstar_R.rho, u_dstar_R.mx, u_dstar_R.my, u_dstar_R.mz, u_dstar_R.E, u_dstar_R.Eint};
	for (int n = 0; n < N_scalars; ++n) {
		const int nstart = fluxdim - N_scalars;
		U_dstar_L_array[nstart + n] = u_dstar_L.scalar[n];
		U_dstar_R_array[nstart + n] = u_dstar_R.scalar[n];
	}

	quokka::valarray<double, fluxdim> F_L_array = {f_L.rho, f_L.mx, f_L.my, f_L.mz, f_L.E, f_L.Eint};
	quokka::valarray<double, fluxdim> F_R_array = {f_R.rho, f_R.mx, f_R.my, f_R.mz, f_R.E, f_R.Eint};
	for (int n = 0; n < N_scalars; ++n) {
		const int nstart = fluxdim - N_scalars;
		F_L_array[nstart + n] = f_L.scalar[n];
		F_R_array[nstart + n] = f_R.scalar[n];
	}

	U_dstar_L_array = spds[1] * (U_dstar_L_array - U_star_L_array);
	U_star_L_array = spds[0] * (U_star_L_array - U_L_array);
	U_dstar_R_array = spds[3] * (U_dstar_R_array - U_star_R_array);
	U_star_R_array = spds[4] * (U_star_R_array - U_R_array);

	//--- Step 6. Compute fluxes

	if (spds[0] >= 0.0) {
		// return u_L if flow is supersonic
		F_x = F_L_array;
	} else if (spds[4] <= 0.0) {
		// return u_R if flow is supersonic
		F_x = F_R_array;
	} else if (spds[1] >= 0.0) {
		// return u_star_L
		F_x = F_L_array + U_star_L_array;
	} else if (spds[2] >= 0.0) {
		// return u_dstar_L
		F_x = F_L_array + U_star_L_array + U_dstar_L_array;
	} else if (spds[3] > 0.0) {
		// return u_dstar_R
		F_x = F_R_array + U_star_R_array + U_dstar_R_array;
	} else {
		// return u_star_R
		F_x = F_R_array + U_star_R_array;
	}

	return std::make_tuple(std::move(F_x), fspd_m, fspd_p);
}
} // namespace quokka::Riemann

#endif // HLLD_HPP_