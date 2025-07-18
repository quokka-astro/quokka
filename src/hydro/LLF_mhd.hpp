#ifndef LLF_MHD_HPP_ // NOLINT
#define LLF_MHD_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include <AMReX.H>
#include <AMReX_REAL.H>

#include "hydro/HydroState.hpp"
#include "util/valarray.hpp"

namespace quokka::Riemann
{
// Local Lax-Friedrichs (LLF) / Rusanov solver
template <typename problem_t, int N_scalars, int N_mscalars, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto LLF_MHD(quokka::HydroState<N_scalars, N_mscalars> const &sL, quokka::HydroState<N_scalars, N_mscalars> const &sR,
						 const double gamma, const double bx) -> std::tuple<quokka::valarray<double, fluxdim>, double, double>
{
	// initialize left and right conserved states
	ConsHydro1D<N_scalars> u_L{};
	ConsHydro1D<N_scalars> u_R{};
	// initialize fluxes at left and right side of the interface
	ConsHydro1D<N_scalars> f_L{};
	ConsHydro1D<N_scalars> f_R{};
	quokka::valarray<double, fluxdim> Du = {};
	quokka::valarray<double, fluxdim> F = {};

	double const bx_sq = SQUARE(bx);

	// compute L/R states for select conserved variables
	// (group transverse vector components for floating-point associativity symmetry)
	// magnetic pressure
	double const pb_L = 0.5 * (bx_sq + (SQUARE(sL.by) + SQUARE(sL.bz)));
	double const pb_R = 0.5 * (bx_sq + (SQUARE(sR.by) + SQUARE(sR.bz)));
	// kinetic energy
	double const ke_L = 0.5 * sL.rho * (SQUARE(sL.u) + (SQUARE(sL.v) + SQUARE(sL.w)));
	double const ke_R = 0.5 * sR.rho * (SQUARE(sR.u) + (SQUARE(sR.v) + SQUARE(sR.w)));

	double ptot_L = sL.P + pb_L;
	double ptot_R = sR.P + pb_R;

	const double fspd_L = FastMagnetoSonicSpeed(gamma, sL, bx);
	const double fspd_R = FastMagnetoSonicSpeed(gamma, sR, bx);
	const double fspd_m = -std::min(0.0, std::min(sL.u - fspd_L, sR.u - fspd_R));
	const double fspd_p = std::max(0.0, std::max(sL.u + fspd_L, sR.u + fspd_R));

	// set left conserved states
	u_L.rho = sL.rho;
	u_L.mx = sL.u * u_L.rho;
	u_L.my = sL.v * u_L.rho;
	u_L.mz = sL.w * u_L.rho;
	u_L.E = ke_L + pb_L + sL.P / (gamma - 1.0);
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

	//--- Step 2.  Compute wave speeds in L,R states (see Toro eq. 10.43)

	const amrex::Real a = 0.5 * std::max(std::abs(sL.u) + fspd_L, std::abs(sR.u) + fspd_R);
	//--- Step 3.  Compute L/R fluxes
	// fluxes on the left side of the interface, non-barotropic case
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

	// convert to arrays for simplified math
	quokka::valarray<double, fluxdim> U_L_array = {u_L.rho, u_L.mx, u_L.my, u_L.mz, u_L.E, u_L.Eint};
	quokka::valarray<double, fluxdim> U_R_array = {u_R.rho, u_R.mx, u_R.my, u_R.mz, u_R.E, u_R.Eint};
	for (int n = 0; n < N_scalars; ++n) {
		const int nstart = fluxdim - N_scalars;
		U_L_array[nstart + n] = u_L.scalar[n];
		U_R_array[nstart + n] = u_R.scalar[n];
	}
	quokka::valarray<double, fluxdim> F_L_array = {f_L.rho, f_L.mx, f_L.my, f_L.mz, f_L.E, f_L.Eint};
	quokka::valarray<double, fluxdim> F_R_array = {f_R.rho, f_R.mx, f_R.my, f_R.mz, f_R.E, f_R.Eint};
	for (int n = 0; n < N_scalars; ++n) {
		const int nstart = fluxdim - N_scalars;
		F_L_array[nstart + n] = f_L.scalar[n];
		F_R_array[nstart + n] = f_R.scalar[n];
	}

	//--- Step 4.  Compute difference in L/R states dU

	Du = U_R_array - U_L_array;

	//--- Step 5.  Compute the LLF flux at interface (see Toro eq. 10.42).

	F = 0.5 * (F_L_array + F_R_array) - a * Du;

	return std::make_tuple(std::move(F), fspd_m, fspd_p);
}
} // namespace quokka::Riemann

#endif // LLF_MHD_HPP_
