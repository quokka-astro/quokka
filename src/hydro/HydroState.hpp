#ifndef HYDROSTATE_HPP_ // NOLINT
#define HYDROSTATE_HPP_

#include "AMReX_Array.H"
#include "util/valarray.hpp"

namespace quokka
{
template <int Nall, int Nmass> struct HydroState {
	double rho;				   // density
	double u;				   // normal velocity component
	double v;				   // transverse velocity component
	double w;				   // 2nd transverse velocity component
	double P;				   // pressure
	double cs;				   // adiabatic sound speed
	double E;				   // total energy density
	double Eint;				   // internal energy density
	double by;				   // transverse bfield component
	double bz;				   // 2nd transverse bfield density
	amrex::GpuArray<double, Nall> scalar;	   // passive scalars
	amrex::GpuArray<double, Nmass> massScalar; // mass scalars
};

struct DustState {
	double rho;				   // density
	double u;				   // normal velocity component
	double v;				   // transverse velocity component
	double w;				   // 2nd transverse velocity component
};

} // namespace quokka

// density, momentum, total energy, transverse magnetic field
template <int N_passiveScalars> struct ConsHydro1D {
	double rho;					   // density
	double mx;					   // x-momentum
	double my;					   // y-momentum
	double mz;					   // z-momentum
	double E;					   // total energy density
	double Eint;					   // specific internal energy
	double by;					   // y-magnetic field
	double bz;					   // z-magnetic field
	quokka::valarray<double, N_passiveScalars> scalar; // passive scalars, problem defined
};

template <class T> constexpr auto SQUARE(const T x) -> T { return x * x; }

template <int N_scalars, int N_mscalars>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto FastMagnetoSonicSpeed(double gamma, quokka::HydroState<N_scalars, N_mscalars> const state, const double bx) -> double
{
	double gamma_pressure = gamma * state.P;
	double byz_sq = SQUARE(state.by) + SQUARE(state.bz);
	double const b_sq = SQUARE(bx) + byz_sq;
	double const b_plus_gamma_pressure = b_sq + gamma_pressure;
	double const b_minus_gamma_pressure = b_sq - gamma_pressure;
	return std::sqrt(0.5 * (b_plus_gamma_pressure + std::sqrt(b_minus_gamma_pressure * b_minus_gamma_pressure + 4.0 * gamma_pressure * byz_sq)) /
			 state.rho);
}

#endif // HYDROSTATE_HPP_
