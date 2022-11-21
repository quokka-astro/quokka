#ifndef HYDROSTATE_HPP_ // NOLINT
#define HYDROSTATE_HPP_

#include <array>
namespace quokka
{
template <int N> struct HydroState {
	double rho;		      // density
	double vx;		      // x-vel
	double vy;		      // y-vel
	double vz;		      // z-vel
	double P;		      // pressure
	double u;		      // normal velocity component
	double cs;		      // adiabatic sound speed
	double E;		      // total energy density
	double Eint;		      // internal energy density
	std::array<double, N> scalar; // passive scalars
};

} // namespace quokka

#endif // HYDROSTATE_HPP_