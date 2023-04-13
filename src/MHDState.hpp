#ifndef MHDSTATE_HPP_ // NOLINT
#define MHDSTATE_HPP_

#include <array>

namespace quokka
{
struct MHDState {
	double rho; // density
	double vx;  // normal velocity component
	double vy;  // transverse velocity component
	double vz;  // 2nd transverse velocity component
	double P;   // pressure
	double E;   // total energy density
	double by;  // transverse bfield component
	double bz;  // 2nd transverse bfield density
};

} // namespace quokka

#endif // MHDSTATE_HPP_