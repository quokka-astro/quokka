#ifndef DUSTSTATE_HPP_ // NOLINT
#define DUSTSTATE_HPP_

#include "AMReX_Array.H"
#include "util/valarray.hpp"

namespace quokka
{
struct DustState {
	double rho; // density
	double u;   // normal velocity component
	double v;   // transverse velocity component
	double w;   // 2nd transverse velocity component
};

} // namespace quokka

#endif // DUSTSTATE_HPP_
