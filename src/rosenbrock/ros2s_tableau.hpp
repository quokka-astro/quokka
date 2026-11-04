#ifndef ROSENBROCK_ROS2S_TABLEAU_HPP_
#define ROSENBROCK_ROS2S_TABLEAU_HPP_
/// \file ros2s_tableau.hpp
/// \brief Coefficients of the ros2s Rosenbrock method (3 stages, order 2, embedded order 1), as in Microphysics
/// (integration/Rosenbrock/rosenbrock_tableau.H). Stage indices are 1-based, as there.

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace rosenbrock
{

struct ros2s_tableau {
	static constexpr int stages = 3;
	static constexpr amrex::Real gamma = 0.292893218813452;

	/// time of stage i, as a fraction of the step
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto ct(const int i) -> amrex::Real { return i == 2 ? 0.585786437626905 : 1.0; }

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto a(const int i, const int j) -> amrex::Real
	{
		if (i == 2 && j == 1) {
			return 2.0000000000000036;
		}
		if (i == 3 && j == 1) {
			return 6.828427124746214;
		}
		if (i == 3 && j == 2) {
			return 3.4142135623731007;
		}
		return 0.0;
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto c(const int i, const int j) -> amrex::Real
	{
		if (i == 2 && j == 1) {
			return -6.828427124746214;
		}
		if (i == 3 && j == 1) {
			return -10.949747468305889;
		}
		if (i == 3 && j == 2) {
			return -7.535533905932761;
		}
		return 0.0;
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto b(const int i) -> amrex::Real
	{
		if (i == 1) {
			return 6.828427124746214;
		}
		if (i == 2) {
			return 3.414213562373101;
		}
		return 1.0;
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto e(const int i) -> amrex::Real
	{
		if (i == 1) {
			return -0.23570226039551292;
		}
		if (i == 2) {
			return -0.23570226039551567;
		}
		return -0.13807118745769906;
	}
};

} // namespace rosenbrock

#endif // ROSENBROCK_ROS2S_TABLEAU_HPP_
