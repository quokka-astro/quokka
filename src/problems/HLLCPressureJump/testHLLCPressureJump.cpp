//==============================================================================
// Copyright 2026 Yaoguang Pei.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testHLLCPressureJump.cpp
/// \brief Tests the HLLC carbuncle sensor on normal pressure jumps.
///

#include <array>
#include <cmath>

#include "AMReX.H"
#include "AMReX_GpuAsyncArray.H"
#include "AMReX_GpuLaunch.H"
#include "AMReX_Print.H"

#include "hydro/HLLC.hpp"
#include "physics_info.hpp"

struct HLLCPressureJumpProblem {};

template <> struct quokka::EOS_Traits<HLLCPressureJumpProblem> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.8821957483241444e4;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<HLLCPressureJumpProblem> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr bool is_hydro_enabled = true;
};

auto problem_main() -> int
{
	constexpr int num_cases = 3;
	std::array<double, num_cases> host_flux = {NAN, NAN, NAN};
	amrex::AsyncArray async_flux(host_flux.data(), num_cases);
	double *const flux_out = async_flux.data();

	amrex::ParallelFor(num_cases, [=] AMREX_GPU_DEVICE(int n) noexcept {
		quokka::HydroState<0, 0> left{};
		quokka::HydroState<0, 0> right{};

		left.u = 1.32835071447571463e4;
		left.v = -1.36306805368190515e5;
		left.w = -6.67287637958019186e4;
		left.cs = quokka::EOS_Traits<HLLCPressureJumpProblem>::cs_isothermal;

		right.u = 5.82950699962116778e3;
		right.v = -2.94541597662591012e4;
		right.w = -2.72877142624116423e4;
		right.cs = quokka::EOS_Traits<HLLCPressureJumpProblem>::cs_isothermal;

		if (n == 0) {
			// This state previously activated the transverse carbuncle sensor
			// strongly enough to reverse the pressure-driven mass flux.
			left.rho = 1.16764051107606764e-22;
			left.P = 4.13655430795964494e-14;
			right.rho = 4.45656880879147046e-22;
			right.P = 1.57881117774304212e-13;
		} else if (n == 1) {
			// Exercise the smooth transition between the corrected and
			// unmodified pressure terms.
			left.rho = 2.4698949200923355e-22;
			left.P = 8.75e-14;
			right.rho = 3.1755791829758596e-22;
			right.P = 1.125e-13;
		} else {
			// Preserve the published correction when pressure is nearly
			// continuous across the current face.
			left.rho = 2.699620473536431e-22;
			left.P = 9.563839720987232e-14;
			right.rho = 2.9245888463311335e-22;
			right.P = 1.0360826364402834e-13;
		}

		left.E = 0.5 * left.rho * (left.u * left.u + left.v * left.v + left.w * left.w);
		right.E = 0.5 * right.rho * (right.u * right.u + right.v * right.v + right.w * right.w);

		constexpr double du = -9.75035757244138222e3;
		constexpr double dw = -1.06702277106021327e5;
		auto const flux =
		    quokka::Riemann::HLLC<HLLCPressureJumpProblem, 0, 0, 6>(left, right, quokka::EOS_Traits<HLLCPressureJumpProblem>::gamma, du, dw);
		flux_out[n] = flux[0];
	});
	async_flux.copyToHost(host_flux.data(), num_cases);

	constexpr std::array<double, num_cases> expected_flux = {-2.1589381771725202e-19, 2.5203897321703885e-18, 2.979327941432850e-18};
	constexpr double relative_tolerance = 1.0e-9;
	bool all_ok = true;
	for (int n = 0; n < num_cases; ++n) {
		const double relative_error = std::abs((host_flux[n] - expected_flux[n]) / expected_flux[n]);
		amrex::Print() << "case " << n << ": mass flux = " << host_flux[n] << ", expected = " << expected_flux[n]
			       << ", relative error = " << relative_error << "\n";
		if (!std::isfinite(host_flux[n]) || relative_error >= relative_tolerance) {
			all_ok = false;
		}
	}
	if (!(host_flux[0] < 0.0)) {
		amrex::Print() << "pressure-jump case has the wrong mass-flux direction\n";
		all_ok = false;
	}

	amrex::Print() << (all_ok ? "test passed\n" : "test failed\n");
	return all_ok ? 0 : 1;
}
