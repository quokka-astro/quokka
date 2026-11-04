#ifndef PHOTOCHEM_SYSTEM_TEST_HPP_
#define PHOTOCHEM_SYSTEM_TEST_HPP_
/// \file PhotochemSystemTest.hpp
/// \brief Unit test of the photochemistry solver system for the network a problem is built with: temperature bounds,
/// in-place cleaning of the photon number density, and the analytic Jacobian against central finite differences of the RHS.

#include <cmath>
#include <format>
#include <iostream>

#include "radiation/photochemistry.hpp"

/// \param dust_kappa dust opacity used in the test state (exercises the dust rows when DUST_CHEMICAL_BAND_ABSORPTION is defined)
/// \param jacobian_at_fixed_T true if the network's species columns hold T fixed (the handwritten networks ignore dT/dn_i);
///        false if they are the full derivative at fixed e (the JAFF-generated networks)
/// \return 0 if all checks pass
inline auto TestPhotochemSystem(const amrex::Real dust_kappa = 0.0, const bool jacobian_at_fixed_T = true) -> int
{
	using amrex::Real;
	using quokka::photochemistry::detail::PhotochemSystem;
	namespace pd = quokka::photochemistry::detail;
	constexpr int N = network::NumEqs;
	int status = 0;
	auto report = [&status](const char *name, bool ok) {
		std::cout << std::format("PhotochemSystem {:<34}{}\n", name, ok ? "PASS" : "FAIL");
		if (!ok) {
			status = 1;
		}
	};

	PhotochemSystem sys{};
	sys.params = network::NetworkParams{};
	sys.state.rho = 1.0e2 * spmasses[Species::H];
	sys.state.c_hat = C::c_light;
	sys.state.dust_kappa = dust_kappa;

	// a partly ionized state at T ~ 8000 K with photons and thermal radiation
	amrex::GpuArray<Real, NumSpec> n{};
	n[Species::H] = 30.0;
	n[Species::H_p] = 70.0;
	n[Species::e] = 70.0;
	rosenbrock::Vector<N> y{};
	for (int i = 0; i < NumSpec; ++i) {
		y[i] = n[i];
	}
	y[pd::i_enuc] = quokka::photochemistry::EnergyFromTemperature<NumSpec>(8000.0, n, spmasses, gammas);
	for (int b = 0; b < NumChemBands; ++b) {
		y[pd::i_rad0 + NumChemRadVarsPerGroup * b] = 1.0e3;
		if (NumChemRadVarsPerGroup > 1) {
			y[pd::i_rad0 + NumChemRadVarsPerGroup * b + 1] = 1.0; // flux factor
		}
	}
	for (int b = 0; b < NumThermalBands; ++b) {
		y[pd::i_thermal0 + b] = 1.0e-12;
	}

	// temperature above the upper bound: the RHS is zero
	{
		rosenbrock::Vector<N> yhot = y;
		yhot[pd::i_enuc] = quokka::photochemistry::EnergyFromTemperature<NumSpec>(1.0e12, n, spmasses, gammas);
		rosenbrock::Vector<N> ydot{};
		ydot[0] = 1.0;
		sys.rhs(0.0, yhot, ydot);
		bool all_zero = true;
		for (int i = 0; i < N; ++i) {
			all_zero = all_zero && (ydot[i] == 0.0);
		}
		report("rhs zero above T_max", all_zero);
	}

	// a slightly negative photon number density is cleaned in place to SMALL_X_SAFE
	{
		rosenbrock::Vector<N> yneg = y;
		yneg[pd::i_rad0] = -1.0e-40;
		rosenbrock::Vector<N> ydot{};
		sys.rhs(0.0, yneg, ydot);
		report("rhs cleans photon number", yneg[pd::i_rad0] == quokka::photochemistry::photochem_small_x_safe);
	}

	// Analytic Jacobian vs central finite differences of the RHS. If the network's Jacobian holds T fixed through dedT (it
	// ignores dT/dn_i), the species columns are compared with e adjusted to keep T fixed. The step is large enough that
	// roundoff in the photoheating-dominated energy row (|f| ~ 1e8) stays below the tolerance; the RHS is bilinear in the
	// species at fixed T, so this adds no truncation error there.
	{
		rosenbrock::Vector<N> ybase = y;
		rosenbrock::Vector<N> f0{};
		sys.rhs(0.0, ybase, f0);
		rosenbrock::Matrix<N> J{};
		sys.jac(0.0, ybase, J);
		const Real T0 = sys.state.T;
		Real jmax = 0.0;
		for (int j = 0; j < N; ++j) {
			for (int i = 0; i < N; ++i) {
				jmax = amrex::max(jmax, std::abs(J(i, j)));
			}
		}
		bool ok = true;
		for (int j = 0; j < N; ++j) {
			const Real h = 1.0e-4 * amrex::max(std::abs(ybase[j]), Real(1.0e-30));
			rosenbrock::Vector<N> yp = ybase;
			rosenbrock::Vector<N> ym = ybase;
			yp[j] += h;
			ym[j] -= h;
			if (jacobian_at_fixed_T && j < NumSpec) { // keep T fixed: recompute e from T0 with the perturbed composition
				amrex::GpuArray<Real, NumSpec> np{};
				amrex::GpuArray<Real, NumSpec> nm{};
				for (int s = 0; s < NumSpec; ++s) {
					np[s] = yp[s];
					nm[s] = ym[s];
				}
				yp[pd::i_enuc] = quokka::photochemistry::EnergyFromTemperature<NumSpec>(T0, np, spmasses, gammas);
				ym[pd::i_enuc] = quokka::photochemistry::EnergyFromTemperature<NumSpec>(T0, nm, spmasses, gammas);
			}
			rosenbrock::Vector<N> fp{};
			rosenbrock::Vector<N> fm{};
			sys.rhs(0.0, yp, fp);
			sys.rhs(0.0, ym, fm);
			for (int i = 0; i < N; ++i) {
				const Real fd = (fp[i] - fm[i]) / (2.0 * h);
				if (std::abs(J(i, j)) > 1.0e-12 * jmax && std::abs(fd - J(i, j)) > 1.0e-5 * std::abs(J(i, j))) {
					std::cout << std::format("  J({}, {}) = {:.8e}, finite difference {:.8e}\n", i, j, J(i, j), fd);
					ok = false;
				}
			}
		}
		report("jac matches finite differences", ok);
	}
	return status;
}

#endif // PHOTOCHEM_SYSTEM_TEST_HPP_
