// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testISMCoolingOneZone.cpp
/// \brief Host-only unit test of the network thermophotochemistry_ism_handwritten (mode "jacobian").
///
/// For four gas states, with G0 = 1 and n_gamma = 1e-3, it checks that
///  - every entry of actual_jac matches a centred finite difference of actual_rhs, and
///  - the specific energy rate of actual_rhs equals the sum of the rate functions of cooling/ISMCooling.hpp called directly.

#include "AMReX_Array.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "cooling/GrackleNTTable.hpp"
#include "cooling/ISMCooling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>

#include "actual_rhs.H"
#include "burn_type.H"
#include "extern_parameters.H"
#include "fundamental_constants.H"
#include "network.H"

namespace ISM = quokka::ISMCooling;

constexpr Real fd_step = 1.0e-6;
constexpr Real jac_rtol = 1.0e-4;
// Roundoff floor of the finite difference, relative to the size of the terms in the row: a difference of two values of size f taken over a
// step h = 1e-6 x has an error of about 1e-10 f / x, and the floor leaves a margin of ten.
constexpr Real fd_noise_floor = 1.0e-9;
constexpr Real sum_rtol = 1.0e-12;
constexpr Real G0_test = 1.0;
constexpr Real n_gamma_test = 1.0e-3;
constexpr Real n_gamma_flux_test = 5.0e-4;
constexpr Real dust_kappa_test = 1.0e-2;
constexpr int max_reports = 10;

struct GasState {
	const char *name;
	Real nH;
	Real T;
	Real xHp;
};

constexpr std::array<GasState, 4> gas_states = {{
    {"CNM", 30.0, 60.0, 1.0e-4},
    {"WNM", 0.5, 8000.0, 0.02},
    {"HII", 10.0, 9000.0, 0.999},
    {"partial", 1.0, 1.2e4, 0.5},
}};

static auto MakeState(const GasState &g) -> burn_t
{
	burn_t state;
	state.xn[e_idx] = g.xHp * g.nH;
	state.xn[Hp_idx] = g.xHp * g.nH;
	state.xn[H_idx] = (1.0 - g.xHp) * g.nH;
	state.T = g.T;
	state.rho = g.nH * C::m_u;
	state.c_hat = C::c_light;
	state.dust_kappa = dust_kappa_test;
	state.aux[0] = G0_test;
	state.rn[0] = n_gamma_test;
	state.rn[1] = n_gamma_flux_test;
	// an ideal monatomic gas of all particles, so that the T column of the Jacobian has a realistic 1/dedT
	state.dedT = 1.5 * C::k_B * (state.xn[e_idx] + state.xn[Hp_idx] + state.xn[H_idx]) / state.rho;
	return state;
}

/// Perturb variable `col` (a column of the Jacobian) of the state by `delta`. Energy is perturbed through T, with dT = delta / dedT.
/// Returns false if the column has no state variable behind it.
static auto Perturb(burn_t &state, int col, Real delta) -> bool
{
	if (col >= 1 && col <= NumSpec) {
		state.xn[col - 1] += delta;
	} else if (col == net_ienuc) {
		state.T += delta / state.dedT;
	} else if (col == i_ngamma) {
		state.rn[0] += delta;
	} else if (col == i_ngamma_flux) {
		state.rn[1] += delta;
	} else {
		return false;
	}
	return true;
}

static auto Variable(const burn_t &state, int col) -> Real
{
	if (col >= 1 && col <= NumSpec) {
		return state.xn[col - 1];
	}
	if (col == net_ienuc) {
		return state.T * state.dedT;
	}
	return (col == i_ngamma) ? state.rn[0] : state.rn[1];
}

static auto CheckJacobian(const GasState &g) -> int
{
	const burn_t state = MakeState(g);
	JacNetArray2D jac;
	actual_jac(state, jac);

	int failures = 0;
	for (int col = 1; col <= INT_NEQS; ++col) {
		burn_t sp = state;
		burn_t sm = state;
		const Real h = fd_step * std::abs(Variable(state, col));
		if (!Perturb(sp, col, h) || !Perturb(sm, col, -h)) {
			// no state variable behind this column (the dust energy): the analytic column must vanish
			for (int row = 1; row <= INT_NEQS; ++row) {
				failures += (jac(row, col) != 0.0) ? 1 : 0;
			}
			continue;
		}
		Array1D<Real, 1, INT_NEQS> yp;
		Array1D<Real, 1, INT_NEQS> ym;
		actual_rhs(sp, yp);
		actual_rhs(sm, ym);
		// the actual step in the variable, which differs from h by roundoff
		const Real dvar = (col == net_ienuc) ? (sp.T - sm.T) * state.dedT : Variable(sp, col) - Variable(sm, col);
		for (int row = 1; row <= INT_NEQS; ++row) {
			const Real fd = (yp(row) - ym(row)) / dvar;
			// the size of the terms in this row: sum over the variables of |d ydot / d x| x
			Real row_scale = 0.0;
			for (int c = 1; c <= net_ienuc + NumChemRadEqs; ++c) {
				row_scale += std::abs(jac(row, c) * Variable(state, c));
			}
			const Real err = std::abs(fd - jac(row, col));
			if (!(err <= jac_rtol * std::abs(jac(row, col)) + fd_noise_floor * row_scale / std::abs(Variable(state, col)))) {
				if (failures < max_reports) {
					std::cout << "  " << g.name << ": jac(" << row << ", " << col << ") = " << jac(row, col) << ", finite difference " << fd
						  << '\n';
				}
				++failures;
			}
		}
	}
	return failures;
}

/// The energy rate [erg cm^-3 s^-1] from the rate functions, written out independently of actual_rhs.
static auto DirectEnergySum(const GasState &g) -> Real
{
	const burn_t state = MakeState(g);
	const Real T = state.T;
	const Real n_e = state.xn[e_idx];
	const Real n_HI = state.xn[H_idx];
	const Real n_Hp = state.xn[Hp_idx];

	// the new terms, on n_e^max
	const auto *table = ISM::GetGrackleNTTable();
	const auto nt = ISM::LookupGrackleNT(n_HI + n_Hp, T, *table);
	const auto ne_max = ISM::ElectronDensityMax(n_HI, n_Hp, T, nt.xe, nt.dxe_dT, nt.dxe_dnH, ism_metallicity, cr_ionization_rate_HI,
						    get_recombination_coefficient(T), dRecombination_coefficient_dT(T));
	const Real Zp = ism_metallicity;
	Real sum = 0.0;
	sum -= ISM::LyaCooling(n_HI, ne_max.ne, T).val;
	sum -= ISM::NebularCooling(n_Hp, ne_max.ne, T, Zp).val;
	sum -= ISM::CIICooling(n_HI, n_Hp, ne_max.ne, T, Zp).val;
	sum -= ISM::OICooling(n_HI, ne_max.ne, T, Zp).val;
	sum -= ISM::GrainRecombinationCooling(n_HI, n_Hp, ne_max.ne, T, G0_test, Zp).val;
	sum += ISM::PEHeating(n_HI, n_Hp, ne_max.ne, T, G0_test, Zp).val;
	sum += ISM::CRHeating(n_HI, cr_ionization_rate_HI, 0.0).val;

	// the kept H terms, on the network n_e
	const Real photoionization = state.c_hat * get_photoionization_coefficient(T) * n_HI * state.rn[0];
	const Real collisional = n_e * n_HI * get_collisional_ionization_coefficient(T);
	sum += get_ionization_heating_coefficient(T) * photoionization;
	sum -= get_recombination_cooling_coefficient(T) * n_e * n_Hp;
	sum -= RydbergEnergy * collisional;
	sum -= 1.3 * 1.427e-27 * std::sqrt(T) * n_e * n_Hp;
	return sum;
}

static auto CheckEnergySum(const GasState &g) -> int
{
	const burn_t state = MakeState(g);
	Array1D<Real, 1, INT_NEQS> ydot;
	actual_rhs(state, ydot);
	const Real got = ydot(net_ienuc) * state.rho;
	const Real expected = DirectEnergySum(g);
	if (!(std::abs(got - expected) <= sum_rtol * std::abs(expected))) {
		std::cout << "  " << g.name << ": energy rate " << got << ", sum of the rate functions " << expected << '\n';
		return 1;
	}
	return 0;
}

auto problem_main() -> int
{
	amrex::ParmParse const pp("ism_test");
	amrex::ParmParse const pp_cooling("ism_cooling");
	std::string mode;
	std::string table_path;
	pp.get("mode", mode);
	pp_cooling.get("grackle_nT_table", table_path);
	if (mode != "jacobian") {
		std::cout << "ISMCoolingOneZone: unknown ism_test.mode '" << mode << "'\n";
		return 1;
	}

	init_extern_parameters();

	// nothing in the simulation registers the table yet, so the test does
	const auto data = ISM::ReadGrackleNTTable(table_path);
	const auto attrs = ISM::ReadGrackleNTAttributes(table_path);
	ISM::RegisterGrackleNTTable({data->const_tables_host(), attrs.zmet, attrs.H_mass_fraction}, {data->const_tables(), attrs.zmet, attrs.H_mass_fraction});

	int failures = 0;
	for (const GasState &g : gas_states) {
		const int jac_failures = CheckJacobian(g);
		const int sum_failures = CheckEnergySum(g);
		std::cout << "ISMCoolingOneZone " << g.name << ": " << jac_failures << " Jacobian entries and " << sum_failures << " energy sums failed\n";
		failures += jac_failures + sum_failures;
	}
	std::cout << (failures == 0 ? "ISMCoolingOneZone: PASSED\n" : "ISMCoolingOneZone: FAILED\n");
	return (failures == 0) ? 0 : 1;
}
