// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testISMCoolingOneZone.cpp
/// \brief Tests of the network thermophotochemistry_ism_handwritten, selected by ism_test.mode.
///
/// Mode "startup" constructs a QuokkaSimulation with a uniform gas cell and runs one step. It exercises the startup checks of the photochemistry
/// cooling path (G0_external, ism_cooling.grackle_nT_table and the metallicity of the table), which abort with a message naming the parameter.
///
/// Mode "equilibrium" evolves a row of cells at fixed n_H from 1e-2 to 1e2 cm^-3 to thermal equilibrium (G0 = 1, Z' = 1, no photons, hydro off) and
/// compares their temperatures with the Python steady state of extern/cooling/ism_cooling_reference.py (ism_equilibrium_reference.csv), then checks that
/// the curve has a warm and a cold phase. It is 1D only.
///
/// Mode "jacobian" is a host-only unit test:
///
/// For six gas states, chosen so that every branch of n_e^max is active in at least one (asserted), with G0 = 1, it checks that
///  - every entry of actual_jac matches a centred finite difference of actual_rhs, with n_gamma = 1e-3 and again with no photons (so that
///    photoheating does not set the scale of the energy row and the entries of the ISM terms are resolved),
///  - get_ism_energy_terms matches a centred finite difference of its own value in T, n_HI and n_H+, and
///  - the specific energy rate of actual_rhs equals the sum of the rate functions of cooling/ISMCooling.hpp called directly.

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "cooling/GrackleNTTable.hpp"
#include "cooling/ISMCooling.hpp"
#include "util/fextract.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "actual_eos_data.H"
#include "actual_rhs.H"
#include "burn_type.H"
#include "eos.H"
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
	int branch; ///< the active branch of n_e^max: 0 network, 1 CIE, 2 C+ floor, 3 CR floor
};

constexpr std::array<GasState, 6> gas_states = {{
    {"CNM", 30.0, 60.0, 1.0e-4, 2},
    {"WNM", 0.5, 8000.0, 0.02, 0},
    {"HII", 10.0, 9000.0, 0.999, 0},
    {"partial", 1.0, 1.2e4, 0.5, 0},
    {"CIE", 1.0, 2.0e4, 1.0e-3, 1},
    {"CR", 1.0, 120.0, 1.0e-5, 3}, // not T = 100 K: the recombination cooling fit switches on there,
}};

static auto MakeState(const GasState &g, bool photons = true) -> burn_t
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
	state.rn[0] = photons ? n_gamma_test : 0.0;
	state.rn[1] = photons ? n_gamma_flux_test : 0.0;
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

static auto CheckJacobian(const GasState &g, bool photons) -> int
{
	const burn_t state = MakeState(g, photons);
	JacNetArray2D jac;
	actual_jac(state, jac);

	int failures = 0;
	for (int col = 1; col <= INT_NEQS; ++col) {
		burn_t sp = state;
		burn_t sm = state;
		const Real h = fd_step * std::abs(Variable(state, col));
		if (h == 0.0) {
			continue; // the photon variables of the pass without photons: no relative step is defined
		}
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
					std::cout << "  " << g.name << (photons ? "" : " (no photons)") << ": jac(" << row << ", " << col
						  << ") = " << jac(row, col) << ", finite difference " << fd << '\n';
				}
				++failures;
			}
		}
	}
	return failures;
}

static auto ElectronDensityOf(const burn_t &state) -> ISM::ElectronDensity
{
	const Real T = state.T;
	const auto nt = ISM::LookupGrackleNT(state.xn[H_idx] + state.xn[Hp_idx], T, *ISM::GetGrackleNTTable());
	return ISM::ElectronDensityMax(state.xn[H_idx], state.xn[Hp_idx], T, nt.xe, nt.dxe_dT, nt.dxe_dnH, ism_metallicity, cr_ionization_rate_HI,
				       get_recombination_coefficient(T), dRecombination_coefficient_dT(T));
}

static auto CheckBranch(const GasState &g) -> int
{
	const int branch = ElectronDensityOf(MakeState(g)).branch;
	if (branch != g.branch) {
		std::cout << "  " << g.name << ": n_e^max branch " << branch << ", expected " << g.branch << '\n';
		return 1;
	}
	return 0;
}

/// get_ism_energy_terms against a centred finite difference of its value in T, n_HI and n_H+.
static auto CheckIsmTerms(const GasState &g) -> int
{
	const burn_t state = MakeState(g);
	const auto an = get_ism_energy_terms(state);
	const Real T = state.T;
	const Real nHI = state.xn[H_idx];
	const Real nHp = state.xn[Hp_idx];
	// the size of the terms: roundoff floor of the finite differences
	const Real scale = std::abs(an.dT * T) + std::abs(an.dnHI * nHI) + std::abs(an.dnHp * nHp);
	int failures = 0;
	for (int v = 0; v < 3; ++v) {
		burn_t sp = state;
		burn_t sm = state;
		const Real x0 = (v == 0) ? T : ((v == 1) ? nHI : nHp);
		const Real h = fd_step * x0;
		Real &xp = (v == 0) ? sp.T : ((v == 1) ? sp.xn[H_idx] : sp.xn[Hp_idx]);
		Real &xm = (v == 0) ? sm.T : ((v == 1) ? sm.xn[H_idx] : sm.xn[Hp_idx]);
		xp += h;
		xm -= h;
		const Real fd = (get_ism_energy_terms(sp).val - get_ism_energy_terms(sm).val) / (xp - xm);
		const Real analytic = (v == 0) ? an.dT : ((v == 1) ? an.dnHI : an.dnHp);
		if (!(std::abs(fd - analytic) <= jac_rtol * std::abs(analytic) + fd_noise_floor * scale / x0)) {
			std::cout << "  " << g.name << ": get_ism_energy_terms partial " << v << " (T, nHI, nHp): analytic " << analytic
				  << ", finite difference " << fd << '\n';
			++failures;
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

struct ISMCoolingOneZone {};

constexpr double c_hat = C::c_light;
constexpr double Erad_floor_ = 1.0e-60; // erg cm^-3

template <> struct quokka::EOS_Traits<ISMCoolingOneZone> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<ISMCoolingOneZone> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int numMassScalars = NumSpec;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr int nGroups = NumThermalBands + NumChemBands;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<ISMCoolingOneZone> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr int beta_order = 0;
	static constexpr auto ChemBandsPowerLawIndex() { return ChemBandsPowerLawIndex_; }
	static constexpr auto ChemBands() { return ChemBandsHeader(); }
};

// The network computes the energy dust absorbs from the chemical band (DUST_CHEMICAL_BAND_ABSORPTION); there is no coupling model, so it is dropped.
template <> struct ISM_Traits<ISMCoolingOneZone> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = false;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = false;
	static constexpr bool thermal_band_photochemistry = false;
	static constexpr bool dust_chemical_band_absorption = true;
};

template <> struct SimulationData<ISMCoolingOneZone> {
	amrex::Real temperature{};
	amrex::Real n_H{};
	amrex::Real x_Hp{};
	bool equilibrium{};			///< mode "equilibrium": the cells span n_H_min <= n_H <= n_H_max, log-spaced, instead of being uniform
	amrex::Real n_H_min{};			///< n_H of the first cell [cm^-3] (mode "equilibrium")
	amrex::Real n_H_max{};			///< n_H of the last cell [cm^-3] (mode "equilibrium")
	std::vector<amrex::Real> time_history_; ///< the time of each recorded step [s] (mode "equilibrium")
	std::vector<std::vector<amrex::Real>> T_history_; ///< the temperature of every cell at each recorded step [K] (mode "equilibrium")
};

template <> void QuokkaSimulation<ISMCoolingOneZone>::preCalculateInitialConditions()
{
	init_extern_parameters();

	amrex::ParmParse const pp("ism_test");
	userData_.temperature = 8000.0_rt;
	userData_.n_H = 1.0_rt;
	userData_.x_Hp = 0.01_rt;
	pp.query("temperature", userData_.temperature);
	pp.query("n_H", userData_.n_H);
	pp.query("x_Hp", userData_.x_Hp);
	std::string mode;
	pp.get("mode", mode);
	userData_.equilibrium = (mode == "equilibrium");
	userData_.n_H_min = 1.0e-2_rt;
	userData_.n_H_max = 1.0e2_rt;
	pp.query("n_H_min", userData_.n_H_min);
	pp.query("n_H_max", userData_.n_H_max);

	amrex::Real small_temp = 1.0e-2_rt;
	amrex::Real small_dens = 1.0e-60_rt;
	eos_init(small_temp, small_dens);
	network_init();
}

// The chemical bands couple to the gas only through photochemistry, so their opacities are zero.
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ISMCoolingOneZone>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ISMCoolingOneZone>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> void QuokkaSimulation<ISMCoolingOneZone>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// In mode "equilibrium" cell i has n_H(i) = n_H_min (n_H_max / n_H_min)^(i / (n - 1)). The state below is that of n_H = 1 cm^-3: at fixed T and
	// composition the specific energy does not depend on the density, and the densities scale with n_H(i).
	const bool equilibrium = userData_.equilibrium;
	const amrex::Real n_H_min = userData_.n_H_min;
	const amrex::Real log_n_H_ratio = std::log(userData_.n_H_max / userData_.n_H_min);
	const int n_cells = geom[0].Domain().length(0);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!equilibrium || (AMREX_SPACEDIM == 1 && n_cells > 1),
					 "ISMCoolingOneZone: mode equilibrium needs a 1D build with n_cell > 1");
	const amrex::Real n_H_base = equilibrium ? 1.0_rt : userData_.n_H;

	burn_t state;
	state.xn[Species::H_p] = userData_.x_Hp * n_H_base;
	state.xn[Species::H] = n_H_base - state.xn[Species::H_p];
	state.xn[Species::e] = state.xn[Species::H_p];
	state.T = userData_.temperature;
	Real rhotot = 0.0_rt;
	for (int n = 0; n < NumSpec; ++n) {
		rhotot += state.xn[n] * spmasses[n];
	}
	state.rho = rhotot;
	eos(eos_input_rt, state);
	const auto Egas0 = state.e * rhotot;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real scale =
		    equilibrium ? n_H_min * std::exp(log_n_H_ratio * static_cast<amrex::Real>(i) / static_cast<amrex::Real>(n_cells - 1)) : 1.0_rt;
		for (int g = 0; g < Physics_Traits<ISMCoolingOneZone>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<ISMCoolingOneZone>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad_floor_;
			state_cc(i, j, k, RadSystem<ISMCoolingOneZone>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<ISMCoolingOneZone>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<ISMCoolingOneZone>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
		}
		state_cc(i, j, k, RadSystem<ISMCoolingOneZone>::gasEnergy_index) = scale * Egas0;
		state_cc(i, j, k, RadSystem<ISMCoolingOneZone>::gasDensity_index) = scale * rhotot;
		state_cc(i, j, k, RadSystem<ISMCoolingOneZone>::gasInternalEnergy_index) = scale * Egas0;
		state_cc(i, j, k, RadSystem<ISMCoolingOneZone>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<ISMCoolingOneZone>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<ISMCoolingOneZone>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<ISMCoolingOneZone>::scalar0_index + nn) = scale * state.xn[nn] * spmasses[nn];
		}
	});
}

/// The gas of every cell along x, in cell order: number densities and temperature from the pure-H photochemistry EOS, as the burn sees them.
struct CellGas {
	std::vector<amrex::Real> n_HI;
	std::vector<amrex::Real> n_Hp;
	std::vector<amrex::Real> T;
};

static auto ReadCellGas(amrex::MultiFab &mf, amrex::Geometry &geom) -> CellGas
{
	using Hydro = HydroSystem<ISMCoolingOneZone>;
	using Rad = RadSystem<ISMCoolingOneZone>;
	auto [position, values] = fextract(mf, geom, 0, 0.0);
	const size_t n = position.size();
	CellGas gas;
	for (size_t i = 0; i < n; ++i) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(i == 0 || position[i] > position[i - 1], "ISMCoolingOneZone: cells are not ordered along x");
		const amrex::Real rho = values.at(Rad::gasDensity_index)[i];
		const amrex::Real Eint = values.at(Rad::gasInternalEnergy_index)[i];
		amrex::GpuArray<amrex::Real, NumSpec> scalars{};
		for (int nn = 0; nn < NumSpec; ++nn) {
			scalars[nn] = values.at(Hydro::scalar0_index + nn)[i];
		}
		gas.n_HI.push_back(scalars[Species::H] / spmasses[Species::H]);
		gas.n_Hp.push_back(scalars[Species::H_p] / spmasses[Species::H_p]);
		quokka::optional<amrex::GpuArray<amrex::Real, NumSpec>> massScalars = scalars;
		gas.T.push_back(quokka::EOS<ISMCoolingOneZone>::ComputeTgasFromEint(rho, Eint, massScalars));
	}
	return gas;
}

/// Mode "equilibrium": record the temperature of every cell after each step, to check that the run has settled.
template <> void QuokkaSimulation<ISMCoolingOneZone>::computeAfterTimestep()
{
	if (!userData_.equilibrium) {
		return;
	}
	userData_.time_history_.push_back(tNew_[0]);
	userData_.T_history_.push_back(ReadCellGas(state_new_cc_[0], geom[0]).T);
}

/// Mode "startup": construct the simulation, so that its startup checks run, set the initial conditions and take the steps of max_timesteps.
static auto RunStartupMode() -> int
{
	QuokkaSimulation<ISMCoolingOneZone> sim;
	if (ISM::GetGrackleNTTable() == nullptr) {
		std::cout << "ISMCoolingOneZone: the simulation did not register the Grackle table\n";
		return 1;
	}
	sim.setInitialConditions();
	sim.plotfileInterval_ = -1;
	sim.evolve();
	std::cout << "ISMCoolingOneZone startup: PASSED\n";
	return 0;
}

/// Mode "jacobian": builds burn_t states directly, so there is no simulation to register the Grackle table and the mode registers it itself.
static auto RunJacobianMode() -> int
{
	amrex::ParmParse const pp_cooling("ism_cooling");
	std::string table_path;
	pp_cooling.get("grackle_nT_table", table_path);

	init_extern_parameters();

	// G0_external is G0 in Habing units: the one the states carry in aux[0] must be the one the input asks for
	amrex::ParmParse const pp_top;
	Real G0_input = 0.0;
	pp_top.getWithParser("G0_external", G0_input);
	if (MakeState(gas_states[0]).aux[0] != G0_input) {
		std::cout << "ISMCoolingOneZone: aux[0] = " << MakeState(gas_states[0]).aux[0] << " but G0_external = " << G0_input << '\n';
		return 1;
	}

	const auto data = ISM::ReadGrackleNTTable(table_path);
	const auto attrs = ISM::ReadGrackleNTAttributes(table_path);
	ISM::RegisterGrackleNTTable({data->const_tables_host(), attrs.zmet, attrs.H_mass_fraction}, {data->const_tables(), attrs.zmet, attrs.H_mass_fraction});

	int failures = 0;
	for (const GasState &g : gas_states) {
		const int branch_failures = CheckBranch(g);
		const int jac_failures = CheckJacobian(g, true) + CheckJacobian(g, false);
		const int ism_failures = CheckIsmTerms(g);
		const int sum_failures = CheckEnergySum(g);
		std::cout << "ISMCoolingOneZone " << g.name << ": " << branch_failures << " branch, " << jac_failures << " Jacobian entries, " << ism_failures
			  << " ISM-term partials and " << sum_failures << " energy sums failed\n";
		failures += branch_failures + jac_failures + ism_failures + sum_failures;
	}
	std::cout << (failures == 0 ? "ISMCoolingOneZone: PASSED\n" : "ISMCoolingOneZone: FAILED\n");
	return (failures == 0) ? 0 : 1;
}

/// The reference of extern/cooling/ism_cooling_reference.py: one row per cell of (n_H, T_eq, x_H+, n_e^max, branch, number of roots).
static auto ReadEquilibriumReference(std::string const &path) -> std::vector<std::array<Real, 6>>
{
	std::ifstream file(path);
	if (!file.is_open()) {
		amrex::Abort("ISMCoolingOneZone: could not open " + path + "; run the test from the <quokka-root>/tests/ folder.");
	}
	std::vector<std::array<Real, 6>> rows;
	std::string line;
	std::getline(file, line); // header
	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string value;
		std::array<Real, 6> row{};
		for (Real &v : row) {
			std::getline(ss, value, ',');
			v = std::stod(value);
		}
		rows.push_back(row);
	}
	return rows;
}

/// Mode "equilibrium": a row of cells at fixed n_H (log-spaced from n_H_min to n_H_max), with G0 = 1, Z' = 1 and no photons, evolved from T = 1e3 K to
/// stop_time with hydro off. Their temperatures must match the thermal equilibrium of the network found by the Python reference to temperature_rtol,
/// must no longer change (the last tenth of the run changes none of them by more than drift_rtol), and the curve must show the two phases.
static auto RunEquilibriumMode() -> int
{
	constexpr Real temperature_rtol = 1.0e-3;
	constexpr Real drift_rtol = 1.0e-4;
	QuokkaSimulation<ISMCoolingOneZone> sim;
	if (ISM::GetGrackleNTTable() == nullptr) {
		std::cout << "ISMCoolingOneZone: the simulation did not register the Grackle table\n";
		return 1;
	}
	amrex::ParmParse const pp("ism_test");
	std::string reference_path;
	pp.get("reference_file", reference_path);
	const auto reference = ReadEquilibriumReference(reference_path);

	sim.setInitialConditions();
	const CellGas initial = ReadCellGas(sim.state_new_cc_[0], sim.geom[0]);
	const size_t n = initial.T.size();
	if (reference.size() != n) {
		std::cout << "ISMCoolingOneZone: " << n << " cells but " << reference.size() << " reference rows\n";
		return 1;
	}
	int failures = 0;
	for (size_t i = 0; i < n; ++i) {
		const Real n_H = initial.n_HI[i] + initial.n_Hp[i];
		if (!(std::abs(n_H - reference[i][0]) <= 1.0e-9 * reference[i][0]) ||
		    !(std::abs(initial.T[i] - sim.userData_.temperature) <= 1.0e-9 * sim.userData_.temperature)) {
			std::cout << "  cell " << i << ": initial n_H = " << n_H << " (reference " << reference[i][0] << "), T = " << initial.T[i] << '\n';
			++failures;
		}
	}

	sim.plotfileInterval_ = -1;
	sim.evolve();
	const CellGas final_gas = ReadCellGas(sim.state_new_cc_[0], sim.geom[0]);

	// how much each cell still changes over the last tenth of the run
	const auto &times = sim.userData_.time_history_;
	const Real t_end = times.back();
	size_t i_late = 0;
	while (times[i_late] < 0.9 * t_end) {
		++i_late;
	}

	std::cout << "ISMCoolingOneZone equilibrium: t = " << t_end << " s\n"
		  << "  n_H [cm^-3]   T_C++ [K]   T_Python [K]   rel. error   drift(last 10%)   branch C++/Python   x_H+ C++/Python\n";
	bool has_cold_phase = false;
	bool has_warm_phase = false;
	for (size_t i = 0; i < n; ++i) {
		const Real n_H = final_gas.n_HI[i] + final_gas.n_Hp[i];
		const Real T = final_gas.T[i];
		const Real T_ref = reference[i][1];
		const Real err = std::abs(T - T_ref) / T_ref;
		const Real drift = std::abs(T - sim.userData_.T_history_[i_late][i]) / T;
		burn_t state{};
		state.xn[H_idx] = final_gas.n_HI[i];
		state.xn[Hp_idx] = final_gas.n_Hp[i];
		state.T = T;
		const int branch = ElectronDensityOf(state).branch;
		std::cout << "  " << std::setw(11) << std::setprecision(5) << n_H << "  " << std::setw(10) << T << "  " << std::setw(12) << T_ref << "  "
			  << std::setw(10) << std::setprecision(3) << err << "  " << std::setw(14) << drift << "  " << std::setw(9) << branch << "/"
			  << static_cast<int>(reference[i][4]) << "  " << std::setw(12) << final_gas.n_Hp[i] / n_H << "/" << reference[i][2] << '\n';
		if (!(err <= temperature_rtol)) {
			++failures;
		}
		if (!(drift <= drift_rtol)) {
			std::cout << "    n_H = " << n_H << ": T still changes by " << drift << " over the last tenth of the run\n";
			++failures;
		}
		has_cold_phase = has_cold_phase || (n_H >= 10.0 && n_H <= 100.0 && T >= 30.0 && T <= 100.0);
		has_warm_phase = has_warm_phase || (n_H <= 1.0 && T >= 5000.0 && T <= 9000.0);
	}
	if (!has_cold_phase) {
		std::cout << "  no cell with 10 <= n_H <= 100 cm^-3 has 30 <= T <= 100 K (no CNM)\n";
		++failures;
	}
	if (!has_warm_phase) {
		std::cout << "  no cell with n_H <= 1 cm^-3 has 5000 <= T <= 9000 K (no WNM)\n";
		++failures;
	}
	std::cout << (failures == 0 ? "ISMCoolingOneZone equilibrium: PASSED\n" : "ISMCoolingOneZone equilibrium: FAILED\n");
	return (failures == 0) ? 0 : 1;
}

auto problem_main() -> int
{
	amrex::ParmParse const pp("ism_test");
	std::string mode;
	pp.get("mode", mode);
	if (mode == "jacobian") {
		return RunJacobianMode();
	}
	if (mode == "startup") {
		return RunStartupMode();
	}
	if (mode == "equilibrium") {
		return RunEquilibriumMode();
	}
	std::cout << "ISMCoolingOneZone: unknown ism_test.mode '" << mode << "'\n";
	return 1;
}
