//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
// \file testDTypeFront1D.cpp
// \brief Defines a 1D planar H II region test: a central ionizing+optical source drives a D-type ionization front into a uniform neutral slab. Momentum
// deposition from the optical radiation is also accounted for in accordance with KM09.

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include "radiation/radiation_dust_system.hpp"
#include "radiation/radiation_system.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct DTypeFront1D {};

constexpr double c_hat = C::c_light / 1000.0;
constexpr double Erad_floor_ = 1.0e-10 * 13.6 * C::ev2erg; // erg cm^-3
constexpr int group_ir = 0;
constexpr int group_optical = 1;
constexpr int group_ionizing = 2;

AMREX_GPU_MANAGED double kappa_ir = 0.0;      // NOLINT
AMREX_GPU_MANAGED double kappa_optical = 0.0; // NOLINT

template <> struct quokka::EOS_Traits<DTypeFront1D> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DTypeFront1D> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int numMassScalars = NumSpec;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr int nGroups = NumThermalBands + NumChemBands;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<DTypeFront1D> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr amrex::GpuArray<double, Physics_Traits<DTypeFront1D>::nGroups + 1> radBoundaries{1.0e-6, 0.413567, ChemBandsHeader().arr[0],
													  ChemBandsHeader().arr[1]};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr auto ChemBandsPowerLawIndex() { return ChemBandsPowerLawIndex_; }
	static constexpr auto ChemBands() { return ChemBandsHeader(); }
};

template <> struct ISM_Traits<DTypeFront1D> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = false;
	static constexpr bool thermal_band_photochemistry =
#ifdef THERMAL_DUST_PHOTOCHEMISTRY
	    true;
#else
	    false;
#endif
};

template <> struct SimulationData<DTypeFront1D> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real n_e_init{};
	amrex::Real n_HI_init{};
	amrex::Real n_HII_init{};
	amrex::Real flux_optical{};
	amrex::Real flux_ion{};
	amrex::Real flux_ir{};
	amrex::Real eps_ir{};
	amrex::Real eps_opt{};
	amrex::Real eps_ion{};
	amrex::Real T_ionized{};
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> xshell_vec_;
	amrex::Vector<amrex::Real> xspitzer_vec_;
	amrex::Vector<amrex::Real> xeff_vec_;
	amrex::Vector<amrex::Real> xode_vec_;
	amrex::Real l_ode_last_t_{};
	amrex::Real l_ode_last_l_{};
	amrex::Real l_ode_last_u_{};
};

namespace
{

// Ionization-fraction-weighted effective ionized length on the +x side of the source, as a distance from the
// source: x_eff = integral_{x_source}^{L} (1 - x_HI) dx, averaged over the transverse (y, z) columns.
auto compute_effective_length(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real x_source) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];
	const amrex::Real x_lo = prob_lo[0];

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		const amrex::Real x = x_lo + (static_cast<amrex::Real>(i) + 0.5_rt) * cell_length;
		if (x <= x_source) {
			return 0.0_rt;
		}
		const amrex::Real n_HI = state[box_no](i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + static_cast<int>(Species::H)) / spmasses[Species::H];
		const amrex::Real n_HII =
		    state[box_no](i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + static_cast<int>(Species::H_p)) / spmasses[Species::H_p];
		const amrex::Real denom = n_HI + n_HII;
		if (denom <= 0.0_rt) {
			return 0.0_rt;
		}
		const amrex::Real x_HI = n_HI / denom;
		return cell_length * (1.0_rt - x_HI);
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real total_ionized_length = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Sum(total_ionized_length, amrex::ParallelContext::CommunicatorSub());

	const amrex::Box &domain = state_mf.boxArray().minimalBox();
	const amrex::GpuArray<int, 3> len3d = domain.length3d();
	const amrex::Long n_transverse = static_cast<amrex::Long>(len3d[1]) * static_cast<amrex::Long>(len3d[2]);
	return total_ionized_length / static_cast<amrex::Real>(n_transverse);
}

// Position of the dense shocked shell on the +x side of the source, returned as a distance from the source.
auto compute_shell_position(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real x_source) -> amrex::Real
{
	const amrex::Real cell_length = dx[0];
	const amrex::Real x_lo = prob_lo[0];

	// Pass 1: the largest gas density outward of the source.
	amrex::Real rho_max = 0.0;
	{
		amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
		amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
		auto const state = state_mf.const_arrays();

		reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
			const amrex::Real x = x_lo + (static_cast<amrex::Real>(i) + 0.5_rt) * cell_length;
			if (x <= x_source) {
				return 0.0_rt;
			}
			return state[box_no](i, j, k, HydroSystem<DTypeFront1D>::density_index);
		});

		auto const &hv = reduce_data.value(reduce_op);
		rho_max = amrex::get<0>(hv);
		amrex::ParallelAllReduce::Max(rho_max, amrex::ParallelContext::CommunicatorSub());
	}

	// Pass 2: the innermost cell attaining that density.
	amrex::Real x_shell = 0.0;
	{
		amrex::ReduceOps<amrex::ReduceOpMin> reduce_op;
		amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
		auto const state = state_mf.const_arrays();
		const amrex::Real threshold = rho_max;
		const amrex::Real x_none = std::numeric_limits<amrex::Real>::max();

		reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
			const amrex::Real x = x_lo + (static_cast<amrex::Real>(i) + 0.5_rt) * cell_length;
			if (x <= x_source || state[box_no](i, j, k, HydroSystem<DTypeFront1D>::density_index) < threshold) {
				return x_none;
			}
			return x;
		});

		auto const &hv = reduce_data.value(reduce_op);
		x_shell = amrex::get<0>(hv);
		amrex::ParallelAllReduce::Min(x_shell, amrex::ParallelContext::CommunicatorSub());
	}

	return x_shell - x_source;
}

auto lambda_rec(double T) -> double
{
	if (T < 100.0) {
		return 0.0;
	}
	return 6.1e-10 * C::k_B * T * std::pow(T, -0.89);
}

auto get_cle_term(double T) -> double
{
	if (T < 1.0e2) {
		return 3.47e-29 * std::pow(T, 1.915);
	}
	if (T < std::pow(10.0, 2.8)) {
		return 2.34e-26 * std::pow(T, 0.500);
	}
	if (T < std::pow(10.0, 3.6)) {
		return 1.11e-24 * std::pow(T, -0.099);
	}
	if (T < 1.0e4) {
		return 1.08e-32 * std::pow(T, 2.127);
	}
	if (T < std::pow(10.0, 4.5)) {
		return 2.67e-30 * std::pow(T, 1.529);
	}
	if (T < 1.0e5) {
		return 1.74e-24 * std::pow(T, 0.237);
	}
	if (T < 1.0e6) {
		return 1.10e-21 * std::pow(T, -0.323);
	}
	return 7.49e-21 * std::pow(T, -0.462);
}

auto lambda_ff(double T) -> double { return 1.3 * 1.427e-27 * std::sqrt(T) + get_cle_term(T); }

auto lambda_KI(double T) -> double { return 2.0e-26 * (1.0e7 * std::exp(-118400.0 / (T + 1.0e3)) + 1.4e-2 * std::sqrt(T) * std::exp(-92.0 / T)); }

// Collisional ionization is omitted
auto net_energy_ionized(double T, double n_e, double eps_ion) -> double
{
	const double alpha_B = 2.6e-13 * std::pow(T / 1.0e4, -0.7);
	const double epsilon = std::max(eps_ion - 13.6 * C::ev2erg, 0.0);
	// alpha_B * n_e^2 = n_gamma
	const double photoheating = alpha_B * n_e * n_e * epsilon;
	const double recombination_cooling = n_e * n_e * lambda_rec(T);
	const double ff_cooling = n_e * n_e * lambda_ff(T);
	const double KI_heating = 0.0;
	const double KI_cooling = 0.0;
	return photoheating - recombination_cooling - ff_cooling + KI_heating - KI_cooling;
}

auto net_energy_neutral(double T, double n_HI) -> double
{
	const double KI_heating = n_HI * 2e-26;
	const double KI_cooling = n_HI * n_HI * lambda_KI(T);
	return KI_heating - KI_cooling;
}

// Temperature at which net_energy_neutral vanishes, by bisection.
auto compute_equilibrium_temperature_neutral(double n_HI) -> double
{
	double T_lo = 1;
	double T_hi = 1000;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(net_energy_neutral(T_lo, n_HI) > 0.0 && net_energy_neutral(T_hi, n_HI) < 0.0,
					 "compute_equilibrium_temperature_neutral: brackets do not straddle a root");
	int const max_iter = 10000;
	for (int iter = 0; iter < max_iter; ++iter) {
		const double T_mid = 0.5 * (T_lo + T_hi);
		if (net_energy_neutral(T_mid, n_HI) > 0.0) {
			T_lo = T_mid;
		} else {
			T_hi = T_mid;
		}
		if ((T_hi - T_lo) < 1e-2) {
			break;
		}
	}
	return 0.5 * (T_lo + T_hi);
}

// Temperature at which net_energy_ionized vanishes, by bisection.
auto compute_equilibrium_temperature_ionized(double n_e, double eps_ion) -> double
{
	double T_lo = 1000.0;
	double T_hi = 1.0e5;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(net_energy_ionized(T_lo, n_e, eps_ion) > 0.0 && net_energy_ionized(T_hi, n_e, eps_ion) < 0.0,
					 "compute_equilibrium_temperature_ionized: brackets do not straddle a root");
	int const max_iter = 10000;
	for (int iter = 0; iter < max_iter; ++iter) {
		const double T_mid = 0.5 * (T_lo + T_hi);
		if (net_energy_ionized(T_mid, n_e, eps_ion) > 0.0) {
			T_lo = T_mid;
		} else {
			T_hi = T_mid;
		}
		if ((T_hi - T_lo) < 1.0) {
			break;
		}
	}
	return 0.5 * (T_lo + T_hi);
}

auto recombination_coefficient(amrex::Real T_i) -> amrex::Real { return 2.6e-13 * std::pow(T_i / 1.0e4, -0.7); }

auto ionized_sound_speed(amrex::Real T_i) -> amrex::Real { return std::sqrt(C::k_B * T_i / (0.5_rt * C::m_p)); }

auto stromgren_column(amrex::Real flux_ion, amrex::Real n_0, amrex::Real T_i) -> amrex::Real { return flux_ion / (recombination_coefficient(T_i) * n_0 * n_0); }

// Planar (1D) analog of the Spitzer D-type expansion law, evaluated at time t. Gas pressure only.
auto spitzer_planar_position(amrex::Real t, amrex::Real flux_ion, amrex::Real n_0, amrex::Real T_i) -> amrex::Real
{
	const amrex::Real c_i = ionized_sound_speed(T_i);
	const amrex::Real x_St = stromgren_column(flux_ion, n_0, T_i);
	return x_St * std::pow(1.0_rt + (5.0_rt / 4.0_rt) * c_i * t / x_St, 4.0_rt / 5.0_rt);
}

// Numerically integrate the planar D-type front ODE including radiation pressure,
// d(l * ldot)/dt = sqrt(l_s / l) * c_s^2  +  (F_ion * eps_ion + F_opt * eps_opt) / (rho_0 * c),
// Integrated as the first-order system for y = (l, u) with u = l * ldot:
// dl/dt = u / l,     du/dt = sqrt(l_s / l) * c_s^2 + Xi.
// The natural start is the end of the R-type phase, l = l_s moving at c_s.
auto integrate_front(amrex::Real dt_target, amrex::Real l0, amrex::Real u0, amrex::Real l_s, amrex::Real c_s, amrex::Real Xi) -> amrex::GpuArray<amrex::Real, 2>
{
	if (dt_target <= 0.0_rt) {
		return {l0, u0};
	}

	const amrex::Real l_floor = 1.0e-10_rt * l_s; // guard against a division by zero in an RK stage
	auto rhs = [&](amrex::GpuArray<amrex::Real, 2> const &y) -> amrex::GpuArray<amrex::Real, 2> {
		const amrex::Real l = std::max(y[0], l_floor);
		return {y[1] / l, std::sqrt(l_s / l) * c_s * c_s + Xi};
	};

	int N = 256;
	const int max_iters = 10;
	const amrex::Real tol = 1.0e-6_rt * std::max(l_s, 1.0_rt);
	amrex::GpuArray<amrex::Real, 2> y_prev{l0, u0};

	for (int iter = 0; iter < max_iters; ++iter) {
		const amrex::Real dt = dt_target / static_cast<amrex::Real>(N);
		amrex::GpuArray<amrex::Real, 2> y{l0, u0};

		for (int step = 0; step < N; ++step) {
			const auto k1 = rhs(y);
			const auto k2 = rhs({y[0] + 0.5_rt * dt * k1[0], y[1] + 0.5_rt * dt * k1[1]});
			const auto k3 = rhs({y[0] + 0.5_rt * dt * k2[0], y[1] + 0.5_rt * dt * k2[1]});
			const auto k4 = rhs({y[0] + dt * k3[0], y[1] + dt * k3[1]});
			y[0] += (dt / 6.0_rt) * (k1[0] + 2.0_rt * k2[0] + 2.0_rt * k3[0] + k4[0]);
			y[1] += (dt / 6.0_rt) * (k1[1] + 2.0_rt * k2[1] + 2.0_rt * k3[1] + k4[1]);
			y[0] = std::max(y[0], 0.0_rt);
		}

		if (iter > 0 && std::abs(y[0] - y_prev[0]) < tol) {
			return y;
		}
		y_prev = y;
		N *= 2;
	}

	amrex::Abort("integrate_front failed to converge within max_iters for dt=" + std::to_string(dt_target));
	return y_prev; // unreachable
}

} // namespace

template <>
void RadSystem<DTypeFront1D>::AddRadSource(array_t &radEnergy, array_t &reducedFlux, const amrex::Box &indexRange,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	amrex::ParmParse const pp("photoionize");
	amrex::Real flux_optical = 1.0e11_rt;
	pp.query("flux_optical", flux_optical);
	amrex::Real flux_ion = 0.0_rt;
	pp.query("flux_ion", flux_ion);
	amrex::Real flux_ir = 0.0_rt;
	pp.query("flux_ir", flux_ir);
	int source_cells = 1;
	pp.query("source_cells", source_cells); // cells per side occupied by the source slab
	int beamed = 1;
	pp.query("beamed", beamed); // 1 = each wing injected beamed outward, 0 = isotropic

	const auto n_cells = static_cast<amrex::Real>(source_cells);
	const amrex::Real eps_ir = RadSystem<DTypeFront1D>::GetThermalBandQuanta(group_ir);
	const amrex::Real eps_opt = RadSystem<DTypeFront1D>::GetThermalBandQuanta(group_optical);
	const amrex::Real eps_ion = RadSystem<DTypeFront1D>::GetChemBandQuanta(0);
	const amrex::Real src_ir = flux_ir * eps_ir / (n_cells * dx[0]);
	const amrex::Real src_optical = flux_optical * eps_opt / (n_cells * dx[0]);
	const amrex::Real src_ionizing = flux_ion * eps_ion / (n_cells * dx[0]);

	const amrex::Real x_source = 0.5_rt * (prob_lo[0] + prob_hi[0]);
	const amrex::Real cell_length = dx[0];
	const amrex::Real x_lo = prob_lo[0];
	const amrex::Real half_width = n_cells * cell_length;
	const amrex::Real beam_factor = (beamed != 0) ? 1.0_rt : 0.0_rt;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const amrex::Real x = x_lo + (static_cast<amrex::Real>(i) + 0.5_rt) * cell_length;
		const bool in_source = std::abs(x - x_source) < half_width;
		// Outward is -x on the left of the source and +x on the right.
		const amrex::Real outward = (x > x_source) ? 1.0_rt : -1.0_rt;
		for (int g = 0; g < Physics_Traits<DTypeFront1D>::nGroups; ++g) {
			amrex::Real src = 0.0_rt;
			if (in_source) {
				if (g == group_ir) {
					src = src_ir;
				} else if (g == group_optical) {
					src = src_optical;
				} else if (g == group_ionizing) {
					src = src_ionizing;
				}
			}
			radEnergy(i, j, k, g) = src;
			reducedFlux(i, j, k, 3 * g + 0) = (src > 0.0_rt) ? outward * beam_factor : 0.0_rt;
			reducedFlux(i, j, k, 3 * g + 1) = 0.0_rt;
			reducedFlux(i, j, k, 3 * g + 2) = 0.0_rt;
		}
	});
}

template <> void QuokkaSimulation<DTypeFront1D>::preCalculateInitialConditions()
{
	// initialize microphysics routines
	init_extern_parameters();

	// parmparse species, temperature, and flux
	amrex::ParmParse const pp("photoionize");
	userData_.small_temp = 1e-2;
	userData_.small_dens = 1e-60;
	userData_.temperature = 1.0e2;
	userData_.n_e_init = 1.0e-10_rt;
	userData_.n_HI_init = 1.0e2_rt;
	userData_.n_HII_init = 1.0e-10_rt;
	userData_.flux_optical = 1.0e11_rt;
	userData_.flux_ion = 1.0e9_rt;
	userData_.flux_ir = 0.0_rt;
	pp.query("kappa_ir", kappa_ir);
	pp.query("kappa_optical", kappa_optical);
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("n_e_init", userData_.n_e_init);
	pp.query("n_HI_init", userData_.n_HI_init);
	pp.query("n_HII_init", userData_.n_HII_init);
	pp.query("flux_optical", userData_.flux_optical);
	pp.query("flux_ion", userData_.flux_ion);
	pp.query("flux_ir", userData_.flux_ir);

	userData_.eps_ir = RadSystem<DTypeFront1D>::GetThermalBandQuanta(group_ir);
	userData_.eps_opt = RadSystem<DTypeFront1D>::GetThermalBandQuanta(group_optical);
	userData_.eps_ion = RadSystem<DTypeFront1D>::GetChemBandQuanta(0);

	userData_.T_ionized = compute_equilibrium_temperature_ionized(userData_.n_HI_init, userData_.eps_ion);
	amrex::Print() << "Band mean photon energies: IR " << userData_.eps_ir / C::ev2erg << " eV, optical " << userData_.eps_opt / C::ev2erg
		       << " eV, ionizing " << userData_.eps_ion / C::ev2erg << " eV\n";
	amrex::Print() << "Photoionization-equilibrium temperature of the ionized gas: " << userData_.T_ionized << " K\n";

	{
		const amrex::Real l_s = stromgren_column(userData_.flux_ion, userData_.n_HI_init, userData_.T_ionized);
		const amrex::Real c_s = ionized_sound_speed(userData_.T_ionized);
		userData_.l_ode_last_t_ = 0.0_rt;
		userData_.l_ode_last_l_ = l_s;
		userData_.l_ode_last_u_ = l_s * c_s;
		amrex::Print() << "Stromgren column l_s = " << l_s << " cm, ionized sound speed c_s = " << c_s << " cm/s\n";
	}

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<DTypeFront1D>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							      const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	const amrex::GpuArray<double, nGroups_> kappa_g{kappa_ir, kappa_optical, 0.0};
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		exponents_and_values[1][i] = (i < nGroups_) ? kappa_g[i] : 0.0;
	}
	return exponents_and_values;
}

template <> void QuokkaSimulation<DTypeFront1D>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	burn_t state;
	std::array<Real, NumSpec> numdens = {-1.0};
	numdens[Species::e] = userData_.n_e_init;
	numdens[Species::H] = userData_.n_HI_init;
	numdens[Species::H_p] = userData_.n_HII_init;

	state.T = userData_.temperature;
	Real rhotot = 0.0_rt;
	for (int n = 0; n < NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * spmasses[n];
	}
	state.rho = rhotot;

	eos(eos_input_rt, state);
	const auto Egas0 = state.e * rhotot;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<DTypeFront1D>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<DTypeFront1D>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad_floor_;
			state_cc(i, j, k, RadSystem<DTypeFront1D>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFront1D>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFront1D>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
		}
		state_cc(i, j, k, RadSystem<DTypeFront1D>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFront1D>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<DTypeFront1D>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFront1D>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFront1D>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFront1D>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + nn) =
			    state.xn[nn] * spmasses[nn]; // scalar indices carry partial densities instead of number densities
		}
	});
}

template <> void QuokkaSimulation<DTypeFront1D>::computeAfterTimestep()
{
	const int lev = 0;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();
	const amrex::Real x_source = 0.5 * (prob_lo[0] + prob_hi[0]);
	const amrex::Real t = tNew_[lev];
	userData_.t_vec_.push_back(t);

	const amrex::Real x_shell = compute_shell_position(state_new_cc_[lev], dx, prob_lo, x_source);
	const amrex::Real x_spitzer = spitzer_planar_position(t, userData_.flux_ion, userData_.n_HI_init, userData_.T_ionized);
	const amrex::Real x_eff = compute_effective_length(state_new_cc_[lev], dx, prob_lo, x_source);

	amrex::Real x_ode = std::numeric_limits<amrex::Real>::quiet_NaN();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const amrex::Real n_0 = userData_.n_HI_init;
		const amrex::Real rho_0 = n_0 * spmasses[Species::H];
		const amrex::Real l_s = stromgren_column(userData_.flux_ion, n_0, userData_.T_ionized);
		const amrex::Real c_s = ionized_sound_speed(userData_.T_ionized);
		const amrex::Real T_i = userData_.T_ionized;
		// Energy per recombination: binding energy plus the kinetic energy carried off. The network emits this into the optical band (ydot(i_optical)
		// in actual_rhs.H), hence its grouping below.
		const amrex::Real eps_rec = 13.6 * C::ev2erg + lambda_rec(T_i) / recombination_coefficient(T_i);
		const amrex::Real Xi =
		    (userData_.flux_ion * userData_.eps_ion + userData_.flux_optical * userData_.eps_opt + userData_.flux_ion * eps_rec) / (rho_0 * C::c_light);

		amrex::Real dt_ode = t - userData_.l_ode_last_t_;
		if (dt_ode < 0.0_rt) {
			// time went backwards or was reset; restart the integration from the R-type endpoint
			userData_.l_ode_last_t_ = 0.0_rt;
			userData_.l_ode_last_l_ = l_s;
			userData_.l_ode_last_u_ = l_s * c_s;
			dt_ode = t;
		}
		const auto y = integrate_front(dt_ode, userData_.l_ode_last_l_, userData_.l_ode_last_u_, l_s, c_s, Xi);
		userData_.l_ode_last_t_ = t;
		userData_.l_ode_last_l_ = y[0];
		userData_.l_ode_last_u_ = y[1];
		x_ode = y[0];
	}
	amrex::ParallelDescriptor::Bcast(&x_ode, 1, amrex::ParallelDescriptor::IOProcessorNumber());

	userData_.xshell_vec_.push_back(x_shell);
	userData_.xspitzer_vec_.push_back(x_spitzer);
	userData_.xeff_vec_.push_back(x_eff);
	userData_.xode_vec_.push_back(x_ode);
}

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 0.3;

	// Problem initialization
	QuokkaSimulation<DTypeFront1D> sim;

	// initialize
	sim.setInitialConditions();
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.plotfileInterval_ = -1;

	sim.evolve();

	int status = 0;

	// The source sits at the middle of the domain and radiates both ways, so each front has Lx / 2 to travel.
	const double half_Lx = 0.5 * (sim.geom[0].ProbHiArray()[0] - sim.geom[0].ProbLoArray()[0]);

	// Check 1: gas temperature in the ionized cavity and in the undisturbed neutral gas.
	{
		const double ne_eq = sim.userData_.n_HI_init;
		const double T_ion_eq = compute_equilibrium_temperature_ionized(ne_eq, sim.userData_.eps_ion);
		const double n_HI_init = sim.userData_.n_HI_init;
		const double T_neu_eq = compute_equilibrium_temperature_neutral(n_HI_init);

		amrex::MultiFab const &state_mf = sim.state_new_cc_[0];

		// Collect temperatures per region: cavity (x_HII > 90%), neutral (x_HI > 99.99%).
		const double v_quiescent = 0.05 * ionized_sound_speed(sim.userData_.T_ionized);
		std::vector<double> cavity_temps;
		std::vector<double> neutral_temps;

		for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
			const amrex::Box &box = mfi.validbox();

			amrex::FArrayBox host_fab(box, state_mf.nComp(), amrex::The_Pinned_Arena());
			static_cast<void>(state_mf[mfi].template copyToMem<amrex::RunOn::Device>(box, 0, state_mf.nComp(), host_fab.dataPtr()));
			amrex::Gpu::synchronize();

			const auto state = host_fab.const_array();

			amrex::LoopOnCpu(box, [&](int i, int j, int k) noexcept {
				const amrex::Real rho = state(i, j, k, HydroSystem<DTypeFront1D>::density_index);
				const amrex::Real Eint = state(i, j, k, RadSystem<DTypeFront1D>::gasInternalEnergy_index);
				const amrex::Real n_HI_cell =
				    state(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + static_cast<int>(Species::H)) / spmasses[Species::H];
				const amrex::Real n_HII_cell =
				    state(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + static_cast<int>(Species::H_p)) / spmasses[Species::H_p];
				const amrex::Real denom = n_HI_cell + n_HII_cell;
				if (denom <= 0.0_rt) {
					return;
				}
				const amrex::Real x_HII = n_HII_cell / denom;
				const amrex::Real x_HI = n_HI_cell / denom;

				burn_t bstate;
				for (int nn = 0; nn < NumSpec; ++nn) {
					bstate.xn[nn] = state(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + nn) / spmasses[nn];
				}
				bstate.rho = rho;
				bstate.e = Eint / rho;
				bstate.T = 1.0e4; // initial guess
				eos(eos_input_re, bstate);
				const double T_cell = bstate.T;

				if (x_HII > 0.90_rt) {
					cavity_temps.push_back(T_cell);
				}
				// Quiescent neutral gas only: the gas the front has set in motion is adiabatically cooled
				// and out of thermal equilibrium, and |vx| is what separates it from the gas still sitting
				// on the KI balance.
				const amrex::Real vx = state(i, j, k, HydroSystem<DTypeFront1D>::x1Momentum_index) / rho;
				if (x_HI > 0.9999_rt && std::abs(vx) < v_quiescent) {
					neutral_temps.push_back(T_cell);
				}
			});
		}

		auto compute_median_and_check = [&](std::vector<double> &local_temps, double T_analytical, const char *region_name, const char *quantity_name,
						    const char *unit) {
			const int num_local = static_cast<int>(local_temps.size());
			auto num_local_vec = amrex::ParallelDescriptor::Gather(num_local, amrex::ParallelDescriptor::IOProcessorNumber());

			amrex::Vector<int> recvcnt;
			amrex::Vector<int> disp;
			std::vector<double> all_temps;
			if (amrex::ParallelDescriptor::IOProcessor()) {
				recvcnt.resize(num_local_vec.size());
				disp.resize(num_local_vec.size());
				int ntot = 0;
				disp[0] = 0;
				for (int r = 0, n = static_cast<int>(num_local_vec.size()); r < n; ++r) {
					recvcnt[r] = num_local_vec[r];
					ntot += num_local_vec[r];
					if (r + 1 < n) {
						disp[r + 1] = disp[r] + num_local_vec[r];
					}
				}
				all_temps.resize(ntot);
			} else {
				recvcnt.resize(1);
				disp.resize(1);
				all_temps.resize(1);
			}

			static double static_val = 0.0;
			const double *send_ptr = local_temps.empty() ? &static_val : local_temps.data();
			double *recv_ptr = all_temps.empty() ? &static_val : all_temps.data();
			amrex::ParallelDescriptor::Gatherv(send_ptr, num_local, recv_ptr, recvcnt, disp, amrex::ParallelDescriptor::IOProcessorNumber());

			if (amrex::ParallelDescriptor::IOProcessor()) {
				const int ntot = static_cast<int>(all_temps.size());
				if (ntot == 0) {
					amrex::Print() << "Test FAILED: no " << region_name << " cells found.\n";
					status = 1;
					return;
				}
				std::sort(all_temps.begin(), all_temps.end());
				const double T_median = (ntot % 2 == 0) ? 0.5 * (all_temps[ntot / 2 - 1] + all_temps[ntot / 2]) : all_temps[ntot / 2];
				const double rel_err = std::abs(T_median - T_analytical) / T_analytical;
				if (rel_err > 0.05) {
					amrex::Print() << "Test FAILED: " << region_name << " median " << quantity_name << " " << T_median << unit
						       << " differs from analytical equilibrium " << T_analytical << unit << " by " << 100.0 * rel_err
						       << "% (tolerance: 5%).\n";
					status = 1;
				} else {
					amrex::Print() << "Test passed: " << region_name << " median " << quantity_name << " " << T_median << unit
						       << " is within 5% of analytical equilibrium " << T_analytical << unit << " (" << ntot << " cells).\n";
				}
			}
		};

		compute_median_and_check(cavity_temps, T_ion_eq, "cavity", "temperature", " K");
		compute_median_and_check(neutral_temps, T_neu_eq, "neutral", "temperature", " K");
	}

	// Check 2: the D-type front radius against the numerically integrated thin-shell solution that carries
	// both the ionized-gas pressure and the radiation pressure. The offset between the two is a systematic,
	// largely resolution-independent fraction of the front radius, so the tolerance is relative rather than a
	// fixed cell count. Either metric matching is sufficient: the shell position and the effective ionized
	// length measure the same front from different sides of its finite thickness.
	{
		const double x_front = sim.userData_.xeff_vec_.back();
		const double x_shell = sim.userData_.xshell_vec_.back();
		const double x_ode = sim.userData_.xode_vec_.back();
		const double rel_diff = (x_front - x_ode) / x_ode;
		const double shell_rel_diff = (x_shell - x_ode) / x_ode;

		const double tol_rel = 0.05;

		amrex::Print() << "Integrated solution (gas + radiation pressure):   " << x_ode << " cm\n";

		if (!(x_ode > 0.0)) {
			amrex::Print() << "Test FAILED: the integrated front solution is not positive; check photoionize.flux_ion.\n";
			status = 1;
		} else if (x_ode >= half_Lx) {
			amrex::Print() << "Test FAILED: the integrated front has left the domain; reduce stop_time.\n";
			status = 1;
		} else {
			const bool eff_ok = std::abs(rel_diff) <= tol_rel;
			const bool shell_ok = std::abs(shell_rel_diff) <= tol_rel;

			if (eff_ok) {
				amrex::Print() << "Test passed: D-type I front matches the integrated radiation + gas pressure solution within "
					       << 100.0 * tol_rel << "% (" << 100.0 * rel_diff << "%).\n";
			} else {
				amrex::Print() << "D-type I front differs from the integrated radiation + gas pressure solution by more than "
					       << 100.0 * tol_rel << "% (" << 100.0 * rel_diff << "%).\n";
			}

			amrex::Print() << "Numerical max-density shell position: " << x_shell << " cm\n";

			if (shell_ok) {
				amrex::Print() << "Test passed: max-density shell matches the integrated radiation + gas pressure solution within "
					       << 100.0 * tol_rel << "% (" << 100.0 * shell_rel_diff << "%).\n";
			} else {
				amrex::Print() << "max-density shell differs from the integrated radiation + gas pressure solution by more than "
					       << 100.0 * tol_rel << "% (" << 100.0 * shell_rel_diff << "%).\n";
			}

			if (!eff_ok && !shell_ok) {
				amrex::Print() << "Test FAILED: neither the effective ionized length nor the max-density shell matches the integrated "
						  "solution within tolerance.\n";
				status = 1;
			}
		}
	}

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr amrex::Real seconds_per_Myr = 3.15576e13;
		constexpr amrex::Real cm_per_pc = 3.085677581491367e18;

		const auto n = static_cast<int>(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> t_Myr(n);
		std::vector<amrex::Real> x_shell_pc(n);
		std::vector<amrex::Real> x_spitzer_pc(n);
		std::vector<amrex::Real> x_eff_pc(n);
		std::vector<amrex::Real> x_ode_pc(n);
		for (int i = 0; i < n; ++i) {
			t_Myr[i] = sim.userData_.t_vec_[i] / seconds_per_Myr;
			x_shell_pc[i] = sim.userData_.xshell_vec_[i] / cm_per_pc;
			x_spitzer_pc[i] = sim.userData_.xspitzer_vec_[i] / cm_per_pc;
			x_eff_pc[i] = sim.userData_.xeff_vec_[i] / cm_per_pc;
			x_ode_pc[i] = sim.userData_.xode_vec_[i] / cm_per_pc;
		}
		matplotlibcpp::clf();
		std::map<std::string, std::string> shell_args;
		shell_args["label"] = "max-density shell";
		shell_args["color"] = "C0";
		std::map<std::string, std::string> eff_args;
		eff_args["label"] = "effective ionized length";
		eff_args["color"] = "C1";
		std::map<std::string, std::string> spitzer_args;
		spitzer_args["label"] = "analytic (planar D-type, 4/5 law)";
		spitzer_args["color"] = "k";
		spitzer_args["linestyle"] = "--";
		std::map<std::string, std::string> ode_args;
		ode_args["label"] = "ODE (gas + radiation pressure)";
		ode_args["color"] = "k";
		ode_args["linestyle"] = ":";
		matplotlibcpp::plot(t_Myr, x_shell_pc, shell_args);
		matplotlibcpp::plot(t_Myr, x_eff_pc, eff_args);
		matplotlibcpp::plot(t_Myr, x_spitzer_pc, spitzer_args);
		matplotlibcpp::plot(t_Myr, x_ode_pc, ode_args);
		matplotlibcpp::xlabel("time (Myr)");
		matplotlibcpp::ylabel("front position from source (pc)");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dtype_front_1d_shell.pdf");
	}
#endif

	amrex::Print() << "Finished." << '\n';
	return status;
}
