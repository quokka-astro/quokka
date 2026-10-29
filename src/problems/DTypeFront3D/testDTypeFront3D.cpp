//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
// \file testDTypeFront3D.cpp
// \brief Defines a 3D spherical H II region test: a central ionizing+optical source drives a D-type ionization front into a uniform neutral medium with dust.
// Momentum deposition from the optical radiation is also accounted for in accordance with KM09.

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "microphysics/burn_state.hpp"
#include "microphysics/microphysics_traits.hpp"
#include "networks/thermophotochemistry_handwritten/network.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_dust_system.hpp"
#include "radiation/radiation_system.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "actual_eos_data.H"
#include "eos.H"
#include "extern_parameters.H"

struct DTypeFront3D {};

constexpr double c_hat = C::c_light / 1000.0;
constexpr double T_gas_limit = 5.0e4; // K; no cell may get hotter than this at any timestep
// Check 2 tolerance on the effective ionized radius and the max-density shell radius, in cells of the finest level.
constexpr double front_radius_tol_cells = 4.0;
constexpr double Erad_floor_ = 1.0e-10 * 13.6 * C::ev2erg; // erg cm^-3
constexpr int group_ir = 0;
constexpr int group_optical = 1;
constexpr int group_ionizing = 2;

AMREX_GPU_MANAGED double kappa_ir = 0.0;      // NOLINT
AMREX_GPU_MANAGED double kappa_optical = 0.0; // NOLINT

template <> struct quokka::EOS_Traits<DTypeFront3D> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Microphysics_Traits<DTypeFront3D> : DefaultMicrophysicsTraits {
	using Network = quokka::networks::Thermophotochemistry;
	using Integrator = quokka::microphysics::Rosenbrock<quokka::microphysics::ROS2S>;
	static constexpr bool photochemistry = true;		    // built with PHOTOCHEMISTRY
	static constexpr bool dust_chemical_band_absorption = true; // built with DUST_CHEMICAL_BAND_ABSORPTION
	static constexpr bool integrate_rad_flux = false;	    // built with SKIP_PHOTOCHEMFLUX
};

using Network = Microphysics_Traits<DTypeFront3D>::Network;

template <> struct Physics_Traits<DTypeFront3D> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int numMassScalars = Network::NumSpec;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr int nGroups = Network::NumThermalBands + Network::NumChemBands;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<DTypeFront3D> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr amrex::GpuArray<double, Physics_Traits<DTypeFront3D>::nGroups + 1> radBoundaries{1.0e-6, 0.413567, Network::ChemBands().arr[0],
													  Network::ChemBands().arr[1]};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr auto ChemBandsPowerLawIndex() { return Network::ChemBandsPowerLawIndex; }
	static constexpr auto ChemBands() { return Network::ChemBands(); }
};

template <> struct ISM_Traits<DTypeFront3D> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = false;
	static constexpr bool thermal_band_photochemistry = true;
};

template <> struct SimulationData<DTypeFront3D> {
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
	amrex::Vector<amrex::Real> reff_vec_;
	amrex::Vector<amrex::Real> rshell_vec_;
	amrex::Vector<amrex::Real> rspitzer_vec_;
	amrex::Vector<amrex::Real> rode_vec_;
	// Running state (R, v = dR/dt) of the front ODE, advanced one timestep at a time in computeAfterTimestep.
	amrex::Real r_ode_last_t_{};
	amrex::Real r_ode_last_R_{};
	amrex::Real r_ode_last_v_{};
	// Largest gas speed and temperature over all cells and timesteps, and the time each first exceeded its limit (-1 if never).
	amrex::Real v_max_run_{};
	amrex::Real T_max_run_{};
	amrex::Real t_first_v_violation_ = -1.0;
	amrex::Real t_first_T_violation_ = -1.0;
	std::ofstream output_file_;
};

namespace
{

auto make_level_mask(amrex::Vector<amrex::MultiFab> const &state_cc, amrex::Vector<amrex::Geometry> const &geom, amrex::Vector<amrex::IntVect> const &ref_ratio,
		     int lev, int finest_level) -> amrex::iMultiFab
{
	if (lev == finest_level) {
		amrex::iMultiFab mask(state_cc[lev].boxArray(), state_cc[lev].DistributionMap(), 1, 0);
		mask.setVal(1);
		return mask;
	}
	return amrex::makeFineMask(state_cc[lev], state_cc[lev + 1], amrex::IntVect(0), ref_ratio[lev], geom[lev].periodicity(), 1, 0);
}

AMREX_GPU_HOST_DEVICE auto wendland_c2(amrex::Real r) -> amrex::Real
{
	if (r > 1.0) {
		return 0.0;
	}
	return (21. / (2. * M_PI)) * std::pow((1.0 - r), 4) * (4.0 * r + 1.0);
}

// Neutral fraction x_HI = n_HI / (n_HI + n_HII) of a cell, or 0 if the cell holds no hydrogen.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto neutral_fraction(amrex::Array4<const amrex::Real> const &state, int i, int j, int k) -> amrex::Real
{
	const amrex::Real n_HI =
	    state(i, j, k, HydroSystem<DTypeFront3D>::scalar0_index + static_cast<int>(Network::Species::H)) / spmasses[Network::Species::H];
	const amrex::Real n_HII =
	    state(i, j, k, HydroSystem<DTypeFront3D>::scalar0_index + static_cast<int>(Network::Species::H_p)) / spmasses[Network::Species::H_p];
	const amrex::Real denom = n_HI + n_HII;
	return (denom > 0.0_rt) ? (n_HI / denom) : 0.0_rt;
}

// Effective ionized radius: the radius of the sphere whose volume equals the total ionized volume
// V_ion = sum_cells (1 - x_HI) dV.
auto compute_effective_radius(QuokkaSimulation<DTypeFront3D> &sim) -> amrex::Real
{
	const amrex::Real total_ionized_volume = sim.computeVolumeIntegral(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state,
				 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const & /*state_fc*/) noexcept -> amrex::Real {
		    const amrex::Real n_HI =
			state(i, j, k, HydroSystem<DTypeFront3D>::scalar0_index + static_cast<int>(Network::Species::H)) / spmasses[Network::Species::H];
		    const amrex::Real n_HII =
			state(i, j, k, HydroSystem<DTypeFront3D>::scalar0_index + static_cast<int>(Network::Species::H_p)) / spmasses[Network::Species::H_p];
		    const amrex::Real denom = n_HI + n_HII;
		    if (denom <= 0.0_rt) {
			    return 0.0_rt;
		    }
		    return 1.0_rt - n_HI / denom;
	    });
	return std::cbrt((3.0_rt * total_ionized_volume) / (4.0_rt * M_PI));
}

// Radius of the dense shocked shell.
auto compute_shell_radius(amrex::Vector<amrex::MultiFab> const &state_cc, amrex::Vector<amrex::Geometry> const &geom,
			  amrex::Vector<amrex::IntVect> const &ref_ratio, int finest_level, int n_bins) -> amrex::Real
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[0].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[0].ProbHiArray();
	const amrex::Real x_c = 0.5_rt * (prob_lo[0] + prob_hi[0]);
	const amrex::Real y_c = 0.5_rt * (prob_lo[1] + prob_hi[1]);
	const amrex::Real z_c = 0.5_rt * (prob_lo[2] + prob_hi[2]);
	// Bin only out to the largest radius fully enclosed by the box; beyond it the bins sample only the corners.
	const amrex::Real r_max = std::min({0.5_rt * (prob_hi[0] - prob_lo[0]), 0.5_rt * (prob_hi[1] - prob_lo[1]), 0.5_rt * (prob_hi[2] - prob_lo[2])});
	if (!(r_max > 0.0_rt) || n_bins <= 0) {
		return -1.0_rt;
	}
	const amrex::Real inv_bin_width = static_cast<amrex::Real>(n_bins) / r_max;

	amrex::Gpu::DeviceVector<amrex::Real> d_mass(n_bins, 0.0_rt);
	amrex::Gpu::DeviceVector<amrex::Real> d_volume(n_bins, 0.0_rt);
	auto *mass_ptr = d_mass.data();
	auto *volume_ptr = d_volume.data();

	for (int lev = 0; lev <= finest_level; ++lev) {
		amrex::MultiFab const &state_mf = state_cc[lev];
		const amrex::iMultiFab mask = make_level_mask(state_cc, geom, ref_ratio, lev, finest_level);
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lev_dx = geom[lev].CellSizeArray();
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lev_lo = geom[lev].ProbLoArray();
		const amrex::Real cell_volume = AMREX_D_TERM(lev_dx[0], *lev_dx[1], *lev_dx[2]);

		for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.validbox();
			auto const &state = state_mf.const_array(mfi);
			auto const &mask_arr = mask.const_array(mfi);

			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				if (mask_arr(i, j, k) == 0) {
					return;
				}
				const amrex::Real x = lev_lo[0] + (static_cast<amrex::Real>(i) + 0.5_rt) * lev_dx[0] - x_c;
				const amrex::Real y = lev_lo[1] + (static_cast<amrex::Real>(j) + 0.5_rt) * lev_dx[1] - y_c;
				const amrex::Real z = lev_lo[2] + (static_cast<amrex::Real>(k) + 0.5_rt) * lev_dx[2] - z_c;
				const amrex::Real r = std::sqrt(x * x + y * y + z * z);
				if (r >= r_max) {
					return;
				}
				int ibin = static_cast<int>(r * inv_bin_width);
				ibin = amrex::max(0, amrex::min(ibin, n_bins - 1));
				const amrex::Real rho = state(i, j, k, HydroSystem<DTypeFront3D>::density_index);
				amrex::Gpu::Atomic::AddNoRet(&mass_ptr[ibin], rho * cell_volume);
				amrex::Gpu::Atomic::AddNoRet(&volume_ptr[ibin], cell_volume);
			});
		}
	}

	amrex::Gpu::streamSynchronize();

	amrex::Gpu::HostVector<amrex::Real> h_mass(n_bins);
	amrex::Gpu::HostVector<amrex::Real> h_volume(n_bins);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_mass.begin(), d_mass.end(), h_mass.begin());
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_volume.begin(), d_volume.end(), h_volume.begin());

	amrex::ParallelAllReduce::Sum(h_mass.data(), n_bins, amrex::ParallelContext::CommunicatorSub());
	amrex::ParallelAllReduce::Sum(h_volume.data(), n_bins, amrex::ParallelContext::CommunicatorSub());

	amrex::Real rho_peak = -1.0_rt;
	int peak_bin = -1;
	for (int b = 0; b < n_bins; ++b) {
		if (h_volume[b] <= 0.0_rt) {
			continue;
		}
		const amrex::Real rho_bin = h_mass[b] / h_volume[b];
		if (rho_bin > rho_peak) {
			rho_peak = rho_bin;
			peak_bin = b;
		}
	}

	if (peak_bin < 0) {
		return -1.0_rt;
	}
	return (static_cast<amrex::Real>(peak_bin) + 0.5_rt) / inv_bin_width;
}

auto compute_group_energy(QuokkaSimulation<DTypeFront3D> &sim, int g) -> amrex::Real
{
	// only pass levels that currently exist; state_new_cc_ is sized max_level+1 and higher levels may be empty
	const int nlev = sim.finestLevel() + 1;
	amrex::Vector<amrex::MultiFab const *> mfs(nlev);
	for (int lev = 0; lev < nlev; ++lev) {
		mfs[lev] = &sim.getNewMF_cc()[lev];
	}
	return amrex::volumeWeightedSum(mfs, RadSystem<DTypeFront3D>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g, sim.Geom(), sim.refRatio());
}

// Largest gas speed and temperature over the valid cells of all levels. A non-finite speed or temperature counts as infinite, so it fails the checks.
auto compute_max_speed_and_temperature(QuokkaSimulation<DTypeFront3D> &sim) -> std::array<amrex::Real, 2>
{
	amrex::Real v_max = 0.0_rt;
	amrex::Real T_max = 0.0_rt;
	for (int lev = 0; lev <= sim.finestLevel(); ++lev) {
		amrex::MultiFab const &state_mf = sim.getNewMF_cc()[lev];
		auto const &state = state_mf.const_arrays();
		const auto level_max = amrex::ParReduce(
		    amrex::TypeList<amrex::ReduceOpMax, amrex::ReduceOpMax>{}, amrex::TypeList<amrex::Real, amrex::Real>{}, state_mf, amrex::IntVect(0),
		    [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
			    const amrex::Real rho = state[bx](i, j, k, HydroSystem<DTypeFront3D>::density_index);
			    const amrex::Real px = state[bx](i, j, k, HydroSystem<DTypeFront3D>::x1Momentum_index);
			    const amrex::Real py = state[bx](i, j, k, HydroSystem<DTypeFront3D>::x2Momentum_index);
			    const amrex::Real pz = state[bx](i, j, k, HydroSystem<DTypeFront3D>::x3Momentum_index);
			    const amrex::Real Eint = state[bx](i, j, k, RadSystem<DTypeFront3D>::gasInternalEnergy_index);
			    const auto massScalars = RadSystem<DTypeFront3D>::ComputeMassScalars(state[bx], i, j, k);
			    const amrex::Real v = std::sqrt(px * px + py * py + pz * pz) / rho;
			    const amrex::Real T = quokka::EOS<DTypeFront3D>::ComputeTgasFromEint(rho, Eint, massScalars);
			    constexpr amrex::Real inf = std::numeric_limits<amrex::Real>::infinity();
			    return {std::isfinite(v) ? v : inf, std::isfinite(T) ? T : inf};
		    });
		v_max = std::max(v_max, amrex::get<0>(level_max));
		T_max = std::max(T_max, amrex::get<1>(level_max));
	}
	amrex::ParallelDescriptor::ReduceRealMax(v_max);
	amrex::ParallelDescriptor::ReduceRealMax(T_max);
	return {v_max, T_max};
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

auto net_energy_ionized(double T, double n_e, double eps_ion) -> double
{
	const double alpha_B = 2.6e-13 * std::pow(T / 1.0e4, -0.7);
	const double epsilon = std::max(eps_ion - 13.6 * C::ev2erg, 0.0);
	const double photoheating = alpha_B * n_e * n_e * epsilon;
	const double recombination_cooling = n_e * n_e * lambda_rec(T);
	const double ff_cooling = n_e * n_e * lambda_ff(T);
	return photoheating - recombination_cooling - ff_cooling;
}

auto net_energy_neutral(double T, double n_HI) -> double
{
	const double KI_heating = n_HI * 2e-26;
	const double KI_cooling = n_HI * n_HI * lambda_KI(T);
	return KI_heating - KI_cooling;
}

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

auto stromgren_radius(amrex::Real flux_ion, amrex::Real n_0, amrex::Real T_i) -> amrex::Real
{
	return std::cbrt((3.0_rt * flux_ion) / (4.0_rt * M_PI * recombination_coefficient(T_i) * n_0 * n_0));
}

auto ionizing_dust_kappa() -> amrex::Real
{
	if constexpr (RadSystem<DTypeFront3D>::dust_chemical_band_absorption_) {
		return Network::DustKappa();
	} else {
		return 0.0_rt;
	}
}

// Spherical dust attenuation factor h(tau) = 3 int_0^1 x^2 e^(tau x) dx = 3 [e^tau (tau^2 - 2 tau + 2) - 2] / tau^3, the ratio of
// the ionizing photon rate needed to keep a dusty sphere of optical radius tau ionized to that needed without dust. h(0) = 1.
// Below tau = 1 the closed form cancels catastrophically, so the series 3 sum_n tau^n / (n! (n + 3)) is used instead.
auto spherical_dust_attenuation_factor(amrex::Real tau) -> amrex::Real
{
	if (tau <= 0.0_rt) {
		return 1.0_rt;
	}
	if (tau < 1.0_rt) {
		amrex::Real sum = 0.0_rt;
		amrex::Real tau_n_over_n_factorial = 1.0_rt;
		for (int n = 0; n < 30; ++n) {
			const amrex::Real term = tau_n_over_n_factorial / static_cast<amrex::Real>(n + 3);
			sum += term;
			if (term < 1.0e-17_rt * sum) {
				break;
			}
			tau_n_over_n_factorial *= tau / static_cast<amrex::Real>(n + 1);
		}
		return 3.0_rt * sum;
	}
	return 3.0_rt * (std::exp(tau) * (tau * tau - 2.0_rt * tau + 2.0_rt) - 2.0_rt) / (tau * tau * tau);
}

// Ionized-to-ambient density ratio s from Q_ion = (4 pi / 3) alpha n_i^2 R^3 h(tau) with tau = kappa rho_0 s R,
// i.e. (R_s / R)^3 = s^2 h(kappa rho_0 s R). The left side is fixed and the right increases with s, so bisect.
auto ionized_density_ratio(amrex::Real R, amrex::Real R_s, amrex::Real kappa_rho_0) -> amrex::Real
{
	const amrex::Real target = std::pow(R_s / R, 3);
	amrex::Real s_lo = 0.0_rt;
	amrex::Real s_hi = std::sqrt(target); // the dust-free value, since h >= 1
	for (int iter = 0; iter < 100; ++iter) {
		const amrex::Real s = 0.5_rt * (s_lo + s_hi);
		if (s * s * spherical_dust_attenuation_factor(kappa_rho_0 * s * R) > target) {
			s_hi = s;
		} else {
			s_lo = s;
		}
	}
	return 0.5_rt * (s_lo + s_hi);
}

// End of the R-type phase (s = 1): R_s^3 = R_d^3 h(kappa rho_0 R_d). Equal to R_s without dust.
auto dusty_stromgren_radius(amrex::Real R_s, amrex::Real kappa_rho_0) -> amrex::Real
{
	amrex::Real R_lo = 0.0_rt;
	amrex::Real R_hi = R_s;
	for (int iter = 0; iter < 100; ++iter) {
		const amrex::Real R = 0.5_rt * (R_lo + R_hi);
		if (R * R * R * spherical_dust_attenuation_factor(kappa_rho_0 * R) > R_s * R_s * R_s) {
			R_hi = R;
		} else {
			R_lo = R;
		}
	}
	return 0.5_rt * (R_lo + R_hi);
}

// Classic spherical Spitzer D-type expansion law. Gas pressure only; reference curve.
auto spitzer_radius(amrex::Real t, amrex::Real flux_ion, amrex::Real n_0, amrex::Real T_i) -> amrex::Real
{
	const amrex::Real c_i = ionized_sound_speed(T_i);
	const amrex::Real r_s = stromgren_radius(flux_ion, n_0, T_i);
	const amrex::Real t_s = r_s / c_i;
	return r_s * std::pow(1.0_rt + 7.0_rt * t / (4.0_rt * t_s), 4.0_rt / 7.0_rt);
}

// Numerically integrate the spherical thin-shell D-type front equation including radiation pressure and dust absorption
// of ionizing photons,
//
//   d/dt [ M(R) Rdot ] = 4 pi R^2 P_i  +  L_abs(R) / c,
//
// with M(R) = (4 pi / 3) rho_0 R^3 the swept-up mass and P_i = s rho_0 c_s^2 the ionized-gas pressure. The density ratio
// s = n_i / n_0 follows from ionization balance with dust (ionized_density_ratio); without dust s = (R_s/R)^{3/2}.
// As the first-order system for y = (R, v):
//
//   dR/dt = v,
//   dv/dt = [ 4 pi R^2 P_i  +  L_abs(R) / c  -  M'(R) v^2 ] / M(R).
//
// L_abs = L_ion + (L_optical + L_rec f_rec) (1 - exp(-kappa_opt Sigma)), with Sigma = M / (4 pi R^2) the shell column and
// f_rec = s^2 (R / R_s)^3 = 1 / h(tau) the fraction of ionizing photons that recombine rather than hit dust. All ionizing
// photons are absorbed inside R, by gas or dust, so L_ion enters in full; the dust re-emits what it absorbs in the IR,
// which the shell barely absorbs (kappa_ir is small), so it is left out.
//
// The integration starts at the dusty Stromgren radius R_d moving at c_s, the end of the R-type phase.
auto integrate_front(amrex::Real dt_target, amrex::Real R0, amrex::Real v0, amrex::Real R_s, amrex::Real rho_0, amrex::Real c_s, amrex::Real L_ion,
		     amrex::Real L_optical, amrex::Real L_rec, amrex::Real kappa_opt, amrex::Real kappa_rho_0) -> amrex::GpuArray<amrex::Real, 2>
{
	if (dt_target <= 0.0_rt) {
		return {R0, v0};
	}

	const amrex::Real R_d = dusty_stromgren_radius(R_s, kappa_rho_0);
	auto rhs = [&](amrex::GpuArray<amrex::Real, 2> const &y) -> amrex::GpuArray<amrex::Real, 2> {
		const amrex::Real R = std::max(y[0], R_d);
		const amrex::Real v = y[1];
		const amrex::Real area = 4.0_rt * M_PI * R * R;
		const amrex::Real mass = (4.0_rt / 3.0_rt) * M_PI * R * R * R * rho_0;
		const amrex::Real dmass_dR = area * rho_0;
		const amrex::Real s = ionized_density_ratio(R, R_s, kappa_rho_0);
		const amrex::Real P_i = s * rho_0 * c_s * c_s;
		const amrex::Real f_rec = s * s * std::pow(R / R_s, 3);
		const amrex::Real Sigma = mass / area;
		const amrex::Real f_absorbed = -std::expm1(-kappa_opt * Sigma); // = 1 - exp(-tau), accurate for small tau
		const amrex::Real L_abs = L_ion + (L_optical + L_rec * f_rec) * f_absorbed;
		const amrex::Real force = area * P_i + L_abs / C::c_light - dmass_dR * v * v;
		return {v, force / mass};
	};

	int N = 256;
	const int max_iters = 10;
	const amrex::Real tol = 1.0e-6_rt * std::max(R_s, 1.0_rt);
	amrex::GpuArray<amrex::Real, 2> y_prev{R0, v0};

	for (int iter = 0; iter < max_iters; ++iter) {
		const amrex::Real dt = dt_target / static_cast<amrex::Real>(N);
		amrex::GpuArray<amrex::Real, 2> y{R0, v0};

		for (int step = 0; step < N; ++step) {
			const auto k1 = rhs(y);
			const auto k2 = rhs({y[0] + 0.5_rt * dt * k1[0], y[1] + 0.5_rt * dt * k1[1]});
			const auto k3 = rhs({y[0] + 0.5_rt * dt * k2[0], y[1] + 0.5_rt * dt * k2[1]});
			const auto k4 = rhs({y[0] + dt * k3[0], y[1] + dt * k3[1]});
			y[0] += (dt / 6.0_rt) * (k1[0] + 2.0_rt * k2[0] + 2.0_rt * k3[0] + k4[0]);
			y[1] += (dt / 6.0_rt) * (k1[1] + 2.0_rt * k2[1] + 2.0_rt * k3[1] + k4[1]);
			y[0] = std::max(y[0], R_d);
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

// The luminosities [erg s^-1] driving the front ODE, returned as {L_ion, L_optical, L_rec}. L_rec is the recombination
// emission if every ionizing photon recombined; the network emits it into the optical band, and the ODE scales it by f_rec.
auto compute_driving_luminosities(amrex::Real flux_ion, amrex::Real flux_optical, amrex::Real eps_ion, amrex::Real eps_opt, amrex::Real T_i)
    -> amrex::GpuArray<amrex::Real, 3>
{
	const amrex::Real eps_rec = 13.6 * C::ev2erg + lambda_rec(T_i) / recombination_coefficient(T_i);
	return {flux_ion * eps_ion, flux_optical * eps_opt, flux_ion * eps_rec};
}

auto is_restart() -> bool
{
	std::string restartfile;
	amrex::ParmParse const pp;
	pp.query("restartfile", restartfile);
	return !restartfile.empty();
}

} // namespace

template <>
void RadSystem<DTypeFront3D>::AddRadSource(array_t &radEnergy, array_t &reducedFlux, const amrex::Box &indexRange,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	amrex::ParmParse const pp("photoionize");
	amrex::Real flux_optical = 3.0e50_rt;
	pp.query("flux_optical", flux_optical);
	amrex::Real flux_ion = 1.0e48_rt;
	pp.query("flux_ion", flux_ion);
	amrex::Real flux_ir = 0.0_rt;
	pp.query("flux_ir", flux_ir);

	const amrex::Real L_ir = flux_ir * RadSystem<DTypeFront3D>::GetThermalBandQuanta(group_ir);
	const amrex::Real L_optical = flux_optical * RadSystem<DTypeFront3D>::GetThermalBandQuanta(group_optical);
	const amrex::Real L_ionizing = flux_ion * RadSystem<DTypeFront3D>::GetChemBandQuanta(0);

	constexpr int N = 2;
	constexpr amrex::Real inv_N = 1.0 / static_cast<amrex::Real>(N);
	constexpr auto cutoff_r2 = static_cast<amrex::Real>(N * N);

	const amrex::Real x0 = 0.5_rt * (prob_lo[0] + prob_hi[0]);
	const amrex::Real y0 = 0.5_rt * (prob_lo[1] + prob_hi[1]);
	const amrex::Real z0 = 0.5_rt * (prob_lo[2] + prob_hi[2]);
	const amrex::Real volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	const amrex::Real inv_volume = 1.0 / volume;

	const int src_i = static_cast<int>(amrex::Math::floor((x0 - prob_lo[0]) / dx[0]));
	const int src_j = static_cast<int>(amrex::Math::floor((y0 - prob_lo[1]) / dx[1]));
	const int src_k = static_cast<int>(amrex::Math::floor((z0 - prob_lo[2]) / dx[2]));
	const amrex::Real frac_x = (x0 - prob_lo[0]) / dx[0] - static_cast<amrex::Real>(src_i);
	const amrex::Real frac_y = (y0 - prob_lo[1]) / dx[1] - static_cast<amrex::Real>(src_j);
	const amrex::Real frac_z = (z0 - prob_lo[2]) / dx[2] - static_cast<amrex::Real>(src_k);

	constexpr int stencil_width = 2 * N + 1;
	amrex::Real norm_sum = 0.0_rt;
	for (int kk = 0; kk < stencil_width; ++kk) {
		const amrex::Real dz = static_cast<amrex::Real>(kk - N) + 0.5 - frac_z;
		for (int jj = 0; jj < stencil_width; ++jj) {
			const amrex::Real dy = static_cast<amrex::Real>(jj - N) + 0.5 - frac_y;
			for (int ii = 0; ii < stencil_width; ++ii) {
				const amrex::Real di = static_cast<amrex::Real>(ii - N) + 0.5 - frac_x;
				const amrex::Real r2 = di * di + dy * dy + dz * dz;
				if (r2 <= cutoff_r2) {
					norm_sum += wendland_c2(std::sqrt(r2) * inv_N);
				}
			}
		}
	}
	const amrex::Real inv_norm = 1.0_rt / norm_sum;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const amrex::Real di = static_cast<amrex::Real>(i - src_i) + 0.5 - frac_x;
		const amrex::Real dj = static_cast<amrex::Real>(j - src_j) + 0.5 - frac_y;
		const amrex::Real dk = static_cast<amrex::Real>(k - src_k) + 0.5 - frac_z;
		const amrex::Real r2 = di * di + dj * dj + dk * dk;
		const amrex::Real weight = (r2 <= cutoff_r2) ? wendland_c2(std::sqrt(r2) * inv_N) * inv_norm * inv_volume : 0.0_rt;

		for (int g = 0; g < Physics_Traits<DTypeFront3D>::nGroups; ++g) {
			amrex::Real luminosity = 0.0_rt;
			if (g == group_ir) {
				luminosity = L_ir;
			} else if (g == group_optical) {
				luminosity = L_optical;
			} else if (g == group_ionizing) {
				luminosity = L_ionizing;
			}
			radEnergy(i, j, k, g) = luminosity * weight;
			reducedFlux(i, j, k, 3 * g + 0) = 0.0_rt;
			reducedFlux(i, j, k, 3 * g + 1) = 0.0_rt;
			reducedFlux(i, j, k, 3 * g + 2) = 0.0_rt;
		}
	});
}

template <> void QuokkaSimulation<DTypeFront3D>::preCalculateInitialConditions()
{
	// initialize microphysics routines
	init_extern_parameters();

	// parmparse species, temperature, and source photon rates
	amrex::ParmParse const pp("photoionize");
	userData_.small_temp = 1e-2;
	userData_.small_dens = 1e-60;
	userData_.temperature = 1.0e2;
	userData_.n_e_init = 1.0e-10_rt;
	userData_.n_HI_init = 1.0e2_rt;
	userData_.n_HII_init = 1.0e-10_rt;
	userData_.flux_optical = 3.0e50_rt;
	userData_.flux_ion = 1.0e48_rt;
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

	userData_.eps_ir = RadSystem<DTypeFront3D>::GetThermalBandQuanta(group_ir);
	userData_.eps_opt = RadSystem<DTypeFront3D>::GetThermalBandQuanta(group_optical);
	userData_.eps_ion = RadSystem<DTypeFront3D>::GetChemBandQuanta(0);

	userData_.T_ionized = compute_equilibrium_temperature_ionized(userData_.n_HI_init, userData_.eps_ion);
	amrex::Print() << "Band mean photon energies: IR " << userData_.eps_ir / C::ev2erg << " eV, optical " << userData_.eps_opt / C::ev2erg
		       << " eV, ionizing " << userData_.eps_ion / C::ev2erg << " eV\n";
	amrex::Print() << "Photoionization-equilibrium temperature of the ionized gas: " << userData_.T_ionized << " K\n";

	{
		const amrex::Real R_s = stromgren_radius(userData_.flux_ion, userData_.n_HI_init, userData_.T_ionized);
		const amrex::Real c_s = ionized_sound_speed(userData_.T_ionized);
		const amrex::Real R_d = dusty_stromgren_radius(R_s, ionizing_dust_kappa() * userData_.n_HI_init * spmasses[Network::Species::H]);
		userData_.r_ode_last_t_ = 0.0_rt;
		userData_.r_ode_last_R_ = R_d;
		userData_.r_ode_last_v_ = c_s;
		amrex::Print() << "Stromgren radius R_s = " << R_s << " cm (with dust: " << R_d << " cm), ionized sound speed c_s = " << c_s << " cm/s\n";
	}

	eos_init(userData_.small_temp, userData_.small_dens);
	// Called once per level built at t = 0. Reopening an open stream fails and silently drops every later write.
	// A restart appends to the file the original run wrote.
	if (amrex::ParallelDescriptor::IOProcessor() && !userData_.output_file_.is_open()) {
		if (is_restart()) {
			userData_.output_file_.open(std::string("dtype_front_3d_radii") + ".csv", std::ios::app);
		} else {
			userData_.output_file_.open(std::string("dtype_front_3d_radii") + ".csv");
			userData_.output_file_ << "time,r_effective,r_shell,r_spitzer,r_ode,E_opt_tot,E_ir_tot\n";
		}
	}
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<DTypeFront3D>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
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

template <> void QuokkaSimulation<DTypeFront3D>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	quokka::microphysics::BurnState<DTypeFront3D> state;
	std::array<Real, Network::NumSpec> numdens = {-1.0};
	numdens[Network::Species::e] = userData_.n_e_init;
	numdens[Network::Species::H] = userData_.n_HI_init;
	numdens[Network::Species::H_p] = userData_.n_HII_init;

	state.T = userData_.temperature;
	Real rhotot = 0.0_rt;
	for (int n = 0; n < Network::NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * spmasses[n];
	}
	state.rho = rhotot;

	eos(eos_input_rt, state);
	const auto Egas0 = state.e * rhotot;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<DTypeFront3D>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<DTypeFront3D>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad_floor_;
			state_cc(i, j, k, RadSystem<DTypeFront3D>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFront3D>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFront3D>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
		}
		state_cc(i, j, k, RadSystem<DTypeFront3D>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFront3D>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<DTypeFront3D>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFront3D>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFront3D>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFront3D>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < Network::NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<DTypeFront3D>::scalar0_index + nn) =
			    state.xn[nn] * spmasses[nn]; // scalar indices carry partial densities instead of number densities
		}
	});
}

template <> void QuokkaSimulation<DTypeFront3D>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real time, int /*ngrow*/)
{
	// Refine wherever the neutral fraction jumps by more than x_HI_grad_threshold between neighbouring cells.
	const amrex::Real x_HI_grad_threshold = 0.3;

	// The initial gas is uniformly neutral, so the gradient tags nothing at t = 0 and the R-type phase would run on the base
	// grid until the first regrid. Refine the sphere the front reaches within the first few coarse steps (2 R_s) from the start.
	const bool tag_source_region = (time <= 0.0_rt);
	const amrex::Real r_tag = 2.0_rt * stromgren_radius(userData_.flux_ion, userData_.n_HI_init, userData_.T_ionized);
	const amrex::Real r_tag2 = r_tag * r_tag;
	const auto prob_lo = geom[lev].ProbLoArray();
	const auto prob_hi = geom[lev].ProbHiArray();
	const auto dx = geom[lev].CellSizeArray();

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			if (tag_source_region) {
				const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5_rt) * dx[0] - 0.5_rt * (prob_lo[0] + prob_hi[0]);
				const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5_rt) * dx[1] - 0.5_rt * (prob_lo[1] + prob_hi[1]);
				const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5_rt) * dx[2] - 0.5_rt * (prob_lo[2] + prob_hi[2]);
				if (x * x + y * y + z * z < r_tag2) {
					tag(i, j, k) = amrex::TagBox::SET;
					return;
				}
			}

			const amrex::Real x0 = neutral_fraction(state, i, j, k);

			const amrex::Real del_x =
			    std::max(std::abs(neutral_fraction(state, i + 1, j, k) - x0), std::abs(x0 - neutral_fraction(state, i - 1, j, k)));
			const amrex::Real del_y =
			    std::max(std::abs(neutral_fraction(state, i, j + 1, k) - x0), std::abs(x0 - neutral_fraction(state, i, j - 1, k)));
			const amrex::Real del_z =
			    std::max(std::abs(neutral_fraction(state, i, j, k + 1) - x0), std::abs(x0 - neutral_fraction(state, i, j, k - 1)));

			if (std::max({del_x, del_y, del_z}) > x_HI_grad_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

template <> void QuokkaSimulation<DTypeFront3D>::computeAfterTimestep()
{
	const int lev = 0;
	const amrex::Real t = tNew_[lev];

	// One bin per cell width of the finest level the inputs allow, the same cells Check 2 measures its tolerance in.
	const int n_bins = geom[maxLevel()].Domain().length(0) / 2;

	const amrex::Real r_effective = compute_effective_radius(*this);
	const amrex::Real r_shell = compute_shell_radius(state_new_cc_, geom, ref_ratio, finest_level, n_bins);
	const amrex::Real r_spitzer = spitzer_radius(t, userData_.flux_ion, userData_.n_HI_init, userData_.T_ionized);

	amrex::Real r_ode = std::numeric_limits<amrex::Real>::quiet_NaN();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const amrex::Real n_0 = userData_.n_HI_init;
		const amrex::Real rho_0 = n_0 * spmasses[Network::Species::H];
		const amrex::Real T_i = userData_.T_ionized;
		const amrex::Real R_s = stromgren_radius(userData_.flux_ion, n_0, T_i);
		const amrex::Real c_s = ionized_sound_speed(T_i);
		const auto L_bands = compute_driving_luminosities(userData_.flux_ion, userData_.flux_optical, userData_.eps_ion, userData_.eps_opt, T_i);
		const amrex::Real kappa_rho_0 = ionizing_dust_kappa() * rho_0;

		amrex::Real dt_ode = t - userData_.r_ode_last_t_;
		if (dt_ode < 0.0_rt) {
			// time went backwards or was reset; restart the integration from the R-type endpoint
			userData_.r_ode_last_t_ = 0.0_rt;
			userData_.r_ode_last_R_ = dusty_stromgren_radius(R_s, kappa_rho_0);
			userData_.r_ode_last_v_ = c_s;
			dt_ode = t;
		}
		const auto y = integrate_front(dt_ode, userData_.r_ode_last_R_, userData_.r_ode_last_v_, R_s, rho_0, c_s, L_bands[0], L_bands[1], L_bands[2],
					       kappa_optical, kappa_rho_0);
		userData_.r_ode_last_t_ = t;
		userData_.r_ode_last_R_ = y[0];
		userData_.r_ode_last_v_ = y[1];
		r_ode = y[0];
	}
	amrex::ParallelDescriptor::Bcast(&r_ode, 1, amrex::ParallelDescriptor::IOProcessorNumber());

	userData_.t_vec_.push_back(t);
	userData_.reff_vec_.push_back(r_effective);
	userData_.rshell_vec_.push_back(r_shell);
	userData_.rspitzer_vec_.push_back(r_spitzer);
	userData_.rode_vec_.push_back(r_ode);

	const auto [v_max, T_max] = compute_max_speed_and_temperature(*this);
	userData_.v_max_run_ = std::max(userData_.v_max_run_, v_max);
	userData_.T_max_run_ = std::max(userData_.T_max_run_, T_max);
	if (v_max > c_hat && userData_.t_first_v_violation_ < 0.0_rt) {
		userData_.t_first_v_violation_ = t;
	}
	if (T_max > T_gas_limit && userData_.t_first_T_violation_ < 0.0_rt) {
		userData_.t_first_T_violation_ = t;
	}

	const amrex::Real E_opt_tot = compute_group_energy(*this, group_optical);
	const amrex::Real E_ir_tot = compute_group_energy(*this, group_ir);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_ << t << ',' << r_effective << ',' << r_shell << ',' << r_spitzer << ',' << r_ode << ',' << E_opt_tot << ',' << E_ir_tot
				       << '\n';
	}
}

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 0.3;

	// Problem initialization
	QuokkaSimulation<DTypeFront3D> sim;

	// A restart reads the state from the checkpoint and skips preCalculateInitialConditions, which reads the problem
	// parameters, initialises the EOS and network, and starts the front ODE at t = 0. The first computeAfterTimestep then
	// integrates the ODE from t = 0 to the restart time. Checks 2 and 3 only see the timesteps after the restart.
	if (is_restart()) {
		sim.preCalculateInitialConditions();
	}

	// initialize
	sim.setInitialConditions();
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;

	sim.evolve();

	int status = 0;

	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim.geom[0].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim.geom[0].ProbHiArray();
	const double r_domain = std::min({0.5 * (prob_hi[0] - prob_lo[0]), 0.5 * (prob_hi[1] - prob_lo[1]), 0.5 * (prob_hi[2] - prob_lo[2])});

	// Check 1: gas temperature in the ionized cavity and in the undisturbed neutral gas.
	{
		const double T_ion_eq = compute_equilibrium_temperature_ionized(sim.userData_.n_HI_init, sim.userData_.eps_ion);
		const double T_neu_eq = compute_equilibrium_temperature_neutral(sim.userData_.n_HI_init);

		amrex::MultiFab const &state_mf = sim.state_new_cc_[0];

		// Collect temperatures per region: cavity (x_HII > 90%), neutral (x_HI > 99.99%). The gas the front has
		// set in motion is adiabatically cooled and out of thermal equilibrium, so the neutral sample is
		// restricted to quiescent gas.
		const double v_quiescent = 0.05 * ionized_sound_speed(sim.userData_.T_ionized);
		std::vector<double> cavity_temps;
		std::vector<double> neutral_temps;

		for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
			const amrex::Box &box = mfi.validbox();

			// In GPU builds, MultiFab data resides on device; copy to pinned host memory before CPU access.
			amrex::FArrayBox host_fab(box, state_mf.nComp(), amrex::The_Pinned_Arena());
			static_cast<void>(state_mf[mfi].template copyToMem<amrex::RunOn::Device>(box, 0, state_mf.nComp(), host_fab.dataPtr()));
			amrex::Gpu::synchronize();

			const auto state = host_fab.const_array();

			amrex::LoopOnCpu(box, [&](int i, int j, int k) noexcept {
				const amrex::Real rho = state(i, j, k, HydroSystem<DTypeFront3D>::density_index);
				const amrex::Real Eint = state(i, j, k, RadSystem<DTypeFront3D>::gasInternalEnergy_index);
				const amrex::Real n_HI_cell = state(i, j, k, HydroSystem<DTypeFront3D>::scalar0_index + static_cast<int>(Network::Species::H)) /
							      spmasses[Network::Species::H];
				const amrex::Real n_HII_cell =
				    state(i, j, k, HydroSystem<DTypeFront3D>::scalar0_index + static_cast<int>(Network::Species::H_p)) /
				    spmasses[Network::Species::H_p];
				const amrex::Real denom = n_HI_cell + n_HII_cell;
				if (denom <= 0.0_rt) {
					return;
				}
				const amrex::Real x_HII = n_HII_cell / denom;
				const amrex::Real x_HI = n_HI_cell / denom;

				quokka::microphysics::BurnState<DTypeFront3D> bstate;
				for (int nn = 0; nn < Network::NumSpec; ++nn) {
					bstate.xn[nn] = state(i, j, k, HydroSystem<DTypeFront3D>::scalar0_index + nn) / spmasses[nn];
				}
				bstate.rho = rho;
				bstate.e = Eint / rho;
				bstate.T = 1.0e4; // initial guess
				eos(eos_input_re, bstate);
				const double T_cell = bstate.T;

				if (x_HII > 0.90_rt) {
					cavity_temps.push_back(T_cell);
				}
				const amrex::Real px = state(i, j, k, HydroSystem<DTypeFront3D>::x1Momentum_index);
				const amrex::Real py = state(i, j, k, HydroSystem<DTypeFront3D>::x2Momentum_index);
				const amrex::Real pz = state(i, j, k, HydroSystem<DTypeFront3D>::x3Momentum_index);
				const amrex::Real v_mag = std::sqrt(px * px + py * py + pz * pz) / rho;
				if (x_HI > 0.9999_rt && v_mag < v_quiescent) {
					neutral_temps.push_back(T_cell);
				}
			});
		}

		auto compute_median_and_check = [&](std::vector<double> &local_temps, double T_analytical, const char *region_name) {
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
				// A NaN would also break std::sort's ordering, so reject non-finite temperatures before sorting.
				const auto n_nonfinite = std::count_if(all_temps.begin(), all_temps.end(), [](double T) { return !std::isfinite(T); });
				if (n_nonfinite > 0) {
					amrex::Print() << "Test FAILED: " << n_nonfinite << " of " << ntot << " " << region_name
						       << " cells have a non-finite temperature.\n";
					status = 1;
					return;
				}
				std::sort(all_temps.begin(), all_temps.end());
				const double T_median = (ntot % 2 == 0) ? 0.5 * (all_temps[ntot / 2 - 1] + all_temps[ntot / 2]) : all_temps[ntot / 2];
				const double rel_err = std::abs(T_median - T_analytical) / T_analytical;
				if (!(rel_err <= 0.05)) {
					amrex::Print()
					    << "Test FAILED: " << region_name << " median temperature " << T_median << " K differs from analytical equilibrium "
					    << T_analytical << " K by " << 100.0 * rel_err << "% (tolerance: 5%).\n";
					status = 1;
				} else {
					amrex::Print() << "Test passed: " << region_name << " median temperature " << T_median
						       << " K is within 5% of analytical equilibrium " << T_analytical << " K (" << ntot << " cells).\n";
				}
			}
		};

		compute_median_and_check(cavity_temps, T_ion_eq, "cavity");
		compute_median_and_check(neutral_temps, T_neu_eq, "neutral");
	}

	// Check 2: the D-type front radius against the integrated thin-shell solution that carries both the
	// ionized-gas pressure and the radiation pressure. Passes if either the effective ionized radius or the
	// max-density shell radius matches, as in DTypeFront1D, since the thin-shell ODE collapses the ionization
	// front and the shock into one surface, while the two can separate in the simulation.
	if (sim.userData_.rode_vec_.empty()) {
		amrex::Print() << "Test FAILED: no timestep was taken, so there is no front radius to check.\n";
		status = 1;
	} else {
		constexpr double cm_per_pc = 3.085677581491367e18;
		const double r_eff = sim.userData_.reff_vec_.back();
		const double r_shell = sim.userData_.rshell_vec_.back();
		const double r_ode = sim.userData_.rode_vec_.back();
		const double r_spitzer = sim.userData_.rspitzer_vec_.back();

		amrex::Print() << "Integrated solution (gas + radiation pressure): " << r_ode << " cm = " << r_ode / cm_per_pc << " pc\n";
		amrex::Print() << "Spitzer solution (gas pressure only):           " << r_spitzer << " cm = " << r_spitzer / cm_per_pc << " pc\n";
		amrex::Print() << "Radiation pressure adds " << 100.0 * (r_ode / r_spitzer - 1.0) << "% over the gas-only Spitzer radius.\n";
		amrex::Print() << "Effective ionized radius:  " << r_eff << " cm = " << r_eff / cm_per_pc << " pc\n";
		amrex::Print() << "Max-density shell radius:  " << r_shell << " cm = " << r_shell / cm_per_pc << " pc\n";

		if (!(r_ode > 0.0)) {
			amrex::Print() << "Test FAILED: the integrated front solution is not positive; check photoionize.flux_ion.\n";
			status = 1;
		} else if (r_ode >= r_domain) {
			amrex::Print() << "Test FAILED: the integrated front has left the domain; reduce stop_time or enlarge the box.\n";
			status = 1;
		} else {
			// Cells of the finest level the inputs allow, so the tolerance is strictest if the front is not refined.
			const double dx_finest = sim.geom[sim.maxLevel()].CellSize(0);
			const double eff_cell_diff = (r_eff - r_ode) / dx_finest;
			const double shell_cell_diff = (r_shell - r_ode) / dx_finest;
			const bool eff_ok = std::abs(eff_cell_diff) <= front_radius_tol_cells;
			const bool shell_ok = std::abs(shell_cell_diff) <= front_radius_tol_cells;

			if (eff_ok) {
				amrex::Print() << "Test passed: the effective ionized radius matches the integrated radiation + gas pressure solution within "
					       << front_radius_tol_cells << " finest cells (" << eff_cell_diff << " cells).\n";
			} else {
				amrex::Print() << "The effective ionized radius differs from the integrated radiation + gas pressure solution by more than "
					       << front_radius_tol_cells << " finest cells (" << eff_cell_diff << " cells).\n";
			}

			if (shell_ok) {
				amrex::Print() << "Test passed: the max-density shell matches the integrated radiation + gas pressure solution within "
					       << front_radius_tol_cells << " finest cells (" << shell_cell_diff << " cells).\n";
			} else {
				amrex::Print() << "The max-density shell differs from the integrated radiation + gas pressure solution by more than "
					       << front_radius_tol_cells << " finest cells (" << shell_cell_diff << " cells).\n";
			}

			if (!eff_ok && !shell_ok) {
				amrex::Print() << "Test FAILED: neither the effective ionized radius nor the max-density shell matches the integrated "
						  "solution within tolerance.\n";
				status = 1;
			}
		}
	}

	// Check 3: at the end of every coarse timestep, no cell on any level moved faster than the reduced speed of light
	// or was hotter than T_gas_limit.
	{
		if (sim.userData_.t_first_v_violation_ >= 0.0) {
			amrex::Print() << "Test FAILED: the gas speed exceeded the reduced speed of light (" << c_hat
				       << " cm/s), first at t = " << sim.userData_.t_first_v_violation_ << " s; maximum " << sim.userData_.v_max_run_
				       << " cm/s.\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: the gas speed stayed below the reduced speed of light (maximum " << sim.userData_.v_max_run_
				       << " cm/s, " << sim.userData_.v_max_run_ / c_hat << " c_hat).\n";
		}

		if (sim.userData_.t_first_T_violation_ >= 0.0) {
			amrex::Print() << "Test FAILED: the gas temperature exceeded " << T_gas_limit
				       << " K, first at t = " << sim.userData_.t_first_T_violation_ << " s; maximum " << sim.userData_.T_max_run_ << " K.\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: the gas temperature stayed below " << T_gas_limit << " K (maximum " << sim.userData_.T_max_run_
				       << " K).\n";
		}
	}

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr amrex::Real seconds_per_Myr = 3.15576e13;
		constexpr amrex::Real cm_per_pc = 3.085677581491367e18;

		const auto n = static_cast<int>(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> t_Myr(n);
		std::vector<amrex::Real> r_eff_pc(n);
		std::vector<amrex::Real> r_shell_pc(n);
		std::vector<amrex::Real> r_spitzer_pc(n);
		std::vector<amrex::Real> r_ode_pc(n);
		// The Check 2 tolerance around the ODE solution.
		const amrex::Real band_pc = front_radius_tol_cells * sim.geom[sim.maxLevel()].CellSize(0) / cm_per_pc;
		std::vector<amrex::Real> r_ode_lower_pc(n);
		std::vector<amrex::Real> r_ode_upper_pc(n);
		for (int i = 0; i < n; ++i) {
			t_Myr[i] = sim.userData_.t_vec_[i] / seconds_per_Myr;
			r_eff_pc[i] = sim.userData_.reff_vec_[i] / cm_per_pc;
			r_shell_pc[i] = sim.userData_.rshell_vec_[i] / cm_per_pc;
			r_spitzer_pc[i] = sim.userData_.rspitzer_vec_[i] / cm_per_pc;
			r_ode_pc[i] = sim.userData_.rode_vec_[i] / cm_per_pc;
			r_ode_lower_pc[i] = r_ode_pc[i] - band_pc;
			r_ode_upper_pc[i] = r_ode_pc[i] + band_pc;
		}
		matplotlibcpp::clf();
		std::map<std::string, std::string> eff_args;
		eff_args["label"] = "effective ionized radius";
		eff_args["color"] = "C1";
		std::map<std::string, std::string> shell_args;
		shell_args["label"] = "max-density shell";
		shell_args["color"] = "C0";
		std::map<std::string, std::string> ode_band_args;
		ode_band_args["label"] = "ODE +/- " + std::to_string(static_cast<int>(front_radius_tol_cells)) + " finest cells (Check 2 tolerance)";
		ode_band_args["color"] = "#00000020";
		std::map<std::string, std::string> spitzer_args;
		spitzer_args["label"] = "Spitzer (gas pressure only, 4/7 law)";
		spitzer_args["color"] = "k";
		spitzer_args["linestyle"] = "--";
		std::map<std::string, std::string> ode_args;
		ode_args["label"] = "ODE (gas + radiation pressure)";
		ode_args["color"] = "k";
		ode_args["linestyle"] = ":";
		matplotlibcpp::fill_between(t_Myr, r_ode_lower_pc, r_ode_upper_pc, ode_band_args);
		matplotlibcpp::plot(t_Myr, r_eff_pc, eff_args);
		matplotlibcpp::plot(t_Myr, r_shell_pc, shell_args);
		matplotlibcpp::plot(t_Myr, r_spitzer_pc, spitzer_args);
		matplotlibcpp::plot(t_Myr, r_ode_pc, ode_args);
		matplotlibcpp::xlabel("time (Myr)");
		matplotlibcpp::ylabel("front radius (pc)");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save(std::string("./dtype_front_3d_radii") + ".pdf");
	}
#endif

	amrex::Print() << "Finished." << '\n';
	return status;
}
