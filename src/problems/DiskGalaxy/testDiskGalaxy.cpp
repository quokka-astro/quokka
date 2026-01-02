//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testDiskGalaxy.cpp
/// \brief Defines a simulation using the AGORA isolated galaxy initial conditions.
///

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArrayBase.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "math/quadrature.hpp"
#include "particles/particle_types.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct AgoraGalaxy {
};

template <> struct quokka::EOS_Traits<AgoraGalaxy> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = 0.6 * C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct HydroSystem_Traits<AgoraGalaxy> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<AgoraGalaxy> {
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	static constexpr bool is_mhd_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = 1;			     // number of radiation groups
};

template <> struct Particle_Traits<AgoraGalaxy> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC | ParticleSwitch::StochasticStellarPop;
};

template <> struct SimulationData<AgoraGalaxy> {
	amrex::Real r_inner{};
	amrex::Real r_outer{};
	amrex::Real vcirc_outer{};
	amrex::Real rho_outer{};
	amrex::Real velr_outer{};
	amrex::Real temp_outer{};

	amrex::Real vcirc_inner{};
	amrex::Real rho_inner{};
	amrex::Real velr_inner{};
	amrex::Real temp_inner{};

	amrex::Gpu::PinnedVector<amrex::Real> radius;
	amrex::Gpu::PinnedVector<amrex::Real> vcirc;
	amrex::Gpu::PinnedVector<amrex::Real> rho_halo;
	amrex::Gpu::PinnedVector<amrex::Real> velr_halo;
	amrex::Gpu::PinnedVector<amrex::Real> temp_halo;
};

template <> void QuokkaSimulation<AgoraGalaxy>::preCalculateInitialConditions()
{
	// 1. read in circular velocity table "vcirc.dat"
	std::vector<amrex::Real> radius_h;
	std::vector<amrex::Real> vcirc_h;
	std::vector<amrex::Real> rho_h;
	std::vector<amrex::Real> velr_h;
	std::vector<amrex::Real> temp_h;

	// get circular velocity profile filename from ParmParse
	amrex::ParmParse const pp("agora_galaxy");
	std::string filename;
	pp.query("vcirc_file", filename);

	std::ifstream fstream(filename, std::ios::in);
	AMREX_ALWAYS_ASSERT(fstream.is_open());
	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values;

		for (double value = NAN; iss >> value;) {
			values.push_back(value);
		}
		if (values.empty()) {
			continue;
		}
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(values.size() == 5,
						 "agora_galaxy.vcirc_file must have 5 columns per row (R, vcirc, rho_halo, velr_halo, temp_halo).");
		Real const R_val = values.at(0);
		Real const vcirc_val = values.at(1);
		Real const rho_val = values.at(2);
		Real const velr_val = values.at(3);
		Real const temp_val = values.at(4);

		radius_h.push_back(R_val);
		vcirc_h.push_back(vcirc_val);
		rho_h.push_back(rho_val);
		velr_h.push_back(velr_val);
		temp_h.push_back(temp_val);
	}

	// 2. copy data to simData_.radius and simData_.vcirc
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!radius_h.empty(), "agora_galaxy.vcirc_file contained no numeric rows.");
	const size_t N = radius_h.size();
	userData_.radius.resize(N);
	userData_.vcirc.resize(N);
	userData_.rho_halo.resize(N);
	userData_.velr_halo.resize(N);
	userData_.temp_halo.resize(N);

	const double length_unit = 1.0e3 * C::parsec; // kpc
	const double vel_unit = 1.0e5;		      // km/s
	for (size_t i = 0; i < N; ++i) {
		userData_.radius[i] = radius_h[i] * length_unit;
		userData_.vcirc[i] = vcirc_h[i] * vel_unit;
		userData_.rho_halo[i] = rho_h[i];
		userData_.velr_halo[i] = velr_h[i];
		userData_.temp_halo[i] = temp_h[i];
	}

	// save min/max radii
	auto min_result = std::min_element(radius_h.begin(), radius_h.end());
	userData_.r_inner = (*min_result) * length_unit;
	userData_.vcirc_inner = vcirc_h[std::distance(radius_h.begin(), min_result)] * vel_unit;
	userData_.rho_inner = rho_h[std::distance(radius_h.begin(), min_result)];
	userData_.velr_inner = velr_h[std::distance(radius_h.begin(), min_result)];
	userData_.temp_inner = temp_h[std::distance(radius_h.begin(), min_result)];

	auto max_result = std::max_element(radius_h.begin(), radius_h.end());
	userData_.r_outer = (*max_result) * length_unit;
	userData_.vcirc_outer = vcirc_h[std::distance(radius_h.begin(), max_result)] * vel_unit;
	userData_.rho_outer = rho_h[std::distance(radius_h.begin(), max_result)];
	userData_.velr_outer = velr_h[std::distance(radius_h.begin(), max_result)];
	userData_.temp_outer = temp_h[std::distance(radius_h.begin(), max_result)];
}

template <> void QuokkaSimulation<AgoraGalaxy>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// read parameters
	//
	amrex::ParmParse const pp("agora_galaxy");

	double magnetic_field_microgauss = 1.0; // default B-field strength
	pp.query("magnetic_field_microgauss", magnetic_field_microgauss);
	const double B_0 = magnetic_field_microgauss * 1.0e-6 / std::sqrt(4.0 * M_PI);

	// disc parameters
	double disk_gas_mass_Msun = NAN;     // disk mass
	double disk_Rscale_kpc = NAN;	     // disk scale length
	double disk_zscale_kpc = NAN;	     // disk scale height
	double T_disk = NAN;		     // K
	double disk_perturb_amplitude = NAN; // amplitude of harmonic mode perturbation
	double disk_perturb_Rmax_kpc = NAN;  // max radius (in kpc) for harmonic mode perturbations
	pp.query("disk_gas_mass_Msun", disk_gas_mass_Msun);
	pp.query("disk_Rscale_kpc", disk_Rscale_kpc);
	pp.query("disk_zscale_kpc", disk_zscale_kpc);
	pp.query("disk_temperature", T_disk);
	pp.query("disk_perturb_amplitude", disk_perturb_amplitude);
	pp.query("disk_perturb_Rmax_kpc", disk_perturb_Rmax_kpc);
	int debug_disk_switch = 0;
	pp.query("debug_disk_switch", debug_disk_switch);
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_gas_mass_Msun));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_Rscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_zscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(T_disk));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_perturb_amplitude));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_perturb_Rmax_kpc));

	const double disk_gas_mass = disk_gas_mass_Msun * C::M_solar;
	const double R_d = disk_Rscale_kpc * (1.0e3 * C::parsec);
	const double z_d = disk_zscale_kpc * (1.0e3 * C::parsec);
	const double R_max_perturb = disk_perturb_Rmax_kpc * (1e3 * C::parsec);
	const double rho_0 = disk_gas_mass / 4. / M_PI / (R_d * R_d) / z_d; // normalization constant

	// read tables

	double const *R_table = userData_.radius.dataPtr();
	double const *vcirc_table = userData_.vcirc.dataPtr();
	double const *rhoH_table = userData_.rho_halo.dataPtr();
	double const *velr_table = userData_.velr_halo.dataPtr();
	double const *temp_table = userData_.temp_halo.dataPtr();

	auto const len_table = static_cast<int>(userData_.radius.size());
	const amrex::Real R_table_min = userData_.r_inner;
	const amrex::Real rho_inner = userData_.rho_inner;
	const amrex::Real vcirc_inner = userData_.vcirc_inner;
	const amrex::Real velr_inner = userData_.velr_inner;
	const amrex::Real temp_inner = userData_.temp_inner;

	const amrex::Real R_table_max = userData_.r_outer;
	const amrex::Real vcirc_outer = userData_.vcirc_outer;
	const amrex::Real rho_outer = userData_.rho_outer;
	const amrex::Real velr_outer = userData_.velr_outer;
	const amrex::Real temp_outer = userData_.temp_outer;

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

#ifndef AMREX_USE_GPU
	if (debug_disk_switch != 0) {
		auto rhoHalo_host = [R_table_min, R_table, R_table_max, rho_inner, rho_outer, rhoH_table, len_table](amrex::Real const R) {
			double rho_H = NAN;
			if (R > R_table_min && R < R_table_max) {
				rho_H = interpolate_value(R, R_table, rhoH_table, len_table);
			} else if (R <= R_table_min) {
				rho_H = rho_inner;
			} else {
				rho_H = rho_outer;
			}
			return rho_H;
		};

		auto tempHalo_host = [R_table_min, R_table, R_table_max, temp_inner, temp_outer, temp_table, len_table](amrex::Real const R) {
			double temp_H = NAN;
			if (R > R_table_min && R < R_table_max) {
				temp_H = interpolate_value(R, R_table, temp_table, len_table);
			} else if (R <= R_table_min) {
				temp_H = temp_inner;
			} else {
				temp_H = temp_outer;
			}
			return temp_H;
		};

		amrex::Real min_disk_r = std::numeric_limits<amrex::Real>::max();
		amrex::Real max_disk_r = 0.0;
		long disk_count = 0;

		for (int k = indexRange.smallEnd(2); k <= indexRange.bigEnd(2); ++k) {
			for (int j = indexRange.smallEnd(1); j <= indexRange.bigEnd(1); ++j) {
				for (int i = indexRange.smallEnd(0); i <= indexRange.bigEnd(0); ++i) {
					amrex::Real const x_mid = prob_lo[0] + (i + 0.5) * dx[0];
					amrex::Real const y_mid = prob_lo[1] + (j + 0.5) * dx[1];
					amrex::Real const z_mid = prob_lo[2] + (k + 0.5) * dx[2];
					amrex::Real const R_mid = std::sqrt((x_mid * x_mid) + (y_mid * y_mid));
					amrex::Real const r_mid = std::sqrt((x_mid * x_mid) + (y_mid * y_mid) + (z_mid * z_mid));

					double const rho_disk_mid = rho_0 * std::exp(-R_mid / R_d) * std::exp(-std::abs(z_mid) / z_d);
					double const rho_halo_mid = rhoHalo_host(r_mid);
					double const temp_halo_mid = tempHalo_host(r_mid);

					if (rho_halo_mid * temp_halo_mid < rho_disk_mid * T_disk) {
						min_disk_r = std::min(min_disk_r, r_mid);
						max_disk_r = std::max(max_disk_r, r_mid);
						++disk_count;
					}
				}
			}
		}

		const amrex::Real kpc = 1.0e3 * C::parsec;
		const amrex::Real min_disk_r_global = amrex::ParallelDescriptor::ReduceRealMin(min_disk_r);
		const amrex::Real max_disk_r_global = amrex::ParallelDescriptor::ReduceRealMax(max_disk_r);
		const long disk_count_global = amrex::ParallelDescriptor::ReduceLongSum(disk_count);

		if (amrex::ParallelDescriptor::IOProcessor()) {
			if (disk_count_global > 0) {
				amrex::Print() << "[DiskGalaxy] disk-selected cells=" << disk_count_global << " r_min_kpc=" << (min_disk_r_global / kpc)
					       << " r_max_kpc=" << (max_disk_r_global / kpc) << "\n";
			} else {
				amrex::Print() << "[DiskGalaxy] disk-selected cells=0\n";
			}
		}
	}
#endif

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// Cartesian coordinates
		amrex::Real const x0 = prob_lo[0] + (i * dx[0]);
		amrex::Real const y0 = prob_lo[1] + (j * dx[1]);
		amrex::Real const z0 = prob_lo[2] + (k * dx[2]);

		amrex::Real const x1 = prob_lo[0] + ((i + 1) * dx[0]);
		amrex::Real const y1 = prob_lo[1] + ((j + 1) * dx[1]);
		amrex::Real const z1 = prob_lo[2] + ((k + 1) * dx[2]);

		amrex::Real const x_mid = 0.5 * (x0 + x1);
		amrex::Real const y_mid = 0.5 * (y0 + y1);
		amrex::Real const z_mid = 0.5 * (z0 + z1);
		amrex::Real const R_mid = std::sqrt((x_mid * x_mid) + (y_mid * y_mid));

		amrex::Real const B_phi = B_0 * std::exp(-R_mid / R_d) * std::exp(-std::abs(z_mid) / z_d);
		amrex::Real Bx = 0.0;
		amrex::Real By = 0.0;
		if (R_mid > 0.0) {
			Bx = -B_phi * y_mid / R_mid;
			By = B_phi * x_mid / R_mid;
		}
		amrex::Real const Emag = 0.5 * ((Bx * Bx) + (By * By));

		// compute density profile
		auto rho_exact = [rho_0, R_d, z_d](double x, double y, double z) {
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			return rho_0 * std::exp(-R / R_d) * std::exp(-std::abs(z) / z_d);
		};

		auto vcirc_exact = [R_table_min, R_table_max, R_table, vcirc_inner, vcirc_outer, vcirc_table, len_table](const amrex::Real R) {
			double vcirc = NAN;
			if (R > R_table_min && R < R_table_max) {
				vcirc = interpolate_value(R, R_table, vcirc_table, len_table);
			} else if (R >= R_table_max) {
				vcirc = vcirc_outer;
			} else if (R <= R_table_min) {
				vcirc = vcirc_inner;
			}
			return vcirc;
		};

		// compute velocity profiles
		auto vx_exact = [vcirc_exact](double x, double y, double /*z*/) {
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const theta = std::atan2(x, y);
			return -vcirc_exact(R) * std::cos(theta); // vx
		};

		auto vy_exact = [vcirc_exact](double x, double y, double /*z*/) {
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const theta = std::atan2(x, y);
			return vcirc_exact(R) * std::sin(theta); // vy
		};

		auto rhoHalo = [R_table_min, R_table, R_table_max, rho_inner, rho_outer, rhoH_table, len_table](const amrex::Real R) {
			double rho_H = NAN;
			if (R > R_table_min && R < R_table_max) {
				rho_H = interpolate_value(R, R_table, rhoH_table, len_table);
			} else if (R <= R_table_min) {
				rho_H = rho_inner;
			} else {
				rho_H = rho_outer;
			}

			return rho_H;
		};

		auto velHalo = [R_table_min, R_table, R_table_max, velr_inner, velr_outer, velr_table, len_table](const amrex::Real R) {
			double vel_H = NAN;
			if (R > R_table_min && R < R_table_max) {
				vel_H = interpolate_value(R, R_table, velr_table, len_table);
			} else if (R <= R_table_min) {
				vel_H = velr_inner;
			} else {
				vel_H = velr_outer;
			}
			return -vel_H;
		};

		auto tempHalo = [R_table_min, R_table, R_table_max, temp_inner, temp_outer, temp_table, len_table](const amrex::Real R) {
			double temp_H = NAN;
			if (R > R_table_min && R < R_table_max) {
				temp_H = interpolate_value(R, R_table, temp_table, len_table);
			} else if (R <= R_table_min) {
				temp_H = temp_inner;
			} else {
				temp_H = temp_outer;
			}
			return temp_H;
		};

		// compute density profiles
		auto rhoHalo_exact = [rhoHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return rhoHalo(r);
		};

		auto tempHalo_exact = [tempHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return tempHalo(r);
		};

		// compute momenta profiles
		auto velx_exact = [velHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return (r > 0.0) ? (velHalo(r) * x / r) : 0.0; // vx
		};

		auto vely_exact = [velHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return (r > 0.0) ? (velHalo(r) * y / r) : 0.0; // vy
		};

		auto velz_exact = [velHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return (r > 0.0) ? (velHalo(r) * z / r) : 0.0; // vz
		};

		// integrate density profile over cell volume
		// TODO(bwibking): use adaptive quadrature with relative tolerance
		const double cell_vol = dx[0] * dx[1] * dx[2];
		const double rho_disk = quad_3d(rho_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double rho_halo = quad_3d(rhoHalo_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double vel_Hx_halo = quad_3d(velx_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double vel_Hy_halo = quad_3d(vely_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double vel_Hz_halo = quad_3d(velz_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double temp_halo = quad_3d(tempHalo_exact, x0, x1, y0, y1, z0, z1) / cell_vol;

		// Compute halo momenta
		const double momx_halo = rho_halo * vel_Hx_halo;
		const double momy_halo = rho_halo * vel_Hy_halo;
		const double momz_halo = rho_halo * vel_Hz_halo;

		// Compute halo total internal energy
		// use mu = 0.61 as in cooling flow solutions
		constexpr double gamma_gas = quokka::EOS_Traits<AgoraGalaxy>::gamma;
		const double eint_halo = rho_halo * C::k_B * temp_halo / (0.61 * C::m_p * (gamma_gas - 1.0));

		AMREX_ALWAYS_ASSERT(!std::isnan(rho_disk));

		double rho = 0;
		double vx = 0;
		double vy = 0;
		double const vz = 0;
		double T = NAN;

		// IMPORTANT: transition between disk and halo at the P_halo == P_disk surface
		if (rho_halo * temp_halo < rho_disk * T_disk) { // we are in the disk
			double const x = 0.5 * (x0 + x1);
			double const y = 0.5 * (y0 + y1);
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const theta = std::atan2(x, y);

			// set density (compute density perturbation)
			// NOTE: jn is the C standard math function for BesselJ. it works everywhere.
			double const drho_over_rho = disk_perturb_amplitude * jn(2, 5.1356 * R / R_max_perturb) * std::sin(2.0 * theta);
			rho = rho_disk * (1 + drho_over_rho);
			AMREX_ALWAYS_ASSERT(rho > 0.);

			// set temperature
			T = T_disk;

			// set velocity (integrate velocity profiles over cell volume)
			// TODO(bwibking): use adaptive quadrature with relative tolerance
			vx = quad_3d(vx_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
			vy = quad_3d(vy_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
			AMREX_ALWAYS_ASSERT(!std::isnan(vx));
			AMREX_ALWAYS_ASSERT(!std::isnan(vy));
		}

		// compute auxiliary quantities (approximate mean molecular weight)
		constexpr double mu = 0.61;
		double const Eint = (rho > 0.0 && std::isfinite(T)) ? (rho * C::k_B * T / (mu * C::m_p * (gamma_gas - 1.0))) : 0.0;

		// Add up disk and halo contributions
		double const rho_disk_halo = rho + rho_halo;
		double const momx_disk_halo = rho * vx + momx_halo;
		double const momy_disk_halo = rho * vy + momy_halo;
		double const momz_disk_halo = rho * vz + momz_halo;
		double const Ekin_disk_halo =
		    0.5 * (momx_disk_halo * momx_disk_halo + momy_disk_halo * momy_disk_halo + momz_disk_halo * momz_disk_halo) / rho_disk_halo;
		double const Eint_disk_halo = Eint + eint_halo;
		double const Etot_disk_halo = Eint_disk_halo + Ekin_disk_halo + Emag;

		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::density_index) = rho_disk_halo;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x1Momentum_index) = momx_disk_halo;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x2Momentum_index) = momy_disk_halo;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x3Momentum_index) = momz_disk_halo;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::energy_index) = Etot_disk_halo;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::internalEnergy_index) = Eint_disk_halo;
	});
}

template <> void QuokkaSimulation<AgoraGalaxy>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	amrex::ParmParse const pp("agora_galaxy");
	double magnetic_field_microgauss = 1.0;
	pp.query("magnetic_field_microgauss", magnetic_field_microgauss);

	// disc parameters
	double disk_Rscale_kpc = NAN; // disk scale length
	double disk_zscale_kpc = NAN; // disk scale height
	pp.query("disk_Rscale_kpc", disk_Rscale_kpc);
	pp.query("disk_zscale_kpc", disk_zscale_kpc);
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_Rscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_zscale_kpc));
	const double R_d = disk_Rscale_kpc * (1.0e3 * C::parsec);
	const double z_d = disk_zscale_kpc * (1.0e3 * C::parsec);

	const double B_0 = magnetic_field_microgauss * 1.0e-6 / std::sqrt(4.0 * M_PI);
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// Cartesian coordinates at this face
		amrex::Real const dx_cen = (dir == quokka::direction::x) ? 0.0 : 0.5 * dx[0];
		amrex::Real const dy_cen = (dir == quokka::direction::y) ? 0.0 : 0.5 * dx[1];
		amrex::Real const dz_cen = (dir == quokka::direction::z) ? 0.0 : 0.5 * dx[2];
		amrex::Real const x = prob_lo[0] + (i * dx[0]) + dx_cen;
		amrex::Real const y = prob_lo[1] + (j * dx[1]) + dy_cen;
		amrex::Real const z = prob_lo[2] + (k * dx[2]) + dz_cen;
		amrex::Real const R = std::sqrt(x * x + y * y);

		amrex::Real const B_phi = B_0 * std::exp(-R / R_d) * std::exp(-std::abs(z) / z_d);
		amrex::Real Bx = 0.0;
		amrex::Real By = 0.0;
		amrex::Real const Bz = 0.0;
		if (R > 0.0) {
			Bx = -B_phi * y / R;
			By = B_phi * x / R;
		}

		constexpr int mhd_index = Physics_Indices<AgoraGalaxy>::mhdFirstIndex;
		if (dir == quokka::direction::x) {
			state_fc(i, j, k, mhd_index) = Bx;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, mhd_index) = By;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, mhd_index) = Bz;
		}
	});
}

template <> void QuokkaSimulation<AgoraGalaxy>::createInitialCICParticles()
{
	// read particles from ASCII file
	amrex::ParmParse const pp("agora_galaxy");
	std::string filename;
	pp.query("particle_file", filename);

	amrex::Print() << "\nReading particles from ASCII file " << filename << "...\n";
	CICParticles->SetVerbose(1);
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->InitFromAsciiFile(filename, nreal_extra, nullptr);
	amrex::Print() << "\n";
}

template <> void QuokkaSimulation<AgoraGalaxy>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// geometrical refinement
	// tag cells within the cylinder defined by R < Rmax and abs(z) < zmax
	amrex::ParmParse const pp("agora_galaxy");
	amrex::Real refine_Rmax_kpc = NAN;
	amrex::Real refine_zmax_kpc = NAN;
	pp.query("refine_Rmax_kpc", refine_Rmax_kpc);
	pp.query("refine_zmax_kpc", refine_zmax_kpc);
	const amrex::Real refine_Rmax = refine_Rmax_kpc * (1.0e3 * C::parsec);
	const amrex::Real refine_zmax = refine_zmax_kpc * (1.0e3 * C::parsec);

	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	const auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// NOTE: must check all nodes of the cell!
		// Otherwise, cells that are too big can completely prevent refinement.
		amrex::Real const x0 = prob_lo[0] + (i * dx[0]);
		amrex::Real const y0 = prob_lo[1] + (j * dx[1]);
		amrex::Real const z0 = prob_lo[2] + (k * dx[2]);

		amrex::Real const x1 = prob_lo[0] + ((i + 1) * dx[0]);
		amrex::Real const y1 = prob_lo[1] + ((j + 1) * dx[1]);
		amrex::Real const z1 = prob_lo[2] + ((k + 1) * dx[2]);

		auto tagIfPointInRegion = [=](amrex::Real x, amrex::Real y, amrex::Real z) {
			amrex::Real const R = std::sqrt(x * x + y * y);
			if ((R < refine_Rmax) && (std::abs(z) < refine_zmax)) {
				tag[bx](i, j, k) = amrex::TagBox::SET;
			}
		};

		for (auto const &x : {x0, x1}) {
			for (auto const &y : {y0, y1}) {
				for (auto const &z : {z0, z1}) {
					tagIfPointInRegion(x, y, z);
				}
			}
		}
	});
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<AgoraGalaxy>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "gpot") {
		const int ncomp = ncomp_cc_in;
		auto const &phi_arr = phi[lev].const_arrays();
		auto output = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
		amrex::Gpu::streamSynchronize();
	}

	if (dname == "temperature") {
		const int ncomp = ncomp_cc_in;
		auto tables = resampledTables_.const_tables();
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<AgoraGalaxy>::density_index);
				Real const x1Mom = state(i, j, k, HydroSystem<AgoraGalaxy>::x1Momentum_index);
				Real const x2Mom = state(i, j, k, HydroSystem<AgoraGalaxy>::x2Momentum_index);
				Real const x3Mom = state(i, j, k, HydroSystem<AgoraGalaxy>::x3Momentum_index);
				Real const Egas = state(i, j, k, HydroSystem<AgoraGalaxy>::energy_index);
				Real const Eint = RadSystem<AgoraGalaxy>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
				Real const Tgas = quokka::ResampledCooling::ComputeTgasFromEgas(rho, Eint, tables);
				output(i, j, k, ncomp) = Tgas;
			});
		}
	}

	if (dname == "pressure") {
		const int ncomp = ncomp_cc_in;
		auto const &state_fc = state_new_fc_[lev];
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
			    AMREX_D_DECL(state_fc[0].const_array(iter), state_fc[1].const_array(iter), state_fc[2].const_array(iter))};
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const Pgas = HydroSystem<AgoraGalaxy>::ComputePressure(state, i, j, k, &cons_fc);
				output(i, j, k, ncomp) = Pgas / C::k_B;
			});
		}
	}

	if (dname == "radius_sph") {
		const int ncomp = ncomp_cc_in;
		auto const geom_data = geom[lev].data();
		auto const prob_lo = geom_data.ProbLo();
		auto const dx = geom_data.CellSize();
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
				const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
				const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
				const amrex::Real r_cm = std::sqrt(x * x + y * y + z * z);
				output(i, j, k, ncomp) = r_cm / 3.08567758e21;
			});
		}
	}

	if (dname == "bfield_strength") {
		static_assert(Physics_Traits<AgoraGalaxy>::is_mhd_enabled, "bfield_strength requires MHD to be enabled.");
		const int ncomp = ncomp_cc_in;
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &bx_fc = state_new_fc_[lev][0].const_array(iter);
			auto const &by_fc = state_new_fc_[lev][1].const_array(iter);
			auto const &bz_fc = state_new_fc_[lev][2].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real bx_cc = 0.5 * (bx_fc(i, j, k, 0) + bx_fc(i + 1, j, k, 0));
				const amrex::Real by_cc = 0.5 * (by_fc(i, j, k, 0) + by_fc(i, j + 1, k, 0));
				const amrex::Real bz_cc = 0.5 * (bz_fc(i, j, k, 0) + bz_fc(i, j, k + 1, 0));
				const amrex::Real b_code = std::sqrt(bx_cc * bx_cc + by_cc * by_cc + bz_cc * bz_cc);
				output(i, j, k, ncomp) = b_code * std::sqrt(4.0 * M_PI) * 1.0e6;
			});
		}
	}

	if (dname == "radial_velocity") {
		const int ncomp = ncomp_cc_in;
		auto const geom_data = geom[lev].data();
		auto const prob_lo = geom_data.ProbLo();
		auto const dx = geom_data.CellSize();
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real rho = state(i, j, k, HydroSystem<AgoraGalaxy>::density_index);
				const amrex::Real vx = state(i, j, k, HydroSystem<AgoraGalaxy>::x1Momentum_index) / rho;
				const amrex::Real vy = state(i, j, k, HydroSystem<AgoraGalaxy>::x2Momentum_index) / rho;
				const amrex::Real vz = state(i, j, k, HydroSystem<AgoraGalaxy>::x3Momentum_index) / rho;
				const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
				const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
				const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
				const amrex::Real r_cm = std::sqrt(x * x + y * y + z * z);
				// Radial velocity follows the input halo table sign convention (implicit minus in the table).
				output(i, j, k, ncomp) = (r_cm > 0.0) ? ((x * vx + y * vy + z * vz) / r_cm) / 1.0e5 : 0.0;
			});
		}
	}

	if (dname == "circular_velocity") {
		const int ncomp = ncomp_cc_in;
		auto const geom_data = geom[lev].data();
		auto const prob_lo = geom_data.ProbLo();
		auto const dx = geom_data.CellSize();
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real rho = state(i, j, k, HydroSystem<AgoraGalaxy>::density_index);
				const amrex::Real vx = state(i, j, k, HydroSystem<AgoraGalaxy>::x1Momentum_index) / rho;
				const amrex::Real vy = state(i, j, k, HydroSystem<AgoraGalaxy>::x2Momentum_index) / rho;
				const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
				const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
				const amrex::Real r_cyl = std::sqrt(x * x + y * y);
				output(i, j, k, ncomp) = (r_cyl > 0.0) ? ((x * vy - y * vx) / r_cyl) / 1.0e5 : 0.0;
			});
		}
	}
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<AgoraGalaxy>(quokka::BCType::reflecting);

	const int nvars_fc = Physics_Indices<AgoraGalaxy>::nvarTotal_fc;
	const int nvars_per_dim_fc = Physics_Indices<AgoraGalaxy>::nvarPerDim_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		int const component_dir = (nvars_per_dim_fc > 0) ? (icomp / nvars_per_dim_fc) : 0;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			int const bc_type = (component_dir == idim) ? amrex::BCType::reflect_even : amrex::BCType::reflect_odd;
			BCs_fc[icomp].setLo(idim, bc_type);
			BCs_fc[icomp].setHi(idim, bc_type);
		}
	}

	// Problem initialization
	QuokkaSimulation<AgoraGalaxy> sim(BCs_cc, BCs_fc);

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	const int status = 0;
	return status;
}
