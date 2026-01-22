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
#include <optional>
#include <sstream>

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArrayBase.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Parser.H"
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
#include "util/DataTable.hpp"

namespace
{
constexpr double keV_in_ergs = 1000.0 * C::ev2erg; // ergs == 1 keV
constexpr double seconds_per_year = 3.15576e7;
} // namespace

struct AgoraGalaxy {
};

static_assert(AMREX_SPACEDIM == 3, "DiskGalaxy problem requires AMREX_SPACEDIM == 3.");

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

	std::string haloVphiExpr;
	bool useHaloVphiParser = false;
	std::optional<amrex::Parser> haloVphiParser;
	std::optional<amrex::ParserExecutor<3>> haloVphiParserExe;
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

	// optional halo v_phi expression (variables: x, y, z)
	userData_.haloVphiExpr.clear();
	pp.query("halo_vphi_expr", userData_.haloVphiExpr);
	userData_.useHaloVphiParser = !userData_.haloVphiExpr.empty();
	if (userData_.useHaloVphiParser) {
		userData_.haloVphiParser.emplace(userData_.haloVphiExpr);
		userData_.haloVphiParser->registerVariables({"x", "y", "z"});
		userData_.haloVphiParserExe = userData_.haloVphiParser->compile<3>();
#ifdef AMREX_USE_GPU
		if (userData_.haloVphiParserExe->m_device_executor == nullptr) {
			amrex::Abort("agora_galaxy.halo_vphi_expr: device parser executor is null after compile<3>()");
		}
#endif
		userData_.haloVphiParser.reset();
	} else {
		userData_.haloVphiParser.reset();
		userData_.haloVphiParserExe.reset();
	}
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
	const bool use_halo_vphi_parser = userData_.useHaloVphiParser;
	amrex::ParserExecutor<3> halo_vphi_parser{};
	if (use_halo_vphi_parser) {
		if (userData_.haloVphiParserExe.has_value()) {
			halo_vphi_parser = *userData_.haloVphiParserExe;
		} else {
			amrex::Abort("agora_galaxy.halo_vphi_expr: parser executor is missing after compile<3>()");
		}
	}

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

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

		auto rhoDisk_exact = [rho_0, R_d, z_d, disk_perturb_amplitude, R_max_perturb](double x, double y, double z) {
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const theta = std::atan2(x, y);
			double const drho_over_rho = disk_perturb_amplitude * jn(2, 5.1356 * R / R_max_perturb) * std::sin(2.0 * theta);
			return rho_0 * std::exp(-R / R_d) * std::exp(-std::abs(z) / z_d) * (1.0 + drho_over_rho);
		};

		// compute momenta profiles
		auto vphiHalo_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			if (use_halo_vphi_parser) {
				return halo_vphi_parser(x, y, z);
			}
			return 0.0;
		};

		auto velx_exact = [velHalo, vphiHalo_exact](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const vr_component = (r > 0.0) ? (velHalo(r) * x / r) : 0.0;
			double const vphi_component = (R > 0.0) ? (-vphiHalo_exact(x, y, z) * y / R) : 0.0;
			return vr_component + vphi_component; // vx
		};

		auto vely_exact = [velHalo, vphiHalo_exact](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const vr_component = (r > 0.0) ? (velHalo(r) * y / r) : 0.0;
			double const vphi_component = (R > 0.0) ? (vphiHalo_exact(x, y, z) * x / R) : 0.0;
			return vr_component + vphi_component; // vy
		};

		auto velz_exact = [velHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return (r > 0.0) ? (velHalo(r) * z / r) : 0.0; // vz
		};

		// integrate profiles over cell volume
		const double cell_vol = dx[0] * dx[1] * dx[2];
		constexpr double gamma_gas = quokka::EOS_Traits<AgoraGalaxy>::gamma;
		constexpr double mu = 0.61;

		auto rho_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) { return rhoDisk_exact(x, y, z) + rhoHalo_exact(x, y, z); };

		auto momx_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			const double rho_disk_local = rhoDisk_exact(x, y, z);
			const double rho_halo_local = rhoHalo_exact(x, y, z);
			return rho_disk_local * vx_exact(x, y, z) + rho_halo_local * velx_exact(x, y, z);
		};

		auto momy_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			const double rho_disk_local = rhoDisk_exact(x, y, z);
			const double rho_halo_local = rhoHalo_exact(x, y, z);
			return rho_disk_local * vy_exact(x, y, z) + rho_halo_local * vely_exact(x, y, z);
		};

		auto momz_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			const double rho_halo_local = rhoHalo_exact(x, y, z);
			return rho_halo_local * velz_exact(x, y, z);
		};

		auto eint_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			const double rho_disk_local = rhoDisk_exact(x, y, z);
			const double rho_halo_local = rhoHalo_exact(x, y, z);
			const double temp_halo_local = tempHalo_exact(x, y, z);
			const double eint_disk_local = (rho_disk_local > 0.0) ? (rho_disk_local * C::k_B * T_disk / (mu * C::m_p * (gamma_gas - 1.0))) : 0.0;
			const double eint_halo_local =
			    (rho_halo_local > 0.0) ? (rho_halo_local * C::k_B * temp_halo_local / (mu * C::m_p * (gamma_gas - 1.0))) : 0.0;
			return eint_disk_local + eint_halo_local;
		};

		const double rho = quad_3d(rho_total_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		AMREX_ALWAYS_ASSERT(rho > 0.0);
		const double momx = quad_3d(momx_total_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double momy = quad_3d(momy_total_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double momz = quad_3d(momz_total_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double Eint = quad_3d(eint_total_exact, x0, x1, y0, y1, z0, z1) / cell_vol;

		// Add up disk and halo contributions
		double const rho_disk_halo = rho;
		double const momx_disk_halo = momx;
		double const momy_disk_halo = momy;
		double const momz_disk_halo = momz;
		double const Ekin_disk_halo =
		    0.5 * (momx_disk_halo * momx_disk_halo + momy_disk_halo * momy_disk_halo + momz_disk_halo * momz_disk_halo) / rho_disk_halo;
		double const Eint_disk_halo = Eint;
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

	if (dname == "entropy") {
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
				Real const K_cgs = quokka::ResampledCooling::ComputeEntropyFromRhoEint(rho, Eint, tables);
				output(i, j, k, ncomp) = K_cgs / keV_in_ergs;
			});
		}
	}

	if (dname == "radius_sph") {
		const int ncomp = ncomp_cc_in;
		const auto prob_lo = geom[lev].ProbLoArray();
		const auto dx = geom[lev].CellSizeArray();
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
		const auto prob_lo = geom[lev].ProbLoArray();
		const auto dx = geom[lev].CellSizeArray();
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
		const auto prob_lo = geom[lev].ProbLoArray();
		const auto dx = geom[lev].CellSizeArray();
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

template <> auto QuokkaSimulation<AgoraGalaxy>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	std::map<std::string, amrex::Real> stats;

	const amrex::Real time_now = tNew_[0];
	const amrex::Real time_window = 1.0e6 * seconds_per_year;
	const amrex::Real sfr_g_per_s = particleRegister_.computeSfrAveragedOverTime(time_now, time_window);
	stats["sfr_1myr"] = (sfr_g_per_s / C::M_solar) * seconds_per_year;

	amrex::ParmParse const pp("agora_galaxy");
	amrex::Real refine_Rmax_kpc = NAN;
	amrex::Real refine_zmax_kpc = NAN;
	pp.query("refine_Rmax_kpc", refine_Rmax_kpc);
	pp.query("refine_zmax_kpc", refine_zmax_kpc);
	AMREX_ALWAYS_ASSERT(!std::isnan(refine_Rmax_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(refine_zmax_kpc));
	const amrex::Real refine_Rmax = refine_Rmax_kpc * (1.0e3 * C::parsec);
	const amrex::Real refine_zmax = refine_zmax_kpc * (1.0e3 * C::parsec);

	amrex::Vector<amrex::MultiFab> refine_mask;
	refine_mask.resize(finest_level + 1);
	for (int lev = 0; lev <= finest_level; ++lev) {
		refine_mask[lev].define(boxArray(lev), DistributionMap(lev), 1, 0);
	}

	for (int lev = 0; lev <= finest_level; ++lev) {
		const auto prob_lo = geom[lev].ProbLoArray();
		const auto dx = geom[lev].CellSizeArray();
		auto const &state = state_new_cc_[lev].const_arrays();
		auto const &result = refine_mask[lev].arrays();
		amrex::ParallelFor(refine_mask[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
			const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
			const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
			const amrex::Real R = std::sqrt(x * x + y * y);
			const amrex::Real rho = state[bx](i, j, k, HydroSystem<AgoraGalaxy>::density_index);
			result[bx](i, j, k) = (R < refine_Rmax && std::abs(z) < refine_zmax) ? rho : 0.0;
		});
	}
	amrex::Gpu::streamSynchronize();

	const amrex::Real disk_mass_refine = amrex::volumeWeightedSum(amrex::GetVecOfConstPtrs(refine_mask), 0, geom, ref_ratio);
	stats["disk_mass_refine_region"] = disk_mass_refine / C::M_solar;

	auto tables = resampledTables_.const_tables();
	const amrex::Real cold_mass = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		const Real rho = state(i, j, k, HydroSystem<AgoraGalaxy>::density_index);
		const Real x1Mom = state(i, j, k, HydroSystem<AgoraGalaxy>::x1Momentum_index);
		const Real x2Mom = state(i, j, k, HydroSystem<AgoraGalaxy>::x2Momentum_index);
		const Real x3Mom = state(i, j, k, HydroSystem<AgoraGalaxy>::x3Momentum_index);
		const Real Egas = state(i, j, k, HydroSystem<AgoraGalaxy>::energy_index);
		const Real Eint = RadSystem<AgoraGalaxy>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
		const Real Tgas = quokka::ResampledCooling::ComputeTgasFromEgas(rho, Eint, tables);
		return (Tgas < 1.0e4) ? rho : 0.0;
	});
	stats["mass_T_lt_1e4"] = cold_mass / C::M_solar;

	const amrex::Real stellar_mass = particleRegister_.computeTotalStellarMass();
	stats["stellar_mass"] = stellar_mass / C::M_solar;
	stats["sn_count"] = sn_count_;

	return stats;
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
