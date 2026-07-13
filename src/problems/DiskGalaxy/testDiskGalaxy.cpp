//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testDiskGalaxy.cpp
/// \brief Defines a simulation using disk galaxy initial conditions.
///

#include <cmath>
#include <optional>

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArrayBase.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelReduce.H"
#include "AMReX_Parser.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_Reduce.H"
#include "AMReX_iMultiFab.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "math/quadrature.hpp"
#include "math/spherical_geometry.hpp"
#include "particles/particle_types.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "util/DataTable.hpp"

namespace
{
constexpr double keV_in_ergs = 1000.0 * C::ev2erg; // ergs == 1 keV
constexpr double seconds_per_year = 3.15576e7;
} // namespace

struct DiskGalaxy {
};

static_assert(AMREX_SPACEDIM == 3, "DiskGalaxy problem requires AMREX_SPACEDIM == 3.");

template <> struct quokka::EOS_Traits<DiskGalaxy> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = 0.6 * C::m_u;
	using EOSBackend = quokka::EOSTabulated<DiskGalaxy>;
};

template <> struct HydroSystem_Traits<DiskGalaxy> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<DiskGalaxy> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int numPassiveScalars = numMassScalars + 1; // number of passive scalars
};

template <> struct Particle_Traits<DiskGalaxy> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC | ParticleSwitch::StochasticStellarPop;
};

template <> struct SimulationData<DiskGalaxy> {
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

template <> void QuokkaSimulation<DiskGalaxy>::preCalculateInitialConditions()
{
	// 1. read in circular velocity table "vcirc.dat"
	// get circular velocity profile filename from ParmParse
	amrex::ParmParse const pp("disk_galaxy");
	std::string filename;
	pp.query("vcirc_file", filename);

	auto halo_table = quokka::DataTable<1, 4, quokka::OutOfBounds::clamp>::CSVReader(filename, quokka::TransformType::linear);
	auto const halo_table_const = halo_table.const_tables_host();
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(halo_table_const.sizes[0] > 0, "disk_galaxy.vcirc_file contained no numeric rows.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(halo_table_const.spacing_types[0] == quokka::TransformType::linear,
					 "disk_galaxy.vcirc_file must use linear spacing for the radius coordinate.");

	// 2. copy data to simData_.radius and simData_.vcirc
	const auto N = static_cast<size_t>(halo_table_const.sizes[0]);
	userData_.radius.resize(N);
	userData_.vcirc.resize(N);
	userData_.rho_halo.resize(N);
	userData_.velr_halo.resize(N);
	userData_.temp_halo.resize(N);

	const double length_unit = 1.0e3 * C::parsec; // kpc
	const double vel_unit = 1.0e5;		      // km/s
	for (size_t i = 0; i < N; ++i) {
		amrex::Real const radius = halo_table_const.coord_min[0] + static_cast<amrex::Real>(i) * halo_table_const.dcoord[0];
		userData_.radius[i] = radius * length_unit;
		userData_.vcirc[i] = halo_table_const.dataViewArrays[0](static_cast<int>(i)) * vel_unit;
		userData_.rho_halo[i] = halo_table_const.dataViewArrays[1](static_cast<int>(i));
		userData_.velr_halo[i] = halo_table_const.dataViewArrays[2](static_cast<int>(i));
		userData_.temp_halo[i] = halo_table_const.dataViewArrays[3](static_cast<int>(i));
	}

	// save min/max radii
	userData_.r_inner = halo_table_const.coord_min[0] * length_unit;
	userData_.vcirc_inner = halo_table_const.dataViewArrays[0](0) * vel_unit;
	userData_.rho_inner = halo_table_const.dataViewArrays[1](0);
	userData_.velr_inner = halo_table_const.dataViewArrays[2](0);
	userData_.temp_inner = halo_table_const.dataViewArrays[3](0);

	userData_.r_outer = halo_table_const.coord_max[0] * length_unit;
	userData_.vcirc_outer = halo_table_const.dataViewArrays[0](static_cast<int>(N - 1)) * vel_unit;
	userData_.rho_outer = halo_table_const.dataViewArrays[1](static_cast<int>(N - 1));
	userData_.velr_outer = halo_table_const.dataViewArrays[2](static_cast<int>(N - 1));
	userData_.temp_outer = halo_table_const.dataViewArrays[3](static_cast<int>(N - 1));

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
			amrex::Abort("disk_galaxy.halo_vphi_expr: device parser executor is null after compile<3>()");
		}
#endif
		userData_.haloVphiParser.reset();
	} else {
		userData_.haloVphiParser.reset();
		userData_.haloVphiParserExe.reset();
	}
}

template <> void QuokkaSimulation<DiskGalaxy>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// read parameters
	//
	amrex::ParmParse const pp("disk_galaxy");

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
	double initial_scalar_density = NAN; // scalar density at cgs units (cm^-3)
	pp.query("disk_gas_mass_Msun", disk_gas_mass_Msun);
	pp.query("disk_Rscale_kpc", disk_Rscale_kpc);
	pp.query("disk_zscale_kpc", disk_zscale_kpc);
	pp.query("disk_temperature", T_disk);
	pp.query("disk_perturb_amplitude", disk_perturb_amplitude);
	pp.query("disk_perturb_Rmax_kpc", disk_perturb_Rmax_kpc);
	pp.query("initial_scalar_density", initial_scalar_density);
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_gas_mass_Msun));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_Rscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_zscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(T_disk));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_perturb_amplitude));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_perturb_Rmax_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(initial_scalar_density));

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
			amrex::Abort("disk_galaxy.halo_vphi_expr: parser executor is missing after compile<3>()");
		}
	}

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// particles.scalar_yield_per_SN must be set as well, and it should be greater than (initial_scalar_density * (128 pc)^3),
	// so that the SN ejected metal density in SN remnant is greater than the background density.
	amrex::ParmParse const pp_particles("particles");
	double scalar_yield_per_SN = NAN;
	pp_particles.query("scalar_yield_per_SN", scalar_yield_per_SN);
	AMREX_ALWAYS_ASSERT(!std::isnan(scalar_yield_per_SN));
	const Real SNR_volume = std::pow(128.0 * C::parsec, 3);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(scalar_yield_per_SN > initial_scalar_density * SNR_volume,
					 "particles.scalar_yield_per_SN must be greater than (initial_scalar_density * (128 pc)^3)");

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
		constexpr double gamma_gas = quokka::EOS_Traits<DiskGalaxy>::gamma;
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

		state_cc(i, j, k, HydroSystem<DiskGalaxy>::density_index) = rho_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy>::x1Momentum_index) = momx_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy>::x2Momentum_index) = momy_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy>::x3Momentum_index) = momz_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy>::energy_index) = Etot_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy>::internalEnergy_index) = Eint_disk_halo;

		// first capture on device
		const auto initial_scalar_density_d = initial_scalar_density;

		// Initialize passive scalar field
		if constexpr (Physics_Traits<DiskGalaxy>::numPassiveScalars > 0) {
			state_cc(i, j, k, HydroSystem<DiskGalaxy>::scalar0_index) = initial_scalar_density_d;
		}
	});
}

template <> void QuokkaSimulation<DiskGalaxy>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	amrex::ParmParse const pp("disk_galaxy");
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

		constexpr int mhd_index = Physics_Indices<DiskGalaxy>::mhdFirstIndex;
		if (dir == quokka::direction::x) {
			state_fc(i, j, k, mhd_index) = Bx;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, mhd_index) = By;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, mhd_index) = Bz;
		}
	});
}

template <> void QuokkaSimulation<DiskGalaxy>::createInitialCICParticles()
{
	// read particles from ASCII file
	amrex::ParmParse const pp("disk_galaxy");
	std::string filename;
	pp.query("particle_file", filename);

	amrex::Print() << "\nReading particles from ASCII file " << filename << "...\n";
	CICParticles->SetVerbose(1);
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->InitFromAsciiFile(filename, nreal_extra, nullptr);
	amrex::Print() << "\n";
}

template <> void QuokkaSimulation<DiskGalaxy>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// geometrical refinement
	// tag cells within the cylinder defined by R < Rmax and abs(z) < zmax
	amrex::ParmParse const pp("disk_galaxy");
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

template <>
void QuokkaSimulation<DiskGalaxy>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
						     amrex::MultiFab const &state_cc, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) const
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
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_cc.const_array(iter);
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
			    AMREX_D_DECL(state_fc[0].const_array(iter), state_fc[1].const_array(iter), state_fc[2].const_array(iter))};
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<DiskGalaxy>::density_index);
				Real const Eint = HydroSystem<DiskGalaxy>::ComputeInternalEnergy(state, i, j, k, &cons_fc);
				Real const Tgas = quokka::EOS<DiskGalaxy>::ComputeTgasFromEint(rho, Eint);
				output(i, j, k, ncomp) = Tgas;
			});
		}
	}

	if (dname == "pressure") {
		const int ncomp = ncomp_cc_in;
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_cc.const_array(iter);
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
			    AMREX_D_DECL(state_fc[0].const_array(iter), state_fc[1].const_array(iter), state_fc[2].const_array(iter))};
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const Pgas = HydroSystem<DiskGalaxy>::ComputePressure(state, i, j, k, &cons_fc);
				output(i, j, k, ncomp) = Pgas / C::k_B;
			});
		}
	}

	if (dname == "entropy") {
		const int ncomp = ncomp_cc_in;
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_cc.const_array(iter);
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
			    AMREX_D_DECL(state_fc[0].const_array(iter), state_fc[1].const_array(iter), state_fc[2].const_array(iter))};
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<DiskGalaxy>::density_index);
				Real const Eint = HydroSystem<DiskGalaxy>::ComputeInternalEnergy(state, i, j, k, &cons_fc);
				Real const K_cgs = quokka::EOS<DiskGalaxy>::ComputeEntropyFromRhoEint(rho, Eint);
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
		static_assert(Physics_Traits<DiskGalaxy>::is_mhd_enabled, "bfield_strength requires MHD to be enabled.");
		const int ncomp = ncomp_cc_in;
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &bx_fc = state_fc[0].const_array(iter);
			auto const &by_fc = state_fc[1].const_array(iter);
			auto const &bz_fc = state_fc[2].const_array(iter);
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
			auto const &state = state_cc.const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real rho = state(i, j, k, HydroSystem<DiskGalaxy>::density_index);
				const amrex::Real vx = state(i, j, k, HydroSystem<DiskGalaxy>::x1Momentum_index) / rho;
				const amrex::Real vy = state(i, j, k, HydroSystem<DiskGalaxy>::x2Momentum_index) / rho;
				const amrex::Real vz = state(i, j, k, HydroSystem<DiskGalaxy>::x3Momentum_index) / rho;
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
			auto const &state = state_cc.const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real rho = state(i, j, k, HydroSystem<DiskGalaxy>::density_index);
				const amrex::Real vx = state(i, j, k, HydroSystem<DiskGalaxy>::x1Momentum_index) / rho;
				const amrex::Real vy = state(i, j, k, HydroSystem<DiskGalaxy>::x2Momentum_index) / rho;
				const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
				const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
				const amrex::Real r_cyl = std::sqrt(x * x + y * y);
				output(i, j, k, ncomp) = (r_cyl > 0.0) ? ((x * vy - y * vx) / r_cyl) / 1.0e5 : 0.0;
			});
		}
	}
}

template <> auto QuokkaSimulation<DiskGalaxy>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	std::map<std::string, amrex::Real> stats;

	const amrex::Real time_now = tNew_[0];
	const amrex::Real time_window = 1.0e6 * seconds_per_year;
	const amrex::Real sfr_g_per_s = particleRegister_.computeSfrAveragedOverTime(time_now, time_window);
	stats["sfr_1myr"] = (sfr_g_per_s / C::M_solar) * seconds_per_year;

	amrex::ParmParse const pp("disk_galaxy");
	amrex::Real refine_Rmax_kpc = NAN;
	amrex::Real refine_zmax_kpc = NAN;
	amrex::Real flux_sphere_radius_kpc = NAN;
	pp.query("refine_Rmax_kpc", refine_Rmax_kpc);
	pp.query("refine_zmax_kpc", refine_zmax_kpc);
	pp.query("flux_sphere_radius_kpc", flux_sphere_radius_kpc);
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
			const amrex::Real rho = state[bx](i, j, k, HydroSystem<DiskGalaxy>::density_index);
			result[bx](i, j, k) = (R < refine_Rmax && std::abs(z) < refine_zmax) ? rho : 0.0;
		});
	}
	amrex::Gpu::streamSynchronize();

	const amrex::Real disk_mass_refine = amrex::volumeWeightedSum(amrex::GetVecOfConstPtrs(refine_mask), 0, geom, ref_ratio);
	stats["disk_mass_refine_region"] = disk_mass_refine / C::M_solar;

	const amrex::Real cold_mass =
	    computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state,
						       std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const &state_fc) noexcept {
		    const Real rho = state(i, j, k, HydroSystem<DiskGalaxy>::density_index);
		    const Real Eint = HydroSystem<DiskGalaxy>::ComputeInternalEnergy(state, i, j, k, &state_fc);
		    const Real Tgas = quokka::EOS<DiskGalaxy>::ComputeTgasFromEint(rho, Eint);
		    return (Tgas < 1.0e4) ? rho : 0.0;
	    });
	stats["mass_T_lt_1e4"] = cold_mass / C::M_solar;

	if (!std::isnan(flux_sphere_radius_kpc)) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(flux_sphere_radius_kpc > 0.0, "disk_galaxy.flux_sphere_radius_kpc must be > 0.");
		const amrex::Real flux_sphere_radius = flux_sphere_radius_kpc * (1.0e3 * C::parsec);
		stats["flux_sphere_radius_kpc"] = flux_sphere_radius_kpc;

		amrex::Vector<amrex::iMultiFab> flux_mask;
		flux_mask.resize(finest_level + 1);
		for (int lev = 0; lev <= finest_level; ++lev) {
			if (lev == finest_level) {
				flux_mask[lev].define(boxArray(lev), DistributionMap(lev), 1, 0);
				flux_mask[lev].setVal(1);
			} else {
				flux_mask[lev] = amrex::makeFineMask(state_new_cc_[lev], state_new_cc_[lev + 1], amrex::IntVect(0), ref_ratio[lev],
								     geom[lev].periodicity(), 1, 0);
			}
		}

		amrex::Real mass_flux_sphere = 0.0;
		amrex::Real hydro_energy_flux_sphere = 0.0;
		amrex::Real mhd_energy_flux_sphere = 0.0;
		amrex::Real passive_scalar_flux_sphere = 0.0;
		for (int lev = 0; lev <= finest_level; ++lev) {
			const auto prob_lo = geom[lev].ProbLoArray();
			const auto dx = geom[lev].CellSizeArray();
			auto const &state = state_new_cc_[lev].const_arrays();
			auto const &state_fc = state_new_fc_[lev];
			auto const &state_fc_x = state_fc[0].const_arrays();
			auto const &state_fc_y = state_fc[1].const_arrays();
			auto const &state_fc_z = state_fc[2].const_arrays();
			auto const &mask = flux_mask[lev].const_arrays();

			auto const level_flux = amrex::ParReduce(
			    amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>{},
			    amrex::TypeList<amrex::Real, amrex::Real, amrex::Real, amrex::Real>{}, state_new_cc_[lev], amrex::IntVect(0),
			    [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
				    if (mask[bx](i, j, k) == 0) {
					    return {0.0, 0.0, 0.0, 0.0};
				    }

				    const amrex::Real x0 = prob_lo[0] + static_cast<amrex::Real>(i) * dx[0];
				    const amrex::Real y0 = prob_lo[1] + static_cast<amrex::Real>(j) * dx[1];
				    const amrex::Real z0 = prob_lo[2] + static_cast<amrex::Real>(k) * dx[2];
				    const amrex::Real x1 = x0 + dx[0];
				    const amrex::Real y1 = y0 + dx[1];
				    const amrex::Real z1 = z0 + dx[2];

				    const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
				    const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
				    const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
				    const amrex::Real r = std::sqrt(x * x + y * y + z * z);

				    const amrex::Real rho = state[bx](i, j, k, HydroSystem<DiskGalaxy>::density_index);
				    if (r <= 0.0 || rho <= 0.0) {
					    return {0.0, 0.0, 0.0, 0.0};
				    }

				    const amrex::Real momx = state[bx](i, j, k, HydroSystem<DiskGalaxy>::x1Momentum_index);
				    const amrex::Real momy = state[bx](i, j, k, HydroSystem<DiskGalaxy>::x2Momentum_index);
				    const amrex::Real momz = state[bx](i, j, k, HydroSystem<DiskGalaxy>::x3Momentum_index);
				    const amrex::Real vx = momx / rho;
				    const amrex::Real vy = momy / rho;
				    const amrex::Real vz = momz / rho;
				    const amrex::Real vr = (x * momx + y * momy + z * momz) / (rho * r);
				    const amrex::Real rhat_x = x / r;
				    const amrex::Real rhat_y = y / r;
				    const amrex::Real rhat_z = z / r;

				    const amrex::Real mass_flux_density = rho * vr;
				    const amrex::Real energy_density = state[bx](i, j, k, HydroSystem<DiskGalaxy>::energy_index);
				    const amrex::Real scalar_density = state[bx](i, j, k, HydroSystem<DiskGalaxy>::scalar0_index);
				    std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
					AMREX_D_DECL(state_fc_x[bx], state_fc_y[bx], state_fc_z[bx])};
				    const amrex::Real Pgas = HydroSystem<DiskGalaxy>::ComputePressure(state[bx], i, j, k, &cons_fc);
				    const amrex::Real Emag = HydroSystem<DiskGalaxy>::ComputeMagneticEnergy(i, j, k, &cons_fc);
				    const amrex::Real Ehydro = energy_density - Emag;

				    const amrex::Real bx1_m = cons_fc[0](i, j, k, Physics_Indices<DiskGalaxy>::mhdFirstIndex);
				    const amrex::Real bx1_p = cons_fc[0](i + 1, j, k, Physics_Indices<DiskGalaxy>::mhdFirstIndex);
				    const amrex::Real bx2_m = cons_fc[1](i, j, k, Physics_Indices<DiskGalaxy>::mhdFirstIndex);
				    const amrex::Real bx2_p = cons_fc[1](i, j + 1, k, Physics_Indices<DiskGalaxy>::mhdFirstIndex);
				    const amrex::Real bx3_m = cons_fc[2](i, j, k, Physics_Indices<DiskGalaxy>::mhdFirstIndex);
				    const amrex::Real bx3_p = cons_fc[2](i, j, k + 1, Physics_Indices<DiskGalaxy>::mhdFirstIndex);
				    const amrex::Real Bx = 0.5 * (bx1_m + bx1_p);
				    const amrex::Real By = 0.5 * (bx2_m + bx2_p);
				    const amrex::Real Bz = 0.5 * (bx3_m + bx3_p);
				    const amrex::Real Bdotv = vx * Bx + vy * By + vz * Bz;
				    const amrex::Real Br = rhat_x * Bx + rhat_y * By + rhat_z * Bz;

				    const amrex::Real hydro_energy_flux_density = (Ehydro + Pgas) * vr;
				    const amrex::Real mhd_energy_flux_density = (energy_density + Pgas + Emag) * vr - Bdotv * Br;
				    const amrex::Real area = quokka::math::sphericalSectionAreaInCell(flux_sphere_radius, x0, x1, y0, y1, z0, z1);
				    if (area <= 0.0) {
					    return {0.0, 0.0, 0.0, 0.0};
				    }

				    return {mass_flux_density * area, hydro_energy_flux_density * area, mhd_energy_flux_density * area,
					    (scalar_density * vr) * area};
			    });

			mass_flux_sphere += amrex::get<0>(level_flux);
			hydro_energy_flux_sphere += amrex::get<1>(level_flux);
			mhd_energy_flux_sphere += amrex::get<2>(level_flux);
			passive_scalar_flux_sphere += amrex::get<3>(level_flux);
		}

		// MPI reduction
		std::array<Real, 4> fluxes_sphere = {mass_flux_sphere, hydro_energy_flux_sphere, mhd_energy_flux_sphere, passive_scalar_flux_sphere};
		amrex::ParallelAllReduce::Sum(fluxes_sphere.data(), fluxes_sphere.size(), amrex::ParallelContext::CommunicatorSub());
		mass_flux_sphere = fluxes_sphere[0];
		hydro_energy_flux_sphere = fluxes_sphere[1];
		mhd_energy_flux_sphere = fluxes_sphere[2];
		passive_scalar_flux_sphere = fluxes_sphere[3];

		stats["mass_flux_sphere"] = mass_flux_sphere;
		stats["hydro_energy_flux_sphere"] = hydro_energy_flux_sphere;
		stats["mhd_energy_flux_sphere"] = mhd_energy_flux_sphere;
		stats["passive_scalar_flux_sphere"] = passive_scalar_flux_sphere;
	}

	const amrex::Real stellar_mass_at_birth = particleRegister_.computeTotalStellarMassAtBirth();
	stats["stellar_mass_at_birth"] = stellar_mass_at_birth / C::M_solar;
	stats["sn_count_cumulative"] = sn_count_cumulative_;

	return stats;
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<DiskGalaxy>(quokka::BCType::reflecting);

	const int nvars_fc = Physics_Indices<DiskGalaxy>::nvarTotal_fc;
	const int nvars_per_dim_fc = Physics_Indices<DiskGalaxy>::nvarPerDim_fc;
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
	QuokkaSimulation<DiskGalaxy> sim(BCs_cc, BCs_fc);

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	const int status = 0;
	return status;
}
