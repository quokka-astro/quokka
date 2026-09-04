//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2026 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testKeplerianDisk.cpp
/// \brief Defines an isolated self-gravitating cooling gas disk with star formation in a fixed spherical gravitational potential.

#include <cmath>
#include <optional>

#include "AMReX_BLassert.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Parser.H"
#include "AMReX_REAL.H"
#include "AMReX_TagBox.H"

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
constexpr double keV_in_ergs = 1000.0 * C::ev2erg;
constexpr double seconds_per_year = 3.15576e7;
} // namespace

struct KeplerianDisk {
};

static_assert(AMREX_SPACEDIM == 3, "KeplerianDisk problem requires AMREX_SPACEDIM == 3.");

template <> struct quokka::EOS_Traits<KeplerianDisk> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = 0.6 * C::m_u;
	using EOSBackend = quokka::EOSTabulated<KeplerianDisk>;
};

template <> struct HydroSystem_Traits<KeplerianDisk> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<KeplerianDisk> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr int numPassiveScalars = 1;
};

template <> struct Particle_Traits<KeplerianDisk> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct SimulationData<KeplerianDisk> {
	amrex::Real r_inner{};
	amrex::Real r_outer{};
	amrex::Real vcirc_inner{};
	amrex::Real vcirc_outer{};
	amrex::Real potential_inner{};
	amrex::Real potential_outer{};
	amrex::Real rho_inner{};
	amrex::Real rho_outer{};
	amrex::Real velr_inner{};
	amrex::Real velr_outer{};
	amrex::Real temp_inner{};
	amrex::Real temp_outer{};
	amrex::Gpu::PinnedVector<amrex::Real> radius;
	amrex::Gpu::PinnedVector<amrex::Real> vcirc;
	amrex::Gpu::PinnedVector<amrex::Real> potential;
	amrex::Gpu::PinnedVector<amrex::Real> rho_halo;
	amrex::Gpu::PinnedVector<amrex::Real> velr_halo;
	amrex::Gpu::PinnedVector<amrex::Real> temp_halo;
	std::string haloVphiExpr;
	bool useHaloVphiParser = false;
	std::optional<amrex::Parser> haloVphiParser;
	std::optional<amrex::ParserExecutor<3>> haloVphiParserExe;
};

template <> void QuokkaSimulation<KeplerianDisk>::preCalculateInitialConditions()
{
	amrex::ParmParse const pp("disk_galaxy");
	std::string filename;
	pp.get("vcirc_file", filename);

	// Use the same four-output table format as DiskGalaxy for the circular
	// velocity and hot-halo density, radial velocity, and temperature profiles.
	auto rotation_curve = quokka::DataTable<1, 4, quokka::OutOfBounds::clamp>::CSVReader(filename, quokka::TransformType::linear);
	auto const rotation_curve_host = rotation_curve.const_tables_host();
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rotation_curve_host.sizes[0] > 1, "disk_galaxy.vcirc_file must contain at least two radii.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rotation_curve_host.spacing_types[0] == quokka::TransformType::linear,
					 "disk_galaxy.vcirc_file must use linear spacing for the radius coordinate.");

	auto const num_radii = static_cast<std::size_t>(rotation_curve_host.sizes[0]);
	userData_.radius.resize(num_radii);
	userData_.vcirc.resize(num_radii);
	userData_.potential.resize(num_radii);
	userData_.rho_halo.resize(num_radii);
	userData_.velr_halo.resize(num_radii);
	userData_.temp_halo.resize(num_radii);

	constexpr double length_unit = 1.0e3 * C::parsec; // kpc
	constexpr double velocity_unit = 1.0e5;		  // km/s
	for (std::size_t n = 0; n < num_radii; ++n) {
		amrex::Real const radius = rotation_curve_host.coord_min[0] + static_cast<amrex::Real>(n) * rotation_curve_host.dcoord[0];
		userData_.radius[n] = radius * length_unit;
		userData_.vcirc[n] = rotation_curve_host.dataViewArrays[0](static_cast<int>(n)) * velocity_unit;
		userData_.rho_halo[n] = rotation_curve_host.dataViewArrays[1](static_cast<int>(n));
		userData_.velr_halo[n] = rotation_curve_host.dataViewArrays[2](static_cast<int>(n));
		userData_.temp_halo[n] = rotation_curve_host.dataViewArrays[3](static_cast<int>(n));
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.radius.front() > 0.0, "disk_galaxy.vcirc_file radii must be positive.");

	userData_.potential[0] = 0.0;
	for (std::size_t n = 1; n < num_radii; ++n) {
		amrex::Real const dr = userData_.radius[n] - userData_.radius[n - 1];
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dr > 0.0, "disk_galaxy.vcirc_file radii must be strictly increasing.");
		amrex::Real const integrand_lo = userData_.vcirc[n - 1] * userData_.vcirc[n - 1] / userData_.radius[n - 1];
		amrex::Real const integrand_hi = userData_.vcirc[n] * userData_.vcirc[n] / userData_.radius[n];
		userData_.potential[n] = userData_.potential[n - 1] + 0.5 * dr * (integrand_lo + integrand_hi);
	}

	userData_.r_inner = userData_.radius.front();
	userData_.r_outer = userData_.radius.back();
	userData_.vcirc_inner = userData_.vcirc.front();
	userData_.vcirc_outer = userData_.vcirc.back();
	userData_.potential_inner = userData_.potential.front();
	userData_.potential_outer = userData_.potential.back();
	userData_.rho_inner = userData_.rho_halo.front();
	userData_.rho_outer = userData_.rho_halo.back();
	userData_.velr_inner = userData_.velr_halo.front();
	userData_.velr_outer = userData_.velr_halo.back();
	userData_.temp_inner = userData_.temp_halo.front();
	userData_.temp_outer = userData_.temp_halo.back();

	pp.query("halo_vphi_expr", userData_.haloVphiExpr);
	userData_.useHaloVphiParser = !userData_.haloVphiExpr.empty();
	if (userData_.useHaloVphiParser) {
		userData_.haloVphiParser.emplace(userData_.haloVphiExpr);
		userData_.haloVphiParser->registerVariables({"x", "y", "z"});
		userData_.haloVphiParserExe = userData_.haloVphiParser->compile<3>();
#ifdef AMREX_USE_GPU
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.haloVphiParserExe->m_device_executor != nullptr,
						 "disk_galaxy.halo_vphi_expr: device parser executor is null after compile<3>().");
#endif
		userData_.haloVphiParser.reset();
	}
}

template <> void QuokkaSimulation<KeplerianDisk>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::ParmParse const pp("disk_galaxy");
	double disk_gas_mass_Msun = NAN;
	double disk_Rscale_kpc = NAN;
	double disk_zscale_kpc = NAN;
	double disk_temperature = NAN;
	double disk_perturb_amplitude = NAN;
	double disk_perturb_Rmax_kpc = NAN;
	double initial_scalar_density = NAN;
	pp.query("disk_gas_mass_Msun", disk_gas_mass_Msun);
	pp.query("disk_Rscale_kpc", disk_Rscale_kpc);
	pp.query("disk_zscale_kpc", disk_zscale_kpc);
	pp.query("disk_temperature", disk_temperature);
	pp.query("disk_perturb_amplitude", disk_perturb_amplitude);
	pp.query("disk_perturb_Rmax_kpc", disk_perturb_Rmax_kpc);
	pp.query("initial_scalar_density", initial_scalar_density);
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_gas_mass_Msun));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_Rscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_zscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_temperature));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_perturb_amplitude));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_perturb_Rmax_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(initial_scalar_density));

	constexpr double kpc = 1.0e3 * C::parsec;
	double const disk_gas_mass = disk_gas_mass_Msun * C::M_solar;
	double const R_d = disk_Rscale_kpc * kpc;
	double const z_d = disk_zscale_kpc * kpc;
	double const R_max_perturb = disk_perturb_Rmax_kpc * kpc;
	double const rho_0 = disk_gas_mass / (4.0 * M_PI * R_d * R_d * z_d);

	double const *radius_table = userData_.radius.dataPtr();
	double const *vcirc_table = userData_.vcirc.dataPtr();
	double const *rho_halo_table = userData_.rho_halo.dataPtr();
	double const *velr_halo_table = userData_.velr_halo.dataPtr();
	double const *temp_halo_table = userData_.temp_halo.dataPtr();
	int const table_length = static_cast<int>(userData_.radius.size());
	amrex::Real const r_table_min = userData_.r_inner;
	amrex::Real const r_table_max = userData_.r_outer;
	amrex::Real const vcirc_inner = userData_.vcirc_inner;
	amrex::Real const vcirc_outer = userData_.vcirc_outer;
	amrex::Real const rho_inner = userData_.rho_inner;
	amrex::Real const rho_outer = userData_.rho_outer;
	amrex::Real const velr_inner = userData_.velr_inner;
	amrex::Real const velr_outer = userData_.velr_outer;
	amrex::Real const temp_inner = userData_.temp_inner;
	amrex::Real const temp_outer = userData_.temp_outer;
	bool const use_halo_vphi_parser = userData_.useHaloVphiParser;
	amrex::ParserExecutor<3> halo_vphi_parser{};
	if (use_halo_vphi_parser) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.haloVphiParserExe.has_value(),
						 "disk_galaxy.halo_vphi_expr: parser executor is missing after compile<3>().");
		halo_vphi_parser = *userData_.haloVphiParserExe;
	}

	amrex::ParmParse const pp_particles("particles");
	double scalar_yield_per_SN = NAN;
	pp_particles.query("scalar_yield_per_SN", scalar_yield_per_SN);
	AMREX_ALWAYS_ASSERT(!std::isnan(scalar_yield_per_SN));
	amrex::Real const SNR_volume = std::pow(128.0 * C::parsec, 3);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(scalar_yield_per_SN > initial_scalar_density * SNR_volume,
					 "particles.scalar_yield_per_SN must be greater than (initial_scalar_density * (128 pc)^3).");

	amrex::Box const &index_range = grid_elem.indexRange_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	amrex::Array4<double> const &state = grid_elem.array_;

	amrex::ParallelFor(index_range, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		double const x0 = prob_lo[0] + static_cast<double>(i) * dx[0];
		double const y0 = prob_lo[1] + static_cast<double>(j) * dx[1];
		double const z0 = prob_lo[2] + static_cast<double>(k) * dx[2];
		double const x1 = x0 + dx[0];
		double const y1 = y0 + dx[1];
		double const z1 = z0 + dx[2];

		auto circularVelocity = [=](double radius) {
			if (radius <= r_table_min) {
				return vcirc_inner;
			}
			if (radius >= r_table_max) {
				return vcirc_outer;
			}
			return interpolate_value(radius, radius_table, vcirc_table, table_length);
		};

		auto diskDensity = [=](double x, double y, double z) {
			double const R = std::sqrt(x * x + y * y);
			double const theta = std::atan2(x, y);
			double const perturbation = disk_perturb_amplitude * jn(2, 5.1356 * R / R_max_perturb) * std::sin(2.0 * theta);
			return rho_0 * std::exp(-R / R_d) * std::exp(-std::abs(z) / z_d) * (1.0 + perturbation);
		};

		auto haloDensity = [=](double x, double y, double z) {
			double const radius = std::sqrt(x * x + y * y + z * z);
			if (radius <= r_table_min) {
				return rho_inner;
			}
			if (radius >= r_table_max) {
				return rho_outer;
			}
			return interpolate_value(radius, radius_table, rho_halo_table, table_length);
		};

		auto haloRadialVelocity = [=](double radius) {
			if (radius <= r_table_min) {
				return -velr_inner;
			}
			if (radius >= r_table_max) {
				return -velr_outer;
			}
			return -interpolate_value(radius, radius_table, velr_halo_table, table_length);
		};

		auto haloTemperature = [=](double x, double y, double z) {
			double const radius = std::sqrt(x * x + y * y + z * z);
			if (radius <= r_table_min) {
				return temp_inner;
			}
			if (radius >= r_table_max) {
				return temp_outer;
			}
			return interpolate_value(radius, radius_table, temp_halo_table, table_length);
		};

		auto haloAzimuthalVelocity = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			return use_halo_vphi_parser ? halo_vphi_parser(x, y, z) : 0.0;
		};

		auto diskXVelocity = [=](double x, double y, double /*z*/) {
			double const R = std::sqrt(x * x + y * y);
			return (R > 0.0) ? -circularVelocity(R) * y / R : 0.0;
		};
		auto diskYVelocity = [=](double x, double y, double /*z*/) {
			double const R = std::sqrt(x * x + y * y);
			return (R > 0.0) ? circularVelocity(R) * x / R : 0.0;
		};

		auto haloXVelocity = [=](double x, double y, double z) {
			double const radius = std::sqrt(x * x + y * y + z * z);
			double const R = std::sqrt(x * x + y * y);
			double const radial_component = (radius > 0.0) ? haloRadialVelocity(radius) * x / radius : 0.0;
			double const azimuthal_component = (R > 0.0) ? -haloAzimuthalVelocity(x, y, z) * y / R : 0.0;
			return radial_component + azimuthal_component;
		};
		auto haloYVelocity = [=](double x, double y, double z) {
			double const radius = std::sqrt(x * x + y * y + z * z);
			double const R = std::sqrt(x * x + y * y);
			double const radial_component = (radius > 0.0) ? haloRadialVelocity(radius) * y / radius : 0.0;
			double const azimuthal_component = (R > 0.0) ? haloAzimuthalVelocity(x, y, z) * x / R : 0.0;
			return radial_component + azimuthal_component;
		};
		auto haloZVelocity = [=](double x, double y, double z) {
			double const radius = std::sqrt(x * x + y * y + z * z);
			return (radius > 0.0) ? haloRadialVelocity(radius) * z / radius : 0.0;
		};

		auto totalDensity = [=](double x, double y, double z) { return diskDensity(x, y, z) + haloDensity(x, y, z); };
		auto xMomentum = [=](double x, double y, double z) {
			return diskDensity(x, y, z) * diskXVelocity(x, y, z) + haloDensity(x, y, z) * haloXVelocity(x, y, z);
		};
		auto yMomentum = [=](double x, double y, double z) {
			return diskDensity(x, y, z) * diskYVelocity(x, y, z) + haloDensity(x, y, z) * haloYVelocity(x, y, z);
		};
		auto zMomentum = [=](double x, double y, double z) { return haloDensity(x, y, z) * haloZVelocity(x, y, z); };
		auto internalEnergy = [=](double x, double y, double z) {
			constexpr double gamma = quokka::EOS_Traits<KeplerianDisk>::gamma;
			constexpr double mu = 0.61;
			double const disk_eint = diskDensity(x, y, z) * C::k_B * disk_temperature / (mu * C::m_p * (gamma - 1.0));
			double const halo_eint = haloDensity(x, y, z) * C::k_B * haloTemperature(x, y, z) / (mu * C::m_p * (gamma - 1.0));
			return disk_eint + halo_eint;
		};

		double const cell_volume = dx[0] * dx[1] * dx[2];
		double const rho = quad_3d(totalDensity, x0, x1, y0, y1, z0, z1) / cell_volume;
		double const xmom = quad_3d(xMomentum, x0, x1, y0, y1, z0, z1) / cell_volume;
		double const ymom = quad_3d(yMomentum, x0, x1, y0, y1, z0, z1) / cell_volume;
		double const zmom = quad_3d(zMomentum, x0, x1, y0, y1, z0, z1) / cell_volume;
		double const eint = quad_3d(internalEnergy, x0, x1, y0, y1, z0, z1) / cell_volume;
		double const kinetic_energy = 0.5 * (xmom * xmom + ymom * ymom + zmom * zmom) / rho;

		state(i, j, k, HydroSystem<KeplerianDisk>::density_index) = rho;
		state(i, j, k, HydroSystem<KeplerianDisk>::x1Momentum_index) = xmom;
		state(i, j, k, HydroSystem<KeplerianDisk>::x2Momentum_index) = ymom;
		state(i, j, k, HydroSystem<KeplerianDisk>::x3Momentum_index) = zmom;
		state(i, j, k, HydroSystem<KeplerianDisk>::energy_index) = eint + kinetic_energy;
		state(i, j, k, HydroSystem<KeplerianDisk>::internalEnergy_index) = eint;
		state(i, j, k, HydroSystem<KeplerianDisk>::scalar0_index) = initial_scalar_density;
	});
}

template <> void QuokkaSimulation<KeplerianDisk>::addProblemPotentialAtLevel(amrex::MultiFab &phi_mf, int lev, bool physical_ghosts_only)
{
	auto const prob_lo = geom[lev].ProbLoArray();
	auto const dx = geom[lev].CellSizeArray();
	auto const domain_lo = geom[lev].Domain().smallEnd();
	auto const domain_hi = geom[lev].Domain().bigEnd();
	auto const phi = phi_mf.arrays();
	double const *radius_table = userData_.radius.dataPtr();
	double const *potential_table = userData_.potential.dataPtr();
	int const table_length = static_cast<int>(userData_.radius.size());
	amrex::Real const r_table_min = userData_.r_inner;
	amrex::Real const r_table_max = userData_.r_outer;
	amrex::Real const vcirc_inner = userData_.vcirc_inner;
	amrex::Real const vcirc_outer = userData_.vcirc_outer;
	amrex::Real const potential_inner = userData_.potential_inner;
	amrex::Real const potential_outer = userData_.potential_outer;

	amrex::ParallelFor(phi_mf, phi_mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		bool const outside_physical_domain =
		    (i < domain_lo[0]) || (i > domain_hi[0]) || (j < domain_lo[1]) || (j > domain_hi[1]) || (k < domain_lo[2]) || (k > domain_hi[2]);
		if (physical_ghosts_only && !outside_physical_domain) {
			return;
		}

		double const x = prob_lo[0] + (static_cast<double>(i) + 0.5) * dx[0];
		double const y = prob_lo[1] + (static_cast<double>(j) + 0.5) * dx[1];
		double const z = prob_lo[2] + (static_cast<double>(k) + 0.5) * dx[2];
		double const radius = std::sqrt(x * x + y * y + z * z);

		double fixed_potential = potential_inner;
		if (radius < r_table_min) {
			double const radius_ratio = radius / r_table_min;
			fixed_potential += 0.5 * vcirc_inner * vcirc_inner * (radius_ratio * radius_ratio - 1.0);
		} else if (radius > r_table_max) {
			fixed_potential = potential_outer + vcirc_outer * vcirc_outer * std::log(radius / r_table_max);
		} else {
			fixed_potential = interpolate_value(radius, radius_table, potential_table, table_length);
		}

		phi[bx](i, j, k) += fixed_potential;
	});
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<KeplerianDisk>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	amrex::ParmParse const pp("disk_galaxy");
	amrex::Real refine_Rmax_kpc = NAN;
	amrex::Real refine_zmax_kpc = NAN;
	pp.query("refine_Rmax_kpc", refine_Rmax_kpc);
	pp.query("refine_zmax_kpc", refine_zmax_kpc);
	AMREX_ALWAYS_ASSERT(!std::isnan(refine_Rmax_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(refine_zmax_kpc));

	constexpr amrex::Real kpc = 1.0e3 * C::parsec;
	amrex::Real const refine_Rmax = refine_Rmax_kpc * kpc;
	amrex::Real const refine_zmax = refine_zmax_kpc * kpc;
	auto const prob_lo = geom[lev].ProbLoArray();
	auto const dx = geom[lev].CellSizeArray();
	auto const tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Real const x0 = prob_lo[0] + static_cast<amrex::Real>(i) * dx[0];
		amrex::Real const y0 = prob_lo[1] + static_cast<amrex::Real>(j) * dx[1];
		amrex::Real const z0 = prob_lo[2] + static_cast<amrex::Real>(k) * dx[2];
		amrex::Real const x1 = x0 + dx[0];
		amrex::Real const y1 = y0 + dx[1];
		amrex::Real const z1 = z0 + dx[2];

		for (amrex::Real const x : {x0, x1}) {
			for (amrex::Real const y : {y0, y1}) {
				for (amrex::Real const z : {z0, z1}) {
					amrex::Real const R = std::sqrt(x * x + y * y);
					if (R < refine_Rmax && std::abs(z) < refine_zmax) {
						tag[bx](i, j, k) = amrex::TagBox::SET;
					}
				}
			}
		}
	});
	amrex::Gpu::streamSynchronize();
}

template <>
void QuokkaSimulation<KeplerianDisk>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp, amrex::MultiFab const &state_cc,
							amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const & /*state_fc*/) const
{
	auto const output = mf.arrays();
	auto const state = state_cc.const_arrays();

	if (dname == "gpot") {
		auto const phi_arr = phi[lev].const_arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
	} else if (dname == "temperature") {
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			amrex::Real const rho = state[bx](i, j, k, HydroSystem<KeplerianDisk>::density_index);
			amrex::Real const eint = HydroSystem<KeplerianDisk>::ComputeInternalEnergy(state[bx], i, j, k, nullptr);
			output[bx](i, j, k, ncomp) = quokka::EOS<KeplerianDisk>::ComputeTgasFromEint(rho, eint);
		});
	} else if (dname == "pressure") {
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			output[bx](i, j, k, ncomp) = HydroSystem<KeplerianDisk>::ComputePressure(state[bx], i, j, k, nullptr) / C::k_B;
		});
	} else if (dname == "entropy") {
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			amrex::Real const rho = state[bx](i, j, k, HydroSystem<KeplerianDisk>::density_index);
			amrex::Real const eint = HydroSystem<KeplerianDisk>::ComputeInternalEnergy(state[bx], i, j, k, nullptr);
			output[bx](i, j, k, ncomp) = quokka::EOS<KeplerianDisk>::ComputeEntropyFromRhoEint(rho, eint) / keV_in_ergs;
		});
	} else if (dname == "radius_sph") {
		auto const prob_lo = geom[lev].ProbLoArray();
		auto const dx = geom[lev].CellSizeArray();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			amrex::Real const x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
			amrex::Real const y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
			amrex::Real const z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
			output[bx](i, j, k, ncomp) = std::sqrt(x * x + y * y + z * z) / (1.0e3 * C::parsec);
		});
	} else if (dname == "radial_velocity") {
		auto const prob_lo = geom[lev].ProbLoArray();
		auto const dx = geom[lev].CellSizeArray();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			amrex::Real const rho = state[bx](i, j, k, HydroSystem<KeplerianDisk>::density_index);
			amrex::Real const x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
			amrex::Real const y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
			amrex::Real const z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
			amrex::Real const radius = std::sqrt(x * x + y * y + z * z);
			amrex::Real const radial_momentum = x * state[bx](i, j, k, HydroSystem<KeplerianDisk>::x1Momentum_index) +
							    y * state[bx](i, j, k, HydroSystem<KeplerianDisk>::x2Momentum_index) +
							    z * state[bx](i, j, k, HydroSystem<KeplerianDisk>::x3Momentum_index);
			output[bx](i, j, k, ncomp) = (radius > 0.0) ? radial_momentum / (rho * radius * 1.0e5) : 0.0;
		});
	} else if (dname == "circular_velocity") {
		auto const prob_lo = geom[lev].ProbLoArray();
		auto const dx = geom[lev].CellSizeArray();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			amrex::Real const rho = state[bx](i, j, k, HydroSystem<KeplerianDisk>::density_index);
			amrex::Real const x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
			amrex::Real const y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
			amrex::Real const radius = std::sqrt(x * x + y * y);
			amrex::Real const angular_momentum = x * state[bx](i, j, k, HydroSystem<KeplerianDisk>::x2Momentum_index) -
							     y * state[bx](i, j, k, HydroSystem<KeplerianDisk>::x1Momentum_index);
			output[bx](i, j, k, ncomp) = (radius > 0.0) ? angular_momentum / (rho * radius * 1.0e5) : 0.0;
		});
	}
	amrex::Gpu::streamSynchronize();
}

template <> auto QuokkaSimulation<KeplerianDisk>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	std::map<std::string, amrex::Real> stats;

	amrex::Real const sfr = particleRegister_.computeSfrAveragedOverTime(tNew_[0], 1.0e6 * seconds_per_year);
	stats["sfr_1myr"] = (sfr / C::M_solar) * seconds_per_year;

	amrex::ParmParse const pp("disk_galaxy");
	amrex::Real refine_Rmax_kpc = NAN;
	amrex::Real refine_zmax_kpc = NAN;
	pp.query("refine_Rmax_kpc", refine_Rmax_kpc);
	pp.query("refine_zmax_kpc", refine_zmax_kpc);
	AMREX_ALWAYS_ASSERT(!std::isnan(refine_Rmax_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(refine_zmax_kpc));
	amrex::Real const refine_Rmax = refine_Rmax_kpc * (1.0e3 * C::parsec);
	amrex::Real const refine_zmax = refine_zmax_kpc * (1.0e3 * C::parsec);

	amrex::Vector<amrex::MultiFab> refine_mask(finest_level + 1);
	for (int lev = 0; lev <= finest_level; ++lev) {
		refine_mask[lev].define(boxArray(lev), DistributionMap(lev), 1, 0);
		auto const prob_lo = geom[lev].ProbLoArray();
		auto const dx = geom[lev].CellSizeArray();
		auto const state = state_new_cc_[lev].const_arrays();
		auto const result = refine_mask[lev].arrays();
		amrex::ParallelFor(refine_mask[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			amrex::Real const x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
			amrex::Real const y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
			amrex::Real const z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
			amrex::Real const radius = std::sqrt(x * x + y * y);
			amrex::Real const rho = state[bx](i, j, k, HydroSystem<KeplerianDisk>::density_index);
			result[bx](i, j, k) = (radius < refine_Rmax && std::abs(z) < refine_zmax) ? rho : 0.0;
		});
	}
	amrex::Gpu::streamSynchronize();
	amrex::Real const disk_mass = amrex::volumeWeightedSum(amrex::GetVecOfConstPtrs(refine_mask), 0, geom, ref_ratio);
	stats["disk_mass_refine_region"] = disk_mass / C::M_solar;

	amrex::Real const cold_mass =
	    computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state,
						       std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const & /*state_fc*/) noexcept {
		    amrex::Real const rho = state(i, j, k, HydroSystem<KeplerianDisk>::density_index);
		    amrex::Real const eint = HydroSystem<KeplerianDisk>::ComputeInternalEnergy(state, i, j, k, nullptr);
		    amrex::Real const temperature = quokka::EOS<KeplerianDisk>::ComputeTgasFromEint(rho, eint);
		    return (temperature < 1.0e4) ? rho : 0.0;
	    });
	stats["mass_T_lt_1e4"] = cold_mass / C::M_solar;
	stats["stellar_mass_at_birth"] = particleRegister_.computeTotalStellarMassAtBirth() / C::M_solar;
	stats["sn_count_cumulative"] = sn_count_cumulative_;

	return stats;
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<KeplerianDisk>(quokka::BCType::reflecting);
	QuokkaSimulation<KeplerianDisk> sim(BCs_cc);
	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
