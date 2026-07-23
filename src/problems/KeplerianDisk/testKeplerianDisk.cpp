//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2026 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testKeplerianDisk.cpp
/// \brief Defines an isolated gas disk in a fixed spherical gravitational potential.

#include <cmath>

#include "AMReX_BLassert.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_TagBox.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "math/quadrature.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "util/DataTable.hpp"

struct KeplerianDisk {
};

static_assert(AMREX_SPACEDIM == 3, "KeplerianDisk problem requires AMREX_SPACEDIM == 3.");

template <> struct quokka::EOS_Traits<KeplerianDisk> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = 0.6 * C::m_u;
	using EOSBackend = quokka::EOSIdeal<KeplerianDisk>;
};

template <> struct HydroSystem_Traits<KeplerianDisk> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<KeplerianDisk> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
};

template <> struct SimulationData<KeplerianDisk> {
	amrex::Real r_inner{};
	amrex::Real r_outer{};
	amrex::Real vcirc_inner{};
	amrex::Real vcirc_outer{};
	amrex::Gpu::PinnedVector<amrex::Real> radius;
	amrex::Gpu::PinnedVector<amrex::Real> vcirc;
};

template <> void QuokkaSimulation<KeplerianDisk>::preCalculateInitialConditions()
{
	amrex::ParmParse const pp("disk_galaxy");
	std::string filename;
	pp.get("vcirc_file", filename);

	// Use the same four-output table format as DiskGalaxy. Only the circular
	// velocity column is needed for this problem.
	auto rotation_curve = quokka::DataTable<1, 4, quokka::OutOfBounds::clamp>::CSVReader(filename, quokka::TransformType::linear);
	auto const rotation_curve_host = rotation_curve.const_tables_host();
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rotation_curve_host.sizes[0] > 1, "disk_galaxy.vcirc_file must contain at least two radii.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rotation_curve_host.spacing_types[0] == quokka::TransformType::linear,
					 "disk_galaxy.vcirc_file must use linear spacing for the radius coordinate.");

	auto const num_radii = static_cast<std::size_t>(rotation_curve_host.sizes[0]);
	userData_.radius.resize(num_radii);
	userData_.vcirc.resize(num_radii);

	constexpr double length_unit = 1.0e3 * C::parsec; // kpc
	constexpr double velocity_unit = 1.0e5;		  // km/s
	for (std::size_t n = 0; n < num_radii; ++n) {
		amrex::Real const radius = rotation_curve_host.coord_min[0] + static_cast<amrex::Real>(n) * rotation_curve_host.dcoord[0];
		userData_.radius[n] = radius * length_unit;
		userData_.vcirc[n] = rotation_curve_host.dataViewArrays[0](static_cast<int>(n)) * velocity_unit;
	}

	userData_.r_inner = userData_.radius.front();
	userData_.r_outer = userData_.radius.back();
	userData_.vcirc_inner = userData_.vcirc.front();
	userData_.vcirc_outer = userData_.vcirc.back();
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
	pp.query("disk_gas_mass_Msun", disk_gas_mass_Msun);
	pp.query("disk_Rscale_kpc", disk_Rscale_kpc);
	pp.query("disk_zscale_kpc", disk_zscale_kpc);
	pp.query("disk_temperature", disk_temperature);
	pp.query("disk_perturb_amplitude", disk_perturb_amplitude);
	pp.query("disk_perturb_Rmax_kpc", disk_perturb_Rmax_kpc);
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_gas_mass_Msun));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_Rscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_zscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_temperature));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_perturb_amplitude));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_perturb_Rmax_kpc));

	constexpr double kpc = 1.0e3 * C::parsec;
	double const disk_gas_mass = disk_gas_mass_Msun * C::M_solar;
	double const R_d = disk_Rscale_kpc * kpc;
	double const z_d = disk_zscale_kpc * kpc;
	double const R_max_perturb = disk_perturb_Rmax_kpc * kpc;
	double const rho_0 = disk_gas_mass / (4.0 * M_PI * R_d * R_d * z_d);
	double const density_floor = densityFloor_;
	bool const use_density_floor_parser = useDensityFloorParser_;
	amrex::ParserExecutor<4> density_floor_parser{};
	if (use_density_floor_parser) {
		density_floor_parser = densityFloorParserExe_.value();
	}

	double const *radius_table = userData_.radius.dataPtr();
	double const *vcirc_table = userData_.vcirc.dataPtr();
	int const table_length = static_cast<int>(userData_.radius.size());
	amrex::Real const r_table_min = userData_.r_inner;
	amrex::Real const r_table_max = userData_.r_outer;
	amrex::Real const vcirc_inner = userData_.vcirc_inner;
	amrex::Real const vcirc_outer = userData_.vcirc_outer;

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

		auto density = [=](double x, double y, double z) {
			double const R = std::sqrt(x * x + y * y);
			double const theta = std::atan2(x, y);
			double const perturbation = disk_perturb_amplitude * jn(2, 5.1356 * R / R_max_perturb) * std::sin(2.0 * theta);
			double const rho_disk = rho_0 * std::exp(-R / R_d) * std::exp(-std::abs(z) / z_d) * (1.0 + perturbation);
			double const local_density_floor = use_density_floor_parser ? density_floor_parser(x, y, z, density_floor) : density_floor;
			return amrex::max(rho_disk, local_density_floor);
		};

		auto xMomentum = [=](double x, double y, double z) {
			double const R = std::sqrt(x * x + y * y);
			return (R > 0.0) ? density(x, y, z) * (-circularVelocity(R) * y / R) : 0.0;
		};
		auto yMomentum = [=](double x, double y, double z) {
			double const R = std::sqrt(x * x + y * y);
			return (R > 0.0) ? density(x, y, z) * (circularVelocity(R) * x / R) : 0.0;
		};
		auto internalEnergy = [=](double x, double y, double z) {
			return quokka::EOS<KeplerianDisk>::ComputeEintFromTgas(density(x, y, z), disk_temperature);
		};

		double const cell_volume = dx[0] * dx[1] * dx[2];
		double const rho = quad_3d(density, x0, x1, y0, y1, z0, z1) / cell_volume;
		double const xmom = quad_3d(xMomentum, x0, x1, y0, y1, z0, z1) / cell_volume;
		double const ymom = quad_3d(yMomentum, x0, x1, y0, y1, z0, z1) / cell_volume;
		double const eint = quad_3d(internalEnergy, x0, x1, y0, y1, z0, z1) / cell_volume;
		double const kinetic_energy = 0.5 * (xmom * xmom + ymom * ymom) / rho;

		state(i, j, k, HydroSystem<KeplerianDisk>::density_index) = rho;
		state(i, j, k, HydroSystem<KeplerianDisk>::x1Momentum_index) = xmom;
		state(i, j, k, HydroSystem<KeplerianDisk>::x2Momentum_index) = ymom;
		state(i, j, k, HydroSystem<KeplerianDisk>::x3Momentum_index) = 0.0;
		state(i, j, k, HydroSystem<KeplerianDisk>::energy_index) = eint + kinetic_energy;
		state(i, j, k, HydroSystem<KeplerianDisk>::internalEnergy_index) = eint;
	});
}

template <> void QuokkaSimulation<KeplerianDisk>::addStrangSplitSources(amrex::MultiFab &state_mf, int lev, amrex::Real /*time*/, amrex::Real dt)
{
	auto const prob_lo = geom[lev].ProbLoArray();
	auto const dx = geom[lev].CellSizeArray();
	auto const state = state_mf.arrays();
	double const *radius_table = userData_.radius.dataPtr();
	double const *vcirc_table = userData_.vcirc.dataPtr();
	int const table_length = static_cast<int>(userData_.radius.size());
	amrex::Real const r_table_min = userData_.r_inner;
	amrex::Real const r_table_max = userData_.r_outer;
	amrex::Real const vcirc_inner = userData_.vcirc_inner;
	amrex::Real const vcirc_outer = userData_.vcirc_outer;

	amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		double const x = prob_lo[0] + (static_cast<double>(i) + 0.5) * dx[0];
		double const y = prob_lo[1] + (static_cast<double>(j) + 0.5) * dx[1];
		double const z = prob_lo[2] + (static_cast<double>(k) + 0.5) * dx[2];
		double const radius_squared = x * x + y * y + z * z;
		if (radius_squared <= 0.0) {
			return;
		}

		double const radius = std::sqrt(radius_squared);
		double vcirc = vcirc_inner;
		if (radius >= r_table_max) {
			vcirc = vcirc_outer;
		} else if (radius > r_table_min) {
			vcirc = interpolate_value(radius, radius_table, vcirc_table, table_length);
		}

		double const acceleration_scale = -(vcirc * vcirc) / radius_squared;
		double const rho = state[bx](i, j, k, HydroSystem<KeplerianDisk>::density_index);
		double &xmom = state[bx](i, j, k, HydroSystem<KeplerianDisk>::x1Momentum_index);
		double &ymom = state[bx](i, j, k, HydroSystem<KeplerianDisk>::x2Momentum_index);
		double &zmom = state[bx](i, j, k, HydroSystem<KeplerianDisk>::x3Momentum_index);
		double const kinetic_energy_old = 0.5 * (xmom * xmom + ymom * ymom + zmom * zmom) / rho;

		xmom += dt * rho * acceleration_scale * x;
		ymom += dt * rho * acceleration_scale * y;
		zmom += dt * rho * acceleration_scale * z;

		double const kinetic_energy_new = 0.5 * (xmom * xmom + ymom * ymom + zmom * zmom) / rho;
		state[bx](i, j, k, HydroSystem<KeplerianDisk>::energy_index) += kinetic_energy_new - kinetic_energy_old;
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

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<KeplerianDisk>(quokka::BCType::reflecting);
	QuokkaSimulation<KeplerianDisk> sim(BCs_cc);
	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
