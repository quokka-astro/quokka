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

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArrayBase.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
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
	static constexpr bool is_mhd_enabled = false;
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
	amrex::Real vcirc_inner{};
	amrex::Real vcirc_outer{};
	amrex::Gpu::PinnedVector<amrex::Real> radius;
	amrex::Gpu::PinnedVector<amrex::Real> vcirc;

	amrex::Real profile_r_min{};
	amrex::Real profile_r_max{};
	amrex::Gpu::PinnedVector<amrex::Real> profile_radius;
	amrex::Gpu::PinnedVector<amrex::Real> profile_rho;
	amrex::Gpu::PinnedVector<amrex::Real> profile_vr;
	amrex::Gpu::PinnedVector<amrex::Real> profile_temp;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto QuokkaSimulation<AgoraGalaxy>::densityFloor(amrex::Real x, amrex::Real y, amrex::Real z,
											  amrex::Real base_density_floor) const -> amrex::Real
{
	amrex::Real constexpr r_break = 10.0e3 * C::parsec; // 10 kpc
	amrex::Real const r = std::sqrt(x * x + y * y + z * z);
	if (r <= r_break) {
		return base_density_floor;
	}
	amrex::Real const ratio = r_break / r;
	return base_density_floor * ratio * ratio;
}

template <> void QuokkaSimulation<AgoraGalaxy>::preCalculateInitialConditions()
{
	// 1. read in circular velocity table "vcirc.dat"
	std::vector<amrex::Real> radius_h;
	std::vector<amrex::Real> vcirc_h;

	// get circular velocity profile filename from ParmParse
	amrex::ParmParse const pp("agora_galaxy");
	std::string filename;
	pp.query("vcirc_file", filename);

	std::ifstream fstream(filename, std::ios::in);
	AMREX_ALWAYS_ASSERT(fstream.is_open());
	std::string header;
	std::getline(fstream, header);

	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values;

		for (double value = NAN; iss >> value;) {
			values.push_back(value);
		}
		Real const R_val = values.at(0);
		Real const vcirc_val = values.at(1);

		radius_h.push_back(R_val);
		vcirc_h.push_back(vcirc_val);
	}

	// 2. copy data to simData_.radius and simData_.vcirc
	const size_t N = radius_h.size();
	userData_.radius.resize(N);
	userData_.vcirc.resize(N);

	const double length_unit = 1.0e3 * C::parsec; // kpc
	const double vel_unit = 1.0e5;		      // km/s
	for (size_t i = 0; i < N; ++i) {
		userData_.radius[i] = radius_h[i] * length_unit;
		userData_.vcirc[i] = vcirc_h[i] * vel_unit;
	}

	// save min/max radii
	auto min_result = std::min_element(radius_h.begin(), radius_h.end());
	userData_.r_inner = (*min_result) * length_unit;
	userData_.vcirc_inner = vcirc_h[std::distance(radius_h.begin(), min_result)] * vel_unit;

	auto max_result = std::max_element(radius_h.begin(), radius_h.end());
	userData_.r_outer = (*max_result) * length_unit;
	userData_.vcirc_outer = vcirc_h[std::distance(radius_h.begin(), max_result)] * vel_unit;

	// read CGM profile
	std::string profile_filename = "extern/agora_data/cooling_flow_MW.txt";
	pp.query("cooling_flow_file", profile_filename);

	std::vector<amrex::Real> profile_radius_h;
	std::vector<amrex::Real> profile_rho_h;
	std::vector<amrex::Real> profile_vr_h;
	std::vector<amrex::Real> profile_temp_h;

	std::ifstream profile_fstream(profile_filename, std::ios::in);
	AMREX_ALWAYS_ASSERT(profile_fstream.is_open());
	std::string profile_header;
	std::getline(profile_fstream, profile_header);

	for (std::string line; std::getline(profile_fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values;

		for (double value = NAN; iss >> value;) {
			values.push_back(value);
		}
		Real const R_val = values.at(0);
		Real const rho_val = values.at(1);
		Real const vr_val = values.at(2);
		Real const temp_val = values.at(3);

		profile_radius_h.push_back(R_val);
		profile_rho_h.push_back(rho_val);
		profile_vr_h.push_back(vr_val);
		profile_temp_h.push_back(temp_val);
	}

	// copy data to simData
	const size_t N_profile = profile_radius_h.size();
	userData_.profile_radius.resize(N_profile);
	userData_.profile_rho.resize(N_profile);
	userData_.profile_vr.resize(N_profile);
	userData_.profile_temp.resize(N_profile);

	for (size_t i = 0; i < N_profile; ++i) {
		userData_.profile_radius[i] = profile_radius_h[i] * length_unit;
		userData_.profile_rho[i] = profile_rho_h[i];
		userData_.profile_vr[i] = profile_vr_h[i];
		userData_.profile_temp[i] = profile_temp_h[i];
	}

	auto min_profile = std::min_element(profile_radius_h.begin(), profile_radius_h.end());
	userData_.profile_r_min = (*min_profile) * length_unit;

	auto max_profile = std::max_element(profile_radius_h.begin(), profile_radius_h.end());
	userData_.profile_r_max = (*max_profile) * length_unit;
}

template <> void QuokkaSimulation<AgoraGalaxy>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// read parameters
	//
	amrex::ParmParse const pp("agora_galaxy");

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
	//
	double const *R_table = userData_.radius.dataPtr();
	double const *vcirc_table = userData_.vcirc.dataPtr();
	auto const len_table = static_cast<int>(userData_.radius.size());
	const amrex::Real R_table_min = userData_.r_inner;
	const amrex::Real R_table_max = userData_.r_outer;
	const amrex::Real vcirc_inner = userData_.vcirc_inner;
	const amrex::Real vcirc_outer = userData_.vcirc_outer;

	double const *profile_R_table = userData_.profile_radius.dataPtr();
	double const *profile_rho_table = userData_.profile_rho.dataPtr();
	double const *profile_vr_table = userData_.profile_vr.dataPtr();
	double const *profile_temp_table = userData_.profile_temp.dataPtr();
	auto const profile_len_table = static_cast<int>(userData_.profile_radius.size());
	const amrex::Real profile_R_min = userData_.profile_r_min;
	const amrex::Real profile_R_max = userData_.profile_r_max;

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

		// compute density profile
		auto rho_exact = [rho_0, R_d, z_d](double x, double y, double z) {
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			return rho_0 * std::exp(-R / R_d) * std::exp(-std::abs(z) / z_d);
		};

		auto vcirc_exact = [R_table_min, R_table_max, R_table, vcirc_inner, vcirc_outer, vcirc_table, len_table](const amrex::Real R) {
			double vcirc = NAN;
			if ((R >= R_table_min) && (R <= R_table_max)) {
				vcirc = interpolate_value(R, R_table, vcirc_table, len_table);
			} else if (R < R_table_min) {
				vcirc = vcirc_inner;
			} else {
				vcirc = vcirc_outer;
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

		double rho = 0;
		double vx = NAN;
		double vy = NAN;
		double vz = NAN;
		double T = NAN;

		double const x = 0.5 * (x0 + x1);
		double const y = 0.5 * (y0 + y1);
		double const z = 0.5 * (z0 + z1);
		double const r = std::sqrt(x * x + y * y + z * z);
		{
			// Cooling flow profile
			double rho_cf = 0;
			double temp_cf = 0;
			double vr_cf = 0;
			if (r <= profile_R_min) {
				rho_cf = profile_rho_table[0];
				vr_cf = profile_vr_table[0];
				temp_cf = profile_temp_table[0];
			} else if (r >= profile_R_max) {
				rho_cf = profile_rho_table[profile_len_table - 1];
				vr_cf = profile_vr_table[profile_len_table - 1];
				temp_cf = profile_temp_table[profile_len_table - 1];
			} else {
				rho_cf = interpolate_value(r, profile_R_table, profile_rho_table, profile_len_table);
				vr_cf = interpolate_value(r, profile_R_table, profile_vr_table, profile_len_table);
				temp_cf = interpolate_value(r, profile_R_table, profile_temp_table, profile_len_table);
			}

			rho += rho_cf;
			T = temp_cf;
			vx = -vr_cf * (x / r);
			vy = -vr_cf * (y / r);
			vz = -vr_cf * (z / r);
		}
		{
			// integrate density profile over cell volume
			const double cell_vol = dx[0] * dx[1] * dx[2];
			const double rho_disk = quad_3d(rho_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
			AMREX_ALWAYS_ASSERT(!std::isnan(rho_disk));

			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const theta = std::atan2(x, y);

			// set density (compute density perturbation)
			// NOTE: jn is the C standard math function for BesselJ. it works everywhere.
			double const drho_over_rho = disk_perturb_amplitude * jn(2, 5.1356 * R / R_max_perturb) * std::sin(2.0 * theta);

			if (rho_disk > rho) {
				// we are in the disk
				// set velocity to disk velocity
				vx = quad_3d(vx_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
				vy = quad_3d(vy_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
				vz = 0;

				// compute harmonic mean of temperatures
				T = (2.0 * T * T_disk) / (T + T_disk);
			}
			// update density
			rho += rho_disk * (1 + drho_over_rho);
		}

		// compute auxiliary quantities
		double const vsq = (vx * vx) + (vy * vy) + (vz * vz);
		double const Eint = quokka::EOS<AgoraGalaxy>::ComputeEintFromTgas(rho, T);

		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::energy_index) = Eint + 0.5 * rho * vsq;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::internalEnergy_index) = Eint;
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
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<AgoraGalaxy>(quokka::BCType::reflecting);

	// Problem initialization
	QuokkaSimulation<AgoraGalaxy> sim(BCs_cc);

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	const int status = 0;
	return status;
}
