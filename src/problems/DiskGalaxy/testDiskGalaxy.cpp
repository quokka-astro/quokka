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
	amrex::Real vcirc_outer{};
	amrex::Real rho_outer{};
	amrex::Real momr_outer{};
	amrex::Real eint_outer{};
	amrex::Real etot_outer{};

	amrex::Gpu::PinnedVector<amrex::Real> radius;
	amrex::Gpu::PinnedVector<amrex::Real> vcirc;
	amrex::Gpu::PinnedVector<amrex::Real> rho_halo;
	amrex::Gpu::PinnedVector<amrex::Real> momr_halo;
	amrex::Gpu::PinnedVector<amrex::Real> eint_halo;
	amrex::Gpu::PinnedVector<amrex::Real> etot_halo;
};

template <> void QuokkaSimulation<AgoraGalaxy>::preCalculateInitialConditions()
{
	// 1. read in circular velocity table "vcirc.dat"
	std::vector<amrex::Real> radius_h;
	std::vector<amrex::Real> vcirc_h;
	std::vector<amrex::Real> rho_h;
	std::vector<amrex::Real> momr_h;
	std::vector<amrex::Real> eint_h;
	std::vector<amrex::Real> etot_h;

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
		Real const rho_val = values.at(2);
		Real const momr_val = values.at(3);
		Real const eint_val = values.at(4);
		Real const etot_val = values.at(5);
		radius_h.push_back(R_val);
		vcirc_h.push_back(vcirc_val);
		rho_h.push_back(rho_val);
		momr_h.push_back(momr_val);
		eint_h.push_back(eint_val);
		etot_h.push_back(etot_val);
	}

	// 2. copy data to simData_.radius and simData_.vcirc
	const size_t N = radius_h.size();
	userData_.radius.resize(N);
	userData_.vcirc.resize(N);
	userData_.rho_halo.resize(N);
	userData_.rho_halo.resize(N);
	userData_.momr_halo.resize(N);
	userData_.eint_halo.resize(N);
	userData_.etot_halo.resize(N);

	const double length_unit = 1.0e3 * C::parsec; // kpc
	const double vel_unit = 1.0e5;		      // km/s
	for (size_t i = 0; i < N; ++i) {
		userData_.radius[i] = radius_h[i] * length_unit;
		userData_.vcirc[i] = vcirc_h[i] * vel_unit;
		userData_.rho_halo[i] = rho_h[i];
		userData_.momr_halo[i] = momr_h[i];
		userData_.eint_halo[i] = eint_h[i];
		userData_.etot_halo[i] = etot_h[i];
	}

	// save min/max radii
	auto min_result = std::min_element(radius_h.begin(), radius_h.end());
	userData_.r_inner = (*min_result) * length_unit;

	auto max_result = std::max_element(radius_h.begin(), radius_h.end());
	userData_.r_outer = (*max_result) * length_unit;
	userData_.vcirc_outer = vcirc_h[std::distance(radius_h.begin(), max_result)] * vel_unit;
	userData_.rho_outer = rho_h[std::distance(radius_h.begin(), max_result)];
	userData_.momr_outer = momr_h[std::distance(radius_h.begin(), max_result)];
	userData_.eint_outer = eint_h[std::distance(radius_h.begin(), max_result)];
	userData_.etot_outer = etot_h[std::distance(radius_h.begin(), max_result)];
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

	// halo parameters
	double T_halo = 1.0e6;	    // K
	double ndens_halo = 1.0e-6; // cm^{-3}
	pp.query("halo_temperature", T_halo);
	pp.query("halo_number_density", ndens_halo);
	AMREX_ALWAYS_ASSERT(!std::isnan(T_halo));
	AMREX_ALWAYS_ASSERT(!std::isnan(ndens_halo));

	// const double rho_halo = ndens_halo * quokka::EOS_Traits<AgoraGalaxy>::mean_molecular_weight;

	// read tables

	double const *R_table = userData_.radius.dataPtr();
	double const *vcirc_table = userData_.vcirc.dataPtr();
	double const *rhoH_table = userData_.rho_halo.dataPtr();
	double const *momr_table = userData_.momr_halo.dataPtr();
	double const *eint_table = userData_.eint_halo.dataPtr();
	double const *etot_table = userData_.etot_halo.dataPtr();

	auto const len_table = static_cast<int>(userData_.radius.size());
	const amrex::Real R_table_max = userData_.r_outer;
	const amrex::Real vcirc_outer = userData_.vcirc_outer;
	const amrex::Real rho_outer = userData_.rho_outer;
	const amrex::Real momr_outer = userData_.momr_outer;
	const amrex::Real eint_outer = userData_.eint_outer;
	const amrex::Real etot_outer = userData_.etot_outer;

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

		auto vcirc_exact = [R_table_max, R_table, vcirc_outer, vcirc_table, len_table](const amrex::Real R) {
			double vcirc = NAN;
			if ((R < R_table_max)) {
				vcirc = interpolate_value(R, R_table, vcirc_table, len_table);
			} else if (R >= R_table_max) {
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

		auto rhoHalo = [R_table, R_table_max, rho_outer, rhoH_table, len_table](const amrex::Real R) {
			double rho_H = NAN;
			if ((R <= R_table_max)) {
				rho_H = interpolate_value(R, R_table, rhoH_table, len_table);
			} else {
				rho_H = rho_outer;
			}

			return rho_H;
		};

		auto momHalo = [R_table, R_table_max, momr_outer, momr_table, len_table](const amrex::Real R) {
			double mom_H = NAN;
			if ((R <= R_table_max)) {
				mom_H = interpolate_value(R, R_table, momr_table, len_table);
			} else {
				mom_H = momr_outer;
			}
			return mom_H;
		};

		auto eintHalo = [R_table, R_table_max, eint_outer, eint_table, len_table](const amrex::Real R) {
			double eint_H = NAN;
			if ((R <= R_table_max)) {
				eint_H = interpolate_value(R, R_table, eint_table, len_table);
			} else {
				eint_H = eint_outer;
			}
			return eint_H;
		};

		auto etotHalo = [R_table, R_table_max, etot_outer, etot_table, len_table](const amrex::Real R) {
			double etot_H = NAN;
			if ((R <= R_table_max)) {
				etot_H = interpolate_value(R, R_table, etot_table, len_table);
			} else {
				etot_H = etot_outer;
			}
			return etot_H;
		};

		// compute density profiles
		auto rhoHalo_exact = [rhoHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return rhoHalo(r);
		};

		// compute eint profiles
		auto eintHalo_exact = [eintHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return eintHalo(r);
		};

		auto etotHalo_exact = [etotHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return etotHalo(r);
		};

		// compute momenta profiles
		auto momx_exact = [momHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const theta = std::atan2(R, z) + (z < 0.0 ? M_PI : 0.0);
			;
			double const phi = std::atan2(y, x);
			return momHalo(r) * std::sin(theta) * std::cos(phi); // vx
		};

		auto momy_exact = [momHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const theta = std::atan2(R, z) + (z < 0.0 ? M_PI : 0.0);
			;
			double const phi = std::atan2(y, x);
			return momHalo(r) * std::sin(theta) * std::sin(phi); // vy
		};

		auto momz_exact = [momHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const theta = std::atan2(R, z) + (z < 0.0 ? M_PI : 0.0);
			;
			return momHalo(r) * std::cos(theta); // vz
		};

		// integrate density profile over cell volume
		// TODO(bwibking): use adaptive quadrature with relative tolerance
		const double cell_vol = dx[0] * dx[1] * dx[2];
		const double rho_disk = quad_3d(rho_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double rho_halo = quad_3d(rhoHalo_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double momx_halo = quad_3d(momx_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double momy_halo = quad_3d(momy_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double momz_halo = quad_3d(momz_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double eint_halo = quad_3d(eintHalo_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double etot_halo = quad_3d(etotHalo_exact, x0, x1, y0, y1, z0, z1) / cell_vol;

		AMREX_ALWAYS_ASSERT(!std::isnan(rho_disk));

		double rho = NAN;
		double vx = NAN;
		double vy = NAN;
		double const vz = 0;
		double T = NAN;

		// IMPORTANT: transition between disk and halo at the P_halo == P_disk surface
		if (rho_halo * T_halo > rho_disk * T_disk) {
			rho = rho_halo;
			T = T_halo;
			vx = 0; // velocity is zero in the halo
			vy = 0;
		} else { // we are in the disk
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

		// compute auxiliary quantities
		double const vsq = (vx * vx) + (vy * vy) + (vz * vz);
		double const Eint = quokka::EOS<AgoraGalaxy>::ComputeEintFromTgas(rho, T);

		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::density_index) = rho + rho_halo;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x1Momentum_index) = rho * vx + momx_halo;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x2Momentum_index) = rho * vy + momy_halo;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x3Momentum_index) = rho * vz + momz_halo;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::energy_index) = Eint + 0.5 * rho * vsq + etot_halo;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::internalEnergy_index) = Eint + eint_halo;
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
