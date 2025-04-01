//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file galaxy.cpp
/// \brief Defines a simulation using the AGORA isolated galaxy initial conditions.
///

#include <algorithm>
#include <cmath>

#include "boost/math/special_functions/bessel.hpp"

#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "galaxy.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "math/quadrature.hpp"
#include "physics_info.hpp"

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
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = 1;			     // number of radiation groups
};

template <> struct Particle_Traits<AgoraGalaxy> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> struct SimulationData<AgoraGalaxy> {
	amrex::Real r_inner{};
	amrex::Real r_outer{};
	amrex::Real vcirc_inner{};
	amrex::Real vcirc_outer{};
	amrex::Gpu::PinnedVector<amrex::Real> radius;
	amrex::Gpu::PinnedVector<amrex::Real> vcirc;
};

constexpr double rho_halo = 1.0e-6 * quokka::EOS_Traits<AgoraGalaxy>::mean_molecular_weight;

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
}

template <> void QuokkaSimulation<AgoraGalaxy>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	double const *R_table = userData_.radius.dataPtr();
	double const *vcirc_table = userData_.vcirc.dataPtr();
	auto const len_table = static_cast<int>(userData_.radius.size());
	const amrex::Real R_table_min = userData_.r_inner;
	const amrex::Real R_table_max = userData_.r_outer;
	const amrex::Real vcirc_inner = userData_.vcirc_inner;
	const amrex::Real vcirc_outer = userData_.vcirc_outer;
	const amrex::Real rho_bg = ::rho_halo; // workaround nvcc compiler bug

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// Cartesian coordinates
		amrex::Real const x0 = prob_lo[0] + (i * dx[0]);
		amrex::Real const y0 = prob_lo[1] + (j * dx[1]);
		amrex::Real const z0 = prob_lo[2] + (k * dx[2]);

		amrex::Real const x1 = prob_lo[0] + ((i + 1) * dx[0]);
		amrex::Real const y1 = prob_lo[1] + ((j + 1) * dx[1]);
		amrex::Real const z1 = prob_lo[2] + ((k + 1) * dx[2]);

		// compute density profile
		auto rho_exact = [rho_bg](double x, double y, double z) {
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			constexpr double M_GAS = 8.59322e9 * C::M_solar;		// disk mass: 8.59322e9 Msun (20% gas fraction)
			constexpr double r_d = 3.43218e3 * C::parsec;			// disk scale length: 3.43218 kpc
			constexpr double z_d = 0.343218e3 * C::parsec;			// disk scale height: 0.343218 kpc
			constexpr double rho_0 = M_GAS / 4. / M_PI / (r_d * r_d) / z_d; // normalization constant
			return rho_0 * std::exp(-R / r_d) * std::exp(-std::abs(z) / z_d);
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

		// integrate density profile over cell volume
		const double cell_vol = dx[0] * dx[1] * dx[2];
		const double rho_disk = quad_3d(rho_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		AMREX_ALWAYS_ASSERT(!std::isnan(rho_disk));

		// set temperature to constant (the disk will cool quasi-instantly)
		const double T_halo = 1.0e6; // K
		const double T_disk = 1.0e4; // K

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
			double const R_max = 20.0e3 * C::parsec;
			double const drho_over_rho = boost::math::cyl_bessel_j(2, 5.1356 * R / R_max) * std::sin(2.0 * theta);
			rho = rho_disk * (1 + drho_over_rho);
			AMREX_ALWAYS_ASSERT(rho > 0.);

			// set temperature
			T = T_disk;

			// set velocity (integrate velocity profiles over cell volume)
			vx = quad_3d(vx_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
			vy = quad_3d(vy_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
			AMREX_ALWAYS_ASSERT(!std::isnan(vx));
			AMREX_ALWAYS_ASSERT(!std::isnan(vy));
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
	amrex::Print() << "Reading particles from file " << filename << "...\n";

	CICParticles->SetVerbose(1);
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->InitFromAsciiFile(filename, nreal_extra, nullptr);
}

template <> void QuokkaSimulation<AgoraGalaxy>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// quasi-Lagrangian refinement:
	// refine if the cell mass is greater than 'mass_refine_threshold'
	amrex::ParmParse const pp("agora_galaxy");
	amrex::Real mass_refine_threshold_Msun = NAN;
	pp.query("mass_refine_threshold", mass_refine_threshold_Msun);
	const amrex::Real mass_refine_threshold = mass_refine_threshold_Msun * C::M_solar;

	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const amrex::Real cell_vol = dx[0] * dx[1] * dx[2];
	const auto state = state_new_cc_[lev].const_arrays();
	const auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Real const rho = state[bx](i, j, k, HydroSystem<AgoraGalaxy>::density_index);
		amrex::Real const cell_mass = rho * cell_vol;
		if (cell_mass > mass_refine_threshold) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
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
		auto tables = grackleTables_.const_tables();
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
				Real const Tgas = quokka::GrackleLikeCooling::ComputeTgasFromEgas(rho, Eint, HydroSystem<AgoraGalaxy>::gamma_, tables);
				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<AgoraGalaxy>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<AgoraGalaxy>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<AgoraGalaxy>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<AgoraGalaxy>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	// Problem initialization
	QuokkaSimulation<AgoraGalaxy> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	const int status = 0;
	return status;
}
