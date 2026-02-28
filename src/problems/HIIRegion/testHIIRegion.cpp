/// \file testHIIRegion.cpp
/// \brief Strömgren temperature-floor integration test with analytic radius comparison.
///

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "util/BC.hpp"

#include <cmath>
#include <fstream>
#include <hdf5.h>
#include <limits>

struct HIIRegionProblem {
};

template <> struct SimulationData<HIIRegionProblem> {
	amrex::Real nH0 = 1.0e3;		 // cm^-3
	amrex::Real T0 = 500.0;			 // K
	amrex::Real alphaB = 2.6e-13;		 // cm^3 s^-1
	amrex::Real ionizingPhotonRate = 1.0e49; // photons s^-1
	amrex::Real source_x = 0.0;		 // cm
	amrex::Real source_y = 0.0;		 // cm
	amrex::Real source_z = 0.0;		 // cm
	std::string stars_file = "hii_stars.txt";
	std::string qh0_file = "hii_qh0_table.h5";
	amrex::Real volume_rel_tol = 0.70; // geometric/discrete tolerance
	amrex::Real core_temp_tol = 0.70;  // core average T must be >= 0.70 * 1e4 K
};

template <> struct quokka::EOS_Traits<HIIRegionProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_p;
};

template <> struct HydroSystem_Traits<HIIRegionProblem> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Particle_Traits<HIIRegionProblem> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct Physics_Traits<HIIRegionProblem> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int nDustGroups = 0;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 0;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

namespace
{
void WriteQH0Table(std::string const &fname, amrex::Real const qh0_rate)
{
	hid_t const file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT(file >= 0);

	hid_t const g_grids = H5Gcreate2(file, "/grids", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	hid_t const g_data = H5Gcreate2(file, "/data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	hid_t const g_meta = H5Gcreate2(file, "/metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT(g_grids >= 0 && g_data >= 0 && g_meta >= 0);

	double const mass[2] = {1.0, 100.0};
	double const age[2] = {1.0e5, 1.0e9};
	double const qh0[4] = {qh0_rate, qh0_rate, qh0_rate, qh0_rate};

	hsize_t const d1[1] = {2};
	hid_t const s1 = H5Screate_simple(1, d1, nullptr);
	AMREX_ALWAYS_ASSERT(s1 >= 0);

	hid_t dset = H5Dcreate2(file, "/grids/mass", H5T_NATIVE_DOUBLE, s1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT(dset >= 0);
	AMREX_ALWAYS_ASSERT(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mass) >= 0);
	H5Dclose(dset);

	dset = H5Dcreate2(file, "/grids/age", H5T_NATIVE_DOUBLE, s1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT(dset >= 0);
	AMREX_ALWAYS_ASSERT(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, age) >= 0);
	H5Dclose(dset);
	H5Sclose(s1);

	hsize_t const d2[2] = {2, 2};
	hid_t const s2 = H5Screate_simple(2, d2, nullptr);
	AMREX_ALWAYS_ASSERT(s2 >= 0);
	dset = H5Dcreate2(file, "/data/QH0", H5T_NATIVE_DOUBLE, s2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT(dset >= 0);
	AMREX_ALWAYS_ASSERT(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, qh0) >= 0);
	H5Dclose(dset);
	H5Sclose(s2);

	int const n_mass = 2;
	int const n_age = 2;
	hid_t const scalar = H5Screate(H5S_SCALAR);
	AMREX_ALWAYS_ASSERT(scalar >= 0);

	hid_t attr = H5Acreate2(g_meta, "n_mass", H5T_NATIVE_INT, scalar, H5P_DEFAULT, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT(attr >= 0);
	AMREX_ALWAYS_ASSERT(H5Awrite(attr, H5T_NATIVE_INT, &n_mass) >= 0);
	H5Aclose(attr);

	attr = H5Acreate2(g_meta, "n_age", H5T_NATIVE_INT, scalar, H5P_DEFAULT, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT(attr >= 0);
	AMREX_ALWAYS_ASSERT(H5Awrite(attr, H5T_NATIVE_INT, &n_age) >= 0);
	H5Aclose(attr);
	H5Sclose(scalar);

	H5Gclose(g_meta);
	H5Gclose(g_data);
	H5Gclose(g_grids);
	H5Fclose(file);
}

void WriteSingleStarFile(std::string const &fname, amrex::Real source_x, amrex::Real source_y, amrex::Real source_z)
{
	std::ofstream ofs(fname);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs.good(), ("Failed to open star file: " + fname).c_str());

	// One particle. Format: x y z + all real components.
	// Real components:
	// mass vx vy vz birth_time death_time birth_x birth_y birth_z death_x death_y death_z death_density mass_at_birth lum0
	ofs << "1\n";
	ofs << source_x << " " << source_y << " " << source_z << " ";
	ofs << 30.0 * C::M_solar << " ";			      // mass
	ofs << "0 0 0 ";					      // velocity
	ofs << -1.0e12 << " ";					      // birth_time (ensures age > 0 at t=0)
	ofs << 1.0e30 << " ";					      // death_time
	ofs << source_x << " " << source_y << " " << source_z << " "; // birth pos
	ofs << source_x << " " << source_y << " " << source_z << " "; // death pos placeholder
	ofs << 1.0e-21 << " ";					      // death_density placeholder
	ofs << 30.0 * C::M_solar << " ";			      // mass_at_birth
	ofs << "0\n";						      // luminosity group 0 (unused here)
}
} // namespace

template <> void QuokkaSimulation<HIIRegionProblem>::createInitialStochasticStellarPopParticles()
{
	const int nreal_extra = quokka::StochasticStellarPopParticleRealComps<HIIRegionProblem>;
	StochasticStellarPopParticles->SetVerbose(0);
	StochasticStellarPopParticles->InitFromAsciiFile(userData_.stars_file, nreal_extra, nullptr);

	for (auto &kv : StochasticStellarPopParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_array = ikv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();
			if (np == 0) {
				continue;
			}
			auto *pdata = particle_array().data();
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
				p.idata(quokka::StochasticStellarPopParticleStageIdx) = static_cast<int>(quokka::StellarEvolutionStage::HighMassNonExploding);
			});
		}
	}
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<HIIRegionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Real nH0 = userData_.nH0;
	const amrex::Real T0 = userData_.T0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real rho = nH0 * C::m_p;
		const amrex::Real eint = quokka::EOS<HIIRegionProblem>::ComputeEintFromTgas(rho, T0, {});
		state_cc(i, j, k, HydroSystem<HIIRegionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<HIIRegionProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<HIIRegionProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<HIIRegionProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<HIIRegionProblem>::internalEnergy_index) = eint;
		state_cc(i, j, k, HydroSystem<HIIRegionProblem>::energy_index) = eint;
	});
}

auto problem_main() -> int
{
	SimulationData<HIIRegionProblem> inputData;
	amrex::ParmParse const pp("problem");
	pp.query("nH0", inputData.nH0);
	pp.query("T0", inputData.T0);
	pp.query("alphaB", inputData.alphaB);
	pp.query("ionizing_photon_rate", inputData.ionizingPhotonRate);
	pp.query("source_x", inputData.source_x);
	pp.query("source_y", inputData.source_y);
	pp.query("source_z", inputData.source_z);
	pp.query("stars_file", inputData.stars_file);
	pp.query("qh0_file", inputData.qh0_file);
	pp.query("volume_rel_tol", inputData.volume_rel_tol);
	pp.query("core_temp_tol", inputData.core_temp_tol);

	std::string stromgren_qh0_file = inputData.qh0_file;
	amrex::ParmParse const pp_particles("particles");
	pp_particles.query("stromgren_qh0_table_hdf5_file", stromgren_qh0_file);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		WriteQH0Table(stromgren_qh0_file, inputData.ionizingPhotonRate);
		WriteSingleStarFile(inputData.stars_file, inputData.source_x, inputData.source_y, inputData.source_z);
	}
	amrex::ParallelDescriptor::Barrier();

	auto BCs_cc = quokka::BC<HIIRegionProblem>(quokka::BCType::int_dir);
	QuokkaSimulation<HIIRegionProblem> sim(BCs_cc);
	sim.userData_ = inputData;

	sim.setInitialConditions();
	sim.evolve();

	const auto prob_lo = sim.geom[0].ProbLoArray();
	const auto dx = sim.geom[0].CellSizeArray();
	const amrex::Real source_x = sim.userData_.source_x;
	const amrex::Real source_y = sim.userData_.source_y;
	const amrex::Real source_z = sim.userData_.source_z;
	const amrex::Real nH = sim.userData_.nH0;
	const amrex::Real alphaB = sim.userData_.alphaB;
	const amrex::Real Q = sim.userData_.ionizingPhotonRate;
	const amrex::Real T_ion = 1.0e4;

	// Analytic Strömgren radius for uniform density.
	const amrex::Real rs = std::cbrt((3.0 * Q) / (4.0 * M_PI * alphaB * nH * nH));
	const amrex::Real core_radius = 0.8 * rs;

	const amrex::Real hot_volume =
	    sim.computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<HIIRegionProblem>::density_index);
		    const amrex::Real px = state(i, j, k, HydroSystem<HIIRegionProblem>::x1Momentum_index);
		    const amrex::Real py = state(i, j, k, HydroSystem<HIIRegionProblem>::x2Momentum_index);
		    const amrex::Real pz = state(i, j, k, HydroSystem<HIIRegionProblem>::x3Momentum_index);
		    const amrex::Real Egas = state(i, j, k, HydroSystem<HIIRegionProblem>::energy_index);
		    const amrex::Real Eint = RadSystem<HIIRegionProblem>::ComputeEintFromEgas(rho, px, py, pz, Egas);
		    const amrex::Real T = quokka::EOS<HIIRegionProblem>::ComputeTgasFromEint(rho, Eint, {});
		    return (T > 0.5 * T_ion) ? 1.0 : 0.0;
	    });

	const amrex::Real core_tmin_proxy =
	    sim.computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state) noexcept {
		    const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
		    const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
		    const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
		    const amrex::Real r = std::sqrt((x - source_x) * (x - source_x) + (y - source_y) * (y - source_y) + (z - source_z) * (z - source_z));
		    if (r > core_radius) {
			    return 0.0;
		    }
		    const amrex::Real rho = state(i, j, k, HydroSystem<HIIRegionProblem>::density_index);
		    const amrex::Real px = state(i, j, k, HydroSystem<HIIRegionProblem>::x1Momentum_index);
		    const amrex::Real py = state(i, j, k, HydroSystem<HIIRegionProblem>::x2Momentum_index);
		    const amrex::Real pz = state(i, j, k, HydroSystem<HIIRegionProblem>::x3Momentum_index);
		    const amrex::Real Egas = state(i, j, k, HydroSystem<HIIRegionProblem>::energy_index);
		    const amrex::Real Eint = RadSystem<HIIRegionProblem>::ComputeEintFromEgas(rho, px, py, pz, Egas);
		    const amrex::Real T = quokka::EOS<HIIRegionProblem>::ComputeTgasFromEint(rho, Eint, {});
		    return T;
	    });

	const amrex::Real core_vol_cells =
	    sim.computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const & /*state*/) noexcept {
		    const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
		    const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
		    const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
		    const amrex::Real r = std::sqrt((x - source_x) * (x - source_x) + (y - source_y) * (y - source_y) + (z - source_z) * (z - source_z));
		    return (r <= core_radius) ? 1.0 : 0.0;
	    });

	const amrex::Real v_exact = (4.0 / 3.0) * M_PI * rs * rs * rs;
	const amrex::Real vol_rel_error = std::abs(hot_volume - v_exact) / v_exact;

	const amrex::Real core_tavg = (core_vol_cells > 0.0) ? (core_tmin_proxy / core_vol_cells) : 0.0;
	const bool core_floor_ok = core_tavg >= (sim.userData_.core_temp_tol * T_ion);

	amrex::Print() << "HIIRegion (Strömgren floor):\n";
	amrex::Print() << "\tR_s (analytic) = " << rs << " cm\n";
	amrex::Print() << "\tHot volume rel. error = " << vol_rel_error << " (tol " << sim.userData_.volume_rel_tol << ")\n";
	amrex::Print() << "\tCore average T / 1e4 K = " << (core_tavg / T_ion) << " (threshold " << sim.userData_.core_temp_tol << ")\n";

	if ((vol_rel_error > sim.userData_.volume_rel_tol) || !core_floor_ok || std::isnan(vol_rel_error) || std::isnan(core_tavg)) {
		amrex::Print() << "HIIRegion test FAILED.\n";
		return 1;
	}
	amrex::Print() << "HIIRegion test passed.\n";
	return 0;
}
