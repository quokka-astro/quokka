/// \file testHIIRegion.cpp
/// \brief Strömgren volume test with analytic radius comparison.
///

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "util/BC.hpp"

#include <array>
#include <cmath>

struct HIIRegionProblem {
};

template <> struct SimulationData<HIIRegionProblem> {
	amrex::Real nH0 = 1.0e3; // cm^-3
	amrex::Real T0 = 500.0;	 // K
	std::string stars_file = "hii_stars.txt";
	amrex::Real volume_rel_tol = 0.70; // geometric/discrete tolerance
	amrex::Real core_temp_tol = 0.70;  // core average T must be >= 0.70 * 1e4 K
};

namespace
{
constexpr amrex::Real alphaB = 2.6e-13; // cm^3 s^-1
} // namespace

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

template <> void QuokkaSimulation<HIIRegionProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "temperature") {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coolingTableType_ == "resampled", "HIIRegion diagnostics require resampled cooling tables.");
		const int ncomp = ncomp_cc_in;
		auto tables = resampledTables_.const_tables();

		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);

			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<HIIRegionProblem>::density_index);
				Real const x1Mom = state(i, j, k, HydroSystem<HIIRegionProblem>::x1Momentum_index);
				Real const x2Mom = state(i, j, k, HydroSystem<HIIRegionProblem>::x2Momentum_index);
				Real const x3Mom = state(i, j, k, HydroSystem<HIIRegionProblem>::x3Momentum_index);
				Real const Egas = state(i, j, k, HydroSystem<HIIRegionProblem>::energy_index);
				Real const Eint = RadSystem<HIIRegionProblem>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
				Real const Tgas = quokka::ResampledCooling::ComputeTgasFromEgas(rho, Eint, tables);
				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

auto problem_main() -> int
{
	SimulationData<HIIRegionProblem> inputData;
	amrex::ParmParse const pp("problem");
	pp.query("nH0", inputData.nH0);
	pp.query("T0", inputData.T0);
	pp.query("stars_file", inputData.stars_file);
	pp.query("volume_rel_tol", inputData.volume_rel_tol);
	pp.query("core_temp_tol", inputData.core_temp_tol);

	std::string stromgren_qh0_file = "../extern/stellar_tables/QH0_mist_9to120_quokka_best.h5";
	amrex::ParmParse const pp_particles("particles");
	pp_particles.query("stromgren_qh0_table_hdf5_file", stromgren_qh0_file);

	QuokkaSimulation<HIIRegionProblem> sim;
	sim.userData_ = inputData;
	sim.setInitialConditions();
	sim.evolve();

	const auto prob_lo = sim.geom[0].ProbLoArray();
	const auto dx = sim.geom[0].CellSizeArray();
	const amrex::Real nH = sim.userData_.nH0;
	const amrex::Real current_time = sim.tNew_[0];
	const auto qh0_table = sim.stochasticStellarPopQH0Table_.const_tables();
	auto *stellar_desc = sim.GetParticleRegister().getParticleDescriptor(quokka::ParticleType::StochasticStellarPop);
	auto const [particle_ids, real_data, int_data] = stellar_desc->getParticleDataAtAllLevels();
	amrex::ignore_unused(particle_ids);

	amrex::Real Q = 0.0; // total ionizing photon luminosity from all stars
	if (amrex::ParallelDescriptor::IOProcessor()) {
		for (std::size_t i = 0; i < real_data.size(); ++i) {
			auto const &r = real_data[i];
			auto const &idata = int_data[i];
			amrex::Real const age = current_time - r[AMREX_SPACEDIM + quokka::StochasticStellarPopParticleBirthTimeIdx];
			amrex::Real const zams_mass = r[AMREX_SPACEDIM + quokka::StochasticStellarPopParticleMassAtBirthIdx];
			amrex::Real const mass_coord = zams_mass / C::M_solar;
			amrex::Real const age_coord = age / 3.15576e7;

			std::array<amrex::Real, 2> point{};
			if (sim.stochastic_stellar_pop_qh0_table_axes_are_mass_age_) {
				point = {mass_coord, age_coord};
			} else {
				point = {age_coord, mass_coord};
			}
			amrex::Real const S = qh0_table.interpolate_single(point, 0);
			Q += S;
		}
	}
	amrex::ParallelDescriptor::Bcast(&Q, 1, amrex::ParallelDescriptor::IOProcessorNumber());
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
		    const amrex::Real r = std::sqrt(x * x + y * y + z * z);
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
		    const amrex::Real r = std::sqrt(x * x + y * y + z * z);
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
