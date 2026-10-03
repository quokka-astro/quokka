//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMolecularCloudDisruption.cpp
/// \brief Defines an isolated molecular-cloud disruption benchmark with stellar feedback.

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <string>

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "particles/particle_deposition.hpp"
#include "particles/particle_types.hpp"

namespace
{
constexpr amrex::Real seconds_per_myr = 1.0e6 * quokka::seconds_per_year;
constexpr amrex::Real hydrogen_mass = C::m_p + C::m_e;
constexpr amrex::Real cloudy_H_mass_fraction = 1.0 / (1.0 + 0.1 * 3.971);
constexpr amrex::Real sn_ejecta_mass = 10.0 * C::M_solar;

struct MolecularCloudDisruption {
};

struct CloudProblemParameters {
	amrex::Real cloudMassMsun = 1.0e5;
	amrex::Real cloudRadiusPc = 20.0;
	amrex::Real cloudTemperature = 100.0;
	amrex::Real densityContrast = 100.0;
	amrex::Real edgeWidthPc = 1.0;
	amrex::Real virialParameter = 2.0;
	amrex::Real stellarMassMsun = 2.0e3;
	amrex::Real denseThresholdNH = 50.0;
	amrex::Real coldThresholdTemperature = 300.0;
	amrex::Real refineCloudFraction = 0.01;
	bool allowStarFormation = false;
	std::string stellarParticlesFile = "../inputs/MolecularCloudDisruption_particles.txt";
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rawTurbulentVelocity(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real radius)
    -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
{
	const amrex::Real phase = std::numbers::pi_v<amrex::Real> / radius;
	return {std::sin(phase * y) + 0.5 * std::sin(2.0 * phase * z), std::sin(phase * z) + 0.5 * std::sin(2.0 * phase * x),
		std::sin(phase * x) + 0.5 * std::sin(2.0 * phase * y)};
}

auto rawTurbulentVelocityRms() -> amrex::Real
{
	// Integrate the analytic field over a unit-density sphere. This normalization is
	// independent of the mesh, so resolution comparisons use the same continuum IC.
	constexpr int num_intervals = 8192;
	constexpr amrex::Real dq = 2.0 / static_cast<amrex::Real>(num_intervals);
	amrex::Real mean_sin2_k1 = 0.0;
	amrex::Real mean_sin2_k2 = 0.0;
	for (int n = 0; n < num_intervals; ++n) {
		const amrex::Real q = -1.0 + (static_cast<amrex::Real>(n) + 0.5) * dq;
		const amrex::Real spherical_weight = 0.75 * (1.0 - q * q);
		mean_sin2_k1 += spherical_weight * std::pow(std::sin(std::numbers::pi_v<amrex::Real> * q), 2) * dq;
		mean_sin2_k2 += spherical_weight * std::pow(std::sin(2.0 * std::numbers::pi_v<amrex::Real> * q), 2) * dq;
	}
	return std::sqrt(3.0 * (mean_sin2_k1 + 0.25 * mean_sin2_k2));
}
} // namespace

template <> struct Physics_Traits<MolecularCloudDisruption> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr int numMassScalars = 2;
	static constexpr int numPassiveScalars = numMassScalars;
};

template <> struct HydroSystem_Traits<MolecularCloudDisruption> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct quokka::EOS_Traits<MolecularCloudDisruption> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
	using EOSBackend = quokka::EOSTabulated<MolecularCloudDisruption>;
};

template <> struct Particle_Traits<MolecularCloudDisruption> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct SimulationData<MolecularCloudDisruption> {
	CloudProblemParameters params{};
	amrex::Real cloudRadius = 0.0;
	amrex::Real edgeWidth = 0.0;
	amrex::Real cloudDensity = 0.0;
	amrex::Real ambientDensity = 0.0;
	amrex::Real equilibriumPressure = 0.0;
	amrex::Real turbulentVelocityScale = 0.0;
	amrex::Real turbulentVelocityRms = 0.0;
	amrex::Real freefallTime = 0.0;
};

template <> void QuokkaSimulation<MolecularCloudDisruption>::preCalculateInitialConditions()
{
	const auto &params = userData_.params;
	userData_.cloudRadius = params.cloudRadiusPc * C::parsec;
	userData_.edgeWidth = params.edgeWidthPc * C::parsec;
	const amrex::Real cloud_mass = params.cloudMassMsun * C::M_solar;
	const amrex::Real stellar_mass = params.stellarMassMsun * C::M_solar;
	const amrex::Real cloud_volume = (4.0 / 3.0) * std::numbers::pi_v<amrex::Real> * std::pow(userData_.cloudRadius, 3);
	userData_.cloudDensity = cloud_mass / cloud_volume;
	userData_.ambientDensity = userData_.cloudDensity / params.densityContrast;

	const amrex::Real cloud_eint = quokka::EOS<MolecularCloudDisruption>::ComputeEintFromTgas(userData_.cloudDensity, params.cloudTemperature);
	userData_.equilibriumPressure = quokka::EOS<MolecularCloudDisruption>::ComputePressure(userData_.cloudDensity, cloud_eint);

	const amrex::Real isothermal_sound_speed_sq = userData_.equilibriumPressure / userData_.cloudDensity;
	const amrex::Real gravitating_mass = cloud_mass + stellar_mass;
	const amrex::Real target_1d_turbulent_variance =
	    params.virialParameter * C::Gconst * gravitating_mass / (5.0 * userData_.cloudRadius) - isothermal_sound_speed_sq;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(target_1d_turbulent_variance > 0.0,
					 "The requested virial parameter is already supplied by thermal support; no positive turbulent variance remains.");
	userData_.turbulentVelocityRms = std::sqrt(3.0 * target_1d_turbulent_variance);
	userData_.turbulentVelocityScale = userData_.turbulentVelocityRms / rawTurbulentVelocityRms();
	userData_.freefallTime = std::sqrt(3.0 * std::numbers::pi_v<amrex::Real> / (32.0 * C::Gconst * userData_.cloudDensity));
}

template <> void QuokkaSimulation<MolecularCloudDisruption>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const auto dx = grid_elem.dx_;
	const auto prob_lo = grid_elem.prob_lo_;
	const auto prob_hi = grid_elem.prob_hi_;
	const amrex::Box &index_range = grid_elem.indexRange_;
	const amrex::Array4<amrex::Real> &state = grid_elem.array_;

	const amrex::Real center_x = 0.5 * (prob_lo[0] + prob_hi[0]);
	const amrex::Real center_y = 0.5 * (prob_lo[1] + prob_hi[1]);
	const amrex::Real center_z = 0.5 * (prob_lo[2] + prob_hi[2]);
	const amrex::Real radius = userData_.cloudRadius;
	const amrex::Real edge_width = userData_.edgeWidth;
	const amrex::Real rho_cloud = userData_.cloudDensity;
	const amrex::Real rho_ambient = userData_.ambientDensity;
	const amrex::Real pressure = userData_.equilibriumPressure;
	const amrex::Real velocity_scale = userData_.turbulentVelocityScale;

	amrex::ParallelFor(index_range, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0] - center_x;
		const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1] - center_y;
		const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2] - center_z;
		const amrex::Real r = std::sqrt(x * x + y * y + z * z);
		const amrex::Real transition = 0.5 * (1.0 + std::tanh((radius - r) / edge_width));

		const amrex::Real partial_cloud_density = rho_cloud * transition;
		const amrex::Real partial_ambient_density = rho_ambient * (1.0 - transition);
		const amrex::Real rho = partial_cloud_density + partial_ambient_density;
		const amrex::Real cloud_mass_fraction = partial_cloud_density / rho;

		auto velocity = rawTurbulentVelocity(x, y, z, radius);
		for (amrex::Real &component : velocity) {
			component *= velocity_scale * cloud_mass_fraction;
		}

		const amrex::Real xmom = rho * velocity[0];
		const amrex::Real ymom = rho * velocity[1];
		const amrex::Real zmom = rho * velocity[2];
		const amrex::Real eint = quokka::EOS<MolecularCloudDisruption>::ComputeEintFromPres(rho, pressure);
		const amrex::Real kinetic_energy = 0.5 * (xmom * xmom + ymom * ymom + zmom * zmom) / rho;

		state(i, j, k, HydroSystem<MolecularCloudDisruption>::density_index) = rho;
		state(i, j, k, HydroSystem<MolecularCloudDisruption>::x1Momentum_index) = xmom;
		state(i, j, k, HydroSystem<MolecularCloudDisruption>::x2Momentum_index) = ymom;
		state(i, j, k, HydroSystem<MolecularCloudDisruption>::x3Momentum_index) = zmom;
		state(i, j, k, HydroSystem<MolecularCloudDisruption>::energy_index) = eint + kinetic_energy;
		state(i, j, k, HydroSystem<MolecularCloudDisruption>::internalEnergy_index) = eint;
		state(i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index) = partial_cloud_density;
		state(i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index + 1) = partial_ambient_density;
	});
}

template <> void QuokkaSimulation<MolecularCloudDisruption>::createInitialStochasticStellarPopParticles()
{
	const int nreal_extra = quokka::StochasticStellarPopParticleRealComps<MolecularCloudDisruption>;
	StochasticStellarPopParticles->SetVerbose(quokka::particle_verbose);
	StochasticStellarPopParticles->InitFromAsciiFile(userData_.params.stellarParticlesFile, nreal_extra, nullptr);

	for (auto &level_entry : StochasticStellarPopParticles->GetParticles()) {
		for (auto &tile_entry : level_entry) {
			auto &particle_array = tile_entry.second.GetArrayOfStructs();
			const int num_particles = particle_array.numParticles();
			if (num_particles == 0) {
				continue;
			}
			auto *particles = particle_array().data();
			amrex::ParallelFor(num_particles, [=] AMREX_GPU_DEVICE(int idx) noexcept {
				auto &particle = particles[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				const amrex::Real death_time = particle.rdata(quokka::StochasticStellarPopParticleDeathTimeIdx);
				const bool is_composite = death_time > 0.5 * std::numeric_limits<amrex::Real>::max();
				particle.idata(quokka::StochasticStellarPopParticleStageIdx) = static_cast<int>(
				    is_composite ? quokka::StellarEvolutionStage::LowMassComposite : quokka::StellarEvolutionStage::SNProgenitor);
			});
		}
	}
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<MolecularCloudDisruption>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	const amrex::Real threshold = userData_.params.refineCloudFraction;
	const auto state = state_new_cc_[lev].const_arrays();
	auto tag = tags.arrays();
	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real rho = state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
		const amrex::Real rho_cloud = state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index);
		if (rho_cloud / rho > threshold) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
}

template <>
void QuokkaSimulation<MolecularCloudDisruption>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp,
								   amrex::MultiFab const &state_cc,
								   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const & /*state_fc*/) const
{
	auto output = mf.arrays();
	auto state = state_cc.const_arrays();

	if (dname == "gpot") {
		auto const potential = phi[lev].const_arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = potential[bx](i, j, k); });
	} else if (dname == "temperature") {
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const amrex::Real rho = state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
			const amrex::Real xmom = state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::x1Momentum_index);
			const amrex::Real ymom = state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::x2Momentum_index);
			const amrex::Real zmom = state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::x3Momentum_index);
			const amrex::Real egas = state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::energy_index);
			const amrex::Real eint = quokka::EOS<MolecularCloudDisruption>::ComputeEintFromEgas(rho, xmom, ymom, zmom, egas, 0.0);
			output[bx](i, j, k, ncomp) = quokka::EOS<MolecularCloudDisruption>::ComputeTgasFromEint(rho, eint);
		});
	} else if (dname == "cloud_fraction" || dname == "ambient_fraction") {
		const int scalar_offset = (dname == "cloud_fraction") ? 0 : 1;
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const amrex::Real rho = state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
			output[bx](i, j, k, ncomp) = state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index + scalar_offset) / rho;
		});
	} else if (dname == "nH") {
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const amrex::Real rho = state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
			output[bx](i, j, k, ncomp) = rho * cloudy_H_mass_fraction / hydrogen_mass;
		});
	} else if (dname == "radial_velocity") {
		const auto prob_lo = geom[lev].ProbLoArray();
		const auto prob_hi = geom[lev].ProbHiArray();
		const auto dx = geom[lev].CellSizeArray();
		const amrex::Real center_x = 0.5 * (prob_lo[0] + prob_hi[0]);
		const amrex::Real center_y = 0.5 * (prob_lo[1] + prob_hi[1]);
		const amrex::Real center_z = 0.5 * (prob_lo[2] + prob_hi[2]);
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0] - center_x;
			const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1] - center_y;
			const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2] - center_z;
			const amrex::Real radius = std::sqrt(x * x + y * y + z * z);
			const amrex::Real rho = state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
			const amrex::Real radial_momentum = x * state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::x1Momentum_index) +
							    y * state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::x2Momentum_index) +
							    z * state[bx](i, j, k, HydroSystem<MolecularCloudDisruption>::x3Momentum_index);
			output[bx](i, j, k, ncomp) = (radius > 0.0) ? radial_momentum / (rho * radius) : 0.0;
		});
	} else {
		amrex::Abort("Unknown MolecularCloudDisruption derived variable: " + dname);
	}
	amrex::Gpu::streamSynchronize();
}

template <> auto QuokkaSimulation<MolecularCloudDisruption>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	using FaceStateArray = std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM>;
	std::map<std::string, amrex::Real> stats;

	const amrex::Real gas_mass =
	    amrex::volumeWeightedSum(amrex::GetVecOfConstPtrs(state_new_cc_), HydroSystem<MolecularCloudDisruption>::density_index, geom, ref_ratio);
	const amrex::Real cloud_mass =
	    amrex::volumeWeightedSum(amrex::GetVecOfConstPtrs(state_new_cc_), HydroSystem<MolecularCloudDisruption>::scalar0_index, geom, ref_ratio);
	const amrex::Real ambient_mass =
	    amrex::volumeWeightedSum(amrex::GetVecOfConstPtrs(state_new_cc_), HydroSystem<MolecularCloudDisruption>::scalar0_index + 1, geom, ref_ratio);
	const amrex::Real dense_threshold_nh = userData_.params.denseThresholdNH;
	const amrex::Real cold_threshold_temperature = userData_.params.coldThresholdTemperature;

	const amrex::Real dense_cloud_mass = computeVolumeIntegral(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
		    const amrex::Real rho_cloud = state(i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index);
		    const amrex::Real n_h = rho * cloudy_H_mass_fraction / hydrogen_mass;
		    return (n_h > dense_threshold_nh) ? rho_cloud : 0.0;
	    });
	const amrex::Real cold_cloud_mass = computeVolumeIntegral(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
		    const amrex::Real xmom = state(i, j, k, HydroSystem<MolecularCloudDisruption>::x1Momentum_index);
		    const amrex::Real ymom = state(i, j, k, HydroSystem<MolecularCloudDisruption>::x2Momentum_index);
		    const amrex::Real zmom = state(i, j, k, HydroSystem<MolecularCloudDisruption>::x3Momentum_index);
		    const amrex::Real egas = state(i, j, k, HydroSystem<MolecularCloudDisruption>::energy_index);
		    const amrex::Real eint = quokka::EOS<MolecularCloudDisruption>::ComputeEintFromEgas(rho, xmom, ymom, zmom, egas, 0.0);
		    const amrex::Real temperature = quokka::EOS<MolecularCloudDisruption>::ComputeTgasFromEint(rho, eint);
		    return (temperature < cold_threshold_temperature) ? state(i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index) : 0.0;
	    });
	const amrex::Real dense_cold_cloud_mass = computeVolumeIntegral(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
		    const amrex::Real xmom = state(i, j, k, HydroSystem<MolecularCloudDisruption>::x1Momentum_index);
		    const amrex::Real ymom = state(i, j, k, HydroSystem<MolecularCloudDisruption>::x2Momentum_index);
		    const amrex::Real zmom = state(i, j, k, HydroSystem<MolecularCloudDisruption>::x3Momentum_index);
		    const amrex::Real egas = state(i, j, k, HydroSystem<MolecularCloudDisruption>::energy_index);
		    const amrex::Real eint = quokka::EOS<MolecularCloudDisruption>::ComputeEintFromEgas(rho, xmom, ymom, zmom, egas, 0.0);
		    const amrex::Real temperature = quokka::EOS<MolecularCloudDisruption>::ComputeTgasFromEint(rho, eint);
		    const amrex::Real n_h = rho * cloudy_H_mass_fraction / hydrogen_mass;
		    return (temperature < cold_threshold_temperature && n_h > dense_threshold_nh)
			       ? state(i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index)
			       : 0.0;
	    });
	const amrex::Real cloud_kinetic_energy = computeVolumeIntegral(
	    [] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
		    const amrex::Real rho_cloud = state(i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index);
		    const amrex::Real xmom = state(i, j, k, HydroSystem<MolecularCloudDisruption>::x1Momentum_index);
		    const amrex::Real ymom = state(i, j, k, HydroSystem<MolecularCloudDisruption>::x2Momentum_index);
		    const amrex::Real zmom = state(i, j, k, HydroSystem<MolecularCloudDisruption>::x3Momentum_index);
		    return 0.5 * (rho_cloud / rho) * (xmom * xmom + ymom * ymom + zmom * zmom) / rho;
	    });
	const amrex::Real cloud_temperature_mass_integral = computeVolumeIntegral(
	    [] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
		    const amrex::Real xmom = state(i, j, k, HydroSystem<MolecularCloudDisruption>::x1Momentum_index);
		    const amrex::Real ymom = state(i, j, k, HydroSystem<MolecularCloudDisruption>::x2Momentum_index);
		    const amrex::Real zmom = state(i, j, k, HydroSystem<MolecularCloudDisruption>::x3Momentum_index);
		    const amrex::Real egas = state(i, j, k, HydroSystem<MolecularCloudDisruption>::energy_index);
		    const amrex::Real eint = quokka::EOS<MolecularCloudDisruption>::ComputeEintFromEgas(rho, xmom, ymom, zmom, egas, 0.0);
		    const amrex::Real temperature = quokka::EOS<MolecularCloudDisruption>::ComputeTgasFromEint(rho, eint);
		    return state(i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index) * temperature;
	    });
	const amrex::Real scalar_closure_l1 = computeVolumeIntegral(
	    [] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
		    const amrex::Real scalar_sum = state(i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index) +
						   state(i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index + 1);
		    return std::abs(scalar_sum - rho);
	    });

	stats["ambient_mass_Msun"] = ambient_mass / C::M_solar;
	stats["cloud_cold_dense_fraction"] = dense_cold_cloud_mass / cloud_mass;
	stats["cloud_cold_fraction"] = cold_cloud_mass / cloud_mass;
	stats["cloud_dense_fraction"] = dense_cloud_mass / cloud_mass;
	stats["cloud_kinetic_energy_erg"] = cloud_kinetic_energy;
	stats["cloud_mass_Msun"] = cloud_mass / C::M_solar;
	stats["cloud_mean_temperature_K"] = cloud_temperature_mass_integral / cloud_mass;
	stats["cloud_rms_speed_km_s"] = std::sqrt(2.0 * cloud_kinetic_energy / cloud_mass) / 1.0e5;
	stats["emf_active_particle_count"] = static_cast<amrex::Real>(emf_active_particle_count_);
	stats["emf_momentum_requested_step_Msun_kmps"] = emf_momentum_requested_ / (C::M_solar * 1.0e5);
	stats["feedback_coupling_radius_pc"] = quokka::SN_stencil_size * geom[finest_level].CellSizeArray()[0] / C::parsec;
	stats["gas_mass_Msun"] = gas_mass / C::M_solar;
	stats["scalar_closure_relative_L1"] = scalar_closure_l1 / gas_mass;
	stats["sn_count_cumulative"] = static_cast<amrex::Real>(sn_count_cumulative_);
	stats["t_over_tff"] = tNew_[0] / userData_.freefallTime;
	return stats;
}

auto problem_main() -> int
{
	CloudProblemParameters params;
	amrex::ParmParse const pp("problem");
	pp.query("cloud_mass_Msun", params.cloudMassMsun);
	pp.query("cloud_radius_pc", params.cloudRadiusPc);
	pp.query("cloud_temperature", params.cloudTemperature);
	pp.query("density_contrast", params.densityContrast);
	pp.query("edge_width_pc", params.edgeWidthPc);
	pp.query("virial_parameter", params.virialParameter);
	pp.query("stellar_mass_Msun", params.stellarMassMsun);
	pp.query("dense_threshold_nH", params.denseThresholdNH);
	pp.query("cold_threshold_temperature", params.coldThresholdTemperature);
	pp.query("refine_cloud_fraction", params.refineCloudFraction);
	pp.query("allow_star_formation", params.allowStarFormation);
	pp.query("stellar_particles_file", params.stellarParticlesFile);

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(params.cloudMassMsun > 0.0, "problem.cloud_mass_Msun must be positive.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(params.cloudRadiusPc > 0.0, "problem.cloud_radius_pc must be positive.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(params.cloudTemperature > 0.0, "problem.cloud_temperature must be positive.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(params.densityContrast > 1.0, "problem.density_contrast must exceed one.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(params.edgeWidthPc > 0.0, "problem.edge_width_pc must be positive.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(params.virialParameter > 0.0, "problem.virial_parameter must be positive.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!params.stellarParticlesFile.empty(), "problem.stellar_particles_file must be specified.");

	QuokkaSimulation<MolecularCloudDisruption> sim;
	sim.userData_.params = params;
	sim.setInitialConditions();

	auto *stellar_descriptor = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop);
	stellar_descriptor->setForceFinestLevel(true);
	stellar_descriptor->setAllowsCreation(params.allowStarFormation);

	const amrex::Real stellar_mass_at_birth = stellar_descriptor->computeStellarMassAtBirth();
	const amrex::Real requested_stellar_mass = params.stellarMassMsun * C::M_solar;
	amrex::Print() << "Loaded stellar birth mass = " << stellar_mass_at_birth / C::M_solar << " Msun (requested " << params.stellarMassMsun << " Msun)\n";
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(stellar_mass_at_birth - requested_stellar_mass) / requested_stellar_mass < 1.0e-12,
					 "The stellar particle file birth mass does not equal problem.stellar_mass_Msun.");

	using FaceStateArray = std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM>;
	const amrex::Real gas_mass = sim.computeVolumeIntegral(
	    [] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    return state(i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
	    });
	const amrex::Real cloud_mass = sim.computeVolumeIntegral(
	    [] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    return state(i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index);
	    });
	const amrex::Real requested_cloud_mass = params.cloudMassMsun * C::M_solar;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(cloud_mass - requested_cloud_mass) / requested_cloud_mass < 0.05,
					 "The mesh-integrated initial cloud mass differs from problem.cloud_mass_Msun by more than 5 percent.");

	const amrex::Real closure_error = sim.computeVolumeIntegral(
	    [] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<MolecularCloudDisruption>::density_index);
		    const amrex::Real scalar_sum = state(i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index) +
						   state(i, j, k, HydroSystem<MolecularCloudDisruption>::scalar0_index + 1);
		    return std::abs(scalar_sum - rho);
	    });
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(closure_error / gas_mass < 1.0e-14, "Initial MassScalars do not sum to the gas density.");

	if (!quokka::disable_SN_feedback) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(quokka::scalar_yield_per_SN - sn_ejecta_mass) / sn_ejecta_mass < 1.0e-12,
						 "For this two-origin tracer, particles.scalar_yield_per_SN must equal the 10 Msun SN ejecta mass.");
	}

	const amrex::Real dx_pc = sim.geom[0].CellSizeArray()[0] / C::parsec;
	amrex::Print() << "\nMolecularCloudDisruption initial conditions:\n"
		       << "  mesh cloud mass = " << cloud_mass / C::M_solar << " Msun\n"
		       << "  stellar birth mass = " << stellar_mass_at_birth / C::M_solar << " Msun\n"
		       << "  cloud density = " << sim.userData_.cloudDensity << " g cm^-3\n"
		       << "  free-fall time = " << sim.userData_.freefallTime / seconds_per_myr << " Myr\n"
		       << "  turbulent 3D RMS speed = " << sim.userData_.turbulentVelocityRms / 1.0e5 << " km s^-1\n"
		       << "  cell width = " << dx_pc << " pc\n"
		       << "  three-cell feedback radius = " << quokka::SN_stencil_size * dx_pc << " pc\n"
		       << "  on-the-fly star formation = " << (params.allowStarFormation ? "enabled" : "disabled") << "\n\n";

	sim.evolve();
	return 0;
}
