// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testPopIII.cpp
/// \brief Defines a test problem for Pop III star formation.
/// Author: Piyush Sharda (Leiden University, 2023)
///
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "util/BC.hpp"
#include <array>
#include <fstream>
#include <string>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_FabArray.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_TableData.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "hydro/hydro_system.hpp"
#include "microphysics/microphysics_traits.hpp"
#include "networks/primordial_chem/network.hpp"
#include "radiation/radiation_system.hpp"
#include "turbulence/TurbDataReader.hpp"

#include "chemistry/Chemistry.hpp"
#include "microphysics/burn_state.hpp"

using amrex::Real;

struct PopIII {};

template <> struct Microphysics_Traits<PopIII> : DefaultMicrophysicsTraits {
	using Network = quokka::networks::PrimordialChem;
	static constexpr bool chemistry = true;
	using Tableau = quokka::math::rosenbrock::Rodas5P;
};

using Network = Microphysics_Traits<PopIII>::Network;
using amrex::literals::operator""_rt;

template <> struct HydroSystem_Traits<PopIII> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<PopIII> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr int numMassScalars = Network::NumSpec;	     // number of chemical species
	static constexpr int numPassiveScalars = numMassScalars + 0; // we only have mass scalars
};

template <> struct SimulationData<PopIII> {
	// real-space perturbation fields
	amrex::TableData<Real, 3> dvx;
	amrex::TableData<Real, 3> dvy;
	amrex::TableData<Real, 3> dvz;
	amrex::Real dv_rms_generated{};
	amrex::Real dv_rms_target{};
	amrex::Real rescale_factor{};

	// cloud parameters
	amrex::Real R_sphere{};
	amrex::Real numdens_init{};
	amrex::Real omega_sphere{};

	amrex::Real temperature{};
	/// the initial number density of each species, as a fraction of the cloud's (primordial_chem.primary_species_<n + 1>)
	std::array<amrex::Real, Network::NumSpec> primary_species{};
};

template <> void QuokkaSimulation<PopIII>::preCalculateInitialConditions()
{

	// parmparse species and temperature
	amrex::ParmParse const pp("primordial_chem");
	userData_.temperature = 1e1;
	pp.query("temperature", userData_.temperature);

	// defaults: all electrons (species 0), as before the port; the input keys number the species from 1
	userData_.primary_species.fill(0.0_rt);
	userData_.primary_species[Network::E] = 1.0_rt;
	for (int n = 0; n < Network::NumSpec; ++n) {
		pp.query(("primary_species_" + std::to_string(n + 1)), userData_.primary_species[n]);
	}

	static bool isSamplingDone = false;
	if (!isSamplingDone) {
		// read perturbations from file
		turb_data turbData;
		amrex::ParmParse const pp("perturb");
		std::string turbdata_filename;
		pp.query("filename", turbdata_filename);
		initialize_turbdata(turbData, turbdata_filename);

		// copy to pinned memory
		auto pinned_dvx = get_tabledata(turbData.dvx);
		auto pinned_dvy = get_tabledata(turbData.dvy);
		auto pinned_dvz = get_tabledata(turbData.dvz);

		// compute normalisation
		userData_.dv_rms_generated = computeRms(pinned_dvx, pinned_dvy, pinned_dvz);
		amrex::Print() << "rms dv = " << userData_.dv_rms_generated << "\n";

		Real rms_dv_target = NAN;
		pp.query("rms_velocity", rms_dv_target);
		const Real rms_dv_actual = userData_.dv_rms_generated;
		userData_.rescale_factor = rms_dv_target / rms_dv_actual;

		// copy to GPU
		userData_.dvx.resize(pinned_dvx.lo(), pinned_dvx.hi());
		userData_.dvx.copy(pinned_dvx);

		userData_.dvy.resize(pinned_dvy.lo(), pinned_dvy.hi());
		userData_.dvy.copy(pinned_dvy);

		userData_.dvz.resize(pinned_dvz.lo(), pinned_dvz.hi());
		userData_.dvz.copy(pinned_dvz);

		isSamplingDone = true;
	}
}

template <> void QuokkaSimulation<PopIII>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const std::array<Real, Network::NumSpec> numdens = userData_.primary_species;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	// cloud parameters
	const double R_sphere = userData_.R_sphere;
	const double omega_sphere = userData_.omega_sphere;
	const double renorm_amp = userData_.rescale_factor;
	const double numdens_init = userData_.numdens_init;
	const double core_temp = userData_.temperature;

	auto const &dvx = userData_.dvx.const_table();
	auto const &dvy = userData_.dvy.const_table();
	auto const &dvz = userData_.dvz.const_table();

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));
		amrex::Real const distxy = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2));

		quokka::microphysics::BurnState<Network::NumSpec> state;
		amrex::Real rhotot = 0.0;

		for (int n = 0; n < Network::NumSpec; ++n) {
			state.xn[n] = numdens[n] * numdens_init;
			rhotot += state.xn[n] * Network::SpeciesMass()[n];
		}

		// normalize -- just in case
		std::array<Real, Network::NumSpec> mfracs = {-1.0};
		Real msum = 0.0;
		for (int n = 0; n < Network::NumSpec; ++n) {
			mfracs[n] = state.xn[n] * Network::SpeciesMass()[n] / rhotot;
			msum += mfracs[n];
		}

		for (int n = 0; n < Network::NumSpec; ++n) {
			mfracs[n] /= msum;
			// use the normalized mass fractions to obtain the corresponding number densities
			state.xn[n] = mfracs[n] * rhotot / Network::SpeciesMass()[n];
		}

		// amrex::Print() << "cell " << i << j << k << " " << rhotot << " " << numdens_init << " " << numdens[0] << std::endl;

		amrex::Real const phi = atan2((y - y0), (x - x0));

		double vx = renorm_amp * dvx(i, j, k);
		double vy = renorm_amp * dvy(i, j, k);
		double const vz = renorm_amp * dvz(i, j, k);

		// calculate eos params for the core
		using EOS = quokka::EOSMultigamma<PopIII>;
		auto eos_state = EOS::FromRhoT(rhotot, core_temp, state.xn);

		if (r <= R_sphere) {
			// add rotation to vx and vy
			vx += (-1.0) * distxy * omega_sphere * std::sin(phi);
			vy += distxy * omega_sphere * std::cos(phi);

		} else {
			// re-calculate eos params outside the core, using pressure equilibrium (so, pressure within the core = pressure outside)
			eos_state = EOS::FromRhoP(0.01 * rhotot, EOS::PressureOf(eos_state), state.xn);
		}
		state.rho = eos_state.rho;
		state.e = eos_state.e;

		// call the EOS to set initial internal energy e
		amrex::Real const e = state.rho * state.e;

		// amrex::Print() << "cell " << i << j << k << " " << state.rho << " " << state.T << " " << e << std::endl;

		state_cc(i, j, k, HydroSystem<PopIII>::density_index) = state.rho;
		state_cc(i, j, k, HydroSystem<PopIII>::x1Momentum_index) = state.rho * vx;
		state_cc(i, j, k, HydroSystem<PopIII>::x2Momentum_index) = state.rho * vy;
		state_cc(i, j, k, HydroSystem<PopIII>::x3Momentum_index) = state.rho * vz;
		state_cc(i, j, k, HydroSystem<PopIII>::internalEnergy_index) = e;

		static_assert(!Physics_Traits<PopIII>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
		Real const Egas = quokka::EOS<PopIII>::ComputeEgasFromEint(state.rho, state.rho * vx, state.rho * vy, state.rho * vz, e, 0.0);
		state_cc(i, j, k, HydroSystem<PopIII>::energy_index) = Egas;

		for (int nn = 0; nn < Network::NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<PopIII>::scalar0_index + nn) = mfracs[nn] * state.rho; // we use partial densities and not mass fractions
		}
	});
}

template <> void QuokkaSimulation<PopIII>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{

	// read-in jeans length refinement runtime params
	amrex::ParmParse const pp("jeansRefine");
	int N_cells = 0;
	pp.query("ncells", N_cells); // inverse of the 'Jeans number' [Truelove et al. (1997)]
	Real jeans_density_threshold = NAN;
	pp.query("density_threshold", jeans_density_threshold);

	const amrex::Real G = Gconst_;
	const amrex::Real dx = geom[lev].CellSizeArray()[0];

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<PopIII>::density_index;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			Real const rho = state(i, j, k, nidx);
			Real const pressure = HydroSystem<PopIII>::ComputePressure(state, i, j, k);
			amrex::GpuArray<Real, Physics_Traits<PopIII>::numMassScalars> massScalars = RadSystem<PopIII>::ComputeMassScalars(state, i, j, k);

			amrex::Real const cs = quokka::EOS<PopIII>::ComputeSoundSpeed(rho, pressure, massScalars);

			const amrex::Real l_Jeans = cs * std::sqrt(M_PI / (G * rho));
			// add a density criterion for refinement so that no initial refinement is ever triggered outside the core
			// typically, a density threshold ~ initial core density works well
			if (l_Jeans < (N_cells * dx) && rho > jeans_density_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

template <>
void QuokkaSimulation<PopIII>::ComputeDerivedVar(int /*lev*/, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
						 amrex::MultiFab const &state_cc, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const & /*state_fc*/) const
{
	// compute derived variables and save in 'mf'
	if (dname == "temperature") {
		const int ncomp = ncomp_cc_in;
		auto const &state = state_cc.const_arrays();
		auto output = mf.arrays();

		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<PopIII>::density_index);
			amrex::Real const Eint = state[bx](i, j, k, HydroSystem<PopIII>::internalEnergy_index);

			amrex::GpuArray<Real, Physics_Traits<PopIII>::numMassScalars> massScalars = RadSystem<PopIII>::ComputeMassScalars(state[bx], i, j, k);

			output[bx](i, j, k, ncomp) = quokka::EOS<PopIII>::ComputeTgasFromEint(rho, Eint, massScalars);
		});
	}

	if (dname == "pressure") {

		const int ncomp = ncomp_cc_in;
		auto const &state = state_cc.const_arrays();
		auto output = mf.arrays();

		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			amrex::Real const Pgas = HydroSystem<PopIII>::ComputePressure(state[bx], i, j, k);
			output[bx](i, j, k, ncomp) = Pgas;
		});
	}

	if (dname == "velx") {

		const int ncomp = ncomp_cc_in;
		auto const &state = state_cc.const_arrays();
		auto output = mf.arrays();

		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<PopIII>::density_index);
			Real const xmom = state[bx](i, j, k, HydroSystem<PopIII>::x1Momentum_index);
			output[bx](i, j, k, ncomp) = xmom / rho;
		});
	}

	if (dname == "sound_speed") {

		const int ncomp = ncomp_cc_in;
		auto const &state = state_cc.const_arrays();
		auto output = mf.arrays();

		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<PopIII>::density_index);
			Real const pressure = HydroSystem<PopIII>::ComputePressure(state[bx], i, j, k);
			amrex::GpuArray<Real, Physics_Traits<PopIII>::numMassScalars> massScalars = RadSystem<PopIII>::ComputeMassScalars(state[bx], i, j, k);

			amrex::Real const cs = quokka::EOS<PopIII>::ComputeSoundSpeed(rho, pressure, massScalars);
			output[bx](i, j, k, ncomp) = cs;
		});
	}
}

auto problem_main() -> int
{
	// read problem parameters
	amrex::ParmParse const pp("perturb");

	// cloud radius
	Real R_sphere{};
	pp.query("cloud_radius", R_sphere);

	// cloud density
	Real numdens_init{};
	pp.query("cloud_numdens", numdens_init);

	// cloud angular velocity
	Real omega_sphere{};
	pp.query("cloud_omega", omega_sphere);

	// Problem initialization
	QuokkaSimulation<PopIII> sim;

	sim.tempFloor_ = 2.73 * (30.0 + 1.0);
	// sim.speedCeiling_ = 3e6;

	sim.userData_.R_sphere = R_sphere;
	sim.userData_.numdens_init = numdens_init;
	sim.userData_.omega_sphere = omega_sphere;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	int const status = 0;
	return status;
}
