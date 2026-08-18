//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testDTypeFrontVC.cpp
/// \brief Direct photoionization O(v/c) momentum-coupling test.
///

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "AMReX_Reduce.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include "radiation/photochemistry.hpp"
#include "radiation/radiation_system.hpp"
#include <array>
#include <cmath>
#include <limits>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct DTypeFrontVC {
};

namespace
{
constexpr double c_hat = C::c_light / 1000.0;
constexpr amrex::Real initial_photon_number_density = 10.0_rt;
constexpr amrex::Real burn_dt = 1.0e8_rt;

struct MomentumCheck {
	amrex::Real gas_px{};
	amrex::Real gas_py{};
	amrex::Real gas_pz{};
	amrex::Real expected_px{};
};

struct EnergyCheck {
	amrex::Real gas_energy{};
	amrex::Real gas_internal_energy{};
	amrex::Real gas_kinetic_energy{};
};
} // namespace

template <> struct quokka::EOS_Traits<DTypeFrontVC> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DTypeFrontVC> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = NumSpec;
	static constexpr int numPassiveScalars = numMassScalars;
	static constexpr bool is_radiation_enabled = true;
};

template <> struct RadSystem_Traits<DTypeFrontVC> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = C::a_rad * 1.0e-8;
	static constexpr int beta_order = 1;
	static constexpr auto ChemBands() { return ChemBandsHeader_; }
};

template <> struct SimulationData<DTypeFrontVC> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real primary_species_1{};
	amrex::Real primary_species_2{};
	amrex::Real primary_species_3{};
};

template <>
void RadSystem<DTypeFrontVC>::SetRadEnergySource(array_t &radEnergy, const amrex::Box &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*dx*/,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real /*time*/)
{
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { radEnergy(i, j, k) = 0.0_rt; });
}

template <> void QuokkaSimulation<DTypeFrontVC>::preCalculateInitialConditions()
{
	init_extern_parameters();

	amrex::ParmParse const pp("photoionization_momentum");
	userData_.small_temp = 1.0e-2;
	userData_.small_dens = 1.0e-60;
	userData_.temperature = 1.0e3;
	userData_.primary_species_1 = 1.0e-10;
	userData_.primary_species_2 = 1.0e2;
	userData_.primary_species_3 = 1.0e-10;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("primary_species_1", userData_.primary_species_1);
	pp.query("primary_species_2", userData_.primary_species_2);
	pp.query("primary_species_3", userData_.primary_species_3);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DTypeFrontVC>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DTypeFrontVC>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> void QuokkaSimulation<DTypeFrontVC>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	burn_t state;
	std::array<Real, NumSpec> numdens = {userData_.primary_species_1, userData_.primary_species_2, userData_.primary_species_3};
	state.T = userData_.temperature;

	Real rhotot = 0.0_rt;
	for (int n = 0; n < NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * spmasses[n];
	}
	state.rho = rhotot;
	eos(eos_input_rt, state);
	const auto Egas0 = state.e * rhotot;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<DTypeFrontVC>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<DTypeFrontVC>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = 1.0e-99_rt;
			state_cc(i, j, k, RadSystem<DTypeFrontVC>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFrontVC>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFrontVC>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
		}
		state_cc(i, j, k, RadSystem<DTypeFrontVC>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFrontVC>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<DTypeFrontVC>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFrontVC>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFrontVC>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFrontVC>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<DTypeFrontVC>::scalar0_index + nn) = state.xn[nn] * spmasses[nn];
		}
	});
}

template <> void QuokkaSimulation<DTypeFrontVC>::computeAfterTimestep() {}

namespace
{
void set_uniform_radiation_beam(amrex::MultiFab &state_mf, amrex::Real const initial_rad_energy, amrex::Real const initial_flux_x)
{
	for (amrex::MFIter iter(state_mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = state_mf.array(iter);
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			state(i, j, k, RadSystem<DTypeFrontVC>::radEnergy_index) = initial_rad_energy;
			state(i, j, k, RadSystem<DTypeFrontVC>::x1RadFlux_index) = initial_flux_x;
			state(i, j, k, RadSystem<DTypeFrontVC>::x2RadFlux_index) = 0.0_rt;
			state(i, j, k, RadSystem<DTypeFrontVC>::x3RadFlux_index) = 0.0_rt;
			state(i, j, k, RadSystem<DTypeFrontVC>::x1GasMomentum_index) = 0.0_rt;
			state(i, j, k, RadSystem<DTypeFrontVC>::x2GasMomentum_index) = 0.0_rt;
			state(i, j, k, RadSystem<DTypeFrontVC>::x3GasMomentum_index) = 0.0_rt;
		});
	}
}

auto compute_momentum_check(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::Real const initial_flux_x)
    -> MomentumCheck
{
	amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	const amrex::Real inv_c2 = 1.0_rt / (C::c_light * C::c_light);

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data,
		       [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
			       const amrex::Real gas_px = cell_volume * state[box_no](i, j, k, RadSystem<DTypeFrontVC>::x1GasMomentum_index);
			       const amrex::Real gas_py = cell_volume * state[box_no](i, j, k, RadSystem<DTypeFrontVC>::x2GasMomentum_index);
			       const amrex::Real gas_pz = cell_volume * state[box_no](i, j, k, RadSystem<DTypeFrontVC>::x3GasMomentum_index);
			       const amrex::Real flux_x_after = state[box_no](i, j, k, RadSystem<DTypeFrontVC>::x1RadFlux_index);
			       const amrex::Real expected_px = cell_volume * (initial_flux_x - flux_x_after) * inv_c2;
			       return {gas_px, gas_py, gas_pz, expected_px};
		       });

	auto [gas_px, gas_py, gas_pz, expected_px] = reduce_data.value();
	amrex::ParallelAllReduce::Sum(gas_px, amrex::ParallelContext::CommunicatorSub());
	amrex::ParallelAllReduce::Sum(gas_py, amrex::ParallelContext::CommunicatorSub());
	amrex::ParallelAllReduce::Sum(gas_pz, amrex::ParallelContext::CommunicatorSub());
	amrex::ParallelAllReduce::Sum(expected_px, amrex::ParallelContext::CommunicatorSub());
	return {.gas_px = gas_px, .gas_py = gas_py, .gas_pz = gas_pz, .expected_px = expected_px};
}

auto compute_energy_check(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> EnergyCheck
{
	amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data,
		       [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real> {
			       const amrex::Real rho = state[box_no](i, j, k, RadSystem<DTypeFrontVC>::gasDensity_index);
			       const amrex::Real px = state[box_no](i, j, k, RadSystem<DTypeFrontVC>::x1GasMomentum_index);
			       const amrex::Real py = state[box_no](i, j, k, RadSystem<DTypeFrontVC>::x2GasMomentum_index);
			       const amrex::Real pz = state[box_no](i, j, k, RadSystem<DTypeFrontVC>::x3GasMomentum_index);
			       const amrex::Real kinetic = (px * px + py * py + pz * pz) / (2.0_rt * rho);
			       const amrex::Real gas_energy = state[box_no](i, j, k, RadSystem<DTypeFrontVC>::gasEnergy_index);
			       const amrex::Real gas_internal_energy = state[box_no](i, j, k, RadSystem<DTypeFrontVC>::gasInternalEnergy_index);
			       return {cell_volume * gas_energy, cell_volume * gas_internal_energy, cell_volume * kinetic};
		       });

	auto [gas_energy, gas_internal_energy, gas_kinetic_energy] = reduce_data.value();
	amrex::ParallelAllReduce::Sum(gas_energy, amrex::ParallelContext::CommunicatorSub());
	amrex::ParallelAllReduce::Sum(gas_internal_energy, amrex::ParallelContext::CommunicatorSub());
	amrex::ParallelAllReduce::Sum(gas_kinetic_energy, amrex::ParallelContext::CommunicatorSub());
	return {.gas_energy = gas_energy, .gas_internal_energy = gas_internal_energy, .gas_kinetic_energy = gas_kinetic_energy};
}
} // namespace

auto problem_main() -> int
{
	QuokkaSimulation<DTypeFrontVC> sim;
	sim.setInitialConditions();

	const amrex::Real photon_energy = RadSystem<DTypeFrontVC>::GetChemBandQuanta(0);
	const amrex::Real initial_rad_energy = initial_photon_number_density * photon_energy;
	const amrex::Real initial_flux_x = C::c_light * initial_rad_energy;
	set_uniform_radiation_beam(sim.state_new_cc_[0], initial_rad_energy, initial_flux_x);
	const EnergyCheck energy_before = compute_energy_check(sim.state_new_cc_[0], sim.geom[0].CellSizeArray());

	std::array<amrex::MultiFab const *, AMREX_SPACEDIM> const fc_ptrs{};
	static_cast<void>(quokka::photochemistry::computePhotoChemistry<DTypeFrontVC>(sim.state_new_cc_[0], fc_ptrs, burn_dt, 1,
										      std::numeric_limits<amrex::Real>::max(), 0.0_rt));

	const MomentumCheck check = compute_momentum_check(sim.state_new_cc_[0], sim.geom[0].CellSizeArray(), initial_flux_x);
	const EnergyCheck energy_after = compute_energy_check(sim.state_new_cc_[0], sim.geom[0].CellSizeArray());

	int status = 0;
	const amrex::Real rel_err = std::abs(check.gas_px - check.expected_px) / check.expected_px;
	const amrex::Real transverse_momentum = std::sqrt(check.gas_py * check.gas_py + check.gas_pz * check.gas_pz);
	const amrex::Real transverse_rel = transverse_momentum / check.expected_px;
	const amrex::Real internal_rel = std::abs(energy_after.gas_internal_energy - energy_before.gas_internal_energy) / energy_before.gas_internal_energy;
	const amrex::Real total_energy_delta = energy_after.gas_energy - energy_before.gas_energy;
	const amrex::Real kinetic_abs = std::abs(total_energy_delta - energy_after.gas_kinetic_energy);
	const amrex::Real kinetic_rel = std::abs(total_energy_delta - energy_after.gas_kinetic_energy) / energy_after.gas_kinetic_energy;

	amrex::Print() << "Gas x-momentum: " << check.gas_px << '\n';
	amrex::Print() << "Expected x-momentum from absorbed flux: " << check.expected_px << '\n';
	amrex::Print() << "Relative x-momentum error: " << rel_err << '\n';
	amrex::Print() << "Relative transverse momentum: " << transverse_rel << '\n';
	amrex::Print() << "Relative internal-energy change from O(v/c) work term: " << internal_rel << '\n';
	amrex::Print() << "Total gas-energy change: " << total_energy_delta << '\n';
	amrex::Print() << "Gas kinetic energy after momentum deposition: " << energy_after.gas_kinetic_energy << '\n';
	amrex::Print() << "Absolute total-energy/kinetic-energy error: " << kinetic_abs << '\n';
	amrex::Print() << "Relative total-energy/kinetic-energy error: " << kinetic_rel << '\n';

	if (!(check.expected_px > 0.0_rt)) {
		amrex::Print() << "Test FAILED: no chem-band flux was absorbed.\n";
		status = 1;
	}
	if (rel_err > 1.0e-12_rt) {
		amrex::Print() << "Test FAILED: gas momentum does not match absorbed photon momentum.\n";
		status = 1;
	}
	if (transverse_rel > 1.0e-14_rt) {
		amrex::Print() << "Test FAILED: transverse momentum was deposited by a purely x-directed beam.\n";
		status = 1;
	}
	if (internal_rel > 1.0e-12_rt) {
		amrex::Print() << "Test FAILED: O(v/c) momentum work changed gas internal energy.\n";
		status = 1;
	}
	if (kinetic_abs > 1.0e-12_rt * energy_after.gas_energy) {
		amrex::Print() << "Test FAILED: absorbed momentum did not increase total gas energy by the kinetic-energy gain.\n";
		status = 1;
	}
	if (status == 0) {
		amrex::Print() << "Test passed: photoionization O(v/c) momentum deposition changes kinetic energy only.\n";
	}

	return status;
}
