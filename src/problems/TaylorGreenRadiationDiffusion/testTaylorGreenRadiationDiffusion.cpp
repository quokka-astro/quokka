/// \file testTaylorGreenRadiationDiffusion.cpp
/// \brief Taylor-Green radiation diffusion MMS with finite-P1 flux relaxation.

#include "AMReX_MultiFab.H"
#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include <cmath>

struct TaylorGreenRadiationDiffusion {
};

namespace
{
constexpr double gamma_gas = 5.0 / 3.0;
constexpr double rho0 = 1.0;
constexpr double pressure0 = 0.1;
constexpr double potential_amplitude = 5.0e-3;
constexpr double Erad0 = 1.0;
constexpr double c_light = 100.0;
constexpr double c_hat = 1.0;
constexpr double velocity_amplitude = 1.0e-2;
constexpr double opacity = 100.0;
constexpr double wavenumber = 2.0 * M_PI;
constexpr double tau_flux = 1.0 / (c_hat * rho0 * opacity);
constexpr double final_time = 5.0 * tau_flux;
constexpr double constant_dt = 1.0e-4;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto mode(amrex::Real x, amrex::Real y) -> amrex::Real
{
	return std::cos(2.0 * wavenumber * x) + std::cos(2.0 * wavenumber * y);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto potential(amrex::Real x, amrex::Real y) -> amrex::Real { return potential_amplitude * mode(x, y); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto relaxedPotentialAmplitude(amrex::Real time) -> amrex::Real
{
	return potential_amplitude * (1.0 - std::exp(-time / tau_flux));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto taylorGreenPressurePotential(amrex::Real x, amrex::Real y) -> amrex::Real
{
	return 0.25 * rho0 * velocity_amplitude * velocity_amplitude * mode(x, y);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto xVelocity(amrex::Real x, amrex::Real y) -> amrex::Real
{
	return velocity_amplitude * std::sin(wavenumber * x) * std::cos(wavenumber * y);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto yVelocity(amrex::Real x, amrex::Real y) -> amrex::Real
{
	return -velocity_amplitude * std::cos(wavenumber * x) * std::sin(wavenumber * y);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto exactRadiationEnergy(amrex::Real x, amrex::Real y) -> amrex::Real { return Erad0 - 3.0 * potential(x, y); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto exactFluxX(amrex::Real x, amrex::Real /*y*/, amrex::Real time) -> amrex::Real
{
	const amrex::Real amp = relaxedPotentialAmplitude(time);
	return -(2.0 * c_light * wavenumber * amp / (rho0 * opacity)) * std::sin(2.0 * wavenumber * x);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto exactFluxY(amrex::Real /*x*/, amrex::Real y, amrex::Real time) -> amrex::Real
{
	const amrex::Real amp = relaxedPotentialAmplitude(time);
	return -(2.0 * c_light * wavenumber * amp / (rho0 * opacity)) * std::sin(2.0 * wavenumber * y);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto radiationEnergySource(amrex::Real x, amrex::Real y, amrex::Real time) -> amrex::Real
{
	const amrex::Real advection_source = 6.0 * wavenumber * potential_amplitude * velocity_amplitude *
					     (std::sin(wavenumber * x) * std::cos(wavenumber * y) * std::sin(2.0 * wavenumber * x) -
					      std::cos(wavenumber * x) * std::sin(wavenumber * y) * std::sin(2.0 * wavenumber * y));

	const amrex::Real diffusion_source = -(4.0 * c_light * wavenumber * wavenumber * relaxedPotentialAmplitude(time) / (rho0 * opacity)) * mode(x, y);
	return advection_source + diffusion_source;
}
} // namespace

template <> struct quokka::EOS_Traits<TaylorGreenRadiationDiffusion> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = gamma_gas;
};

template <> struct Physics_Traits<TaylorGreenRadiationDiffusion> {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = true;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = ::c_light;
	static constexpr double radiation_constant = 1.0;
};

template <> struct RadSystem_Traits<TaylorGreenRadiationDiffusion> {
	static constexpr double c_hat_over_c = c_hat / c_light;
	static constexpr double Erad_floor = 0.0;
	static constexpr int beta_order = 0;
	static constexpr bool allow_signed_radiation_energy_source = true;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<TaylorGreenRadiationDiffusion>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/)
    -> amrex::Real
{
	return 0.0;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<TaylorGreenRadiationDiffusion>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/)
    -> amrex::Real
{
	return opacity;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<TaylorGreenRadiationDiffusion>::ComputeEnergyMeanOpacity(const double /*rho*/, const double /*Tgas*/)
    -> amrex::Real
{
	return 0.0;
}

template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<TaylorGreenRadiationDiffusion>::ComputeEddingtonFactor(double /*f*/) -> double
{
	return 1.0 / 3.0;
}

template <>
void RadSystem<TaylorGreenRadiationDiffusion>::SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real time)
{
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
		const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
		radEnergySource(i, j, k, 0) += radiationEnergySource(x, y, time);
	});
}

template <> void QuokkaSimulation<TaylorGreenRadiationDiffusion>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	amrex::Box const &indexRange = grid_elem.indexRange_;
	amrex::Array4<double> const &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
		const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];

		const amrex::Real rho = rho0;
		const amrex::Real vx = xVelocity(x, y);
		const amrex::Real vy = yVelocity(x, y);
		const amrex::Real pressure = pressure0 + taylorGreenPressurePotential(x, y);
		const amrex::Real Eint = pressure / (gamma_gas - 1.0);

		state_cc(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::radEnergy_index) = exactRadiationEnergy(x, y);
		state_cc(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::x1RadFlux_index) = 0.0;
		state_cc(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::x2RadFlux_index) = 0.0;
		state_cc(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::x3RadFlux_index) = 0.0;

		state_cc(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::x1GasMomentum_index) = rho * vx;
		state_cc(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::x2GasMomentum_index) = rho * vy;
		state_cc(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::x3GasMomentum_index) = 0.0;
		state_cc(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::gasInternalEnergy_index) = Eint;
		state_cc(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::gasEnergy_index) = Eint + 0.5 * rho * (vx * vx + vy * vy);
	});
}

auto problem_main() -> int
{
	QuokkaSimulation<TaylorGreenRadiationDiffusion> sim;

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3;
	sim.cflNumber_ = 0.8;
	sim.radiationCflNumber_ = 0.8;
	sim.constantDt_ = constant_dt;
	sim.maxDt_ = constant_dt;
	sim.stopTime_ = final_time;
	sim.maxTimesteps_ = 100000;
	sim.plotfileInterval_ = -1;

	sim.setInitialConditions();
	sim.evolve();

	amrex::MultiFab error_mf(sim.state_new_cc_[0].boxArray(), sim.state_new_cc_[0].DistributionMap(), 3, 0);
	amrex::MultiFab exact_mf(sim.state_new_cc_[0].boxArray(), sim.state_new_cc_[0].DistributionMap(), 3, 0);

	const auto dx = sim.Geom(0).CellSizeArray();
	const auto prob_lo = sim.Geom(0).ProbLoArray();
	const amrex::Real time = sim.tNew_[0];

	for (amrex::MFIter iter(sim.state_new_cc_[0]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = sim.state_new_cc_[0].const_array(iter);
		auto const &err = error_mf.array(iter);
		auto const &exact = exact_mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
			const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];

			const amrex::Real Erad_exact = exactRadiationEnergy(x, y);
			const amrex::Real Fx_exact = exactFluxX(x, y, time);
			const amrex::Real Fy_exact = exactFluxY(x, y, time);

			exact(i, j, k, 0) = Erad_exact;
			exact(i, j, k, 1) = Fx_exact;
			exact(i, j, k, 2) = Fy_exact;

			err(i, j, k, 0) = state(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::radEnergy_index) - Erad_exact;
			err(i, j, k, 1) = state(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::x1RadFlux_index) - Fx_exact;
			err(i, j, k, 2) = state(i, j, k, RadSystem<TaylorGreenRadiationDiffusion>::x2RadFlux_index) - Fy_exact;
		});
	}

	const amrex::Real n_cells = static_cast<amrex::Real>(error_mf.boxArray().numPts());
	const amrex::Real energy_rel_l1 = (error_mf.norm1(0) / n_cells) / (exact_mf.norm1(0) / n_cells);
	const amrex::Real flux_abs_l1 = (error_mf.norm1(1) + error_mf.norm1(2)) / n_cells;
	const amrex::Real flux_ref_l1 = (exact_mf.norm1(1) + exact_mf.norm1(2)) / n_cells;
	const amrex::Real flux_rel_l1 = flux_abs_l1 / flux_ref_l1;

	amrex::Print() << "Taylor-Green radiation diffusion MMS:\n";
	amrex::Print() << "  t / tau_F = " << time / tau_flux << "\n";
	amrex::Print() << "  radiation energy relative L1 error = " << energy_rel_l1 << "\n";
	amrex::Print() << "  radiation flux relative L1 error = " << flux_rel_l1 << "\n";

	const bool passed = std::isfinite(energy_rel_l1) && std::isfinite(flux_rel_l1) && (energy_rel_l1 < 5.0e-3) && (flux_rel_l1 < 7.5e-2);
	if (!passed) {
		amrex::Print() << "Test failed.\n";
		return 1;
	}

	amrex::Print() << "Test passed.\n";
	return 0;
}
