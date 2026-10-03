// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef QUOKKA_NESTED_COUPLING_HPP_
#define QUOKKA_NESTED_COUPLING_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

// The default hooks fail closed. A problem must declare domain-wide opacity
// assumptions and explicitly allow estimated results if its evaluator is unverified.
template <typename problem_t> AMREX_GPU_DEVICE auto RadSystem<problem_t>::NestedCouplingOptions() -> mgsolve::Options { return {}; }

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::NestedCouplingContract(double /*rho*/, amrex::GpuArray<double, nGroups_ + 1> const & /*boundaries*/)
    -> mgsolve::Contract<nGroups_>
{
	return {};
}

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::NestedCouplingValues(double rho, double temperature, quokka::valarray<double, nGroups_> const &radiation,
								 amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> mgsolve::GroupValues<nGroups_>
{
	mgsolve::GroupValues<nGroups_> values;
	if constexpr (nGroups_ == 1) {
		values.alpha[0] = rho * ComputeEnergyMeanOpacity(rho, temperature);
		values.p[0] = rho * ComputePlanckOpacity(rho, temperature);
		values.B[0] = ComputeThermalRadiationSingleGroup(temperature);
	} else {
		amrex::GpuArray<double, nGroups_> ratios{};
		for (int g = 0; g < nGroups_; ++g) {
			ratios[g] = boundaries[g + 1] / boundaries[g];
		}
		auto const bands = ComputeThermalRadiationMultiGroup(temperature, boundaries);
		// The full-spectrum fit must not change its reference radiation during
		// iteration. That model is excluded by the adapter's static assertion.
		auto const opacity = ComputeModelDependentKappaEAndKappaP(temperature, rho, boundaries, ratios, bands, radiation, 0);
		for (int g = 0; g < nGroups_; ++g) {
			values.alpha[g] = rho * opacity.kappaE[g];
			values.p[g] = rho * opacity.kappaP[g];
			values.B[g] = bands[g];
		}
	}
	// Zero masks deliberately remain false: a sampled zero does not establish
	// an identically-zero coefficient or band. Specialize this hook to declare one.
	return values;
}

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveNestedRadiationCoupling(double gas_energy, quokka::valarray<double, nGroups_> const &radiation, double rho,
									 double coeff_n, double dt, amrex::GpuArray<Real, nmscalars_> const &massScalars,
									 quokka::valarray<double, nGroups_> const &source,
									 amrex::GpuArray<double, nGroups_ + 1> const &boundaries, double temperature_floor,
									 int *iterations) -> NewtonIterationResult<problem_t>
{
	static_assert(NestedRadiationCoupling_Traits<problem_t>::thermal_only,
		      "Nested coupling requires thermal_only: no extra cooling, heating, or chemistry.");
	static_assert(std::is_same_v<typename quokka::detail::EOSBackendHelper<problem_t>::type, quokka::EOSIdeal<problem_t>> && gamma_ > 1.0,
		      "Nested coupling requires the constant heat capacity ideal gas EOS.");
	static_assert(enable_dust_gas_thermal_coupling_model_, "Nested coupling currently requires dust with positive collision coefficient.");
	static_assert(beta_order_ == 0, "The live nested path currently excludes velocity work and Lorentz corrections.");
	static_assert(!enable_photoelectric_heating_ && !thermal_band_photochemistry_ && nGroupsThermal_ == nGroups_,
		      "Nested coupling currently supports thermal groups only.");
	static_assert(opacity_model_ != OpacityModel::PPL_opacity_full_spectrum,
		      "The evolving spectral-fit opacity model is outside this adapter's equations.");
	mgsolve::Problem<nGroups_> problem;
	problem.T = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, gas_energy, massScalars);
	problem.A = gas_energy / problem.T;
	problem.chi = c_light_ / c_hat_;
	problem.D = coeff_n * problem.chi; // caller stores dt*K divided by c/chat
	problem.h = dt * c_hat_;
	quokka::valarray<double, nGroups_> adjusted{};
	for (int g = 0; g < nGroups_; ++g) {
		problem.r[g] = radiation[g] + source[g];
		adjusted[g] = problem.r[g];
	}
	auto const options = NestedCouplingOptions();
	auto const contract = NestedCouplingContract(rho, boundaries);
	auto oracle = [=] AMREX_GPU_DEVICE(double x, mgsolve::GroupValues<nGroups_> &values) {
		values = NestedCouplingValues(rho, x, adjusted, boundaries);
		return true;
	};
	auto const solved = mgsolve::solve(problem, oracle, contract, options);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
	    mgsolve::accepted(solved.status),
	    "Nested radiation coupling failed: check domain, callback contracts, positive coupling, arithmetic range, and requested tolerance.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(solved.gas_temperature >= temperature_floor,
					 "Nested coupling reached a gas floor; floor projection is outside the accuracy contract.");
	NewtonIterationResult<problem_t> result{};
	result.Egas = solved.gas_energy;
	result.T_gas = solved.gas_temperature;
	result.T_d = solved.dust_temperature;
	for (int g = 0; g < nGroups_; ++g) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(solved.radiation[g] >= Erad_floor_,
						 "Nested coupling reached a radiation floor; projection is outside the accuracy contract.");
		result.EradVec[g] = solved.radiation[g];
	}
	if constexpr (nGroups_ == 1) {
		result.opacity_terms.kappaE[0] = ComputeEnergyMeanOpacity(rho, result.T_d);
		result.opacity_terms.kappaP[0] = ComputePlanckOpacity(rho, result.T_d);
		result.opacity_terms.kappaF[0] = ComputeFluxMeanOpacity(rho, result.T_d);
	} else {
		amrex::GpuArray<double, nGroups_> ratios{};
		for (int g = 0; g < nGroups_; ++g) {
			ratios[g] = boundaries[g + 1] / boundaries[g];
		}
		auto const bands = ComputeThermalRadiationMultiGroup(result.T_d, boundaries);
		result.opacity_terms = ComputeModelDependentKappaEAndKappaP(result.T_d, rho, boundaries, ratios, bands, adjusted, 0);
		ComputeModelDependentKappaFAndDeltaTerms(result.T_d, rho, boundaries, bands, result.opacity_terms);
	}
	// These existing counters measure outer thermal iterations, not opacity or inner calls.
	amrex::Gpu::Atomic::Add(&iterations[0], 1);
	amrex::Gpu::Atomic::Add(&iterations[1], static_cast<int>(solved.outer_iterations));
	amrex::Gpu::Atomic::Max(&iterations[2], static_cast<int>(solved.outer_iterations));
	return result;
}
#endif
