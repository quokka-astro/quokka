#include "AMReX_FArrayBox.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_Print.H"
#include "extern_parameters.H"
#include "radiation/radiation_system.hpp"
#include <cmath>

// Two instantiations exercise both live source-update entry points on CPU/GPU.
template <int N> struct NestedProblem {};
template <int N> struct Physics_Traits<NestedProblem<N>> : DefaultPhysicsTraits {
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = N;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double c_light = 4.0;
	static constexpr double radiation_constant = 1.0;
	static constexpr double boltzmann_constant = 8.0 * (5.0 / 3.0 - 1.0);
};
template <int N> struct quokka::EOS_Traits<NestedProblem<N>> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5.0 / 3.0;
};
template <int N> struct ISM_Traits<NestedProblem<N>> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr bool enable_photoelectric_heating = false;
	static constexpr bool thermal_band_photochemistry = false;
	static constexpr bool dust_chemical_band_absorption = false;
	static constexpr double gas_dust_coupling_threshold = 0;
};
template <int N> struct RadSystem_Traits<NestedProblem<N>> {
	static constexpr double c_hat_over_c = 0.25;
	static constexpr double Erad_floor = 0;
	static constexpr double energy_unit = 1;
	static constexpr int beta_order = 0;
	static constexpr OpacityModel opacity_model = N == 1 ? OpacityModel::single_group : OpacityModel::piecewise_constant_opacity;
	static constexpr amrex::GpuArray<double, N + 1> radBoundaries = []() constexpr {
		if constexpr (N == 1) {
			return amrex::GpuArray<double, 2>{1, 2};
		} else {
			return amrex::GpuArray<double, 3>{1, 2, 3};
		}
	}();
};
template <int N> struct NestedRadiationCoupling_Traits<NestedProblem<N>> {
	static constexpr bool enabled = true;
	static constexpr bool thermal_only = true;
};

// Analytic positive bands with elasticity four isolate source mapping, c/chat,
// collision scaling, and positive reconstruction from interpolation error.
template <> AMREX_GPU_DEVICE auto RadSystem<NestedProblem<1>>::NestedCouplingOptions() -> mgsolve::Options
{
	mgsolve::Options options;
	options.x_min = 0.125;
	options.x_max = 16;
	options.allow_estimated = true;
	options.relative_tolerance = 1.e-11;
	return options;
}
template <>
AMREX_GPU_DEVICE auto RadSystem<NestedProblem<1>>::NestedCouplingContract(double /*unused*/, amrex::GpuArray<double, 1 + 1> const & /*unused*/)
    -> mgsolve::Contract<1>
{
	mgsolve::Contract<1> contract;
	for (int g = 0; g < 1; ++g) {
		contract.sensitivity[g] = 4;
	}
	return contract;
}
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<NestedProblem<1>>::ComputePlanckOpacity(double /*unused*/, double /*unused*/) -> double { return 1; }
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<NestedProblem<1>>::ComputeEnergyMeanOpacity(double /*unused*/, double /*unused*/) -> double { return 1; }
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<NestedProblem<1>>::ComputeFluxMeanOpacity(double /*unused*/, double /*unused*/) -> double { return 1; }

template <> AMREX_GPU_DEVICE auto RadSystem<NestedProblem<2>>::NestedCouplingOptions() -> mgsolve::Options
{
	mgsolve::Options options;
	options.x_min = 0.125;
	options.x_max = 16;
	options.allow_estimated = true;
	options.relative_tolerance = 1.e-11;
	return options;
}
template <>
AMREX_GPU_DEVICE auto RadSystem<NestedProblem<2>>::NestedCouplingContract(double /*unused*/, amrex::GpuArray<double, 2 + 1> const & /*unused*/)
    -> mgsolve::Contract<2>
{
	mgsolve::Contract<2> contract;
	for (int g = 0; g < 2; ++g) {
		contract.sensitivity[g] = 4;
	}
	return contract;
}
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<NestedProblem<2>>::ComputePlanckOpacity(double /*unused*/, double /*unused*/) -> double { return 1; }
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<NestedProblem<2>>::ComputeEnergyMeanOpacity(double /*unused*/, double /*unused*/) -> double { return 1; }
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<NestedProblem<2>>::ComputeFluxMeanOpacity(double /*unused*/, double /*unused*/) -> double { return 1; }

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<NestedProblem<2>>::ComputeThermalRadiationMultiGroup(double x, amrex::GpuArray<double, 3> const & /*unused*/)
    -> quokka::valarray<double, 2>
{
	return {0.5 * x * x * x * x, 0.5 * x * x * x * x};
}
template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<NestedProblem<2>>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, 3> /*unused*/, double /*unused*/,
											     double /*unused*/)
    -> amrex::GpuArray<amrex::GpuArray<double, 3>, 2>
{
	return {{{0, 0, 0}, {1, 1, 1}}};
}

template <int N> auto test_live(double initial_temperature, double dust_root) -> int
{
	using RS = RadSystem<NestedProblem<N>>;
	amrex::Box const box(amrex::IntVect(0), amrex::IntVect(0));
	amrex::FArrayBox state(box, Physics_Indices<NestedProblem<N>>::nvarTotal_cc);
	amrex::FArrayBox source(box, N);
	amrex::FArrayBox flux(box, 3 * N);
	amrex::FArrayBox heating(box, 1);
	state.setVal<amrex::RunOn::Device>(0);
	source.setVal<amrex::RunOn::Device>(0);
	flux.setVal<amrex::RunOn::Device>(0);
	heating.setVal<amrex::RunOn::Device>(0);
	auto const a = state.array();
	auto const src = source.array();
	constexpr double gas_root = 4.0;
	// A=8, D=1, h=1, chi=4. These states solve the frozen equations.
	double const B = dust_root * dust_root * dust_root * dust_root;
	double const radiation_input = B + 4.0 * (gas_root - initial_temperature);
	double const radiation_expected = 0.5 * (radiation_input + B) / N;
	double const initial_energy = quokka::EOS<NestedProblem<N>>::ComputeEintFromTgas(1.0, initial_temperature);
	amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		a(i, j, k, RS::gasDensity_index) = 1;
		a(i, j, k, RS::gasEnergy_index) = initial_energy;
		a(i, j, k, RS::gasInternalEnergy_index) = initial_energy;
		for (int g = 0; g < N; ++g) {
			a(i, j, k, RS::radEnergy_index + RS::numRadVars_ * g) = radiation_input / N - 0.125;
			src(i, j, k, g) = 0.5; // dt/chat correction gives 0.125 injected energy
		}
	});
	amrex::Gpu::DeviceVector<int> counts(3, 0);
	amrex::Gpu::DeviceVector<int> failures(3, 0);
	auto const energy_source = source.const_array();
	auto const flux_source = flux.const_array();
	amrex::GpuArray<double, 0> const scalars{};
	double const nH = RS::ComputeNumberDensityH(1.0, scalars);
	double const dust_coefficient = 1.0 / (nH * nH);
	if constexpr (N == 1) {
		RS::AddSourceTermsSingleGroup(a, energy_source, flux_source, box, 1, 1, dust_coefficient, 1.e-10, 0, 0, counts.data(), failures.data(), {});
	} else {
		RS::AddSourceTermsMultiGroup(a, energy_source, flux_source, box, 1, 1, dust_coefficient, 1.e-10, 0, 0, counts.data(), failures.data(),
					     heating.const_array(), {});
	}
	amrex::Gpu::DeviceVector<double> results(N + 1);
	auto *out = results.data();
	amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		out[0] = a(i, j, k, RS::gasInternalEnergy_index);
		for (int g = 0; g < N; ++g) {
			out[g + 1] = a(i, j, k, RS::radEnergy_index + RS::numRadVars_ * g);
		}
	});
	amrex::Gpu::HostVector<double> host(N + 1);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, results.begin(), results.end(), host.begin());
	double worst = std::abs(host[0] / 32.0 - 1);
	for (int g = 0; g < N; ++g) {
		worst = std::max(worst, std::abs(host[g + 1] / radiation_expected - 1));
	}
	amrex::Print() << "Nested live groups=" << N << " initial T=" << initial_temperature << " relative error=" << worst << "\n";
	return worst < 1.e-10 ? 0 : 1;
}

auto problem_main() -> int
{
	init_extern_parameters();
	double small_temp = 1.e-10;
	double small_density = 1.e-10;
	eos_init(small_temp, small_density);
	int status = 0;
	status += test_live<1>(3.0, 8.0);
	status += test_live<2>(3.0, 8.0);
	status += test_live<1>(4.5, 2.0);
	status += test_live<2>(4.5, 2.0);
	status += test_live<1>(4.0, 4.0);
	status += test_live<2>(4.0, 4.0);
	return status == 0 ? 0 : 1;
}
