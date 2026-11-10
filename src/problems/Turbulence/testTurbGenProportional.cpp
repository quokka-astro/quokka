#include <TurbGen.h>

#include <array>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using Parameters = std::map<std::string, std::string>;
using Components = std::array<double, 3>;

void Check(const bool condition, const std::string &description)
{
	if (!condition) {
		throw std::runtime_error(description);
	}
}

void CheckNear(const double actual, const double expected, const std::string &description)
{
	Check(std::isfinite(actual) && std::abs(actual - expected) <= 1.0e-12 * std::max(1.0, std::abs(expected)), description);
}

// These are the complete TurbGen map inputs forwarded by QuokkaSimulation.
// Power-law parameters are supplied explicitly, and dt = 1/32 is exact.
auto DrivingParameters() -> Parameters
{
	return {{"ndim", "3"},
		{"length", "1"},
		{"target_vdisp", "2"},
		{"ampl_factor", "1,2,3"},
		{"ampl_auto_adjust", "1"},
		{"ampl_auto_adjust_method", "proportional"},
		{"ampl_proportional_gain", "60"},
		{"ampl_max_amplitude", "20"},
		{"k_driv", "2"},
		{"k_min", "1"},
		{"k_max", "2"},
		{"sol_weight", "0.5"},
		{"spect_form", "2"},
		{"power_law_exp", "-2"},
		{"angles_exp", "1"},
		{"random_seed", "140281"},
		{"nsteps_per_t_turb", "8"}};
}

struct GeneratorState {
	Components amplitudes;
	std::vector<double> phases;
	std::array<std::vector<double>, 3> real;
	std::array<std::vector<double>, 3> imaginary;
	int randomSeed;
	int ouStep;

	auto operator==(const GeneratorState &) const -> bool = default;
};

class TurbGenProbe final : public TurbGen
{
      public:
	// A non-writing rank keeps this standalone test free of evolution files.
	TurbGenProbe() : TurbGen(1) { set_verbose(0); }

	[[nodiscard]] auto State() const -> GeneratorState
	{
		return {.amplitudes = {ampl_factor[0], ampl_factor[1], ampl_factor[2]},
			.phases = OUphases,
			.real = {aka[0], aka[1], aka[2]},
			.imaginary = {akb[0], akb[1], akb[2]},
			.randomSeed = seed,
			.ouStep = step};
	}

	[[nodiscard]] auto UpdateInterval() const -> double { return dt; }
};

auto ComponentRms(const Components &values) -> double { return std::hypot(values[0], values[1], values[2]) / std::numbers::sqrt3; }

void CheckSameOu(const GeneratorState &controlled, const GeneratorState &fixed)
{
	Check(!controlled.phases.empty(), "the test must exercise actual OU modes");
	Check(controlled.phases == fixed.phases && controlled.real == fixed.real && controlled.imaginary == fixed.imaginary &&
		  controlled.randomSeed == fixed.randomSeed && controlled.ouStep == fixed.ouStep,
	      "amplitude control must leave the underlying OU sequence unchanged");
}

void TestProportionalDriving()
{
	TurbGenProbe controlled;
	TurbGenProbe fixed;
	auto params = DrivingParameters();
	Check(controlled.init_driving(params) == 0, "initialize proportional driver");
	Check(controlled.uses_proportional_control(), "select proportional method");
	params["ampl_auto_adjust"] = "0";
	Check(fixed.init_driving(params) == 0, "initialize fixed-amplitude driver");
	Check(!fixed.uses_proportional_control(), "honor disabled amplitude control");
	const auto initial = controlled.State();
	Check(initial == fixed.State(), "identical seeded initial conditions");
	const double reference = ComponentRms(initial.amplitudes);
	CheckNear(reference, std::sqrt(12.0), "reference uses powered amplitudes");

	const std::array<Components, 10> observations = {{{0.0, 0.0, 0.0},
							  {0.0, 1.2, 1.6},
							  {1.2, 0.0, 1.6},
							  {1.2, 1.6, 0.0},
							  {4.0, 0.0, 0.0},
							  {4.0, 0.0, 0.0},
							  {1.98, 0.0, 0.0},
							  {1.98, 0.0, 0.0},
							  {2.0, 0.0, 0.0},
							  {0.0, 0.0, 0.0}}};
	const std::array<int, 10> requestedSteps = {0, 1, 2, 3, 6, 7, 10, 11, 12, 15};
	std::feclearexcept(FE_ALL_EXCEPT);
	for (std::size_t i = 0; i < observations.size(); ++i) {
		const double time = (requestedSteps[i] + 0.25) * controlled.UpdateInterval();
		const auto previous = controlled.State();
		Check(controlled.check_for_update(time, observations[i].data()), "scheduled proportional update");
		Check(fixed.check_for_update(time, observations[i].data()), "scheduled fixed-amplitude update");
		const auto current = controlled.State();
		CheckSameOu(current, fixed.State());
		Check(current.ouStep == requestedSteps[i], "advance to the requested OU step");
		Check(current.phases != previous.phases, "OU noise keeps evolving even while the amplitude is zero");
		Check(fixed.State().amplitudes == initial.amplitudes, "disabled control keeps the original amplitudes");

		const double normalized = std::hypot(observations[i][0], observations[i][1], observations[i][2]) / 2.0;
		const double expected = std::clamp(reference + 60.0 * (1.0 - normalized), 0.0, 20.0);
		const double amplitude = ComponentRms(current.amplitudes);
		CheckNear(amplitude, expected, "bounded positional proportional amplitude");
		Check(amplitude >= 0.0 && amplitude <= 20.0 * (1.0 + 1.0e-14), "the common RMS amplitude respects its absolute cap");
		for (std::size_t d = 0; d < 3; ++d) {
			CheckNear(current.amplitudes[d], initial.amplitudes[d] * expected / reference, "one common scalar preserves the original anisotropy");
		}

		const Components position = {0.13, 0.37, 0.71};
		Components field{};
		Components fixedField{};
		controlled.get_turb_vector(position.data(), field.data());
		fixed.get_turb_vector(position.data(), fixedField.data());
		for (std::size_t d = 0; d < 3; ++d) {
			CheckNear(field[d], fixedField[d] * expected / reference, "the physical forcing has the same common amplitude scale");
			if (expected == 0.0) {
				Check(field[d] == 0.0, "zero clipped amplitude produces zero forcing");
			}
		}

		const Components changedObservation = {100.0, 100.0, 100.0};
		Check(!controlled.check_for_update(time, changedObservation.data()), "a repeated time must not perform another controller update");
		Check(!controlled.check_for_update(time + 0.25 * controlled.UpdateInterval(), changedObservation.data()),
		      "observations between OU steps must not alter held forcing");
		Check(controlled.State() == current, "unscheduled calls preserve amplitudes, phases, coefficients, and "
						     "seed");
	}
	Check(std::fetestexcept(FE_OVERFLOW | FE_INVALID | FE_DIVBYZERO) == 0, "zero and one-zero-component observations remain numerically safe");
}

void TestLegacyDefault()
{
	auto params = DrivingParameters();
	params.erase("ampl_auto_adjust_method");
	params.erase("ampl_proportional_gain");
	params.erase("ampl_max_amplitude");
	TurbGenProbe implicitLegacy;
	TurbGenProbe explicitLegacy;
	Check(implicitLegacy.init_driving(params) == 0, "initialize default method");
	params["ampl_auto_adjust_method"] = "legacy";
	Check(explicitLegacy.init_driving(params) == 0, "initialize explicit legacy method");
	Check(!implicitLegacy.uses_proportional_control(), "legacy remains the default");
	const Components observation = {0.4, 0.5, 0.6};
	for (const int step : {0, 2, 5}) {
		const double time = (step + 0.25) * implicitLegacy.UpdateInterval();
		Check(implicitLegacy.check_for_update(time, observation.data()), "default update");
		Check(explicitLegacy.check_for_update(time, observation.data()), "legacy update");
		Check(implicitLegacy.State() == explicitLegacy.State(), "omitted controller options preserve legacy behavior");
	}
}

void CheckRejected(const Parameters &params, const double time, const std::string &description)
{
	bool rejected = false;
	try {
		TurbGenProbe generator;
		static_cast<void>(generator.init_driving(params, time));
	} catch (const std::exception &) {
		rejected = true;
	}
	Check(rejected, description);
}

void TestRejectedMeasurements()
{
	for (int axis = 0; axis < 3; ++axis) {
		for (const double invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -2.0, -1.0}) {
			TurbGenProbe generator;
			generator.init_driving(DrivingParameters());
			const auto before = generator.State();
			Components observation = {0.0, 0.0, 0.0};
			observation[axis] = invalid;
			bool rejected = false;
			try {
				generator.check_for_update(0.0, observation.data());
			} catch (const std::invalid_argument &) {
				rejected = true;
			}
			Check(rejected, "reject invalid component measurement");
			Check(generator.State() == before, "invalid measurement cannot advance OU state");
		}
	}
	TurbGenProbe generator;
	generator.init_driving(DrivingParameters());
	const auto before = generator.State();
	Check(generator.check_for_update(0.0), "no-measurement overload advances OU");
	Check(generator.State().amplitudes == before.amplitudes, "all-negative sentinel skips amplitude feedback");
}

void TestRejectedInitialization()
{
	const std::vector<std::pair<std::string, std::string>> invalid = {{"ampl_auto_adjust_method", "unknown"},
									  {"ampl_proportional_gain", "0"},
									  {"ampl_proportional_gain", "-1"},
									  {"ampl_proportional_gain", "nan"},
									  {"ampl_proportional_gain", "inf"},
									  {"ampl_max_amplitude", "0"},
									  {"ampl_max_amplitude", "-1"},
									  {"ampl_max_amplitude", "nan"},
									  {"ampl_max_amplitude", "inf"},
									  {"ampl_max_amplitude", "1.7e308"},
									  {"target_vdisp", "0"},
									  {"target_vdisp", "-1"},
									  {"target_vdisp", "nan"},
									  {"target_vdisp", "inf"},
									  {"ampl_factor", "0,2,3"},
									  {"ampl_factor", "1,-2,3"},
									  {"ampl_factor", "1,2,0"},
									  {"ampl_factor", "1,2,1e300"},
									  {"ampl_factor", "nan"},
									  {"ampl_factor", "inf"}};
	for (const auto &[key, value] : invalid) {
		auto params = DrivingParameters();
		params[key] = value;
		std::string description = "reject invalid ";
		description.append(key).append("=").append(value);
		CheckRejected(params, 0.0, description);
	}
	CheckRejected(DrivingParameters(), 0.25, "reject proportional restart without checkpointed controller state");
}

} // namespace

auto main() -> int
{
	try {
		TestProportionalDriving();
		TestLegacyDefault();
		TestRejectedInitialization();
		TestRejectedMeasurements();
	} catch (const std::exception &error) {
		std::cerr << "TurbGen proportional integration test failed: " << error.what() << '\n';
		return 1;
	}
	std::cout << "TurbGen proportional integration tests passed.\n";
	return 0;
}
