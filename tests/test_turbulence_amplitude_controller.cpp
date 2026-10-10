#include "../extern/turbulence/AmplitudeController.h"

#include <array>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using NameSpaceTurbGen::ProportionalAmplitude;

void Check(const bool condition, const std::string &description) {
  if (!condition) {
    throw std::runtime_error(description);
  }
}

void CheckNear(const double actual, const double expected,
               const std::string &description) {
  Check(std::isfinite(actual) &&
            std::abs(actual - expected) <=
                1.0e-13 * std::max(1.0, std::abs(expected)),
        description);
}

void CheckRejected(const double dispersion, const double reference,
                   const double kp, const double cap,
                   const std::string &description) {
  bool rejected = false;
  try {
    static_cast<void>(ProportionalAmplitude(dispersion, reference, kp, cap));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  Check(rejected, description);
}

void TestExpectedAmplitudes() {
  CheckNear(ProportionalAmplitude(0.0, 2.0, 60.0, 20.0), 20.0,
            "zero dispersion");
  CheckNear(ProportionalAmplitude(-0.0, 2.0, 60.0, 20.0), 20.0,
            "negative zero dispersion");
  CheckNear(ProportionalAmplitude(1.0e-300, 2.0, 60.0, 20.0), 20.0,
            "tiny dispersion");
  CheckNear(ProportionalAmplitude(0.8, 2.0, 60.0, 20.0), 14.0,
            "below target without saturation");
  CheckNear(ProportionalAmplitude(1.0, 2.0, 60.0, 20.0), 2.0,
            "at target returns reference");
  CheckNear(ProportionalAmplitude(1.0 + 1.0 / 60.0, 2.0, 60.0, 20.0), 1.0,
            "negative proportional correction");
  CheckNear(ProportionalAmplitude(1.0 + 2.0 / 60.0, 2.0, 60.0, 20.0), 0.0,
            "lower saturation boundary");
  CheckNear(ProportionalAmplitude(2.0, 2.0, 60.0, 20.0), 0.0,
            "above target clamps at zero");
  CheckNear(ProportionalAmplitude(0.0, 2.0, 60.0, 3.0), 3.0,
            "custom upper cap");
  CheckNear(ProportionalAmplitude(0.0, 2.0, 0.5, 20.0), 2.5,
            "custom proportional coefficient");
  CheckNear(ProportionalAmplitude(0.0, 0.0, 60.0, 20.0), 20.0,
            "zero reference still drives below target");
  CheckNear(ProportionalAmplitude(1.0, 0.0, 60.0, 20.0), 0.0,
            "zero reference at target");
  CheckNear(ProportionalAmplitude(2.0, -0.0, 60.0, 20.0), 0.0,
            "negative zero reference above target");
  CheckNear(ProportionalAmplitude(0.9, 1.0e-300, 60.0, 20.0), 6.0,
            "tiny reference does not weaken the proportional correction");
  CheckNear(ProportionalAmplitude(1.0, 100.0, 60.0, 20.0), 20.0,
            "reference above cap is clamped at target");
  CheckNear(ProportionalAmplitude(2.0, 100.0, 60.0, 20.0), 20.0,
            "reference above cap can remain saturated above target");
  CheckNear(ProportionalAmplitude(2.0, 100.0, 90.0, 20.0), 10.0,
            "correction is applied before clamping reference");
  CheckNear(ProportionalAmplitude(2.0, 100.0, 2.0, 200.0), 98.0,
            "large excess and coefficient do not prematurely saturate");
  CheckNear(ProportionalAmplitude(1.125, 0.5, 2.0, 20.0), 0.25,
            "subunit reference amplitude");
}

void TestInvalidInputs() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  for (const double invalid : {-1.0, -infinity, infinity, nan}) {
    CheckRejected(invalid, 2.0, 60.0, 20.0, "reject invalid dispersion");
    CheckRejected(1.0, invalid, 60.0, 20.0, "reject invalid reference");
    CheckRejected(1.0, 2.0, invalid, 20.0, "reject invalid coefficient");
    CheckRejected(1.0, 2.0, 60.0, invalid, "reject invalid cap");
  }
  for (const double zero : {0.0, -0.0}) {
    CheckRejected(1.0, 2.0, zero, 20.0, "reject zero coefficient");
    CheckRejected(1.0, 2.0, 60.0, zero, "reject zero cap");
  }
  const double negativeTiny = -std::numeric_limits<double>::denorm_min();
  CheckRejected(negativeTiny, 2.0, 60.0, 20.0,
                "reject tiny negative dispersion");
  CheckRejected(1.0, negativeTiny, 60.0, 20.0,
                "reject tiny negative reference");
  CheckRejected(1.0, 2.0, negativeTiny, 20.0,
                "reject tiny negative coefficient");
  CheckRejected(1.0, 2.0, 60.0, negativeTiny, "reject tiny negative cap");
}

void TestMonotonicAndStateless() {
  const std::array<std::array<double, 3>, 7> configurations = {
      {{2.0, 60.0, 20.0},
       {0.0, 60.0, 20.0},
       {1.0e-300, 60.0, 20.0},
       {100.0, 90.0, 20.0},
       {1.0, 0.1, 0.2},
       {0.5, 0.5, 100.0},
       {0.25, 1000.0, 7.0}}};
  for (const auto &configuration : configurations) {
    const double reference = configuration[0];
    const double kp = configuration[1];
    const double cap = configuration[2];
    double previous = ProportionalAmplitude(0.0, reference, kp, cap);
    for (int i = 0; i <= 10000; ++i) {
      const double dispersion = static_cast<double>(i) / 2500.0;
      const double amplitude =
          ProportionalAmplitude(dispersion, reference, kp, cap);
      Check(amplitude <= previous,
            "amplitude is nonincreasing with dispersion");
      const double expected =
          std::max(0.0, std::min(cap, reference + kp * (1.0 - dispersion)));
      CheckNear(amplitude, expected, "matches bounded proportional law");
      static_cast<void>(ProportionalAmplitude(100.0, 50.0, 2.0, 20.0));
      static_cast<void>(ProportionalAmplitude(0.0, 0.0, 30.0, 0.5));
      Check(ProportionalAmplitude(dispersion, reference, kp, cap) == amplitude,
            "unrelated calls do not change the result");
      previous = amplitude;
    }
  }
}

void TestExtremeFiniteInputs() {
  const double largest = std::numeric_limits<double>::max();
  const double smallest = std::numeric_limits<double>::denorm_min();
  const std::array<double, 9> parameters = {
      smallest,
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::epsilon(),
      0.5,
      1.0,
      2.0,
      60.0,
      1.0e200,
      largest};
  const std::array<double, 10> references = {
      0.0,
      smallest,
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::epsilon(),
      0.5,
      1.0,
      2.0,
      60.0,
      1.0e200,
      largest};
  const std::array<double, 12> dispersions = {
      0.0,
      smallest,
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::epsilon(),
      0.5,
      std::nextafter(1.0, 0.0),
      1.0,
      std::nextafter(1.0, 2.0),
      1.5,
      2.0,
      1.0e200,
      largest};

  std::feclearexcept(FE_ALL_EXCEPT);
  for (const double reference : references) {
    for (const double kp : parameters) {
      for (const double cap : parameters) {
        double previous = cap;
        for (const double dispersion : dispersions) {
          const double amplitude =
              ProportionalAmplitude(dispersion, reference, kp, cap);
          Check(std::isfinite(amplitude) && amplitude >= 0.0 &&
                    amplitude <= cap,
                "extreme finite inputs produce a finite bounded amplitude");
          Check(amplitude <= previous, "extreme inputs remain monotonic");
          previous = amplitude;
        }
      }
    }
  }
  Check(ProportionalAmplitude(0.0, largest, largest, largest) == largest,
        "largest reference saturates before adding the correction");
  Check(ProportionalAmplitude(0.0, largest / 2.0, largest, largest) == largest,
        "largest correction saturates before adding the reference");
  Check(ProportionalAmplitude(largest, largest, largest, largest) == 0.0,
        "largest inputs saturate before multiplying");
  Check(ProportionalAmplitude(2.0, largest, largest, largest) == 0.0,
        "largest reference and correction cancel safely");
  CheckNear(ProportionalAmplitude(1.5, largest, largest, largest),
            largest / 2.0, "large values retain an unsaturated finite result");
  CheckNear(ProportionalAmplitude(largest, 1.0, smallest, 20.0), 1.0,
            "tiny coefficient and huge dispersion are not prematurely zeroed");
  Check(ProportionalAmplitude(1.0, smallest, largest, largest) == smallest,
        "tiny reference is preserved at target");
  Check(std::fetestexcept(FE_OVERFLOW | FE_INVALID | FE_DIVBYZERO) == 0,
        "valid finite inputs do not raise unsafe floating-point exceptions");
}

} // namespace

auto main() -> int {
  try {
    TestExpectedAmplitudes();
    TestInvalidInputs();
    TestMonotonicAndStateless();
    TestExtremeFiniteInputs();
  } catch (const std::exception &error) {
    std::cerr << "Turbulence amplitude controller test failed: " << error.what()
              << '\n';
    return 1;
  }
  std::cout << "Turbulence amplitude controller tests passed.\n";
  return 0;
}
