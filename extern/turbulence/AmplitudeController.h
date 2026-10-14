#ifndef TURBULENCE_AMPLITUDE_CONTROLLER_H
#define TURBULENCE_AMPLITUDE_CONTROLLER_H

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace NameSpaceTurbGen {

// Stateless proportional amplitude for the instantaneous total velocity
// dispersion, normalized by its positive target. The caller supplies the
// reference amplitude, proportional coefficient, and absolute amplitude cap.
// Returns clamp(referenceAmplitude + kp * (1 - normalizedDispersion),
//               0, maxAmplitude).
inline auto ProportionalAmplitude(const double normalizedDispersion,
                                  const double referenceAmplitude,
                                  const double kp, const double maxAmplitude)
    -> double {
  if (!std::isfinite(normalizedDispersion) || normalizedDispersion < 0.0) {
    throw std::invalid_argument(
        "Normalized turbulence dispersion must be finite and nonnegative.");
  }
  if (!std::isfinite(referenceAmplitude) || referenceAmplitude < 0.0) {
    throw std::invalid_argument(
        "Reference turbulence amplitude must be finite and nonnegative.");
  }
  if (!std::isfinite(kp) || kp <= 0.0 || !std::isfinite(maxAmplitude) ||
      maxAmplitude <= 0.0) {
    throw std::invalid_argument("Turbulence proportional coefficient and "
                                "maximum amplitude must be finite "
                                "and positive.");
  }

  if (normalizedDispersion <= 1.0) {
    if (referenceAmplitude >= maxAmplitude) {
      return maxAmplitude;
    }
    // The error is at most one, so this multiplication cannot overflow.
    const double increase = kp * (1.0 - normalizedDispersion);
    // Saturate before adding, including when either amplitude is near DBL_MAX.
    if (increase >= maxAmplitude - referenceAmplitude) {
      return maxAmplitude;
    }
    return referenceAmplitude + increase;
  }

  const double excess = normalizedDispersion - 1.0;
  // Division by an excess of at least one cannot overflow. Above this
  // threshold the correction exceeds the reference, so the amplitude is zero.
  if (excess >= 1.0 && kp > referenceAmplitude / excess) {
    return 0.0;
  }
  // At the rounded threshold, a separate product could overflow before
  // cancellation with the reference. Fusing the subtraction keeps it finite.
  const double amplitude = std::fma(-kp, excess, referenceAmplitude);
  return std::max(0.0, std::min(maxAmplitude, amplitude));
}

} // namespace NameSpaceTurbGen

#endif // TURBULENCE_AMPLITUDE_CONTROLLER_H
