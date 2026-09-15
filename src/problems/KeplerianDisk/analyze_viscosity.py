#!/usr/bin/env python3
"""Evaluate the current 2D AV stencil and a proposed face Balsara factor.

Standard library only. This samples the analytic initial velocity field; it does
not evolve the disk or measure the full reconstructed artificial-viscosity flux.
The constants match testKeplerianDisk.cpp. Run from any working directory.
"""

import argparse
import math


def velocity(x, y):
    omega = max(x * x + y * y, 0.25**2) ** -0.75
    return -omega * y, omega * x


def sensor(x, y, h, field=velocity):
    """X-face estimator; Y faces have identical statistics by disk symmetry."""
    left, right = field(x - h / 2, y), field(x + h / 2, y)
    low = [field(a, y - h) for a in (x - h / 2, x + h / 2)]
    high = [field(a, y + h) for a in (x - h / 2, x + h / 2)]
    transverse = [
        min(hi[1] - mid[1], mid[1] - lo[1])
        for lo, mid, hi in zip(low, (left, right), high)
    ]
    # hydro_system.hpp: du + 0.5*(dvl+dvr), in VELOCITY units.
    compression = right[0] - left[0] + sum(transverse) / 2
    # Independent face-centered gradient, all terms in inverse-time units.
    div = (right[0] - left[0]) / h + sum(hi[1] - lo[1] for lo, hi in zip(low, high)) / (4 * h)
    curl = (right[1] - left[1]) / h - sum(hi[0] - lo[0] for lo, hi in zip(low, high)) / (4 * h)
    return compression, div, abs(curl)


def sample(n, coefficient, epsilon):
    h = 4 / n
    count = active = 0
    av_sum = limited_sum = factor_sum = 0.0
    for i in range(n + 1):
        x = -2 + i * h
        for j in range(n):
            y = -2 + (j + 0.5) * h
            r2 = x * x + y * y
            if not 0.7**2 < r2 < 1.3**2:
                continue
            compression, div, curl = sensor(x, y, h)
            rho = 0.01 + math.exp(-((r2 - 1) / 0.3) ** 2)
            cs = math.sqrt((5 / 3) * 0.001 / rho)
            denominator = abs(div) + curl + epsilon * cs / h
            factor = abs(div) / denominator if denominator else 0.0
            av = coefficient * max(-compression, 0.0)
            count += 1
            active += compression < 0
            av_sum += av
            limited_sum += av * factor
            factor_sum += factor
    return active / count, av_sum / count, factor_sum / count, limited_sum / av_sum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--coefficient", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    args = parser.parse_args()
    if any(n < 16 for n in args.resolutions) or args.coefficient <= 0 or args.epsilon < 0:
        parser.error("resolutions must be >=16, coefficient positive, and epsilon nonnegative")

    # Exact affine controls test the signs/normalization independently of the disk.
    for field in (lambda x, y: (y, 0.0), lambda x, y: (x + y, -x - y)):
        compression, div, _ = sensor(1.0, 0.5, 0.125, field)
        assert compression == 0.0 and div == 0.0
    compression, div, curl = sensor(1.0, 0.5, 0.125, lambda x, y: (-x, 0.0))
    assert compression == -0.125 and div == -1.0 and curl == 0.0

    print("# analytic x-faces, 0.7 < r < 1.3; K =", args.coefficient, "epsilon =", args.epsilon)
    print("# N active_fraction mean_AV_speed mean_Balsara_factor sum_limited_AV/sum_AV")
    for n in args.resolutions:
        print(n, *(f"{value:.8e}" for value in sample(n, args.coefficient, args.epsilon)))


if __name__ == "__main__":
    main()
