#include "AMReX.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuLaunch.H"
#include "spherical_geometry.hpp"
#include <array>
#include <cmath>
#include <iostream>

auto main(int argc, char **argv) -> int
{
	amrex::Initialize(argc, argv);
	int failures = 0;
	{
		for (const amrex::Real scale : {1., 1.e-14, 1.e14}) {
			amrex::Gpu::DeviceVector<amrex::Real> results(10);
			auto *areas = results.data();
			amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int) {
				using quokka::math::detail::planeBoxSectionArea;
				const auto s = scale;
				areas[0] = quokka::math::sphericalSectionAreaInCell(s, .5 * s, 1.5 * s, -.5 * s, .5 * s, -.5 * s, .5 * s);
				areas[1] = planeBoxSectionArea(-.5 * s, .5 * s, -.5 * s, .5 * s, -.5 * s, .5 * s, 1., 0., 0., .5 * s);
				const auto n = 1. / std::sqrt(2.);
				areas[2] = planeBoxSectionArea(-.5 * s, .5 * s, -.5 * s, .5 * s, -.5 * s, .5 * s, n, n, 0., n * s);
				areas[3] = planeBoxSectionArea(-.5 * s, .5 * s, -.5 * s, .5 * s, -.5 * s, .5 * s, 1., 0., 0., 2. * s);
				areas[4] = planeBoxSectionArea(7.5 * s, 8.5 * s, -4.5 * s, -3.5 * s, 1.5 * s, 2.5 * s, 1., 0., 0., 8. * s);
				const auto diagonal = 1. / std::sqrt(3.);
				areas[5] = planeBoxSectionArea(-.5 * s, .5 * s, -.5 * s, .5 * s, -.5 * s, .5 * s, diagonal, diagonal, diagonal, 0.);
				areas[6] = planeBoxSectionArea(0., 0., -.5 * s, .5 * s, -.5 * s, .5 * s, 1., 0., 0., 0.);
				areas[7] = planeBoxSectionArea(.5 * s, -.5 * s, -.5 * s, .5 * s, -.5 * s, .5 * s, 1., 0., 0., 0.);
				areas[8] = planeBoxSectionArea(7.5 * s, 8.5 * s, -4.5 * s, -3.5 * s, 1.5 * s, 2.5 * s, diagonal, diagonal, diagonal,
							       6. * s * diagonal);
				areas[9] = planeBoxSectionArea(-.5 * s, .5 * s, -s, s, -1.5 * s, 1.5 * s, 1., 0., 0., 0.);
			});
			std::array<amrex::Real, 10> actual{};
			amrex::Gpu::copy(amrex::Gpu::deviceToHost, results.begin(), results.end(), actual.begin());
			const std::array<amrex::Real, 10> expected{1., 1., 0., 0., 1., 3. * std::sqrt(3.) / 4., 0., 0., 3. * std::sqrt(3.) / 4., 6.};
			for (int i = 0; i < 10; ++i) {
				const auto normalized = actual[i] / (scale * scale);
				if (!std::isfinite(normalized) || std::abs(normalized - expected[i]) > 1.e-12) {
					std::cerr << "scale " << scale << " case " << i << ": got " << normalized << ", expected " << expected[i] << '\n';
					++failures;
				}
			}
		}
	}
	amrex::Finalize();
	return failures == 0 ? 0 : 1;
}
