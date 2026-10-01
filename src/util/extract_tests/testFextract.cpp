#include "fextract.hpp"
#include <cmath>
#include <iostream>

auto main(int argc, char **argv) -> int
{
	amrex::Initialize(argc, argv);
	int failures = 0;
	{
		for (const int origin : {0, 5}) {
			const amrex::Box domain(amrex::IntVect(origin), amrex::IntVect(origin + 3));
			const amrex::RealBox physical(amrex::Array<amrex::Real, AMREX_SPACEDIM>{AMREX_D_DECL(2., 2., 2.)},
						      amrex::Array<amrex::Real, AMREX_SPACEDIM>{AMREX_D_DECL(6., 6., 6.)});
			const amrex::Array<int, AMREX_SPACEDIM> periodic{};
			amrex::Geometry geometry(domain, physical, 0, periodic);
			for (const int boxSize : {4, 2}) {
				amrex::BoxArray boxes(domain);
				boxes.maxSize(boxSize);
				amrex::DistributionMapping mapping(boxes);
				amrex::MultiFab data(boxes, mapping, 2, 0);
				for (amrex::MFIter mfi(data); mfi.isValid(); ++mfi) {
					const auto field = data.array(mfi);
					amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
						const auto value = AMREX_D_TERM((2.5 + i - origin), +10. * (2.5 + j - origin), +100. * (2.5 + k - origin));
						field(i, j, k, 0) = value;
						field(i, j, k, 1) = -value;
					});
				}
				amrex::Gpu::streamSynchronize();
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> coordinates{AMREX_D_DECL(2.25, 3.25, 4.25)};
					coordinates[dir] = -1.e100; // The longitudinal coordinate must be ignored.
					const auto [positions, values] = fextract(data, geometry, dir, coordinates);
					if (amrex::ParallelDescriptor::IOProcessor()) {
						if (positions.size() != 4) {
							++failures;
							continue;
						}
						for (int cell = 0; cell < 4; ++cell) {
							amrex::Real expected = 0.;
							amrex::Real weight = 1.;
							for (int d = 0; d < AMREX_SPACEDIM; ++d) {
								expected += weight * (d == dir ? 2.5 + cell : 2.5 + d);
								weight *= 10.;
							}
							if (positions[cell] != 2.5 + cell || values[0][cell] != expected || values[1][cell] != -expected) {
								std::cerr << "Distinct transverse coordinates selected the wrong profile\n";
								++failures;
							}
						}
					}
				}

				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					for (const amrex::Real coordinate : {2.25, 4.25, -1., 10.}) {
						for (const bool center : {false, true}) {
							const auto [positions, values] = fextract(data, geometry, dir, coordinate, center);
							if (!amrex::ParallelDescriptor::IOProcessor()) {
								continue;
							}
							if (positions.size() != 4 || values.size() != 2 || values[0].size() != 4 || values[1].size() != 4) {
								std::cerr << "Incorrect profile size at origin " << origin << '\n';
								++failures;
								continue;
							}
							const auto transverse =
							    center ? 4.5 : (coordinate < 2. ? 2.5 : (coordinate >= 6. ? 5.5 : std::floor(coordinate) + .5));
							for (int cell = 0; cell < 4; ++cell) {
								amrex::Real expected = 0.;
								amrex::Real weight = 1.;
								for (int d = 0; d < AMREX_SPACEDIM; ++d) {
									expected += weight * (d == dir ? 2.5 + cell : transverse);
									weight *= 10.;
								}
								if (std::abs(positions[cell] - (2.5 + cell)) > 1.e-12 ||
								    std::abs(values[0][cell] - expected) > 1.e-12 ||
								    std::abs(values[1][cell] + expected) > 1.e-12) {
									std::cerr << "Wrong slice or position: origin=" << origin << " direction=" << dir
										  << " coordinate=" << coordinate << " center=" << center << '\n';
									++failures;
									break;
								}
							}
						}
					}
				}
			}
		}
	}
	amrex::ParallelDescriptor::ReduceIntSum(failures);
	amrex::Finalize();
	return failures == 0 ? 0 : 1;
}
