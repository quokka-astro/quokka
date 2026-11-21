/// \file FieldLineContainer.cpp
/// \brief Field lines as AMReX particles: seeding, RK4 push, and point recording.

#include "FieldLineContainer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>

#include "Interp.hpp"

namespace ffieldlines
{

FieldLineContainer::FieldLineContainer(FieldData const &fd) : amrex::AmrParticleContainer<NReal, NInt>(fd.geom, fd.dmap, fd.grids, fd.refRatio), fd_(fd) {}

auto FieldLineContainer::AddSeeds(std::vector<Point3> const &seeds, const TraceDirection direction) -> int64_t
{
	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto &ptile = DefineAndReturnParticleTile(0, 0, 0);
		for (int i = 0; i < std::ssize(seeds); ++i) {
			for (const int dir : {+1, -1}) {
				if ((dir > 0 && direction == TraceDirection::Backward) || (dir < 0 && direction == TraceDirection::Forward)) {
					continue;
				}
				ParticleType p;
				p.id() = ParticleType::NextID();
				p.cpu() = amrex::ParallelDescriptor::MyProc();
				for (int d = 0; d < 3; ++d) {
					p.pos(d) = seeds[i][d];
				}
				p.rdata(ArcLength) = 0.0;
				p.rdata(SeedX) = seeds[i][0];
				p.rdata(SeedY) = seeds[i][1];
				p.rdata(SeedZ) = seeds[i][2];
				p.idata(SeedId) = i;
				p.idata(Dir) = dir;
				p.idata(Steps) = 0;
				ptile.push_back(p);
			}
		}
	}
	Redistribute();
	return TotalNumberOfParticles();
}

void FieldLineContainer::Sweep(TraceParams const &params, const bool terminateAll)
{
	using amrex::Real;
	using namespace amrex::literals;

	const int nsteps = terminateAll ? 0 : params.stepsPerSweep;
	const int maxRec = (params.stepsPerSweep / params.outputEvery) + 3;
	const int outputEvery = params.outputEvery;
	const Real maxLength = params.maxLength;
	const Real bMin = params.bMin;
	const int64_t maxSteps = params.maxSteps;
	const amrex::GpuArray<int, 3> periodic{params.periodic[0], params.periodic[1], params.periodic[2]};

	for (int lev = 0; lev <= fd_.finestLevel; ++lev) {
		auto const &geom = fd_.geom[lev];
		const Vec3 plo = geom.ProbLoArray();
		const Vec3 phi = geom.ProbHiArray();
		const Vec3 dx = geom.CellSizeArray();
		const Vec3 dxi = geom.InvCellSizeArray();
		const amrex::IntVect domLo = geom.Domain().smallEnd();
		const Real dxMin = std::min({dx[0], dx[1], dx[2]});
		const Real dxMax = std::max({dx[0], dx[1], dx[2]});
		const Real h = params.stepFraction * dxMin;
		const Real loopRadius = params.loopTolerance * dxMin;
		const Real loopMinArc = 4.0_rt * dxMax;
		const bool hasMask = lev < fd_.finestLevel;

		for (ParIterType pti(*this, lev); pti.isValid(); ++pti) {
			const int np = pti.numParticles();
			if (np == 0) {
				continue;
			}
			ParticleType *pstruct = pti.GetArrayOfStructs()().data();
			const amrex::Box validBox = fd_.grids[lev][pti.index()];
			const auto state = fd_.state[lev].const_array(pti);
			const amrex::Array4<int const> mask = hasMask ? fd_.fineMask[lev].const_array(pti) : amrex::Array4<int const>{};

			amrex::Gpu::DeviceVector<Real> buffer(static_cast<size_t>(np) * maxRec * NRecordCols);
			amrex::Gpu::DeviceVector<int> counts(np, 0);
			Real *bufferPtr = buffer.data();
			int *countPtr = counts.data();

			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int ip) noexcept {
				ParticleType &p = pstruct[ip];
				countPtr[ip] = 0;
				if (!p.id().is_valid()) {
					return;
				}
				Real *out = bufferPtr + (static_cast<size_t>(ip) * maxRec * NRecordCols);
				int n = 0;
				const Real dir = static_cast<Real>(p.idata(Dir));
				const Real nan = std::numeric_limits<Real>::quiet_NaN();
				auto position = [&]() -> Vec3 { return {p.pos(0), p.pos(1), p.pos(2)}; };

				auto record = [&](const int status) {
					Real *r = out + (static_cast<size_t>(n) * NRecordCols);
					const Vec3 pos = position();
					r[ColSeed] = static_cast<Real>(p.idata(SeedId));
					r[ColDir] = dir;
					r[ColSteps] = static_cast<Real>(p.idata(Steps));
					r[ColStatus] = static_cast<Real>(status);
					r[ColX] = pos[0];
					r[ColY] = pos[1];
					r[ColZ] = pos[2];
					r[ColArc] = p.rdata(ArcLength);
					FieldSample s{};
					if (SampleFields(state, plo, dxi, domLo, pos, s) && s.rho > 0.0_rt) {
						r[ColRho] = s.rho;
						r[ColTemp] = s.temp;
						r[ColBMag] = std::sqrt(Dot(s.b, s.b));
						r[ColVMag] = std::sqrt(Dot(s.m, s.m)) / s.rho;
						r[ColVDotB] = Dot(s.m, s.b) / s.rho;
					} else {
						r[ColRho] = nan;
						r[ColTemp] = nan;
						r[ColBMag] = nan;
						r[ColVMag] = nan;
						r[ColVDotB] = nan;
					}
					++n;
				};

				// unit direction dir * B/|B| at x; returns a termination status or Active
				auto direction = [&](Vec3 const &x, Vec3 &k) -> int {
					FieldSample s{};
					if (!SampleFields(state, plo, dxi, domLo, x, s)) {
						return BadSample;
					}
					const Real bmag = std::sqrt(Dot(s.b, s.b));
					if (!std::isfinite(bmag)) {
						return BadSample;
					}
					if (!(bmag > bMin) || bmag == 0.0_rt) {
						return WeakField;
					}
					for (int d = 0; d < 3; ++d) {
						k[d] = dir * s.b[d] / bmag;
					}
					return Active;
				};

				// RK4 stage point x0 + c * k
				auto stage = [](Vec3 const &x0, const Real c, Vec3 const &k) -> Vec3 {
					return {x0[0] + c * k[0], x0[1] + c * k[1], x0[2] + c * k[2]};
				};

				if (nsteps == 0) {
					record(MaxSweeps);
					p.id().make_invalid();
					countPtr[ip] = n;
					return;
				}

				for (int step = 0; step < nsteps; ++step) {
					const Vec3 x0 = position();
					const amrex::IntVect cell = CellIndex(x0, plo, dxi, domLo);
					if (!validBox.contains(cell)) {
						break; // left this grid: wait for Redistribute
					}
					if (hasMask && mask(cell) != 0) {
						break; // entered a finer level: wait for Redistribute
					}
					if (p.idata(Steps) == 0) {
						record(Active); // the seed point
					}

					const Real hs = amrex::min(h, maxLength - std::abs(p.rdata(ArcLength)));
					Vec3 k1{};
					Vec3 k2{};
					Vec3 k3{};
					Vec3 k4{};
					int status = direction(x0, k1);
					if (status == Active) {
						status = direction(stage(x0, 0.5_rt * hs, k1), k2);
					}
					if (status == Active) {
						status = direction(stage(x0, 0.5_rt * hs, k2), k3);
					}
					if (status == Active) {
						status = direction(stage(x0, hs, k3), k4);
					}
					if (status != Active) {
						record(status); // terminate at the step start
						p.id().make_invalid();
						break;
					}

					for (int d = 0; d < 3; ++d) {
						p.pos(d) = x0[d] + (hs / 6.0_rt) * (k1[d] + 2.0_rt * k2[d] + 2.0_rt * k3[d] + k4[d]);
					}
					p.rdata(ArcLength) += dir * hs;
					p.idata(Steps) += 1;
					const Real arc = std::abs(p.rdata(ArcLength));

					bool outside = false;
					Real dist2 = 0.0_rt;
					for (int d = 0; d < 3; ++d) {
						Real delta = p.pos(d) - p.rdata(SeedX + d);
						if (periodic[d] != 0) {
							const Real length = phi[d] - plo[d];
							delta -= length * std::round(delta / length); // minimum image
						} else if (p.pos(d) < plo[d] || p.pos(d) >= phi[d]) {
							outside = true;
						}
						dist2 += delta * delta;
					}

					if (outside) {
						status = DomainExit;
					} else if (arc >= maxLength * (1.0_rt - 1.0e-12_rt)) {
						status = MaxLength;
					} else if (p.idata(Steps) >= maxSteps) {
						status = MaxSteps;
					} else if (arc > loopMinArc && dist2 <= loopRadius * loopRadius) {
						status = ClosedLoop;
					}
					if (status != Active || p.idata(Steps) % outputEvery == 0) {
						record(status);
					}
					if (status != Active) {
						p.id().make_invalid();
						break;
					}
				}
				countPtr[ip] = n;
			});

			// compact this tile's records into the rank-local buffer
			std::vector<int> hostCounts(np);
			amrex::Gpu::copy(amrex::Gpu::deviceToHost, counts.begin(), counts.end(), hostCounts.begin());
			std::vector<Real> hostBuffer(buffer.size());
			amrex::Gpu::copy(amrex::Gpu::deviceToHost, buffer.begin(), buffer.end(), hostBuffer.begin());
			for (int ip = 0; ip < np; ++ip) {
				const auto first = hostBuffer.begin() + static_cast<std::ptrdiff_t>(ip) * maxRec * NRecordCols;
				records_.insert(records_.end(), first, first + static_cast<std::ptrdiff_t>(hostCounts[ip]) * NRecordCols);
			}
		}
	}
}

} // namespace ffieldlines
