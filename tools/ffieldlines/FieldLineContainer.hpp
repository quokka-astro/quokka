#ifndef FFIELDLINES_FIELDLINECONTAINER_HPP_
#define FFIELDLINES_FIELDLINECONTAINER_HPP_
/// \file FieldLineContainer.hpp
/// \brief Field lines as AMReX particles: seeding, RK4 push, and point recording.

#include <cstdint>
#include <vector>

#include <AMReX_AmrParticles.H>

#include "FieldData.hpp"
#include "Options.hpp"
#include "Records.hpp"
#include "Seeds.hpp"

namespace ffieldlines
{

/// Particle real components.
enum RealComp : int { ArcLength = 0, SeedX, SeedY, SeedZ, NReal };
/// Particle integer components.
enum IntComp : int { SeedId = 0, Dir, Steps, NInt };

/// Integration settings shared by every particle.
struct TraceParams {
	double stepFraction = 0.25;
	double maxLength = 1.0;
	double bMin = 0.0;
	double loopTolerance = 0.5;
	int64_t maxSteps = 1000000;
	int stepsPerSweep = 64;
	int outputEvery = 4;
	std::array<int, 3> periodic{0, 0, 0};
};

/// One particle per half-line. A particle integrates only while its cell is
/// valid on its own grid and not covered by a finer level; otherwise it waits
/// for Redistribute() to move it to the owning grid, level, and rank. Results
/// therefore do not depend on the decomposition or on stepsPerSweep.
class FieldLineContainer : public amrex::AmrParticleContainer<NReal, NInt>
{
      public:
	explicit FieldLineContainer(FieldData const &fd);

	/// Create the particles for `seeds` on the I/O rank and redistribute them.
	/// Returns the global number of particles created.
	auto AddSeeds(std::vector<Point3> const &seeds, TraceDirection direction) -> int64_t;

	/// Advance every active particle by up to params.stepsPerSweep steps,
	/// appending recorded points to the local record buffer. With
	/// `terminateAll`, no steps are taken: every remaining particle records its
	/// current point with status MaxSweeps and is removed.
	void Sweep(TraceParams const &params, bool terminateAll);

	/// Local recorded points, NRecordCols doubles per point.
	[[nodiscard]] auto Records() const -> std::vector<double> const & { return records_; }

      private:
	FieldData const &fd_;
	std::vector<double> records_;
};

} // namespace ffieldlines

#endif // FFIELDLINES_FIELDLINECONTAINER_HPP_
