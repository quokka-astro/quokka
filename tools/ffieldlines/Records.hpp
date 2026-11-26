#ifndef FFIELDLINES_RECORDS_HPP_
#define FFIELDLINES_RECORDS_HPP_
/// \file Records.hpp
/// \brief Layout of recorded field-line points and termination statuses.

namespace ffieldlines
{

/// Termination status of a half-line. `Active` marks intermediate points.
enum Status : int { Active = 0, MaxLength = 1, MaxSteps = 2, WeakField = 3, DomainExit = 4, ClosedLoop = 5, MaxSweeps = 6, BadSample = 7, NStatus = 8 };

/// Human-readable status names, indexed by Status.
constexpr const char *statusNames[NStatus] = {"active", "max_length", "max_steps", "weak_field", "domain_exit", "closed_loop", "max_sweeps", "bad_sample"};

/// Record layout: one row of NRecordCols doubles per recorded point.
enum RecordCol : int {
	ColSeed = 0, ///< seed index
	ColDir,	     ///< +1 forward, -1 backward
	ColSteps,    ///< integration steps taken on this half-line
	ColStatus,   ///< Status (non-zero only on the final point of a half-line)
	ColX,
	ColY,
	ColZ,
	ColArc,	  ///< signed arc length from the seed
	ColRho,	  ///< density
	ColTemp,  ///< temperature (NaN when not sampled)
	ColBMag,  ///< |B|
	ColVMag,  ///< |v|, with v = m / rho
	ColVDotB, ///< v . B
	NRecordCols
};

} // namespace ffieldlines

#endif // FFIELDLINES_RECORDS_HPP_
