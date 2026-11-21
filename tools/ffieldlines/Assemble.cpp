/// \file Assemble.cpp
/// \brief Turn recorded points into polylines ready for VTK output.

#include "Assemble.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "Records.hpp"

namespace ffieldlines
{
namespace
{

/// Records of one half-line, ordered by step.
using HalfLine = std::vector<const double *>;

auto FinalStatus(HalfLine const &half) -> int32_t { return half.empty() ? statusNotTraced : static_cast<int32_t>(half.back()[ColStatus]); }

auto FinalLength(HalfLine const &half) -> double { return half.empty() ? 0.0 : std::abs(half.back()[ColArc]); }

} // namespace

auto AssemblePolylines(std::vector<double> records, std::array<int, 3> const &periodic, std::array<double, 3> const &domainLength, const bool hasTemperature)
    -> Polylines
{
	const size_t nrec = records.size() / NRecordCols;
	std::vector<size_t> order(nrec);
	std::iota(order.begin(), order.end(), size_t{0});

	// (seed, dir, steps, status): backward half first, and for duplicate steps the
	// terminal (non-zero status) record last so deduplication keeps it. Full ties
	// (identical keys) fall back to the original index so the order is total.
	std::sort(order.begin(), order.end(), [&records](const size_t a, const size_t b) {
		for (const int col : {ColSeed, ColDir, ColSteps, ColStatus}) {
			const double va = records[(a * NRecordCols) + col];
			const double vb = records[(b * NRecordCols) + col];
			if (va != vb) {
				return va < vb;
			}
		}
		return a < b;
	});
	std::vector<const double *> rows(nrec);
	for (size_t i = 0; i < nrec; ++i) {
		rows[i] = records.data() + (order[i] * NRecordCols);
	}
	std::vector<const double *> unique;
	unique.reserve(rows.size());
	for (size_t i = 0; i < rows.size(); ++i) {
		const bool sameAsNext = (i + 1 < rows.size()) && rows[i][ColSeed] == rows[i + 1][ColSeed] && rows[i][ColDir] == rows[i + 1][ColDir] &&
					rows[i][ColSteps] == rows[i + 1][ColSteps];
		if (!sameAsNext) {
			unique.push_back(rows[i]);
		}
	}

	Polylines out;
	std::vector<double> arc;
	std::vector<double> rho;
	std::vector<double> temp;
	std::vector<double> bmag;
	std::vector<double> vmag;
	std::vector<double> vdotb;
	std::vector<double> cosvb;
	out.offsets.push_back(0);

	auto appendPoint = [&](const double *r) {
		out.points.insert(out.points.end(), {r[ColX], r[ColY], r[ColZ]});
		arc.push_back(r[ColArc]);
		rho.push_back(r[ColRho]);
		temp.push_back(r[ColTemp]);
		bmag.push_back(r[ColBMag]);
		vmag.push_back(r[ColVMag]);
		vdotb.push_back(r[ColVDotB]);
		const double denom = r[ColVMag] * r[ColBMag];
		cosvb.push_back((denom > 1.0e-300) ? std::clamp(r[ColVDotB] / denom, -1.0, 1.0) : std::numeric_limits<double>::quiet_NaN());
	};

	size_t i = 0;
	while (i < unique.size()) {
		const double seed = unique[i][ColSeed];
		HalfLine backward;
		HalfLine forward;
		for (; i < unique.size() && unique[i][ColSeed] == seed; ++i) {
			(unique[i][ColDir] < 0.0 ? backward : forward).push_back(unique[i]);
		}

		const int32_t statusF = FinalStatus(forward);
		const int32_t statusB = FinalStatus(backward);
		const int32_t primary = forward.empty() ? statusB : statusF;
		if (primary >= 0 && primary < NStatus) {
			++out.statusCounts[primary];
		}
		++out.nLines;

		// point sequence: reversed backward half (its seed point is shared with the
		// forward half), then the forward half. A closed forward loop already traces
		// the whole line, so the backward half is dropped and the loop is closed.
		std::vector<const double *> sequence;
		if (statusF == ClosedLoop) {
			sequence = forward;
			sequence.push_back(forward.front());
		} else {
			for (auto it = backward.rbegin(); it != backward.rend(); ++it) {
				if (!forward.empty() && (*it)[ColSteps] == 0.0) {
					continue;
				}
				sequence.push_back(*it);
			}
			sequence.insert(sequence.end(), forward.begin(), forward.end());
		}

		// split into pieces at periodic wraps
		std::vector<size_t> breaks{0};
		for (size_t k = 1; k < sequence.size(); ++k) {
			for (int d = 0; d < 3; ++d) {
				if (periodic[d] != 0 && std::abs(sequence[k][ColX + d] - sequence[k - 1][ColX + d]) > 0.5 * domainLength[d]) {
					breaks.push_back(k);
					break;
				}
			}
		}
		breaks.push_back(sequence.size());

		int32_t pieceIndex = 0;
		for (size_t b = 0; b + 1 < breaks.size(); ++b) {
			if (breaks[b + 1] - breaks[b] < 2) {
				continue; // a single point is not a line segment
			}
			for (size_t k = breaks[b]; k < breaks[b + 1]; ++k) {
				appendPoint(sequence[k]);
			}
			out.offsets.push_back(static_cast<int64_t>(out.points.size() / 3));
			out.pieceSeed.push_back(static_cast<int64_t>(seed));
			out.pieceIndex.push_back(pieceIndex++);
			out.statusForward.push_back(statusF);
			out.statusBackward.push_back(statusB);
			out.lengthForward.push_back(FinalLength(forward));
			out.lengthBackward.push_back(FinalLength(backward));
		}
		if (pieceIndex == 0) {
			++out.nDegenerate;
		}
	}

	out.pointData.emplace_back("arc_length", std::move(arc));
	out.pointData.emplace_back("density", std::move(rho));
	if (hasTemperature) {
		out.pointData.emplace_back("temperature", std::move(temp));
	}
	out.pointData.emplace_back("B_mag", std::move(bmag));
	out.pointData.emplace_back("v_mag", std::move(vmag));
	out.pointData.emplace_back("v_dot_B", std::move(vdotb));
	out.pointData.emplace_back("cos_vB", std::move(cosvb));
	return out;
}

} // namespace ffieldlines
