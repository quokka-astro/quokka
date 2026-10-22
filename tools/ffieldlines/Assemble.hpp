#ifndef FFIELDLINES_ASSEMBLE_HPP_
#define FFIELDLINES_ASSEMBLE_HPP_
/// \file Assemble.hpp
/// \brief Turn recorded points into polylines ready for VTK output.

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ffieldlines
{

/// Polylines in CSR form. Each line (seed) is split into one or more pieces;
/// a new piece starts wherever the line wraps through a periodic boundary.
struct Polylines {
	std::vector<double> points;   ///< 3 * P coordinates
	std::vector<int64_t> offsets; ///< Q + 1 offsets into points (in points, not coordinates)

	/// per-point arrays, each of length P
	std::vector<std::pair<std::string, std::vector<double>>> pointData;

	/// per-piece arrays, each of length Q
	std::vector<int64_t> pieceSeed;
	std::vector<int32_t> pieceIndex;
	std::vector<int32_t> statusForward;
	std::vector<int32_t> statusBackward;
	std::vector<double> lengthForward;
	std::vector<double> lengthBackward;

	/// number of lines per final status (forward half, or backward if forward was not traced)
	std::array<int64_t, 8> statusCounts{};
	int64_t nLines = 0;
	int64_t nDegenerate = 0; ///< seeds whose line has fewer than two points
};

/// Status value reported for a half-line that was not traced (e.g. --direction forward).
constexpr int32_t statusNotTraced = -1;

/// Assemble gathered records (NRecordCols doubles per point, in any order) into
/// polylines. The result depends only on the record contents, not their order.
/// `domainLength[d]` is used to detect periodic wraps on axes with periodic[d] != 0.
auto AssemblePolylines(std::vector<double> records, std::array<int, 3> const &periodic, std::array<double, 3> const &domainLength, bool hasTemperature)
    -> Polylines;

} // namespace ffieldlines

#endif // FFIELDLINES_ASSEMBLE_HPP_
