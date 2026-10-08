/// \file VtkWriter.cpp
/// \brief Write polylines as VTK XML PolyData (.vtp) or legacy binary VTK (.vtk).

#include "VtkWriter.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <type_traits>
#include <vector>

#include <AMReX.H>

namespace ffieldlines
{
namespace
{

template <typename T> constexpr auto VtkTypeName() -> const char *
{
	if constexpr (std::is_same_v<T, double>) {
		return "Float64";
	} else if constexpr (std::is_same_v<T, int64_t>) {
		return "Int64";
	} else {
		static_assert(std::is_same_v<T, int32_t>);
		return "Int32";
	}
}

/// Accumulates the appended-data section and the matching DataArray headers.
class AppendedData
{
      public:
	template <typename T> auto Add(std::string const &name, std::vector<T> const &values, int ncomp = 1) -> std::string
	{
		std::ostringstream header;
		header << "<DataArray type=\"" << VtkTypeName<T>() << "\"";
		if (!name.empty()) {
			header << " Name=\"" << name << "\"";
		}
		// NumberOfTuples is required for FieldData arrays, which have no point or cell count to infer it from
		header << " NumberOfComponents=\"" << ncomp << "\" NumberOfTuples=\"" << values.size() / ncomp << "\" format=\"appended\" offset=\""
		       << bytes_.size() << "\"/>\n";

		const uint64_t nbytes = values.size() * sizeof(T);
		Append(&nbytes, sizeof(nbytes));
		Append(values.data(), nbytes);
		return header.str();
	}

	[[nodiscard]] auto Bytes() const -> std::vector<char> const & { return bytes_; }

      private:
	void Append(const void *data, size_t n)
	{
		const auto *p = static_cast<const char *>(data);
		bytes_.insert(bytes_.end(), p, p + n);
	}

	std::vector<char> bytes_;
};

/// XML comments may not contain "--".
auto XmlCommentSafe(std::string text) -> std::string
{
	for (size_t pos = text.find("--"); pos != std::string::npos; pos = text.find("--", pos)) {
		text.replace(pos, 2, "- -");
	}
	return text;
}

void CheckStream(std::ofstream const &out, std::string const &path)
{
	if (!out.good()) {
		amrex::Abort("ffieldlines: failed writing '" + path + "'");
	}
}

/// Write values in big-endian order (legacy VTK binary).
template <typename T> void WriteBigEndian(std::ofstream &out, std::vector<T> const &values)
{
	std::vector<char> buffer(values.size() * sizeof(T));
	std::memcpy(buffer.data(), values.data(), buffer.size());
	if constexpr (std::endian::native == std::endian::little) {
		for (auto element = buffer.begin(); element != buffer.end(); element += sizeof(T)) {
			std::reverse(element, element + sizeof(T));
		}
	}
	out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
	out << "\n";
}

} // namespace

void WriteVtp(std::string const &path, Polylines const &lines, VtkMetadata const &meta)
{
	const int64_t npoints = static_cast<int64_t>(lines.points.size() / 3);
	const int64_t npieces = static_cast<int64_t>(lines.offsets.size()) - 1;

	AppendedData data;
	std::ostringstream xml;
	xml << "<?xml version=\"1.0\"?>\n";
	xml << "<!--\n" << XmlCommentSafe(meta.description) << "\n-->\n";
	xml << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"" << (std::endian::native == std::endian::little ? "LittleEndian" : "BigEndian")
	    << "\" header_type=\"UInt64\">\n";
	xml << "<PolyData>\n";
	xml << "<FieldData>\n";
	xml << data.Add("TimeValue", std::vector<double>{meta.time});
	xml << data.Add("TIME", std::vector<double>{meta.time});
	xml << data.Add("CYCLE", std::vector<int64_t>{meta.cycle});
	xml << "</FieldData>\n";
	xml << "<Piece NumberOfPoints=\"" << npoints << "\" NumberOfVerts=\"0\" NumberOfLines=\"" << npieces
	    << "\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";

	xml << "<PointData Scalars=\"cos_vB\">\n";
	for (auto const &[name, values] : lines.pointData) {
		xml << data.Add(name, values);
	}
	xml << "</PointData>\n";

	xml << "<CellData Scalars=\"seed_id\">\n";
	xml << data.Add("seed_id", lines.pieceSeed);
	xml << data.Add("piece_index", lines.pieceIndex);
	xml << data.Add("status_forward", lines.statusForward);
	xml << data.Add("status_backward", lines.statusBackward);
	xml << data.Add("length_forward", lines.lengthForward);
	xml << data.Add("length_backward", lines.lengthBackward);
	xml << "</CellData>\n";

	xml << "<Points>\n" << data.Add("Points", lines.points, 3) << "</Points>\n";

	std::vector<int64_t> connectivity(npoints);
	for (int64_t i = 0; i < npoints; ++i) {
		connectivity[i] = i;
	}
	const std::vector<int64_t> endOffsets(lines.offsets.begin() + 1, lines.offsets.end());
	xml << "<Lines>\n";
	xml << data.Add("connectivity", connectivity);
	xml << data.Add("offsets", endOffsets);
	xml << "</Lines>\n";
	xml << "</Piece>\n</PolyData>\n";
	xml << "<AppendedData encoding=\"raw\">\n_";

	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	if (!out.good()) {
		amrex::Abort("ffieldlines: cannot open '" + path + "' for writing");
	}
	const std::string head = xml.str();
	out.write(head.data(), static_cast<std::streamsize>(head.size()));
	out.write(data.Bytes().data(), static_cast<std::streamsize>(data.Bytes().size()));
	out << "\n</AppendedData>\n</VTKFile>\n";
	CheckStream(out, path);
}

void WriteLegacyVtk(std::string const &path, Polylines const &lines, VtkMetadata const &meta)
{
	const int64_t npoints = static_cast<int64_t>(lines.points.size() / 3);
	const int64_t npieces = static_cast<int64_t>(lines.offsets.size()) - 1;
	if (npoints + npieces > std::numeric_limits<int32_t>::max()) {
		amrex::Abort("ffieldlines: too many points for legacy VTK; use --format vtp");
	}

	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	if (!out.good()) {
		amrex::Abort("ffieldlines: cannot open '" + path + "' for writing");
	}
	out << "# vtk DataFile Version 3.0\n";
	out << "ffieldlines field lines\n";
	out << "BINARY\n";
	out << "DATASET POLYDATA\n";
	out << "FIELD FieldData 2\n";
	out << "TIME 1 1 double\n";
	WriteBigEndian(out, std::vector<double>{meta.time});
	out << "CYCLE 1 1 int\n";
	WriteBigEndian(out, std::vector<int32_t>{static_cast<int32_t>(meta.cycle)});

	out << "POINTS " << npoints << " double\n";
	WriteBigEndian(out, lines.points);

	std::vector<int32_t> cells;
	cells.reserve(npoints + npieces);
	for (int64_t q = 0; q < npieces; ++q) {
		cells.push_back(static_cast<int32_t>(lines.offsets[q + 1] - lines.offsets[q]));
		for (int64_t i = lines.offsets[q]; i < lines.offsets[q + 1]; ++i) {
			cells.push_back(static_cast<int32_t>(i));
		}
	}
	out << "LINES " << npieces << " " << cells.size() << "\n";
	WriteBigEndian(out, cells);

	const std::vector<int32_t> seeds(lines.pieceSeed.begin(), lines.pieceSeed.end());
	out << "CELL_DATA " << npieces << "\n";
	out << "FIELD FieldData 6\n";
	out << "seed_id 1 " << npieces << " int\n";
	WriteBigEndian(out, seeds);
	out << "piece_index 1 " << npieces << " int\n";
	WriteBigEndian(out, lines.pieceIndex);
	out << "status_forward 1 " << npieces << " int\n";
	WriteBigEndian(out, lines.statusForward);
	out << "status_backward 1 " << npieces << " int\n";
	WriteBigEndian(out, lines.statusBackward);
	out << "length_forward 1 " << npieces << " double\n";
	WriteBigEndian(out, lines.lengthForward);
	out << "length_backward 1 " << npieces << " double\n";
	WriteBigEndian(out, lines.lengthBackward);

	out << "POINT_DATA " << npoints << "\n";
	out << "FIELD FieldData " << lines.pointData.size() << "\n";
	for (auto const &[name, values] : lines.pointData) {
		out << name << " 1 " << npoints << " double\n";
		WriteBigEndian(out, values);
	}
	CheckStream(out, path);
}

} // namespace ffieldlines
