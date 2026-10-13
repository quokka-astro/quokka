#ifndef FFIELDLINES_VTKWRITER_HPP_
#define FFIELDLINES_VTKWRITER_HPP_
/// \file VtkWriter.hpp
/// \brief Write polylines as VTK XML PolyData (.vtp) or legacy binary VTK (.vtk).

#include <string>

#include "Assemble.hpp"

namespace ffieldlines
{

/// Provenance written alongside the geometry.
struct VtkMetadata {
	double time = 0.0;
	int64_t cycle = 0;
	std::string description; ///< free text (command line, plotfile metadata); written as an XML comment in .vtp
};

/// VTK XML PolyData with raw appended binary data. Readable by ParaView and VisIt.
void WriteVtp(std::string const &path, Polylines const &lines, VtkMetadata const &meta);

/// Legacy binary (big-endian) VTK POLYDATA, for readers without XML support.
void WriteLegacyVtk(std::string const &path, Polylines const &lines, VtkMetadata const &meta);

} // namespace ffieldlines

#endif // FFIELDLINES_VTKWRITER_HPP_
