//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
//! \file openPMD.cpp
///  \brief openPMD I/O for snapshots

#include <cstdint>
#include <string>

// AMReX headers
#include "AMReX.H"
#include "AMReX_Geometry.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"

// openPMD headers
#include "openPMD/openPMD.hpp"

namespace quokka::OpenPMDOutput
{

namespace detail
{
/** \brief
 * Convert an IntVect to a std::vector<std::uint64_t>
 * and reverse the order of the elements
 * (used for compatibility with the openPMD API)
 */
auto getReversedVec(const amrex::IntVect &v) -> std::vector<std::uint64_t>
{
	// Convert the IntVect v to and std::vector u
	std::vector<std::uint64_t> u = {AMREX_D_DECL(static_cast<std::uint64_t>(v[0]), static_cast<std::uint64_t>(v[1]), static_cast<std::uint64_t>(v[2]))};
	// Reverse the order of elements, if v corresponds to the indices of a Fortran-order array (like an AMReX FArrayBox)
	// but u is intended to be used with a C-order API (like openPMD)
	std::reverse(u.begin(), u.end());
	return u;
}

/** \brief
 * Convert Real* pointer to a std::vector<double>,
 * and reverse the order of the elements
 * (used for compatibility with the openPMD API)
 */
auto getReversedVec(const amrex::Real *v) -> std::vector<double>
{
	// Convert Real* v to and std::vector u
	std::vector<double> u = {AMREX_D_DECL(static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2]))}; // NOLINT
	// Reverse the order of elements, if v corresponds to the indices of a Fortran-order array (like an AMReX FArrayBox)
	// but u is intended to be used with a C-order API (like openPMD)
	std::reverse(u.begin(), u.end());
	return u;
}

void SetupMeshComponent(openPMD::Mesh &mesh, amrex::Geometry &full_geom)
{
	amrex::Box const &global_box = full_geom.Domain();
	auto global_size = getReversedVec(global_box.size());
	std::vector<double> const grid_spacing = getReversedVec(full_geom.CellSize());
	std::vector<double> const global_offset = getReversedVec(full_geom.ProbLo());
	std::vector<std::string> const coordinateLabels{AMREX_D_DECL("x", "y", "z")};
	std::vector<std::string> const grid_axes{coordinateLabels.rbegin(), coordinateLabels.rend()}; // reverse order

	// Prepare the type of dataset that will be written
	mesh.setDataOrder(openPMD::Mesh::DataOrder::C);
	mesh.setGridSpacing(grid_spacing);
	mesh.setGridGlobalOffset(global_offset);
	mesh.setAxisLabels(grid_axes); // use C-ordering
	mesh.setAttribute("fieldSmoothing", "none");

	auto mesh_comp = mesh[openPMD::MeshRecordComponent::SCALAR];
	auto const dataset = openPMD::Dataset(openPMD::determineDatatype<amrex::Real>(), global_size);
	mesh_comp.resetDataset(dataset);
	std::vector<amrex::Real> const relativePosition{0.5, 0.5, 0.5}; // cell-centered only (for now)
	mesh_comp.setPosition(relativePosition);
}

auto GetMeshComponentName(int meshLevel, std::string const &field_name) -> std::string
{
	std::string new_field_name = field_name;
	if (meshLevel > 0) {
		new_field_name += std::string("_lvl").append(std::to_string(meshLevel));
	}
	return new_field_name;
}
} // namespace detail
//----------------------------------------------------------------------------------------
//! \fn void OpenPMDOutput:::WriteOutputFile(Mesh *pm)
//  \brief  Write cell-centered MultiFab using openPMD
void WriteFile(const std::vector<std::string> &varnames, int const output_levels, amrex::Vector<const amrex::MultiFab *> &mf,
	       amrex::Vector<amrex::Geometry> &geom, const std::string &output_basename, amrex::Real const time, int const file_number)
{
	// open file
	std::string const filename = output_basename + ".bp";
	auto series = openPMD::Series(filename, openPMD::Access::CREATE, amrex::ParallelDescriptor::Communicator());
	series.setSoftware("Quokka", "1.0");

	auto series_iteration = series.iterations[file_number];
	series_iteration.open();
	series_iteration.setTime(time);
	auto meshes = series_iteration.meshes;

	// loop over levels up to output_levels
	for (int lev = 0; lev < output_levels; lev++) {
		amrex::Geometry full_geom = geom[lev];
		amrex::Box const &global_box = full_geom.Domain();
		int const ncomp = mf[lev]->nComp();

		for (int icomp = 0; icomp < ncomp; icomp++) {
			const std::string field_name = detail::GetMeshComponentName(lev, varnames[icomp]);
			if (!meshes.contains(field_name)) {
				auto mesh = meshes[field_name];
				detail::SetupMeshComponent(mesh, full_geom);
			}
		}

		// pass data pointers for each box to ADIOS
		for (int icomp = 0; icomp < ncomp; icomp++) {
			std::string const field_name = detail::GetMeshComponentName(lev, varnames[icomp]);
			openPMD::MeshRecordComponent mesh = series_iteration.meshes[field_name][openPMD::MeshRecordComponent::SCALAR];

			// Loop through the multifab, and store each box as a chunk in the openPMD file.
			for (amrex::MFIter mfi(*mf[lev]); mfi.isValid(); ++mfi) {
				amrex::FArrayBox const &fab = (*mf[lev])[mfi];
				amrex::Box const &local_box = fab.box();

				// Determine the offset and size of this chunk
				amrex::IntVect const box_offset = local_box.smallEnd() - global_box.smallEnd();
				auto chunk_offset = detail::getReversedVec(box_offset); // this overflows if ghost zones are used (!)
				auto chunk_size = detail::getReversedVec(local_box.size());

				// pass device pointer directly to ADIOS
				amrex::Real const *local_data = fab.dataPtr(icomp);
				mesh.storeChunkRaw(local_data, chunk_offset, chunk_size);
			}
		}

		// flush this level to disk
		series.flush();
	}

	// close file
	series.close();
}

} // namespace quokka::OpenPMDOutput
