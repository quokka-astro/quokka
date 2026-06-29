// ABOUTME: Implementation for resampled cooling tables that interpolate on (rho, e_int) grid
// ABOUTME: Reads HDF5-format tables produced by extern/cooling/resample_cooling_tables.py
//==============================================================================
//  TwoMomentRad - a radiation transport library for patch-based AMR codes
//  Copyright 2020 Benjamin Wibking.
//  Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file ResampledCooling.cpp
/// \brief Implements methods for interpolating cooling rates from resampled
/// tables.
///

#include "cooling/ResampledCooling.hpp"

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"
#include <hdf5.h>
#include <format>

namespace quokka::ResampledCooling
{

constexpr double cloudy_H_mass_fraction = 1. / (1. + 0.1 * 3.971);

// return: is_include_pe
auto readResampledData(std::string const &hdf5_file, resampled_tables &resampledTables) -> bool
{
	amrex::Print() << "Initializing resampled cooling.\n";
	amrex::Print() << std::format("resampled_table_file: {}.\n", hdf5_file);

	// Check if file exists
	if (!amrex::FileSystem::Exists(hdf5_file)) {
		amrex::Abort("Resampled cooling table file does not exist!");
	}

	// Read the combined table (Nout=5: cooling_rate, temperature, sound_speed, pressure, entropy)
	resampledTables.table = quokka::DataTable<2, 5>::H5Reader(hdf5_file, "/data");

	// Read physical eint bounds (xlo[1], xhi[1]) and file-level include_pe flag
	amrex::Real eint_min = 0.0;
	amrex::Real eint_max = 0.0;
	int include_pe_val = 0;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		const herr_t h5_error = -1;
		hid_t const file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file_id != h5_error, ("Failed to open HDF5 file: " + hdf5_file).c_str());

		// Read xlo/xhi from /data group to get eint bounds (index 1)
		hid_t const group_id = H5Gopen2(file_id, "/data", H5P_DEFAULT);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(group_id != h5_error, "Failed to open /data group!");
		double xlo[2] = {0.0, 0.0};
		double xhi[2] = {0.0, 0.0};
		hid_t attr_id = H5Aopen(group_id, "xlo", H5P_DEFAULT);
		H5Aread(attr_id, H5T_NATIVE_DOUBLE, xlo);
		H5Aclose(attr_id);
		attr_id = H5Aopen(group_id, "xhi", H5P_DEFAULT);
		H5Aread(attr_id, H5T_NATIVE_DOUBLE, xhi);
		H5Aclose(attr_id);
		H5Gclose(group_id);
		eint_min = xlo[1];
		eint_max = xhi[1];

		// Read file-level include_pe attribute
		if (H5Aexists(file_id, "include_pe") > 0) {
			attr_id = H5Aopen(file_id, "include_pe", H5P_DEFAULT);
			H5Aread(attr_id, H5T_NATIVE_INT, &include_pe_val);
			H5Aclose(attr_id);
		}
		H5Fclose(file_id);
	}

	amrex::ParallelDescriptor::Bcast(&eint_min, 1, amrex::ParallelDescriptor::IOProcessorNumber());
	amrex::ParallelDescriptor::Bcast(&eint_max, 1, amrex::ParallelDescriptor::IOProcessorNumber());
	amrex::ParallelDescriptor::Bcast(&include_pe_val, 1, amrex::ParallelDescriptor::IOProcessorNumber());

	resampledTables.eint_min = eint_min;
	resampledTables.eint_max = eint_max;
	resampledTables.cloudy_H_mass_fraction = cloudy_H_mass_fraction;

	const int n_rho = resampledTables.table.size(0);
	const int n_eint = resampledTables.table.size(1);
	amrex::Print() << std::format("\tDensity range: {} to {} g/cm^3 ({} steps).\n", resampledTables.table.xlo()[0],
				      resampledTables.table.xhi()[0], n_rho);
	amrex::Print() << std::format("\tSpecific energy range: {} to {} erg/g ({} steps).\n", eint_min, eint_max, n_eint);
	amrex::Print() << std::format("\tPhotoelectric heating: {}.\n", (include_pe_val != 0) ? "enabled" : "disabled");

	return (include_pe_val != 0);
}

auto resampled_tables::const_tables() const -> resampledGpuConstTables
{
	return resampledGpuConstTables{table.const_tables(), eint_min, eint_max, cloudy_H_mass_fraction};
}

} // namespace quokka::ResampledCooling
