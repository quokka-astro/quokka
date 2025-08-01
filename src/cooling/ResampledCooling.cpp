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

#include <H5Dpublic.h>
#include <H5Ppublic.h>
#include <hdf5.h>

#include "AMReX_BLassert.H"
#include "AMReX_Print.H"

namespace quokka::ResampledCooling
{

void readResampledData(std::string const &hdf5_file, resampled_tables &resampledTables)
{
	amrex::Print() << "Initializing resampled cooling.\n";
	amrex::Print() << fmt::format("resampled_table_file: {}.\n", hdf5_file);

	// Read cooling data from HDF5 file
	hid_t file_id = 0;
	hid_t attr_id = 0;
	herr_t status = 0;
	herr_t const h5_error = -1;

	file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file_id != h5_error, "Failed to open resampled cooling data file!");

	// Read remaining metadata (hydrogen mass fraction only)
	hid_t const metadata_group = H5Gopen2(file_id, "/metadata", H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(metadata_group != h5_error, "Failed to open metadata group!");

	attr_id = H5Aopen(metadata_group, "cloudy_H_mass_fraction", H5P_DEFAULT);
	status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &resampledTables.cloudy_H_mass_fraction);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read cloudy_H_mass_fraction!");
	H5Aclose(attr_id);

	H5Gclose(metadata_group);

	// Define coordinate names and fast_log setting
	const std::vector<std::string> coord_names = {"rho", "eint"};
	const int is_fast_log = 1;

	// Coordinate bounds will be read by H5Reader
	std::array<std::pair<amrex::Real, amrex::Real>, 2> coord_bounds;

	// Read all 2D datasets using generic DataTable H5Reader (includes coordinate bounds)
	resampledTables.cooling_rates = quokka::DataTable<2, 1>::H5Reader(file_id, "/data/cooling_rates", coord_names, is_fast_log, &coord_bounds);
	resampledTables.temperatures = quokka::DataTable<2, 1>::H5Reader(file_id, "/data/temperatures", coord_names, is_fast_log);
	resampledTables.sound_speeds = quokka::DataTable<2, 1>::H5Reader(file_id, "/data/sound_speeds", coord_names, is_fast_log);
	resampledTables.pressures = quokka::DataTable<2, 1>::H5Reader(file_id, "/data/pressures", coord_names, is_fast_log);
	resampledTables.entropies = quokka::DataTable<2, 1>::H5Reader(file_id, "/data/entropies", coord_names, is_fast_log);

	// Set coordinate bounds from H5Reader output
	resampledTables.rho_min = coord_bounds[0].first;
	resampledTables.rho_max = coord_bounds[0].second;
	resampledTables.eint_min = coord_bounds[1].first;
	resampledTables.eint_max = coord_bounds[1].second;

	H5Fclose(file_id);

	// Get grid dimensions from the DataTable objects for logging
	const int n_rho = resampledTables.cooling_rates.size(0);
	const int n_eint = resampledTables.cooling_rates.size(1);

	amrex::Print() << fmt::format("\tDensity range: {} to {} g/cm^3 ({} steps).\n", resampledTables.rho_min, resampledTables.rho_max, n_rho);
	amrex::Print() << fmt::format("\tSpecific energy range: {} to {} erg/g ({} steps).\n", resampledTables.eint_min, resampledTables.eint_max, n_eint);
}

auto resampled_tables::const_tables() const -> resampledGpuConstTables
{
	resampledGpuConstTables tables{cooling_rates.const_tables(),
				       temperatures.const_tables(),
				       sound_speeds.const_tables(),
				       pressures.const_tables(),
				       entropies.const_tables(),
				       rho_min,
				       rho_max,
				       eint_min,
				       eint_max,
				       cloudy_H_mass_fraction};
	return tables;
}

} // namespace quokka::ResampledCooling