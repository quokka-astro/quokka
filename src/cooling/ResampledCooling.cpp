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
	hid_t dset_id = 0;
	hid_t attr_id = 0;
	herr_t status = 0;
	herr_t const h5_error = -1;

	file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file_id != h5_error, "Failed to open resampled cooling data file!");

	// Read metadata
	hid_t const metadata_group = H5Gopen2(file_id, "/metadata", H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(metadata_group != h5_error, "Failed to open metadata group!");

	// Read grid dimensions
	int n_rho = 0;
	int n_eint = 0;
	attr_id = H5Aopen(metadata_group, "n_rho", H5P_DEFAULT);
	status = H5Aread(attr_id, H5T_NATIVE_INT, &n_rho);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read n_rho!");
	H5Aclose(attr_id);

	attr_id = H5Aopen(metadata_group, "n_eint", H5P_DEFAULT);
	status = H5Aread(attr_id, H5T_NATIVE_INT, &n_eint);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read n_eint!");
	H5Aclose(attr_id);

	// Read bounds
	attr_id = H5Aopen(metadata_group, "rho_min", H5P_DEFAULT);
	status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &resampledTables.rho_min);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read rho_min!");
	H5Aclose(attr_id);

	attr_id = H5Aopen(metadata_group, "rho_max", H5P_DEFAULT);
	status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &resampledTables.rho_max);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read rho_max!");
	H5Aclose(attr_id);

	attr_id = H5Aopen(metadata_group, "eint_min", H5P_DEFAULT);
	status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &resampledTables.eint_min);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read eint_min!");
	H5Aclose(attr_id);

	attr_id = H5Aopen(metadata_group, "eint_max", H5P_DEFAULT);
	status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &resampledTables.eint_max);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read eint_max!");
	H5Aclose(attr_id);

	attr_id = H5Aopen(metadata_group, "cloudy_H_mass_fraction", H5P_DEFAULT);
	status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &resampledTables.cloudy_H_mass_fraction);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read cloudy_H_mass_fraction!");
	H5Aclose(attr_id);

	H5Gclose(metadata_group);

	// Read coordinate grids
	amrex::Vector<amrex::Real> rho_coords(n_rho);
	amrex::Vector<amrex::Real> eint_coords(n_eint);

	{
		auto *temp_data = new double[n_rho]; // NOLINT(cppcoreguidelines-owning-memory)
		dset_id = H5Dopen2(file_id, "/grids/fast_log_rho", H5P_DEFAULT);
		status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read fast_log_rho dataset!");
		H5Dclose(dset_id);

		for (int i = 0; i < n_rho; ++i) {
			rho_coords[i] = temp_data[i];
		}
		delete[] temp_data; // NOLINT(cppcoreguidelines-owning-memory)
	}

	{
		auto *temp_data = new double[n_eint]; // NOLINT(cppcoreguidelines-owning-memory)
		dset_id = H5Dopen2(file_id, "/grids/fast_log_eint", H5P_DEFAULT);
		status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read fast_log_eint dataset!");
		H5Dclose(dset_id);

		for (int i = 0; i < n_eint; ++i) {
			eint_coords[i] = temp_data[i];
		}
		delete[] temp_data; // NOLINT(cppcoreguidelines-owning-memory)
	}

	// Prepare coordinate arrays for DataTable initialization
	const std::array<amrex::Vector<amrex::Real>, 2> coord_arrays = {rho_coords, eint_coords};

	// Read all 2D datasets using DataTable H5Reader
	resampledTables.cooling_rates = quokka::DataTable<2, 1>::H5Reader(file_id, "/data/cooling_rates", coord_arrays);
	resampledTables.temperatures = quokka::DataTable<2, 1>::H5Reader(file_id, "/data/temperatures", coord_arrays);
	resampledTables.sound_speeds = quokka::DataTable<2, 1>::H5Reader(file_id, "/data/sound_speeds", coord_arrays);
	resampledTables.pressures = quokka::DataTable<2, 1>::H5Reader(file_id, "/data/pressures", coord_arrays);
	resampledTables.entropies = quokka::DataTable<2, 1>::H5Reader(file_id, "/data/entropies", coord_arrays);

	H5Fclose(file_id);

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