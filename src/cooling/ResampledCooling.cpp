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

#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_Print.H"
#include "AMReX_TableData.H"

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

	H5Gclose(metadata_group);

	// Read grid data
	{
		auto *temp_data = new double[n_rho]; // NOLINT(cppcoreguidelines-owning-memory)
		dset_id = H5Dopen2(file_id, "/grids/fast_log_rho", H5P_DEFAULT);
		status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read fast_log_rho dataset!");
		H5Dclose(dset_id);

		resampledTables.fast_log_rho =
		    std::make_unique<amrex::TableData<double, 1>>(amrex::Array<int, 1>{0}, amrex::Array<int, 1>{n_rho - 1}, amrex::The_Pinned_Arena());
		auto rho_table = resampledTables.fast_log_rho->table();
		for (int i = 0; i < n_rho; ++i) {
			rho_table(i) = temp_data[i];
		}
		delete[] temp_data; // NOLINT(cppcoreguidelines-owning-memory)
	}

	{
		auto *temp_data = new double[n_eint]; // NOLINT(cppcoreguidelines-owning-memory)
		dset_id = H5Dopen2(file_id, "/grids/fast_log_eint", H5P_DEFAULT);
		status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read fast_log_eint dataset!");
		H5Dclose(dset_id);

		resampledTables.fast_log_eint =
		    std::make_unique<amrex::TableData<double, 1>>(amrex::Array<int, 1>{0}, amrex::Array<int, 1>{n_eint - 1}, amrex::The_Pinned_Arena());
		auto eint_table = resampledTables.fast_log_eint->table();
		for (int i = 0; i < n_eint; ++i) {
			eint_table(i) = temp_data[i];
		}
		delete[] temp_data; // NOLINT(cppcoreguidelines-owning-memory)
	}

	// Read 2D data tables
	const int64_t data_size = static_cast<int64_t>(n_rho) * static_cast<int64_t>(n_eint);

	{
		auto *temp_data = new double[data_size]; // NOLINT(cppcoreguidelines-owning-memory)
		dset_id = H5Dopen2(file_id, "/data/cooling_rates", H5P_DEFAULT);
		status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read cooling_rates dataset!");
		H5Dclose(dset_id);

		resampledTables.cooling_rates = std::make_unique<amrex::TableData<double, 2>>(
		    amrex::Array<int, 2>{0, 0}, amrex::Array<int, 2>{n_rho - 1, n_eint - 1}, amrex::The_Pinned_Arena());
		auto cooling_table = resampledTables.cooling_rates->table();

		// Copy data with proper indexing (HDF5 uses C-order, AMReX tables use F-order)
		for (int i = 0; i < n_rho; ++i) {
			for (int j = 0; j < n_eint; ++j) {
				cooling_table(i, j) = temp_data[i * n_eint + j];
			}
		}
		delete[] temp_data; // NOLINT(cppcoreguidelines-owning-memory)
	}

	{
		auto *temp_data = new double[data_size]; // NOLINT(cppcoreguidelines-owning-memory)
		dset_id = H5Dopen2(file_id, "/data/temperatures", H5P_DEFAULT);
		status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read temperatures dataset!");
		H5Dclose(dset_id);

		resampledTables.temperatures = std::make_unique<amrex::TableData<double, 2>>(
		    amrex::Array<int, 2>{0, 0}, amrex::Array<int, 2>{n_rho - 1, n_eint - 1}, amrex::The_Pinned_Arena());
		auto temp_table = resampledTables.temperatures->table();

		// Copy data with proper indexing (HDF5 uses C-order, AMReX tables use F-order)
		for (int i = 0; i < n_rho; ++i) {
			for (int j = 0; j < n_eint; ++j) {
				temp_table(i, j) = temp_data[i * n_eint + j];
			}
		}
		delete[] temp_data; // NOLINT(cppcoreguidelines-owning-memory)
	}

	{
		auto *temp_data = new double[data_size]; // NOLINT(cppcoreguidelines-owning-memory)
		dset_id = H5Dopen2(file_id, "/data/sound_speeds", H5P_DEFAULT);
		status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read sound_speeds dataset!");
		H5Dclose(dset_id);

		resampledTables.sound_speeds = std::make_unique<amrex::TableData<double, 2>>(
		    amrex::Array<int, 2>{0, 0}, amrex::Array<int, 2>{n_rho - 1, n_eint - 1}, amrex::The_Pinned_Arena());
		auto sound_speed_table = resampledTables.sound_speeds->table();

		// Copy data with proper indexing (HDF5 uses C-order, AMReX tables use F-order)
		for (int i = 0; i < n_rho; ++i) {
			for (int j = 0; j < n_eint; ++j) {
				sound_speed_table(i, j) = temp_data[i * n_eint + j];
			}
		}
		delete[] temp_data; // NOLINT(cppcoreguidelines-owning-memory)
	}

	{
		auto *temp_data = new double[data_size]; // NOLINT(cppcoreguidelines-owning-memory)
		dset_id = H5Dopen2(file_id, "/data/pressures", H5P_DEFAULT);
		status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read pressures dataset!");
		H5Dclose(dset_id);

		resampledTables.pressures = std::make_unique<amrex::TableData<double, 2>>(
		    amrex::Array<int, 2>{0, 0}, amrex::Array<int, 2>{n_rho - 1, n_eint - 1}, amrex::The_Pinned_Arena());
		auto pressure_table = resampledTables.pressures->table();

		// Copy data with proper indexing (HDF5 uses C-order, AMReX tables use F-order)
		for (int i = 0; i < n_rho; ++i) {
			for (int j = 0; j < n_eint; ++j) {
				pressure_table(i, j) = temp_data[i * n_eint + j];
			}
		}
		delete[] temp_data; // NOLINT(cppcoreguidelines-owning-memory)
	}

	{
		auto *temp_data = new double[data_size]; // NOLINT(cppcoreguidelines-owning-memory)
		dset_id = H5Dopen2(file_id, "/data/entropies", H5P_DEFAULT);
		status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read entropies dataset!");
		H5Dclose(dset_id);

		resampledTables.entropies = std::make_unique<amrex::TableData<double, 2>>(
		    amrex::Array<int, 2>{0, 0}, amrex::Array<int, 2>{n_rho - 1, n_eint - 1}, amrex::The_Pinned_Arena());
		auto entropy_table = resampledTables.entropies->table();

		// Copy data with proper indexing (HDF5 uses C-order, AMReX tables use F-order)
		for (int i = 0; i < n_rho; ++i) {
			for (int j = 0; j < n_eint; ++j) {
				entropy_table(i, j) = temp_data[i * n_eint + j];
			}
		}
		delete[] temp_data; // NOLINT(cppcoreguidelines-owning-memory)
	}

	H5Fclose(file_id);

	amrex::Print() << fmt::format("\tDensity range: {} to {} g/cm^3 ({} steps).\n", resampledTables.rho_min, resampledTables.rho_max, n_rho);
	amrex::Print() << fmt::format("\tSpecific energy range: {} to {} erg/g ({} steps).\n", resampledTables.eint_min, resampledTables.eint_max, n_eint);
}

auto resampled_tables::const_tables() const -> resampledGpuConstTables
{
	resampledGpuConstTables tables{fast_log_rho->const_table(),
				       fast_log_eint->const_table(),
				       cooling_rates->const_table(),
				       temperatures->const_table(),
				       sound_speeds->const_table(),
				       pressures->const_table(),
				       entropies->const_table(),
				       rho_min,
				       rho_max,
				       eint_min,
				       eint_max};
	return tables;
}

} // namespace quokka::ResampledCooling