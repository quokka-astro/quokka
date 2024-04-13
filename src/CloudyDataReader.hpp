#ifndef CLOUDYDATAREADER_HPP_ // NOLINT
#define CLOUDYDATAREADER_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file GrackleDataReader.hpp
/// \brief Defines methods for reading the cooling rate tables used by Grackle.
///

#include <cstdint>
#include <limits>
#include <string>

#include <H5Dpublic.h>
#include <H5Ppublic.h>
#include <hdf5.h>

#include "AMReX_GpuContainers.H"
#include "AMReX_TableData.H"

/* HDF5 definitions */
#define HDF5_FILE_I4 H5T_STD_I32BE
#define HDF5_FILE_I8 H5T_STD_I64BE
#define HDF5_FILE_R4 H5T_IEEE_F32BE
#define HDF5_FILE_R8 H5T_IEEE_F64BE
#define HDF5_FILE_B8 H5T_STD_B8BE

#define HDF5_I4 H5T_NATIVE_INT
#define HDF5_I8 H5T_NATIVE_LLONG
#define HDF5_R4 H5T_NATIVE_FLOAT
#define HDF5_R8 H5T_NATIVE_DOUBLE
#define HDF5_R16 H5T_NATIVE_LDOUBLE

namespace quokka::TabulatedCooling
{

constexpr double SMALL_LOG_VALUE = -99.0;
constexpr int CLOUDY_MAX_DIMENSION = 2; // we are using amrex::Table2D

// Cooling table storage

using cloudy_cooling_tools_data = struct cloudy_cooling_tools_data {
	// Rank of dataset.
	int64_t grid_rank = 0;

	// Dimension of dataset.
	std::vector<int64_t> grid_dimension;

	// Dataset parameter values.
	std::vector<amrex::Table1D<double>> grid_parameters;
	std::vector<amrex::Gpu::PinnedVector<double>> grid_parametersVec;

	// Heating values
	amrex::Table2D<double> heating_data;
	// Backing allocation in pinned memory
	amrex::Gpu::PinnedVector<double> heating_dataVec;

	// Cooling values
	amrex::Table2D<double> cooling_data;
	// Backing allocation in pinned memory
	amrex::Gpu::PinnedVector<double> cooling_dataVec;

	// Mean Molecular Weight values
	amrex::Table2D<double> mmw_data;
	// Backing allocation in pinned memory
	amrex::Gpu::PinnedVector<double> mmw_dataVec;

	// Length of 1D flattened data
	int64_t data_size = 0;

	// temperature range
	amrex::Real T_min{std::numeric_limits<amrex::Real>::max()};
	amrex::Real T_max{std::numeric_limits<amrex::Real>::min()};

	// mean molecular weight range
	amrex::Real mmw_min{std::numeric_limits<amrex::Real>::max()};
	amrex::Real mmw_max{std::numeric_limits<amrex::Real>::min()};
};

using code_units = struct code_units {
	double density_units = 1;
	double length_units = 1;
	double time_units = 1;
	double velocity_units = 1;
};

void initialize_cloudy_data(cloudy_cooling_tools_data &my_cloudy, std::string const &grackle_data_file, code_units const &my_units);

auto extract_2d_table(amrex::Table2D<double> const &table2D) -> amrex::TableData<double, 2>;

auto copy_1d_table(amrex::Table1D<double> const &table1D) -> amrex::TableData<double, 1>;

} // namespace quokka::TabulatedCooling

#endif // CLOUDYDATAREADER_HPP_
