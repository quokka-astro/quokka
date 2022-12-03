#ifndef GRACKLEDATAREADER_HPP_ // NOLINT
#define GRACKLEDATAREADER_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file GrackleDataReader.hpp
/// \brief Defines methods for reading the cooling rate tables used by Grackle.
///

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

#include "fmt/core.h"
#include <H5Dpublic.h>
#include <H5Ppublic.h>
#include <hdf5.h>

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_TableData.H"

using Real = amrex::Real;

#define SMALL_LOG_VALUE -99.0
#define CLOUDY_MAX_DIMENSION 3 // we are using amrex::Table3D

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

// Cooling table storage

using cloudy_data = struct cloudy_data {
	// Rank of dataset.
	int64_t grid_rank = 0;

	// Dimension of dataset.
	std::vector<int64_t> grid_dimension;

	// Dataset parameter values.
	std::vector<amrex::Table1D<double>> grid_parameters;

	// Heating values
	amrex::Table3D<double> heating_data;

	// Cooling values
	amrex::Table3D<double> cooling_data;

	// Mean Molecular Weight values
	amrex::Table3D<double> mmw_data;

	// Length of 1D flattened data
	int64_t data_size = 0;
};

using code_units = struct code_units {
	double density_units = 1;
	double length_units = 1;
	double time_units = 1;
	double velocity_units = 1;
};

void initialize_cloudy_data(cloudy_data &my_cloudy, char const *group_name, std::string &grackle_data_file, code_units &my_units);

auto extract_2d_table(amrex::Table3D<double> const &table3D, int redshift_index) -> amrex::TableData<double, 2>;

auto copy_1d_table(amrex::Table1D<double> const &table1D) -> amrex::TableData<double, 1>;

#endif // GRACKLEDATAREADER_HPP_
