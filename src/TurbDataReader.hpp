#ifndef TURBDATAREADER_HPP_ // NOLINT
#define TURBDATAREADER_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file TurbDataReader.hpp
/// \brief Reads turbulent driving fields generated as cubic HDF5 arrays.
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

using turb_data = struct turb_data {
	// values
	amrex::Table3D<double> dvx;
	amrex::Table3D<double> dvy;
	amrex::Table3D<double> dvz;
};

void initialize_turbdata(turb_data &data, std::string &data_file);

auto read_dataset(hid_t &file_id, char const *dataset_name) -> amrex::Table3D<double>;

auto get_tabledata(amrex::Table3D<double> &in_t) -> amrex::TableData<double, 3>;

auto computeRms(amrex::TableData<amrex::Real, 3> &dvx, amrex::TableData<amrex::Real, 3> &dvy, amrex::TableData<amrex::Real, 3> &dvz) -> amrex::Real;

#endif // TURBDATAREADER_HPP_
