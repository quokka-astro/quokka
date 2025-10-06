#ifndef DATATABLE_HPP_
#define DATATABLE_HPP_

#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_TableData.H"

// HDF5 includes for H5Reader functionality
#include <H5Dpublic.h>
#include <H5Ppublic.h>
#include <hdf5.h>

#include "math/FastMath.hpp"
#include <array>
#include <fstream>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

// For descriptive error messages
#include <fmt/format.h>

namespace quokka
{

// Structure to hold interpolation indices and normalized coordinates
template <int Ndim> struct InterpData {
	std::array<int, Ndim> indices{};	    // grid indices for each dimension (lower bounds)
	std::array<amrex::Real, Ndim> normalized{}; // normalized coordinates in [0,1] for each dimension

	// Default constructor
	AMREX_GPU_HOST_DEVICE InterpData() = default;
};

// GPU-friendly struct containing const table references
template <int Ndim, int Nout = 1> struct DataTableGpuConst {
	std::array<amrex::Table1D<const amrex::Real>, Ndim> coords;
	// Array of data tables for multiple outputs - each has the same coordinate dimensionality
	using single_data_table_type =
	    std::conditional_t<Ndim == 1, amrex::Table1D<const amrex::Real>,
			       std::conditional_t<Ndim == 2, amrex::Table2D<const amrex::Real>,
						  std::conditional_t<Ndim == 3, amrex::Table3D<const amrex::Real>, amrex::Table4D<const amrex::Real>>>>;
	std::array<single_data_table_type, Nout> dataViewArrays;

	std::array<amrex::Real, Ndim> coord_min{};
	std::array<amrex::Real, Ndim> coord_max{};
	std::array<std::string, Ndim> spacing_types{};

	// Precomputed grid spacing for optimization
	std::array<amrex::Real, Ndim> dcoord{};

	std::array<int, Ndim> sizes{};
	
	// Output spacing for return values: "linear", "log", or "fast_log"
	std::string output_spacing = "linear";

	/// @brief Find interpolation indices and normalized coordinates for n-dimensional interpolation
	///
	/// This function locates the hypercube containing the given point and computes normalized
	/// coordinates within that hypercube for efficient n-linear interpolation.
	///
	/// @param point Physical coordinates to interpolate at (size Ndim)
	/// @return InterpData structure containing grid indices, coordinates, and normalized params
	///
	/// Grid Layout and Coordinate Mapping, for 2D as an example:
	/// ```
	///   y2  z3 -------- z4     (x1,y2) -------- (x2,y2)
	///       |     *     |         |     *     |
	///       | (h,v)     |         | (h,v)     |
	///       |           |         |           |
	///   y1  z1 -------- z2     (x1,y1) -------- (x2,y1)
	///      x1          x2
	///
	///   where: h = (x - x1)/(x2 - x1), v = (y - y1)/(y2 - y1)
	///
	///   Normalized coordinate mapping:
	///   - z1 = f(0,0) -> data(ix, iy)   = (x1,y1) bottom-left
	///   - z2 = f(1,0) -> data(iix, iy)  = (x2,y1) bottom-right
	///   - z3 = f(0,1) -> data(ix, iiy)  = (x1,y2) top-left
	///   - z4 = f(1,1) -> data(iix, iiy) = (x2,y2) top-right
	/// ```
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto find_interpolation_data(const std::array<amrex::Real, Ndim> &point) const
	    -> InterpData<Ndim>
	{
		InterpData<Ndim> interp;

		for (int dim = 0; dim < Ndim; ++dim) {
			// Get table bounds - use precomputed coord_min/coord_max instead of accessing coords table
			amrex::Real const coord_start = coord_min[dim];
			amrex::Real const coord_end = coord_max[dim];

			// Clamp coordinates to valid table bounds (extrapolation not supported)
			amrex::Real clamped_coord = amrex::max(coord_start, amrex::min(point[dim], coord_end));

			// Find grid cell indices containing the point
			// indices are the "lower" indices of the containing hypercube
			interp.indices[dim] =
			    amrex::max(0, amrex::min(static_cast<int>(std::floor((clamped_coord - coord_start) / dcoord[dim])), sizes[dim] - 1));

			// if indices is end - 1, then set indices to end - 2 (so that upper_indices is end - 1, the last index)
			if (interp.indices[dim] == sizes[dim] - 1) {
				interp.indices[dim] = sizes[dim] - 2;
			}

			// Compute normalized coordinate
			// For linear spacing: coord = coord_min + index * dcoord (no table lookup needed!)
			amrex::Real const coord_at_index = coord_min[dim] + static_cast<amrex::Real>(interp.indices[dim]) * dcoord[dim];
			interp.normalized[dim] = (clamped_coord - coord_at_index) / dcoord[dim];
		}

		return interp;
	}

	/// @brief Perform n-dimensional linear interpolation for multiple outputs
	///
	/// This method performs n-linear interpolation by recursively interpolating
	/// along each dimension. For 2D this becomes bilinear, for 3D trilinear, etc.
	/// Returns all output values sharing the same coordinate interpolation.
	///
	/// @param point Physical coordinates to interpolate at (size Ndim)
	/// @return Array of interpolated values (size Nout)
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate(const std::array<amrex::Real, Ndim> &point) const
	    -> std::array<amrex::Real, Nout>
	{
		// Take log or fast_log if the spacing types are log or fast_log
		std::array<amrex::Real, Ndim> point_{};
		for (int dim = 0; dim < Ndim; ++dim) {
			if (spacing_types[dim] == "linear") {
				point_[dim] = point[dim];
			} else if (spacing_types[dim] == "log") {
				point_[dim] = std::log10(point[dim]);
			} else if (spacing_types[dim] == "fast_log") {
				point_[dim] = FastMath::log10(point[dim]);
			}
		}

		// Part 1: Find interpolation indices and normalized coordinates (shared for all outputs)
		InterpData<Ndim> const interp = find_interpolation_data(point_);

		// Part 2: Perform n-dimensional interpolation for all outputs
		auto values = interpolate_from_indices(interp);
		
		// Part 3: Convert from log space if output values are stored in log10
		if (output_spacing == "fast_log") {
			for (int i = 0; i < Nout; ++i) {
				values[i] = FastMath::pow10(values[i]);
			}
		} else if (output_spacing == "log") {
			for (int i = 0; i < Nout; ++i) {
				values[i] = std::pow(10.0, values[i]);
			}
		}
		
		return values;
	}

	/// @brief Perform n-dimensional linear interpolation for a single output (backward compatibility)
	///
	/// @param point Physical coordinates to interpolate at (size Ndim)
	/// @param output_index Index of the output to interpolate (0 to Nout-1)
	/// @return Single interpolated value
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate_single(const std::array<amrex::Real, Ndim> &point, int output_index = 0) const
	    -> amrex::Real
	{
		// Take log or fast_log if the spacing types are log or fast_log
		std::array<amrex::Real, Ndim> point_{};
		for (int dim = 0; dim < Ndim; ++dim) {
			if (spacing_types[dim] == "linear") {
				point_[dim] = point[dim];
			} else if (spacing_types[dim] == "log") {
				point_[dim] = std::log10(point[dim]);
			} else if (spacing_types[dim] == "fast_log") {
				point_[dim] = FastMath::log10(point[dim]);
			}
		}

		// Part 1: Find interpolation indices and normalized coordinates
		InterpData<Ndim> const interp = find_interpolation_data(point_);

		// Part 2: Perform n-dimensional interpolation for single output
		amrex::Real value = interpolate_single_from_indices(interp, output_index);
		
		// Part 3: Convert from log space if output values are stored in log10
		if (output_spacing == "fast_log") {
			value = FastMath::pow10(value);
		} else if (output_spacing == "log") {
			value = std::pow(10.0, value);
		}
		
		return value;
	}

      private:
	/// @brief Helper for n-dimensional interpolation (multiple outputs)
	///
	/// This function performs n-linear interpolation for 1D-4D cases for all outputs.
	/// Supports linear, bilinear, trilinear, and quadrilinear interpolation.
	///
	/// @param interp Interpolation data containing indices and normalized coordinates
	/// @return Array of interpolated values (size Nout)
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate_from_indices(const InterpData<Ndim> &interp) const
	    -> std::array<amrex::Real, Nout>
	{
		std::array<amrex::Real, Nout> results{};

		// Interpolate all outputs using the same coordinate weights
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			results[out_idx] = interpolate_single_from_indices(interp, out_idx);
		}

		return results;
	}

	/// @brief Helper for n-dimensional interpolation (single output)
	///
	/// This function performs n-linear interpolation for 1D-4D cases for a single output.
	/// Supports linear, bilinear, trilinear, and quadrilinear interpolation.
	///
	/// @param interp Interpolation data containing indices and normalized coordinates
	/// @param output_index Index of the output to interpolate (0 to Nout-1)
	/// @return Single interpolated value
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate_single_from_indices(const InterpData<Ndim> &interp, int output_index) const
	    -> amrex::Real
	{
		auto dataView_ = dataViewArrays[output_index];

		if constexpr (Ndim == 1) {
			// 1D case (linear interpolation)
			const int ix = interp.indices[0];

			const std::array<amrex::Real, 2> w = {1.0 - interp.normalized[0], interp.normalized[0]};

			const amrex::Real value = w[0] * dataView_(ix) + w[1] * dataView_(ix + 1);

			AMREX_ASSERT(!std::isnan(value));
			return value;
		} else if constexpr (Ndim == 2) {
			// 2D case (bilinear interpolation)
			const int ix1 = interp.indices[0];
			const int ix2 = interp.indices[1];

			const std::array<amrex::Real, 2> w1 = {1.0 - interp.normalized[0], interp.normalized[0]};
			const std::array<amrex::Real, 2> w2 = {1.0 - interp.normalized[1], interp.normalized[1]};

			// Spiner formula (https://github.com/lanl/spiner/blob/main/spiner/databox.hpp, line 461):
			// const amrex::Real value = (w2[0] * (w1[0] * dataView_(ix2, ix1) + w1[1] * dataView_(ix2, ix1 + 1)) +
			// 			   w2[1] * (w1[0] * dataView_(ix2 + 1, ix1) + w1[1] * dataView_(ix2 + 1, ix1 + 1)));
			// I inverted the indices because Spiner uses (ix2, ix1) indexing, but we use (ix1, ix2) indexing
			const amrex::Real value = (w2[0] * (w1[0] * dataView_(ix1, ix2) + w1[1] * dataView_(ix1 + 1, ix2)) +
						   w2[1] * (w1[0] * dataView_(ix1, ix2 + 1) + w1[1] * dataView_(ix1 + 1, ix2 + 1)));

			AMREX_ASSERT(!std::isnan(value));
			return value;
		} else if constexpr (Ndim == 3) {
			// 3D case (trilinear interpolation)
			const auto ix = interp.indices;

			const std::array<std::array<amrex::Real, 2>, 3> w = {{{1.0 - interp.normalized[0], interp.normalized[0]},
									      {1.0 - interp.normalized[1], interp.normalized[1]},
									      {1.0 - interp.normalized[2], interp.normalized[2]}}};

			// Spiner formula (https://github.com/lanl/spiner/blob/main/spiner/databox.hpp):
			// I inverted the indices because Spiner uses (ix2, ix1) indexing, but we use (ix1, ix2) indexing
			// clang-format off
			const amrex::Real value = (
				w[2][0] * (w[1][0] * (w[0][0] * dataView_(ix[0], ix[1], ix[2]) +
															w[0][1] * dataView_(ix[0] + 1, ix[1], ix[2])) +
									 w[1][1] * (w[0][0] * dataView_(ix[0], ix[1] + 1, ix[2]) +
															w[0][1] * dataView_(ix[0] + 1, ix[1] + 1, ix[2]))) +
				w[2][1] *
						(w[1][0] * (w[0][0] * dataView_(ix[0], ix[1], ix[2] + 1) +
												w[0][1] * dataView_(ix[0] + 1, ix[1], ix[2] + 1)) +
						 w[1][1] * (w[0][0] * dataView_(ix[0], ix[1] + 1, ix[2] + 1) +
												w[0][1] * dataView_(ix[0] + 1, ix[1] + 1, ix[2] + 1))));
			// clang-format on

			AMREX_ASSERT(!std::isnan(value));
			return value;
		} else if constexpr (Ndim == 4) {
			// 4D case (quadrilinear interpolation)
			const auto ix = interp.indices;

			const std::array<std::array<amrex::Real, 2>, 4> w = {{
			    {1.0 - interp.normalized[0], interp.normalized[0]},
			    {1.0 - interp.normalized[1], interp.normalized[1]},
			    {1.0 - interp.normalized[2], interp.normalized[2]},
			    {1.0 - interp.normalized[3], interp.normalized[3]},
			}};

			// Spiner formula (https://github.com/lanl/spiner/blob/main/spiner/databox.hpp):
			// I inverted the indices because Spiner uses (ix2, ix1) indexing, but we use (ix1, ix2) indexing
			// clang-format off
			const amrex::Real value = (
				w[3][0] *
						(w[2][0] *
								 (w[1][0] *
											(w[0][0] * dataView_(ix[0], ix[1], ix[2], ix[3]) +
											 w[0][1] * dataView_(ix[0] + 1, ix[1], ix[2], ix[3])) +
									w[1][1] *
											(w[0][0] * dataView_(ix[0], ix[1] + 1, ix[2], ix[3]) +
											 w[0][1] * dataView_(ix[0] + 1, ix[1] + 1, ix[2], ix[3]))) +
						 w[2][1] *
								 (w[1][0] *
											(w[0][0] * dataView_(ix[0], ix[1], ix[2] + 1, ix[3]) +
											 w[0][1] * dataView_(ix[0] + 1, ix[1], ix[2] + 1, ix[3])) +
									w[1][1] *
											(w[0][0] * dataView_(ix[0], ix[1] + 1, ix[2] + 1, ix[3]) +
											 w[0][1] *
													 dataView_(ix[0] + 1, ix[1] + 1, ix[2] + 1, ix[3])))) +
				w[3][1] *
						(w[2][0] *
								 (w[1][0] *
											(w[0][0] * dataView_(ix[0], ix[1], ix[2], ix[3] + 1) +
											 w[0][1] * dataView_(ix[0] + 1, ix[1], ix[2], ix[3] + 1)) +
									w[1][1] *
											(w[0][0] * dataView_(ix[0], ix[1] + 1, ix[2], ix[3] + 1) +
											 w[0][1] *
													 dataView_(ix[0] + 1, ix[1] + 1, ix[2], ix[3] + 1))) +
						 w[2][1] * (w[1][0] * (w[0][0] * dataView_(ix[0], ix[1], ix[2] + 1, ix[3] + 1) +
																	 w[0][1] * dataView_(ix[0] + 1, ix[1], ix[2] + 1, ix[3] + 1)) +
												w[1][1] * (w[0][0] * dataView_(ix[0], ix[1] + 1, ix[2] + 1, ix[3] + 1) +
																	 w[0][1] * dataView_(ix[0] + 1, ix[1] + 1, ix[2] + 1, ix[3] + 1))))
			);
			// clang-format on

			AMREX_ASSERT(!std::isnan(value));
			return value;
		}
	}
};

// Generic n-dimensional data table class with multiple outputs
template <int Ndim, int Nout = 1> class DataTable
{
      private:
	std::array<std::unique_ptr<amrex::TableData<amrex::Real, 1>>, Ndim> coords_;
	std::array<std::unique_ptr<amrex::TableData<amrex::Real, Ndim>>, Nout> data_; // Array of tables for multiple outputs

	// Type aliases for different dimensional data structures
	using data_1d_type = std::array<amrex::Vector<amrex::Real>, Nout>;
	using data_2d_type = std::array<amrex::Vector<amrex::Vector<amrex::Real>>, Nout>;
	using data_3d_type = std::array<amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>>, Nout>;
	using data_4d_type = std::array<amrex::Vector<amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>>>, Nout>;

	std::array<amrex::Real, Ndim> coord_min_{};
	std::array<amrex::Real, Ndim> coord_max_{};
	std::array<std::string, Ndim> spacing_types_{};

	// Precomputed grid spacing for optimization
	std::array<amrex::Real, Ndim> dcoord_{};

	std::array<int, Ndim> sizes_{};

	// Metadata for dimension and output names/units
	std::array<std::string, Ndim> input_names_{};
	std::array<std::string, Nout> output_names_{};
	std::array<std::string, Ndim> input_units_{};
	std::array<std::string, Nout> output_units_{};

	// Output spacing type: "linear" or "fast_log"
	std::string output_spacing_{"linear"};

      public:
	// Default constructor
	DataTable() = default;

	// Constructor with coordinate arrays and data - general n-dimensional interface
	// For multiple outputs, data is organized as data[output_index][flattened_coords][last_dim]
	DataTable(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const std::array<amrex::Vector<amrex::Vector<amrex::Real>>, Nout> &data)
	{
		initialize(coords, data);
	}

	// Backward compatibility constructor for single output (Nout = 1)
	template <int N = Nout, typename = std::enable_if_t<N == 1>>
	DataTable(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const amrex::Vector<amrex::Vector<amrex::Real>> &data)
	{
		std::array<amrex::Vector<amrex::Vector<amrex::Real>>, 1> data_array = {data};
		initialize(coords, data_array);
	}

	// Destructor
	~DataTable() = default;

	// Move constructor and assignment
	DataTable(DataTable &&) = default;
	auto operator=(DataTable &&) -> DataTable & = default;

	// Delete copy constructor and assignment (expensive operations)
	DataTable(const DataTable &) = delete;
	auto operator=(const DataTable &) -> DataTable & = delete;

	// Initializer for backward compatibility with single output (Nout = 1), using fast_log on x and linear on y
	void initialize(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const amrex::Vector<amrex::Vector<amrex::Real>> &data)
	{
		static_assert(Ndim >= 1 && Ndim <= 4, "Only 1D-4D tables are supported");
		std::array<amrex::Vector<amrex::Vector<amrex::Real>>, 1> data_array = {data};
		initialize(coords, data_array);
	}

	// Initialize from coordinate arrays - 1D interface
	template <int N = Ndim, typename = std::enable_if_t<N == 1>>
	void initialize(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const data_1d_type &data)
	{
		static_assert(Ndim == 1, "This initialize overload is for 1D tables only");

		// Validate inputs
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coords[dim].empty(), fmt::format("Coordinates for dimension {} cannot be empty!", dim));
		}

		// Validate data dimensions for each output
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    data[out_idx].size() == coords[0].size(),
			    fmt::format("1D data must match coordinate size! (expected: {}, actual: {})", coords[0].size(), data[out_idx].size()));
		}

		initialize_common(coords, data);
	}

	// Initialize from coordinate arrays - 2D interface
	template <int N = Ndim, typename = std::enable_if_t<N == 2>>
	void initialize(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const data_2d_type &data)
	{
		static_assert(Ndim == 2, "This initialize overload is for 2D tables only");

		// Validate inputs
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coords[dim].empty(), fmt::format("Coordinates for dimension {} cannot be empty!", dim));
		}

		// Validate data dimensions for each output
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!data[out_idx].empty(), fmt::format("Data for output {} cannot be empty!", out_idx));
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    data[out_idx].size() == coords[0].size(),
			    fmt::format("Data first dimension must match first coordinate size for output {}! (expected: {}, actual: {})", out_idx,
					coords[0].size(), data[out_idx].size()));
			// Verify data dimensions
			for (const auto &row : data[out_idx]) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
				    row.size() == coords[1].size(),
				    fmt::format("All data rows must match second coordinate size for output {}! (expected: {}, actual: {})", out_idx,
						coords[1].size(), row.size()));
			}
		}

		initialize_common(coords, data);
	}

	// Initialize from coordinate arrays - 3D interface
	template <int N = Ndim, typename = std::enable_if_t<N == 3>>
	void initialize(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const data_3d_type &data)
	{
		static_assert(Ndim == 3, "This initialize overload is for 3D tables only");

		// Validate inputs
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coords[dim].empty(), fmt::format("Coordinates for dimension {} cannot be empty!", dim));
		}

		// Validate data dimensions for each output
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!data[out_idx].empty(), fmt::format("Data for output {} cannot be empty!", out_idx));
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    data[out_idx].size() == coords[0].size(),
			    fmt::format("Data first dimension must match first coordinate size for output {}! (expected: {}, actual: {})", out_idx,
					coords[0].size(), data[out_idx].size()));
			for (const auto &plane : data[out_idx]) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
				    plane.size() == coords[1].size(),
				    fmt::format("Data second dimension must match second coordinate size for output {}! (expected: {}, actual: {})", out_idx,
						coords[1].size(), plane.size()));
				for (const auto &row : plane) {
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
					    row.size() == coords[2].size(),
					    fmt::format("Data third dimension must match third coordinate size for output {}! (expected: {}, actual: {})",
							out_idx, coords[2].size(), row.size()));
				}
			}
		}

		initialize_common(coords, data);
	}

	// Initialize from coordinate arrays - 4D interface
	template <int N = Ndim, typename = std::enable_if_t<N == 4>>
	void initialize(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const data_4d_type &data)
	{
		static_assert(Ndim == 4, "This initialize overload is for 4D tables only");

		// Validate inputs
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coords[dim].empty(), fmt::format("Coordinates for dimension {} cannot be empty!", dim));
		}

		// Validate data dimensions for each output
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!data[out_idx].empty(), fmt::format("Data for output {} cannot be empty!", out_idx));
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    data[out_idx].size() == coords[0].size(),
			    fmt::format("Data first dimension must match first coordinate size for output {}! (expected: {}, actual: {})", out_idx,
					coords[0].size(), data[out_idx].size()));
			for (const auto &volume : data[out_idx]) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
				    volume.size() == coords[1].size(),
				    fmt::format("Data second dimension must match second coordinate size for output {}! (expected: {}, actual: {})", out_idx,
						coords[1].size(), volume.size()));
				for (const auto &plane : volume) {
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
					    plane.size() == coords[2].size(),
					    fmt::format("Data third dimension must match third coordinate size for output {}! (expected: {}, actual: {})",
							out_idx, coords[2].size(), plane.size()));
					for (const auto &row : plane) {
						AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
						    row.size() == coords[3].size(),
						    fmt::format(
							"Data fourth dimension must match fourth coordinate size for output {}! (expected: {}, actual: {})",
							out_idx, coords[3].size(), row.size()));
					}
				}
			}
		}

		initialize_common(coords, data);
	}

	// Get GPU-friendly const tables
	[[nodiscard]] auto const_tables() const -> DataTableGpuConst<Ndim, Nout>
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_initialized(), "DataTable must be initialized before getting const tables!");

		std::array<amrex::Table1D<const amrex::Real>, Ndim> coord_tables{};
		for (int i = 0; i < Ndim; ++i) {
			coord_tables[i] = coords_[i]->const_table();
		}

		std::array<typename DataTableGpuConst<Ndim, Nout>::single_data_table_type, Nout> data_tables{};
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			data_tables[out_idx] = data_[out_idx]->const_table();
		}

		DataTableGpuConst<Ndim, Nout> tables{
		    coord_tables,
		    data_tables,	// array of data tables
		    coord_min_,		// coord_min array
		    coord_max_,		// coord_max array
		    spacing_types_,	// spacing types array
		    dcoord_,		// dcoord array
		    sizes_,		// sizes array
		    output_spacing_	// output spacing
		};
		return tables;
	}

	// Check if table is initialized
	[[nodiscard]] auto is_initialized() const -> bool
	{
		// Check all coordinate arrays
		for (int dim = 0; dim < Ndim; ++dim) {
			if (coords_[dim] == nullptr) {
				return false;
			}
		}
		// Check all data tables
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			if (data_[out_idx] == nullptr) {
				return false;
			}
		}
		return true;
	}

	// Get dimension sizes
	[[nodiscard]] auto sizes() const -> std::array<int, Ndim> { return sizes_; }

	// Get size for specific dimension
	[[nodiscard]] auto size(int dim) const -> int
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dim >= 0 && dim < Ndim,
						 fmt::format("Dimension index out of bounds! (provided: {}, valid range: [0, {}])", dim, Ndim - 1));
		return sizes_[dim];
	}

	// Get number of outputs
	[[nodiscard]] constexpr auto num_outputs() const -> int { return Nout; }

	// Get metadata accessors
	[[nodiscard]] auto input_names() const -> std::array<std::string, Ndim> { return input_names_; }
	[[nodiscard]] auto output_names() const -> std::array<std::string, Nout> { return output_names_; }
	[[nodiscard]] auto input_units() const -> std::array<std::string, Ndim> { return input_units_; }
	[[nodiscard]] auto output_units() const -> std::array<std::string, Nout> { return output_units_; }

	// Get individual metadata by index
	[[nodiscard]] auto input_name(int dim) const -> std::string
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dim >= 0 && dim < Ndim,
						 fmt::format("Dimension index out of bounds! (provided: {}, valid range: [0, {}])", dim, Ndim - 1));
		return input_names_[dim];
	}
	[[nodiscard]] auto output_name(int idx) const -> std::string
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(idx >= 0 && idx < Nout,
						 fmt::format("Output index out of bounds! (provided: {}, valid range: [0, {}])", idx, Nout - 1));
		return output_names_[idx];
	}
	[[nodiscard]] auto input_unit(int dim) const -> std::string
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dim >= 0 && dim < Ndim,
						 fmt::format("Dimension index out of bounds! (provided: {}, valid range: [0, {}])", dim, Ndim - 1));
		return input_units_[dim];
	}
	[[nodiscard]] auto output_unit(int idx) const -> std::string
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(idx >= 0 && idx < Nout,
						 fmt::format("Output index out of bounds! (provided: {}, valid range: [0, {}])", idx, Nout - 1));
		return output_units_[idx];
	}

      private:
	// Optimized initialization that takes bounds, sizes, and spacing directly
	// coords parameter is optional - if empty, coordinates will be generated based on spacing type
	template <typename DataType>
	void initialize_common(const std::array<amrex::Real, Ndim> &x_mins, const std::array<amrex::Real, Ndim> &x_maxs, const std::array<int, Ndim> &n_xs,
			       const std::array<std::string, Ndim> &spacing_types, const std::array<amrex::Vector<amrex::Real>, Ndim> &coords,
			       const DataType &data)
	{
		static_assert(Ndim >= 1 && Ndim <= 4, "Only 1D-4D tables are supported");

		// Store metadata
		coord_min_ = x_mins;
		coord_max_ = x_maxs;
		sizes_ = n_xs;
		spacing_types_ = spacing_types;

		// Validate bounds and spacing types
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coord_max_[dim] > coord_min_[dim], fmt::format("Invalid coordinate bounds for dimension {}: [{}, {}]",
													dim, coord_min_[dim], coord_max_[dim]));
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sizes_[dim] > 0, fmt::format("Invalid dimension size {} for dimension {}", sizes_[dim], dim));
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    spacing_types_[dim] == "linear" || spacing_types_[dim] == "log" || spacing_types_[dim] == "fast_log",
			    fmt::format("Invalid spacing type '{}' for dimension {}. Must be 'linear', 'log', or 'fast_log'", spacing_types_[dim], dim));
		}

		// Create coordinate tables - either from provided coords or generate them
		for (int dim = 0; dim < Ndim; ++dim) {
			coords_[dim] = std::make_unique<amrex::TableData<amrex::Real, 1>>(amrex::Array<int, 1>{0}, amrex::Array<int, 1>{sizes_[dim] - 1},
											  amrex::The_Pinned_Arena());

			// Generate coordinates based on spacing type
			if (spacing_types_[dim] == "irregular") {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(coords[dim].size()) == sizes_[dim],
								 fmt::format("Provided coordinates size mismatch for dimension {}! (expected: {}, actual: {})",
									     dim, sizes_[dim], coords[dim].size()));
				// TODO(cch): this is not used anywhere
				auto coord_table = coords_[dim]->table();
				for (int i = 0; i < sizes_[dim]; ++i) {
					coord_table(i) = coords[dim][i];
				}
			} else if (spacing_types_[dim] == "linear") {
				// Linear spacing: coordinates computed on-the-fly in GPU code, no need to populate table
				// Table structure exists but remains unpopulated for memory efficiency
			} else if (spacing_types_[dim] == "log") {
				// Logarithmic spacing: store actual values, not logarithms
				// Update coordinates to their log values in plance
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coord_min_[dim] > 0.0 && coord_max_[dim] > 0.0,
								 fmt::format("Log spacing requires positive bounds for dimension {}", dim));
				coord_min_[dim] = std::log10(coord_min_[dim]);
				coord_max_[dim] = std::log10(coord_max_[dim]);
			} else if (spacing_types_[dim] == "fast_log") {
				// Fast log spacing: store log10(value) for fast interpolation
				// Update coordinates to their log values in place
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coord_min_[dim] > 0.0 && coord_max_[dim] > 0.0,
								 fmt::format("Fast log spacing requires positive bounds for dimension {}", dim));
				coord_min_[dim] = FastMath::log10(coord_min_[dim]);
				coord_max_[dim] = FastMath::log10(coord_max_[dim]);
			}
		}

		// Calculate grid spacing (after taking necessary log of the coordinates)
		for (int dim = 0; dim < Ndim; ++dim) {
			dcoord_[dim] = (coord_max_[dim] - coord_min_[dim]) / static_cast<amrex::Real>(sizes_[dim] - 1);
		}

		// Create n-dimensional data tables for each output
		amrex::Array<int, Ndim> lo{};
		amrex::Array<int, Ndim> hi{};
		for (int dim = 0; dim < Ndim; ++dim) {
			lo[dim] = 0;
			hi[dim] = sizes_[dim] - 1;
		}

		// Create and populate data tables for each output
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			data_[out_idx] = std::make_unique<amrex::TableData<amrex::Real, Ndim>>(lo, hi, amrex::The_Pinned_Arena());
			auto data_table = data_[out_idx]->table();

			// Copy data for different dimensions
			if constexpr (Ndim == 1) {
				// Copy 1D data: data[out_idx][i] -> table(i)
				for (int i = 0; i < sizes_[0]; ++i) {
					data_table(i) = data[out_idx][i];
				}
			} else if constexpr (Ndim == 2) {
				// Copy 2D data: data[out_idx][i][j] -> table(i,j)
				for (int i = 0; i < sizes_[0]; ++i) {
					for (int j = 0; j < sizes_[1]; ++j) {
						data_table(i, j) = data[out_idx][i][j];
					}
				}
			} else if constexpr (Ndim == 3) {
				// Copy 3D data: data[out_idx][i][j][k] -> table(i,j,k)
				for (int i = 0; i < sizes_[0]; ++i) {
					for (int j = 0; j < sizes_[1]; ++j) {
						for (int k = 0; k < sizes_[2]; ++k) {
							data_table(i, j, k) = data[out_idx][i][j][k];
						}
					}
				}
			} else if constexpr (Ndim == 4) {
				// Copy 4D data: data[out_idx][i][j][k][l] -> table(i,j,k,l)
				for (int i = 0; i < sizes_[0]; ++i) {
					for (int j = 0; j < sizes_[1]; ++j) {
						for (int k = 0; k < sizes_[2]; ++k) {
							for (int l = 0; l < sizes_[3]; ++l) {
								data_table(i, j, k, l) = data[out_idx][i][j][k][l];
							}
						}
					}
				}
			}
		}
	}

	// Backward compatibility wrapper: derive bounds, sizes, and spacing from coords
	template <typename DataType> void initialize_common(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const DataType &data)
	{
		// Validate inputs
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coords[dim].empty(), fmt::format("Coordinates for dimension {} cannot be empty!", dim));
		}

		// Derive bounds and sizes from coordinates
		std::array<amrex::Real, Ndim> x_mins{};
		std::array<amrex::Real, Ndim> x_maxs{};
		std::array<int, Ndim> n_xs{};
		std::array<std::string, Ndim> spacing_types{};

		for (int dim = 0; dim < Ndim; ++dim) {
			n_xs[dim] = static_cast<int>(coords[dim].size());
			x_mins[dim] = coords[dim].front();
			x_maxs[dim] = coords[dim].back();
			// This is a hack for backward compatibility to the cooling table, which uses log input and linear output
			spacing_types[dim] = "linear";
		}

		// Pass empty coords - for linear spacing they will be computed on-the-fly
		std::array<amrex::Vector<amrex::Real>, Ndim> empty_coords{};
		for (int dim = 0; dim < Ndim; ++dim) {
			empty_coords[dim] = amrex::Vector<amrex::Real>();
		}

		// Call the optimized initialize_common with empty coords for efficiency
		initialize_common(x_mins, x_maxs, n_xs, spacing_types, empty_coords, data);
	}

      public:
	// CSVReader: Generic static method to read n-dimensional data from CSV file and create DataTable
	// CSV format:
	//   Line 1: Ndim (number of input dimensions)
	//   Line 2: Nx (comma-separated sizes for each dimension)
	//   Line 3: Nout (number of outputs)
	//   Line 4: input_names (comma-separated names for each input dimension)
	//   Line 5: output_names (comma-separated names for each output)
	//   Line 6: input_units (comma-separated units for each input dimension)
	//   Line 7: output_units (comma-separated units for each output)
	//   Line 8: xlo (comma-separated lower bounds for each dimension)
	//   Line 9: xhi (comma-separated upper bounds for each dimension)
	//   Line 10: spacing (comma-separated spacing types: linear, log, fast_log)
	//   Remaining lines: data values
	//     For 2D: nx2 rows × nx1 columns (last dimension varies fastest in rows)
	//     For 3D: (nx3 × nx2) rows × nx1 columns
	//     For 4D: (nx4 × nx3 × nx2) rows × nx1 columns
	//
	// @param file_path Path to the CSV file
	// @param output_spacing Spacing type for output values: "linear" or "fast_log" or "log"
	//                      If "fast_log", output values are converted to log10 before storage
	static auto CSVReader(const std::string &file_path, const std::string &output_spacing) -> DataTable
	{
		static_assert(Ndim >= 1 && Ndim <= 4, "CSVReader supports 1D-4D tables");

		// Validate output_spacing parameter
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(output_spacing == "linear" || output_spacing == "fast_log" || output_spacing == "log",
						 fmt::format("Invalid output_spacing '{}'. Must be 'linear' or 'fast_log' or 'log'", output_spacing));

		std::ifstream file(file_path);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file.is_open(), ("Failed to open CSV file: " + file_path).c_str());

		// Read header information
		int n_dim = 0;
		int n_out = 0;
		std::array<int, Ndim> sizes{};
		std::array<std::pair<amrex::Real, amrex::Real>, Ndim> coord_bounds{};
		std::array<std::string, Ndim> spacing_types{};

		// Line 1: Ndim
		file >> n_dim;
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
		    n_dim == Ndim, fmt::format("CSV file dimension mismatch! File has {} dimensions, but DataTable is {}-dimensional", n_dim, Ndim));

		// Line 2: Nx (comma-separated)
		std::string nx_line;
		std::getline(file >> std::ws, nx_line);
		{
			std::stringstream ss(nx_line);
			for (int dim = 0; dim < Ndim; ++dim) {
				char comma = ' ';
				ss >> sizes[dim];
				if (dim < Ndim - 1) {
					ss >> comma;
				}
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sizes[dim] > 0, fmt::format("Invalid dimension size {} for dimension {}", sizes[dim], dim));
			}
		}

		// Line 3: Nout
		file >> n_out;
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_out == Nout,
						 fmt::format("CSV file output dimension mismatch! File has {} outputs, but DataTable expects {}", n_out, Nout));

		// Line 4: input_names (comma-separated, in metadata order)
		std::array<std::string, Ndim> input_names{};
		std::string input_names_line;
		std::getline(file >> std::ws, input_names_line);
		{
			std::stringstream ss(input_names_line);
			for (int i = 0; i < Ndim; ++i) {
				if (i < Ndim - 1) {
					std::getline(ss, input_names[i], ',');
				} else {
					ss >> input_names[i];
				}
			}
		}

		// Line 5: output_names (comma-separated)
		std::array<std::string, Nout> output_names{};
		std::string output_names_line;
		std::getline(file >> std::ws, output_names_line);
		{
			std::stringstream ss(output_names_line);
			for (int i = 0; i < Nout; ++i) {
				if (i < Nout - 1) {
					std::getline(ss, output_names[i], ',');
				} else {
					ss >> output_names[i];
				}
			}
		}

		// Line 6: input_units (comma-separated)
		std::array<std::string, Ndim> input_units{};
		std::string input_units_line;
		std::getline(file >> std::ws, input_units_line);
		{
			std::stringstream ss(input_units_line);
			for (int i = 0; i < Ndim; ++i) {
				if (i < Ndim - 1) {
					std::getline(ss, input_units[i], ',');
				} else {
					ss >> input_units[i];
				}
			}
		}

		// Line 7: output_units (comma-separated)
		std::array<std::string, Nout> output_units{};
		std::string output_units_line;
		std::getline(file >> std::ws, output_units_line);
		{
			std::stringstream ss(output_units_line);
			for (int i = 0; i < Nout; ++i) {
				if (i < Nout - 1) {
					std::getline(ss, output_units[i], ',');
				} else {
					ss >> output_units[i];
				}
			}
		}

		// Read metadata values (xlo, xhi, spacing) - these are in input_names order (metadata order)
		std::array<amrex::Real, Ndim> xlo_metadata{};
		std::array<amrex::Real, Ndim> xhi_metadata{};
		std::array<std::string, Ndim> spacing_metadata{};

		// Line 8: xlo (comma-separated, in metadata order)
		std::string xlo_line;
		std::getline(file >> std::ws, xlo_line);
		{
			std::stringstream ss(xlo_line);
			for (int i = 0; i < Ndim; ++i) {
				char comma = ' ';
				ss >> xlo_metadata[i];
				if (i < Ndim - 1) {
					ss >> comma;
				}
			}
		}

		// Line 9: xhi (comma-separated, in metadata order)
		std::string xhi_line;
		std::getline(file >> std::ws, xhi_line);
		{
			std::stringstream ss(xhi_line);
			for (int i = 0; i < Ndim; ++i) {
				char comma = ' ';
				ss >> xhi_metadata[i];
				if (i < Ndim - 1) {
					ss >> comma;
				}
			}
		}

		// Line 10: spacing (comma-separated, in metadata order)
		std::string spacing_line;
		std::getline(file >> std::ws, spacing_line);
		{
			std::stringstream ss(spacing_line);
			for (int i = 0; i < Ndim; ++i) {
				if (i < Ndim - 1) {
					std::getline(ss, spacing_metadata[i], ',');
				} else {
					ss >> spacing_metadata[i];
				}
			}
		}

		// Copy metadata from input_names order (which should match Nx order)
		// Nx = [nx1, nx2, ...] where x1, x2, ... correspond to dimensions in order
		// metadata = [meta0, meta1, ...] in input_names order
		// Assume input_names order matches Nx order (e.g., input_names="age,mass" means x1=age, x2=mass)
		for (int dim = 0; dim < Ndim; ++dim) {
			coord_bounds[dim].first = xlo_metadata[dim];
			coord_bounds[dim].second = xhi_metadata[dim];
			spacing_types[dim] = spacing_metadata[dim];

			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    coord_bounds[dim].second > coord_bounds[dim].first,
			    fmt::format("Invalid coordinate bounds for dimension {}: [{}, {}]", dim, coord_bounds[dim].first, coord_bounds[dim].second));
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    spacing_types[dim] == "linear" || spacing_types[dim] == "log" || spacing_types[dim] == "fast_log",
			    fmt::format("Invalid spacing type '{}' for dimension {}. Must be 'linear', 'log', or 'fast_log'", spacing_types[dim], dim));
		}

		// Prepare bounds and sizes for optimized initialization
		std::array<amrex::Real, Ndim> x_mins{};
		std::array<amrex::Real, Ndim> x_maxs{};
		for (int dim = 0; dim < Ndim; ++dim) {
			x_mins[dim] = coord_bounds[dim].first;
			x_maxs[dim] = coord_bounds[dim].second;
		}

		// Empty coord arrays - will be generated automatically based on spacing type
		std::array<amrex::Vector<amrex::Real>, Ndim> empty_coords{};
		for (int dim = 0; dim < Ndim; ++dim) {
			empty_coords[dim] = amrex::Vector<amrex::Real>(); // Empty vector
		}

		// lambda function for log10
		auto log10_ = [output_spacing](amrex::Real x) -> amrex::Real {
			if (output_spacing == "fast_log") {
				return FastMath::log10(x);
			}
			return std::log10(x);
		};

		// Read data values - layout is transposed from internal representation
		// CSV layout: last dimensions as rows, first dimension as columns
		if constexpr (Ndim == 1) {
			// For 1D: single row with nx1 columns
			data_1d_type data_array;
			for (int out_idx = 0; out_idx < Nout; ++out_idx) {
				data_array[out_idx].resize(sizes[0]);
				for (int i = 0; i < sizes[0]; ++i) {
					char comma = ' ';
					file >> data_array[out_idx][i];
					if (i < sizes[0] - 1) {
						file >> comma;
					}
				}
			}

			// Apply log10 transformation if output_spacing is "fast_log"
			if (output_spacing == "fast_log" || output_spacing == "log") {
				for (int out_idx = 0; out_idx < Nout; ++out_idx) {
					for (int i = 0; i < sizes[0]; ++i) {
						AMREX_ALWAYS_ASSERT_WITH_MESSAGE(data_array[out_idx][i] > 0.0,
										 fmt::format("fast_log output spacing requires positive values, got {} at output {} index {}",
											     data_array[out_idx][i], out_idx, i));
						data_array[out_idx][i] = log10_(data_array[out_idx][i]);
					}
				}
			}

			// Create and initialize DataTable using optimized path
			DataTable table;
			table.initialize_common(x_mins, x_maxs, sizes, spacing_types, empty_coords, data_array);
			
			// Store metadata
			table.input_names_ = input_names;
			table.output_names_ = output_names;
			table.input_units_ = input_units;
			table.output_units_ = output_units;
			table.output_spacing_ = output_spacing;
			
			file.close();
			return table;

		} else if constexpr (Ndim == 2) {
			// For 2D: nx2 rows × nx1 columns
			// CSV: data[row=i2][col=i1] -> DataTable: data[out_idx][i1][i2]
			data_2d_type data_array;
			for (int out_idx = 0; out_idx < Nout; ++out_idx) {
				data_array[out_idx].resize(sizes[0]);
				for (int i1 = 0; i1 < sizes[0]; ++i1) {
					data_array[out_idx][i1].resize(sizes[1]);
				}

				// Read data in transposed order
				for (int i2 = 0; i2 < sizes[1]; ++i2) {
					for (int i1 = 0; i1 < sizes[0]; ++i1) {
						char comma = ' ';
						file >> data_array[out_idx][i1][i2];
						if (i1 < sizes[0] - 1) {
							file >> comma;
						}
					}
				}
			}

			// Apply log10 transformation if output_spacing is "fast_log"
			if (output_spacing == "fast_log" || output_spacing == "log") {
				for (int out_idx = 0; out_idx < Nout; ++out_idx) {
					for (int i1 = 0; i1 < sizes[0]; ++i1) {
						for (int i2 = 0; i2 < sizes[1]; ++i2) {
							AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
							    data_array[out_idx][i1][i2] > 0.0,
							    fmt::format("fast_log output spacing requires positive values, got {} at output {} index ({}, {})",
									data_array[out_idx][i1][i2], out_idx, i1, i2));
							data_array[out_idx][i1][i2] = log10_(data_array[out_idx][i1][i2]);
						}
					}
				}
			}

			// Create and initialize DataTable using optimized path
			DataTable table;
			table.initialize_common(x_mins, x_maxs, sizes, spacing_types, empty_coords, data_array);
			
			// Store metadata
			table.input_names_ = input_names;
			table.output_names_ = output_names;
			table.input_units_ = input_units;
			table.output_units_ = output_units;
			table.output_spacing_ = output_spacing;
			
			file.close();
			return table;

		} else if constexpr (Ndim == 3) {
			// For 3D: (nx3 × nx2) rows × nx1 columns
			// CSV: data[row=(i3*nx2+i2)][col=i1] -> DataTable: data[out_idx][i1][i2][i3]
			data_3d_type data_array;
			for (int out_idx = 0; out_idx < Nout; ++out_idx) {
				data_array[out_idx].resize(sizes[0]);
				for (int i1 = 0; i1 < sizes[0]; ++i1) {
					data_array[out_idx][i1].resize(sizes[1]);
					for (int i2 = 0; i2 < sizes[1]; ++i2) {
						data_array[out_idx][i1][i2].resize(sizes[2]);
					}
				}

				// Read data in transposed order
				for (int i3 = 0; i3 < sizes[2]; ++i3) {
					for (int i2 = 0; i2 < sizes[1]; ++i2) {
						for (int i1 = 0; i1 < sizes[0]; ++i1) {
							char comma = ' ';
							file >> data_array[out_idx][i1][i2][i3];
							if (i1 < sizes[0] - 1) {
								file >> comma;
							}
						}
					}
				}
			}

			// Apply log10 transformation if output_spacing is "fast_log"
			if (output_spacing == "fast_log" || output_spacing == "log") {
				for (int out_idx = 0; out_idx < Nout; ++out_idx) {
					for (int i1 = 0; i1 < sizes[0]; ++i1) {
						for (int i2 = 0; i2 < sizes[1]; ++i2) {
							for (int i3 = 0; i3 < sizes[2]; ++i3) {
								AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
								    data_array[out_idx][i1][i2][i3] > 0.0,
								    fmt::format("fast_log output spacing requires positive values, got {} at output {} index ({}, {}, {})",
										data_array[out_idx][i1][i2][i3], out_idx, i1, i2, i3));
								data_array[out_idx][i1][i2][i3] = log10_(data_array[out_idx][i1][i2][i3]);
							}
						}
					}
				}
			}

			// Create and initialize DataTable using optimized path
			DataTable table;
			table.initialize_common(x_mins, x_maxs, sizes, spacing_types, empty_coords, data_array);
			
			// Store metadata
			table.input_names_ = input_names;
			table.output_names_ = output_names;
			table.input_units_ = input_units;
			table.output_units_ = output_units;
			table.output_spacing_ = output_spacing;
			
			file.close();
			return table;

		} else if constexpr (Ndim == 4) {
			// For 4D: (nx4 × nx3 × nx2) rows × nx1 columns
			// CSV: data[row=(i4*nx3*nx2+i3*nx2+i2)][col=i1] -> DataTable: data[out_idx][i1][i2][i3][i4]
			data_4d_type data_array;
			for (int out_idx = 0; out_idx < Nout; ++out_idx) {
				data_array[out_idx].resize(sizes[0]);
				for (int i1 = 0; i1 < sizes[0]; ++i1) {
					data_array[out_idx][i1].resize(sizes[1]);
					for (int i2 = 0; i2 < sizes[1]; ++i2) {
						data_array[out_idx][i1][i2].resize(sizes[2]);
						for (int i3 = 0; i3 < sizes[2]; ++i3) {
							data_array[out_idx][i1][i2][i3].resize(sizes[3]);
						}
					}
				}

				// Read data in transposed order
				for (int i4 = 0; i4 < sizes[3]; ++i4) {
					for (int i3 = 0; i3 < sizes[2]; ++i3) {
						for (int i2 = 0; i2 < sizes[1]; ++i2) {
							for (int i1 = 0; i1 < sizes[0]; ++i1) {
								char comma = ' ';
								file >> data_array[out_idx][i1][i2][i3][i4];
								if (i1 < sizes[0] - 1) {
									file >> comma;
								}
							}
						}
					}
				}
			}

			// Apply log10 transformation if output_spacing is "fast_log"
			if (output_spacing == "fast_log" || output_spacing == "log") {
				for (int out_idx = 0; out_idx < Nout; ++out_idx) {
					for (int i1 = 0; i1 < sizes[0]; ++i1) {
						for (int i2 = 0; i2 < sizes[1]; ++i2) {
							for (int i3 = 0; i3 < sizes[2]; ++i3) {
								for (int i4 = 0; i4 < sizes[3]; ++i4) {
									AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
									    data_array[out_idx][i1][i2][i3][i4] > 0.0,
									    fmt::format("fast_log output spacing requires positive values, got {} at output {} index ({}, {}, "
											"{}, {})",
											data_array[out_idx][i1][i2][i3][i4], out_idx, i1, i2, i3, i4));
									data_array[out_idx][i1][i2][i3][i4] = log10_(data_array[out_idx][i1][i2][i3][i4]);
								}
							}
						}
					}
				}
			}

			// Create and initialize DataTable using optimized path
			DataTable table;
			table.initialize_common(x_mins, x_maxs, sizes, spacing_types, empty_coords, data_array);
			
			// Store metadata
			table.input_names_ = input_names;
			table.output_names_ = output_names;
			table.input_units_ = input_units;
			table.output_units_ = output_units;
			table.output_spacing_ = output_spacing;
			
			file.close();
			return table;
		}
	}

	// H5Reader: Generic static method to read n-dimensional data from HDF5 file and create DataTable
	// Reads metadata, coordinates, and data all from the HDF5 file
	// Optionally returns coordinate bounds via coord_bounds parameter
	static auto H5Reader(const std::string &file_path, const std::string &dataset_path, const std::vector<std::string> &coord_names, int is_fast_log = 0,
			     std::array<std::pair<amrex::Real, amrex::Real>, Ndim> *coord_bounds = nullptr) -> DataTable
	{
		static_assert(Ndim >= 1 && Ndim <= 4, "H5Reader supports 1D-4D tables");
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
		    coord_names.size() == Ndim,
		    fmt::format("H5Reader requires exactly Ndim coordinate names! (expected: {}, provided: {})", Ndim, coord_names.size()));

		herr_t status = 0;
		herr_t const h5_error = -1;
		hid_t file_id = 0;
		hid_t dset_id = 0;
		hid_t attr_id = 0;

		// Open HDF5 file
		file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file_id != h5_error, ("Failed to open HDF5 file: " + file_path).c_str());

		// Read metadata group to get grid dimensions
		hid_t const metadata_group = H5Gopen2(file_id, "/metadata", H5P_DEFAULT);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(metadata_group != h5_error, "Failed to open metadata group!");

		// Read grid dimensions using generic names
		std::vector<int> n_coords(Ndim);
		std::vector<std::string> n_coord_attrs(Ndim);

		for (int dim = 0; dim < Ndim; ++dim) {
			n_coord_attrs[dim] = "n_" + coord_names[dim];
			attr_id = H5Aopen(metadata_group, n_coord_attrs[dim].c_str(), H5P_DEFAULT);
			status = H5Aread(attr_id, H5T_NATIVE_INT, &n_coords[dim]);
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, ("Failed to read " + n_coord_attrs[dim] + "!").c_str());
			H5Aclose(attr_id);
		}

		// Read coordinate bounds if requested
		if (coord_bounds != nullptr) {
			for (int dim = 0; dim < Ndim; ++dim) {
				const std::string min_attr = coord_names[dim] + "_min";
				const std::string max_attr = coord_names[dim] + "_max";

				attr_id = H5Aopen(metadata_group, min_attr.c_str(), H5P_DEFAULT);
				status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &(*coord_bounds)[dim].first);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, ("Failed to read " + min_attr + "!").c_str());
				H5Aclose(attr_id);

				attr_id = H5Aopen(metadata_group, max_attr.c_str(), H5P_DEFAULT);
				status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &(*coord_bounds)[dim].second);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, ("Failed to read " + max_attr + "!").c_str());
				H5Aclose(attr_id);
			}
		}

		H5Gclose(metadata_group);

		// Read coordinate grids
		std::vector<amrex::Vector<amrex::Real>> coords(Ndim);
		for (int dim = 0; dim < Ndim; ++dim) {
			coords[dim].resize(n_coords[dim]);
		}

		// Construct coordinate dataset names based on is_fast_log parameter
		const std::string prefix = (is_fast_log == 1) ? "fast_log_" : "";
		std::vector<std::string> coord_datasets(Ndim);
		for (int dim = 0; dim < Ndim; ++dim) {
			coord_datasets[dim] = "/grids/" + prefix + coord_names[dim];
		}

		// Read coordinates using for loop
		for (int dim = 0; dim < Ndim; ++dim) {
			std::vector<double> temp_data(n_coords[dim]);
			dset_id = H5Dopen2(file_id, coord_datasets[dim].c_str(), H5P_DEFAULT);
			status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data.data());
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, ("Failed to read " + coord_datasets[dim] + " dataset!").c_str());
			H5Dclose(dset_id);

			for (int i = 0; i < n_coords[dim]; ++i) {
				coords[dim][i] = temp_data[i];
			}
		}

		// Read n-dimensional dataset from HDF5 file
		// Calculate data_size as product of all dimensions
		auto data_size = static_cast<int64_t>(Nout);
		for (int dim = 0; dim < Ndim; ++dim) {
			data_size *= static_cast<int64_t>(n_coords[dim]);
		}
		std::vector<double> temp_data(data_size);

		dset_id = H5Dopen2(file_id, dataset_path.c_str(), H5P_DEFAULT);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dset_id != h5_error, ("Failed to open HDF5 dataset: " + dataset_path).c_str());

		status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data.data());
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, ("Failed to read HDF5 dataset: " + dataset_path).c_str());

		H5Dclose(dset_id);

		// Create coordinate arrays for any dimension
		std::array<amrex::Vector<amrex::Real>, Ndim> coord_arrays;
		for (int dim = 0; dim < Ndim; ++dim) {
			coord_arrays[dim] = coords[dim];
		}

		// Convert HDF5 C-order data to natural dimensional format
		if constexpr (Ndim == 1) {
			// For 1D: data[out_idx][i]
			data_1d_type data_array;
			for (int out_idx = 0; out_idx < Nout; ++out_idx) {
				data_array[out_idx].resize(n_coords[0]);
				for (int i = 0; i < n_coords[0]; ++i) {
					data_array[out_idx][i] = temp_data[out_idx * n_coords[0] + i];
				}
			}

			// Create and initialize DataTable
			DataTable table;
			table.initialize(coord_arrays, data_array);

			// Close HDF5 file
			H5Fclose(file_id);
			return table;

		} else if constexpr (Ndim == 2) {
			// For 2D: data[out_idx][i][j]
			data_2d_type data_array;
			for (int out_idx = 0; out_idx < Nout; ++out_idx) {
				data_array[out_idx].resize(n_coords[0]);
				for (int i = 0; i < n_coords[0]; ++i) {
					data_array[out_idx][i].resize(n_coords[1]);
					for (int j = 0; j < n_coords[1]; ++j) {
						data_array[out_idx][i][j] = temp_data[out_idx * n_coords[0] * n_coords[1] + i * n_coords[1] + j];
					}
				}
			}

			// Create and initialize DataTable
			DataTable table;
			table.initialize(coord_arrays, data_array);

			// Close HDF5 file
			H5Fclose(file_id);
			return table;

		} else if constexpr (Ndim == 3) {
			// For 3D: data[out_idx][i][j][k]
			data_3d_type data_array;
			for (int out_idx = 0; out_idx < Nout; ++out_idx) {
				data_array[out_idx].resize(n_coords[0]);
				for (int i = 0; i < n_coords[0]; ++i) {
					data_array[out_idx][i].resize(n_coords[1]);
					for (int j = 0; j < n_coords[1]; ++j) {
						data_array[out_idx][i][j].resize(n_coords[2]);
						for (int k = 0; k < n_coords[2]; ++k) {
							data_array[out_idx][i][j][k] = temp_data[out_idx * n_coords[0] * n_coords[1] * n_coords[2] +
												 i * n_coords[1] * n_coords[2] + j * n_coords[2] + k];
						}
					}
				}
			}

			// Create and initialize DataTable
			DataTable table;
			table.initialize(coord_arrays, data_array);

			// Close HDF5 file
			H5Fclose(file_id);
			return table;

		} else if constexpr (Ndim == 4) {
			// For 4D: data[out_idx][i][j][k][l]
			data_4d_type data_array;
			for (int out_idx = 0; out_idx < Nout; ++out_idx) {
				data_array[out_idx].resize(n_coords[0]);
				for (int i = 0; i < n_coords[0]; ++i) {
					data_array[out_idx][i].resize(n_coords[1]);
					for (int j = 0; j < n_coords[1]; ++j) {
						data_array[out_idx][i][j].resize(n_coords[2]);
						for (int k = 0; k < n_coords[2]; ++k) {
							data_array[out_idx][i][j][k].resize(n_coords[3]);
							for (int l = 0; l < n_coords[3]; ++l) {
								data_array[out_idx][i][j][k][l] =
								    temp_data[out_idx * n_coords[0] * n_coords[1] * n_coords[2] * n_coords[3] +
									      i * n_coords[1] * n_coords[2] * n_coords[3] + j * n_coords[2] * n_coords[3] +
									      k * n_coords[3] + l];
							}
						}
					}
				}
			}

			// Create and initialize DataTable
			DataTable table;
			table.initialize(coord_arrays, data_array);

			// Close HDF5 file
			H5Fclose(file_id);
			return table;
		}
	}
};

} // namespace quokka

#endif // DATATABLE_HPP_