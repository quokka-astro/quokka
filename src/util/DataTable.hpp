#ifndef DATATABLE_HPP_
#define DATATABLE_HPP_

#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include "AMReX_Enum.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_TableData.H"

// HDF5 includes for H5Reader functionality
#include <H5Dpublic.h>
#include <H5Ppublic.h>
#include <hdf5.h>

#include "math/FastMath.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// For descriptive error messages
#include <format>

namespace quokka
{

// Enum for spacing types (GPU-compatible, supports ParmParse)
AMREX_ENUM(TransformType, linear, log, fast_log); // NOLINT

// Enum for out-of-bounds handling (GPU-compatible, supports ParmParse)
AMREX_ENUM(OutOfBounds, clamp, fail); // NOLINT

// Structure to hold interpolation indices and normalized coordinates
template <int Ndim> struct InterpData {
	std::array<int, Ndim> indices{};	    // grid indices for each dimension (lower bounds)
	std::array<amrex::Real, Ndim> normalized{}; // normalized coordinates in [0,1] for each dimension

	// Default constructor
	AMREX_GPU_HOST_DEVICE InterpData() = default;
};

// GPU-friendly struct containing const table references
template <int Ndim, int Nout = 1, OutOfBounds oob_policy = OutOfBounds::clamp> struct DataTableGpuConst {
	std::array<amrex::Table1D<const amrex::Real>, Ndim> coords;
	// Array of data tables for multiple outputs - each has the same coordinate dimensionality
	using single_data_table_type =
	    std::conditional_t<Ndim == 1, amrex::Table1D<const amrex::Real>,
			       std::conditional_t<Ndim == 2, amrex::Table2D<const amrex::Real>,
						  std::conditional_t<Ndim == 3, amrex::Table3D<const amrex::Real>, amrex::Table4D<const amrex::Real>>>>;
	std::array<single_data_table_type, Nout> dataViewArrays;

	std::array<amrex::Real, Ndim> coord_min{};
	std::array<amrex::Real, Ndim> coord_max{};
	std::array<TransformType, Ndim> spacing_types{};

	// Precomputed grid spacing for optimization
	std::array<amrex::Real, Ndim> dcoord{};

	std::array<int, Ndim> sizes{};

	// Per-output transform for return values: linear, log, or fast_log
	std::array<TransformType, Nout> output_transforms{};

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

			// Handle out-of-bounds based on policy
			amrex::Real clamped_coord = point[dim];
			if constexpr (oob_policy == OutOfBounds::clamp) {
				// Clamp coordinates to valid table bounds (extrapolation not supported)
				clamped_coord = amrex::max(coord_start, amrex::min(point[dim], coord_end));
			} else if constexpr (oob_policy == OutOfBounds::fail) {
				// Check if point is out of bounds and abort if so
				if (point[dim] < coord_start || point[dim] > coord_end) {
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
					    false, std::format("Point is out of bounds for dimension {}! (value: {}, valid range: [{}, {}])", dim, point[dim],
							       coord_start, coord_end)
						       .c_str());
				}
			}

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
		InterpData<Ndim> const interp = find_interpolation_data(transform_point(point));
		auto values = interpolate_from_indices(interp);
		for (int i = 0; i < Nout; ++i) {
			if (output_transforms[i] == TransformType::fast_log) {
				values[i] = FastMath::pow2(values[i]);
			} else if (output_transforms[i] == TransformType::log) {
				values[i] = std::exp(values[i]);
			}
		}
		return values;
	}

	/// @brief Compute the partial derivative of output[output_idx] w.r.t. physical input axis `axis`.
	///
	/// Uses a centered finite difference with a step equal to half the local grid spacing
	/// in the axis's coordinate space, converted to physical units. This ties the step to
	/// the table resolution and avoids absolute-floor problems at extreme parameter values.
	///
	/// @param point      Physical coordinates to differentiate at (size Ndim)
	/// @param axis       Input dimension index to differentiate with respect to (0-indexed)
	/// @param output_idx Output quantity index to differentiate (0-indexed)
	/// @return ∂output[output_idx] / ∂point[axis] in physical units.
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto partial_derivative(const std::array<amrex::Real, Ndim> &point, int axis,
											int output_idx = 0) const -> amrex::Real
	{
		// Half a grid cell in transform space, converted to physical units.
		// For log/fast_log axes, point[axis] must be positive (precondition for log spacing).
		const amrex::Real h_t = amrex::Real(0.5) * dcoord[axis];
		amrex::Real h_p;
		if (spacing_types[axis] == TransformType::log) {
			AMREX_ASSERT(point[axis] > amrex::Real(0.0));
			h_p = point[axis] * (std::exp(h_t) - amrex::Real(1.0));
		} else if (spacing_types[axis] == TransformType::fast_log) {
			AMREX_ASSERT(point[axis] > amrex::Real(0.0));
			h_p = point[axis] * (FastMath::pow2(h_t) - amrex::Real(1.0));
		} else {
			h_p = h_t;
		}
		auto pt_hi = point;
		auto pt_lo = point;
		pt_hi[axis] += h_p;
		pt_lo[axis] -= h_p;
		return (interpolate_single(pt_hi, output_idx) - interpolate_single(pt_lo, output_idx)) / (amrex::Real(2.0) * h_p);
	}

	/// @brief Perform n-dimensional linear interpolation for a single output (backward compatibility)
	///
	/// @param point Physical coordinates to interpolate at (size Ndim)
	/// @param output_index Index of the output to interpolate (0 to Nout-1)
	/// @return Single interpolated value
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate_single(const std::array<amrex::Real, Ndim> &point, int output_index = 0) const
	    -> amrex::Real
	{
		InterpData<Ndim> const interp = find_interpolation_data(transform_point(point));
		amrex::Real value = interpolate_single_from_indices(interp, output_index);
		if (output_transforms[output_index] == TransformType::fast_log) {
			value = FastMath::pow2(value);
		} else if (output_transforms[output_index] == TransformType::log) {
			value = std::exp(value);
		}
		return value;
	}

      private:
	/// @brief Transform physical coordinates to interpolation space (log/fast_log per dimension)
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto transform_point(const std::array<amrex::Real, Ndim> &point) const
	    -> std::array<amrex::Real, Ndim>
	{
		std::array<amrex::Real, Ndim> pt{};
		for (int dim = 0; dim < Ndim; ++dim) {
			if (spacing_types[dim] == TransformType::log) {
				if constexpr (oob_policy == OutOfBounds::clamp) {
					constexpr amrex::Real epsilon = std::numeric_limits<amrex::Real>::min() * 1.0e10; // ~1e-298 for double
					pt[dim] = std::log(amrex::max(point[dim], epsilon));
				} else if constexpr (oob_policy == OutOfBounds::fail) {
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
					    point[dim] > 0.0,
					    std::format("Invalid value for log interpolation in dimension {}: {} (must be positive)", dim, point[dim]).c_str());
					pt[dim] = std::log(point[dim]);
				}
			} else if (spacing_types[dim] == TransformType::fast_log) {
				if constexpr (oob_policy == OutOfBounds::clamp) {
					constexpr amrex::Real epsilon = std::numeric_limits<amrex::Real>::min() * 1.0e10; // ~1e-298 for double
					pt[dim] = FastMath::lg(amrex::max(point[dim], epsilon));
				} else if constexpr (oob_policy == OutOfBounds::fail) {
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
					    point[dim] > 0.0,
					    std::format("Invalid value for fast_log interpolation in dimension {}: {} (must be positive)", dim, point[dim])
						.c_str());
					pt[dim] = FastMath::lg(point[dim]);
				}
			} else {
				pt[dim] = point[dim];
			}
		}
		return pt;
	}

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

			// NOSONAR
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
template <int Ndim, int Nout = 1, OutOfBounds oob_policy = OutOfBounds::clamp> class DataTable
{
      private:
	using coord_table_type = amrex::TableData<amrex::Real, 1>;
	using data_table_type = amrex::TableData<amrex::Real, Ndim>;

	// Host staging tables (pinned for efficient H2D copies in GPU builds).
	std::array<std::unique_ptr<coord_table_type>, Ndim> coords_h_;
	std::array<std::unique_ptr<data_table_type>, Nout> data_h_;
	// Device-visible tables used by kernels via const_tables().
	std::array<std::unique_ptr<coord_table_type>, Ndim> coords_d_;
	std::array<std::unique_ptr<data_table_type>, Nout> data_d_;

	// Type aliases for different dimensional data structures
	using data_1d_type = std::array<amrex::Vector<amrex::Real>, Nout>;
	using data_2d_type = std::array<amrex::Vector<amrex::Vector<amrex::Real>>, Nout>;
	using data_3d_type = std::array<amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>>, Nout>;
	using data_4d_type = std::array<amrex::Vector<amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>>>, Nout>;

	std::array<amrex::Real, Ndim> coord_min_{};
	std::array<amrex::Real, Ndim> coord_max_{};
	std::array<amrex::Real, Ndim> xlo_physical_{}; // bounds in physical units (before any log transform)
	std::array<amrex::Real, Ndim> xhi_physical_{};
	std::array<TransformType, Ndim> spacing_types_{};

	// Precomputed grid spacing for optimization
	std::array<amrex::Real, Ndim> dcoord_{};

	std::array<int, Ndim> sizes_{};

	// Metadata for dimension and output names/units
	std::array<std::string, Ndim> input_names_{};
	std::array<std::string, Nout> output_names_{};
	std::array<std::string, Ndim> input_units_{};
	std::array<std::string, Nout> output_units_{};

	// Per-output transform type: linear, log, or fast_log
	std::array<TransformType, Nout> output_transforms_{};

	[[nodiscard]] static constexpr auto ioProcessorNumber() -> int { return amrex::ParallelDescriptor::IOProcessorNumber(); }

	template <typename T> static void bcastScalar(T &value) { amrex::ParallelDescriptor::Bcast(&value, 1, ioProcessorNumber()); }

	template <typename T, std::size_t N> static void bcastArray(std::array<T, N> &values)
	{
		amrex::ParallelDescriptor::Bcast(values.data(), values.size(), ioProcessorNumber());
	}

	static void bcastString(std::string &value)
	{
		std::int64_t len = amrex::ParallelDescriptor::IOProcessor() ? static_cast<std::int64_t>(value.size()) : 0;
		bcastScalar(len);
		if (!amrex::ParallelDescriptor::IOProcessor()) {
			value.resize(static_cast<std::size_t>(len));
		}
		if (len > 0) {
			amrex::ParallelDescriptor::Bcast(value.data(), static_cast<std::size_t>(len), ioProcessorNumber());
		}
	}

	template <std::size_t N> static void bcastStringArray(std::array<std::string, N> &values)
	{
		for (auto &value : values) {
			bcastString(value);
		}
	}

	template <typename T> static void bcastVector(amrex::Vector<T> &values)
	{
		std::int64_t count = amrex::ParallelDescriptor::IOProcessor() ? static_cast<std::int64_t>(values.size()) : 0;
		bcastScalar(count);
		if (!amrex::ParallelDescriptor::IOProcessor()) {
			values.resize(static_cast<std::size_t>(count));
		}
		if (count > 0) {
			amrex::ParallelDescriptor::Bcast(values.data(), static_cast<std::size_t>(count), ioProcessorNumber());
		}
	}

	static void bcastTransformType(TransformType &value)
	{
		int spacing_value = amrex::ParallelDescriptor::IOProcessor() ? static_cast<int>(value) : 0;
		bcastScalar(spacing_value);
		if (!amrex::ParallelDescriptor::IOProcessor()) {
			value = static_cast<TransformType>(spacing_value);
		}
	}

	static void bcastTransformTypes(std::array<TransformType, Ndim> &values)
	{
		std::array<int, Ndim> serialized{};
		if (amrex::ParallelDescriptor::IOProcessor()) {
			for (int dim = 0; dim < Ndim; ++dim) {
				serialized[dim] = static_cast<int>(values[dim]);
			}
		}
		bcastArray(serialized);
		if (!amrex::ParallelDescriptor::IOProcessor()) {
			for (int dim = 0; dim < Ndim; ++dim) {
				values[dim] = static_cast<TransformType>(serialized[dim]);
			}
		}
	}

	[[nodiscard]] static auto flatDataSize(const std::array<int, Ndim> &sizes) -> amrex::Long
	{
		auto count = static_cast<amrex::Long>(Nout);
		for (int dim = 0; dim < Ndim; ++dim) {
			count *= static_cast<amrex::Long>(sizes[dim]);
		}
		return count;
	}

	[[nodiscard]] static auto flatDataIndex(int out_idx, const std::array<int, Ndim> &sizes, const std::array<int, Ndim> &indices) -> std::size_t
	{
		auto index = static_cast<std::size_t>(out_idx);
		for (int dim = 0; dim < Ndim; ++dim) {
			index = index * static_cast<std::size_t>(sizes[dim]) + static_cast<std::size_t>(indices[dim]);
		}
		return index;
	}

      public:
	void setMetadata(const std::array<std::string, Ndim> &input_names, const std::array<std::string, Nout> &output_names,
			 const std::array<std::string, Ndim> &input_units, const std::array<std::string, Nout> &output_units,
			 std::array<TransformType, Nout> output_transforms)
	{
		input_names_ = input_names;
		output_names_ = output_names;
		input_units_ = input_units;
		output_units_ = output_units;
		output_transforms_ = output_transforms;
	}

	// Default constructor
	DataTable() = default;

	// Constructor with coordinate arrays and data - general n-dimensional interface
	// For multiple outputs, data is organized as data[output_index][flattened_coords][last_dim]
	DataTable(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const std::array<amrex::Vector<amrex::Vector<amrex::Real>>, Nout> &data)
	{
		initialize(coords, data);
	}

	// Backward compatibility constructor for single output (Nout = 1)
	template <int N = Nout>
	DataTable(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const amrex::Vector<amrex::Vector<amrex::Real>> &data) requires(N == 1)
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
	template <int N = Ndim> void initialize(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const data_1d_type &data) requires(N == 1)
	{
		static_assert(Ndim == 1, "This initialize overload is for 1D tables only");

		// Validate inputs
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coords[dim].empty(), std::format("Coordinates for dimension {} cannot be empty!", dim));
		}

		// Validate data dimensions for each output
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    data[out_idx].size() == coords[0].size(),
			    std::format("1D data must match coordinate size! (expected: {}, actual: {})", coords[0].size(), data[out_idx].size()));
		}

		initialize_common(coords, data);
	}

	// Initialize from coordinate arrays - 2D interface
	template <int N = Ndim> void initialize(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const data_2d_type &data) requires(N == 2)
	{
		static_assert(Ndim == 2, "This initialize overload is for 2D tables only");

		// Validate inputs
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coords[dim].empty(), std::format("Coordinates for dimension {} cannot be empty!", dim));
		}

		// Validate data dimensions for each output
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!data[out_idx].empty(), std::format("Data for output {} cannot be empty!", out_idx));
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    data[out_idx].size() == coords[0].size(),
			    std::format("Data first dimension must match first coordinate size for output {}! (expected: {}, actual: {})", out_idx,
					coords[0].size(), data[out_idx].size()));
			// Verify data dimensions
			for (const auto &row : data[out_idx]) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
				    row.size() == coords[1].size(),
				    std::format("All data rows must match second coordinate size for output {}! (expected: {}, actual: {})", out_idx,
						coords[1].size(), row.size()));
			}
		}

		initialize_common(coords, data);
	}

	// Initialize from coordinate arrays - 3D interface
	template <int N = Ndim> void initialize(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const data_3d_type &data) requires(N == 3)
	{
		static_assert(Ndim == 3, "This initialize overload is for 3D tables only");

		// Validate inputs
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coords[dim].empty(), std::format("Coordinates for dimension {} cannot be empty!", dim));
		}

		// Validate data dimensions for each output
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!data[out_idx].empty(), std::format("Data for output {} cannot be empty!", out_idx));
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    data[out_idx].size() == coords[0].size(),
			    std::format("Data first dimension must match first coordinate size for output {}! (expected: {}, actual: {})", out_idx,
					coords[0].size(), data[out_idx].size()));
			for (const auto &plane : data[out_idx]) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
				    plane.size() == coords[1].size(),
				    std::format("Data second dimension must match second coordinate size for output {}! (expected: {}, actual: {})", out_idx,
						coords[1].size(), plane.size()));
				for (const auto &row : plane) {
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
					    row.size() == coords[2].size(),
					    std::format("Data third dimension must match third coordinate size for output {}! (expected: {}, actual: {})",
							out_idx, coords[2].size(), row.size()));
				}
			}
		}

		initialize_common(coords, data);
	}

	// Initialize from coordinate arrays - 4D interface
	template <int N = Ndim> void initialize(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const data_4d_type &data) requires(N == 4)
	{
		static_assert(Ndim == 4, "This initialize overload is for 4D tables only");

		// Validate inputs
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coords[dim].empty(), std::format("Coordinates for dimension {} cannot be empty!", dim));
		}

		// Validate data dimensions for each output
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!data[out_idx].empty(), std::format("Data for output {} cannot be empty!", out_idx));
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    data[out_idx].size() == coords[0].size(),
			    std::format("Data first dimension must match first coordinate size for output {}! (expected: {}, actual: {})", out_idx,
					coords[0].size(), data[out_idx].size()));
			for (const auto &volume : data[out_idx]) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
				    volume.size() == coords[1].size(),
				    std::format("Data second dimension must match second coordinate size for output {}! (expected: {}, actual: {})", out_idx,
						coords[1].size(), volume.size()));
				for (const auto &plane : volume) {
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
					    plane.size() == coords[2].size(),
					    std::format("Data third dimension must match third coordinate size for output {}! (expected: {}, actual: {})",
							out_idx, coords[2].size(), plane.size()));
					for (const auto &row : plane) { // NOSONAR
						AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
						    row.size() == coords[3].size(),
						    std::format(
							"Data fourth dimension must match fourth coordinate size for output {}! (expected: {}, actual: {})",
							out_idx, coords[3].size(), row.size()));
					}
				}
			}
		}

		initialize_common(coords, data);
	}

	// Get GPU-friendly const tables (device-backed; only safe to access from GPU kernels).
	[[nodiscard]] auto const_tables() const -> DataTableGpuConst<Ndim, Nout, oob_policy>
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_initialized(), "DataTable must be initialized before getting const tables!");

		std::array<amrex::Table1D<const amrex::Real>, Ndim> coord_tables{};
		for (int i = 0; i < Ndim; ++i) {
			coord_tables[i] = coords_d_[i]->const_table();
		}

		std::array<typename DataTableGpuConst<Ndim, Nout, oob_policy>::single_data_table_type, Nout> data_tables{};
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			data_tables[out_idx] = data_d_[out_idx]->const_table();
		}

		DataTableGpuConst<Ndim, Nout, oob_policy> tables{
		    coord_tables,
		    data_tables,       // array of data tables
		    coord_min_,	       // coord_min array
		    coord_max_,	       // coord_max array
		    spacing_types_,    // spacing types array (converted to enum)
		    dcoord_,	       // dcoord array
		    sizes_,	       // sizes array
		    output_transforms_ // per-output transform
		};
		return tables;
	}

	// Get host-accessible const tables (pinned memory; safe to read on the CPU).
	// Use this for host-side initialization code that reads table values directly.
	// GPU kernels should use const_tables() instead for optimal performance.
	[[nodiscard]] auto const_tables_host() const -> DataTableGpuConst<Ndim, Nout, oob_policy>
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_initialized(), "DataTable must be initialized before getting const tables!");

		std::array<amrex::Table1D<const amrex::Real>, Ndim> coord_tables{};
		for (int i = 0; i < Ndim; ++i) {
			coord_tables[i] = coords_h_[i]->const_table();
		}

		std::array<typename DataTableGpuConst<Ndim, Nout, oob_policy>::single_data_table_type, Nout> data_tables{};
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			data_tables[out_idx] = data_h_[out_idx]->const_table();
		}

		DataTableGpuConst<Ndim, Nout, oob_policy> tables{coord_tables,	 data_tables, coord_min_, coord_max_,
								 spacing_types_, dcoord_,     sizes_,	  output_transforms_};
		return tables;
	}

	// Check if table is initialized
	[[nodiscard]] auto is_initialized() const -> bool
	{
		// Check all coordinate arrays
		for (int dim = 0; dim < Ndim; ++dim) {
			if (coords_h_[dim] == nullptr || coords_d_[dim] == nullptr) {
				return false;
			}
		}
		// Check all data tables
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			if (data_h_[out_idx] == nullptr || data_d_[out_idx] == nullptr) {
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
						 std::format("Dimension index out of bounds! (provided: {}, valid range: [0, {}])", dim, Ndim - 1));
		return sizes_[dim];
	}

	// Get number of outputs
	[[nodiscard]] constexpr auto num_outputs() const -> int { return Nout; }

	// Get physical coordinate bounds (before any log transform)
	[[nodiscard]] auto coord_xlo() const -> std::array<amrex::Real, Ndim> { return xlo_physical_; }
	[[nodiscard]] auto coord_xhi() const -> std::array<amrex::Real, Ndim> { return xhi_physical_; }

	// Get metadata accessors
	[[nodiscard]] auto input_names() const -> std::array<std::string, Ndim> { return input_names_; }
	[[nodiscard]] auto output_names() const -> std::array<std::string, Nout> { return output_names_; }
	[[nodiscard]] auto input_units() const -> std::array<std::string, Ndim> { return input_units_; }
	[[nodiscard]] auto output_units() const -> std::array<std::string, Nout> { return output_units_; }

	// Get individual metadata by index
	[[nodiscard]] auto input_name(int dim) const -> std::string
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dim >= 0 && dim < Ndim,
						 std::format("Dimension index out of bounds! (provided: {}, valid range: [0, {}])", dim, Ndim - 1));
		return input_names_[dim];
	}
	[[nodiscard]] auto output_name(int idx) const -> std::string
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(idx >= 0 && idx < Nout,
						 std::format("Output index out of bounds! (provided: {}, valid range: [0, {}])", idx, Nout - 1));
		return output_names_[idx];
	}
	[[nodiscard]] auto input_unit(int dim) const -> std::string
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dim >= 0 && dim < Ndim,
						 std::format("Dimension index out of bounds! (provided: {}, valid range: [0, {}])", dim, Ndim - 1));
		return input_units_[dim];
	}
	[[nodiscard]] auto output_unit(int idx) const -> std::string
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(idx >= 0 && idx < Nout,
						 std::format("Output index out of bounds! (provided: {}, valid range: [0, {}])", idx, Nout - 1));
		return output_units_[idx];
	}

      private:
	// Optimized initialization that takes bounds, sizes, and spacing directly
	// coords parameter is optional - if empty, coordinates will be generated based on spacing type
	void sync_tables_to_device()
	{
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coords_h_[dim] != nullptr && coords_d_[dim] != nullptr,
							 "Coordinate tables must be allocated before H2D copy.");
			coords_d_[dim]->copy(*coords_h_[dim]);
		}
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(data_h_[out_idx] != nullptr && data_d_[out_idx] != nullptr,
							 "Data tables must be allocated before H2D copy.");
			data_d_[out_idx]->copy(*data_h_[out_idx]);
		}
		amrex::Gpu::streamSynchronize();
	}

	void initializeStorage(const std::array<amrex::Real, Ndim> &x_mins, const std::array<amrex::Real, Ndim> &x_maxs, const std::array<int, Ndim> &n_xs,
			       const std::array<TransformType, Ndim> &spacing_types)
	{
		static_assert(Ndim >= 1 && Ndim <= 4, "Only 1D-4D tables are supported");

		coord_min_ = x_mins;
		coord_max_ = x_maxs;
		xlo_physical_ = x_mins; // save physical bounds before any log transform
		xhi_physical_ = x_maxs;
		sizes_ = n_xs;
		spacing_types_ = spacing_types;

		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coord_max_[dim] > coord_min_[dim], std::format("Invalid coordinate bounds for dimension {}: [{}, {}]",
													dim, coord_min_[dim], coord_max_[dim]));
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sizes_[dim] > 0, std::format("Invalid dimension size {} for dimension {}", sizes_[dim], dim));
		}

		for (int dim = 0; dim < Ndim; ++dim) {
			coords_h_[dim] =
			    std::make_unique<coord_table_type>(amrex::Array<int, 1>{0}, amrex::Array<int, 1>{sizes_[dim] - 1}, amrex::The_Pinned_Arena());
			coords_d_[dim] = std::make_unique<coord_table_type>(amrex::Array<int, 1>{0}, amrex::Array<int, 1>{sizes_[dim] - 1});

			if (spacing_types_[dim] == TransformType::linear) {
				// Linear spacing: coordinates computed on-the-fly in GPU code, no need to populate table.
			} else if (spacing_types_[dim] == TransformType::log) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coord_min_[dim] > 0.0 && coord_max_[dim] > 0.0,
								 std::format("Log spacing requires positive bounds for dimension {}", dim));
				coord_min_[dim] = std::log(coord_min_[dim]);
				coord_max_[dim] = std::log(coord_max_[dim]);
			} else if (spacing_types_[dim] == TransformType::fast_log) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coord_min_[dim] > 0.0 && coord_max_[dim] > 0.0,
								 std::format("Fast log spacing requires positive bounds for dimension {}", dim));
				coord_min_[dim] = FastMath::lg(coord_min_[dim]);
				coord_max_[dim] = FastMath::lg(coord_max_[dim]);
			}

			// NOSONAR
			// For future support of irregular spacing
			// } else if (spacing_types_[dim] == TransformType::irregular) {
			// 	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(coords[dim].size()) == sizes_[dim],
			// 					 std::format("Provided coordinates size mismatch for dimension {}! (expected: {}, actual: {})",
			// 						     dim, sizes_[dim], coords[dim].size()));
			// 	auto coord_table = coords_h_[dim]->table();
			// 	for (int i = 0; i < sizes_[dim]; ++i) {
			// 		coord_table(i) = coords[dim][i];
			// 	}
		}

		// Calculate grid spacing (after taking necessary log of the coordinates)
		for (int dim = 0; dim < Ndim; ++dim) {
			dcoord_[dim] = (coord_max_[dim] - coord_min_[dim]) / static_cast<amrex::Real>(sizes_[dim] - 1);
		}

		amrex::Array<int, Ndim> lo{};
		amrex::Array<int, Ndim> hi{};
		for (int dim = 0; dim < Ndim; ++dim) {
			lo[dim] = 0;
			hi[dim] = sizes_[dim] - 1;
		}

		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			data_h_[out_idx] = std::make_unique<data_table_type>(lo, hi, amrex::The_Pinned_Arena());
			data_d_[out_idx] = std::make_unique<data_table_type>(lo, hi);
		}
	}

	template <typename DataType> void fillDataTables(const DataType &data)
	{
		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			auto data_table = data_h_[out_idx]->table();

			if constexpr (Ndim == 1) {
				for (int i = 0; i < sizes_[0]; ++i) {
					data_table(i) = data[out_idx][i];
				}
			} else if constexpr (Ndim == 2) {
				for (int i = 0; i < sizes_[0]; ++i) {
					for (int j = 0; j < sizes_[1]; ++j) {
						data_table(i, j) = data[out_idx][i][j];
					}
				}
			} else if constexpr (Ndim == 3) {
				for (int i = 0; i < sizes_[0]; ++i) {
					for (int j = 0; j < sizes_[1]; ++j) {
						for (int k = 0; k < sizes_[2]; ++k) {
							data_table(i, j, k) = data[out_idx][i][j][k];
						}
					}
				}
			} else if constexpr (Ndim == 4) {
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
		sync_tables_to_device();
	}

	void fillDataTablesFlat(const amrex::Vector<amrex::Real> &flat_data)
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(flat_data.size() == flatDataSize(sizes_),
						 std::format("Flat data size mismatch! (expected: {}, actual: {})", flatDataSize(sizes_), flat_data.size()));

		for (int out_idx = 0; out_idx < Nout; ++out_idx) {
			auto data_table = data_h_[out_idx]->table();

			if constexpr (Ndim == 1) {
				for (int i = 0; i < sizes_[0]; ++i) {
					data_table(i) = flat_data[flatDataIndex(out_idx, sizes_, std::array<int, Ndim>{i})];
				}
			} else if constexpr (Ndim == 2) {
				for (int i = 0; i < sizes_[0]; ++i) {
					for (int j = 0; j < sizes_[1]; ++j) {
						data_table(i, j) = flat_data[flatDataIndex(out_idx, sizes_, std::array<int, Ndim>{i, j})];
					}
				}
			} else if constexpr (Ndim == 3) {
				for (int i = 0; i < sizes_[0]; ++i) {
					for (int j = 0; j < sizes_[1]; ++j) {
						for (int k = 0; k < sizes_[2]; ++k) {
							data_table(i, j, k) = flat_data[flatDataIndex(out_idx, sizes_, std::array<int, Ndim>{i, j, k})];
						}
					}
				}
			} else if constexpr (Ndim == 4) {
				for (int i = 0; i < sizes_[0]; ++i) {
					for (int j = 0; j < sizes_[1]; ++j) {
						for (int k = 0; k < sizes_[2]; ++k) {
							for (int l = 0; l < sizes_[3]; ++l) {
								data_table(i, j, k, l) =
								    flat_data[flatDataIndex(out_idx, sizes_, std::array<int, Ndim>{i, j, k, l})];
							}
						}
					}
				}
			}
		}
		sync_tables_to_device();
	}

	void initializeCommonFlat(const std::array<amrex::Real, Ndim> &x_mins, const std::array<amrex::Real, Ndim> &x_maxs, const std::array<int, Ndim> &n_xs,
				  const std::array<TransformType, Ndim> &spacing_types, const amrex::Vector<amrex::Real> &flat_data)
	{
		initializeStorage(x_mins, x_maxs, n_xs, spacing_types);
		fillDataTablesFlat(flat_data);
	}

	template <typename DataType>
	void initialize_common(const std::array<amrex::Real, Ndim> &x_mins, const std::array<amrex::Real, Ndim> &x_maxs, const std::array<int, Ndim> &n_xs,
			       const std::array<TransformType, Ndim> &spacing_types, const std::array<amrex::Vector<amrex::Real>, Ndim> & /*coords*/,
			       const DataType &data)
	{
		initializeStorage(x_mins, x_maxs, n_xs, spacing_types);
		fillDataTables(data);
	}

	// Backward compatibility wrapper: derive bounds, sizes, and spacing from coords
	template <typename DataType> void initialize_common(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const DataType &data)
	{
		// Validate inputs
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coords[dim].empty(), std::format("Coordinates for dimension {} cannot be empty!", dim));
		}

		// Derive bounds and sizes from coordinates
		std::array<amrex::Real, Ndim> x_mins;
		std::array<amrex::Real, Ndim> x_maxs;
		std::array<int, Ndim> n_xs{};
		std::array<TransformType, Ndim> spacing_types{};

		for (int dim = 0; dim < Ndim; ++dim) {
			n_xs[dim] = static_cast<int>(coords[dim].size());
			x_mins[dim] = coords[dim].front();
			x_maxs[dim] = coords[dim].back();
			// This is a hack for backward compatibility to the cooling table, which uses log input and linear output
			spacing_types[dim] = TransformType::linear;
		}

		// Pass empty coords - for linear spacing they will be computed on-the-fly
		std::array<amrex::Vector<amrex::Real>, Ndim> empty_coords;
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
	// @param output_transform Transform type applied to each output value: linear (no transform), log, or fast_log
	//                      If fast_log, output values are converted to log before storage
	static auto CSVReader(const std::string &file_path, TransformType output_transform) -> DataTable
	{
		static_assert(Ndim >= 1 && Ndim <= 4, "CSVReader supports 1D-4D tables");

		std::array<amrex::Real, Ndim> x_mins{};
		std::array<amrex::Real, Ndim> x_maxs{};
		std::array<int, Ndim> sizes{};
		std::array<TransformType, Ndim> spacing_types_enum{};
		std::array<std::string, Ndim> input_names{};
		std::array<std::string, Nout> output_names{};
		std::array<std::string, Ndim> input_units{};
		std::array<std::string, Nout> output_units{};
		TransformType output_transform_bcast = output_transform;
		amrex::Vector<amrex::Real> flat_data;

		if (amrex::ParallelDescriptor::IOProcessor()) {
			std::ifstream file(file_path);
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file.is_open(), ("Failed to open CSV file: " + file_path).c_str());

			int n_dim = 0;
			int n_out = 0;
			std::array<std::pair<amrex::Real, amrex::Real>, Ndim> coord_bounds{};
			std::array<std::string, Ndim> spacing_types{};

			file >> n_dim;
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    n_dim == Ndim, std::format("CSV file dimension mismatch! File has {} dimensions, but DataTable is {}-dimensional", n_dim, Ndim));

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
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sizes[dim] > 0,
									 std::format("Invalid dimension size {} for dimension {}", sizes[dim], dim));
				}
			}

			file >> n_out;
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    n_out == Nout, std::format("CSV file output dimension mismatch! File has {} outputs, but DataTable expects {}", n_out, Nout));

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

			std::array<amrex::Real, Ndim> xlo_metadata{};
			std::array<amrex::Real, Ndim> xhi_metadata{};
			std::array<std::string, Ndim> spacing_metadata{};

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

			for (int dim = 0; dim < Ndim; ++dim) {
				coord_bounds[dim].first = xlo_metadata[dim];
				coord_bounds[dim].second = xhi_metadata[dim];
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coord_bounds[dim].second > coord_bounds[dim].first,
								 std::format("Invalid coordinate bounds for dimension {}: [{}, {}]", dim,
									     coord_bounds[dim].first, coord_bounds[dim].second));

				try {
					spacing_types_enum[dim] = amrex::getEnumCaseInsensitive<TransformType>(spacing_metadata[dim]);
				} catch (const std::runtime_error &) {
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
					    false, std::format("Invalid spacing type '{}' for dimension {}. Must be 'linear', 'log', or 'fast_log'",
							       spacing_metadata[dim], dim));
				}

				x_mins[dim] = coord_bounds[dim].first;
				x_maxs[dim] = coord_bounds[dim].second;
			}

			auto log_ = [output_transform_bcast](amrex::Real x) {
				if (output_transform_bcast == TransformType::fast_log) {
					return FastMath::inverse_pow2(x);
				}
				return std::log(x);
			};

			flat_data.resize(flatDataSize(sizes));

			if constexpr (Ndim == 1) {
				for (int out_idx = 0; out_idx < Nout; ++out_idx) {
					for (int i = 0; i < sizes[0]; ++i) {
						char comma = ' ';
						amrex::Real value = 0.0;
						file >> value;
						if (i < sizes[0] - 1) {
							file >> comma;
						}
						if (output_transform_bcast == TransformType::fast_log || output_transform_bcast == TransformType::log) {
							AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
							    value > 0.0,
							    std::format("log output transform requires positive values, got {} at output {} index {}", value,
									out_idx, i));
							value = log_(value);
						}
						flat_data[flatDataIndex(out_idx, sizes, std::array<int, Ndim>{i})] = value;
					}
				}
			} else if constexpr (Ndim == 2) {
				for (int out_idx = 0; out_idx < Nout; ++out_idx) {
					for (int i2 = 0; i2 < sizes[1]; ++i2) {
						for (int i1 = 0; i1 < sizes[0]; ++i1) {
							char comma = ' ';
							amrex::Real value = 0.0;
							file >> value;
							if (i1 < sizes[0] - 1) {
								file >> comma;
							}
							if (output_transform_bcast == TransformType::fast_log || output_transform_bcast == TransformType::log) {
								AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
								    value > 0.0,
								    std::format(
									"log output transform requires positive values, got {} at output {} index ({}, {})",
									value, out_idx, i1, i2));
								value = log_(value);
							}
							flat_data[flatDataIndex(out_idx, sizes, std::array<int, Ndim>{i1, i2})] = value;
						}
					}
				}
			} else if constexpr (Ndim == 3) {
				for (int out_idx = 0; out_idx < Nout; ++out_idx) {
					for (int i3 = 0; i3 < sizes[2]; ++i3) {
						for (int i2 = 0; i2 < sizes[1]; ++i2) {
							for (int i1 = 0; i1 < sizes[0]; ++i1) {
								char comma = ' ';
								amrex::Real value = 0.0;
								file >> value;
								if (i1 < sizes[0] - 1) {
									file >> comma;
								}
								if (output_transform_bcast == TransformType::fast_log ||
								    output_transform_bcast == TransformType::log) {
									AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
									    value > 0.0, std::format("log output transform requires positive values, got {} at "
												     "output {} index ({}, {}, {})",
												     value, out_idx, i1, i2, i3));
									value = log_(value);
								}
								flat_data[flatDataIndex(out_idx, sizes, std::array<int, Ndim>{i1, i2, i3})] = value;
							}
						}
					}
				}
			} else if constexpr (Ndim == 4) {
				for (int out_idx = 0; out_idx < Nout; ++out_idx) {
					for (int i4 = 0; i4 < sizes[3]; ++i4) {
						for (int i3 = 0; i3 < sizes[2]; ++i3) {
							for (int i2 = 0; i2 < sizes[1]; ++i2) {
								for (int i1 = 0; i1 < sizes[0]; ++i1) {
									char comma = ' ';
									amrex::Real value = 0.0;
									file >> value;
									if (i1 < sizes[0] - 1) {
										file >> comma;
									}
									if (output_transform_bcast == TransformType::fast_log ||
									    output_transform_bcast == TransformType::log) {
										AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
										    value > 0.0,
										    std::format("log output transform requires positive values, got "
												"{} at output {} index ({}, {}, {}, {})",
												value, out_idx, i1, i2, i3, i4));
										value = log_(value);
									}
									flat_data[flatDataIndex(out_idx, sizes, std::array<int, Ndim>{i1, i2, i3, i4})] = value;
								}
							}
						}
					}
				}
			}
		}

		bcastArray(sizes);
		bcastArray(x_mins);
		bcastArray(x_maxs);
		bcastTransformTypes(spacing_types_enum);
		bcastStringArray(input_names);
		bcastStringArray(output_names);
		bcastStringArray(input_units);
		bcastStringArray(output_units);
		bcastTransformType(output_transform_bcast);
		bcastVector(flat_data);

		DataTable table;
		table.initializeCommonFlat(x_mins, x_maxs, sizes, spacing_types_enum, flat_data);
		std::array<TransformType, Nout> output_transforms_arr{};
		output_transforms_arr.fill(output_transform_bcast);
		table.setMetadata(input_names, output_names, input_units, output_units, output_transforms_arr);
		return table;
	}

	// H5Reader: Read an n-dimensional DataTable from an HDF5 group following the standard format.
	//
	// The group must contain the following attributes:
	//   Ndim (int), Nout (int), Nx (int[Ndim]), xlo (double[Ndim]), xhi (double[Ndim]),
	//   spacing (str[Ndim]: "linear"/"log"/"fast_log"),
	//   input_names, output_names, input_units, output_units (str arrays, optional)
	//
	// The group must contain a dataset named "data" of shape [Nout, Nx[0], Nx[1], ...] in row-major order.
	// If a "grids" subgroup exists it is ignored (irregular grids are not supported; use linear/log/fast_log spacing).
	//
	// All data is broadcast to non-IO MPI ranks automatically.
	static auto H5Reader(const std::string &file_path, const std::string &group_name = "tab1", std::array<TransformType, Nout> output_transforms = {})
	    -> DataTable
	{
		static_assert(Ndim >= 1 && Ndim <= 4, "H5Reader supports 1D-4D tables");

		std::array<int, Ndim> sizes{};
		std::array<amrex::Real, Ndim> x_mins{};
		std::array<amrex::Real, Ndim> x_maxs{};
		std::array<TransformType, Ndim> spacing_types{};
		std::array<std::string, Ndim> input_names{};
		std::array<std::string, Nout> output_names{};
		std::array<std::string, Ndim> input_units{};
		std::array<std::string, Nout> output_units{};
		amrex::Vector<amrex::Real> flat_data;

		if (amrex::ParallelDescriptor::IOProcessor()) {
			herr_t status = 0;
			herr_t const h5_error = -1;

			hid_t const file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file_id >= 0, ("Failed to open HDF5 file: " + file_path).c_str());

			hid_t const group_id = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(group_id >= 0, ("Failed to open HDF5 group: " + group_name).c_str());

			// Read and validate Ndim
			{
				int file_ndim = 0;
				hid_t const attr_id = H5Aopen(group_id, "Ndim", H5P_DEFAULT);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(attr_id >= 0, "Failed to open Ndim attribute");
				status = H5Aread(attr_id, H5T_NATIVE_INT, &file_ndim);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read Ndim attribute");
				H5Aclose(attr_id);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
				    file_ndim == Ndim,
				    std::format("Ndim mismatch in group '{}': file has {}, DataTable expects {}", group_name, file_ndim, Ndim));
			}

			// Read and validate Nout
			{
				int file_nout = 0;
				hid_t const attr_id = H5Aopen(group_id, "Nout", H5P_DEFAULT);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(attr_id >= 0, "Failed to open Nout attribute");
				status = H5Aread(attr_id, H5T_NATIVE_INT, &file_nout);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read Nout attribute");
				H5Aclose(attr_id);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
				    file_nout == Nout,
				    std::format("Nout mismatch in group '{}': file has {}, DataTable expects {}", group_name, file_nout, Nout));
			}

			// Read Nx
			{
				hid_t const attr_id = H5Aopen(group_id, "Nx", H5P_DEFAULT);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(attr_id >= 0, "Failed to open Nx attribute");
				{
					hid_t const space_id = H5Aget_space(attr_id);
					hssize_t const nelem = H5Sget_simple_extent_npoints(space_id);
					H5Sclose(space_id);
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nelem == static_cast<hssize_t>(Ndim),
									 std::format("Nx attribute has {} elements, expected {}", nelem, Ndim));
				}
				status = H5Aread(attr_id, H5T_NATIVE_INT, sizes.data());
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read Nx attribute");
				H5Aclose(attr_id);
				for (int dim = 0; dim < Ndim; ++dim) {
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sizes[dim] > 0, std::format("Invalid Nx[{}] = {}", dim, sizes[dim]));
				}
			}

			// Read xlo and xhi
			{
				std::array<double, Ndim> xlo_d{};
				std::array<double, Ndim> xhi_d{};
				hid_t attr_id = H5Aopen(group_id, "xlo", H5P_DEFAULT);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(attr_id >= 0, "Failed to open xlo attribute");
				{
					hid_t const space_id = H5Aget_space(attr_id);
					hssize_t const nelem = H5Sget_simple_extent_npoints(space_id);
					H5Sclose(space_id);
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nelem == static_cast<hssize_t>(Ndim),
									 std::format("xlo attribute has {} elements, expected {}", nelem, Ndim));
				}
				status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, xlo_d.data());
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read xlo attribute");
				H5Aclose(attr_id);
				attr_id = H5Aopen(group_id, "xhi", H5P_DEFAULT);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(attr_id >= 0, "Failed to open xhi attribute");
				{
					hid_t const space_id = H5Aget_space(attr_id);
					hssize_t const nelem = H5Sget_simple_extent_npoints(space_id);
					H5Sclose(space_id);
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nelem == static_cast<hssize_t>(Ndim),
									 std::format("xhi attribute has {} elements, expected {}", nelem, Ndim));
				}
				status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, xhi_d.data());
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read xhi attribute");
				H5Aclose(attr_id);
				for (int dim = 0; dim < Ndim; ++dim) {
					x_mins[dim] = static_cast<amrex::Real>(xlo_d[dim]);
					x_maxs[dim] = static_cast<amrex::Real>(xhi_d[dim]);
				}
			}

			// Read spacing (fixed-size string array written by numpy dtype='S')
			{
				hid_t const attr_id = H5Aopen(group_id, "spacing", H5P_DEFAULT);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(attr_id >= 0, "Failed to open spacing attribute");
				{
					hid_t const space_id = H5Aget_space(attr_id);
					hssize_t const nelem = H5Sget_simple_extent_npoints(space_id);
					H5Sclose(space_id);
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nelem == static_cast<hssize_t>(Ndim),
									 std::format("spacing attribute has {} elements, expected {}", nelem, Ndim));
				}
				hid_t const atype = H5Aget_type(attr_id);
				hid_t const native_type = H5Tget_native_type(atype, H5T_DIR_ASCEND);
				size_t const type_size = H5Tget_size(native_type);
				std::vector<char> raw(static_cast<std::size_t>(Ndim) * type_size);
				H5Aread(attr_id, native_type, raw.data());
				H5Tclose(native_type);
				H5Tclose(atype);
				H5Aclose(attr_id);
				for (int i = 0; i < Ndim; ++i) {
					std::string s(raw.data() + static_cast<std::size_t>(i) * type_size, type_size);
					s.erase(std::find(s.begin(), s.end(), '\0'), s.end());
					try {
						spacing_types[i] = amrex::getEnumCaseInsensitive<TransformType>(s);
					} catch (const std::runtime_error &) {
						AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
						    false, std::format("Invalid spacing '{}' for dim {}. Must be 'linear', 'log', or 'fast_log'", s, i));
					}
				}
			}

			// Helper: read a fixed-size numpy-style string array attribute
			auto read_str_array = [&](const char *attr_name, int count) -> std::vector<std::string> {
				std::vector<std::string> result(static_cast<std::size_t>(count));
				if (H5Aexists(group_id, attr_name) <= 0) {
					return result;
				}
				hid_t const aid = H5Aopen(group_id, attr_name, H5P_DEFAULT);
				if (aid < 0) {
					return result;
				}
				{
					hid_t const space_id = H5Aget_space(aid);
					hssize_t const nelem = H5Sget_simple_extent_npoints(space_id);
					H5Sclose(space_id);
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nelem == static_cast<hssize_t>(count),
									 std::format("{} attribute has {} elements, expected {}", attr_name, nelem, count));
				}
				hid_t const atype_id = H5Aget_type(aid);
				hid_t const native_type_id = H5Tget_native_type(atype_id, H5T_DIR_ASCEND);
				size_t const str_size = H5Tget_size(native_type_id);
				std::vector<char> raw_buf(static_cast<std::size_t>(count) * str_size);
				H5Aread(aid, native_type_id, raw_buf.data());
				H5Tclose(native_type_id);
				H5Tclose(atype_id);
				H5Aclose(aid);
				for (int i = 0; i < count; ++i) {
					std::string s(raw_buf.data() + static_cast<std::size_t>(i) * str_size, str_size);
					s.erase(std::find(s.begin(), s.end(), '\0'), s.end());
					result[static_cast<std::size_t>(i)] = s;
				}
				return result;
			};

			auto in_names = read_str_array("input_names", Ndim);
			auto out_names = read_str_array("output_names", Nout);
			auto in_units = read_str_array("input_units", Ndim);
			auto out_units = read_str_array("output_units", Nout);
			for (int i = 0; i < Ndim; ++i) {
				input_names[i] = in_names[static_cast<std::size_t>(i)];
				input_units[i] = in_units[static_cast<std::size_t>(i)];
			}
			for (int i = 0; i < Nout; ++i) {
				output_names[i] = out_names[static_cast<std::size_t>(i)];
				output_units[i] = out_units[static_cast<std::size_t>(i)];
			}

			// Read 'data' dataset: shape [Nout, Nx[0], Nx[1], ...] row-major
			{
				amrex::Long const total = flatDataSize(sizes);
				hid_t const dset_id = H5Dopen2(group_id, "data", H5P_DEFAULT);
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dset_id >= 0, ("Failed to open 'data' dataset in group: " + group_name).c_str());
				{
					hid_t const space_id = H5Dget_space(dset_id);
					hssize_t const file_total = H5Sget_simple_extent_npoints(space_id);
					H5Sclose(space_id);
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
					    file_total == static_cast<hssize_t>(total),
					    std::format("'data' dataset in group '{}' has {} elements, expected {} (Nout={}, Nx as read)", group_name,
							file_total, total, Nout));
				}
				std::vector<double> raw_data(static_cast<std::size_t>(total));
				status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw_data.data());
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, ("Failed to read 'data' dataset in group: " + group_name).c_str());
				H5Dclose(dset_id);
				// Apply per-output transform (inverse_pow2 for fast_log, log for log).
				// HDF5 stores raw physical values; the transform converts to interpolation space.
				flat_data.resize(static_cast<std::size_t>(total));
				amrex::Long const spatial_size = total / Nout;
				for (amrex::Long k = 0; k < total; ++k) {
					auto const val = static_cast<amrex::Real>(raw_data[static_cast<std::size_t>(k)]);
					int const out_idx = static_cast<int>(k / spatial_size);
					if (output_transforms[static_cast<std::size_t>(out_idx)] == TransformType::fast_log) {
						flat_data[static_cast<std::size_t>(k)] = FastMath::inverse_pow2(val);
					} else if (output_transforms[static_cast<std::size_t>(out_idx)] == TransformType::log) {
						flat_data[static_cast<std::size_t>(k)] = std::log(val);
					} else {
						flat_data[static_cast<std::size_t>(k)] = val;
					}
				}
			}

			H5Gclose(group_id);
			H5Fclose(file_id);
		}

		// Broadcast all data to non-IO ranks
		bcastArray(sizes);
		bcastArray(x_mins);
		bcastArray(x_maxs);
		bcastTransformTypes(spacing_types);
		bcastStringArray(input_names);
		bcastStringArray(output_names);
		bcastStringArray(input_units);
		bcastStringArray(output_units);
		bcastVector(flat_data);

		DataTable table;
		table.initializeCommonFlat(x_mins, x_maxs, sizes, spacing_types, flat_data);
		table.setMetadata(input_names, output_names, input_units, output_units, output_transforms);
		return table;
	}
};

} // namespace quokka

#endif // DATATABLE_HPP_
