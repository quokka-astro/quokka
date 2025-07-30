#ifndef DATATABLE_HPP_
#define DATATABLE_HPP_

#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_TableData.H"

#include <array>
#include <memory>
#include <type_traits>

namespace quokka
{

// Structure to hold interpolation indices and normalized coordinates
template <int Ndim> struct InterpData {
	std::array<int, Ndim> indices{}; // grid indices for each dimension (lower bounds)
	// std::array<int, Ndim> upper_indices{};  // upper bound indices for each dimension
	// std::array<amrex::Real, Ndim> coords_lower{};  // actual coordinate values at lower grid points
	// std::array<amrex::Real, Ndim> coords_upper{};  // actual coordinate values at upper grid points
	std::array<amrex::Real, Ndim> normalized{}; // normalized coordinates in [0,1] for each dimension

	// Default constructor
	AMREX_GPU_HOST_DEVICE InterpData() = default;
};

// GPU-friendly struct containing const table references
template <int Ndim> struct DataTableGpuConst {
	static_assert(Ndim >= 1 && Ndim <= 4, "Only 1D-4D interpolation is supported");

	std::array<amrex::Table1D<const amrex::Real>, Ndim> coords;
	// Conditional type for arbitrary dimensions (1-4)
	using data_table_type =
	    std::conditional_t<Ndim == 1, amrex::Table1D<const amrex::Real>,
			       std::conditional_t<Ndim == 2, amrex::Table2D<const amrex::Real>,
						  std::conditional_t<Ndim == 3, amrex::Table3D<const amrex::Real>, amrex::Table4D<const amrex::Real>>>>;
	data_table_type data;

	std::array<amrex::Real, Ndim> coord_min{};
	std::array<amrex::Real, Ndim> coord_max{};

	// Precomputed grid spacing for optimization
	std::array<amrex::Real, Ndim> dcoord{};

	std::array<int, Ndim> sizes{};

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
			// Get table bounds for this dimension - assumes uniform grid spacing
			amrex::Real const coord_start = coords[dim](coords[dim].begin); // First coordinate, begin = 0
			amrex::Real const coord_end = coords[dim](coords[dim].end - 1); // Last coordinate, end = size

			// Clamp coordinates to valid table bounds (extrapolation not supported)
			amrex::Real clamped_coord = amrex::max(coord_start, amrex::min(point[dim], coord_end));

			// Find grid cell indices containing the point
			// indices are the "lower" indices of the containing hypercube
			interp.indices[dim] = amrex::max(
			    coords[dim].begin, amrex::min(static_cast<int>(std::floor((clamped_coord - coord_start) / dcoord[dim])), coords[dim].end - 1));

			// if indices is end - 1, then set indices to end - 2 (so that upper_indices is end - 1, the last index)
			if (interp.indices[dim] == coords[dim].end - 1) {
				interp.indices[dim] = coords[dim].end - 2;
			}

			// This can be greater than 1 if the point is outside the grid and not clamped
			interp.normalized[dim] = (clamped_coord - coords[dim](interp.indices[dim])) / dcoord[dim];
		}

		return interp;
	}

	/// @brief Perform n-dimensional linear interpolation
	///
	/// This method performs n-linear interpolation by recursively interpolating
	/// along each dimension. For 2D this becomes bilinear, for 3D trilinear, etc.
	///
	/// @param point Physical coordinates to interpolate at (size Ndim)
	/// @return Interpolated value
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate(const std::array<amrex::Real, Ndim> &point) const -> amrex::Real
	{
		// Part 1: Find interpolation indices and normalized coordinates
		InterpData<Ndim> const interp = find_interpolation_data(point);

		// Part 2: Perform n-dimensional interpolation
		return interpolate_from_indices(interp);
	}

      private:
	/// @brief Helper for n-dimensional interpolation
	///
	/// This function performs n-linear interpolation for 1D-4D cases.
	/// Supports linear, bilinear, trilinear, and quadrilinear interpolation.
	///
	/// @param interp Interpolation data containing indices and normalized coordinates
	/// @return Interpolated value
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate_from_indices(const InterpData<Ndim> &interp) const -> amrex::Real
	{
		auto const dataView_ = data;

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

			// Spiner formula (https://github.com/lanl/spiner/blob/main/spiner/databox.hpp):
			// const amrex::Real value = (w2[0] * (w1[0] * dataView_(ix2, ix1) + w1[1] * dataView_(ix2, ix1 + 1)) +
			// 			   w2[1] * (w1[0] * dataView_(ix2 + 1, ix1) + w1[1] * dataView_(ix2 + 1, ix1 + 1)));
			// Need to swap indices because Spiner uses (ix2, ix1) indexing, but we use (ix1, ix2) indexing
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
		} else {
			static_assert(false, "Only 1D-4D interpolation is supported");
			return 0.0; // This line should never be reached
		}
	}
};

// Generic n-dimensional data table class
template <int Ndim> class DataTable
{
      private:
	std::array<std::unique_ptr<amrex::TableData<amrex::Real, 1>>, Ndim> coords_;
	std::unique_ptr<amrex::TableData<amrex::Real, Ndim>> data_; // Now supports arbitrary dimensions

	std::array<amrex::Real, Ndim> coord_min_{};
	std::array<amrex::Real, Ndim> coord_max_{};

	// Precomputed grid spacing for optimization
	std::array<amrex::Real, Ndim> dcoord_{};

	std::array<int, Ndim> sizes_{};

      public:
	// Default constructor
	DataTable() = default;

	// Constructor with coordinate arrays and data - general n-dimensional interface
	DataTable(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const amrex::Vector<amrex::Vector<amrex::Real>> &data)
	{
		initialize(coords, data);
	}

	// Destructor
	~DataTable() = default;

	// Move constructor and assignment
	DataTable(DataTable &&) = default;
	auto operator=(DataTable &&) -> DataTable & = default;

	// Delete copy constructor and assignment (expensive operations)
	DataTable(const DataTable &) = delete;
	auto operator=(const DataTable &) -> DataTable & = delete;

	// Initialize from coordinate arrays - general n-dimensional interface
	// For now, this implementation still expects 2D input data for backward compatibility
	// TODO(cche): Extend to support true n-dimensional input data formats
	void initialize(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const amrex::Vector<amrex::Vector<amrex::Real>> &data)
	{
		static_assert(Ndim >= 1 && Ndim <= 4, "Only 1D-4D tables are supported");

		// Validate inputs
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coords[dim].empty(), "Coordinates cannot be empty!");
		}
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!data.empty(), "Data cannot be empty!");

		// For 2D case, maintain backward compatibility with existing data format
		if constexpr (Ndim == 2) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(data.size() == coords[0].size(), "Data first dimension must match first coordinate size!");
			// Verify data dimensions
			for (const auto &row : data) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(row.size() == coords[1].size(), "All data rows must match second coordinate size!");
			}
		} else {
			// For non-2D cases, you'll need to implement appropriate data format validation
			// This is a placeholder - extend as needed for your specific use cases
			static_assert(Ndim == 2, "Non-2D data initialization not yet implemented. Please extend this method for your use case.");
		}

		// Store sizes
		for (int dim = 0; dim < Ndim; ++dim) {
			sizes_[dim] = static_cast<int>(coords[dim].size());
		}

		// Store coordinate bounds (assuming ascending order) and calculate grid spacing
		for (int dim = 0; dim < Ndim; ++dim) {
			coord_min_[dim] = coords[dim].front();
			coord_max_[dim] = coords[dim].back();
			dcoord_[dim] = (coord_max_[dim] - coord_min_[dim]) / static_cast<amrex::Real>(sizes_[dim] - 1);
		}

		// Create coordinate tables
		for (int dim = 0; dim < Ndim; ++dim) {
			coords_[dim] = std::make_unique<amrex::TableData<amrex::Real, 1>>(amrex::Array<int, 1>{0}, amrex::Array<int, 1>{sizes_[dim] - 1},
											  amrex::The_Pinned_Arena());
			auto coord_table = coords_[dim]->table();
			for (int i = 0; i < sizes_[dim]; ++i) {
				coord_table(i) = coords[dim][i];
			}
		}

		// Create n-dimensional data table
		amrex::Array<int, Ndim> lo{};
		amrex::Array<int, Ndim> hi{};
		for (int dim = 0; dim < Ndim; ++dim) {
			lo[dim] = 0;
			hi[dim] = sizes_[dim] - 1;
		}

		data_ = std::make_unique<amrex::TableData<amrex::Real, Ndim>>(lo, hi, amrex::The_Pinned_Arena());
		auto data_table = data_->table();

		// Copy data - for now only handle 2D input format
		if constexpr (Ndim == 2) {
			// Copy data (input is data[i][j], table is accessed as table(i,j))
			for (int i = 0; i < sizes_[0]; ++i) {
				for (int j = 0; j < sizes_[1]; ++j) {
					data_table(i, j) = data[i][j];
				}
			}
		} else {
			// For other dimensions, you'll need to implement appropriate data copying
			// This is a placeholder - extend as needed for your specific use cases
			static_assert(Ndim == 2, "Non-2D data copying not yet implemented. Please extend this method for your use case.");
		}
	}

	// Get GPU-friendly const tables
	[[nodiscard]] auto const_tables() const -> DataTableGpuConst<Ndim>
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_initialized(), "DataTable must be initialized before getting const tables!");

		std::array<amrex::Table1D<const amrex::Real>, Ndim> coord_tables{};
		for (int i = 0; i < Ndim; ++i) {
			coord_tables[i] = coords_[i]->const_table();
		}

		DataTableGpuConst<Ndim> tables{
		    coord_tables,
		    data_->const_table(), // data
		    coord_min_,		  // coord_min array
		    coord_max_,		  // coord_max array
		    dcoord_,		  // dcoord array
		    sizes_		  // sizes array
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
		return (data_ != nullptr);
	}

	// Get dimension sizes
	[[nodiscard]] auto sizes() const -> std::array<int, Ndim> { return sizes_; }

	// Get size for specific dimension
	[[nodiscard]] auto size(int dim) const -> int
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dim >= 0 && dim < Ndim, "Dimension index out of bounds!");
		return sizes_[dim];
	}
};

} // namespace quokka

#endif // DATATABLE_HPP_