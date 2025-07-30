#ifndef DATATABLE_HPP_
#define DATATABLE_HPP_

#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_TableData.H"
#include "math/Interpolate2D.hpp"
#include <array>
#include <memory>
#include <type_traits>

namespace quokka
{

// Structure to hold interpolation indices and normalized coordinates
template <int Ndim>
struct InterpData {
	std::array<int, Ndim> indices{};        // grid indices for each dimension (lower bounds)
	std::array<int, Ndim> upper_indices{};  // upper bound indices for each dimension
	std::array<amrex::Real, Ndim> coords_lower{};  // actual coordinate values at lower grid points
	std::array<amrex::Real, Ndim> coords_upper{};  // actual coordinate values at upper grid points
	std::array<amrex::Real, Ndim> normalized{};    // normalized coordinates in [0,1] for each dimension

	// Default constructor
	AMREX_GPU_HOST_DEVICE InterpData() = default;
};

// GPU-friendly struct containing const table references
template <int Ndim>
struct DataTableGpuConst {
	std::array<amrex::Table1D<const amrex::Real>, Ndim> coords;
	amrex::Table2D<const amrex::Real> data;  // Keep 2D for now, can be generalized later

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
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto find_interpolation_data(const std::array<amrex::Real, Ndim>& point) const -> InterpData<Ndim>
	{
		InterpData<Ndim> interp;

		for (int dim = 0; dim < Ndim; ++dim) {
			// Get table bounds for this dimension - assumes uniform grid spacing
			amrex::Real const coord_start = coords[dim](coords[dim].begin);   // First coordinate
			amrex::Real const coord_end = coords[dim](coords[dim].end - 1);   // Last coordinate

			// Clamp coordinates to valid table bounds (extrapolation not supported)
			amrex::Real clamped_coord = amrex::max(coord_start, amrex::min(point[dim], coord_end));

			// Find grid cell indices containing the point
			// indices are the "lower" indices of the containing hypercube
			interp.indices[dim] = amrex::max(coords[dim].begin, 
				amrex::min(static_cast<int>(std::floor((clamped_coord - coord_start) / dcoord[dim])), 
					coords[dim].end - 1));

			// upper_indices are the "upper" indices (handle boundary case)
			interp.upper_indices[dim] = (interp.indices[dim] == coords[dim].end - 1) ? 
				interp.indices[dim] : interp.indices[dim] + 1;

			// Get actual coordinate values at the grid points
			interp.coords_lower[dim] = coords[dim](interp.indices[dim]);       // Lower coordinate
			interp.coords_upper[dim] = coords[dim](interp.upper_indices[dim]); // Upper coordinate

			// Compute normalized coordinates within the grid cell [0,1]
			// normalized[dim] = 0 at coords_lower[dim], normalized[dim] = 1 at coords_upper[dim]
			if (interp.indices[dim] != interp.upper_indices[dim]) {
				interp.normalized[dim] = (clamped_coord - interp.coords_lower[dim]) / 
					(interp.coords_upper[dim] - interp.coords_lower[dim]);
			} else {
				interp.normalized[dim] = 0.0; // No variation in this dimension (boundary case)
			}
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
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate(const std::array<amrex::Real, Ndim>& point) const -> amrex::Real
	{
		// Part 1: Find interpolation indices and normalized coordinates
		InterpData<Ndim> const interp = find_interpolation_data(point);

		// Part 2: Perform n-dimensional interpolation
		return interpolate_from_indices(interp);
	}

	// Backward compatibility wrapper for 2D
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate(amrex::Real x, amrex::Real y) const -> amrex::Real
	{
		static_assert(Ndim == 2, "This overload only works for 2D tables");
		return interpolate(std::array<amrex::Real, 2>{x, y});
	}

private:
	/// @brief Helper for n-dimensional interpolation
	///
	/// This function performs n-linear interpolation. Currently optimized for 2D case.
	/// Can be extended to support true recursive n-dimensional interpolation in the future.
	///
	/// @param interp Interpolation data containing indices and normalized coordinates
	/// @return Interpolated value
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate_from_indices(const InterpData<Ndim>& interp) const -> amrex::Real
	{
		if constexpr (Ndim == 2) {
			// Optimized 2D case (bilinear interpolation)
			// Note: data table is currently 2D only, so we use direct indexing
			amrex::Real const z1 = data(interp.indices[0], interp.indices[1]);
			amrex::Real const z2 = data(interp.upper_indices[0], interp.indices[1]);
			amrex::Real const z3 = data(interp.indices[0], interp.upper_indices[1]);
			amrex::Real const z4 = data(interp.upper_indices[0], interp.upper_indices[1]);

			// f(h, v) = (1 - v)((1 - h) z1 + h z2) + v((1 - h) z3 + h z4)
			amrex::Real const value = (1.0 - interp.normalized[1]) * ((1.0 - interp.normalized[0]) * z1 + interp.normalized[0] * z2) + 
				interp.normalized[1] * ((1.0 - interp.normalized[0]) * z3 + interp.normalized[0] * z4);
			
			AMREX_ASSERT(!std::isnan(value));
			return value;
		} else {
			// General n-dimensional case would go here
			// For now, only 2D is supported due to data table limitations
			static_assert(Ndim == 2, "Only 2D interpolation is currently supported due to data table structure");
			return 0.0; // This line should never be reached
		}
	}
};

// Generic n-dimensional data table class
template <int Ndim>
class DataTable
{
      public:
	// Default constructor
	DataTable() = default;

	// Constructor with data - specialized for 2D
	template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
	DataTable(const amrex::Vector<amrex::Real> &x_coords, const amrex::Vector<amrex::Real> &y_coords,
		  const amrex::Vector<amrex::Vector<amrex::Real>> &data)
	{
		initialize(x_coords, y_coords, data);
	}

	// Constructor with coordinate arrays and data - general n-dimensional interface
	DataTable(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords,
		  const amrex::Vector<amrex::Vector<amrex::Real>> &data)
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

	// Initialize from vectors - specialized for 2D (backward compatibility)
	template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
	void initialize(const amrex::Vector<amrex::Real> &x_coords, const amrex::Vector<amrex::Real> &y_coords,
			const amrex::Vector<amrex::Vector<amrex::Real>> &data)
	{
		std::array<amrex::Vector<amrex::Real>, 2> coord_arrays = {x_coords, y_coords};
		initialize(coord_arrays, data);
	}

	// Initialize from coordinate arrays - general n-dimensional interface
	void initialize(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords,
			const amrex::Vector<amrex::Vector<amrex::Real>> &data)
	{
		static_assert(Ndim == 2, "Currently only 2D tables are supported");
		
		// Validate inputs
		for (int dim = 0; dim < Ndim; ++dim) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coords[dim].empty(), "Coordinates cannot be empty!");
		}
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!data.empty(), "Data cannot be empty!");
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(data.size() == coords[0].size(), "Data first dimension must match first coordinate size!");

		// Store sizes
		for (int dim = 0; dim < Ndim; ++dim) {
			sizes_[dim] = static_cast<int>(coords[dim].size());
		}

		// Verify data dimensions
		for (const auto &row : data) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(row.size() == coords[1].size(), "All data rows must match second coordinate size!");
		}

		// Store coordinate bounds (assuming ascending order) and calculate grid spacing
		for (int dim = 0; dim < Ndim; ++dim) {
			coord_min_[dim] = coords[dim].front();
			coord_max_[dim] = coords[dim].back();
			dcoord_[dim] = (coord_max_[dim] - coord_min_[dim]) / static_cast<amrex::Real>(sizes_[dim] - 1);
		}

		// Create coordinate tables
		for (int dim = 0; dim < Ndim; ++dim) {
			coords_[dim] = std::make_unique<amrex::TableData<amrex::Real, 1>>(
			    amrex::Array<int, 1>{0}, amrex::Array<int, 1>{sizes_[dim] - 1}, amrex::The_Pinned_Arena());
			auto coord_table = coords_[dim]->table();
			for (int i = 0; i < sizes_[dim]; ++i) {
				coord_table(i) = coords[dim][i];
			}
		}

		// All above is generic to arbitrary Ndim, but this part is only for 2D
		{
			// Create 2D data table
			data_ = std::make_unique<amrex::TableData<amrex::Real, 2>>(amrex::Array<int, 2>{0, 0}, amrex::Array<int, 2>{sizes_[0] - 1, sizes_[1] - 1},
											amrex::The_Pinned_Arena());
			auto data_table = data_->table();

			// Copy data (input is data[i][j], table is accessed as table(i,j))
			for (int i = 0; i < sizes_[0]; ++i) {
				for (int j = 0; j < sizes_[1]; ++j) {
					data_table(i, j) = data[i][j];
				}
			}
		}
	}

	// Get GPU-friendly const tables
	[[nodiscard]] auto const_tables() const -> DataTableGpuConst<Ndim>
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_initialized(), "DataTable must be initialized before getting const tables!");

		if constexpr (Ndim == 2) {
			DataTableGpuConst<Ndim> tables{
			    {coords_[0]->const_table(), coords_[1]->const_table()}, // coords array
			    data_->const_table(),                                   // data
			    coord_min_,                                             // coord_min array
			    coord_max_,                                             // coord_max array
			    dcoord_,                                                // dcoord array
			    sizes_                                                  // sizes array
			};
			return tables;
		} else {
			static_assert(Ndim == 2, "Only 2D tables are currently supported");
		}
	}

	// Check if table is initialized
	[[nodiscard]] auto is_initialized() const -> bool
	{
		if constexpr (Ndim == 2) {
			return (coords_[0] != nullptr && coords_[1] != nullptr && data_ != nullptr);
		} else {
			// For general case, check all coordinate arrays
			for (int dim = 0; dim < Ndim; ++dim) {
				if (coords_[dim] == nullptr) {
					return false;
				}
			}
			return (data_ != nullptr);
		}
	}

	// Get dimension sizes
	[[nodiscard]] auto sizes() const -> std::array<int, Ndim>
	{
		return sizes_;
	}
	
	// Get size for specific dimension
	[[nodiscard]] auto size(int dim) const -> int
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dim >= 0 && dim < Ndim, "Dimension index out of bounds!");
		return sizes_[dim];
	}

	// Backward compatibility methods for 2D
	template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
	[[nodiscard]] auto x_size() const -> int { return sizes_[0]; }

	template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
	[[nodiscard]] auto y_size() const -> int { return sizes_[1]; }

      private:
	std::array<std::unique_ptr<amrex::TableData<amrex::Real, 1>>, Ndim> coords_;
	std::unique_ptr<amrex::TableData<amrex::Real, 2>> data_;  // Still 2D for now

	std::array<amrex::Real, Ndim> coord_min_{};
	std::array<amrex::Real, Ndim> coord_max_{};

	// Precomputed grid spacing for optimization
	std::array<amrex::Real, Ndim> dcoord_{};

	std::array<int, Ndim> sizes_{};
};

} // namespace quokka

#endif // DATATABLE_HPP_