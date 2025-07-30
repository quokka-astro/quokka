#ifndef DATATABLE_HPP_
#define DATATABLE_HPP_

#include "AMReX.H"
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

	// Original interpolation method (for backward compatibility) - specialized for 2D
	template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate0(amrex::Real x, amrex::Real y) const -> amrex::Real
	{
		// Clamp x and y to valid bounds
		x = amrex::max(coord_min[0], amrex::min(x, coord_max[0]));
		y = amrex::max(coord_min[1], amrex::min(y, coord_max[1]));

		return interpolate2d(x, y, coords[0], coords[1], data);
	}

	/// @brief Find interpolation indices and normalized coordinates for bilinear interpolation
	///
	/// This function locates the grid cell containing point (x,y) and computes normalized
	/// coordinates within that cell for efficient bilinear interpolation.
	///
	/// @param x Physical x-coordinate to interpolate at
	/// @param y Physical y-coordinate to interpolate at
	/// @return InterpData structure containing grid indices, coordinates, and normalized params
	///
	/// Grid Layout and Coordinate Mapping:
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
	template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto find_interpolation_data(amrex::Real x, amrex::Real y) const -> InterpData<2>
	{
		InterpData<2> interp;

		// Get table bounds - assumes uniform grid spacing
		amrex::Real const xi = coords[0](coords[0].begin);   // First x coordinate
		amrex::Real const xf = coords[0](coords[0].end - 1); // Last x coordinate
		amrex::Real const yi = coords[1](coords[1].begin);   // First y coordinate
		amrex::Real const yf = coords[1](coords[1].end - 1); // Last y coordinate

		// Clamp coordinates to valid table bounds (extrapolation not supported)
		x = amrex::max(xi, amrex::min(x, xf));
		y = amrex::max(yi, amrex::min(y, yf));

		// Find grid cell indices containing the point (x,y)
		// indices are the "lower-left" indices of the containing cell
		interp.indices[0] = amrex::max(coords[0].begin, amrex::min(static_cast<int>(std::floor((x - xi) / dcoord[0])), coords[0].end - 1));
		interp.indices[1] = amrex::max(coords[1].begin, amrex::min(static_cast<int>(std::floor((y - yi) / dcoord[1])), coords[1].end - 1));

		// upper_indices are the "upper-right" indices (handle boundary case)
		interp.upper_indices[0] = (interp.indices[0] == coords[0].end - 1) ? interp.indices[0] : interp.indices[0] + 1;
		interp.upper_indices[1] = (interp.indices[1] == coords[1].end - 1) ? interp.indices[1] : interp.indices[1] + 1;

		// Get actual coordinate values at the four grid points
		interp.coords_lower[0] = coords[0](interp.indices[0]);       // Left x-coordinate
		interp.coords_upper[0] = coords[0](interp.upper_indices[0]); // Right x-coordinate
		interp.coords_lower[1] = coords[1](interp.indices[1]);       // Bottom y-coordinate
		interp.coords_upper[1] = coords[1](interp.upper_indices[1]); // Top y-coordinate

		// Compute normalized coordinates within the grid cell [0,1] x [0,1]
		// normalized[0] = 0 at coords_lower[0], normalized[0] = 1 at coords_upper[0]
		// normalized[1] = 0 at coords_lower[1], normalized[1] = 1 at coords_upper[1]
		if (interp.indices[0] != interp.upper_indices[0]) {
			interp.normalized[0] = (x - interp.coords_lower[0]) / (interp.coords_upper[0] - interp.coords_lower[0]);
		} else {
			interp.normalized[0] = 0.0; // No variation in x direction (boundary case)
		}

		if (interp.indices[1] != interp.upper_indices[1]) {
			interp.normalized[1] = (y - interp.coords_lower[1]) / (interp.coords_upper[1] - interp.coords_lower[1]);
		} else {
			interp.normalized[1] = 0.0; // No variation in y direction (boundary case)
		}

		return interp;
	}

	// Convenience method: find interpolation data and compute value in one call
	template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate(amrex::Real x, amrex::Real y) const -> amrex::Real
	{
		// Part 1: Find interpolation indices and normalized coordinates
		InterpData<2> const interp = find_interpolation_data(x, y);

		// Part 2: Compute interpolated value using precomputed indices and normalized coordinates
		amrex::Real const z1 = data(interp.indices[0], interp.indices[1]);
		amrex::Real const z2 = data(interp.upper_indices[0], interp.indices[1]);
		amrex::Real const z3 = data(interp.indices[0], interp.upper_indices[1]);
		amrex::Real const z4 = data(interp.upper_indices[0], interp.upper_indices[1]);

		// f(h, v) = (1 - v)((1 - h) z1 + h z2) + v((1 - h) z3 + h z4)
		amrex::Real const value = (1.0 - interp.normalized[1]) * ((1.0 - interp.normalized[0]) * z1 + interp.normalized[0] * z2) + interp.normalized[1] * ((1.0 - interp.normalized[0]) * z3 + interp.normalized[0] * z4);
		AMREX_ASSERT(!std::isnan(value));

		return value;
	}
};

// Generic 2D data table class
class DataTable
{
      public:
	// Default constructor
	DataTable() = default;

	// Constructor with data
	DataTable(const amrex::Vector<amrex::Real> &x_coords, const amrex::Vector<amrex::Real> &y_coords,
		  const amrex::Vector<amrex::Vector<amrex::Real>> &data);

	// Destructor
	~DataTable() = default;

	// Move constructor and assignment
	DataTable(DataTable &&) = default;
	auto operator=(DataTable &&) -> DataTable & = default;

	// Delete copy constructor and assignment (expensive operations)
	DataTable(const DataTable &) = delete;
	auto operator=(const DataTable &) -> DataTable & = delete;

	// Initialize from vectors
	void initialize(const amrex::Vector<amrex::Real> &x_coords, const amrex::Vector<amrex::Real> &y_coords,
			const amrex::Vector<amrex::Vector<amrex::Real>> &data);

	// Get GPU-friendly const tables
	[[nodiscard]] auto const_tables() const -> DataTableGpuConst<2>;

	// Check if table is initialized
	[[nodiscard]] auto is_initialized() const -> bool;

	// Get dimensions
	[[nodiscard]] auto x_size() const -> int;
	[[nodiscard]] auto y_size() const -> int;

      private:
	std::unique_ptr<amrex::TableData<amrex::Real, 1>> x_coords_;
	std::unique_ptr<amrex::TableData<amrex::Real, 1>> y_coords_;
	std::unique_ptr<amrex::TableData<amrex::Real, 2>> data_;

	amrex::Real x_min_ = 0.0;
	amrex::Real x_max_ = 0.0;
	amrex::Real y_min_ = 0.0;
	amrex::Real y_max_ = 0.0;

	// Precomputed grid spacing for optimization
	amrex::Real dx_ = 0.0;
	amrex::Real dy_ = 0.0;

	int x_size_ = 0;
	int y_size_ = 0;
};

} // namespace quokka

#endif // DATATABLE_HPP_