#ifndef DATATABLE_HPP_
#define DATATABLE_HPP_

#include "AMReX.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_TableData.H"
#include "math/Interpolate2D.hpp"
#include <memory>

namespace quokka
{

// Structure to hold interpolation indices and normalized coordinates
struct InterpData {
	int ix{}, iy{}, iix{}, iiy{};	    // grid indices
	amrex::Real x1{}, x2{}, y1{}, y2{}; // actual coordinate values at grid points
	amrex::Real h{}, v{};	    // normalized coordinates: h = (x-x1)/(x2-x1), v = (y-y1)/(y2-y1)

	// Default constructor
	AMREX_GPU_HOST_DEVICE InterpData() = default;
};

// GPU-friendly struct containing const table references
struct DataTableGpuConst {
	amrex::Table1D<const amrex::Real> x_coords;
	amrex::Table1D<const amrex::Real> y_coords;
	amrex::Table2D<const amrex::Real> data;

	amrex::Real x_min{};
	amrex::Real x_max{};
	amrex::Real y_min{};
	amrex::Real y_max{};

	int x_size{};
	int y_size{};

	// Original interpolation method (for backward compatibility)
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate0(amrex::Real x, amrex::Real y) const -> amrex::Real
	{
		// Clamp x and y to valid bounds
		x = amrex::max(x_min, amrex::min(x, x_max));
		y = amrex::max(y_min, amrex::min(y, y_max));

		return interpolate2d(x, y, x_coords, y_coords, data);
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
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto find_interpolation_data(amrex::Real x, amrex::Real y) const -> InterpData
	{
		InterpData interp;

		// Get table bounds - assumes uniform grid spacing
		amrex::Real xi = x_coords(x_coords.begin);   // First x coordinate
		amrex::Real xf = x_coords(x_coords.end - 1); // Last x coordinate
		amrex::Real yi = y_coords(y_coords.begin);   // First y coordinate
		amrex::Real yf = y_coords(y_coords.end - 1); // Last y coordinate

		// Compute uniform grid spacing
		amrex::Real dx = (xf - xi) / static_cast<amrex::Real>(x_coords.end - x_coords.begin - 1);
		amrex::Real dy = (yf - yi) / static_cast<amrex::Real>(y_coords.end - y_coords.begin - 1);

		// Clamp coordinates to valid table bounds (extrapolation not supported)
		x = amrex::max(xi, amrex::min(x, xf));
		y = amrex::max(yi, amrex::min(y, yf));

		// Find grid cell indices containing the point (x,y)
		// ix, iy are the "lower-left" indices of the containing cell
		interp.ix = amrex::max(x_coords.begin, amrex::min(static_cast<int>(std::floor((x - xi) / dx)), x_coords.end - 1));
		interp.iy = amrex::max(y_coords.begin, amrex::min(static_cast<int>(std::floor((y - yi) / dy)), y_coords.end - 1));

		// iix, iiy are the "upper-right" indices (handle boundary case)
		interp.iix = (interp.ix == x_coords.end - 1) ? interp.ix : interp.ix + 1;
		interp.iiy = (interp.iy == y_coords.end - 1) ? interp.iy : interp.iy + 1;

		// Get actual coordinate values at the four grid points
		interp.x1 = x_coords(interp.ix);  // Left x-coordinate
		interp.x2 = x_coords(interp.iix); // Right x-coordinate
		interp.y1 = y_coords(interp.iy);  // Bottom y-coordinate
		interp.y2 = y_coords(interp.iiy); // Top y-coordinate

		// Compute normalized coordinates within the grid cell [0,1] x [0,1]
		// h = 0 at x1, h = 1 at x2
		// v = 0 at y1, v = 1 at y2
		if (interp.ix != interp.iix) {
			interp.h = (x - interp.x1) / (interp.x2 - interp.x1);
		} else {
			interp.h = 0.0; // No variation in x direction (boundary case)
		}

		if (interp.iy != interp.iiy) {
			interp.v = (y - interp.y1) / (interp.y2 - interp.y1);
		} else {
			interp.v = 0.0; // No variation in y direction (boundary case)
		}

		return interp;
	}

	// Convenience method: find interpolation data and compute value in one call
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate(amrex::Real x, amrex::Real y) const -> amrex::Real
	{
		// Part 1: Find interpolation indices and normalized coordinates
		InterpData interp = find_interpolation_data(x, y);

		// Part 2: Compute interpolated value using precomputed indices and normalized coordinates
		amrex::Real z1 = data(interp.ix, interp.iy);
		amrex::Real z2 = data(interp.iix, interp.iy);
		amrex::Real z3 = data(interp.ix, interp.iiy);
		amrex::Real z4 = data(interp.iix, interp.iiy);

		// f(h, v) = (1 - v)((1 - h) z1 + h z2) + v((1 - h) z3 + h z4)
		amrex::Real value = (1.0 - interp.v) * ((1.0 - interp.h) * z1 + interp.h * z2) + interp.v * ((1.0 - interp.h) * z3 + interp.h * z4);
		AMREX_ASSERT(!std::isnan(value));

		return value;
	}

	// Compute numeric derivatives (∂f/∂x, ∂f/∂y) using normalized coordinate algorithm
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto numeric_derivative(amrex::Real x, amrex::Real y) const -> amrex::Array<amrex::Real, 2>
	{
		// Part 1: Get interpolation data (includes precomputed h and v)
		InterpData interp = find_interpolation_data(x, y);

		// Part 2: Compute derivatives in normalized coordinates
		amrex::Real z1 = data(interp.ix, interp.iy);
		amrex::Real z2 = data(interp.iix, interp.iy);
		amrex::Real z3 = data(interp.ix, interp.iiy);
		amrex::Real z4 = data(interp.iix, interp.iiy);

		amrex::Real f_h = 0.0;
		amrex::Real f_v = 0.0;

		// Compute derivatives in normalized coordinates
		if (interp.ix != interp.iix && interp.iy != interp.iiy) {
			// Full bilinear case
			// f_h = v (z4 - z3) + (1 - v) (z2 - z1)
			f_h = interp.v * (z4 - z3) + (1.0 - interp.v) * (z2 - z1);

			// f_v = h (z4 - z2) + (1 - h) (z3 - z1)
			f_v = interp.h * (z4 - z2) + (1.0 - interp.h) * (z3 - z1);

		} else if (interp.ix == interp.iix && interp.iy != interp.iiy) {
			// Linear interpolation in y direction only
			f_h = 0.0;     // No variation in x direction
			f_v = z3 - z1; // Linear derivative in normalized v coordinate

		} else if (interp.ix != interp.iix && interp.iy == interp.iiy) {
			// Linear interpolation in x direction only
			f_h = z2 - z1; // Linear derivative in normalized h coordinate
			f_v = 0.0;     // No variation in y direction

		} else {
			// Point interpolation - no derivatives
			f_h = 0.0;
			f_v = 0.0;
		}

		// Part 3: Convert to physical coordinates: f_x = f_h / (x2 - x1), f_y = f_v / (y2 - y1)
		amrex::Real dfdx = (interp.ix != interp.iix) ? f_h / (interp.x2 - interp.x1) : 0.0;
		amrex::Real dfdy = (interp.iy != interp.iiy) ? f_v / (interp.y2 - interp.y1) : 0.0;

		return {dfdx, dfdy};
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
	[[nodiscard]] auto const_tables() const -> DataTableGpuConst;

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

	int x_size_ = 0;
	int y_size_ = 0;
};

} // namespace quokka

#endif // DATATABLE_HPP_