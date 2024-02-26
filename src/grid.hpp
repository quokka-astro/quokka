#ifndef GRID_HPP_ // NOLINT
#define GRID_HPP_

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_REAL.H>

namespace quokka
{
enum class centering { cc = 0, fc, ec };
enum class direction { na = -1, x, y, z };
const std::array<const std::string, 3> face_dir_str = {"x", "y", "z"}; // NOLINT

struct grid {
	amrex::Array4<double> array_;
	amrex::Box indexRange_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi_;
	enum centering cen_;
	enum direction dir_;

	grid(amrex::Array4<double> const &array, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
	     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi, centering cen, direction dir)
	    : array_(array), indexRange_(indexRange), dx_(dx), prob_lo_(prob_lo), prob_hi_(prob_hi), cen_(cen), dir_(dir)
	{
	}
};
} // namespace quokka

#endif // GRID_HPP_