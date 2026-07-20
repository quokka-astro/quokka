#ifndef QUOKKA_PRIMORDIAL_GENERATED_TYPES_HPP_
#define QUOKKA_PRIMORDIAL_GENERATED_TYPES_HPP_

#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace quokka::chemistry::primordial_detail
{

constexpr int NumSpec = 14;
constexpr int neqs = NumSpec + 1;
constexpr int net_ienuc = NumSpec + 1;

namespace Rates
{
enum NetworkRates : int {};
}

struct PrimordialRhsState {
	amrex::Real density = 0.0;
	amrex::Real temperature = 0.0;
	amrex::Real temperature_derivative = 0.0;
	amrex::GpuArray<amrex::Real, NumSpec> species{};
};

template <int N> struct OneBasedVector {
	amrex::GpuArray<amrex::Real, N> values{};

	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto operator()(int index) noexcept -> amrex::Real & { return values[index - 1]; }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto operator()(int index) const noexcept -> amrex::Real { return values[index - 1]; }
};

} // namespace quokka::chemistry::primordial_detail

#endif
