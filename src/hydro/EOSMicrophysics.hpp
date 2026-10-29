#ifndef EOS_MICROPHYSICS_HPP_ // NOLINT
#define EOS_MICROPHYSICS_HPP_
/// \file EOSMicrophysics.hpp
/// \brief The EOSMicrophysics backend of EOS.hpp: the default backend of the problems with chemistry or photochemistry
/// (see DefaultEOSBackend). It is in its own header because it uses spmasses, which only the chemistry EOSs declare;
/// chemistry/Chemistry.hpp and radiation/photochemistry.hpp include it, and problems without chemistry never do.
/// Likely transitional: this file may go once it is decided how Quokka does the EOS (the EOS port).
/// James471: This file was an agent's suggestion. It was not given a review by me and I recommend not putting effort into reviewing it, until the EOS port is
/// decided.

#include <tuple>

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "actual_eos_data.H"
#include "eos.H"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "microphysics/microphysics_traits.hpp"
#include "network_properties.H"
#include "physics_info.hpp"
#include "util/Optional.hpp"

namespace quokka
{

template <typename problem_t> struct EOSMicrophysics {
	// Until the EOS reads its species from the network (the EOS port), it sizes its composition by the species table on
	// the include path (network_properties.H), so the problem's network must match that table.
	static_assert(Microphysics_Traits<problem_t>::Network::NumSpec == NumSpec,
		      "The problem's Network does not match the species table of the EOS it is built with.");

	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr bool is_tabulated = false;
	static constexpr amrex::Real gamma_ = EOS_Traits<problem_t>::gamma;
	static constexpr amrex::Real boltzmann_constant_ = EOSIdeal<problem_t>::boltzmann_constant_;

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.e = Eint / rho;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_re, chemstate);
		return chemstate.T;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.T = Tgas;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_rt, chemstate);
		return chemstate.e * chemstate.rho;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.p = Pressure;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_rp, chemstate);
		return chemstate.e * chemstate.rho;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.T = Tgas;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_rt, chemstate);
		return chemstate.dedT * chemstate.rho;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeOtherDerivatives(const amrex::Real rho, const amrex::Real P, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	{
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.p = P;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_rp, chemstate);
		const amrex::Real deint_dRho = chemstate.dedr;
		const amrex::Real deint_dP = 1.0 / chemstate.dpde;
		amrex::Real dRho_dP = 1.0 / (chemstate.dpdr * C::k_B / boltzmann_constant_);
		const amrex::Real dP_dRho_s = chemstate.cs * chemstate.cs;
		const amrex::Real G = chemstate.G;
		return std::make_tuple(deint_dRho, deint_dP, dRho_dP, dP_dRho_s, G);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		static_assert(gamma_ != 1.0, "EOSMicrophysics does not support isothermal (gamma=1) problems.");
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.e = Eint / rho;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_re, chemstate);
		return chemstate.p;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		static_assert(gamma_ != 1.0, "EOSMicrophysics does not support isothermal (gamma=1) problems.");
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.p = Pressure;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_rp, chemstate);
		return chemstate.cs;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeIsothermalSoundSpeed(amrex::Real rho, amrex::Real Pressure) -> amrex::Real
	{
		static_assert(gamma_ == 1.0, "ComputeIsothermalSoundSpeed does not support general EOS");
		static_assert(EOS_Traits<problem_t>::cs_isothermal > 0.0, "EOS_Traits<problem_t>::cs_isothermal must be set when gamma=1.");
		amrex::ignore_unused(rho);
		amrex::ignore_unused(Pressure);
		return EOS_Traits<problem_t>::cs_isothermal;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEntropyFromRhoEint(amrex::Real /*rho*/, amrex::Real /*Eint*/,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/ = {}) -> amrex::Real
	{
		// sizeof(problem_t)==0: fires only on instantiation, not at parse time (C++20 ill-formed NDR workaround)
		static_assert(sizeof(problem_t) == 0, "ComputeEntropyFromRhoEint is only supported by the EOSTabulated backend");
		return 0.0;
	}
};

} // namespace quokka

#endif // EOS_MICROPHYSICS_HPP_
