/// \file testRadDustBandsSweep.cpp
/// \brief Parameter sweep of the dust-absorption-band matter-radiation source step.
///
/// Each cell of a 1D MultiFab holds one combination of gas density, temperature, gas velocity (magnitude
/// and direction relative to the radiation flux), radiation energy and reduced flux of each band, stellar
/// source, and time step. RadSystem::AddSourceTermsMultiGroup is applied once to every cell, which runs
/// SolveDustAbsorptionBands followed by UpdateFlux and the outer work-term iteration. The test fails if
/// any finite input produces a non-finite output, and it reports the cells where the radiation energy or
/// the internal energy leaves its physical range.
///
/// The radiation traits are those of TallBoxSfFuv (FUV and LW bands, Weingartner & Draine dust opacity,
/// photoelectric efficiency 0.05, reduced speed of light c/100). The EOS is ideal rather than tabulated:
/// in this path the gas temperature feeds only the opacity, which does not depend on it.

#include <cmath>
#include <format>
#include <limits>

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"

struct DustBandsSweep {};

constexpr int n_groups = 2;
constexpr double gamma_gas = 5. / 3.;
constexpr double mu = C::m_p;

constexpr double gas_to_dust_mass_ratio = 1.653e2;
constexpr amrex::GpuArray<double, n_groups + 1> K_abs_dust = {4.486e4, 8.246e4, 1.310e5}; // cm^2 per g of dust
constexpr amrex::GpuArray<double, n_groups + 1> kappa_dust = {K_abs_dust.arr[0] / gas_to_dust_mass_ratio, K_abs_dust.arr[1] / gas_to_dust_mass_ratio,
							      K_abs_dust.arr[2] / gas_to_dust_mass_ratio}; // cm^2 per g of gas
constexpr double Erad_floor = 1.0e-25;
constexpr double chat_over_c = 1.0e-2;

template <> struct quokka::EOS_Traits<DustBandsSweep> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = gamma_gas;
};

template <> struct Physics_Traits<DustBandsSweep> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = n_groups;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<DustBandsSweep> {
	static constexpr double c_hat_over_c = chat_over_c;
	static constexpr double Erad_floor = ::Erad_floor;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr amrex::GpuArray<double, n_groups + 1> radBoundaries = {6.0, 11.2, 13.6}; // eV
	static constexpr int beta_order = 1;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr bool dust_absorption_only = true;
	static constexpr amrex::GpuArray<double, n_groups> pe_heating_efficiency = {0.05, 0.05};
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<DustBandsSweep>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, n_groups + 1> rad_boundaries,
											   const double /*rho*/, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, n_groups + 1>, 2>
{
	// same as TallBoxSfFuv
	const amrex::GpuArray<double, n_groups + 1> kappa = kappa_dust;
	amrex::GpuArray<amrex::GpuArray<double, n_groups + 1>, 2> exponents_and_values{};
	for (int g = 0; g < n_groups; ++g) {
		exponents_and_values[0][g] = std::log(kappa[g + 1] / kappa[g]) / std::log(rad_boundaries[g + 1] / rad_boundaries[g]);
		exponents_and_values[1][g] = kappa[g];
	}
	exponents_and_values[0][n_groups] = 0.0;
	exponents_and_values[1][n_groups] = kappa[n_groups];
	return exponents_and_values;
}

// Sweep axes. A negative temperature entry stands for a negative internal energy of the same magnitude,
// which is what E_tot - E_kin gives in a cell where the hydro update has lost the internal energy.
constexpr amrex::GpuArray<double, 8> sweep_rho = {1.0e-28, 1.0e-26, 1.0e-24, 1.0e-22, 1.0e-21, 3.4e-21, 1.0e-20, 1.0e-18}; // g cm^-3
constexpr amrex::GpuArray<double, 7> sweep_T = {1.0, 10.0, 100.0, 1.0e4, 1.0e6, 1.0e8, -100.0};				   // K
constexpr amrex::GpuArray<double, 5> sweep_v = {0.0, 1.0e5, 1.0e7, 1.0e8, 1.0e9};					   // cm s^-1
constexpr int n_vdir = 3; // velocity along +x (with the flux), -x (against it), +y (across it)
// The smallest denormal is what an unilluminated opaque cell decays to when nothing enforces the floor.
constexpr amrex::GpuArray<double, 6> sweep_Erad = {
    std::numeric_limits<double>::denorm_min(), 0.5 * Erad_floor, 1.0e-22, 1.0e-18, 1.0e-14, 1.0e-10}; // erg cm^-3, per band
constexpr amrex::GpuArray<double, 4> sweep_f = {0.0, 0.5, 0.99, 1.0};				      // |F| / (c E), both bands, along +x
constexpr amrex::GpuArray<double, 4> sweep_src = {0.0, 1.0e-20, 1.0e-16, 1.0e-12};		      // L / V, erg s^-1 cm^-3, per band
constexpr amrex::GpuArray<double, 3> sweep_dt = {1.0e7, 2.759e9, 1.0e11};			      // s

constexpr int n_rho = 8;
constexpr int n_T = 7;
constexpr int n_v = 5;
constexpr int n_E = 6;
constexpr int n_f = 4;
constexpr int n_src = 4;
constexpr int n_dt = 3;
constexpr int n_cells_per_dt = n_rho * n_T * n_v * n_vdir * n_E * n_E * n_f * n_src;

struct SweepPoint {
	double rho;
	double T;
	double v;
	int vdir;
	double E0;
	double E1;
	double f;
	double src;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto DecodeSweepPoint(int idx) -> SweepPoint
{
	// mixed-radix decoding of the cell index; the time step is the outermost axis and is handled by the caller
	const amrex::GpuArray<double, 8> rho_list = sweep_rho;
	const amrex::GpuArray<double, 7> T_list = sweep_T;
	const amrex::GpuArray<double, 5> v_list = sweep_v;
	const amrex::GpuArray<double, 6> E_list = sweep_Erad;
	const amrex::GpuArray<double, 4> f_list = sweep_f;
	const amrex::GpuArray<double, 4> src_list = sweep_src;

	SweepPoint pt{};
	pt.src = src_list[idx % n_src];
	idx /= n_src;
	pt.f = f_list[idx % n_f];
	idx /= n_f;
	pt.E1 = E_list[idx % n_E];
	idx /= n_E;
	pt.E0 = E_list[idx % n_E];
	idx /= n_E;
	pt.vdir = idx % n_vdir;
	idx /= n_vdir;
	pt.v = v_list[idx % n_v];
	idx /= n_v;
	pt.T = T_list[idx % n_T];
	idx /= n_T;
	pt.rho = rho_list[idx % n_rho];
	return pt;
}

auto problem_main() -> int
{
	using RadSys = RadSystem<DustBandsSweep>;

	// initialize the Microphysics EOS, as the QuokkaSimulation constructor does
	amrex::ParmParse eos("eos");
	eos.add("eos_gamma", gamma_gas);
	init_extern_parameters();
	amrex::Real small_temp = 1.0e-10;
	amrex::Real small_dens = 1.0e-100;
	eos_init(small_temp, small_dens);
	constexpr int nvars = Physics_Indices<DustBandsSweep>::nvarTotal_cc;
	constexpr int nrv = Physics_NumVars::numRadVarsPerGroup;

	const amrex::Box domain(amrex::IntVect(AMREX_D_DECL(0, 0, 0)), amrex::IntVect(AMREX_D_DECL(n_cells_per_dt - 1, 0, 0)));
	amrex::BoxArray ba(domain);
	ba.maxSize(8192);
	const amrex::DistributionMapping dm(ba);

	amrex::MultiFab state(ba, dm, nvars, 0);
	amrex::MultiFab state_in(ba, dm, nvars, 0);
	amrex::MultiFab radEnergySource(ba, dm, n_groups, 0);
	amrex::MultiFab radFluxSource(ba, dm, 3 * n_groups, 0);

	long n_nonfinite_total = 0;	   // NOLINT(google-runtime-int)
	long n_erad_nonpositive_total = 0; // NOLINT(google-runtime-int)
	long n_acausal_total = 0;	   // NOLINT(google-runtime-int)
	long n_eint_nonpositive_total = 0; // NOLINT(google-runtime-int)
	long n_eint_decreased_total = 0;   // NOLINT(google-runtime-int)

	for (int idt = 0; idt < n_dt; ++idt) {
		const amrex::GpuArray<double, 3> dt_list = sweep_dt;
		const double dt = dt_list[idt];

		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.validbox();
			auto const s = state.array(mfi);
			auto const src_E = radEnergySource.array(mfi);
			auto const src_F = radFluxSource.array(mfi);
			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const SweepPoint pt = DecodeSweepPoint(i);
				const double c = RadSys::c_light_;
				for (int n = 0; n < nvars; ++n) {
					s(i, j, k, n) = 0.0;
				}
				const double Eint_mag = pt.rho * C::k_B * std::abs(pt.T) / ((gamma_gas - 1.0) * mu);
				const double Eint = (pt.T > 0.0) ? Eint_mag : -Eint_mag;
				const double vx = (pt.vdir == 0) ? pt.v : ((pt.vdir == 1) ? -pt.v : 0.0);
				const double vy = (pt.vdir == 2) ? pt.v : 0.0;
				s(i, j, k, RadSys::gasDensity_index) = pt.rho;
				s(i, j, k, RadSys::x1GasMomentum_index) = pt.rho * vx;
				s(i, j, k, RadSys::x2GasMomentum_index) = pt.rho * vy;
				s(i, j, k, RadSys::gasInternalEnergy_index) = Eint;
				s(i, j, k, RadSys::gasEnergy_index) = Eint + 0.5 * pt.rho * (vx * vx + vy * vy);
				const double Eg[2] = {pt.E0, pt.E1};
				for (int g = 0; g < n_groups; ++g) {
					s(i, j, k, RadSys::radEnergy_index + nrv * g) = Eg[g];
					s(i, j, k, RadSys::x1RadFlux_index + nrv * g) = pt.f * c * Eg[g];
					src_E(i, j, k, g) = pt.src;
					for (int n = 0; n < 3; ++n) {
						src_F(i, j, k, 3 * g + n) = 0.0;
					}
				}
			});
		}
		amrex::MultiFab::Copy(state_in, state, 0, 0, nvars, 0);

		amrex::Gpu::Buffer<int> iteration_counter({0, 0, 0, 0});
		amrex::Gpu::Buffer<int> iteration_failure_counter({0, 0, 0});
		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.validbox();
			auto const s = state.array(mfi);
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
			RadSys::AddSourceTermsMultiGroup(s, radEnergySource.const_array(mfi), radFluxSource.const_array(mfi), bx, dt, 1.0, 0.0, 1.0e-11,
							 1.0e-11, 0.0, iteration_counter.data(), iteration_failure_counter.data(),
							 amrex::Array4<const amrex::Real>{}, cons_fc);
		}
		amrex::Gpu::streamSynchronize();
		const int n_outer_failed = iteration_failure_counter.copyToHost()[2];

		// classify the outputs
		amrex::Gpu::Buffer<int> counts({0, 0, 0, 0, 0}); // non-finite, Erad < floor, |F| > c E, Eint <= 0 (from Eint > 0), Eint decreased
		int *p_counts = counts.data();
		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.validbox();
			auto const s = state.const_array(mfi);
			auto const s0 = state_in.const_array(mfi);
			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double c = RadSys::c_light_;
				bool nonfinite = false;
				for (int n = 0; n < nvars; ++n) {
					nonfinite = nonfinite || !std::isfinite(s(i, j, k, n));
				}
				bool erad_nonpositive = false;
				bool acausal = false;
				for (int g = 0; g < n_groups; ++g) {
					const double E = s(i, j, k, RadSys::radEnergy_index + nrv * g);
					const double Fx = s(i, j, k, RadSys::x1RadFlux_index + nrv * g);
					const double Fy = s(i, j, k, RadSys::x2RadFlux_index + nrv * g);
					const double Fz = s(i, j, k, RadSys::x3RadFlux_index + nrv * g);
					erad_nonpositive = erad_nonpositive || !(E >= RadSys::Erad_floor_);
					acausal = acausal || (std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz) > (1.0 + 1.0e-10) * c * E);
				}
				const bool eint_nonpositive =
				    (s0(i, j, k, RadSys::gasInternalEnergy_index) > 0.0) && !(s(i, j, k, RadSys::gasInternalEnergy_index) > 0.0);
				// Only the work term, the cosmic-ray heating and the photoelectric heating reach the gas energy, and the
				// work term goes to the kinetic energy, so the internal energy of a cell that starts positive must not fall.
				const double eint0 = s0(i, j, k, RadSys::gasInternalEnergy_index);
				const bool eint_decreased = (eint0 > 0.0) && (s(i, j, k, RadSys::gasInternalEnergy_index) < eint0 * (1.0 - 1.0e-10));
				int const flags[5] = {static_cast<int>(nonfinite), static_cast<int>(erad_nonpositive), static_cast<int>(acausal),
						      static_cast<int>(eint_nonpositive), static_cast<int>(eint_decreased)};
				for (int m = 0; m < 5; ++m) {
					if (flags[m] != 0) {
						const int nth = amrex::Gpu::Atomic::Add(&p_counts[m], 1);
						if (nth < 5) {
							const SweepPoint pt = DecodeSweepPoint(i);
							printf("[sweep] flag=%d cell=%d rho=%.3e T=%.3e v=%.3e vdir=%d E0=%.3e E1=%.3e f=%.3f "
							       "src=%.3e | " // NOLINT
							       "Eint=%.6e mom=(%.6e,%.6e) E=(%.6e,%.6e) Fx=(%.6e,%.6e)\n",
							       m, i, pt.rho, pt.T, pt.v, pt.vdir, pt.E0, pt.E1, pt.f, pt.src,
							       s(i, j, k, RadSys::gasInternalEnergy_index), s(i, j, k, RadSys::x1GasMomentum_index),
							       s(i, j, k, RadSys::x2GasMomentum_index), s(i, j, k, RadSys::radEnergy_index),
							       s(i, j, k, RadSys::radEnergy_index + nrv), s(i, j, k, RadSys::x1RadFlux_index),
							       s(i, j, k, RadSys::x1RadFlux_index + nrv));
						}
					}
				}
			});
		}
		amrex::Gpu::streamSynchronize();
		const int *h_counts = counts.copyToHost();
		amrex::Print() << std::format("dt = {:.3e} s: {} cells, non-finite = {}, Erad < floor = {}, |F| > cE = {}, Eint <= 0 = {}, "
					      "Eint decreased = {}, outer iteration not converged = {}\n",
					      dt, n_cells_per_dt, h_counts[0], h_counts[1], h_counts[2], h_counts[3], h_counts[4], n_outer_failed);
		n_nonfinite_total += h_counts[0];
		n_erad_nonpositive_total += h_counts[1];
		n_acausal_total += h_counts[2];
		n_eint_nonpositive_total += h_counts[3];
		n_eint_decreased_total += h_counts[4];
	}

	amrex::Print() << std::format("total: non-finite = {}, Erad < floor = {}, |F| > cE = {}, Eint <= 0 = {}, Eint decreased = {}\n", n_nonfinite_total,
				      n_erad_nonpositive_total, n_acausal_total, n_eint_nonpositive_total, n_eint_decreased_total);

	// Regression case: the cell of sigma13.v2 (TallBoxSfFuv) that turned NaN at step 747454, with the inputs of its stage-2
	// source step printed by the chong/debug/fuv-nan build. The LW band had decayed to the smallest denormal, so the
	// implicit absorption rounded it to zero and UpdateFlux divided the flux by it.
	bool regression_ok = true;
	{
		const amrex::Box one(amrex::IntVect(AMREX_D_DECL(0, 0, 0)), amrex::IntVect(AMREX_D_DECL(0, 0, 0)));
		const amrex::BoxArray ba1(one);
		const amrex::DistributionMapping dm1(ba1);
		amrex::MultiFab s1(ba1, dm1, nvars, 0);
		amrex::MultiFab srcE1(ba1, dm1, n_groups, 0);
		amrex::MultiFab srcF1(ba1, dm1, 3 * n_groups, 0);
		s1.setVal(0.0);
		srcE1.setVal(0.0);
		srcF1.setVal(0.0);
		const double dt = 3.56651514527451706e+09;
		for (amrex::MFIter mfi(s1); mfi.isValid(); ++mfi) {
			auto const s = s1.array(mfi);
			amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				s(i, j, k, RadSys::gasDensity_index) = 3.44528865903911935e-21;
				s(i, j, k, RadSys::x1GasMomentum_index) = -7.11244513376165101e-16;
				s(i, j, k, RadSys::x2GasMomentum_index) = -1.10431829499197874e-15;
				s(i, j, k, RadSys::x3GasMomentum_index) = 2.86234867662342772e-16;
				s(i, j, k, RadSys::gasEnergy_index) = 2.70619468665379918e-10;
				s(i, j, k, RadSys::gasInternalEnergy_index) = 8.33113337805112455e-12;
				s(i, j, k, RadSys::radEnergy_index) = 9.64119066885580445e-16;
				s(i, j, k, RadSys::x1RadFlux_index) = -3.58264761111917227e-06;
				s(i, j, k, RadSys::x2RadFlux_index) = -1.51695682434051296e-05;
				s(i, j, k, RadSys::x3RadFlux_index) = -6.82526723628103076e-07;
				s(i, j, k, RadSys::radEnergy_index + nrv) = 4.94065645841246544e-324;
				s(i, j, k, RadSys::x1RadFlux_index + nrv) = -5.54715622802528779e-316;
				s(i, j, k, RadSys::x2RadFlux_index + nrv) = -3.54126130163053471e-315;
				s(i, j, k, RadSys::x3RadFlux_index + nrv) = 9.61404151240106481e-314;
			});
		}
		amrex::Gpu::Buffer<int> iteration_counter({0, 0, 0, 0});
		amrex::Gpu::Buffer<int> iteration_failure_counter({0, 0, 0});
		for (amrex::MFIter mfi(s1); mfi.isValid(); ++mfi) {
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
			RadSys::AddSourceTermsMultiGroup(s1.array(mfi), srcE1.const_array(mfi), srcF1.const_array(mfi), mfi.validbox(), dt, 1.0, 0.0, 1.0e-11,
							 1.0e-11, 0.0, iteration_counter.data(), iteration_failure_counter.data(),
							 amrex::Array4<const amrex::Real>{}, cons_fc);
		}
		amrex::Gpu::streamSynchronize();
		regression_ok =
		    !s1.contains_nan(0, nvars, 0) && !s1.contains_inf(0, nvars, 0) && (s1.min(RadSys::radEnergy_index + nrv) >= RadSys::Erad_floor_);
		amrex::Print() << std::format("regression cell (sigma13.v2 step 747454): Eint = {:.6e}, E_LW = {:.6e}, F_LW,z = {:.6e} -> {}\n",
					      s1.max(RadSys::gasInternalEnergy_index), s1.max(RadSys::radEnergy_index + nrv),
					      s1.max(RadSys::x3RadFlux_index + nrv), regression_ok ? "ok" : "FAILED");
	}

	// Only non-finite output fails the test. The other counts are diagnostics: where the outer work-term iteration does
	// not converge (only at extreme dt and velocity), AddSourceTermsMultiGroup leaves the radiation state as it found
	// it, so an input below the floor stays below it; the next source step floors it.
	const int status = (n_nonfinite_total == 0 && regression_ok) ? 0 : 1;
	amrex::Print() << (status == 0 ? "RadDustBandsSweep Success\n" : "RadDustBandsSweep FAILED\n");
	return status;
}
