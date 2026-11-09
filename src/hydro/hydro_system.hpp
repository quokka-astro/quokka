// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef HYDRO_SYSTEM_HPP_ // NOLINT
#define HYDRO_SYSTEM_HPP_
/// \file hydro_system.hpp
/// \brief Defines a class for solving the Euler equations.
///

// c++ headers
#include <algorithm>
#include <array>
#include <cmath>

// library headers
#include "AMReX.H"
#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_iMultiFab.H"

// internal headers
#include "EOS.hpp"
#include "HLLC.hpp"
#include "HLLD.hpp"
#include "LLF.hpp"
#include "LLF_mhd.hpp"
#include "dust/dust_system.hpp"
#include "hyperbolic_system.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "util/ArrayView.hpp"
#include "util/valarray.hpp"

// Microphysics headers
#include "extern_parameters.H"

// this struct is specialized by the user application code
//
template <typename problem_t> struct HydroSystem_Traits {
	// if true, reconstruct e_int instead of pressure
	static constexpr bool reconstruct_eint = true;
};

// Helper for dependent static_assert that always evaluates to false
// This is needed to make static_assert work inside constexpr if blocks
template <typename T> struct dependent_false : std::false_type {};
template <typename T> inline constexpr bool dependent_false_v = dependent_false<T>::value;

enum class RiemannSolver { HLLC, LLF, LLF_MHD, HLLD };

namespace quokka
{
// hydro.march_kchunk: when > 0 (GPU, 3D), the hydro face/update/flattening kernels are launched with each 64 x 4 (i, j) block
// marching through k-chunks of this many cells, so the k-neighbour planes of the stencils are reused from cache instead of
// being re-read after a whole plane has been streamed. Only the iteration order changes. Set from QuokkaSimulation.
inline int hydroMarchKChunk = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// amrex::ParallelFor(mf, ng, f) with f(box_index, i, j, k), optionally in k-marching order (see hydroMarchKChunk)
template <typename MF, typename F> void ParallelForHydro(MF const &mf, amrex::IntVect const &ng, F const &f)
{
#if defined(AMREX_USE_GPU) && !defined(AMREX_USE_SYCL) && (AMREX_SPACEDIM == 3)
	const int kchunk = hydroMarchKChunk;
	if (kchunk > 0) {
		constexpr int TX = 64;
		constexpr int TY = 4;
		for (amrex::MFIter mfi(mf, amrex::MFItInfo().DisableDeviceSync()); mfi.isValid(); ++mfi) {
			const int lbx = mfi.LocalIndex();
			const amrex::Box bx = amrex::grow(mfi.validbox(), ng);
			const auto lo = amrex::lbound(bx);
			const auto hi = amrex::ubound(bx);
			const int ntx = (hi.x - lo.x + TX) / TX;
			const int nty = (hi.y - lo.y + TY) / TY;
			const int nkc = (hi.z - lo.z + kchunk) / kchunk;
			amrex::launch<TX * TY>(ntx * nty * nkc, 0, amrex::Gpu::gpuStream(), [=] AMREX_GPU_DEVICE() noexcept {
				const int t = static_cast<int>(threadIdx.x);
				int b = static_cast<int>(blockIdx.x);
				const int i = lo.x + TX * (b % ntx) + (t % TX);
				b /= ntx;
				const int j = lo.y + TY * (b % nty) + (t / TX);
				const int k0 = lo.z + kchunk * (b / nty);
				if (i > hi.x || j > hi.y) {
					return;
				}
				const int k1 = amrex::min(k0 + kchunk - 1, hi.z);
				for (int k = k0; k <= k1; ++k) {
					f(lbx, i, j, k);
				}
			});
		}
		return;
	}
#endif
	amrex::ParallelFor(mf, ng, f);
}

// per-cell fixup that does nothing (fused stage kernels without in-kernel floors)
struct NoCellFixup {
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(amrex::Array4<amrex::Real> const & /*state*/, int /*i*/, int /*j*/, int /*k*/) const {}
};

// Read-only view of an interface state held in registers, with the (i, j, k, n) call signature of Array4View,
// so that the Riemann-solver kernel body can read a state reconstructed in the kernel (fusion level >= 2)
struct RegisterStateView {
	double const *v;
	[[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto operator()(int /*i*/, int /*j*/, int /*k*/, int n) const -> double { return v[n]; }
};
} // namespace quokka

/// Class for the Euler equations of inviscid hydrodynamics
///
template <typename problem_t> class HydroSystem : public HyperbolicSystem<problem_t>
{
      public:
	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr int nscalars_ = Physics_Traits<problem_t>::numPassiveScalars;
	static constexpr int nvar_ = Physics_NumVars::numHydroVars + nscalars_ +
				     (Physics_Traits<problem_t>::nDustGroups * Physics_NumVars::numDustVarsPerGroup) *
					 static_cast<int>(Physics_Traits<problem_t>::is_dust_enabled); // total number of variables
	static constexpr int nHydroScalars_ = Physics_NumVars::numHydroVars + nscalars_;
	static constexpr int numDustVars_ = Physics_NumVars::numDustVarsPerGroup; // number of dust variables for each dust group

	enum consVarIndex { // NOLINT
		density_index = Physics_Indices<problem_t>::hydroFirstIndex,
		x1Momentum_index,
		x2Momentum_index,
		x3Momentum_index,
		energy_index,
		internalEnergy_index, // auxiliary internal energy (rho * e)
		scalar0_index	      // first passive scalar (only present if nscalars > 0!)
	};

	enum primVarIndex { // NOLINT
		primDensity_index = 0,
		x1Velocity_index,
		x2Velocity_index,
		x3Velocity_index,
		pressure_index,
		primEint_index,	   // auxiliary internal energy (rho * e)
		primScalar0_index, // first passive scalar (only present if nscalars > 0!)
	};

	enum dustVarIndex { // NOLINT
		dustDensity_index = Physics_Indices<problem_t>::dustFirstIndex,
		x1DustMomentum_index,
		x2DustMomentum_index,
		x3DustMomentum_index
	};

	static constexpr int primDustFirstIndex = primScalar0_index + nscalars_;
	enum primDustVarIndex { primDustDensity_index = primDustFirstIndex, x1DustVelocity_index, x2DustVelocity_index, x3DustVelocity_index }; // NOLINT

	static void ConservedToPrimitive(amrex::MultiFab const &cons_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf,
					 amrex::MultiFab &primVar_mf, int nghost);

	static auto maxSignalSpeedLocal(amrex::MultiFab const &cons, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf) -> amrex::Real;

	static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons_cc,
					  std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const &cons_fc, array_t &maxSignal,
					  amrex::Box const &indexRange);

	// per-cell characteristic speed shared by maxSignalSpeedLocal and ComputeMaxSignalSpeed; includes gas, MHD fast
	// magnetosonic, and dust contributions so both callers agree on what limits the timestep
	AMREX_GPU_DEVICE static auto ComputeCellSignalSpeed(amrex::Array4<const amrex::Real> const &cons_cc, int i, int j, int k,
							    std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr)
	    -> amrex::Real;

	static auto CheckStatesValid(amrex::MultiFab const &cons_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf) -> bool;

	AMREX_GPU_DEVICE static auto ComputePrimVars(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k,
						     std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr)

	    -> quokka::valarray<amrex::Real, nHydroScalars_>;

	AMREX_GPU_DEVICE static auto ComputeConsVars(quokka::valarray<amrex::Real, nHydroScalars_> const &prim)
	    -> quokka::valarray<amrex::Real, nHydroScalars_>;

	AMREX_GPU_DEVICE static auto ComputePressure(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k,
						     std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeInternalEnergy(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k,
							   std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr)
	    -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeSoundSpeed(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k,
						       std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeIsothermalSoundSpeed(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k,
								 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr)
	    -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeMagneticEnergy(int i, int j, int k,
							   std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr)
	    -> amrex::Real;

	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto
	ComputeCellCenteredMagneticEnergy(int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const &fc) -> amrex::Real
	{
		return ::ComputeCellCenteredMagneticEnergy<problem_t>(i, j, k, fc);
	}

	AMREX_GPU_DEVICE static auto ComputePlasmaBeta(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k,
						       std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeVelocityX1(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeVelocityX2(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeVelocityX3(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool;

	static void ComputeRhsFromFluxes(amrex::MultiFab &rhs_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray,
					 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int nvars);

	static void PredictStep(amrex::MultiFab const &consVarOld, amrex::MultiFab &consVarNew, amrex::MultiFab const &rhs, double dt, int nvars,
				amrex::iMultiFab &redoFlag_mf);

	static void AddFluxesRK2(amrex::MultiFab &Unew_mf, amrex::MultiFab const &U0_mf, amrex::MultiFab const &U1_mf, amrex::MultiFab const &rhs_mf, double dt,
				 int nvars, amrex::iMultiFab &redoFlag_mf);

	AMREX_GPU_DEVICE static auto GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec) -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>;

	template <typename DensityFloorFunc>
	// If syncDualEnergy is true, the dual-energy sync (SyncDualEnergy) is applied to each cell in the same kernel, after its floors
	// (one pass over the state instead of two; the per-cell result is identical)
	static void EnforceLimits(amrex::Real densityFloor, amrex::Real dustDensityFloor, amrex::Real tempFloor, amrex::MultiFab &state_mf,
				  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc_mf, amrex::Geometry const &geom,
				  DensityFloorFunc const &density_floor_func, bool syncDualEnergy = false);

	static void AddInternalEnergyPdV(amrex::MultiFab &rhs_mf, amrex::MultiFab const &consVar_mf, amrex::MultiFab const &primVar_mf,
					 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
					 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArray, amrex::iMultiFab const &redoFlag_mf);

	static void SyncDualEnergy(amrex::MultiFab &consVar_mf, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &faceVar_mf);

	// per-cell body of EnforceLimits (the face arrays are read only for MHD)
	template <typename DensityFloorFunc>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void
	EnforceLimitsCell(amrex::Array4<amrex::Real> const &state, int i, int j, int k, amrex::Real densityFloor, amrex::Real dustDensityFloor,
			  amrex::Real tempFloor, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
			  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, DensityFloorFunc const &density_floor_func,
			  amrex::Array4<const amrex::Real> const &fc_x, amrex::Array4<const amrex::Real> const &fc_y,
			  amrex::Array4<const amrex::Real> const &fc_z);

	// per-cell body of SyncDualEnergy (the face arrays are read only for MHD)
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void SyncDualEnergyCell(amrex::Array4<amrex::Real> const &consVar, int i, int j, int k,
									   amrex::Array4<const amrex::Real> const &faceVar_x,
									   amrex::Array4<const amrex::Real> const &faceVar_y,
									   amrex::Array4<const amrex::Real> const &faceVar_z);

	// FUSED_ORDER == 0: read the interface states from x1LeftState_mf / x1RightState_mf.
	// FUSED_ORDER == 2 (PLM) or 3 (PPM) (fusion level >= 2): reconstruct the flattened interface states of each face in registers
	// from primVar_mf and *chiCell_mf; x1LeftState_mf / x1RightState_mf are not read.
	// If accumulate is true (fusion level >= 3), the outputs are averaged in place: out = accumWeightOld * out + accumWeightNew * new.
	// MT: GPU threads per block. MT = 1024 makes __launch_bounds__ cap the kernel at 128 VGPRs (4 waves/SIMD on gfx90a).
	template <RiemannSolver RIEMANN, FluxDir DIR, int FUSED_ORDER = 0, SlopeLimiter limiter = SlopeLimiter::minmod, int MT = AMREX_GPU_MAX_THREADS>
	static void
	ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab &x1FaceVel_mf, amrex::MultiFab const &x1LeftState_mf, amrex::MultiFab const &x1RightState_mf,
		      amrex::MultiFab const &leftState_bfield_mf, amrex::MultiFab const &rightState_bfield_mf, amrex::MultiFab const &primVar_mf,
		      amrex::Real K_visc, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, amrex::Real shearViscosity, amrex::Real bulkViscosity,
		      amrex::MultiFab *x1FSpds_mf = nullptr, amrex::MultiFab const *x1ConsVar_fc_mf = nullptr, int nghost_vel = 2,
		      amrex::MultiFab const *chiCell_mf = nullptr, bool accumulate = false, amrex::Real accumWeightOld = 0., amrex::Real accumWeightNew = 1.);

	struct FaceFlux {
		quokka::valarray<double, nHydroScalars_> F;
		double faceVel;
	};

	// Riemann flux and face velocity at one face from its interface states (the per-face body of ComputeFluxes)
	template <RiemannSolver RIEMANN, FluxDir DIR, typename StateViewL, typename StateViewR>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto
	RiemannFaceFlux(StateViewL const &x1LeftState, StateViewR const &x1RightState, quokka::Array4View<const amrex::Real, DIR> const &x1LeftState_bfield,
			quokka::Array4View<const amrex::Real, DIR> const &x1RightState_bfield, amrex::Array4<const amrex::Real> const &x1ConsVar_fc_ref,
			amrex::Array4<amrex::Real> const &x1FSpds_ref, quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k,
			amrex::Real K_visc, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::Real shearViscosity, amrex::Real bulkViscosity)
	    -> FaceFlux;

	template <FluxDir DIR>
	AMREX_GPU_DEVICE static auto ComputeViscousFlux(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k, int velN_index,
							int velV_index, int velW_index, amrex::Real dx_n, amrex::Real dx_v, amrex::Real dx_w,
							amrex::Real shearViscosity, amrex::Real bulkViscosity)
	    -> quokka::valarray<amrex::Real, 4>; // {sigma_nn, sigma_nv, sigma_nw, v.sigma}

	template <FluxDir DIR>
	static void ComputeFirstOrderFluxes(amrex::Array4<const amrex::Real> const &consVar, array_t &x1FluxDiffusive, amrex::Box const &indexRange);

	template <FluxDir DIR> static void ComputeFlatteningCoefficients(amrex::MultiFab const &primVar_mf, amrex::MultiFab &x1Chi_mf, int nghost);

	// hydro-05: all directions in one kernel, reading the stored flattening pressure (primPflat_index) of primVar_mf
	static void ComputeFlatteningCoefficientsAllDirs(amrex::MultiFab const &primVar_mf, std::array<amrex::MultiFab, 3> &flatCoefs, int nghost);

	// hydro-05: flattening coefficient chi in direction DIR at cell (i_in, j_in, k_in), given the stored primitives
	// (pressure taken from primPflat_index) and the cell's K_S = rho * c_s^2 (Eq. 76 of Miller & Colella 2002)
	template <FluxDir DIR>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto FlatteningChi(amrex::Array4<const amrex::Real> const &primVar_in, int i_in, int j_in, int k_in,
								      amrex::Real K_S) -> amrex::Real;

	// hydro-05: K_S = rho * c_s^2 at a cell, from the stored primitives (same formula as ComputeFlatteningCoefficients)
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto FlatteningKS(amrex::Array4<const amrex::Real> const &primVar_in, int i, int j, int k) -> amrex::Real;

	template <FluxDir DIR>
	static void FlattenShocks(amrex::MultiFab const &q_mf, amrex::MultiFab const &x1Chi_mf, amrex::MultiFab const &x2Chi_mf,
				  amrex::MultiFab const &x3Chi_mf, amrex::MultiFab &x1LeftState_mf, amrex::MultiFab &x1RightState_mf, int nghost, int nvars);

	// fusion level >= 1: the per-cell flattening coefficient chi_ijk (minimum of the 3 * 3 directional coefficients
	// around the cell, Eq. 78 of Miller & Colella 2002) in one array, computed in one kernel
	static void ComputeFlatteningChiCell(amrex::MultiFab const &primVar_mf, amrex::MultiFab &chiCell_mf, int nghost);

	// fusion level >= 1: flattened interface values (a_minus, a_plus) of zone (i, j, k) [permuted to DIR] for component n,
	// for PLM (ORDER == 2) or PPM (ORDER == 3); same blend as FlattenShocks
	template <FluxDir DIR, int ORDER, SlopeLimiter limiter>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto FlattenedEdges(quokka::Array4View<const amrex::Real, DIR> const &q, int n, int i, int j, int k,
								       amrex::Real chi) -> std::pair<double, double>;

	// fusion level 1: reconstruction with flattening applied in registers (replaces ReconstructStates* + FlattenShocks)
	template <FluxDir DIR, int ORDER, SlopeLimiter limiter>
	static void ReconstructStatesFlattened(amrex::MultiFab const &q_mf, amrex::MultiFab const &chiCell_mf, amrex::MultiFab &x1LeftState_mf,
					       amrex::MultiFab &x1RightState_mf, int nghost, int nvars);

	// fusion level >= 3: flux divergence + P dV + predict + validity flag in one cell kernel
	// (replaces ComputeRhsFromFluxes + AddInternalEnergyPdV + PredictStep; no rhs array)
	// fixup(consVarNew, i, j, k) is applied to each cell that passes the validity check (see FinishFusedStage)
	template <typename Fixup = quokka::NoCellFixup>
	static void UpdateStageFused(amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf,
				     std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray,
				     std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArray, amrex::MultiFab const &primVar_mf,
				     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, double dt, int nvars, amrex::iMultiFab &redoFlag_mf, bool useRedoFlag,
				     Fixup const &fixup = Fixup{});

	// fusion level 4: the HLLC flux and face velocity at the low-DIR face of cell (i, j, k) (unpermuted indices), reconstructed
	// in registers from the primitive stencil (same arithmetic as ComputeFluxes<HLLC, DIR, ORDER, limiter>)
	template <FluxDir DIR, int ORDER, SlopeLimiter limiter>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto
	FusedFaceFlux(amrex::Array4<const amrex::Real> const &primVar, amrex::Array4<const amrex::Real> const &chiCell, int i, int j, int k, amrex::Real K_visc,
		      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::Real shearViscosity, amrex::Real bulkViscosity) -> FaceFlux;

	// fusion level 4: one kernel per RK stage. Each cell computes the fluxes on its 2 * AMREX_SPACEDIM faces in registers, then
	// applies the flux divergence, P dV and the forward-Euler update, and sets the validity flag. Nothing face-centred is stored.
	// Stage 1 (STAGE == 1): U_new = U_old + dt * (D1 - P_old div v1); stores stage1Weight * (D1, div v1) in stageDiv_mf.
	// Stage 2 (STAGE == 2): U_new = U_old + dt * (D - P_old div v) with D = stage1Weight * D1 + 0.5 * D2 (and the same for div v),
	//   which equals the flux divergence of the RK-averaged fluxes used by fusion levels 0-3 up to round-off.
	// D is the flux divergence of the nvars conserved components; primVarOld supplies the pressure of the P dV term.
	// fusion level 5: same result as UpdateStageFusedAllFaces, with each face flux computed once per tile and shared through LDS
	// (GPU, 3D; elsewhere it calls UpdateStageFusedAllFaces)
	template <int STAGE, int ORDER, SlopeLimiter limiter, typename Fixup>
	static void UpdateStageFusedLDS(amrex::MultiFab const &primVar_mf, amrex::MultiFab const &chiCell_mf, amrex::MultiFab const &primVarOld_mf,
					amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf, amrex::MultiFab &stageDiv_mf,
					amrex::Real stage1Weight, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, double dt, amrex::Real K_visc,
					amrex::Real shearViscosity, amrex::Real bulkViscosity, amrex::iMultiFab &redoFlag_mf, Fixup const &fixup);

	// fusion level 6: same result as UpdateStageFusedAllFaces, with a 2.5D sliding window (after Woodward's PPMstar pencil
	// pipelining): a block owns a TX x TY column of cells and marches along z. The z-face flux between consecutive planes is
	// kept in registers, and only the x- and y-faces of the current plane go through LDS. (GPU, 3D; elsewhere it calls
	// UpdateStageFusedAllFaces)
	template <int STAGE, int ORDER, SlopeLimiter limiter, typename Fixup>
	static void UpdateStageFusedMarch(amrex::MultiFab const &primVar_mf, amrex::MultiFab const &chiCell_mf, amrex::MultiFab const &primVarOld_mf,
					  amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf, amrex::MultiFab &stageDiv_mf,
					  amrex::Real stage1Weight, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, double dt, amrex::Real K_visc,
					  amrex::Real shearViscosity, amrex::Real bulkViscosity, amrex::iMultiFab &redoFlag_mf, Fixup const &fixup);

	// fusion level 7: the DIR part of a fused stage in its own kernel. Each block of MT threads owns a tile of MT cells
	// (64 along x, the rest along y for DIR = X1/X2 or along z for DIR = X3); the tile's DIR-faces are computed once into LDS and
	// each cell adds its flux divergence and div v in DIR to the accumulator acc_mf (nvar_ + 1 components): X1 writes it,
	// X2 adds to it, and X3 adds its part and finishes the stage (FinishFusedStage, incl. the fixup). Same per-cell result as
	// UpdateStageFusedAllFaces, with the per-direction kernel size (and register use) of ComputeFluxes. (GPU, 3D only.)
	template <int STAGE, int ORDER, SlopeLimiter limiter, FluxDir DIR, int MT, typename Fixup>
	static void UpdateStageSplitDir(amrex::MultiFab const &primVar_mf, amrex::MultiFab const &chiCell_mf, amrex::MultiFab const &primVarOld_mf,
					amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf, amrex::MultiFab &acc_mf,
					amrex::MultiFab &stageDiv_mf, amrex::Real stage1Weight, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, double dt,
					amrex::Real K_visc, amrex::Real shearViscosity, amrex::Real bulkViscosity, amrex::iMultiFab &redoFlag_mf,
					Fixup const &fixup);

	// fusion levels 4 and 5: per-cell end of a fused stage (stage combination, P dV, forward-Euler update, validity flag)
	// fixup(consVarNew, i, j, k) is applied to valid cells after the validity flag is set (the floors and dual-energy sync of
	// ApplyHydroStateFixup, or a no-op)
	template <int STAGE, typename Fixup>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void
	FinishFusedStage(quokka::valarray<double, nvar_> &D, double div_v, amrex::Array4<amrex::Real> const &stageDiv, amrex::Real stage1Weight,
			 amrex::Array4<const amrex::Real> const &primVarOld, amrex::Array4<const amrex::Real> const &consVarOld,
			 amrex::Array4<amrex::Real> const &consVarNew, amrex::Array4<int> const &redoFlag, int i, int j, int k, double dt, Fixup const &fixup);

	template <int STAGE, int ORDER, SlopeLimiter limiter, typename Fixup>
	static void UpdateStageFusedAllFaces(amrex::MultiFab const &primVar_mf, amrex::MultiFab const &chiCell_mf, amrex::MultiFab const &primVarOld_mf,
					     amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf, amrex::MultiFab &stageDiv_mf,
					     amrex::Real stage1Weight, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, double dt, amrex::Real K_visc,
					     amrex::Real shearViscosity, amrex::Real bulkViscosity, amrex::iMultiFab &redoFlag_mf, Fixup const &fixup);

	// hydro-k01: PPM reconstruction with the shock flattening (FlattenShocks) applied in registers before the single store
	template <FluxDir DIR>
	static void ReconstructStatesPPMFlattened(amrex::MultiFab const &q_mf, amrex::MultiFab const &x1Chi_mf, amrex::MultiFab const &x2Chi_mf,
						  amrex::MultiFab const &x3Chi_mf, amrex::MultiFab &x1LeftState_mf, amrex::MultiFab &x1RightState_mf,
						  int nghost, int nvars);

	// C++ does not allow constexpr to be uninitialized, even in a templated
	// class!
	static constexpr double gamma_ = ::quokka::EOS_Traits<problem_t>::gamma;
	static constexpr double cs_iso_ = ::quokka::EOS_Traits<problem_t>::cs_isothermal;
	static constexpr auto is_eos_isothermal() -> bool { return (gamma_ == 1.0); }

	static constexpr bool reconstruct_eint = HydroSystem_Traits<problem_t>::reconstruct_eint;

	// hydro-05: persistent primitive variables. The primitive MultiFab produced by ConservedToPrimitive may carry
	// nPrimExtra_ extra components after the nvar_ primitive variables:
	//   primPgas_index  = gas pressure ComputePressure(cons, i, j, k, cons_fc) (used by AddInternalEnergyPdV)
	//   primPflat_index = the pressure used by the PPM shock-flattening stencil
	// When reconstruct_eint is false, both are already stored in pressure_index and no extra components are needed.
	static constexpr int nPrimExtra_ = reconstruct_eint ? 2 : 0;
	static constexpr int nPrimVar_ = nvar_ + nPrimExtra_; // number of components to allocate for a persistent primVar MultiFab
	static constexpr int primPgas_index = reconstruct_eint ? nvar_ : static_cast<int>(pressure_index);
	static constexpr int primPflat_index = reconstruct_eint ? (nvar_ + 1) : static_cast<int>(pressure_index);
};

template <typename problem_t>
void HydroSystem<problem_t>::ConservedToPrimitive(amrex::MultiFab const &cons_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf,
						  amrex::MultiFab &primVar_mf, const int nghost)
{
	// convert conserved to primitive variables
	auto const &cons_fc_x0 = cons_fc_mf[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto const &cons_fc_x1 = cons_fc_mf[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto const &cons_fc_x2 = cons_fc_mf[2].const_arrays();
#endif
	auto const &cons_cc = cons_cc_mf.const_arrays();
	auto const &primVar = primVar_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};
	// hydro-05: also store the extra pressure components if the caller allocated them
	const bool storeExtraPressures = (nPrimExtra_ > 0) && (primVar_mf.nComp() >= nPrimVar_);

	amrex::ParallelFor(cons_cc_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		// storeExtraPressures is only read inside `if constexpr` below; nvcc can't first-capture
		// a variable from inside constexpr-if, so touch it unconditionally here first
		[[maybe_unused]] bool const storeExtraPressures_captured = storeExtraPressures;
		// First-capture the magnetic field arrays by accessing them early
		// This forces NVCC to capture them before the constexpr if block
		std::remove_cv_t<std::remove_reference_t<decltype(cons_fc_x0[bx])>> fc_x0_ref{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			// If not wrapped in this if clause, this will fail in Debug mode due to nan values
			fc_x0_ref = cons_fc_x0[bx];
		}
#if AMREX_SPACEDIM >= 2
		std::remove_cv_t<std::remove_reference_t<decltype(cons_fc_x1[bx])>> fc_x1_ref{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			fc_x1_ref = cons_fc_x1[bx];
		}
#endif
#if AMREX_SPACEDIM == 3
		std::remove_cv_t<std::remove_reference_t<decltype(cons_fc_x2[bx])>> fc_x2_ref{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			fc_x2_ref = cons_fc_x2[bx];
		}
#endif

		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
		cons_fc[0] = fc_x0_ref;
#if AMREX_SPACEDIM >= 2
		cons_fc[1] = fc_x1_ref;
#endif
#if AMREX_SPACEDIM == 3
		cons_fc[2] = fc_x2_ref;
#endif
		const amrex::Real rho = cons_cc[bx](i, j, k, density_index);
		const amrex::Real px = cons_cc[bx](i, j, k, x1Momentum_index);
		const amrex::Real py = cons_cc[bx](i, j, k, x2Momentum_index);
		const amrex::Real pz = cons_cc[bx](i, j, k, x3Momentum_index);
		const amrex::Real E = cons_cc[bx](i, j, k, energy_index); // *total* gas energy per unit volume
		const amrex::Real Eint_aux = cons_cc[bx](i, j, k, internalEnergy_index);

		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(px));
		AMREX_ASSERT(!std::isnan(py));
		AMREX_ASSERT(!std::isnan(pz));
		AMREX_ASSERT(!std::isnan(E));

		const amrex::Real vx = px / rho;
		const amrex::Real vy = py / rho;
		const amrex::Real vz = pz / rho;
		const amrex::Real kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
		amrex::Real magnetic_energy = 0.;

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			// note, bx is the box, and bxi is the magnetic-field component
			const amrex::Real b_x0_m = fc_x0_ref(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const amrex::Real b_x0_p = fc_x0_ref(i + 1, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const amrex::Real b_x0 = 0.5 * (b_x0_m + b_x0_p);
#if AMREX_SPACEDIM >= 2
			const amrex::Real b_x1_m = fc_x1_ref(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const amrex::Real b_x1_p = fc_x1_ref(i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const amrex::Real b_x1 = 0.5 * (b_x1_m + b_x1_p);
#endif
#if AMREX_SPACEDIM == 3
			const amrex::Real b_x2_m = fc_x2_ref(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const amrex::Real b_x2_p = fc_x2_ref(i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex);
			const amrex::Real b_x2 = 0.5 * (b_x2_m + b_x2_p);
#endif
			magnetic_energy = 0.5 * (AMREX_D_TERM(b_x0 * b_x0, +b_x1 * b_x1, +b_x2 * b_x2));
		}
		const amrex::Real Eint_cons = E - kinetic_energy - magnetic_energy;

		const amrex::Real Pgas = ComputePressure(cons_cc[bx], i, j, k, &cons_fc);
		const amrex::Real eint_cons = Eint_cons / rho;
		const amrex::Real eint_aux = Eint_aux / rho;

		AMREX_ASSERT(rho > 0.);
		if constexpr (!is_eos_isothermal()) {
			AMREX_ASSERT(Pgas > 0.);
		}

		primVar[bx](i, j, k, primDensity_index) = rho;
		primVar[bx](i, j, k, x1Velocity_index) = vx;
		primVar[bx](i, j, k, x2Velocity_index) = vy;
		primVar[bx](i, j, k, x3Velocity_index) = vz;

		if constexpr (reconstruct_eint) {
			// save specific internal energy (SIE) == (Etot - KE) / rho
			primVar[bx](i, j, k, pressure_index) = eint_cons;
			// save auxiliary specific internal energy (SIE) == Eint_aux / rho
			primVar[bx](i, j, k, primEint_index) = eint_aux;
		} else {
			// save pressure
			primVar[bx](i, j, k, pressure_index) = Pgas;
			// save auxiliary internal energy (rho * e)
			primVar[bx](i, j, k, primEint_index) = Eint_aux;
		}

		// copy any passive scalars
		for (int nc = 0; nc < nscalars_; ++nc) {
			primVar[bx](i, j, k, primScalar0_index + nc) = cons_cc[bx](i, j, k, scalar0_index + nc);
		}

		// handle dust
		if constexpr (Physics_Traits<problem_t>::is_dust_enabled) {
			for (int g = 0; g < Physics_Traits<problem_t>::nDustGroups; ++g) {
				const amrex::Real dust_rho = cons_cc[bx](i, j, k, dustDensity_index + numDustVars_ * g);
				const amrex::Real dust_px = cons_cc[bx](i, j, k, x1DustMomentum_index + numDustVars_ * g);
				const amrex::Real dust_py = cons_cc[bx](i, j, k, x2DustMomentum_index + numDustVars_ * g);
				const amrex::Real dust_pz = cons_cc[bx](i, j, k, x3DustMomentum_index + numDustVars_ * g);

				AMREX_ASSERT(!std::isnan(dust_rho));
				AMREX_ASSERT(!std::isnan(dust_px));
				AMREX_ASSERT(!std::isnan(dust_py));
				AMREX_ASSERT(!std::isnan(dust_pz));

				AMREX_ASSERT(dust_rho > 0.);

				const amrex::Real dust_vx = dust_px / dust_rho;
				const amrex::Real dust_vy = dust_py / dust_rho;
				const amrex::Real dust_vz = dust_pz / dust_rho;

				primVar[bx](i, j, k, primDustDensity_index + numDustVars_ * g) = dust_rho;
				primVar[bx](i, j, k, x1DustVelocity_index + numDustVars_ * g) = dust_vx;
				primVar[bx](i, j, k, x2DustVelocity_index + numDustVars_ * g) = dust_vy;
				primVar[bx](i, j, k, x3DustVelocity_index + numDustVars_ * g) = dust_vz;
			}
		}

		// hydro-05: persistent pressures (only needed when pressure_index holds the specific internal energy)
		if constexpr (nPrimExtra_ > 0) {
			if (storeExtraPressures) {
				// gas pressure, exactly as AddInternalEnergyPdV computed it from the conserved state
				primVar[bx](i, j, k, primPgas_index) = Pgas;
				if constexpr (!is_eos_isothermal()) {
					// flattening pressure, exactly as ComputeFlatteningCoefficients computed it from the primitives:
					// EOS(rho_prim, rho_prim * eint_prim, massScalars read from the primitive array)
					// (all primitive components it reads have been written above by this thread)
					const amrex::Real rho_prim = primVar[bx](i, j, k, primDensity_index);
					const amrex::Real eint_prim = primVar[bx](i, j, k, pressure_index);
					amrex::GpuArray<Real, nmscalars_> massScalars_prim = RadSystem<problem_t>::ComputeMassScalars(primVar[bx], i, j, k);
					primVar[bx](i, j, k, primPflat_index) =
					    ::quokka::EOS<problem_t>::ComputePressure(rho_prim, rho_prim * eint_prim, massScalars_prim);
				}
			}
		}
	});
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
HydroSystem<problem_t>::ComputeCellSignalSpeed(amrex::Array4<const amrex::Real> const &cons_cc, int i, int j, int k,
					       std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc) -> amrex::Real
{
	// First-capture by early access
	const auto rho = cons_cc(i, j, k, density_index);
	const auto px = cons_cc(i, j, k, x1Momentum_index);
	const auto py = cons_cc(i, j, k, x2Momentum_index);
	const auto pz = cons_cc(i, j, k, x3Momentum_index);
	AMREX_ASSERT(!std::isnan(rho));
	AMREX_ASSERT(!std::isnan(px));
	AMREX_ASSERT(!std::isnan(py));
	AMREX_ASSERT(!std::isnan(pz));

	const auto vx = px / rho;
	const auto vy = py / rho;
	const auto vz = pz / rho;
	const double vel_magnitude = std::sqrt(vx * vx + vy * vy + vz * vz);
	double fastest_wavespeed = NAN;

	if (Physics_Traits<problem_t>::is_mhd_enabled) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(cons_fc != nullptr, "ComputeCellSignalSpeed called without face-centered fields for MHD problem");
		amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons_cc, i, j, k);
		const auto total_energy = cons_cc(i, j, k, energy_index); // *total* gas energy per unit volume
		const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
		const auto magnetic_energy = ComputeMagneticEnergy(i, j, k, cons_fc);
		const auto thermal_energy = total_energy - kinetic_energy - magnetic_energy;
		const auto bx1_m = (*cons_fc)[0](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx1_p = (*cons_fc)[0](i + 1, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx2_m = (*cons_fc)[1](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx2_p = (*cons_fc)[1](i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx3_m = (*cons_fc)[2](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx3_p = (*cons_fc)[2](i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx1 = 0.5 * (bx1_m + bx1_p);
		const auto bx2 = 0.5 * (bx2_m + bx2_p);
		const auto bx3 = 0.5 * (bx3_m + bx3_p);
		double b_sq = bx1 * bx1 + bx2 * bx2 + bx3 * bx3;
		const auto pressure = ::quokka::EOS<problem_t>::ComputePressure(rho, thermal_energy, massScalars);
		double gp = ::quokka::EOS<problem_t>::gamma_ * pressure;

		double bgp_p = b_sq + gp;
		double const bgp_m = b_sq - gp;
		fastest_wavespeed = std::max({std::sqrt(0.5 * (bgp_p + std::sqrt(bgp_m * bgp_m + 4.0 * gp * (bx2 * bx2 + bx3 * bx3))) / rho),
					      std::sqrt(0.5 * (bgp_p + std::sqrt(bgp_m * bgp_m + 4.0 * gp * (bx1 * bx1 + bx3 * bx3))) / rho),
					      std::sqrt(0.5 * (bgp_p + std::sqrt(bgp_m * bgp_m + 4.0 * gp * (bx1 * bx1 + bx2 * bx2))) / rho)});
	} else {
		if constexpr (is_eos_isothermal()) {
			fastest_wavespeed = cs_iso_;
		} else {
			fastest_wavespeed = ComputeSoundSpeed(cons_cc, i, j, k, cons_fc);
		}
		AMREX_ASSERT(fastest_wavespeed > 0.);
	}

	double signal_max = fastest_wavespeed + vel_magnitude;

	if constexpr (Physics_Traits<problem_t>::is_dust_enabled) {
		for (int g = 0; g < Physics_Traits<problem_t>::nDustGroups; ++g) {
			const amrex::Real dust_rho = cons_cc(i, j, k, dustDensity_index + g * numDustVars_);
			const amrex::Real dust_px = cons_cc(i, j, k, x1DustMomentum_index + g * numDustVars_);
			const amrex::Real dust_py = cons_cc(i, j, k, x2DustMomentum_index + g * numDustVars_);
			const amrex::Real dust_pz = cons_cc(i, j, k, x3DustMomentum_index + g * numDustVars_);
			AMREX_ASSERT(!std::isnan(dust_rho));
			AMREX_ASSERT(!std::isnan(dust_px));
			AMREX_ASSERT(!std::isnan(dust_py));
			AMREX_ASSERT(!std::isnan(dust_pz));

			const amrex::Real dust_vx = dust_px / dust_rho;
			const amrex::Real dust_vy = dust_py / dust_rho;
			const amrex::Real dust_vz = dust_pz / dust_rho;
			const amrex::Real dust_vel_mag = std::sqrt(dust_vx * dust_vx + dust_vy * dust_vy + dust_vz * dust_vz);
			signal_max = std::max(signal_max, dust_vel_mag + fastest_wavespeed);
		}
	}
	return signal_max;
}

template <typename problem_t>
auto HydroSystem<problem_t>::maxSignalSpeedLocal(amrex::MultiFab const &cons_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf) -> amrex::Real
{
	// return maximum signal speed on local grids
	auto const &cons_fc_x0 = cons_fc_mf[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto const &cons_fc_x1 = cons_fc_mf[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto const &cons_fc_x2 = cons_fc_mf[2].const_arrays();
#endif

	auto const &cons = cons_mf.const_arrays();
	return amrex::ParReduce(amrex::TypeList<amrex::ReduceOpMax>{}, amrex::TypeList<amrex::Real>{}, cons_mf,
				amrex::IntVect(0), // no ghost cells
				[=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real> {
					std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
					if (Physics_Traits<problem_t>::is_mhd_enabled) { // if instead of if constexpr to avoid nvcc issues
						cons_fc[0] = cons_fc_x0[bx];
#if AMREX_SPACEDIM >= 2
						cons_fc[1] = cons_fc_x1[bx];
#endif
#if AMREX_SPACEDIM == 3
						cons_fc[2] = cons_fc_x2[bx];
#endif
					}
					return {ComputeCellSignalSpeed(cons[bx], i, j, k, &cons_fc)};
				});
}

template <typename problem_t>
void HydroSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons_cc,
						   std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const &cons_fc, array_t &maxSignal,
						   amrex::Box const &indexRange)
{
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) { maxSignal(i, j, k) = ComputeCellSignalSpeed(cons_cc, i, j, k, &cons_fc); });
}

template <typename problem_t>
auto HydroSystem<problem_t>::CheckStatesValid(amrex::MultiFab const &cons_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf) -> bool
{
	// check whether density or pressure are negative
	auto const &cons = cons_mf.const_arrays();
	auto const &cons_fc_x0 = cons_fc_mf[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto const &cons_fc_x1 = cons_fc_mf[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto const &cons_fc_x2 = cons_fc_mf[2].const_arrays();
#endif

	return amrex::ParReduce(amrex::TypeList<amrex::ReduceOpLogicalAnd>{}, amrex::TypeList<bool>{}, cons_mf,
				amrex::IntVect(0), // no ghost cells
				[=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<bool> {
					std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
					if (Physics_Traits<problem_t>::is_mhd_enabled) {
						cons_fc[0] = cons_fc_x0[bx];
#if AMREX_SPACEDIM >= 2
						cons_fc[1] = cons_fc_x1[bx];
#endif
#if AMREX_SPACEDIM == 3
						cons_fc[2] = cons_fc_x2[bx];
#endif
					}
					const auto rho = cons[bx](i, j, k, density_index);
					const auto px = cons[bx](i, j, k, x1Momentum_index);
					const auto py = cons[bx](i, j, k, x2Momentum_index);
					const auto pz = cons[bx](i, j, k, x3Momentum_index);
					const auto E = cons[bx](i, j, k, energy_index);
					const auto vx = px / rho;
					const auto vy = py / rho;
					const auto vz = pz / rho;
					const auto magnetic_energy = ComputeMagneticEnergy(i, j, k, &cons_fc);
					const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
					const auto thermal_energy = E - kinetic_energy - magnetic_energy;
					const auto P = ComputePressure(cons[bx], i, j, k, &cons_fc);

					bool negativeDensity = (rho <= 0.);
					bool const negativePressure = (P <= 0.);

					if constexpr (is_eos_isothermal()) {
						if (negativeDensity) {
							printf("invalid state at (%d, %d, %d): rho %g\n", i, j, k, rho);
							return {false};
						}
					} else {
						if (negativeDensity || negativePressure) {
							printf("invalid state at (%d, %d, %d): rho %g, Etot %g, Eint %g, P %g\n", i, j, k, rho, E,
							       thermal_energy, P);
							return {false};
						}
					}
					return {true};
				});
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputePrimVars(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k,
										 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc)
    -> quokka::valarray<amrex::Real, nHydroScalars_>
{
	// convert to primitive vars
	const auto rho = cons(i, j, k, density_index);
	const auto px = cons(i, j, k, x1Momentum_index);
	const auto py = cons(i, j, k, x2Momentum_index);
	const auto pz = cons(i, j, k, x3Momentum_index);
	const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume
	const auto Eint_aux = cons(i, j, k, internalEnergy_index);
	const auto vx = px / rho;
	const auto vy = py / rho;
	const auto vz = pz / rho;
	const auto magnetic_energy = ComputeMagneticEnergy(i, j, k, cons_fc);
	const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
	const auto thermal_energy = E - kinetic_energy - magnetic_energy;

	amrex::Real P = NAN;
	if constexpr (is_eos_isothermal()) {
		P = rho * cs_iso_ * cs_iso_;
	} else {
		amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
		P = ::quokka::EOS<problem_t>::ComputePressure(rho, thermal_energy, massScalars);
	}

	quokka::valarray<amrex::Real, nHydroScalars_> primVars{rho, vx, vy, vz, P, Eint_aux};

	for (int n = 0; n < nscalars_; ++n) {
		primVars[primScalar0_index + n] = cons(i, j, k, scalar0_index + n);
	}
	return primVars;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeConsVars(quokka::valarray<amrex::Real, nHydroScalars_> const &prim)
    -> quokka::valarray<amrex::Real, nHydroScalars_>
{
	// convert to conserved vars
	Real const rho = prim[0];
	Real const v1 = prim[1];
	Real const v2 = prim[2];
	Real const v3 = prim[3];
	Real const P = prim[4];
	Real const Eint_aux = prim[5];

	Real const Eint = ::quokka::EOS<problem_t>::ComputeEintFromPres(rho, P);
	Real const Egas = Eint + 0.5 * rho * (v1 * v1 + v2 * v2 + v3 * v3);

	quokka::valarray<amrex::Real, nHydroScalars_> consVars{rho, rho * v1, rho * v2, rho * v3, Egas, Eint_aux};

	for (int i = 0; i < nscalars_; ++i) {
		consVars[scalar0_index + i] = prim[primScalar0_index + i];
	}
	return consVars;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputePressure(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k,
										 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc)
    -> amrex::Real
{
	const auto rho = cons(i, j, k, density_index);
	const auto px = cons(i, j, k, x1Momentum_index);
	const auto py = cons(i, j, k, x2Momentum_index);
	const auto pz = cons(i, j, k, x3Momentum_index);
	const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume
	const auto vx = px / rho;
	const auto vy = py / rho;
	const auto vz = pz / rho;
	const auto magnetic_energy = ComputeMagneticEnergy(i, j, k, cons_fc);
	const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
	const auto thermal_energy = E - kinetic_energy - magnetic_energy;
	amrex::Real P = NAN;

	if constexpr (is_eos_isothermal()) {
		P = rho * cs_iso_ * cs_iso_;
	} else {
		amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
		P = ::quokka::EOS<problem_t>::ComputePressure(rho, thermal_energy, massScalars);
	}
	return P;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
HydroSystem<problem_t>::ComputeInternalEnergy(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k,
					      std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc) -> amrex::Real
{
	const auto rho = cons(i, j, k, density_index);
	const auto px = cons(i, j, k, x1Momentum_index);
	const auto py = cons(i, j, k, x2Momentum_index);
	const auto pz = cons(i, j, k, x3Momentum_index);
	const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume
	const auto kinetic_energy = 0.5 * (px * px + py * py + pz * pz) / rho;
	const auto magnetic_energy = ComputeMagneticEnergy(i, j, k, cons_fc);
	const auto internal_energy = E - kinetic_energy - magnetic_energy;

	return internal_energy;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeSoundSpeed(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k,
										   std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc)
    -> amrex::Real
{
	const auto rho = cons(i, j, k, density_index);
	const auto Eint = ComputeInternalEnergy(cons, i, j, k, cons_fc);

	amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
	amrex::Real P = ::quokka::EOS<problem_t>::ComputePressure(rho, Eint, massScalars);
	amrex::Real cs = ::quokka::EOS<problem_t>::ComputeSoundSpeed(rho, P, massScalars);

	return cs;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
HydroSystem<problem_t>::ComputeIsothermalSoundSpeed(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k,
						    std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc) -> amrex::Real
{
	const auto rho = cons(i, j, k, density_index);
	const auto px = cons(i, j, k, x1Momentum_index);
	const auto py = cons(i, j, k, x2Momentum_index);
	const auto pz = cons(i, j, k, x3Momentum_index);
	const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume
	const auto vx = px / rho;
	const auto vy = py / rho;
	const auto vz = pz / rho;
	const auto magnetic_energy = ComputeMagneticEnergy(i, j, k, cons_fc);
	const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
	const auto thermal_energy = E - kinetic_energy - magnetic_energy;

	amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
	amrex::Real P = ::quokka::EOS<problem_t>::ComputePressure(rho, thermal_energy, massScalars);
	amrex::Real cs = ::quokka::EOS<problem_t>::ComputeIsothermalSoundSpeed(rho, P);

	return cs;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
HydroSystem<problem_t>::ComputeMagneticEnergy(int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc) -> amrex::Real
{
	if constexpr (!Physics_Traits<problem_t>::is_mhd_enabled) {
		return 0.0;
	} else {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(cons_fc != nullptr, "ComputeMagneticEnergy called without face-centered fields for MHD problem");

		const auto bx1_m = (*cons_fc)[0](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx1_p = (*cons_fc)[0](i + 1, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx2_m = (*cons_fc)[1](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx2_p = (*cons_fc)[1](i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx3_m = (*cons_fc)[2](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx3_p = (*cons_fc)[2](i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx1 = 0.5 * (bx1_m + bx1_p);
		const auto bx2 = 0.5 * (bx2_m + bx2_p);
		const auto bx3 = 0.5 * (bx3_m + bx3_p);
		const auto b_sq = bx1 * bx1 + bx2 * bx2 + bx3 * bx3;
		return 0.5 * b_sq;
	}
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputePlasmaBeta(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k,
										   std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc)
    -> amrex::Real
{
	if constexpr (!Physics_Traits<problem_t>::is_mhd_enabled) {
		return std::numeric_limits<amrex::Real>::max();
	} else {
		const amrex::Real pressure_thermal = ComputePressure(cons, i, j, k, cons_fc);
		const amrex::Real magnetic_energy = ComputeMagneticEnergy(i, j, k, cons_fc);
		if (magnetic_energy <= 0.0) {
			return std::numeric_limits<amrex::Real>::max();
		}
		return pressure_thermal / magnetic_energy;
	}
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeVelocityX1(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
    -> amrex::Real
{
	amrex::Real const rho = cons(i, j, k, density_index);
	amrex::Real const vel_x = cons(i, j, k, x1Momentum_index) / rho;
	return vel_x;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeVelocityX2(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
    -> amrex::Real
{
	amrex::Real const rho = cons(i, j, k, density_index);
	amrex::Real const vel_y = cons(i, j, k, x2Momentum_index) / rho;
	return vel_y;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeVelocityX3(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
    -> amrex::Real
{
	amrex::Real const rho = cons(i, j, k, density_index);
	amrex::Real const vel_z = cons(i, j, k, x3Momentum_index) / rho;
	return vel_z;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool
{
	// check if cons(i, j, k) is a valid state
	const amrex::Real rho = cons(i, j, k, density_index);
	bool isDensityPositive = (rho > 0.);

	if constexpr (Physics_Traits<problem_t>::is_dust_enabled) {
		for (int g = 0; g < Physics_Traits<problem_t>::nDustGroups; ++g) {
			const amrex::Real dust_rho = cons(i, j, k, dustDensity_index + g * numDustVars_);
			isDensityPositive = isDensityPositive && (dust_rho > 0.);
		}
	}

	bool isMassScalarPositive = true;
	if constexpr (nmscalars_ > 0) {
		amrex::GpuArray<Real, nmscalars_> massScalars_ = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);

		for (int idx = 0; idx < nmscalars_; ++idx) {
			if (massScalars_[idx] < 0.0) {
				isMassScalarPositive = false;
				break; // Exit the loop early if any element is not positive
			}
		}
	}
	// when the dual energy method is used, we *cannot* reset on pressure
	// failures. on the other hand, we don't need to -- the auxiliary internal
	// energy is used instead!

	return isDensityPositive && isMassScalarPositive;
}

template <typename problem_t>
void HydroSystem<problem_t>::ComputeRhsFromFluxes(amrex::MultiFab &rhs_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray,
						  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const int nvars)
{
	// compute the total right-hand-side for the MOL integration

	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const x1Flux = fluxArray[0].const_arrays();
#if (AMREX_SPACEDIM >= 2)
	auto const x2Flux = fluxArray[1].const_arrays();
#endif
#if (AMREX_SPACEDIM == 3)
	auto const x3Flux = fluxArray[2].const_arrays();
#endif
	auto rhs = rhs_mf.arrays();

	amrex::ParallelFor(rhs_mf, amrex::IntVect{0}, nvars, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) noexcept {
		rhs[bx](i, j, k, n) = AMREX_D_TERM((1.0 / dx[0]) * (x1Flux[bx](i, j, k, n) - x1Flux[bx](i + 1, j, k, n)),
						   +(1.0 / dx[1]) * (x2Flux[bx](i, j, k, n) - x2Flux[bx](i, j + 1, k, n)),
						   +(1.0 / dx[2]) * (x3Flux[bx](i, j, k, n) - x3Flux[bx](i, j, k + 1, n)));
	});
}

template <typename problem_t>
void HydroSystem<problem_t>::PredictStep(amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf, amrex::MultiFab const &rhs_mf, const double dt,
					 const int nvars, amrex::iMultiFab &redoFlag_mf)
{
	const BL_PROFILE("HydroSystem::PredictStep()");

	auto const &consVarOld = consVarOld_mf.const_arrays();
	auto const &rhs = rhs_mf.const_arrays();
	auto consVarNew = consVarNew_mf.arrays();
	auto redoFlag = redoFlag_mf.arrays();

	amrex::ParallelFor(consVarNew_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		for (int n = 0; n < nvars; ++n) {
			consVarNew[bx](i, j, k, n) = consVarOld[bx](i, j, k, n) + dt * rhs[bx](i, j, k, n);
		}
		// check if state is valid -- flag for re-do if not
		if (!isStateValid(consVarNew[bx], i, j, k)) {
			redoFlag[bx](i, j, k) = quokka::redoFlag::redo;
		} else {
			redoFlag[bx](i, j, k) = quokka::redoFlag::none;
		}
	});
}

template <typename problem_t>
void HydroSystem<problem_t>::AddFluxesRK2(amrex::MultiFab &Unew_mf, amrex::MultiFab const &U0_mf, amrex::MultiFab const &U1_mf, amrex::MultiFab const &rhs_mf,
					  const double dt, const int nvars, amrex::iMultiFab &redoFlag_mf)
{
	const BL_PROFILE("HydroSystem::AddFluxesRK2()");

	auto const &U0 = U0_mf.const_arrays();
	auto const &U1 = U1_mf.const_arrays();
	auto const &rhs = rhs_mf.const_arrays();
	auto U_new = Unew_mf.arrays();
	auto redoFlag = redoFlag_mf.arrays();

	amrex::ParallelFor(Unew_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		for (int n = 0; n < nvars; ++n) {
			// RK-SSP2 integrator
			const double U_0 = U0[bx](i, j, k, n);
			const double U_1 = U1[bx](i, j, k, n);
			const double FU = dt * rhs[bx](i, j, k, n);

			// save results in U_new
			U_new[bx](i, j, k, n) = (0.5 * U_0 + 0.5 * U_1) + 0.5 * FU;
		}

		// check if state is valid -- flag for re-do if not
		if (!isStateValid(U_new[bx], i, j, k)) {
			redoFlag[bx](i, j, k) = quokka::redoFlag::redo;
		} else {
			redoFlag[bx](i, j, k) = quokka::redoFlag::none;
		}
	});
}

template <typename problem_t>
template <FluxDir DIR>
void HydroSystem<problem_t>::ComputeFlatteningCoefficients(amrex::MultiFab const &primVar_mf, amrex::MultiFab &x1Chi_mf, const int nghost)
{
	// compute the PPM shock flattening coefficient following
	//   Appendix B1 of Mignone+ 2005 [this description has typos].
	// Method originally from Miller & Colella,
	//   Journal of Computational Physics 183, 26–82 (2002) [no typos].

	constexpr double beta_max = 0.85;
	constexpr double beta_min = 0.75;
	constexpr double Zmax = 0.75;
	constexpr double Zmin = 0.25;

	auto const &primVar_in = primVar_mf.const_arrays();
	auto x1Chi_in = x1Chi_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	// cell-centered kernel
	amrex::ParallelFor(primVar_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in) {
		quokka::Array4View<const amrex::Real, DIR> primVar(primVar_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1Chi(x1Chi_in[bx]);
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		amrex::Real Pplus2 = primVar(i + 2, j, k, pressure_index);
		amrex::Real Pplus1 = primVar(i + 1, j, k, pressure_index);
		amrex::Real P = primVar(i, j, k, pressure_index);
		amrex::Real Pminus1 = primVar(i - 1, j, k, pressure_index);
		amrex::Real Pminus2 = primVar(i - 2, j, k, pressure_index);

		if constexpr (reconstruct_eint) {
			// compute (rho e) (gamma - 1)
			amrex::GpuArray<Real, nmscalars_> massScalars_plus2 = RadSystem<problem_t>::ComputeMassScalars(primVar, i + 2, j, k);
			Pplus2 = ::quokka::EOS<problem_t>::ComputePressure(primVar(i + 2, j, k, primDensity_index),
									   primVar(i + 2, j, k, primDensity_index) * Pplus2, massScalars_plus2);
			amrex::GpuArray<Real, nmscalars_> massScalars_plus1 = RadSystem<problem_t>::ComputeMassScalars(primVar, i + 1, j, k);
			Pplus1 = ::quokka::EOS<problem_t>::ComputePressure(primVar(i + 1, j, k, primDensity_index),
									   primVar(i + 1, j, k, primDensity_index) * Pplus1, massScalars_plus1);
			amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(primVar, i, j, k);
			P = ::quokka::EOS<problem_t>::ComputePressure(primVar(i, j, k, primDensity_index), primVar(i, j, k, primDensity_index) * P,
								      massScalars);
			amrex::GpuArray<Real, nmscalars_> massScalars_minus1 = RadSystem<problem_t>::ComputeMassScalars(primVar, i - 1, j, k);
			Pminus1 = ::quokka::EOS<problem_t>::ComputePressure(primVar(i - 1, j, k, primDensity_index),
									    primVar(i - 1, j, k, primDensity_index) * Pminus1, massScalars_minus1);
			amrex::GpuArray<Real, nmscalars_> massScalars_minus2 = RadSystem<problem_t>::ComputeMassScalars(primVar, i - 2, j, k);
			Pminus2 = ::quokka::EOS<problem_t>::ComputePressure(primVar(i - 2, j, k, primDensity_index),
									    primVar(i - 2, j, k, primDensity_index) * Pminus2, massScalars_minus2);
		}

		if constexpr (is_eos_isothermal()) {
			const amrex::Real cs_sq = cs_iso_ * cs_iso_;
			Pplus2 = primVar(i + 2, j, k, primDensity_index) * cs_sq;
			Pplus1 = primVar(i + 1, j, k, primDensity_index) * cs_sq;
			P = primVar(i, j, k, primDensity_index) * cs_sq;
			Pminus1 = primVar(i - 1, j, k, primDensity_index) * cs_sq;
			Pminus2 = primVar(i - 2, j, k, primDensity_index) * cs_sq;
		}

		// beta is a measure of shock resolution (Eq. 74 of Miller & Colella 2002)
		// Miller & Collela note: "If beta is 1/2, then pressure is linear across
		//   four computational cells. If beta is small enough, then we assume that
		//   any discontinuity is already sufficiently well resolved that additional
		//   dissipation (flattening) is not required."
		const double beta_denom = std::abs(Pplus2 - Pminus2);
		// avoid division by zero (in this case, chi = 1 anyway)
		const double beta = (beta_denom != 0) ? (std::abs(Pplus1 - Pminus1) / beta_denom) : 0;

		// Eq. 75 of Miller & Colella 2002
		const double chi_min = std::max(0., std::min(1., (beta_max - beta) / (beta_max - beta_min)));

		// Z is a measure of shock strength (Eq. 76 of Miller & Colella 2002)
		amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(primVar, i, j, k);
		double K_S = std::pow(::quokka::EOS<problem_t>::ComputeSoundSpeed(primVar(i, j, k, primDensity_index), P, massScalars), 2) *
			     primVar(i, j, k, primDensity_index);
		if constexpr (is_eos_isothermal()) {
			K_S = primVar(i, j, k, primDensity_index) * cs_iso_ * cs_iso_;
		}

		const double Z = std::abs(Pplus1 - Pminus1) / K_S;

		// check for converging flow along the normal direction DIR (Eq. 77)
		int velocity_index = 0;
		if constexpr (DIR == FluxDir::X1) {
			velocity_index = x1Velocity_index;
		} else if constexpr (DIR == FluxDir::X2) {
			velocity_index = x2Velocity_index;
		} else if constexpr (DIR == FluxDir::X3) {
			velocity_index = x3Velocity_index;
		}
		double chi = 1.0;
		if (primVar(i + 1, j, k, velocity_index) < primVar(i - 1, j, k, velocity_index)) {
			chi = std::max(chi_min, std::min(1., (Zmax - Z) / (Zmax - Zmin)));
		}

		x1Chi(i, j, k) = chi;
	});
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::FlatteningKS(amrex::Array4<const amrex::Real> const &primVar, int i, int j, int k)
    -> amrex::Real
{
	// Eq. 76 of Miller & Colella 2002 (denominator of Z); identical to ComputeFlatteningCoefficients
	double K_S = NAN;
	if constexpr (is_eos_isothermal()) {
		K_S = primVar(i, j, k, primDensity_index) * cs_iso_ * cs_iso_;
	} else {
		const amrex::Real P = primVar(i, j, k, primPflat_index);
		amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(primVar, i, j, k);
		K_S = std::pow(::quokka::EOS<problem_t>::ComputeSoundSpeed(primVar(i, j, k, primDensity_index), P, massScalars), 2) *
		      primVar(i, j, k, primDensity_index);
	}
	return K_S;
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::FlatteningChi(amrex::Array4<const amrex::Real> const &primVar_in, int i_in, int j_in, int k_in,
									       amrex::Real K_S) -> amrex::Real
{
	// same arithmetic as ComputeFlatteningCoefficients<DIR>, with the pressure read from the stored primPflat_index
	constexpr double beta_max = 0.85;
	constexpr double beta_min = 0.75;
	constexpr double Zmax = 0.75;
	constexpr double Zmin = 0.25;

	quokka::Array4View<const amrex::Real, DIR> primVar(primVar_in);
	auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

	amrex::Real Pplus2 = NAN;
	amrex::Real Pplus1 = NAN;
	amrex::Real Pminus1 = NAN;
	amrex::Real Pminus2 = NAN;
	if constexpr (is_eos_isothermal()) {
		const amrex::Real cs_sq = cs_iso_ * cs_iso_;
		Pplus2 = primVar(i + 2, j, k, primDensity_index) * cs_sq;
		Pplus1 = primVar(i + 1, j, k, primDensity_index) * cs_sq;
		Pminus1 = primVar(i - 1, j, k, primDensity_index) * cs_sq;
		Pminus2 = primVar(i - 2, j, k, primDensity_index) * cs_sq;
	} else {
		Pplus2 = primVar(i + 2, j, k, primPflat_index);
		Pplus1 = primVar(i + 1, j, k, primPflat_index);
		Pminus1 = primVar(i - 1, j, k, primPflat_index);
		Pminus2 = primVar(i - 2, j, k, primPflat_index);
	}

	// beta is a measure of shock resolution (Eq. 74 of Miller & Colella 2002)
	const double beta_denom = std::abs(Pplus2 - Pminus2);
	// avoid division by zero (in this case, chi = 1 anyway)
	const double beta = (beta_denom != 0) ? (std::abs(Pplus1 - Pminus1) / beta_denom) : 0;

	// Eq. 75 of Miller & Colella 2002
	const double chi_min = std::max(0., std::min(1., (beta_max - beta) / (beta_max - beta_min)));

	// Z is a measure of shock strength (Eq. 76 of Miller & Colella 2002)
	const double Z = std::abs(Pplus1 - Pminus1) / K_S;

	// check for converging flow along the normal direction DIR (Eq. 77)
	int velocity_index = 0;
	if constexpr (DIR == FluxDir::X1) {
		velocity_index = x1Velocity_index;
	} else if constexpr (DIR == FluxDir::X2) {
		velocity_index = x2Velocity_index;
	} else if constexpr (DIR == FluxDir::X3) {
		velocity_index = x3Velocity_index;
	}
	double chi = 1.0;
	if (primVar(i + 1, j, k, velocity_index) < primVar(i - 1, j, k, velocity_index)) {
		chi = std::max(chi_min, std::min(1., (Zmax - Z) / (Zmax - Zmin)));
	}
	return chi;
}

template <typename problem_t>
void HydroSystem<problem_t>::ComputeFlatteningCoefficientsAllDirs(amrex::MultiFab const &primVar_mf, std::array<amrex::MultiFab, 3> &flatCoefs,
								  const int nghost)
{
	// PPM shock flattening coefficients (Miller & Colella 2002) for all directions in one kernel,
	// from the pressure stored once per stage in primVar_mf (component primPflat_index)
	auto const &primVar_in = primVar_mf.const_arrays();
	auto x1Chi_in = flatCoefs[0].arrays();
#if AMREX_SPACEDIM >= 2
	auto x2Chi_in = flatCoefs[1].arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto x3Chi_in = flatCoefs[2].arrays();
#endif
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	amrex::ParallelFor(primVar_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		amrex::Array4<const amrex::Real> const &primVar = primVar_in[bx];
		const amrex::Real K_S = FlatteningKS(primVar, i, j, k);

		{
			quokka::Array4View<amrex::Real, FluxDir::X1> const x1Chi(x1Chi_in[bx]);
			auto [ii, jj, kk] = quokka::reorderMultiIndex<FluxDir::X1>(i, j, k);
			x1Chi(ii, jj, kk) = FlatteningChi<FluxDir::X1>(primVar, i, j, k, K_S);
		}
#if AMREX_SPACEDIM >= 2
		{
			quokka::Array4View<amrex::Real, FluxDir::X2> const x2Chi(x2Chi_in[bx]);
			auto [ii, jj, kk] = quokka::reorderMultiIndex<FluxDir::X2>(i, j, k);
			x2Chi(ii, jj, kk) = FlatteningChi<FluxDir::X2>(primVar, i, j, k, K_S);
		}
#endif
#if AMREX_SPACEDIM == 3
		{
			quokka::Array4View<amrex::Real, FluxDir::X3> const x3Chi(x3Chi_in[bx]);
			auto [ii, jj, kk] = quokka::reorderMultiIndex<FluxDir::X3>(i, j, k);
			x3Chi(ii, jj, kk) = FlatteningChi<FluxDir::X3>(primVar, i, j, k, K_S);
		}
#endif
	});
}

template <typename problem_t>
template <FluxDir DIR>
void HydroSystem<problem_t>::FlattenShocks(amrex::MultiFab const &q_mf, amrex::MultiFab const &x1Chi_mf, amrex::MultiFab const &x2Chi_mf,
					   amrex::MultiFab const &x3Chi_mf, amrex::MultiFab &x1LeftState_mf, amrex::MultiFab &x1RightState_mf, const int nghost,
					   const int nvars)
{
	// Apply shock flattening based on Miller & Colella (2002)
	// [This is necessary to get a reasonable solution to the slow-moving
	// shock problem, and reduces post-shock oscillations in other cases.]

	auto const &q_in = q_mf.const_arrays();
	auto const &x1Chi_in = x1Chi_mf.const_arrays();
	auto const &x2Chi_in = x2Chi_mf.const_arrays();
	auto const &x3Chi_in = x3Chi_mf.const_arrays();
	auto x1LeftState_in = x1LeftState_mf.arrays();
	auto x1RightState_in = x1RightState_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	// cell-centered kernel
	amrex::ParallelFor(q_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in) {
		quokka::Array4View<const amrex::Real, DIR> q(q_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1LeftState(x1LeftState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1RightState(x1RightState_in[bx]);

		// compute coefficient as the minimum from adjacent cells along *each
		// axis*
		//  (Eq. 86 of Miller & Colella 2001; Eq. 78 of Miller & Colella 2002)
		double const chi_ijk = std::min({
		    x1Chi_in[bx](i_in - 1, j_in, k_in),
		    x1Chi_in[bx](i_in, j_in, k_in),
		    x1Chi_in[bx](i_in + 1, j_in, k_in),
#if (AMREX_SPACEDIM >= 2)
		    x2Chi_in[bx](i_in, j_in - 1, k_in),
		    x2Chi_in[bx](i_in, j_in, k_in),
		    x2Chi_in[bx](i_in, j_in + 1, k_in),
#endif
#if (AMREX_SPACEDIM == 3)
		    x3Chi_in[bx](i_in, j_in, k_in - 1),
		    x3Chi_in[bx](i_in, j_in, k_in),
		    x3Chi_in[bx](i_in, j_in, k_in + 1),
#endif
		});

		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		for (int n = 0; n < nvars; ++n) {
			// get interfaces
			const double a_minus = x1RightState(i, j, k, n);
			const double a_plus = x1LeftState(i + 1, j, k, n);
			const double a_mean = q(i, j, k, n);

			// left side of zone i (Eq. 70a)
			const double new_a_minus = chi_ijk * a_minus + (1. - chi_ijk) * a_mean;

			// right side of zone i (Eq. 70b)
			const double new_a_plus = chi_ijk * a_plus + (1. - chi_ijk) * a_mean;

			x1RightState(i, j, k, n) = new_a_minus;
			x1LeftState(i + 1, j, k, n) = new_a_plus;
		}
	});

#if (AMREX_SPACEDIM < 2)
	amrex::ignore_unused(x2Chi_in);
#endif
#if (AMREX_SPACEDIM < 3)
	amrex::ignore_unused(x3Chi_in);
#endif
}

template <typename problem_t>
template <FluxDir DIR>
void HydroSystem<problem_t>::ReconstructStatesPPMFlattened(amrex::MultiFab const &q_mf, amrex::MultiFab const &x1Chi_mf, amrex::MultiFab const &x2Chi_mf,
							   amrex::MultiFab const &x3Chi_mf, amrex::MultiFab &x1LeftState_mf, amrex::MultiFab &x1RightState_mf,
							   const int nghost, const int nvars)
{
	// hydro-k01: fused ReconstructStatesPPM<DIR> + FlattenShocks<DIR>. Same iteration space (cells of q_mf grown by nghost),
	// same chi_ijk (min over the 9 stored chi values, computed once per cell instead of once per component), and the same
	// blend formula and operation order as FlattenShocks, applied to the PPM edge values in registers; each (cell, n)
	// thread is the only writer of rightState(i) and leftState(i + 1), so the result is identical to the two-kernel version.
	const BL_PROFILE("HydroSystem::ReconstructStatesPPMFlattened()");

	auto const &q_in = q_mf.const_arrays();
	auto const &x1Chi_in = x1Chi_mf.const_arrays();
	auto const &x2Chi_in = x2Chi_mf.const_arrays();
	auto const &x3Chi_in = x3Chi_mf.const_arrays();
	auto x1LeftState_in = x1LeftState_mf.arrays();
	auto x1RightState_in = x1RightState_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	// cell-centered kernel
	amrex::ParallelFor(q_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in) noexcept {
		quokka::Array4View<const amrex::Real, DIR> q(q_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1LeftState(x1LeftState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1RightState(x1RightState_in[bx]);

		// compute coefficient as the minimum from adjacent cells along *each
		// axis*
		//  (Eq. 86 of Miller & Colella 2001; Eq. 78 of Miller & Colella 2002)
		double const chi_ijk = std::min({
		    x1Chi_in[bx](i_in - 1, j_in, k_in),
		    x1Chi_in[bx](i_in, j_in, k_in),
		    x1Chi_in[bx](i_in + 1, j_in, k_in),
#if (AMREX_SPACEDIM >= 2)
		    x2Chi_in[bx](i_in, j_in - 1, k_in),
		    x2Chi_in[bx](i_in, j_in, k_in),
		    x2Chi_in[bx](i_in, j_in + 1, k_in),
#endif
#if (AMREX_SPACEDIM == 3)
		    x3Chi_in[bx](i_in, j_in, k_in - 1),
		    x3Chi_in[bx](i_in, j_in, k_in),
		    x3Chi_in[bx](i_in, j_in, k_in + 1),
#endif
		});

		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		for (int n = 0; n < nvars; ++n) {
			// PPM interfaces of zone i (before flattening)
			const std::pair<double, double> edges = HyperbolicSystem<problem_t>::template PPMInterfaceValues<DIR>(q, n, i, j, k);
			const double a_minus = edges.first;
			const double a_plus = edges.second;
			const double a_mean = q(i, j, k, n);

			// left side of zone i (Eq. 70a)
			const double new_a_minus = chi_ijk * a_minus + (1. - chi_ijk) * a_mean;

			// right side of zone i (Eq. 70b)
			const double new_a_plus = chi_ijk * a_plus + (1. - chi_ijk) * a_mean;

			x1RightState(i, j, k, n) = new_a_minus;
			x1LeftState(i + 1, j, k, n) = new_a_plus;
		}
	});

#if (AMREX_SPACEDIM < 2)
	amrex::ignore_unused(x2Chi_in);
#endif
#if (AMREX_SPACEDIM < 3)
	amrex::ignore_unused(x3Chi_in);
#endif
}

template <typename problem_t>
void HydroSystem<problem_t>::ComputeFlatteningChiCell(amrex::MultiFab const &primVar_mf, amrex::MultiFab &chiCell_mf, const int nghost)
{
	// chi_ijk = min of the directional coefficients of the cell and its two neighbours along each axis
	// (Eq. 86 of Miller & Colella 2001; Eq. 78 of Miller & Colella 2002). The directional coefficients are evaluated
	// inline (same arithmetic as ComputeFlatteningCoefficientsAllDirs), so no per-direction arrays are stored.
	const BL_PROFILE("HydroSystem::ComputeFlatteningChiCell()");

	auto const &primVar_in = primVar_mf.const_arrays();
	auto chiCell_out = chiCell_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	quokka::ParallelForHydro(chiCell_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Array4<const amrex::Real> const &q = primVar_in[bx];
		auto KS = [&](int ii, int jj, int kk) { return FlatteningKS(q, ii, jj, kk); };
		const amrex::Real KS_c = KS(i, j, k);
		amrex::Real chi = std::min({FlatteningChi<FluxDir::X1>(q, i - 1, j, k, KS(i - 1, j, k)), FlatteningChi<FluxDir::X1>(q, i, j, k, KS_c),
					    FlatteningChi<FluxDir::X1>(q, i + 1, j, k, KS(i + 1, j, k))});
#if AMREX_SPACEDIM >= 2
		chi = std::min({chi, FlatteningChi<FluxDir::X2>(q, i, j - 1, k, KS(i, j - 1, k)), FlatteningChi<FluxDir::X2>(q, i, j, k, KS_c),
				FlatteningChi<FluxDir::X2>(q, i, j + 1, k, KS(i, j + 1, k))});
#endif
#if AMREX_SPACEDIM == 3
		chi = std::min({chi, FlatteningChi<FluxDir::X3>(q, i, j, k - 1, KS(i, j, k - 1)), FlatteningChi<FluxDir::X3>(q, i, j, k, KS_c),
				FlatteningChi<FluxDir::X3>(q, i, j, k + 1, KS(i, j, k + 1))});
#endif
		chiCell_out[bx](i, j, k) = chi;
	});
}

template <typename problem_t>
template <FluxDir DIR, int ORDER, SlopeLimiter limiter>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::FlattenedEdges(quokka::Array4View<const amrex::Real, DIR> const &q, const int n, const int i,
										const int j, const int k, const amrex::Real chi) -> std::pair<double, double>
{
	static_assert(ORDER == 2 || ORDER == 3, "FlattenedEdges supports PLM (2) and PPM (3)");
	const double a_mean = q(i, j, k, n);
	double a_minus = NAN;
	double a_plus = NAN;
	if constexpr (ORDER == 3) {
		const std::pair<double, double> edges = HyperbolicSystem<problem_t>::template PPMInterfaceValues<DIR>(q, n, i, j, k);
		a_minus = edges.first;
		a_plus = edges.second;
	} else {
		// same as the pointwise ReconstructStatesPLM
		const auto slope =
		    HyperbolicSystem<problem_t>::template SlopeFunc<limiter>(q(i + 1, j, k, n) - q(i, j, k, n), q(i, j, k, n) - q(i - 1, j, k, n));
		a_plus = q(i, j, k, n) + 0.5 * slope;
		a_minus = q(i, j, k, n) - 0.5 * slope;
	}
	// same blend as FlattenShocks (Eqs. 70a and 70b of Miller & Colella 2002)
	return std::make_pair(chi * a_minus + (1. - chi) * a_mean, chi * a_plus + (1. - chi) * a_mean);
}

template <typename problem_t>
template <FluxDir DIR, int ORDER, SlopeLimiter limiter>
void HydroSystem<problem_t>::ReconstructStatesFlattened(amrex::MultiFab const &q_mf, amrex::MultiFab const &chiCell_mf, amrex::MultiFab &x1LeftState_mf,
							amrex::MultiFab &x1RightState_mf, const int nghost, const int nvars)
{
	const BL_PROFILE("HydroSystem::ReconstructStatesFlattened()");

	auto const &q_in = q_mf.const_arrays();
	auto const &chiCell_in = chiCell_mf.const_arrays();
	auto x1LeftState_in = x1LeftState_mf.arrays();
	auto x1RightState_in = x1RightState_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	amrex::ParallelFor(q_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in) noexcept {
		quokka::Array4View<const amrex::Real, DIR> q(q_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1LeftState(x1LeftState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1RightState(x1RightState_in[bx]);
		const amrex::Real chi = chiCell_in[bx](i_in, j_in, k_in);
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);
		for (int n = 0; n < nvars; ++n) {
			const std::pair<double, double> edges = FlattenedEdges<DIR, ORDER, limiter>(q, n, i, j, k, chi);
			x1RightState(i, j, k, n) = edges.first;
			x1LeftState(i + 1, j, k, n) = edges.second;
		}
	});
}

template <typename problem_t>
template <typename Fixup>
void HydroSystem<problem_t>::UpdateStageFused(amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf,
					      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray,
					      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArray, amrex::MultiFab const &primVar_mf,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const double dt, const int nvars, amrex::iMultiFab &redoFlag_mf,
					      const bool useRedoFlag, Fixup const &fixup)
{
	// ComputeRhsFromFluxes + AddInternalEnergyPdV + PredictStep in one kernel, with the same per-cell arithmetic.
	// If useRedoFlag is false, every cell is treated as redoFlag::none for the P dV term (as after redoFlag.setVal(none)).
	const BL_PROFILE("HydroSystem::UpdateStageFused()");
	AMREX_ALWAYS_ASSERT(primVar_mf.nComp() > primPgas_index);

	auto const x1Flux = fluxArray[0].const_arrays();
	auto const vel_x = faceVelArray[0].const_arrays();
#if (AMREX_SPACEDIM >= 2)
	auto const x2Flux = fluxArray[1].const_arrays();
	auto const vel_y = faceVelArray[1].const_arrays();
#endif
#if (AMREX_SPACEDIM == 3)
	auto const x3Flux = fluxArray[2].const_arrays();
	auto const vel_z = faceVelArray[2].const_arrays();
#endif
	auto const &consVarOld = consVarOld_mf.const_arrays();
	auto const &primVar = primVar_mf.const_arrays();
	auto consVarNew = consVarNew_mf.arrays();
	auto redoFlag = redoFlag_mf.arrays();

	quokka::ParallelForHydro(consVarNew_mf, amrex::IntVect(0), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// P dV source term for the internal energy equation (as AddInternalEnergyPdV)
		const amrex::Real Pgas = primVar[bx](i, j, k, primPgas_index);
		amrex::Real div_v = NAN;
		if (!useRedoFlag || redoFlag[bx](i, j, k) == quokka::redoFlag::none) {
			div_v = AMREX_D_TERM((vel_x[bx](i + 1, j, k) - vel_x[bx](i, j, k)) / dx[0], +(vel_y[bx](i, j + 1, k) - vel_y[bx](i, j, k)) / dx[1],
					     +(vel_z[bx](i, j, k + 1) - vel_z[bx](i, j, k)) / dx[2]);
		} else {
			div_v =
			    0.5 * (AMREX_D_TERM((ComputeVelocityX1(consVarOld[bx], i + 1, j, k) - ComputeVelocityX1(consVarOld[bx], i - 1, j, k)) / dx[0],
						+(ComputeVelocityX2(consVarOld[bx], i, j + 1, k) - ComputeVelocityX2(consVarOld[bx], i, j - 1, k)) / dx[1],
						+(ComputeVelocityX3(consVarOld[bx], i, j, k + 1) - ComputeVelocityX3(consVarOld[bx], i, j, k - 1)) / dx[2]));
		}

		// All loads first, then the stores: the Array4 pointers are not restrict-qualified, so interleaving them would
		// serialise each component's loads behind the previous component's store. Same arithmetic per component.
		AMREX_ASSERT(nvars <= nvar_);
		amrex::Real uNew[nvar_];
		for (int n = 0; n < nvar_; ++n) {
			if (n >= nvars) {
				break;
			}
			// flux divergence (as ComputeRhsFromFluxes)
			amrex::Real rhs = AMREX_D_TERM((1.0 / dx[0]) * (x1Flux[bx](i, j, k, n) - x1Flux[bx](i + 1, j, k, n)),
						       +(1.0 / dx[1]) * (x2Flux[bx](i, j, k, n) - x2Flux[bx](i, j + 1, k, n)),
						       +(1.0 / dx[2]) * (x3Flux[bx](i, j, k, n) - x3Flux[bx](i, j, k + 1, n)));
			if (n == internalEnergy_index) {
				rhs += -Pgas * div_v;
			}
			// forward-Euler predict (as PredictStep)
			uNew[n] = consVarOld[bx](i, j, k, n) + dt * rhs;
		}
		for (int n = 0; n < nvar_; ++n) {
			if (n >= nvars) {
				break;
			}
			consVarNew[bx](i, j, k, n) = uNew[n];
		}

		// check if state is valid -- flag for re-do if not
		if (!isStateValid(consVarNew[bx], i, j, k)) {
			redoFlag[bx](i, j, k) = quokka::redoFlag::redo;
		} else {
			redoFlag[bx](i, j, k) = quokka::redoFlag::none;
			fixup(consVarNew[bx], i, j, k);
		}
	});
}

template <typename problem_t>
template <FluxDir DIR, int ORDER, SlopeLimiter limiter>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
HydroSystem<problem_t>::FusedFaceFlux(amrex::Array4<const amrex::Real> const &primVar, amrex::Array4<const amrex::Real> const &chiCell, const int i_in,
				      const int j_in, const int k_in, const amrex::Real K_visc, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
				      const amrex::Real shearViscosity, const amrex::Real bulkViscosity) -> FaceFlux
{
	quokka::Array4View<const amrex::Real, DIR> q(primVar);
	quokka::Array4View<const amrex::Real, DIR> chi(chiCell);
	auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

	double regLeft[nvar_];
	double regRight[nvar_];
	const amrex::Real chi_minus = chi(i - 1, j, k);
	const amrex::Real chi_plus = chi(i, j, k);
	for (int n = 0; n < nvar_; ++n) {
		regLeft[n] = FlattenedEdges<DIR, ORDER, limiter>(q, n, i - 1, j, k, chi_minus).second;
		regRight[n] = FlattenedEdges<DIR, ORDER, limiter>(q, n, i, j, k, chi_plus).first;
	}
	const quokka::Array4View<const amrex::Real, DIR> noBfield(amrex::Array4<const amrex::Real>{});
	return RiemannFaceFlux<RiemannSolver::HLLC, DIR>(quokka::RegisterStateView{regLeft}, quokka::RegisterStateView{regRight}, noBfield, noBfield,
							 amrex::Array4<const amrex::Real>{}, amrex::Array4<amrex::Real>{}, q, i, j, k, K_visc, dx,
							 shearViscosity, bulkViscosity);
}

template <typename problem_t>
template <int STAGE, int ORDER, SlopeLimiter limiter, typename Fixup>
void HydroSystem<problem_t>::UpdateStageFusedAllFaces(amrex::MultiFab const &primVar_mf, amrex::MultiFab const &chiCell_mf,
						      amrex::MultiFab const &primVarOld_mf, amrex::MultiFab const &consVarOld_mf,
						      amrex::MultiFab &consVarNew_mf, amrex::MultiFab &stageDiv_mf, const amrex::Real stage1Weight,
						      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const double dt, const amrex::Real K_visc,
						      const amrex::Real shearViscosity, const amrex::Real bulkViscosity, amrex::iMultiFab &redoFlag_mf,
						      Fixup const &fixup)
{
	static_assert(STAGE == 1 || STAGE == 2);
	static_assert(!Physics_Traits<problem_t>::is_mhd_enabled && !Physics_Traits<problem_t>::is_dust_enabled,
		      "the fused stage kernel is only implemented for pure hydro without dust");
	static_assert(nvar_ == nHydroScalars_);
	const BL_PROFILE("HydroSystem::UpdateStageFusedAllFaces()");
	AMREX_ALWAYS_ASSERT(primVarOld_mf.nComp() > primPgas_index);
	AMREX_ALWAYS_ASSERT(stageDiv_mf.nComp() >= nvar_ + 1);

	auto const &primVar_in = primVar_mf.const_arrays();
	auto const &chiCell_in = chiCell_mf.const_arrays();
	auto const &primVarOld_in = primVarOld_mf.const_arrays();
	auto const &consVarOld = consVarOld_mf.const_arrays();
	auto consVarNew = consVarNew_mf.arrays();
	auto stageDiv = stageDiv_mf.arrays();
	auto redoFlag = redoFlag_mf.arrays();

	amrex::ParallelFor(consVarNew_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Array4<const amrex::Real> const &q = primVar_in[bx];
		amrex::Array4<const amrex::Real> const &chi = chiCell_in[bx];

		// flux divergence, accumulated direction by direction in the same order and form as ComputeRhsFromFluxes
		quokka::valarray<double, nvar_> D{};
		double div_v = 0.;
		{
			const FaceFlux lo = FusedFaceFlux<FluxDir::X1, ORDER, limiter>(q, chi, i, j, k, K_visc, dx, shearViscosity, bulkViscosity);
			const FaceFlux hi = FusedFaceFlux<FluxDir::X1, ORDER, limiter>(q, chi, i + 1, j, k, K_visc, dx, shearViscosity, bulkViscosity);
			for (int n = 0; n < nvar_; ++n) {
				D[n] = (1.0 / dx[0]) * (lo.F[n] - hi.F[n]);
			}
			div_v = (hi.faceVel - lo.faceVel) / dx[0];
		}
#if AMREX_SPACEDIM >= 2
		{
			const FaceFlux lo = FusedFaceFlux<FluxDir::X2, ORDER, limiter>(q, chi, i, j, k, K_visc, dx, shearViscosity, bulkViscosity);
			const FaceFlux hi = FusedFaceFlux<FluxDir::X2, ORDER, limiter>(q, chi, i, j + 1, k, K_visc, dx, shearViscosity, bulkViscosity);
			for (int n = 0; n < nvar_; ++n) {
				D[n] += (1.0 / dx[1]) * (lo.F[n] - hi.F[n]);
			}
			div_v += (hi.faceVel - lo.faceVel) / dx[1];
		}
#endif
#if AMREX_SPACEDIM == 3
		{
			const FaceFlux lo = FusedFaceFlux<FluxDir::X3, ORDER, limiter>(q, chi, i, j, k, K_visc, dx, shearViscosity, bulkViscosity);
			const FaceFlux hi = FusedFaceFlux<FluxDir::X3, ORDER, limiter>(q, chi, i, j, k + 1, K_visc, dx, shearViscosity, bulkViscosity);
			for (int n = 0; n < nvar_; ++n) {
				D[n] += (1.0 / dx[2]) * (lo.F[n] - hi.F[n]);
			}
			div_v += (hi.faceVel - lo.faceVel) / dx[2];
		}
#endif

		FinishFusedStage<STAGE, Fixup>(D, div_v, stageDiv[bx], stage1Weight, primVarOld_in[bx], consVarOld[bx], consVarNew[bx], redoFlag[bx], i, j, k,
					       dt, fixup);
	});
}

template <typename problem_t>
template <int STAGE, typename Fixup>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
HydroSystem<problem_t>::FinishFusedStage(quokka::valarray<double, nvar_> &D, double div_v, amrex::Array4<amrex::Real> const &stageDiv,
					 const amrex::Real stage1Weight, amrex::Array4<const amrex::Real> const &primVarOld,
					 amrex::Array4<const amrex::Real> const &consVarOld, amrex::Array4<amrex::Real> const &consVarNew,
					 amrex::Array4<int> const &redoFlag, const int i, const int j, const int k, const double dt, Fixup const &fixup)
{
	if constexpr (STAGE == 1) {
		// keep the weighted stage-1 divergence for stage 2
		for (int n = 0; n < nvar_; ++n) {
			stageDiv(i, j, k, n) = stage1Weight * D[n];
		}
		stageDiv(i, j, k, nvar_) = stage1Weight * div_v;
	} else {
		for (int n = 0; n < nvar_; ++n) {
			D[n] = stageDiv(i, j, k, n) + 0.5 * D[n];
		}
		div_v = stageDiv(i, j, k, nvar_) + 0.5 * div_v;
	}

	const amrex::Real Pgas = primVarOld(i, j, k, primPgas_index);
	for (int n = 0; n < nvar_; ++n) {
		amrex::Real rhs = D[n];
		if (n == internalEnergy_index) {
			rhs += -Pgas * div_v;
		}
		consVarNew(i, j, k, n) = consVarOld(i, j, k, n) + dt * rhs;
	}

	// check if state is valid -- flag for re-do if not
	if (!isStateValid(consVarNew, i, j, k)) {
		redoFlag(i, j, k) = quokka::redoFlag::redo;
	} else {
		redoFlag(i, j, k) = quokka::redoFlag::none;
		// floors and dual-energy sync of the stage output (ApplyHydroStateFixup), applied to this cell only
		fixup(consVarNew, i, j, k);
	}
}

template <typename problem_t>
template <int STAGE, int ORDER, SlopeLimiter limiter, typename Fixup>
void HydroSystem<problem_t>::UpdateStageFusedLDS(amrex::MultiFab const &primVar_mf, amrex::MultiFab const &chiCell_mf, amrex::MultiFab const &primVarOld_mf,
						 amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf, amrex::MultiFab &stageDiv_mf,
						 const amrex::Real stage1Weight, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const double dt,
						 const amrex::Real K_visc, const amrex::Real shearViscosity, const amrex::Real bulkViscosity,
						 amrex::iMultiFab &redoFlag_mf, Fixup const &fixup)
{
#if defined(AMREX_USE_GPU) && !defined(AMREX_USE_SYCL) && (AMREX_SPACEDIM == 3)
	// Same per-cell result as UpdateStageFusedAllFaces, but each face flux is computed once per tile and shared through LDS:
	// a block of TX * TY * TZ threads owns a tile of cells. For each direction in turn, every thread computes the low face of
	// its cell, the first threads compute the extra face layer at the high end of the tile, and after a barrier each cell
	// reads its two faces from LDS.
	static_assert(STAGE == 1 || STAGE == 2);
	static_assert(!Physics_Traits<problem_t>::is_mhd_enabled && !Physics_Traits<problem_t>::is_dust_enabled,
		      "the fused stage kernel is only implemented for pure hydro without dust");
	const BL_PROFILE("HydroSystem::UpdateStageFusedLDS()");
	AMREX_ALWAYS_ASSERT(primVarOld_mf.nComp() > primPgas_index);
	AMREX_ALWAYS_ASSERT(stageDiv_mf.nComp() >= nvar_ + 1);

	constexpr int TX = 16;
	constexpr int TY = 4;
	constexpr int TZ = 4;
	constexpr int MT = TX * TY * TZ;
	constexpr int NC = nvar_ + 1;				       // flux components + face velocity
	constexpr int NF = MT + std::max({TY * TZ, TX * TZ, TX * TY}); // faces stored per direction (component stride in LDS)
	const std::size_t shmem_bytes = sizeof(double) * NC * NF;

	for (amrex::MFIter mfi(consVarNew_mf); mfi.isValid(); ++mfi) {
		const amrex::Box &vbx = mfi.validbox();
		const auto lo = amrex::lbound(vbx);
		const auto hi = amrex::ubound(vbx);
		const int ntx = (hi.x - lo.x + TX) / TX;
		const int nty = (hi.y - lo.y + TY) / TY;
		const int ntz = (hi.z - lo.z + TZ) / TZ;
		const int nblocks = ntx * nty * ntz;

		auto const q = primVar_mf.const_array(mfi);
		auto const chi = chiCell_mf.const_array(mfi);
		auto const primVarOld = primVarOld_mf.const_array(mfi);
		auto const consVarOld = consVarOld_mf.const_array(mfi);
		auto const consVarNew = consVarNew_mf.array(mfi);
		auto const stageDiv = stageDiv_mf.array(mfi);
		auto const redoFlag = redoFlag_mf.array(mfi);

		amrex::launch<MT>(nblocks, shmem_bytes, amrex::Gpu::gpuStream(), [=] AMREX_GPU_DEVICE() noexcept {
			amrex::Gpu::SharedMemory<double> gsm;
			double *const lds = gsm.dataPtr();
			const int t = static_cast<int>(threadIdx.x);
			int b = static_cast<int>(blockIdx.x);
			const int i0 = lo.x + TX * (b % ntx);
			b /= ntx;
			const int j0 = lo.y + TY * (b % nty);
			const int k0 = lo.z + TZ * (b / nty);
			const int ci = t % TX;
			const int cj = (t / TX) % TY;
			const int ck = t / (TX * TY);
			const int i = i0 + ci;
			const int j = j0 + cj;
			const int k = k0 + ck;
			const bool cellValid = (i <= hi.x) && (j <= hi.y) && (k <= hi.z);

			auto storeFace = [&](const int f, FaceFlux const &ff) {
				for (int n = 0; n < nvar_; ++n) {
					lds[n * NF + f] = ff.F[n];
				}
				lds[nvar_ * NF + f] = ff.faceVel;
			};

			quokka::valarray<double, nvar_> D{};
			double div_v = 0.;

			// x-faces: LDS index fi + (TX + 1) * (fj + TY * fk)
			{
				auto fidx = [](int fi, int fj, int fk) { return fi + (TX + 1) * (fj + TY * fk); };
				if (i <= hi.x + 1 && j <= hi.y && k <= hi.z) {
					storeFace(fidx(ci, cj, ck),
						  FusedFaceFlux<FluxDir::X1, ORDER, limiter>(q, chi, i, j, k, K_visc, dx, shearViscosity, bulkViscosity));
				}
				if (t < TY * TZ) {
					const int fj = t % TY;
					const int fk = t / TY;
					const int ii = i0 + TX;
					if (ii <= hi.x + 1 && j0 + fj <= hi.y && k0 + fk <= hi.z) {
						storeFace(fidx(TX, fj, fk), FusedFaceFlux<FluxDir::X1, ORDER, limiter>(q, chi, ii, j0 + fj, k0 + fk, K_visc, dx,
														       shearViscosity, bulkViscosity));
					}
				}
				__syncthreads();
				if (cellValid) {
					const int flo = fidx(ci, cj, ck);
					const int fhi = fidx(ci + 1, cj, ck);
					for (int n = 0; n < nvar_; ++n) {
						D[n] = (1.0 / dx[0]) * (lds[n * NF + flo] - lds[n * NF + fhi]);
					}
					div_v = (lds[nvar_ * NF + fhi] - lds[nvar_ * NF + flo]) / dx[0];
				}
				__syncthreads();
			}
			// y-faces: LDS index fi + TX * (fj + (TY + 1) * fk)
			{
				auto fidx = [](int fi, int fj, int fk) { return fi + TX * (fj + (TY + 1) * fk); };
				if (i <= hi.x && j <= hi.y + 1 && k <= hi.z) {
					storeFace(fidx(ci, cj, ck),
						  FusedFaceFlux<FluxDir::X2, ORDER, limiter>(q, chi, i, j, k, K_visc, dx, shearViscosity, bulkViscosity));
				}
				if (t < TX * TZ) {
					const int fi = t % TX;
					const int fk = t / TX;
					const int jj = j0 + TY;
					if (i0 + fi <= hi.x && jj <= hi.y + 1 && k0 + fk <= hi.z) {
						storeFace(fidx(fi, TY, fk), FusedFaceFlux<FluxDir::X2, ORDER, limiter>(q, chi, i0 + fi, jj, k0 + fk, K_visc, dx,
														       shearViscosity, bulkViscosity));
					}
				}
				__syncthreads();
				if (cellValid) {
					const int flo = fidx(ci, cj, ck);
					const int fhi = fidx(ci, cj + 1, ck);
					for (int n = 0; n < nvar_; ++n) {
						D[n] += (1.0 / dx[1]) * (lds[n * NF + flo] - lds[n * NF + fhi]);
					}
					div_v += (lds[nvar_ * NF + fhi] - lds[nvar_ * NF + flo]) / dx[1];
				}
				__syncthreads();
			}
			// z-faces: LDS index fi + TX * (fj + TY * fk)
			{
				auto fidx = [](int fi, int fj, int fk) { return fi + TX * (fj + TY * fk); };
				if (i <= hi.x && j <= hi.y && k <= hi.z + 1) {
					storeFace(fidx(ci, cj, ck),
						  FusedFaceFlux<FluxDir::X3, ORDER, limiter>(q, chi, i, j, k, K_visc, dx, shearViscosity, bulkViscosity));
				}
				if (t < TX * TY) {
					const int fi = t % TX;
					const int fj = t / TX;
					const int kk = k0 + TZ;
					if (i0 + fi <= hi.x && j0 + fj <= hi.y && kk <= hi.z + 1) {
						storeFace(fidx(fi, fj, TZ), FusedFaceFlux<FluxDir::X3, ORDER, limiter>(q, chi, i0 + fi, j0 + fj, kk, K_visc, dx,
														       shearViscosity, bulkViscosity));
					}
				}
				__syncthreads();
				if (cellValid) {
					const int flo = fidx(ci, cj, ck);
					const int fhi = fidx(ci, cj, ck + 1);
					for (int n = 0; n < nvar_; ++n) {
						D[n] += (1.0 / dx[2]) * (lds[n * NF + flo] - lds[n * NF + fhi]);
					}
					div_v += (lds[nvar_ * NF + fhi] - lds[nvar_ * NF + flo]) / dx[2];
				}
			}

			if (cellValid) {
				FinishFusedStage<STAGE, Fixup>(D, div_v, stageDiv, stage1Weight, primVarOld, consVarOld, consVarNew, redoFlag, i, j, k, dt,
							       fixup);
			}
		});
	}
#else
	UpdateStageFusedAllFaces<STAGE, ORDER, limiter>(primVar_mf, chiCell_mf, primVarOld_mf, consVarOld_mf, consVarNew_mf, stageDiv_mf, stage1Weight, dx, dt,
							K_visc, shearViscosity, bulkViscosity, redoFlag_mf, fixup);
#endif
}

template <typename problem_t>
template <int STAGE, int ORDER, SlopeLimiter limiter, typename Fixup>
void HydroSystem<problem_t>::UpdateStageFusedMarch(amrex::MultiFab const &primVar_mf, amrex::MultiFab const &chiCell_mf, amrex::MultiFab const &primVarOld_mf,
						   amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf, amrex::MultiFab &stageDiv_mf,
						   const amrex::Real stage1Weight, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const double dt,
						   const amrex::Real K_visc, const amrex::Real shearViscosity, const amrex::Real bulkViscosity,
						   amrex::iMultiFab &redoFlag_mf, Fixup const &fixup)
{
#if defined(AMREX_USE_GPU) && !defined(AMREX_USE_SYCL) && (AMREX_SPACEDIM == 3)
	static_assert(STAGE == 1 || STAGE == 2);
	static_assert(!Physics_Traits<problem_t>::is_mhd_enabled && !Physics_Traits<problem_t>::is_dust_enabled,
		      "the fused stage kernel is only implemented for pure hydro without dust");
	const BL_PROFILE("HydroSystem::UpdateStageFusedMarch()");
	AMREX_ALWAYS_ASSERT(primVarOld_mf.nComp() > primPgas_index);
	AMREX_ALWAYS_ASSERT(stageDiv_mf.nComp() >= nvar_ + 1);

	constexpr int TX = 16;
	constexpr int TY = 8;
	constexpr int MT = TX * TY;
	constexpr int NC = nvar_ + 1;	       // flux components + face velocity
	constexpr int NFX = (TX + 1) * TY;     // x-faces of one plane of the column
	constexpr int NFY = TX * (TY + 1);     // y-faces of one plane of the column
	constexpr int NF = std::max(NFX, NFY); // component stride in LDS
	const std::size_t shmem_bytes = sizeof(double) * NC * NF;
	// cells along z per block: long enough to amortize the first z-face, short enough to give many blocks
	int zchunk = 32;
	amrex::ParmParse const hpp("hydro");
	hpp.query("fused_march_zchunk", zchunk);

	for (amrex::MFIter mfi(consVarNew_mf); mfi.isValid(); ++mfi) {
		const amrex::Box &vbx = mfi.validbox();
		const auto lo = amrex::lbound(vbx);
		const auto hi = amrex::ubound(vbx);
		const int ntx = (hi.x - lo.x + TX) / TX;
		const int nty = (hi.y - lo.y + TY) / TY;
		const int ntz = (hi.z - lo.z + zchunk) / zchunk;
		const int nblocks = ntx * nty * ntz;

		auto const q = primVar_mf.const_array(mfi);
		auto const chi = chiCell_mf.const_array(mfi);
		auto const primVarOld = primVarOld_mf.const_array(mfi);
		auto const consVarOld = consVarOld_mf.const_array(mfi);
		auto const consVarNew = consVarNew_mf.array(mfi);
		auto const stageDiv = stageDiv_mf.array(mfi);
		auto const redoFlag = redoFlag_mf.array(mfi);

		amrex::launch<MT>(nblocks, shmem_bytes, amrex::Gpu::gpuStream(), [=] AMREX_GPU_DEVICE() noexcept {
			amrex::Gpu::SharedMemory<double> gsm;
			double *const lds = gsm.dataPtr();
			const int t = static_cast<int>(threadIdx.x);
			int b = static_cast<int>(blockIdx.x);
			const int i0 = lo.x + TX * (b % ntx);
			b /= ntx;
			const int j0 = lo.y + TY * (b % nty);
			const int kbeg = lo.z + zchunk * (b / nty);
			const int kend = amrex::min(kbeg + zchunk - 1, hi.z);
			const int ci = t % TX;
			const int cj = t / TX;
			const int i = i0 + ci;
			const int j = j0 + cj;
			const bool columnValid = (i <= hi.x) && (j <= hi.y);

			auto storeFace = [&](const int f, FaceFlux const &ff) {
				for (int n = 0; n < nvar_; ++n) {
					lds[n * NF + f] = ff.F[n];
				}
				lds[nvar_ * NF + f] = ff.faceVel;
			};
			auto fidxX = [](int fi, int fj) { return fi + (TX + 1) * fj; };
			auto fidxY = [](int fi, int fj) { return fi + TX * fj; };

			// z-face at the bottom of the first plane of this chunk; afterwards it is carried in registers
			FaceFlux zlo{};
			if (columnValid) {
				zlo = FusedFaceFlux<FluxDir::X3, ORDER, limiter>(q, chi, i, j, kbeg, K_visc, dx, shearViscosity, bulkViscosity);
			}

			for (int k = kbeg; k <= kend; ++k) {
				quokka::valarray<double, nvar_> D{};
				double div_v = 0.;

				// x-faces of this plane
				if (i <= hi.x + 1 && j <= hi.y) {
					storeFace(fidxX(ci, cj),
						  FusedFaceFlux<FluxDir::X1, ORDER, limiter>(q, chi, i, j, k, K_visc, dx, shearViscosity, bulkViscosity));
				}
				if (t < TY) {
					const int ii = i0 + TX;
					if (ii <= hi.x + 1 && j0 + t <= hi.y) {
						storeFace(fidxX(TX, t), FusedFaceFlux<FluxDir::X1, ORDER, limiter>(q, chi, ii, j0 + t, k, K_visc, dx,
														   shearViscosity, bulkViscosity));
					}
				}
				__syncthreads();
				if (columnValid) {
					const int flo = fidxX(ci, cj);
					const int fhi = fidxX(ci + 1, cj);
					for (int n = 0; n < nvar_; ++n) {
						D[n] = (1.0 / dx[0]) * (lds[n * NF + flo] - lds[n * NF + fhi]);
					}
					div_v = (lds[nvar_ * NF + fhi] - lds[nvar_ * NF + flo]) / dx[0];
				}
				__syncthreads();

				if (i <= hi.x && j <= hi.y + 1) {
					storeFace(fidxY(ci, cj),
						  FusedFaceFlux<FluxDir::X2, ORDER, limiter>(q, chi, i, j, k, K_visc, dx, shearViscosity, bulkViscosity));
				}
				if (t < TX) {
					const int jj = j0 + TY;
					if (i0 + t <= hi.x && jj <= hi.y + 1) {
						storeFace(fidxY(t, TY), FusedFaceFlux<FluxDir::X2, ORDER, limiter>(q, chi, i0 + t, jj, k, K_visc, dx,
														   shearViscosity, bulkViscosity));
					}
				}
				__syncthreads();
				if (columnValid) {
					const int flo = fidxY(ci, cj);
					const int fhi = fidxY(ci, cj + 1);
					for (int n = 0; n < nvar_; ++n) {
						D[n] += (1.0 / dx[1]) * (lds[n * NF + flo] - lds[n * NF + fhi]);
					}
					div_v += (lds[nvar_ * NF + fhi] - lds[nvar_ * NF + flo]) / dx[1];
				}
				__syncthreads();

				// z-faces: the top face of this plane becomes the bottom face of the next one
				if (columnValid) {
					const FaceFlux zhi =
					    FusedFaceFlux<FluxDir::X3, ORDER, limiter>(q, chi, i, j, k + 1, K_visc, dx, shearViscosity, bulkViscosity);
					for (int n = 0; n < nvar_; ++n) {
						D[n] += (1.0 / dx[2]) * (zlo.F[n] - zhi.F[n]);
					}
					div_v += (zhi.faceVel - zlo.faceVel) / dx[2];
					zlo = zhi;

					FinishFusedStage<STAGE, Fixup>(D, div_v, stageDiv, stage1Weight, primVarOld, consVarOld, consVarNew, redoFlag, i, j, k,
								       dt, fixup);
				}
			}
		});
	}
#else
	UpdateStageFusedAllFaces<STAGE, ORDER, limiter>(primVar_mf, chiCell_mf, primVarOld_mf, consVarOld_mf, consVarNew_mf, stageDiv_mf, stage1Weight, dx, dt,
							K_visc, shearViscosity, bulkViscosity, redoFlag_mf, fixup);
#endif
}

template <typename problem_t>
template <int STAGE, int ORDER, SlopeLimiter limiter, FluxDir DIR, int MT, typename Fixup>
void HydroSystem<problem_t>::UpdateStageSplitDir(amrex::MultiFab const &primVar_mf, amrex::MultiFab const &chiCell_mf, amrex::MultiFab const &primVarOld_mf,
						 amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf, amrex::MultiFab &acc_mf,
						 amrex::MultiFab &stageDiv_mf, const amrex::Real stage1Weight, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
						 const double dt, const amrex::Real K_visc, const amrex::Real shearViscosity, const amrex::Real bulkViscosity,
						 amrex::iMultiFab &redoFlag_mf, Fixup const &fixup)
{
#if defined(AMREX_USE_GPU) && !defined(AMREX_USE_SYCL) && (AMREX_SPACEDIM == 3)
	static_assert(STAGE == 1 || STAGE == 2);
	static_assert(MT % 64 == 0);
	static_assert(!Physics_Traits<problem_t>::is_mhd_enabled && !Physics_Traits<problem_t>::is_dust_enabled,
		      "the fused stage kernel is only implemented for pure hydro without dust");
	const BL_PROFILE("HydroSystem::UpdateStageSplitDir()");
	AMREX_ALWAYS_ASSERT(acc_mf.nComp() >= nvar_ + 1);

	constexpr int d = static_cast<int>(DIR);
	constexpr int TX = 64;
	constexpr int TY = (d == 2) ? 1 : MT / 64;
	constexpr int TZ = (d == 2) ? MT / 64 : 1;
	constexpr int FX = TX + ((d == 0) ? 1 : 0); // face extents of the tile
	constexpr int FY = TY + ((d == 1) ? 1 : 0);
	constexpr int NEXTRA = (d == 0) ? (TY * TZ) : ((d == 1) ? (TX * TZ) : (TX * TY)); // faces of the high layer
	constexpr int NC = nvar_ + 1;
	constexpr int NF = MT + NEXTRA;
	const std::size_t shmem_bytes = sizeof(double) * NC * NF;

	for (amrex::MFIter mfi(consVarNew_mf); mfi.isValid(); ++mfi) {
		const amrex::Box &vbx = mfi.validbox();
		const auto lo = amrex::lbound(vbx);
		const auto hi = amrex::ubound(vbx);
		const int ntx = (hi.x - lo.x + TX) / TX;
		const int nty = (hi.y - lo.y + TY) / TY;
		const int ntz = (hi.z - lo.z + TZ) / TZ;
		const int nblocks = ntx * nty * ntz;

		auto const q = primVar_mf.const_array(mfi);
		auto const chi = chiCell_mf.const_array(mfi);
		auto const primVarOld = primVarOld_mf.const_array(mfi);
		auto const consVarOld = consVarOld_mf.const_array(mfi);
		auto const consVarNew = consVarNew_mf.array(mfi);
		auto const acc = acc_mf.array(mfi);
		auto const stageDiv = stageDiv_mf.array(mfi);
		auto const redoFlag = redoFlag_mf.array(mfi);
		// face range along DIR extends one past the cells
		const int hix = hi.x + ((d == 0) ? 1 : 0);
		const int hiy = hi.y + ((d == 1) ? 1 : 0);
		const int hiz = hi.z + ((d == 2) ? 1 : 0);

		amrex::launch<MT>(nblocks, shmem_bytes, amrex::Gpu::gpuStream(), [=] AMREX_GPU_DEVICE() noexcept {
			amrex::Gpu::SharedMemory<double> gsm;
			double *const lds = gsm.dataPtr();
			const int t = static_cast<int>(threadIdx.x);
			int b = static_cast<int>(blockIdx.x);
			const int i0 = lo.x + TX * (b % ntx);
			b /= ntx;
			const int j0 = lo.y + TY * (b % nty);
			const int k0 = lo.z + TZ * (b / nty);
			const int ci = t % TX;
			const int cj = (t / TX) % TY;
			const int ck = t / (TX * TY);
			const int i = i0 + ci;
			const int j = j0 + cj;
			const int k = k0 + ck;
			const bool cellValid = (i <= hi.x) && (j <= hi.y) && (k <= hi.z);
			auto fidx = [](int fi, int fj, int fk) { return fi + FX * (fj + FY * fk); };

			auto computeStore = [&](int fi, int fj, int fk) {
				const int ii = i0 + fi;
				const int jj = j0 + fj;
				const int kk = k0 + fk;
				if (ii <= hix && jj <= hiy && kk <= hiz) {
					const FaceFlux ff = FusedFaceFlux<DIR, ORDER, limiter>(q, chi, ii, jj, kk, K_visc, dx, shearViscosity, bulkViscosity);
					const int f = fidx(fi, fj, fk);
					for (int n = 0; n < nvar_; ++n) {
						lds[n * NF + f] = ff.F[n];
					}
					lds[nvar_ * NF + f] = ff.faceVel;
				}
			};

			// low face of each cell
			computeStore(ci, cj, ck);
			// high face layer of the tile, on the first NEXTRA threads
			if (t < NEXTRA) {
				if constexpr (d == 0) {
					computeStore(TX, t % TY, t / TY);
				} else if constexpr (d == 1) {
					computeStore(t % TX, TY, t / TX);
				} else {
					computeStore(t % TX, t / TX, TZ);
				}
			}
			__syncthreads();
			if (!cellValid) {
				return;
			}
			const int flo = fidx(ci, cj, ck);
			const int fhi = fidx(ci + ((d == 0) ? 1 : 0), cj + ((d == 1) ? 1 : 0), ck + ((d == 2) ? 1 : 0));

			// this direction's part, in the same form as UpdateStageFusedAllFaces
			if constexpr (d == 0) {
				for (int n = 0; n < nvar_; ++n) {
					acc(i, j, k, n) = (1.0 / dx[0]) * (lds[n * NF + flo] - lds[n * NF + fhi]);
				}
				acc(i, j, k, nvar_) = (lds[nvar_ * NF + fhi] - lds[nvar_ * NF + flo]) / dx[0];
			} else if constexpr (d == 1) {
				for (int n = 0; n < nvar_; ++n) {
					acc(i, j, k, n) += (1.0 / dx[1]) * (lds[n * NF + flo] - lds[n * NF + fhi]);
				}
				acc(i, j, k, nvar_) += (lds[nvar_ * NF + fhi] - lds[nvar_ * NF + flo]) / dx[1];
			} else {
				quokka::valarray<double, nvar_> D{};
				for (int n = 0; n < nvar_; ++n) {
					D[n] = acc(i, j, k, n) + (1.0 / dx[2]) * (lds[n * NF + flo] - lds[n * NF + fhi]);
				}
				const double div_v = acc(i, j, k, nvar_) + (lds[nvar_ * NF + fhi] - lds[nvar_ * NF + flo]) / dx[2];
				FinishFusedStage<STAGE, Fixup>(D, div_v, stageDiv, stage1Weight, primVarOld, consVarOld, consVarNew, redoFlag, i, j, k, dt,
							       fixup);
			}
		});
	}
#else
	amrex::ignore_unused(primVar_mf, chiCell_mf, primVarOld_mf, consVarOld_mf, consVarNew_mf, acc_mf, stageDiv_mf, stage1Weight, dx, dt, K_visc,
			     shearViscosity, bulkViscosity, redoFlag_mf, fixup);
	amrex::Abort("UpdateStageSplitDir requires a 3D GPU build");
#endif
}

// to ensure that physical quantities are within reasonable
// floors and ceilings which can be set in the param file
template <typename problem_t>
template <typename DensityFloorFunc>
void HydroSystem<problem_t>::EnforceLimits(amrex::Real const densityFloor, amrex::Real const dustDensityFloor, amrex::Real const tempFloor,
					   amrex::MultiFab &state_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc_mf,
					   amrex::Geometry const &geom, DensityFloorFunc const &density_floor_func, const bool syncDualEnergy)
{
	auto state = state_mf.arrays();
	auto const &state_fc_x0 = state_fc_mf[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto const &state_fc_x1 = state_fc_mf[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto const &state_fc_x2 = state_fc_mf[2].const_arrays();
#endif
	auto const prob_lo = geom.ProbLoArray();
	auto const dx = geom.CellSizeArray();

	amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Array4<const amrex::Real> fc_x{};
		amrex::Array4<const amrex::Real> fc_y{};
		amrex::Array4<const amrex::Real> fc_z{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			fc_x = state_fc_x0[bx];
#if AMREX_SPACEDIM >= 2
			fc_y = state_fc_x1[bx];
#endif
#if AMREX_SPACEDIM == 3
			fc_z = state_fc_x2[bx];
#endif
		}
		EnforceLimitsCell(state[bx], i, j, k, densityFloor, dustDensityFloor, tempFloor, prob_lo, dx, density_floor_func, fc_x, fc_y, fc_z);
		if (syncDualEnergy) {
			SyncDualEnergyCell(state[bx], i, j, k, fc_x, fc_y, fc_z);
		}
	});
}

template <typename problem_t>
template <typename DensityFloorFunc>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
HydroSystem<problem_t>::EnforceLimitsCell(amrex::Array4<amrex::Real> const &state, const int i, const int j, const int k, amrex::Real const densityFloor,
					  amrex::Real const dustDensityFloor, amrex::Real const tempFloor,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  DensityFloorFunc const &density_floor_func, amrex::Array4<const amrex::Real> const &fc_x,
					  amrex::Array4<const amrex::Real> const &fc_y, amrex::Array4<const amrex::Real> const &fc_z)
{
	// per-cell body of EnforceLimits
	amrex::ignore_unused(fc_x, fc_y, fc_z);
	amrex::Real const x = prob_lo[0] + (static_cast<amrex::Real>(i) + static_cast<amrex::Real>(0.5)) * dx[0];
#if (AMREX_SPACEDIM >= 2)
	amrex::Real const y = prob_lo[1] + (static_cast<amrex::Real>(j) + static_cast<amrex::Real>(0.5)) * dx[1];
#else
	amrex::Real const y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
	amrex::Real const z = prob_lo[2] + (static_cast<amrex::Real>(k) + static_cast<amrex::Real>(0.5)) * dx[2];
#else
	amrex::Real const z = 0.0;
#endif

	amrex::Real localDensityFloor = density_floor_func(x, y, z, densityFloor);

	// Enforce density floor (do not adjust energies here!!)
	amrex::Real rho_new = NAN;
	{
		amrex::Real const rho = state(i, j, k, density_index);
		rho_new = rho;

		if (rho < localDensityFloor) {
			rho_new = localDensityFloor;
			state(i, j, k, density_index) = rho_new;

			if (nscalars_ > 0) {
				for (int n = 0; n < nscalars_; ++n) {
					if (rho_new == 0.0) {
						state(i, j, k, scalar0_index + n) = 0.0;
					} else {
						state(i, j, k, scalar0_index + n) *= rho / rho_new;
					}
				}
			}
		}
	}

	// Enforce mass scalars floor
	if (nmscalars_ > 0) {
		amrex::Real sp_sum = 0.0;
		for (int idx = 0; idx < nmscalars_; ++idx) {
			if (state(i, j, k, scalar0_index + idx) < 0.0) {
				state(i, j, k, scalar0_index + idx) = network_rp::small_x * rho_new;
			}
			// get sum to renormalize
			sp_sum += state(i, j, k, scalar0_index + idx);
		}

		if ((sp_sum > std::numeric_limits<amrex::Real>::min()) && (rho_new > std::numeric_limits<amrex::Real>::min())) {
			sp_sum /= rho_new; // get mass fractions
			for (int idx = 0; idx < nmscalars_; ++idx) {
				// renormalize
				state(i, j, k, scalar0_index + idx) /= sp_sum;
			}
		}
	}

	// Enforce dust density floor
	amrex::Real const dust_floor = dustDensityFloor;
	if constexpr (Physics_Traits<problem_t>::is_dust_enabled) {
		for (int g = 0; g < Physics_Traits<problem_t>::nDustGroups; ++g) {
			amrex::Real dust_rho = state(i, j, k, dustDensity_index + g * numDustVars_);
			if (dust_rho < dust_floor) {
				state(i, j, k, dustDensity_index + g * numDustVars_) = dust_floor;
			}
		}
	}

	// Enforce temperature floor
	if ((rho_new > std::numeric_limits<amrex::Real>::min()) && !HydroSystem<problem_t>::is_eos_isothermal()) {
		amrex::Real const vx1 = state(i, j, k, x1Momentum_index) / rho_new;
		amrex::Real const vx2 = state(i, j, k, x2Momentum_index) / rho_new;
		amrex::Real const vx3 = state(i, j, k, x3Momentum_index) / rho_new;
		amrex::Real const Ekin = 0.5 * rho_new * (vx1 * vx1 + vx2 * vx2 + vx3 * vx3);

		// Enforce temperature floor (for total energy)
		// First-capture face-centered arrays before any constexpr-if context for CUDA.
		amrex::Array4<const amrex::Real> const &state_fc_x0_ref = fc_x;
#if AMREX_SPACEDIM >= 2
		amrex::Array4<const amrex::Real> const &state_fc_x1_ref = fc_y;
#endif
#if AMREX_SPACEDIM == 3
		amrex::Array4<const amrex::Real> const &state_fc_x2_ref = fc_z;
#endif
		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> state_fc{};
		state_fc[0] = state_fc_x0_ref;
#if AMREX_SPACEDIM >= 2
		state_fc[1] = state_fc_x1_ref;
#endif
#if AMREX_SPACEDIM == 3
		state_fc[2] = state_fc_x2_ref;
#endif

		amrex::Real const Emag = ComputeMagneticEnergy(i, j, k, &state_fc);
		amrex::Real const Eint = ComputeInternalEnergy(state, i, j, k, &state_fc);
		amrex::GpuArray<Real, nmscalars_> const massScalars = RadSystem<problem_t>::ComputeMassScalars(state, i, j, k);
		amrex::Real const primTemp = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho_new, Eint, massScalars);

		if (primTemp < tempFloor) {
			amrex::Real const prim_eint = ::quokka::EOS<problem_t>::ComputeEintFromTgas(rho_new, tempFloor, massScalars);
			state(i, j, k, energy_index) = Ekin + Emag + prim_eint;
		}

		// Enforce temperature floor (for auxiliary internal energy)
		amrex::Real const auxEint = state(i, j, k, internalEnergy_index);
		amrex::Real const auxTemp = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho_new, auxEint, massScalars);

		if (auxTemp < tempFloor) {
			amrex::Real const new_Eint = ::quokka::EOS<problem_t>::ComputeEintFromTgas(rho_new, tempFloor, massScalars);
			state(i, j, k, internalEnergy_index) = new_Eint;
			// total energy should NOT be updated here
		}
	}
}

template <typename problem_t>
void HydroSystem<problem_t>::AddInternalEnergyPdV(amrex::MultiFab &rhs_mf, amrex::MultiFab const &consVar_mf, amrex::MultiFab const &primVar_mf,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf,
						  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArray, amrex::iMultiFab const &redoFlag_mf)
{
	// compute P dV source term for the internal energy equation,
	// using the face-centered velocities in faceVelArray and the pressure
	// hydro-05: the pressure is read from primVar_mf (component primPgas_index), which must be
	// ConservedToPrimitive(consVar_mf, cons_fc_mf) of the same state (allocated with nPrimVar_ components)
	amrex::ignore_unused(cons_fc_mf);
	AMREX_ALWAYS_ASSERT(primVar_mf.nComp() > primPgas_index);
	auto vel_x = faceVelArray[0].const_arrays();

#if AMREX_SPACEDIM >= 2
	auto vel_y = faceVelArray[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto vel_z = faceVelArray[2].const_arrays();
#endif

	auto const &consVar = consVar_mf.const_arrays();
	auto const &primVar = primVar_mf.const_arrays();
	auto const &redoFlag = redoFlag_mf.const_arrays();
	auto rhs = rhs_mf.arrays();

	amrex::ParallelFor(rhs_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		// get cell-centered pressure (stored by ConservedToPrimitive)
		const amrex::Real Pgas = primVar[bx](i, j, k, primPgas_index);

		// compute div v from face-centered velocities
		amrex::Real div_v = NAN;

		if (redoFlag[bx](i, j, k) == quokka::redoFlag::none) {
			div_v = AMREX_D_TERM((vel_x[bx](i + 1, j, k) - vel_x[bx](i, j, k)) / dx[0], +(vel_y[bx](i, j + 1, k) - vel_y[bx](i, j, k)) / dx[1],
					     +(vel_z[bx](i, j, k + 1) - vel_z[bx](i, j, k)) / dx[2]);
		} else {
			div_v = 0.5 * (AMREX_D_TERM((ComputeVelocityX1(consVar[bx], i + 1, j, k) - ComputeVelocityX1(consVar[bx], i - 1, j, k)) / dx[0],
						    +(ComputeVelocityX2(consVar[bx], i, j + 1, k) - ComputeVelocityX2(consVar[bx], i, j - 1, k)) / dx[1],
						    +(ComputeVelocityX3(consVar[bx], i, j, k + 1) - ComputeVelocityX3(consVar[bx], i, j, k - 1)) / dx[2]));
		}

		// add P dV term to rhs array
		rhs[bx](i, j, k, internalEnergy_index) += -Pgas * div_v;
	});
}

template <typename problem_t>
void HydroSystem<problem_t>::SyncDualEnergy(amrex::MultiFab &consVar_mf, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &faceVar_mf)
{
	// sync internal energy and total energy
	// this step must be done as an operator-split step after *each* RK stage

	auto consVar = consVar_mf.arrays();
	auto faceVar_x = faceVar_mf[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto faceVar_y = faceVar_mf[1].const_arrays();
#if AMREX_SPACEDIM == 3
	auto faceVar_z = faceVar_mf[2].const_arrays();
#endif
#endif

	amrex::ParallelFor(consVar_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		amrex::Array4<const amrex::Real> fc_x{};
		amrex::Array4<const amrex::Real> fc_y{};
		amrex::Array4<const amrex::Real> fc_z{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			fc_x = faceVar_x[bx];
#if AMREX_SPACEDIM >= 2
			fc_y = faceVar_y[bx];
#endif
#if AMREX_SPACEDIM == 3
			fc_z = faceVar_z[bx];
#endif
		}
		SyncDualEnergyCell(consVar[bx], i, j, k, fc_x, fc_y, fc_z);
	});
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void HydroSystem<problem_t>::SyncDualEnergyCell(amrex::Array4<amrex::Real> const &consVar, const int i, const int j,
										    const int k, amrex::Array4<const amrex::Real> const &faceVar_x,
										    amrex::Array4<const amrex::Real> const &faceVar_y,
										    amrex::Array4<const amrex::Real> const &faceVar_z)
{
	amrex::ignore_unused(faceVar_x, faceVar_y, faceVar_z);
	const amrex::Real eta = 1.0e-3; // dual energy parameter 'eta'
	amrex::Real const rho = consVar(i, j, k, density_index);
	amrex::Real const px = consVar(i, j, k, x1Momentum_index);
	amrex::Real const py = consVar(i, j, k, x2Momentum_index);
	amrex::Real const pz = consVar(i, j, k, x3Momentum_index);
	amrex::Real const Etot = consVar(i, j, k, energy_index);
	amrex::Real const Eint_aux = consVar(i, j, k, internalEnergy_index);

	// abort if density is negative (can't compute kinetic energy)
	if (rho <= 0.) {
		amrex::Abort("density is negative in SyncDualEnergy! abort!!");
	}

	// compute kinetic energy
	amrex::Real const Ekin = (px * px + py * py + pz * pz) / (2.0 * rho);

	// compute magnetic energy
	amrex::Real Emag = 0;
	if (Physics_Traits<problem_t>::is_mhd_enabled) { // cannot be 'if constexpr' due to CUDA limitation
		constexpr int mhd_idx = Physics_Indices<problem_t>::mhdFirstIndex;
		amrex::Real const Bx = 0.5 * (faceVar_x(i, j, k, mhd_idx) + faceVar_x(i + 1, j, k, mhd_idx));
#if AMREX_SPACEDIM >= 2
		amrex::Real const By = 0.5 * (faceVar_y(i, j, k, mhd_idx) + faceVar_y(i, j + 1, k, mhd_idx));
#endif
#if AMREX_SPACEDIM == 3
		amrex::Real const Bz = 0.5 * (faceVar_z(i, j, k, mhd_idx) + faceVar_z(i, j, k + 1, mhd_idx));
#endif
		Emag = 0.5 * (AMREX_D_TERM(Bx * Bx, +By * By, +Bz * Bz));
	}

	// compute internal energy from conserved vars
	amrex::Real const Eint_cons = Etot - Ekin - Emag;

	// Li et al. sync method
	// replace Eint with Eint_cons == (Etot - Ekin) if (Eint_cons / E) > eta
	if (Eint_cons > eta * Etot) {
		consVar(i, j, k, internalEnergy_index) = Eint_cons;
	} else { // non-conservative sync
		consVar(i, j, k, internalEnergy_index) = Eint_aux;
		consVar(i, j, k, energy_index) = Eint_aux + Ekin + Emag;
	}
}

template <typename problem_t, FluxDir DIR>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeViscosity(int /*i*/, int /*j*/, int /*k*/, quokka::Array4View<const amrex::Real, DIR> const & /*q*/)
    -> quokka::valarray<amrex::Real, 2> // {shear_viscosity, bulk_viscosity}
{
	static_assert(sizeof(problem_t) == 0, "computeViscosity must be specialized in the problem file when using ViscosityModel::problem_defined");
	return {0.0, 0.0};
}

template <typename problem_t>
template <RiemannSolver RIEMANN, FluxDir DIR, int FUSED_ORDER, SlopeLimiter limiter, int MT>
void HydroSystem<problem_t>::ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab &x1FaceVel_mf, amrex::MultiFab const &x1LeftState_mf,
					   amrex::MultiFab const &x1RightState_mf, amrex::MultiFab const &x1LeftState_bfield_mf,
					   amrex::MultiFab const &x1RightState_bfield_mf, amrex::MultiFab const &primVar_mf, const amrex::Real K_visc,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const amrex::Real shearViscosity, const amrex::Real bulkViscosity,
					   amrex::MultiFab *x1FSpds_mf, amrex::MultiFab const *x1ConsVar_fc_mf, const int nghost_vel,
					   amrex::MultiFab const *chiCell_mf, const bool accumulate, const amrex::Real accumWeightOld,
					   const amrex::Real accumWeightNew)
{
	static_assert(FUSED_ORDER == 0 || FUSED_ORDER == 2 || FUSED_ORDER == 3, "FUSED_ORDER must be 0, 2 (PLM) or 3 (PPM)");
	static_assert(FUSED_ORDER == 0 || (!Physics_Traits<problem_t>::is_mhd_enabled && !Physics_Traits<problem_t>::is_dust_enabled),
		      "fused reconstruction is only implemented for pure hydro without dust");

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	// interface states are read from memory only when they are not reconstructed in the kernel (FUSED_ORDER == 0)
	amrex::MultiArray4<const amrex::Real> x1LeftState_in;
	amrex::MultiArray4<const amrex::Real> x1RightState_in;
	amrex::MultiArray4<const amrex::Real> chiCell_in;
	if constexpr (FUSED_ORDER == 0) {
		x1LeftState_in = x1LeftState_mf.const_arrays();
		x1RightState_in = x1RightState_mf.const_arrays();
	} else {
		AMREX_ALWAYS_ASSERT(chiCell_mf != nullptr);
		chiCell_in = chiCell_mf->const_arrays();
	}
	// B-field interface states exist only for MHD (hydro-02: they are empty MultiFabs in pure hydro)
	amrex::MultiArray4<const amrex::Real> x1LeftState_bfield_in;
	amrex::MultiArray4<const amrex::Real> x1RightState_bfield_in;
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		x1LeftState_bfield_in = x1LeftState_bfield_mf.const_arrays();
		x1RightState_bfield_in = x1RightState_bfield_mf.const_arrays();
	}
	auto const &primVar_in = primVar_mf.const_arrays();
	auto x1Flux_in = x1Flux_mf.arrays();
	auto x1FaceVel_in = x1FaceVel_mf.arrays();

	amrex::MultiArray4<double> x1FSpds_in;
	amrex::MultiArray4<const double> x1ConsVar_fc_in;
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		if (RIEMANN == RiemannSolver::HLLD || RIEMANN == RiemannSolver::LLF_MHD) {
			x1FSpds_in = (*x1FSpds_mf).arrays();
		}
		x1ConsVar_fc_in = (*x1ConsVar_fc_mf).const_arrays();
	}

	// Include ghost cells when computing face velocities
	amrex::IntVect ng{AMREX_D_DECL(nghost_vel, nghost_vel, nghost_vel)};
	auto const faceKernel = [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in) {
		// first capture
		[[maybe_unused]] std::remove_cv_t<std::remove_reference_t<decltype(x1ConsVar_fc_in[bx])>> x1ConsVar_fc_ref{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			x1ConsVar_fc_ref = x1ConsVar_fc_in[bx];
		}
		[[maybe_unused]] std::remove_cv_t<std::remove_reference_t<decltype(x1FSpds_in[bx])>> x1FSpds_ref{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			x1FSpds_ref = x1FSpds_in[bx];
		}
		// dx/shearViscosity/bulkViscosity are only read inside `if constexpr` below; nvcc can't first-capture
		// a variable from inside constexpr-if, so touch them unconditionally here first
		[[maybe_unused]] auto const dx_captured = dx;
		[[maybe_unused]] amrex::Real const shearViscosity_captured = shearViscosity;
		[[maybe_unused]] amrex::Real const bulkViscosity_captured = bulkViscosity;

		// FUSED_ORDER / accumulate: touch the captured variables unconditionally (nvcc first-capture inside constexpr-if)
		[[maybe_unused]] auto const x1LeftState_captured = x1LeftState_in;
		[[maybe_unused]] auto const x1RightState_captured = x1RightState_in;
		[[maybe_unused]] auto const chiCell_captured = chiCell_in;

		quokka::Array4View<const amrex::Real, DIR> q(primVar_in[bx]);
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		// fused reconstruction: the left state of face i is the flattened a_plus of zone i - 1,
		// the right state is the flattened a_minus of zone i (as ReconstructStates* + FlattenShocks store them)
		[[maybe_unused]] double regLeft[(FUSED_ORDER != 0) ? nvar_ : 1];
		[[maybe_unused]] double regRight[(FUSED_ORDER != 0) ? nvar_ : 1];
		if constexpr (FUSED_ORDER != 0) {
			quokka::Array4View<const amrex::Real, DIR> chiCell(chiCell_in[bx]);
			const amrex::Real chi_minus = chiCell(i - 1, j, k);
			const amrex::Real chi_plus = chiCell(i, j, k);
			for (int n = 0; n < nvar_; ++n) {
				regLeft[n] = FlattenedEdges<DIR, FUSED_ORDER, limiter>(q, n, i - 1, j, k, chi_minus).second;
				regRight[n] = FlattenedEdges<DIR, FUSED_ORDER, limiter>(q, n, i, j, k, chi_plus).first;
			}
		}
		auto const x1LeftState = [&]() {
			if constexpr (FUSED_ORDER != 0) {
				return quokka::RegisterStateView{regLeft};
			} else {
				return quokka::Array4View<const amrex::Real, DIR>(x1LeftState_in[bx]);
			}
		}();
		auto const x1RightState = [&]() {
			if constexpr (FUSED_ORDER != 0) {
				return quokka::RegisterStateView{regRight};
			} else {
				return quokka::Array4View<const amrex::Real, DIR>(x1RightState_in[bx]);
			}
		}();
		[[maybe_unused]] amrex::Array4<const amrex::Real> x1LeftState_bfield_ref{};
		[[maybe_unused]] amrex::Array4<const amrex::Real> x1RightState_bfield_ref{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			x1LeftState_bfield_ref = x1LeftState_bfield_in[bx];
			x1RightState_bfield_ref = x1RightState_bfield_in[bx];
		}
		[[maybe_unused]] quokka::Array4View<const amrex::Real, DIR> x1LeftState_bfield(x1LeftState_bfield_ref);
		[[maybe_unused]] quokka::Array4View<const amrex::Real, DIR> x1RightState_bfield(x1RightState_bfield_ref);
		quokka::Array4View<amrex::Real, DIR> x1Flux(x1Flux_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1FaceVel(x1FaceVel_in[bx]);

		// Riemann flux and face velocity (pure function of the interface states and the primitive stencil)
		auto const [F, faceVelNew] = RiemannFaceFlux<RIEMANN, DIR>(x1LeftState, x1RightState, x1LeftState_bfield, x1RightState_bfield, x1ConsVar_fc_ref,
									   x1FSpds_ref, q, i, j, k, K_visc, dx, shearViscosity, bulkViscosity);
		if (accumulate) {
			const double v_old = accumWeightOld * x1FaceVel(i, j, k);
			x1FaceVel(i, j, k) = v_old + accumWeightNew * faceVelNew;
		} else {
			x1FaceVel(i, j, k) = faceVelNew;
		}

		// copy all flux components to the flux array
		for (int nc = 0; nc < nHydroScalars_; ++nc) {
			AMREX_ASSERT(!std::isnan(F[nc])); // check flux is valid
			if (accumulate) {
				// same operations as MultiFab::Saxpy into a zeroed array (stage 1) followed by Saxpy (stage 2)
				const double F_old = accumWeightOld * x1Flux(i, j, k, nc);
				x1Flux(i, j, k, nc) = F_old + accumWeightNew * F[nc];
			} else {
				x1Flux(i, j, k, nc) = F[nc];
			}
		}

		// calculate dust fluxes if enabled
		if constexpr (Physics_Traits<problem_t>::is_dust_enabled) {
			DustSystem<problem_t>::template ComputeDustFluxes<DIR>(x1Flux, x1LeftState, x1RightState, i, j, k);
		}
	};
	if constexpr (MT == AMREX_GPU_MAX_THREADS) {
		quokka::ParallelForHydro(x1Flux_mf, ng, faceKernel);
	} else {
		// amrex::ParallelFor<MT>(MultiFab, ...) ignores MT when the MultiFab has a single box, so launch box by box
		for (amrex::MFIter mfi(x1Flux_mf, amrex::MFItInfo().DisableDeviceSync()); mfi.isValid(); ++mfi) {
			const int lbx = mfi.LocalIndex();
			amrex::ParallelFor<MT>(amrex::grow(mfi.validbox(), ng),
					       [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { faceKernel(lbx, i, j, k); });
		}
	}
}

template <typename problem_t>
template <RiemannSolver RIEMANN, FluxDir DIR, typename StateViewL, typename StateViewR>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::RiemannFaceFlux(
    StateViewL const &x1LeftState, StateViewR const &x1RightState, quokka::Array4View<const amrex::Real, DIR> const &x1LeftState_bfield,
    quokka::Array4View<const amrex::Real, DIR> const &x1RightState_bfield, amrex::Array4<const amrex::Real> const &x1ConsVar_fc_ref,
    amrex::Array4<amrex::Real> const &x1FSpds_ref, quokka::Array4View<const amrex::Real, DIR> const &q, const int i, const int j, const int k,
    const amrex::Real K_visc, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, const amrex::Real shearViscosity, const amrex::Real bulkViscosity)
    -> FaceFlux
{
	// Riemann flux and face-centered normal velocity at the face (i, j, k) [permuted to DIR], given its left and right
	// interface states (body of the ComputeFluxes kernel)
	amrex::ignore_unused(x1LeftState_bfield, x1RightState_bfield, x1ConsVar_fc_ref, x1FSpds_ref, dx, shearViscosity, bulkViscosity);
	double faceVelOut = NAN;

	// gather left- and right- state variables

	const double rho_L = x1LeftState(i, j, k, primDensity_index);
	const double rho_R = x1RightState(i, j, k, primDensity_index);

	const double vx_L = x1LeftState(i, j, k, x1Velocity_index);
	const double vx_R = x1RightState(i, j, k, x1Velocity_index);

	const double vy_L = x1LeftState(i, j, k, x2Velocity_index);
	const double vy_R = x1RightState(i, j, k, x2Velocity_index);

	const double vz_L = x1LeftState(i, j, k, x3Velocity_index);
	const double vz_R = x1RightState(i, j, k, x3Velocity_index);

	const double ke_L = 0.5 * rho_L * (vx_L * vx_L + vy_L * vy_L + vz_L * vz_L);
	const double ke_R = 0.5 * rho_R * (vx_R * vx_R + vy_R * vy_R + vz_R * vz_R);

	// auxiliary Eint (rho * e)
	// this is evolved as a passive scalar by the Riemann solver
	double Eint_L = NAN;
	double Eint_R = NAN;

	double P_L = NAN;
	double P_R = NAN;

	double E_L = NAN;
	double E_R = NAN;

	double cs_L = NAN;
	double cs_R = NAN;

	// EOS quantities of the two interface states, one EOS evaluation per side (also used by HLLC)
	::quokka::EOS_RPState rp_L{};
	::quokka::EOS_RPState rp_R{};

	double bx1 = 0.0;
	double by_L = 0.0;
	double bz_L = 0.0;
	double by_R = 0.0;
	double bz_R = 0.0;
	double magnetic_energy_L = 0.0;
	double magnetic_energy_R = 0.0;
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		quokka::Array4View<const amrex::Real, DIR> x1ConsVar_fc(x1ConsVar_fc_ref);
		bx1 = x1ConsVar_fc(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		by_L = x1LeftState_bfield(i, j, k, 0);
		bz_L = x1LeftState_bfield(i, j, k, 1);
		by_R = x1RightState_bfield(i, j, k, 0);
		bz_R = x1RightState_bfield(i, j, k, 1);
		magnetic_energy_L = 0.5 * (bx1 * bx1 + by_L * by_L + bz_L * bz_L);
		magnetic_energy_R = 0.5 * (bx1 * bx1 + by_R * by_R + bz_R * bz_R);
	}

	if constexpr (is_eos_isothermal()) {
		P_L = rho_L * (cs_iso_ * cs_iso_);
		P_R = rho_R * (cs_iso_ * cs_iso_);

		cs_L = cs_iso_;
		cs_R = cs_iso_;

		// total and auxiliary internal energy are not evolved under an isothermal EOS;
		// zero them here so the Riemann solvers never see NaN placeholders
		E_L = 0.0;
		E_R = 0.0;
		Eint_L = 0.0;
		Eint_R = 0.0;
	} else {
		if constexpr (reconstruct_eint) {
			// compute pressure from specific internal energy
			// (pressure_index is actually eint)
			const double eint_L = x1LeftState(i, j, k, pressure_index);
			const double eint_R = x1RightState(i, j, k, pressure_index);
			amrex::GpuArray<Real, nmscalars_> massScalars_L = RadSystem<problem_t>::ComputeMassScalars(x1LeftState, i, j, k);
			P_L = ::quokka::EOS<problem_t>::ComputePressure(rho_L, eint_L * rho_L, massScalars_L);
			amrex::GpuArray<Real, nmscalars_> massScalars_R = RadSystem<problem_t>::ComputeMassScalars(x1RightState, i, j, k);
			P_R = ::quokka::EOS<problem_t>::ComputePressure(rho_R, eint_R * rho_R, massScalars_R);

			// auxiliary Eint is actually (auxiliary) specific internal energy
			Eint_L = rho_L * x1LeftState(i, j, k, primEint_index);
			Eint_R = rho_R * x1RightState(i, j, k, primEint_index);
		} else {
			// pressure_index is actually pressure
			P_L = x1LeftState(i, j, k, pressure_index);
			P_R = x1RightState(i, j, k, pressure_index);

			// primEint_index is actually (rho * e)
			Eint_L = x1LeftState(i, j, k, primEint_index);
			Eint_R = x1RightState(i, j, k, primEint_index);
		}

		amrex::GpuArray<Real, nmscalars_> massScalars_L = RadSystem<problem_t>::ComputeMassScalars(x1LeftState, i, j, k);
		rp_L = ::quokka::EOS<problem_t>::ComputeRPState(rho_L, P_L, massScalars_L);
		cs_L = rp_L.cs;
		E_L = rp_L.eint + ke_L + magnetic_energy_L;

		amrex::GpuArray<Real, nmscalars_> massScalars_R = RadSystem<problem_t>::ComputeMassScalars(x1RightState, i, j, k);
		rp_R = ::quokka::EOS<problem_t>::ComputeRPState(rho_R, P_R, massScalars_R);
		cs_R = rp_R.cs;
		E_R = rp_R.eint + ke_R + magnetic_energy_R;
	}

	AMREX_ASSERT(cs_L > 0.0);
	AMREX_ASSERT(cs_R > 0.0);

	// assign normal component of velocity according to DIR

	int velN_index = x1Velocity_index;
	int velV_index = x2Velocity_index;
	int velW_index = x3Velocity_index;

	// grid spacing along normal/transverse directions, for the viscous stress below (3D only)
	[[maybe_unused]] amrex::Real dx_n = NAN;
	[[maybe_unused]] amrex::Real dx_v = NAN;
	[[maybe_unused]] amrex::Real dx_w = NAN;

	if constexpr (DIR == FluxDir::X1) {
		velN_index = x1Velocity_index;
		velV_index = x2Velocity_index;
		velW_index = x3Velocity_index;
		if constexpr (AMREX_SPACEDIM == 3) {
			dx_n = dx[0];
			dx_v = dx[1];
			dx_w = dx[2];
		}
	} else if constexpr (DIR == FluxDir::X2) {
#if (AMREX_SPACEDIM == 2)
		velN_index = x2Velocity_index;
		velV_index = x1Velocity_index;
		velW_index = x3Velocity_index; // unchanged in 2D
#endif
#if (AMREX_SPACEDIM == 3)
		velN_index = x2Velocity_index;
		velV_index = x3Velocity_index;
		velW_index = x1Velocity_index;
		dx_n = dx[1];
		dx_v = dx[2];
		dx_w = dx[0];
#endif
	} else if constexpr (DIR == FluxDir::X3) {
		velN_index = x3Velocity_index;
		velV_index = x1Velocity_index;
		velW_index = x2Velocity_index;
		if constexpr (AMREX_SPACEDIM == 3) {
			dx_n = dx[2];
			dx_v = dx[0];
			dx_w = dx[1];
		}
	}

	quokka::HydroState<nscalars_, nmscalars_> sL{};
	sL.rho = rho_L;
	sL.u = x1LeftState(i, j, k, velN_index);
	sL.v = x1LeftState(i, j, k, velV_index);
	sL.w = x1LeftState(i, j, k, velW_index);
	sL.P = P_L;
	sL.cs = cs_L;
	sL.E = E_L;
	sL.Eint = Eint_L;
	sL.by = by_L;
	sL.bz = bz_L;

	quokka::HydroState<nscalars_, nmscalars_> sR{};
	sR.rho = rho_R;
	sR.u = x1RightState(i, j, k, velN_index);
	sR.v = x1RightState(i, j, k, velV_index);
	sR.w = x1RightState(i, j, k, velW_index);
	sR.P = P_R;
	sR.cs = cs_R;
	sR.E = E_R;
	sR.Eint = Eint_R;
	sR.by = by_R;
	sR.bz = bz_R;

	// The remaining components are mass scalars and passive scalars, so just copy them from
	// x1LeftState and x1RightState into the (left, right) state vectors U_L and
	// U_R
	for (int n = 0; n < nscalars_; ++n) {
		sL.scalar[n] = x1LeftState(i, j, k, scalar0_index + n);
		sR.scalar[n] = x1RightState(i, j, k, scalar0_index + n);
		// also store mass scalars separately
		if (n < nmscalars_) {
			sL.massScalar[n] = x1LeftState(i, j, k, scalar0_index + n);
			sR.massScalar[n] = x1RightState(i, j, k, scalar0_index + n);
		}
	}

	// difference in normal velocity along normal axis
	const double du = q(i, j, k, velN_index) - q(i - 1, j, k, velN_index);

	// difference in transverse velocity
#if AMREX_SPACEDIM == 1
	const double dw = 0.;
#else
	amrex::Real dvl = std::min(q(i - 1, j + 1, k, velV_index) - q(i - 1, j, k, velV_index), q(i - 1, j, k, velV_index) - q(i - 1, j - 1, k, velV_index));
	amrex::Real dvr = std::min(q(i, j + 1, k, velV_index) - q(i, j, k, velV_index), q(i, j, k, velV_index) - q(i, j - 1, k, velV_index));
	double dw = std::min(dvl, dvr);
#endif
#if AMREX_SPACEDIM == 3
	amrex::Real dwl = std::min(q(i - 1, j, k + 1, velW_index) - q(i - 1, j, k, velW_index), q(i - 1, j, k, velW_index) - q(i - 1, j, k - 1, velW_index));
	amrex::Real dwr = std::min(q(i, j, k + 1, velW_index) - q(i, j, k, velW_index), q(i, j, k, velW_index) - q(i, j, k - 1, velW_index));
	dw = std::min({dwl, dwr, dw});
#endif

	// solve the Riemann problem in canonical form (i.e., where the x-dir is the normal direction)
	quokka::valarray<double, nHydroScalars_> F_canonical{};

	if constexpr (RIEMANN == RiemannSolver::HLLC) {
		static_assert(!Physics_Traits<problem_t>::is_mhd_enabled, "Cannot use HLLC solver for MHD problems!");
		if constexpr (is_eos_isothermal()) {
			F_canonical = quokka::Riemann::HLLC<problem_t, nscalars_, nmscalars_, nHydroScalars_>(sL, sR, gamma_, du, dw);
		} else {
			// reuse the EOS derivatives evaluated above (same values HLLC would compute itself)
			F_canonical = quokka::Riemann::HLLC<problem_t, nscalars_, nmscalars_, nHydroScalars_>(sL, sR, gamma_, du, dw, &rp_L, &rp_R);
		}
	} else if constexpr (RIEMANN == RiemannSolver::LLF) {
		F_canonical = quokka::Riemann::LLF<problem_t, nscalars_, nmscalars_, nHydroScalars_>(sL, sR);
	} else if constexpr (RIEMANN == RiemannSolver::LLF_MHD) {
		quokka::Array4View<amrex::Real, DIR> x1FSpds(x1FSpds_ref);
		auto [F_canonical_tmp, fspd_m, fspd_p] = quokka::Riemann::LLF_MHD<problem_t, nscalars_, nmscalars_, nHydroScalars_>(sL, sR, gamma_, bx1);
		F_canonical = F_canonical_tmp;
		x1FSpds(i, j, k, 0) = fspd_m;
		x1FSpds(i, j, k, 1) = fspd_p;
	} else if constexpr (RIEMANN == RiemannSolver::HLLD) {
		quokka::Array4View<amrex::Real, DIR> x1FSpds(x1FSpds_ref);
		auto [F_canonical_tmp, fspd_m, fspd_p] = quokka::Riemann::HLLD<problem_t, nscalars_, nmscalars_, nHydroScalars_>(sL, sR, gamma_, bx1, dw);
		F_canonical = F_canonical_tmp;
		x1FSpds(i, j, k, 0) = fspd_m;
		x1FSpds(i, j, k, 1) = fspd_p;
	}

	// add artificial viscosity
	// following Colella & Woodward (1984), eq. (4.2)
	const double div_v = AMREX_D_TERM(du, +0.5 * (dvl + dvr), +0.5 * (dwl + dwr));
	const double viscosity = K_visc * std::max(-div_v, 0.);

	quokka::valarray<double, nHydroScalars_> U_L = {sL.rho, sL.rho * sL.u, sL.rho * sL.v, sL.rho * sL.w, sL.E, sL.Eint};
	quokka::valarray<double, nHydroScalars_> U_R = {sR.rho, sR.rho * sR.u, sR.rho * sR.v, sR.rho * sR.w, sR.E, sR.Eint};

	// conserve flux of mass scalars
	// based on Plewa and Muller 1999, A&A, 342, 179 (equations 8 and 12)
	amrex::Real fluxSum_U_L = 0;
	amrex::Real fluxSum_U_R = 0;

	for (int n = 0; n < nscalars_; ++n) {
		const int nstart = nHydroScalars_ - nscalars_;
		U_L[nstart + n] = sL.scalar[n];
		U_R[nstart + n] = sR.scalar[n];

		if (n < nmscalars_) {
			fluxSum_U_L += U_L[nstart + n];
			fluxSum_U_R += U_R[nstart + n];
		}
	}

	F_canonical = F_canonical + viscosity * (U_L - U_R);

	// physical shear/bulk viscosity, added the same way as the artificial viscosity above
	if constexpr (AMREX_SPACEDIM == 3 && Physics_Traits<problem_t>::viscosity_model != ViscosityModel::none) {
		amrex::Real localShearViscosity = shearViscosity;
		amrex::Real localBulkViscosity = bulkViscosity;
		if constexpr (Physics_Traits<problem_t>::viscosity_model == ViscosityModel::problem_defined) {
			const auto local_viscosity = computeViscosity<problem_t, DIR>(i, j, k, q);
			localShearViscosity = local_viscosity[0];
			localBulkViscosity = local_viscosity[1];
		}
		if (localShearViscosity != 0.0 || localBulkViscosity != 0.0) {
			const auto sigma =
			    ComputeViscousFlux<DIR>(q, i, j, k, velN_index, velV_index, velW_index, dx_n, dx_v, dx_w, localShearViscosity, localBulkViscosity);
			F_canonical[x1Momentum_index] -= sigma[0];
			F_canonical[x2Momentum_index] -= sigma[1];
			F_canonical[x3Momentum_index] -= sigma[2];
			F_canonical[energy_index] -= sigma[3]; // zeroed below if EOS is isothermal
		}
	}

	quokka::valarray<double, nHydroScalars_> F = F_canonical;

	// permute momentum components according to flux direction DIR
	F[velN_index] = F_canonical[x1Momentum_index];
	F[velV_index] = F_canonical[x2Momentum_index];
	F[velW_index] = F_canonical[x3Momentum_index];

	// set energy fluxes to zero if EOS is isothermal
	if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
		F[energy_index] = 0;
		F[internalEnergy_index] = 0;
	}

	// compute face-centered normal velocity using HLL star state; matches Mignone21a eqn. (29), feeds
	// ComputeEMF_Quokka2026's face velocity.
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		quokka::Array4View<amrex::Real, DIR> x1FSpds(x1FSpds_ref);
		amrex::Real const fspd_m = x1FSpds(i, j, k, 0);
		amrex::Real const fspd_p = x1FSpds(i, j, k, 1);
		if (fspd_p + fspd_m > 0.0) {
			faceVelOut = (fspd_p * sL.u + fspd_m * sR.u) / (fspd_p + fspd_m);
		} else {
			faceVelOut = 0.5 * (sL.u + sR.u);
		}
	} else {
		// compute face-centered normal velocity
		double v_norm = 0.0;
		if (F[density_index] >= 0.) {
			if (rho_R > 0.) {
				v_norm = F[density_index] / rho_R;
			}
		} else {
			if (rho_L > 0.) {
				v_norm = F[density_index] / rho_L;
			}
		}
		faceVelOut = v_norm;
	}

	// use the same logic as above to scale and conserve specie fluxes
	if (F[density_index] >= 0.) {
		for (int n = 0; n < nmscalars_; ++n) {
			const int nstart = nHydroScalars_ - nscalars_;
			if (fluxSum_U_L > 0.) {
				F[nstart + n] = F[density_index] * U_L[nstart + n] / fluxSum_U_L;
			} else {
				F[nstart + n] = 0.;
			}
		}
	} else {
		for (int n = 0; n < nmscalars_; ++n) {
			const int nstart = nHydroScalars_ - nscalars_;
			if (fluxSum_U_R > 0.) {
				F[nstart + n] = F[density_index] * U_R[nstart + n] / fluxSum_U_R;
			} else {
				F[nstart + n] = 0.;
			}
		}
	}

	return FaceFlux{F, faceVelOut};
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE auto HydroSystem<problem_t>::ComputeViscousFlux(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k, int velN_index,
								 int velV_index, int velW_index, amrex::Real dx_n, amrex::Real dx_v, amrex::Real dx_w,
								 amrex::Real shearViscosity, amrex::Real bulkViscosity) -> quokka::valarray<amrex::Real, 4>
{
	static_assert(AMREX_SPACEDIM == 3 || dependent_false_v<problem_t>, "HydroSystem::ComputeViscousFlux currently only supports 3D.");

	// dq<comp>_d<dir> = derivative of q's <comp>-component along <dir>; n/v/w are the local orthonormal basis
	// normal-direction derivatives: one-sided difference across the face, between cells i-1 and i
	const amrex::Real dqn_dn = (q(i, j, k, velN_index) - q(i - 1, j, k, velN_index)) / dx_n;
	const amrex::Real dqv_dn = (q(i, j, k, velV_index) - q(i - 1, j, k, velV_index)) / dx_n;
	const amrex::Real dqw_dn = (q(i, j, k, velW_index) - q(i - 1, j, k, velW_index)) / dx_n;

	// transverse derivatives: averaged centered difference
	const amrex::Real dqv_dv =
	    0.25 / dx_v * ((q(i, j + 1, k, velV_index) - q(i, j - 1, k, velV_index)) + (q(i - 1, j + 1, k, velV_index) - q(i - 1, j - 1, k, velV_index)));
	const amrex::Real dqn_dv =
	    0.25 / dx_v * ((q(i, j + 1, k, velN_index) - q(i, j - 1, k, velN_index)) + (q(i - 1, j + 1, k, velN_index) - q(i - 1, j - 1, k, velN_index)));
	const amrex::Real dqw_dw =
	    0.25 / dx_w * ((q(i, j, k + 1, velW_index) - q(i, j, k - 1, velW_index)) + (q(i - 1, j, k + 1, velW_index) - q(i - 1, j, k - 1, velW_index)));
	const amrex::Real dqn_dw =
	    0.25 / dx_w * ((q(i, j, k + 1, velN_index) - q(i, j, k - 1, velN_index)) + (q(i - 1, j, k + 1, velN_index) - q(i - 1, j, k - 1, velN_index)));

	const amrex::Real div_q = dqn_dn + dqv_dv + dqw_dw;

	// sigma_ni = 2*nu_shear*S_ni + nu_bulk*div[q]*delta_ni, i in {n,v,w}: sigma's normal row, not the full tensor
	const amrex::Real sigma_nn = 2.0 * shearViscosity * dqn_dn + (bulkViscosity - (2.0 / 3.0) * shearViscosity) * div_q;
	const amrex::Real sigma_nv = shearViscosity * (dqv_dn + dqn_dv);
	const amrex::Real sigma_nw = shearViscosity * (dqw_dn + dqn_dw);

	// viscous heating dot[q,sigma] at the face; caller drops this for isothermal EOS
	const amrex::Real q_face_n = 0.5 * (q(i, j, k, velN_index) + q(i - 1, j, k, velN_index));
	const amrex::Real q_face_v = 0.5 * (q(i, j, k, velV_index) + q(i - 1, j, k, velV_index));
	const amrex::Real q_face_w = 0.5 * (q(i, j, k, velW_index) + q(i - 1, j, k, velW_index));
	const amrex::Real q_dot_sigma = q_face_n * sigma_nn + q_face_v * sigma_nv + q_face_w * sigma_nw;

	// sigma's remaining rows are computed by the other two DIR sweeps, not here
	return {sigma_nn, sigma_nv, sigma_nw, q_dot_sigma};
}

#endif // HYDRO_SYSTEM_HPP_
