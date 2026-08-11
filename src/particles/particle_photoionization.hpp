#ifndef PARTICLE_PHOTOIONIZATION_HPP_
#define PARTICLE_PHOTOIONIZATION_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>

#include "AMReX_Array.H"
#include "AMReX_FillPatchUtil.H"
#include "AMReX_GMRES_MLMG.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MLABecLaplacian.H"
#include "AMReX_MLMG.H"
#include "AMReX_Math.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_Reduce.H"

#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "math/NewtonKrylovSolver.hpp"
#include "particles/particle_types.hpp"
#include "util/DataTable.hpp"

namespace quokka::photoionization
{

amrex::Real constexpr mass_to_table_units = 1.0 / C::M_solar;
amrex::Real constexpr age_to_table_units = 1.0 / 3.15576e7;
amrex::Real constexpr mH = 1.67e-24;
amrex::Real constexpr mean_particle_mass_mu = 1.27;
amrex::Real constexpr alphaB = 2.6e-13;
amrex::Real constexpr sigma_HI = 6.3e-18; // cm^2
bool constexpr table_axes_are_mass_age = true;

#if AMREX_SPACEDIM == 3
template <typename problem_t> class StromgrenHierarchyNewtonProblem
{
      public:
	using State = amrex::Vector<amrex::MultiFab>;
	using RT = amrex::Real;
	using FaceCoefficients = amrex::Vector<std::array<amrex::MultiFab, AMREX_SPACEDIM>>;

	StromgrenHierarchyNewtonProblem(amrex::Vector<amrex::Geometry> const &geometry, amrex::Vector<amrex::BoxArray> const &grids,
					amrex::Vector<amrex::DistributionMapping> const &distribution_map, amrex::Vector<amrex::IntVect> const &ref_ratio,
					amrex::Vector<amrex::MultiFab> const &hydro_state, State const &rhs, amrex::Vector<amrex::iMultiFab> const &masks,
					RT phi_scale, RT operator_rate, RT dt, RT residual_tolerance)
	    : geometry_(geometry), grids_(grids), distributionMap_(distribution_map), refRatio_(ref_ratio), hydroState_(hydro_state), rhs_(rhs), masks_(masks),
	      phiScale_(phi_scale), operatorRate_(operator_rate), dt_(dt), residualTolerance_(residual_tolerance), reactionRate_(makeState(0)),
	      trialReactionRate_(makeState(0)), jacobianReactionRate_(makeState(0)), zeroReactionRate_(makeState(0)), diffusionCoeff_(makeFaceCoefficients()),
	      trialDiffusionCoeff_(makeFaceCoefficients()), jacobianDiffusionDerivative_(makeFaceCoefficients())
	{
		AMREX_ALWAYS_ASSERT(dt_ > 0.0);
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			auto const bc = geometry_[0].isPeriodic(dir) ? amrex::LinOpBCType::Periodic : amrex::LinOpBCType::Dirichlet;
			bcLo_[dir] = bc;
			bcHi_[dir] = bc;
		}
		for (int lev = 0; lev < static_cast<int>(masks_.size()); ++lev) {
			auto const dx = geometry_[lev].CellSizeArray();
			RT const volume = dx[0] * dx[1] * dx[2];
			totalVolume_ += volume * static_cast<RT>(masks_[lev].sum(0, 0, true));
		}
		amrex::ParallelDescriptor::ReduceRealSum(totalVolume_);
	}

	void prepare(State const &state, RT pseudo_transient_shift = 0.0)
	{
		fillCoefficients(state, reactionRate_, diffusionCoeff_);
		for (auto &level : reactionRate_) {
			level.plus(pseudo_transient_shift * operatorRate_, 0, 1, 0);
		}
		operator_ = makeOperator(reactionRate_, diffusionCoeff_);
		mlmg_ = std::make_unique<amrex::MLMG>(*operator_);
		mlmg_->setVerbose(0);
		mlmg_->setBottomVerbose(0);
		mlmg_->setFixedIter(1);
		mlmg_->setBottomSolver(amrex::BottomSolver::smoother);
		preconditioner_ = std::make_unique<amrex::GMRESMLMG>(*mlmg_);
		preconditioner_->setPrecondNumIters(2);
	}

	void residual(State const &state, State &output) { evaluateResidual(state, output); }
	void linearized_residual(State const &state, State &output) { evaluateResidual(state, output); }
	void applyJacobian(State const &state, State const &direction, State &output)
	{
		fillJacobianCoefficients(state, direction, jacobianReactionRate_, jacobianDiffusionDerivative_);
		if (jacobianOperator_ == nullptr) {
			jacobianOperator_ = makeOperator(jacobianReactionRate_, diffusionCoeff_);
			jacobianMlmg_ = std::make_unique<amrex::MLMG>(*jacobianOperator_);
		} else {
			setOperatorCoefficients(*jacobianOperator_, jacobianReactionRate_, diffusionCoeff_);
		}
		State direction_with_ghosts = clone(direction);
		fillGhosts(direction_with_ghosts);
		jacobianMlmg_->apply(amrex::GetVecOfPtrs(output), amrex::GetVecOfPtrs(direction_with_ghosts));

		if (diffusionDerivativeOperator_ == nullptr) {
			diffusionDerivativeOperator_ = makeOperator(zeroReactionRate_, jacobianDiffusionDerivative_);
			diffusionDerivativeMlmg_ = std::make_unique<amrex::MLMG>(*diffusionDerivativeOperator_);
		} else {
			setOperatorCoefficients(*diffusionDerivativeOperator_, zeroReactionRate_, jacobianDiffusionDerivative_);
		}
		State state_with_ghosts = clone(state);
		fillGhosts(state_with_ghosts);
		State diffusion_derivative = makeVecRHS();
		diffusionDerivativeMlmg_->apply(amrex::GetVecOfPtrs(diffusion_derivative), amrex::GetVecOfPtrs(state_with_ghosts));

		for (int lev = 0; lev < static_cast<int>(output.size()); ++lev) {
			amrex::MultiFab::Add(output[lev], diffusion_derivative[lev], 0, 0, 1, 0);
			output[lev].mult(1.0 / operatorRate_, 0, 1, 0);
		}
	}

	void predict(State &state, RT damping)
	{
		AMREX_ALWAYS_ASSERT((damping > 0.0) && (damping <= 1.0));
		fillCoefficients(state, trialReactionRate_, trialDiffusionCoeff_);
		auto picard_operator = makeOperator(trialReactionRate_, trialDiffusionCoeff_);
		picard_operator->prepareForSolve();
		State state_with_ghosts = clone(state);
		fillGhosts(state_with_ghosts);
		State fixed_point_residual = makeVecRHS();
		for (int lev = 0; lev < static_cast<int>(state.size()); ++lev) {
			amrex::MultiFab const *coarse_state = (lev > 0) ? &state_with_ghosts[lev - 1] : nullptr;
			picard_operator->solutionResidual(lev, fixed_point_residual[lev], state_with_ghosts[lev], rhs_[lev], coarse_state);
			amrex::MultiFab::Saxpy(state[lev], damping / operatorRate_, fixed_point_residual[lev], 0, 0, 1, 0);
			auto arrays = state[lev].arrays();
			amrex::ParallelFor(state[lev], [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
				arrays[nbx](i, j, k, 0) = amrex::max(arrays[nbx](i, j, k, 0), 0.0);
			});
		}
		averageDown(state);
	}

	[[nodiscard]] auto makeVecRHS() const -> State { return makeState(0); }
	[[nodiscard]] auto makeVecLHS() const -> State { return makeState(1); }

	void precondition(State &output, State const &rhs)
	{
		AMREX_ALWAYS_ASSERT(preconditioner_ != nullptr);
		setToZero(output);
		State scaled_rhs = clone(rhs);
		scale(scaled_rhs, operatorRate_);
		amrex::GMRESMLMG::VEC output_vector;
		amrex::GMRESMLMG::VEC rhs_vector;
		for (int lev = 0; lev < static_cast<int>(output.size()); ++lev) {
			output_vector.emplace_back(output[lev], amrex::make_alias, 0, 1);
			rhs_vector.emplace_back(scaled_rhs[lev], amrex::make_alias, 0, 1);
		}
		preconditioner_->precond(output_vector, rhs_vector);
	}

	void assign(State &lhs, State const &rhs) const
	{
		for (int lev = 0; lev < static_cast<int>(lhs.size()); ++lev) {
			amrex::MultiFab::Copy(lhs[lev], rhs[lev], 0, 0, 1, std::min(lhs[lev].nGrow(), rhs[lev].nGrow()));
		}
	}

	[[nodiscard]] auto dotProduct(State const &lhs, State const &rhs) const -> RT
	{
		amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
		amrex::ReduceData<RT> reduce_data(reduce_op);
		using Tuple = typename decltype(reduce_data)::Type;
		for (int lev = 0; lev < static_cast<int>(lhs.size()); ++lev) {
			auto const dx = geometry_[lev].CellSizeArray();
			RT const volume = dx[0] * dx[1] * dx[2];
			for (amrex::MFIter mfi(lhs[lev]); mfi.isValid(); ++mfi) {
				auto const x = lhs[lev].const_array(mfi);
				auto const y = rhs[lev].const_array(mfi);
				auto const mask = masks_[lev].const_array(mfi);
				reduce_op.eval(mfi.validbox(), reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> Tuple {
					return {mask(i, j, k) != 0 ? volume * x(i, j, k, 0) * y(i, j, k, 0) : 0.0};
				});
			}
		}
		RT result = amrex::get<0>(reduce_data.value(reduce_op));
		amrex::ParallelDescriptor::ReduceRealSum(result);
		return result / totalVolume_;
	}

	void increment(State &lhs, State const &rhs, RT factor) const
	{
		for (int lev = 0; lev < static_cast<int>(lhs.size()); ++lev) {
			amrex::MultiFab::Saxpy(lhs[lev], factor, rhs[lev], 0, 0, 1, 0);
		}
	}

	void linComb(State &lhs, RT a, State const &rhs_a, RT b, State const &rhs_b) const
	{
		for (int lev = 0; lev < static_cast<int>(lhs.size()); ++lev) {
			amrex::MultiFab::LinComb(lhs[lev], a, rhs_a[lev], 0, b, rhs_b[lev], 0, 0, 1, 0);
		}
	}

	[[nodiscard]] auto norm2(State const &state) const -> RT { return std::sqrt(amrex::max(dotProduct(state, state), 0.0)); }
	void scale(State &state, RT factor) const
	{
		for (auto &level : state) {
			level.mult(factor, 0, 1, 0);
		}
	}
	void setToZero(State &state) const
	{
		for (auto &level : state) {
			level.setVal(0.0);
		}
	}
	[[nodiscard]] auto clone(State const &state) const -> State
	{
		State result;
		result.reserve(state.size());
		for (auto const &level : state) {
			result.emplace_back(level.boxArray(), level.DistributionMap(), 1, level.nGrow());
			amrex::MultiFab::Copy(result.back(), level, 0, 0, 1, level.nGrow());
		}
		return result;
	}

	void fill_candidate(State &candidate, State const &state, State const &correction, RT step_length) const
	{
		linComb(candidate, 1.0, state, step_length, correction);
		for (auto &level : candidate) {
			auto arrays = level.arrays();
			amrex::ParallelFor(level, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
				arrays[nbx](i, j, k, 0) = amrex::max(arrays[nbx](i, j, k, 0), 0.0);
			});
		}
		averageDown(candidate);
	}

	[[nodiscard]] auto admissible(State const &state) const -> bool
	{
		for (auto const &level : state) {
			RT const minimum = level.min(0, 0, false);
			RT const maximum = level.max(0, 0, false);
			if (!std::isfinite(minimum) || !std::isfinite(maximum) || (minimum < 0.0)) {
				return false;
			}
		}
		return true;
	}

	[[nodiscard]] auto relative_change(State const &lhs, State const &rhs) const -> RT
	{
		RT maximum = 0.0;
		for (int lev = 0; lev < static_cast<int>(lhs.size()); ++lev) {
			amrex::MultiFab change(grids_[lev], distributionMap_[lev], 1, 0);
			auto const lhs_arr = lhs[lev].const_arrays();
			auto const rhs_arr = rhs[lev].const_arrays();
			auto const hydro_arr = hydroState_[lev].const_arrays();
			auto change_arr = change.arrays();
			RT const phi_scale = phiScale_;
			RT const tolerance = residualTolerance_;
			RT const n_to_rho = mean_particle_mass_mu * mH;
			amrex::ParallelFor(change, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
				RT const nH = amrex::max(hydro_arr[nbx](i, j, k, HydroSystem<problem_t>::density_index) / n_to_rho, 0.0);
				auto ion_fraction = [=] AMREX_GPU_DEVICE(RT u) noexcept {
					RT const ng = amrex::max(u * phi_scale, 0.0);
					if ((nH <= 0.0) || (ng <= 0.0)) {
						return 0.0;
					}
					RT const ratio = 4.0 * alphaB * nH / (C::c_light * sigma_HI * ng);
					return 2.0 / (1.0 + std::sqrt(1.0 + ratio));
				};
				RT const xl = ion_fraction(lhs_arr[nbx](i, j, k, 0));
				RT const xr = ion_fraction(rhs_arr[nbx](i, j, k, 0));
				change_arr[nbx](i, j, k, 0) = tolerance * std::abs(xl - xr) / (1.0e-3 + tolerance * 0.5 * (std::abs(xl) + std::abs(xr)));
			});
			maximum = amrex::max(maximum, change.norm0(0, 0, false));
		}
		return maximum;
	}

      private:
	[[nodiscard]] auto makeState(int nghost) const -> State
	{
		State result;
		result.reserve(grids_.size());
		for (int lev = 0; lev < static_cast<int>(grids_.size()); ++lev) {
			result.emplace_back(grids_[lev], distributionMap_[lev], 1, nghost);
			result.back().setVal(0.0);
		}
		return result;
	}

	[[nodiscard]] auto makeFaceCoefficients() const -> FaceCoefficients
	{
		FaceCoefficients result(grids_.size());
		for (int lev = 0; lev < static_cast<int>(grids_.size()); ++lev) {
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				auto const face_grids = amrex::convert(grids_[lev], amrex::IntVect::TheDimensionVector(dir));
				result[lev][dir].define(face_grids, distributionMap_[lev], 1, 0);
			}
		}
		return result;
	}

	void fillGhosts(State &state) const
	{
		for (int lev = 0; lev < static_cast<int>(state.size()); ++lev) {
			state[lev].FillBoundary(geometry_[lev].periodicity());
			if (lev > 0) {
				amrex::Vector<amrex::MultiFab *> coarse{&state[lev - 1]};
				amrex::Vector<amrex::MultiFab *> fine{&state[lev]};
				amrex::Vector<RT> const times{0.0};
				amrex::Vector<amrex::BCRec> bcs(1);
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					int const type = geometry_[lev].isPeriodic(dir) ? amrex::BCType::int_dir : amrex::BCType::foextrap;
					bcs[0].setLo(dir, type);
					bcs[0].setHi(dir, type);
				}
				amrex::FillPatchTwoLevels(state[lev], amrex::IntVect(state[lev].nGrow()), amrex::IntVect(0), 0.0, coarse, times, fine, times, 0,
							  0, 1, geometry_[lev - 1], geometry_[lev], refRatio_[lev - 1], &amrex::cell_cons_interp, bcs, 0);
			}
		}
	}

	void averageDown(State &state) const
	{
		for (int lev = static_cast<int>(state.size()) - 2; lev >= 0; --lev) {
			amrex::average_down(state[lev + 1], state[lev], geometry_[lev + 1], geometry_[lev], 0, 1, refRatio_[lev]);
		}
	}

	void fillCoefficients(State const &dimensionless_state, State &reaction_rate, FaceCoefficients &diffusion_coeff) const
	{
		State state = clone(dimensionless_state);
		fillGhosts(state);
		for (int lev = 0; lev < static_cast<int>(state.size()); ++lev) {
			auto const state_arr = state[lev].const_arrays();
			auto const hydro_arr = hydroState_[lev].const_arrays();
			auto reaction_arr = reaction_rate[lev].arrays();
			RT const phi_scale = phiScale_;
			RT const n_to_rho = mean_particle_mass_mu * mH;
			RT const inv_dt = 1.0 / dt_;
			amrex::ParallelFor(reaction_rate[lev], [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
				RT const nH = amrex::max(hydro_arr[nbx](i, j, k, HydroSystem<problem_t>::density_index) / n_to_rho, 0.0);
				RT const ng = amrex::max(state_arr[nbx](i, j, k, 0) * phi_scale, 0.0);
				RT x = 0.0;
				if ((nH > 0.0) && (ng > 0.0)) {
					RT const ratio = 4.0 * alphaB * nH / (C::c_light * sigma_HI * ng);
					x = 2.0 / (1.0 + std::sqrt(1.0 + ratio));
				}
				reaction_arr[nbx](i, j, k, 0) = inv_dt + C::c_light * sigma_HI * nH * (1.0 - x);
			});

			auto const domain = geometry_[lev].Domain();
			auto const is_per = geometry_[lev].periodicity().intVect();
			auto const dx = geometry_[lev].CellSizeArray();
			auto const prob_lo = geometry_[lev].ProbLoArray();
			auto const prob_hi = geometry_[lev].ProbHiArray();
			RT const Lbox = std::min({prob_hi[0] - prob_lo[0], prob_hi[1] - prob_lo[1], prob_hi[2] - prob_lo[2]});
			RT const kappa_ref = 1.0 / Lbox;
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				for (amrex::MFIter mfi(diffusion_coeff[lev][dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
					auto const u = state[lev].const_array(mfi);
					auto const diffusion = diffusion_coeff[lev][dir].array(mfi);
					amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
						int il = i;
						int jl = j;
						int kl = k;
						if (dir == 0) {
							--il;
						} else if (dir == 1) {
							--jl;
						} else {
							--kl;
						}
						int const face_index = (dir == 0) ? i : ((dir == 1) ? j : k);
						bool const at_lo = (is_per[dir] == 0) && (face_index == domain.smallEnd(dir));
						bool const at_hi = (is_per[dir] == 0) && (face_index == domain.bigEnd(dir) + 1);
						RT const phi_lo = at_lo ? 0.0 : amrex::max(u(il, jl, kl, 0) * phi_scale, 0.0);
						RT const phi_hi = at_hi ? 0.0 : amrex::max(u(i, j, k, 0) * phi_scale, 0.0);
						RT const distance = (at_lo || at_hi) ? 0.5 * dx[dir] : dx[dir];
						RT const gradient = (phi_hi - phi_lo) / distance;
						RT const phi_face = 0.5 * (phi_lo + phi_hi);
						RT const R = std::abs(gradient) / amrex::max(kappa_ref * phi_face, 1.0e-30);
						RT const lambda = (2.0 + R) / (6.0 + 3.0 * R + R * R);
						diffusion(i, j, k, 0) = C::c_light * lambda / kappa_ref;
					});
				}
			}
		}
	}

	void fillJacobianCoefficients(State const &dimensionless_state, State const &direction, State &reaction_tangent,
				      FaceCoefficients &diffusion_derivative) const
	{
		State state = clone(dimensionless_state);
		State tangent = clone(direction);
		fillGhosts(state);
		fillGhosts(tangent);
		for (int lev = 0; lev < static_cast<int>(state.size()); ++lev) {
			auto const state_arr = state[lev].const_arrays();
			auto const hydro_arr = hydroState_[lev].const_arrays();
			auto reaction_arr = reaction_tangent[lev].arrays();
			RT const phi_scale = phiScale_;
			RT const n_to_rho = mean_particle_mass_mu * mH;
			RT const inv_dt = 1.0 / dt_;
			amrex::ParallelFor(reaction_tangent[lev], [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
				RT const nH = amrex::max(hydro_arr[nbx](i, j, k, HydroSystem<problem_t>::density_index) / n_to_rho, 0.0);
				RT const ng = amrex::max(state_arr[nbx](i, j, k, 0) * phi_scale, 0.0);
				RT const absorption_rate = C::c_light * sigma_HI * nH;
				RT absorption_tangent = absorption_rate;
				if ((nH > 0.0) && (ng > 0.0)) {
					RT const s = std::sqrt(1.0 + 4.0 * alphaB * nH / (C::c_light * sigma_HI * ng));
					RT const x = 2.0 / (1.0 + s);
					RT const ng_dx_dng = (s - 1.0) / (s * (s + 1.0));
					absorption_tangent = absorption_rate * (1.0 - x - ng_dx_dng);
				}
				reaction_arr[nbx](i, j, k, 0) = inv_dt + absorption_tangent;
			});

			auto const domain = geometry_[lev].Domain();
			auto const is_per = geometry_[lev].periodicity().intVect();
			auto const dx = geometry_[lev].CellSizeArray();
			auto const prob_lo = geometry_[lev].ProbLoArray();
			auto const prob_hi = geometry_[lev].ProbHiArray();
			RT const Lbox = std::min({prob_hi[0] - prob_lo[0], prob_hi[1] - prob_lo[1], prob_hi[2] - prob_lo[2]});
			RT const kappa_ref = 1.0 / Lbox;
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				for (amrex::MFIter mfi(diffusion_derivative[lev][dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
					auto const u = state[lev].const_array(mfi);
					auto const v = tangent[lev].const_array(mfi);
					auto const diffusion_delta = diffusion_derivative[lev][dir].array(mfi);
					amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
						int il = i;
						int jl = j;
						int kl = k;
						if (dir == 0) {
							--il;
						} else if (dir == 1) {
							--jl;
						} else {
							--kl;
						}
						int const face_index = (dir == 0) ? i : ((dir == 1) ? j : k);
						bool const at_lo = (is_per[dir] == 0) && (face_index == domain.smallEnd(dir));
						bool const at_hi = (is_per[dir] == 0) && (face_index == domain.bigEnd(dir) + 1);
						RT const phi_lo = at_lo ? 0.0 : amrex::max(u(il, jl, kl, 0) * phi_scale, 0.0);
						RT const phi_hi = at_hi ? 0.0 : amrex::max(u(i, j, k, 0) * phi_scale, 0.0);
						RT const delta_phi_lo = at_lo ? 0.0 : v(il, jl, kl, 0) * phi_scale;
						RT const delta_phi_hi = at_hi ? 0.0 : v(i, j, k, 0) * phi_scale;
						RT const distance = (at_lo || at_hi) ? 0.5 * dx[dir] : dx[dir];
						RT const gradient = (phi_hi - phi_lo) / distance;
						RT const delta_gradient = (delta_phi_hi - delta_phi_lo) / distance;
						RT const phi_face = 0.5 * (phi_lo + phi_hi);
						RT const delta_phi_face = 0.5 * (delta_phi_lo + delta_phi_hi);
						RT const raw_denominator = kappa_ref * phi_face;
						RT const denominator = amrex::max(raw_denominator, 1.0e-30);
						RT const delta_denominator = (raw_denominator > 1.0e-30) ? kappa_ref * delta_phi_face : 0.0;
						RT const abs_gradient = std::abs(gradient);
						RT const delta_abs_gradient = (gradient > 0.0) ? delta_gradient : ((gradient < 0.0) ? -delta_gradient : 0.0);
						RT const R = abs_gradient / denominator;
						RT const delta_R =
						    delta_abs_gradient / denominator - abs_gradient * delta_denominator / (denominator * denominator);
						RT const limiter_denominator = 6.0 + 3.0 * R + R * R;
						RT const lambda_derivative = -R * (4.0 + R) / (limiter_denominator * limiter_denominator);
						diffusion_delta(i, j, k, 0) = (C::c_light / kappa_ref) * lambda_derivative * delta_R;
					});
				}
			}
		}
	}

	void configureOperator(amrex::MLABecLaplacian &op, State const &reaction_rate, FaceCoefficients const &diffusion_coeff) const
	{
		op.setDomainBC(bcLo_, bcHi_);
		op.setScalars(1.0, 1.0);
		op.setMaxOrder(2);
		for (int lev = 0; lev < static_cast<int>(reaction_rate.size()); ++lev) {
			op.setLevelBC(lev, nullptr);
			op.setACoeffs(lev, reaction_rate[lev]);
			op.setBCoeffs(lev, amrex::GetArrOfConstPtrs(diffusion_coeff[lev]));
		}
	}

	void setOperatorCoefficients(amrex::MLABecLaplacian &op, State const &reaction_rate, FaceCoefficients const &diffusion_coeff) const
	{
		for (int lev = 0; lev < static_cast<int>(reaction_rate.size()); ++lev) {
			op.setACoeffs(lev, reaction_rate[lev]);
			op.setBCoeffs(lev, amrex::GetArrOfConstPtrs(diffusion_coeff[lev]));
		}
	}

	[[nodiscard]] auto makeOperator(State const &reaction_rate, FaceCoefficients const &diffusion_coeff) const -> std::unique_ptr<amrex::MLABecLaplacian>
	{
		amrex::LPInfo info;
		info.setDeterministic(true);
		if (geometry_.size() > 1) {
			info.setAgglomeration(false);
			info.setConsolidation(false);
		}
		auto op = std::make_unique<amrex::MLABecLaplacian>(geometry_, grids_, distributionMap_, info);
		configureOperator(*op, reaction_rate, diffusion_coeff);
		return op;
	}

	void evaluateResidual(State const &state, State &output)
	{
		fillCoefficients(state, trialReactionRate_, trialDiffusionCoeff_);
		auto trial_operator = makeOperator(trialReactionRate_, trialDiffusionCoeff_);
		amrex::MLMG trial_mlmg(*trial_operator);
		State state_with_ghosts = clone(state);
		fillGhosts(state_with_ghosts);
		trial_mlmg.apply(amrex::GetVecOfPtrs(output), amrex::GetVecOfPtrs(state_with_ghosts));
		for (int lev = 0; lev < static_cast<int>(output.size()); ++lev) {
			output[lev].mult(1.0 / operatorRate_, 0, 1, 0);
			amrex::MultiFab::Saxpy(output[lev], -1.0 / operatorRate_, rhs_[lev], 0, 0, 1, 0);
		}
	}

	amrex::Vector<amrex::Geometry> const &geometry_;
	amrex::Vector<amrex::BoxArray> const &grids_;
	amrex::Vector<amrex::DistributionMapping> const &distributionMap_;
	amrex::Vector<amrex::IntVect> const &refRatio_;
	amrex::Vector<amrex::MultiFab> const &hydroState_;
	State const &rhs_;
	amrex::Vector<amrex::iMultiFab> const &masks_;
	RT phiScale_;
	RT operatorRate_;
	RT dt_;
	RT residualTolerance_;
	RT totalVolume_ = 0.0;
	State reactionRate_;
	State trialReactionRate_;
	State jacobianReactionRate_;
	State zeroReactionRate_;
	FaceCoefficients diffusionCoeff_;
	FaceCoefficients trialDiffusionCoeff_;
	FaceCoefficients jacobianDiffusionDerivative_;
	amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bcLo_{};
	amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bcHi_{};
	std::unique_ptr<amrex::MLABecLaplacian> operator_;
	std::unique_ptr<amrex::MLMG> mlmg_;
	std::unique_ptr<amrex::GMRESMLMG> preconditioner_;
	std::unique_ptr<amrex::MLABecLaplacian> jacobianOperator_;
	std::unique_ptr<amrex::MLMG> jacobianMlmg_;
	std::unique_ptr<amrex::MLABecLaplacian> diffusionDerivativeOperator_;
	std::unique_ptr<amrex::MLMG> diffusionDerivativeMlmg_;
};

template <typename problem_t, quokka::OutOfBounds oob_policy>
void DepositStromgrenPhotonSourceAtLevel(quokka::StochasticStellarPopParticleContainer<problem_t> *stellar_particles, int lev, amrex::Real time,
					 amrex::BoxArray const &ba_lev, amrex::DistributionMapping const &dm_lev, amrex::Geometry const &geom_lev,
					 amrex::MultiFab &source_q, quokka::DataTableGpuConst<2, 1, oob_policy> const &qh0_table)
{
	if (stellar_particles == nullptr) {
		source_q.setVal(0.0);
		return;
	}

	auto const p_lo = geom_lev.ProbLoArray();
	auto const dxi = geom_lev.InvCellSizeArray();
	AMREX_ALWAYS_ASSERT(source_q.boxArray() == ba_lev);
	AMREX_ALWAYS_ASSERT(source_q.DistributionMap() == dm_lev);
	AMREX_ALWAYS_ASSERT(source_q.nComp() == 1 && source_q.nGrow() >= 1);
	source_q.setVal(0.0);

	auto const domain = geom_lev.Domain();
	auto const dom_lo = amrex::lbound(domain);
	auto const dom_hi = amrex::ubound(domain);
	auto const is_per = geom_lev.periodicity().intVect();

	// Deposit ionizing photon luminosity (photons/s) from individually sampled stars only.
	// We exclude LowMassComposite, SNRemnant, and Removed particles.
	for (quokka::StochasticStellarPopParticleIterator<problem_t> pti(*stellar_particles, lev); pti.isValid(); ++pti) {
		auto &particles = pti.GetArrayOfStructs();
		auto *pData = particles().data();
		auto const np = pti.numParticles();
		auto const src = source_q.array(pti);
		auto const box = amrex::grow(pti.validbox(), 1);
		auto const lo = amrex::lbound(box);
		auto const hi = amrex::ubound(box);

		amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) noexcept {
			auto const &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			int const stage = p.idata(quokka::StochasticStellarPopParticleStageIdx);
			bool const is_individual_ionizing_star = (stage == static_cast<int>(quokka::StellarEvolutionStage::HighMassNonExploding)) ||
								 (stage == static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor));
			if (!is_individual_ionizing_star) {
				return;
			}

			amrex::Real const age = time - p.rdata(quokka::StochasticStellarPopParticleBirthTimeIdx);
			if (age <= 0.0) {
				return;
			}

			amrex::Real const zams_mass = p.rdata(quokka::StochasticStellarPopParticleMassAtBirthIdx);
			if (zams_mass <= 0.0) {
				return;
			}

			amrex::Real const mass_coord = zams_mass * mass_to_table_units;
			amrex::Real const age_coord = age * age_to_table_units;
			std::array<amrex::Real, 2> point{};
			if (table_axes_are_mass_age) {
				point = {mass_coord, age_coord};
			} else {
				point = {age_coord, mass_coord};
			}

			amrex::Real const S = qh0_table.interpolate_single(point, 0);
			if (!(S > 0.0) || !std::isfinite(S)) {
				return;
			}

			// CIC deposit in cell-centered index space to avoid half-cell bias for sources on faces.
			amrex::Real const x_idx = ((p.pos(0) - p_lo[0]) * dxi[0]) - 0.5;
			amrex::Real const y_idx = ((p.pos(1) - p_lo[1]) * dxi[1]) - 0.5;
			amrex::Real const z_idx = ((p.pos(2) - p_lo[2]) * dxi[2]) - 0.5;

			int const i0 = static_cast<int>(amrex::Math::floor(x_idx));
			int const j0 = static_cast<int>(amrex::Math::floor(y_idx));
			int const k0 = static_cast<int>(amrex::Math::floor(z_idx));

			amrex::Real const fx = x_idx - static_cast<amrex::Real>(i0);
			amrex::Real const fy = y_idx - static_cast<amrex::Real>(j0);
			amrex::Real const fz = z_idx - static_cast<amrex::Real>(k0);

			int const nx = dom_hi.x - dom_lo.x + 1;
			int const ny = dom_hi.y - dom_lo.y + 1;
			int const nz = dom_hi.z - dom_lo.z + 1;

			for (int kk = 0; kk <= 1; ++kk) {
				int kz = k0 + kk;
				if (is_per[2] != 0) {
					while (kz < lo.z) {
						kz += nz;
					}
					while (kz > hi.z) {
						kz -= nz;
					}
				}
				amrex::Real const wz = (kk == 0) ? (1.0 - fz) : fz;

				for (int jj = 0; jj <= 1; ++jj) {
					int jy = j0 + jj;
					if (is_per[1] != 0) {
						while (jy < lo.y) {
							jy += ny;
						}
						while (jy > hi.y) {
							jy -= ny;
						}
					}
					amrex::Real const wy = (jj == 0) ? (1.0 - fy) : fy;

					for (int ii = 0; ii <= 1; ++ii) {
						int ix = i0 + ii;
						if (is_per[0] != 0) {
							while (ix < lo.x) {
								ix += nx;
							}
							while (ix > hi.x) {
								ix -= nx;
							}
						}

						if ((ix < lo.x) || (ix > hi.x) || (jy < lo.y) || (jy > hi.y) || (kz < lo.z) || (kz > hi.z)) {
							continue;
						}

						amrex::Real const wx = (ii == 0) ? (1.0 - fx) : fx;
						amrex::Real const w = wx * wy * wz;
						amrex::Gpu::Atomic::AddNoRet(&src(ix, jy, kz, 0), w * S);
					}
				}
			}
		});
	}
	source_q.SumBoundary(geom_lev.periodicity());
}

template <typename problem_t, quokka::OutOfBounds oob_policy>
[[nodiscard]] auto AdvanceStromgrenPhotonFieldAllLevels(quokka::StochasticStellarPopParticleContainer<problem_t> *stellar_particles, amrex::Real time,
							amrex::Real dt, amrex::Vector<amrex::Geometry> const &geometry,
							amrex::Vector<amrex::BoxArray> const &grids,
							amrex::Vector<amrex::DistributionMapping> const &distribution_map,
							amrex::Vector<amrex::IntVect> const &ref_ratio, amrex::Vector<amrex::MultiFab> const &hydro_state,
							amrex::Vector<amrex::MultiFab> &n_gamma, quokka::DataTableGpuConst<2, 1, oob_policy> const &qh0_table,
							int max_nonlinear_iterations, amrex::Real residual_tolerance) -> bool
{
	int const nlevels = static_cast<int>(geometry.size());
	AMREX_ALWAYS_ASSERT(nlevels > 0 && static_cast<int>(grids.size()) == nlevels && static_cast<int>(distribution_map.size()) == nlevels);
	AMREX_ALWAYS_ASSERT(static_cast<int>(hydro_state.size()) >= nlevels && static_cast<int>(n_gamma.size()) == nlevels);

	amrex::Vector<amrex::MultiFab> source;
	amrex::Vector<amrex::MultiFab> old_state;
	amrex::Vector<amrex::iMultiFab> masks(nlevels);
	source.reserve(nlevels);
	old_state.reserve(nlevels);
	amrex::Real rhs_scale = 0.0;
	amrex::Real operator_rate = 1.0 / dt;
	for (int lev = 0; lev < nlevels; ++lev) {
		source.emplace_back(grids[lev], distribution_map[lev], 1, 1);
		DepositStromgrenPhotonSourceAtLevel<problem_t, oob_policy>(stellar_particles, lev, time, grids[lev], distribution_map[lev], geometry[lev],
									   source[lev], qh0_table);
		auto const dx = geometry[lev].CellSizeArray();
		source[lev].mult(1.0 / (dx[0] * dx[1] * dx[2]), 0, 1, 0);

		old_state.emplace_back(grids[lev], distribution_map[lev], 1, 1);
		amrex::MultiFab::Copy(old_state[lev], n_gamma[lev], 0, 0, 1, 0);
		old_state[lev].FillBoundary(geometry[lev].periodicity());
		rhs_scale = amrex::max(rhs_scale, source[lev].norm0(0, 0, false));
		rhs_scale = amrex::max(rhs_scale, old_state[lev].norm0(0, 0, false) / dt);

		amrex::Real const rho_max = hydro_state[lev].max(HydroSystem<problem_t>::density_index, 0, false);
		amrex::Real const nH_max = amrex::max(rho_max / (mean_particle_mass_mu * mH), 0.0);
		auto const prob_lo = geometry[lev].ProbLoArray();
		auto const prob_hi = geometry[lev].ProbHiArray();
		amrex::Real const Lbox = std::min({prob_hi[0] - prob_lo[0], prob_hi[1] - prob_lo[1], prob_hi[2] - prob_lo[2]});
		amrex::Real const max_diffusion = C::c_light * Lbox / 3.0;
		amrex::Real const diffusion_rate = 2.0 * max_diffusion * ((1.0 / (dx[0] * dx[0])) + (1.0 / (dx[1] * dx[1])) + (1.0 / (dx[2] * dx[2])));
		operator_rate = amrex::max(operator_rate, (1.0 / dt) + diffusion_rate + C::c_light * sigma_HI * nH_max);

		if (lev + 1 < nlevels) {
			masks[lev] = amrex::makeFineMask(grids[lev], distribution_map[lev], amrex::IntVect(0), grids[lev + 1], ref_ratio[lev],
							 geometry[lev].periodicity(), 1, 0);
		} else {
			masks[lev].define(grids[lev], distribution_map[lev], 1, 0);
			masks[lev].setVal(1);
		}
	}
	amrex::ParallelDescriptor::ReduceRealMax(rhs_scale);
	amrex::ParallelDescriptor::ReduceRealMax(operator_rate);
	if (!(rhs_scale > 0.0)) {
		for (auto &level : n_gamma) {
			level.setVal(0.0);
		}
		return true;
	}

	amrex::Real const phi_scale = rhs_scale / operator_rate;
	amrex::Vector<amrex::MultiFab> dimensionless_state;
	amrex::Vector<amrex::MultiFab> rhs;
	dimensionless_state.reserve(nlevels);
	rhs.reserve(nlevels);
	for (int lev = 0; lev < nlevels; ++lev) {
		dimensionless_state.emplace_back(grids[lev], distribution_map[lev], 1, 1);
		amrex::MultiFab::Copy(dimensionless_state[lev], old_state[lev], 0, 0, 1, 0);
		dimensionless_state[lev].mult(1.0 / phi_scale, 0, 1, 0);
		dimensionless_state[lev].FillBoundary(geometry[lev].periodicity());

		rhs.emplace_back(grids[lev], distribution_map[lev], 1, 0);
		amrex::MultiFab::Copy(rhs[lev], source[lev], 0, 0, 1, 0);
		rhs[lev].mult(1.0 / phi_scale, 0, 1, 0);
		amrex::MultiFab::Saxpy(rhs[lev], 1.0 / dt, dimensionless_state[lev], 0, 0, 1, 0);
	}

	StromgrenHierarchyNewtonProblem<problem_t> problem(geometry, grids, distribution_map, ref_ratio, hydro_state, rhs, masks, phi_scale, operator_rate, dt,
							   residual_tolerance);
	int constexpr picard_predictor_steps = 1;
	for (int step = 0; step < picard_predictor_steps; ++step) {
		problem.predict(dimensionless_state, 0.7);
	}
	quokka::math::NewtonKrylovOptions options;
	options.nonlinear_tolerance = residual_tolerance;
	options.linear_tolerance = amrex::min(1.0e-2, 10.0 * residual_tolerance);
	options.maximum_nonlinear_iterations = max_nonlinear_iterations;
	options.maximum_linear_iterations = 50;
	options.krylov_restart_length = 50;
	options.maximum_line_search_iterations = 24;
	options.initial_pseudo_transient_shift = 0.0;
	options.maximum_pseudo_transient_retries = 6;
	options.minimum_acceptable_step_length = 0.25;
	options.adaptive_linear_tolerance = false;
	options.centered_difference = true;
	options.require_change_convergence = false;
	options.linear_verbosity = 0;
	options.problem_name = "The time-dependent AMR H II-region FLD system";
	quokka::math::NewtonKrylovSolver<StromgrenHierarchyNewtonProblem<problem_t>> newton(problem, options);
	auto const result = newton.solve(dimensionless_state);
	amrex::Print() << "Strömgren AMR backward-Euler solve: Picard predictor steps = " << picard_predictor_steps
		       << ", Newton iterations = " << result.nonlinear_iterations << ", Newton GMRES iterations = " << result.total_linear_iterations
		       << " (max " << result.maximum_linear_iterations << "), pseudo-transient retries = " << result.pseudo_transient_retries
		       << ", final pseudo shift = " << result.final_pseudo_transient_shift << ", final residual = " << result.final_residual_norm << '\n';
	if (!result.converged) {
		return false;
	}

	for (int lev = 0; lev < nlevels; ++lev) {
		amrex::MultiFab::Copy(n_gamma[lev], dimensionless_state[lev], 0, 0, 1, 0);
		n_gamma[lev].mult(phi_scale, 0, 1, 0);
		n_gamma[lev].FillBoundary(geometry[lev].periodicity());
	}
	return true;
}
#endif

} // namespace quokka::photoionization

#endif // PARTICLE_PHOTOIONIZATION_HPP_
