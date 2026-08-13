#ifndef NEWTON_KRYLOV_SOLVER_HPP_
#define NEWTON_KRYLOV_SOLVER_HPP_

#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "AMReX_GMRES.H"

namespace quokka::math
{

struct NewtonKrylovOptions {
	amrex::Real nonlinear_tolerance = 1.0e-7;
	amrex::Real linear_tolerance = 1.0e-4;
	amrex::Real armijo_coefficient = 1.0e-4;
	amrex::Real initial_pseudo_transient_shift = 0.0;
	amrex::Real pseudo_transient_growth_factor = 10.0;
	amrex::Real pseudo_transient_decay_factor = 0.2;
	amrex::Real minimum_acceptable_step_length = 0.0;
	int maximum_nonlinear_iterations = 20;
	int maximum_linear_iterations = 100;
	int krylov_restart_length = 30;
	int maximum_line_search_iterations = 14;
	int maximum_pseudo_transient_retries = 0;
	int linear_verbosity = 0;
	bool centered_difference = true;
	bool require_change_convergence = true;
	bool adaptive_linear_tolerance = false;
	std::string problem_name = "nonlinear system";
};

struct NewtonKrylovResult {
	bool converged = false;
	int nonlinear_iterations = 0;
	int total_linear_iterations = 0;
	int maximum_linear_iterations = 0;
	int pseudo_transient_retries = 0;
	amrex::Real final_change = std::numeric_limits<amrex::Real>::max();
	amrex::Real final_residual_norm = std::numeric_limits<amrex::Real>::max();
	amrex::Real final_pseudo_transient_shift = 0.0;
};

// Jacobian-free Newton--Krylov solver with Armijo backtracking. Problem
// supplies state allocation, residual/preconditioner evaluation, vector
// algebra, admissibility, candidate construction, and a change metric.
template <typename Problem> class NewtonKrylovSolver
{
      public:
	using State = typename Problem::State;
	using RT = typename Problem::RT;

	explicit NewtonKrylovSolver(Problem &problem, NewtonKrylovOptions options = {}) : problem_(problem), options_(std::move(options))
	{
		AMREX_ALWAYS_ASSERT(options_.nonlinear_tolerance > RT(0));
		AMREX_ALWAYS_ASSERT(options_.linear_tolerance > RT(0));
		AMREX_ALWAYS_ASSERT(options_.armijo_coefficient > RT(0));
		AMREX_ALWAYS_ASSERT(options_.initial_pseudo_transient_shift >= RT(0));
		AMREX_ALWAYS_ASSERT(options_.pseudo_transient_growth_factor > RT(1));
		AMREX_ALWAYS_ASSERT(options_.pseudo_transient_decay_factor > RT(0) && options_.pseudo_transient_decay_factor < RT(1));
		AMREX_ALWAYS_ASSERT(options_.minimum_acceptable_step_length >= RT(0) && options_.minimum_acceptable_step_length <= RT(1));
		AMREX_ALWAYS_ASSERT(options_.maximum_nonlinear_iterations > 0);
		AMREX_ALWAYS_ASSERT(options_.maximum_linear_iterations > 0);
		AMREX_ALWAYS_ASSERT(options_.krylov_restart_length > 0);
		AMREX_ALWAYS_ASSERT(options_.maximum_line_search_iterations > 0);
		AMREX_ALWAYS_ASSERT(options_.maximum_pseudo_transient_retries >= 0);
	}

	[[nodiscard]] auto solve(State &state) -> NewtonKrylovResult
	{
		NewtonKrylovResult result;
		State residual = problem_.makeVecRHS();
		State candidate_residual = problem_.makeVecRHS();
		problem_.residual(state, residual);
		result.final_residual_norm = problem_.norm2(residual);
		if (result.final_residual_norm <= options_.nonlinear_tolerance) {
			result.final_change = RT(0);
			result.converged = true;
			return result;
		}

		RT pseudo_transient_shift = options_.initial_pseudo_transient_shift;
		RT forcing_term = options_.adaptive_linear_tolerance ? RT(1.0e-4) : options_.linear_tolerance;
		for (int iteration = 0; iteration < options_.maximum_nonlinear_iterations; ++iteration) {
			RT const residual_norm = problem_.norm2(residual);
			State candidate = problem_.clone(state);
			bool accepted = false;
			for (int retry = 0; retry <= options_.maximum_pseudo_transient_retries; ++retry) {
				if constexpr (requires { problem_.prepare(state, pseudo_transient_shift); }) {
					problem_.prepare(state, pseudo_transient_shift);
				} else {
					problem_.prepare(state);
				}
				LinearizedOperator linearized(problem_, state, residual, options_.centered_difference, pseudo_transient_shift);
				amrex::GMRES<State, LinearizedOperator> gmres;
				gmres.define(linearized);
				gmres.setRestartLength(options_.krylov_restart_length);
				gmres.setMaxIters(options_.maximum_linear_iterations);
				gmres.setVerbose(options_.linear_verbosity);

				State correction = linearized.makeVecLHS();
				linearized.setToZero(correction);
				State linear_rhs = linearized.makeVecRHS();
				linearized.linComb(linear_rhs, RT(-1), residual, RT(0), residual);
				RT const linear_tolerance = amrex::max(options_.linear_tolerance, forcing_term);
				gmres.solve(correction, linear_rhs, linear_tolerance, RT(0));
				result.total_linear_iterations += gmres.getNumIters();
				result.maximum_linear_iterations = amrex::max(result.maximum_linear_iterations, gmres.getNumIters());
				if constexpr (requires { problem_.constrain_correction(state, correction); }) {
					problem_.constrain_correction(state, correction);
				}

				auto const try_line_search = [&]() {
					RT step_length = RT(1);
					if constexpr (requires { problem_.maximum_step_length(state, correction); }) {
						step_length = amrex::min(step_length, problem_.maximum_step_length(state, correction));
					}
					for (int line_search = 0; line_search < options_.maximum_line_search_iterations; ++line_search) {
						problem_.fill_candidate(candidate, state, correction, step_length);
						if (problem_.admissible(candidate)) {
							problem_.residual(candidate, candidate_residual);
							RT const candidate_norm = problem_.norm2(candidate_residual);
							if (options_.linear_verbosity > 0) {
								amrex::Print() << "Newton line search: pseudo shift = " << pseudo_transient_shift
									       << ", step = " << step_length << ", residual = " << candidate_norm << '\n';
							}
							if ((step_length >= options_.minimum_acceptable_step_length) &&
							    (candidate_norm <= (RT(1) - options_.armijo_coefficient * step_length) * residual_norm)) {
								result.final_residual_norm = candidate_norm;
								return true;
							}
						}
						if (step_length <= options_.minimum_acceptable_step_length) {
							break;
						}
						step_length *= RT(0.5);
					}
					return false;
				};

				accepted = try_line_search();
				if (accepted) {
					break;
				}
				if (retry == options_.maximum_pseudo_transient_retries) {
					break;
				}
				pseudo_transient_shift =
				    (pseudo_transient_shift > RT(0)) ? pseudo_transient_shift * options_.pseudo_transient_growth_factor : RT(1.0e-3);
				++result.pseudo_transient_retries;
			}
			if (!accepted) {
				result.final_residual_norm = residual_norm;
				result.final_pseudo_transient_shift = pseudo_transient_shift;
				return result;
			}

			result.final_change = problem_.relative_change(candidate, state);
			problem_.assign(state, candidate);
			problem_.assign(residual, candidate_residual);
			++result.nonlinear_iterations;
			RT const reduction = result.final_residual_norm / residual_norm;
			if (options_.adaptive_linear_tolerance) {
				forcing_term = amrex::min(RT(1.0e-4), amrex::max(options_.linear_tolerance, RT(0.9) * reduction * reduction));
			}
			if (pseudo_transient_shift > RT(0)) {
				pseudo_transient_shift *=
				    (reduction < RT(0.75)) ? options_.pseudo_transient_decay_factor : std::sqrt(options_.pseudo_transient_decay_factor);
				if (pseudo_transient_shift < RT(1.0e-12)) {
					pseudo_transient_shift = RT(0);
				}
			}
			result.final_pseudo_transient_shift = pseudo_transient_shift;
			bool const change_converged = !options_.require_change_convergence || (result.final_change <= options_.nonlinear_tolerance);
			if (change_converged && (result.final_residual_norm <= options_.nonlinear_tolerance)) {
				result.converged = true;
				break;
			}
		}
		return result;
	}

      private:
	class LinearizedOperator
	{
	      public:
		using RT = typename Problem::RT;

		LinearizedOperator(Problem &problem, State const &state, State const &residual, bool centered_difference, RT pseudo_transient_shift)
		    : problem_(problem), state_(state), residual_(residual), centered_difference_(centered_difference),
		      pseudoTransientShift_(pseudo_transient_shift)
		{
		}

		[[nodiscard]] auto makeVecRHS() const -> State { return problem_.makeVecRHS(); }
		[[nodiscard]] auto makeVecLHS() const -> State { return problem_.makeVecLHS(); }

		void apply(State &output, State const &direction)
		{
			RT const direction_norm = norm2(direction);
			if (direction_norm == RT(0)) {
				setToZero(output);
				return;
			}
			if constexpr (requires { problem_.applyJacobian(state_, direction, output); }) {
				problem_.applyJacobian(state_, direction, output);
				increment(output, direction, pseudoTransientShift_);
				return;
			}
			RT const epsilon = centered_difference_ ? std::cbrt(std::numeric_limits<RT>::epsilon()) : std::sqrt(std::numeric_limits<RT>::epsilon());
			RT const step = epsilon * (RT(1) + norm2(state_)) / direction_norm;
			State trial_plus = problem_.clone(state_);
			increment(trial_plus, direction, step);
			problem_.linearized_residual(trial_plus, output);
			if (centered_difference_) {
				State trial_minus = problem_.clone(state_);
				increment(trial_minus, direction, -step);
				State residual_minus = makeVecRHS();
				problem_.linearized_residual(trial_minus, residual_minus);
				increment(output, residual_minus, RT(-1));
				scale(output, RT(0.5) / step);
			} else {
				increment(output, residual_, RT(-1));
				scale(output, RT(1) / step);
			}
			increment(output, direction, pseudoTransientShift_);
		}

		void precond(State &output, State const &rhs) { problem_.precondition(output, rhs); }
		void assign(State &lhs, State const &rhs) const { problem_.assign(lhs, rhs); }
		[[nodiscard]] auto dotProduct(State const &lhs, State const &rhs) const -> RT { return problem_.dotProduct(lhs, rhs); }
		void increment(State &lhs, State const &rhs, RT scale_factor) const { problem_.increment(lhs, rhs, scale_factor); }
		void linComb(State &lhs, RT a, State const &rhs_a, RT b, State const &rhs_b) const { problem_.linComb(lhs, a, rhs_a, b, rhs_b); }
		[[nodiscard]] auto norm2(State const &vector) const -> RT { return problem_.norm2(vector); }
		void scale(State &vector, RT factor) const { problem_.scale(vector, factor); }
		void setToZero(State &vector) const { problem_.setToZero(vector); }

	      private:
		Problem &problem_;
		State const &state_;
		State const &residual_;
		bool centered_difference_;
		RT pseudoTransientShift_;
	};

	Problem &problem_;
	NewtonKrylovOptions options_;
};

} // namespace quokka::math

#endif // NEWTON_KRYLOV_SOLVER_HPP_
