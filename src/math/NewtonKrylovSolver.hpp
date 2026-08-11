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
	int maximum_nonlinear_iterations = 20;
	int maximum_linear_iterations = 100;
	int krylov_restart_length = 30;
	int maximum_line_search_iterations = 14;
	int linear_verbosity = 0;
	bool centered_difference = true;
	bool require_change_convergence = true;
	std::string problem_name = "nonlinear system";
};

struct NewtonKrylovResult {
	bool converged = false;
	int nonlinear_iterations = 0;
	int total_linear_iterations = 0;
	int maximum_linear_iterations = 0;
	amrex::Real final_change = std::numeric_limits<amrex::Real>::max();
	amrex::Real final_residual_norm = std::numeric_limits<amrex::Real>::max();
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
		AMREX_ALWAYS_ASSERT(options_.maximum_nonlinear_iterations > 0);
		AMREX_ALWAYS_ASSERT(options_.maximum_linear_iterations > 0);
		AMREX_ALWAYS_ASSERT(options_.krylov_restart_length > 0);
		AMREX_ALWAYS_ASSERT(options_.maximum_line_search_iterations > 0);
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

		for (int iteration = 0; iteration < options_.maximum_nonlinear_iterations; ++iteration) {
			problem_.prepare(state);
			LinearizedOperator linearized(problem_, state, residual, options_.centered_difference);
			amrex::GMRES<State, LinearizedOperator> gmres;
			gmres.define(linearized);
			gmres.setRestartLength(options_.krylov_restart_length);
			gmres.setMaxIters(options_.maximum_linear_iterations);
			gmres.setVerbose(options_.linear_verbosity);

			State correction = linearized.makeVecLHS();
			linearized.setToZero(correction);
			State linear_rhs = linearized.makeVecRHS();
			linearized.linComb(linear_rhs, RT(-1), residual, RT(0), residual);
			gmres.solve(correction, linear_rhs, options_.linear_tolerance, RT(0));
			result.total_linear_iterations += gmres.getNumIters();
			result.maximum_linear_iterations = amrex::max(result.maximum_linear_iterations, gmres.getNumIters());

			RT const residual_norm = problem_.norm2(residual);
			State candidate = problem_.clone(state);
			auto const try_line_search = [&]() {
				RT step_length = RT(1);
				for (int line_search = 0; line_search < options_.maximum_line_search_iterations; ++line_search) {
					problem_.fill_candidate(candidate, state, correction, step_length);
					if (problem_.admissible(candidate)) {
						problem_.residual(candidate, candidate_residual);
						RT const candidate_norm = problem_.norm2(candidate_residual);
						if (options_.linear_verbosity > 0) {
							amrex::Print()
							    << "Newton line search: step = " << step_length << ", residual = " << candidate_norm << '\n';
						}
						if (candidate_norm <= (RT(1) - options_.armijo_coefficient * step_length) * residual_norm) {
							result.final_residual_norm = candidate_norm;
							return true;
						}
					}
					step_length *= RT(0.5);
				}
				return false;
			};

			bool accepted = try_line_search();
			if (!accepted) {
				problem_.precondition(correction, linear_rhs);
				accepted = try_line_search();
			}
			if (!accepted) {
				result.final_residual_norm = residual_norm;
				return result;
			}

			result.final_change = problem_.relative_change(candidate, state);
			problem_.assign(state, candidate);
			problem_.assign(residual, candidate_residual);
			++result.nonlinear_iterations;
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

		LinearizedOperator(Problem &problem, State const &state, State const &residual, bool centered_difference)
		    : problem_(problem), state_(state), residual_(residual), centered_difference_(centered_difference)
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
	};

	Problem &problem_;
	NewtonKrylovOptions options_;
};

} // namespace quokka::math

#endif // NEWTON_KRYLOV_SOLVER_HPP_
