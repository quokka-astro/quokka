// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT
#ifndef ONEZONEHARNESS_HPP_
#define ONEZONEHARNESS_HPP_

/// \file OneZoneHarness.hpp
/// \brief Test harness for one-zone problems: uniform initial state, evolution with QuokkaSimulation, and comparison with a reference solution.
///
/// A one-zone test supplies three things: the initial cell (FillUniform), the observables recorded after each step (Recorder::addObservable), and the
/// checks against a reference (Check, Predicate). Run() then initializes, evolves, and evaluates the checks.
///
/// Usage in a problem file:
///   - derive SimulationData<problem_t> from quokka::onezone::Recorder;
///   - call quokka::onezone::FillUniform<problem_t>(grid_elem, cell) from setInitialConditionsOnGrid();
///   - call userData_.record(state_new_cc_[0], Geom(0), tNew_[0]) from computeAfterTimestep();
///   - in problem_main(), register observables, then return quokka::onezone::Run(sim, checks).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "AMReX_Array.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuLaunch.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "grid.hpp"
#include "math/interpolate.hpp"
#include "physics_info.hpp"
#include "util/fextract.hpp"

namespace quokka::onezone
{

/// Values of all state components in one cell.
template <typename problem_t> using Cell = amrex::GpuArray<amrex::Real, Physics_Indices<problem_t>::nvarTotal_cc>;

/// Values of all state components of the recorded cell, indexed by component.
using CellValues = std::vector<amrex::Real>;

/// A scalar derived from the recorded cell (for example a temperature).
using Observable = std::function<double(CellValues const &)>;

/// Reference values at the given (ascending) times.
using Reference = std::function<std::vector<double>(std::vector<double> const &)>;

/// Measure of the difference between a recorded series y_i and its reference r_i.
enum class Norm {
	L1Relative,   ///< sum_i |y_i - r_i| / sum_i |r_i|
	MeanRelative, ///< mean_i |y_i - r_i| / |r_i|
};

/// One recorded series compared with its reference.
struct Comparison {
	std::string observable;
	Reference reference;
};

/// A tolerance on the norm of the differences of one or more comparisons. With several comparisons the sums run over all of them.
struct Check {
	std::string name;
	std::vector<Comparison> comparisons;
	double tolerance{};
	Norm norm = Norm::L1Relative;
};

class Recorder;

/// A pass/fail condition on the recorded data. It is called on every MPI rank, so it may call collective functions such as fextract; the recorded
/// data are only valid on the I/O rank.
struct Predicate {
	std::string name;
	std::function<bool(Recorder const &)> passes;
};

/// Records observables of the centre cell as a function of time.
class Recorder
{
      public:
	/// Add a series; call before Run().
	void addObservable(std::string name, Observable function)
	{
		names_.push_back(std::move(name));
		functions_.push_back(std::move(function));
		series_.emplace_back();
	}

	/// Record all observables at the given time. Collective: call on every rank.
	void record(amrex::MultiFab &state, amrex::Geometry &geom, amrex::Real time)
	{
		auto const values = std::get<1>(fextract(state, geom, 0, 0.5));
		if (amrex::ParallelDescriptor::IOProcessor()) {
			CellValues cell;
			cell.reserve(values.size());
			for (auto const &component : values) {
				cell.push_back(component[0]);
			}
			times_.push_back(time);
			for (size_t k = 0; k < functions_.size(); ++k) {
				series_[k].push_back(functions_[k](cell));
			}
		}
	}

	[[nodiscard]] auto times() const -> std::vector<double> const & { return times_; }
	[[nodiscard]] auto names() const -> std::vector<std::string> const & { return names_; }
	[[nodiscard]] auto series(std::string const &name) const -> std::vector<double> const &
	{
		for (size_t k = 0; k < names_.size(); ++k) {
			if (names_[k] == name) {
				return series_[k];
			}
		}
		amrex::Abort("OneZoneHarness: unknown observable '" + name + "'");
		return series_.front(); // not reached
	}

      private:
	std::vector<std::string> names_;
	std::vector<Observable> functions_;
	std::vector<double> times_;
	std::vector<std::vector<double>> series_;
};

/// Set every cell of the grid to the same state.
template <typename problem_t> void FillUniform(quokka::grid const &grid_elem, Cell<problem_t> const &cell)
{
	constexpr int nvar = Physics_Indices<problem_t>::nvarTotal_cc;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state_cc.nComp() == nvar, "OneZoneHarness: the state has a different number of components than Physics_Indices");
	amrex::ParallelFor(grid_elem.indexRange_, nvar, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) { state_cc(i, j, k, n) = cell[n]; });
}

/// Reference given by a function of time.
inline auto AnalyticReference(std::function<double(double)> function) -> Reference
{
	return [function = std::move(function)](std::vector<double> const &times) {
		std::vector<double> values(times.size());
		std::transform(times.begin(), times.end(), values.begin(), function);
		return values;
	};
}

/// Reference read from a CSV file: column 0 is time (ascending), `column` is the value. Lines whose first field is not a number (the header) are
/// skipped. Values are linearly interpolated onto the recorded times, which must lie within the table.
inline auto TableReference(std::string const &filename, int column) -> Reference
{
	std::ifstream file(filename);
	if (!file.is_open()) {
		amrex::Abort("OneZoneHarness: could not open reference file " + filename + " (run the test from the <quokka-root>/tests folder)");
	}
	std::vector<double> t_ref;
	std::vector<double> y_ref;
	std::string line;
	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string field;
		std::vector<double> row;
		try {
			while (std::getline(ss, field, ',')) {
				row.push_back(std::stod(field));
			}
		} catch (const std::exception &) {
			continue; // header or malformed line
		}
		if (std::cmp_greater(row.size(), column)) {
			t_ref.push_back(row[0]);
			y_ref.push_back(row[column]);
		}
	}
	if (t_ref.size() < 3) {
		amrex::Abort("OneZoneHarness: fewer than three data rows in reference file " + filename);
	}
	return [t_ref = std::move(t_ref), y_ref = std::move(y_ref), filename](std::vector<double> const &times) {
		if (times.empty()) {
			return std::vector<double>{};
		}
		if (times.front() < t_ref.front() || times.back() > t_ref.back()) {
			amrex::Abort("OneZoneHarness: recorded times lie outside the range of reference file " + filename);
		}
		std::vector<double> x = times;
		std::vector<double> values(times.size());
		interpolate_arrays(x.data(), values.data(), static_cast<int>(x.size()), const_cast<double *>(t_ref.data()), y_ref.data(), // NOLINT
				   static_cast<int>(t_ref.size()));
		return values;
	};
}

/// Passes if every recorded value of every observable is finite.
inline auto AllFinite() -> Predicate
{
	return {.name = "all observables finite", .passes = [](Recorder const &recorder) {
			for (auto const &name : recorder.names()) {
				auto const &y = recorder.series(name);
				if (!std::all_of(y.begin(), y.end(), [](double v) { return std::isfinite(v); })) {
					return false;
				}
			}
			return true;
		}};
}

namespace detail
{
inline auto CheckError(Recorder const &recorder, Check const &check) -> double
{
	double err_sum = 0.;
	double ref_sum = 0.;
	double count = 0.;
	for (auto const &comparison : check.comparisons) {
		auto const &y = recorder.series(comparison.observable);
		auto const r = comparison.reference(recorder.times());
		for (size_t i = 0; i < y.size(); ++i) {
			if (check.norm == Norm::L1Relative) {
				err_sum += std::abs(y[i] - r[i]);
				ref_sum += std::abs(r[i]);
			} else {
				err_sum += std::abs(y[i] - r[i]) / std::abs(r[i]);
				count += 1.0;
			}
		}
	}
	return (check.norm == Norm::L1Relative) ? err_sum / ref_sum : err_sum / count;
}
} // namespace detail

/// Evaluate the checks and predicates on the recorded data, print a report, and write all series to `csv_file` (if not empty).
/// Returns 0 if everything passed, 1 otherwise.
inline auto Evaluate(Recorder const &recorder, std::vector<Check> const &checks, std::vector<Predicate> const &predicates = {},
		     std::string const &csv_file = "") -> int
{
	int status = 0;
	const bool io = amrex::ParallelDescriptor::IOProcessor();
	if (io) {
		for (auto const &check : checks) {
			const double err = detail::CheckError(recorder, check);
			const bool ok = (err <= check.tolerance); // written so that NaN fails
			amrex::Print() << std::format("{:<40}error = {:>12.4e}  tolerance = {:>10.2e}{}\n", check.name, err, check.tolerance,
						      ok ? "" : "  FAIL");
			status = ok ? status : 1;
		}
		if (!csv_file.empty()) {
			std::ofstream out(csv_file);
			out << "time";
			for (auto const &name : recorder.names()) {
				out << "," << name;
			}
			out << "\n" << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10);
			for (size_t i = 0; i < recorder.times().size(); ++i) {
				out << recorder.times()[i];
				for (auto const &name : recorder.names()) {
					out << "," << recorder.series(name)[i];
				}
				out << "\n";
			}
		}
	}
	for (auto const &predicate : predicates) {
		const bool ok = predicate.passes(recorder);
		if (io) {
			amrex::Print() << std::format("{:<40}{}\n", predicate.name, ok ? "passed" : "FAIL");
		}
		status = ok ? status : 1;
	}
	if (io) {
		amrex::Print() << (status == 0 ? "One-zone test: all checks passed.\n" : "One-zone test: FAILED.\n");
	}
	return status;
}

/// Initialize, evolve and check a one-zone simulation. sim.userData_ must derive from Recorder.
template <typename Simulation>
auto Run(Simulation &sim, std::vector<Check> const &checks, std::vector<Predicate> const &predicates = {}, std::string const &csv_file = "") -> int
{
	sim.setInitialConditions();
	sim.userData_.record(sim.state_new_cc_[0], sim.Geom(0), sim.tNew_[0]);
	sim.evolve();
	return Evaluate(sim.userData_, checks, predicates, csv_file);
}

} // namespace quokka::onezone

#endif // ONEZONEHARNESS_HPP_
