/// \file Options.cpp
/// \brief Command-line parsing for the ffieldlines utility.

#include "Options.hpp"

#include <cmath>
#include <limits>
#include <string>

#include <AMReX.H>
#include <AMReX_Print.H>

namespace ffieldlines
{
namespace
{

class ArgReader
{
      public:
	ArgReader() : narg_(amrex::command_argument_count()) {}

	[[nodiscard]] auto Done() const -> bool { return pos_ > narg_; }
	[[nodiscard]] auto Peek() const -> std::string { return amrex::get_command_argument(pos_); }
	void Advance() { ++pos_; }

	/// Consume the value following option `name`.
	auto Next(std::string const &name) -> std::string
	{
		++pos_;
		if (pos_ > narg_) {
			amrex::Abort("ffieldlines: option " + name + " requires a value");
		}
		return amrex::get_command_argument(pos_);
	}

	auto NextDouble(std::string const &name) -> double
	{
		const std::string value = Next(name);
		try {
			size_t used = 0;
			const double result = std::stod(value, &used);
			if (used != value.size()) {
				throw std::invalid_argument(value);
			}
			return result;
		} catch (std::exception const &) {
			amrex::Abort("ffieldlines: option " + name + " expects a number, got '" + value + "'");
		}
		return 0.0;
	}

	auto NextInt(std::string const &name) -> int64_t
	{
		const double value = NextDouble(name);
		if (value != std::floor(value)) {
			amrex::Abort("ffieldlines: option " + name + " expects an integer");
		}
		return static_cast<int64_t>(value);
	}

      private:
	int narg_;
	int pos_ = 1;
};

} // namespace

void PrintUsage()
{
	amrex::Print() << "\n"
		       << " Trace magnetic field lines through an AMReX plotfile and write VTK PolyData.\n"
		       << "\n"
		       << " Usage:\n"
		       << "    ffieldlines --periodic PX PY PZ <seed option> [options] plotfile\n"
		       << "\n"
		       << " seeds (exactly one):\n"
		       << "   -s, --seeds FILE                      text file with one 'x y z' per line ('#' comments)\n"
		       << "       --seed-line X0 Y0 Z0 X1 Y1 Z1 N   N points from (X0,Y0,Z0) to (X1,Y1,Z1)\n"
		       << "       --seed-disk CX CY CZ NX NY NZ R N N points on a disk (center, normal, radius)\n"
		       << "\n"
		       << " output:\n"
		       << "   -o, --output FILE         default: <plotfile>.fieldlines.vtp (.vtk for --format vtk)\n"
		       << "       --format vtp|vtk      VTK XML PolyData (default) or legacy binary VTK\n"
		       << "       --output-every N      record every Nth integration step (default 4)\n"
		       << "       --max-gather-gb G     abort if gathered output exceeds G GiB (default 4)\n"
		       << "\n"
		       << " fields:\n"
		       << "       --bfield BX BY BZ     default: x-BField y-BField z-BField\n"
		       << "       --density NAME        default: gasDensity\n"
		       << "       --momentum MX MY MZ   default: x-GasMomentum y-GasMomentum z-GasMomentum\n"
		       << "       --temperature NAME    default: temperature ('none' to skip)\n"
		       << "\n"
		       << " domain:\n"
		       << "       --periodic PX PY PZ   0/1 per axis; REQUIRED (plotfiles do not record periodicity)\n"
		       << "       --finest-level L      ignore levels finer than L\n"
		       << "\n"
		       << " integration:\n"
		       << "       --direction both|forward|backward   default: both\n"
		       << "       --step-fraction F     RK4 step h = F * dx of the current level, 0 < F <= 0.5 (default 0.25)\n"
		       << "       --max-length L        maximum arc length per half-line (default: 2 x domain diagonal)\n"
		       << "       --max-steps N         maximum steps per half-line (default 1000000)\n"
		       << "       --b-min B             stop where |B| <= B (default 0)\n"
		       << "       --loop-tolerance T    closed-loop radius in units of the local dx (default 0.5)\n"
		       << "       --steps-per-sweep K   steps between particle redistributions (default 64)\n"
		       << "       --max-sweeps N        safety cap on redistributions (default 100000)\n"
		       << "\n"
		       << "   -n, --dry-run             read the plotfile and seeds, report, and exit\n"
		       << "   -v, --verbose\n"
		       << "   -h, --help\n"
		       << "\n";
}

auto ParseOptions() -> Options
{
	Options opts;
	ArgReader args;

	while (!args.Done()) {
		const std::string name = args.Peek();
		if (name == "-h" || name == "--help") {
			opts.showHelp = true;
		} else if (name == "-s" || name == "--seeds") {
			opts.seedFile = args.Next(name);
		} else if (name == "--seed-line") {
			opts.seedLine.clear();
			for (int i = 0; i < 7; ++i) {
				opts.seedLine.push_back(args.NextDouble(name));
			}
		} else if (name == "--seed-disk") {
			opts.seedDisk.clear();
			for (int i = 0; i < 8; ++i) {
				opts.seedDisk.push_back(args.NextDouble(name));
			}
		} else if (name == "-o" || name == "--output") {
			opts.output = args.Next(name);
		} else if (name == "--format") {
			const std::string value = args.Next(name);
			if (value == "vtp") {
				opts.format = OutputFormat::Vtp;
			} else if (value == "vtk") {
				opts.format = OutputFormat::LegacyVtk;
			} else {
				amrex::Abort("ffieldlines: --format must be 'vtp' or 'vtk'");
			}
		} else if (name == "--output-every") {
			opts.outputEvery = static_cast<int>(args.NextInt(name));
		} else if (name == "--max-gather-gb") {
			opts.maxGatherBytes = args.NextDouble(name) * 1024.0 * 1024.0 * 1024.0;
		} else if (name == "--bfield") {
			for (auto &field : opts.bfield) {
				field = args.Next(name);
			}
		} else if (name == "--density") {
			opts.density = args.Next(name);
		} else if (name == "--momentum") {
			for (auto &field : opts.momentum) {
				field = args.Next(name);
			}
		} else if (name == "--temperature") {
			const std::string value = args.Next(name);
			opts.temperature = (value == "none") ? std::string{} : value;
		} else if (name == "--periodic") {
			for (auto &flag : opts.periodic) {
				const int64_t value = args.NextInt(name);
				if (value != 0 && value != 1) {
					amrex::Abort("ffieldlines: --periodic expects three values of 0 or 1");
				}
				flag = static_cast<int>(value);
			}
			opts.periodicSet = true;
		} else if (name == "--finest-level") {
			opts.finestLevel = static_cast<int>(args.NextInt(name));
		} else if (name == "--direction") {
			const std::string value = args.Next(name);
			if (value == "both") {
				opts.direction = TraceDirection::Both;
			} else if (value == "forward") {
				opts.direction = TraceDirection::Forward;
			} else if (value == "backward") {
				opts.direction = TraceDirection::Backward;
			} else {
				amrex::Abort("ffieldlines: --direction must be 'both', 'forward', or 'backward'");
			}
		} else if (name == "--step-fraction") {
			opts.stepFraction = args.NextDouble(name);
		} else if (name == "--max-length") {
			opts.maxLength = args.NextDouble(name);
		} else if (name == "--max-steps") {
			opts.maxSteps = args.NextInt(name);
		} else if (name == "--b-min") {
			opts.bMin = args.NextDouble(name);
		} else if (name == "--loop-tolerance") {
			opts.loopTolerance = args.NextDouble(name);
		} else if (name == "--steps-per-sweep") {
			opts.stepsPerSweep = static_cast<int>(args.NextInt(name));
		} else if (name == "--max-sweeps") {
			opts.maxSweeps = args.NextInt(name);
		} else if (name == "-n" || name == "--dry-run") {
			opts.dryRun = true;
		} else if (name == "-v" || name == "--verbose") {
			opts.verbose = true;
		} else if (!name.empty() && name[0] == '-') {
			amrex::Abort("ffieldlines: unknown option '" + name + "' (see --help)");
		} else {
			if (!opts.plotfile.empty()) {
				amrex::Abort("ffieldlines: more than one plotfile given ('" + opts.plotfile + "' and '" + name + "')");
			}
			opts.plotfile = name;
		}
		args.Advance();
	}

	if (!opts.plotfile.empty() && opts.plotfile.back() == '/') {
		opts.plotfile.pop_back();
	}
	if (opts.output.empty() && !opts.plotfile.empty()) {
		opts.output = opts.plotfile + ((opts.format == OutputFormat::Vtp) ? ".fieldlines.vtp" : ".fieldlines.vtk");
	}
	return opts;
}

void ValidateOptions(Options const &opts)
{
	auto fail = [](std::string const &msg) { amrex::Abort("ffieldlines: " + msg); };

	if (opts.plotfile.empty()) {
		fail("no plotfile given (see --help)");
	}
	if (!opts.periodicSet) {
		fail("--periodic PX PY PZ is required: plotfiles do not record domain periodicity");
	}
	const int nSeedOptions = static_cast<int>(!opts.seedFile.empty()) + static_cast<int>(!opts.seedLine.empty()) + static_cast<int>(!opts.seedDisk.empty());
	if (nSeedOptions != 1) {
		fail("exactly one of --seeds, --seed-line, --seed-disk is required");
	}
	if (!(opts.stepFraction > 0.0 && opts.stepFraction <= 0.5)) {
		fail("--step-fraction must satisfy 0 < F <= 0.5");
	}
	if (opts.maxLength != -1.0 && !(opts.maxLength > 0.0 && std::isfinite(opts.maxLength))) {
		fail("--max-length must be positive and finite");
	}
	if (opts.maxSteps < 1 || opts.maxSteps > std::numeric_limits<int>::max()) {
		fail("--max-steps must be in [1, 2^31 - 1]");
	}
	if (!(opts.bMin >= 0.0 && std::isfinite(opts.bMin))) {
		fail("--b-min must be non-negative and finite");
	}
	if (!(opts.loopTolerance >= 0.0 && std::isfinite(opts.loopTolerance))) {
		fail("--loop-tolerance must be non-negative and finite");
	}
	if (opts.stepsPerSweep < 1) {
		fail("--steps-per-sweep must be at least 1");
	}
	if (opts.maxSweeps < 1) {
		fail("--max-sweeps must be at least 1");
	}
	if (opts.outputEvery < 1) {
		fail("--output-every must be at least 1");
	}
	if (!(opts.maxGatherBytes > 0.0)) {
		fail("--max-gather-gb must be positive");
	}
}

} // namespace ffieldlines
