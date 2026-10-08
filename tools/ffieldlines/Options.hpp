#ifndef FFIELDLINES_OPTIONS_HPP_
#define FFIELDLINES_OPTIONS_HPP_
/// \file Options.hpp
/// \brief Command-line options for the ffieldlines utility.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ffieldlines
{

enum class TraceDirection { Both, Forward, Backward };

enum class OutputFormat { Vtp, LegacyVtk };

/// All settings parsed from the command line. Negative sentinels mark values
/// that are resolved later from the plotfile (e.g. the default maximum length).
struct Options {
	std::string plotfile;
	std::string output;
	OutputFormat format = OutputFormat::Vtp;
	int outputEvery = 4;
	double maxGatherBytes = 4.0 * 1024.0 * 1024.0 * 1024.0;

	// seed specification (exactly one of these is set)
	std::string seedFile;
	std::vector<double> seedLine; // x0 y0 z0 x1 y1 z1 n
	std::vector<double> seedDisk; // cx cy cz nx ny nz radius n

	std::array<std::string, 3> bfield{"x-BField", "y-BField", "z-BField"};
	std::string density = "gasDensity";
	std::array<std::string, 3> momentum{"x-GasMomentum", "y-GasMomentum", "z-GasMomentum"};
	std::string temperature = "temperature"; // empty string: do not sample

	bool periodicSet = false;
	std::array<int, 3> periodic{0, 0, 0};
	int finestLevel = -1;

	TraceDirection direction = TraceDirection::Both;
	double stepFraction = 0.25;
	double maxLength = -1.0;
	int64_t maxSteps = 1000000;
	double bMin = 0.0;
	double loopTolerance = 0.5;
	int stepsPerSweep = 64;
	int64_t maxSweeps = 100000;

	bool verbose = false;
	bool dryRun = false;
	bool showHelp = false;
};

/// Parse the command line (as returned by amrex::get_command_argument).
/// Aborts with a message on malformed input.
auto ParseOptions() -> Options;

/// Validate option values that do not depend on the plotfile.
/// Aborts with a message listing the first problem found.
void ValidateOptions(Options const &opts);

/// Print usage to amrex::Print().
void PrintUsage();

} // namespace ffieldlines

#endif // FFIELDLINES_OPTIONS_HPP_
