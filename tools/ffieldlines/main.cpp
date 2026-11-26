/// \file main.cpp
/// \brief ffieldlines: trace magnetic field lines through an AMReX plotfile.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include "Assemble.hpp"
#include "FieldData.hpp"
#include "FieldLineContainer.hpp"
#include "Options.hpp"
#include "Records.hpp"
#include "Seeds.hpp"
#include "VtkWriter.hpp"

namespace ffieldlines
{
namespace
{

auto CommandLine() -> std::string
{
	std::string line = "ffieldlines";
	for (int i = 1; i <= amrex::command_argument_count(); ++i) {
		line += " " + amrex::get_command_argument(i);
	}
	return line;
}

auto ReadTextFile(std::string const &path) -> std::string
{
	const std::ifstream in(path);
	if (!in.good()) {
		return {};
	}
	std::ostringstream text;
	text << in.rdbuf();
	return text.str();
}

void PrintDatasetSummary(FieldData const &fd, Options const &opts, size_t nseeds, double maxLength)
{
	amrex::Print() << "ffieldlines: " << opts.plotfile << "\n";
	amrex::Print() << "  time " << fd.time << ", levels 0.." << fd.finestLevel << ", periodic (" << opts.periodic[0] << ", " << opts.periodic[1] << ", "
		       << opts.periodic[2] << ")\n";
	double bytes = 0.0;
	for (int lev = 0; lev <= fd.finestLevel; ++lev) {
		const auto dx = fd.geom[lev].CellSizeArray();
		amrex::Print() << "  level " << lev << ": " << fd.grids[lev].size() << " grids, " << fd.grids[lev].numPts() << " cells, dx = (" << dx[0] << ", "
			       << dx[1] << ", " << dx[2] << ")\n";
		for (int i = 0; i < fd.grids[lev].size(); ++i) {
			bytes += static_cast<double>(amrex::grow(fd.grids[lev][i], nGhost).numPts()) * static_cast<double>(NComp) * sizeof(amrex::Real);
		}
	}
	amrex::Print() << "  field memory (all ranks) " << bytes / (1024.0 * 1024.0) << " MiB\n";
	amrex::Print() << "  " << nseeds << " seeds, max length " << maxLength << ", temperature " << (fd.hasTemperature ? opts.temperature : "(not sampled)")
		       << "\n";
}

/// Gather every rank's records on the I/O rank (empty elsewhere).
auto GatherRecords(std::vector<double> const &local, double maxBytes) -> std::vector<double>
{
	const int nprocs = amrex::ParallelDescriptor::NProcs();
	const int root = amrex::ParallelDescriptor::IOProcessorNumber();

	auto total = static_cast<amrex::Long>(local.size());
	amrex::ParallelDescriptor::ReduceLongSum(total);
	const double totalBytes = static_cast<double>(total) * sizeof(double);
	if (totalBytes > maxBytes || total > std::numeric_limits<int>::max()) {
		std::ostringstream msg;
		msg << "ffieldlines: recorded output is " << totalBytes / (1024.0 * 1024.0 * 1024.0)
		    << " GiB, above --max-gather-gb; increase --output-every or reduce --max-length / the number of seeds";
		amrex::Abort(msg.str());
	}

	const int localCount = static_cast<int>(local.size());
	std::vector<int> counts(nprocs, 0);
	amrex::ParallelDescriptor::Gather(&localCount, 1, counts.data(), 1, root);
	std::vector<int> displs(nprocs, 0);
	std::vector<double> all;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
		all.resize(static_cast<size_t>(total));
	}
	amrex::ParallelDescriptor::Gatherv(local.data(), localCount, all.data(), counts, displs, root);
	return all;
}

void Run()
{
	const Options opts = ParseOptions();
	if (opts.showHelp || amrex::command_argument_count() == 0) {
		PrintUsage();
		return;
	}
	ValidateOptions(opts);

	amrex::PlotFileData pf(opts.plotfile);
	const FieldData fd = LoadFieldData(pf, opts);

	const auto plo = pf.probLo();
	const auto phi = pf.probHi();
	const Point3 probLo{plo[0], plo[1], plo[2]};
	const Point3 probHi{phi[0], phi[1], phi[2]};
	const std::array<double, 3> domainLength{phi[0] - plo[0], phi[1] - plo[1], phi[2] - plo[2]};

	const std::vector<Point3> seeds = MakeSeeds(opts);
	ValidateSeeds(seeds, probLo, probHi);

	TraceParams params;
	params.stepFraction = opts.stepFraction;
	params.maxLength = (opts.maxLength > 0.0)
			       ? opts.maxLength
			       : 2.0 * std::sqrt(domainLength[0] * domainLength[0] + domainLength[1] * domainLength[1] + domainLength[2] * domainLength[2]);
	params.bMin = opts.bMin;
	params.loopTolerance = opts.loopTolerance;
	params.maxSteps = opts.maxSteps;
	params.stepsPerSweep = opts.stepsPerSweep;
	params.outputEvery = opts.outputEvery;
	params.periodic = opts.periodic;

	PrintDatasetSummary(fd, opts, seeds.size(), params.maxLength);
	if (opts.dryRun) {
		return;
	}

	FieldLineContainer pc(fd);
	const int64_t expected = static_cast<int64_t>(seeds.size()) * ((opts.direction == TraceDirection::Both) ? 2 : 1);
	const int64_t created = pc.AddSeeds(seeds, opts.direction);
	if (created != expected) {
		amrex::Abort("ffieldlines: created " + std::to_string(created) + " of " + std::to_string(expected) + " field-line particles");
	}

	int64_t active = created;
	int64_t sweep = 0;
	for (; sweep < opts.maxSweeps && active > 0; ++sweep) {
		pc.Sweep(params, false);
		pc.Redistribute();
		active = pc.TotalNumberOfParticles();
		if (opts.verbose) {
			amrex::Print() << "  sweep " << sweep + 1 << ": " << active << " half-lines active\n";
		}
	}
	if (active > 0) {
		amrex::Print() << "  warning: " << active << " half-lines still active after " << opts.maxSweeps
			       << " sweeps; stopping them (status max_sweeps)\n";
		pc.Sweep(params, true);
		pc.Redistribute();
	}

	const std::vector<double> records = GatherRecords(pc.Records(), opts.maxGatherBytes);
	if (!amrex::ParallelDescriptor::IOProcessor()) {
		return;
	}

	const Polylines lines = AssemblePolylines(records, opts.periodic, domainLength, fd.hasTemperature);

	VtkMetadata meta;
	meta.time = fd.time;
	meta.cycle = pf.levelStep(0);
	meta.description = CommandLine() + "\nplotfile: " + std::filesystem::absolute(opts.plotfile).string();
	const std::string yaml = ReadTextFile(opts.plotfile + "/metadata.yaml");
	if (!yaml.empty()) {
		meta.description += "\nmetadata.yaml:\n" + yaml;
	}
	if (opts.format == OutputFormat::Vtp) {
		WriteVtp(opts.output, lines, meta);
	} else {
		WriteLegacyVtk(opts.output, lines, meta);
	}

	amrex::Print() << "  " << sweep << " sweeps, " << lines.nLines << " lines, " << (lines.offsets.size() - 1) << " polyline pieces, "
		       << lines.points.size() / 3 << " points\n";
	amrex::Print() << "  final status (forward half):";
	for (int s = 1; s < NStatus; ++s) {
		if (lines.statusCounts[s] > 0) {
			amrex::Print() << " " << statusNames[s] << "=" << lines.statusCounts[s];
		}
	}
	amrex::Print() << "\n";
	if (lines.nDegenerate > 0) {
		amrex::Print() << "  " << lines.nDegenerate << " seeds produced fewer than two points (e.g. zero field at the seed)\n";
	}
	amrex::Print() << "  wrote " << opts.output << "\n";
}

} // namespace
} // namespace ffieldlines

auto main(int argc, char *argv[]) -> int
{
	amrex::Initialize(argc, argv, false);
	ffieldlines::Run();
	amrex::Finalize();
	return 0;
}
