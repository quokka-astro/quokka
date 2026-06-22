#include <algorithm>
#include <filesystem>
#include <string>

#include "AMReX.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_Print.H"
#include "AMReX_VisMF.H"

#include "cooling/ResampledCooling.hpp"
#include "fundamental_constants.H"

namespace
{

constexpr double keV_in_ergs = 1000.0 * C::ev2erg; // ergs == 1 keV

struct ComponentIndices {
	int density = -1;
	int xmom = -1;
	int ymom = -1;
	int zmom = -1;
	int energy = -1;
	int bx = -1;
	int by = -1;
	int bz = -1;
	int temperature = -1;
	int entropy = -1;
};

struct CommandLineOptions {
	std::string plotfile;
	std::string cooling_table;
	bool keep_backup = true;
	bool show_help = false;
	bool tiny_profiler_enabled = false;
	bool tiny_profiler_memprof_enabled = false;
};

auto parseBool(std::string const &value) -> bool
{
	if (value == "1" || value == "t" || value == "true" || value == "T" || value == "True" || value == "TRUE") {
		return true;
	}
	if (value == "0" || value == "f" || value == "false" || value == "F" || value == "False" || value == "FALSE") {
		return false;
	}
	amrex::Abort("Invalid boolean value '" + value + "'.");
	return false;
}

auto parseCommandLine(int argc, char *argv[]) -> CommandLineOptions
{
	CommandLineOptions options;
	for (int n = 1; n < argc; ++n) {
		std::string const arg = argv[n];
		if (arg == "-h" || arg == "--help") {
			options.show_help = true;
			continue;
		}
		auto const equals = arg.find('=');
		if (equals == std::string::npos) {
			if (options.plotfile.empty()) {
				options.plotfile = arg;
				continue;
			}
			amrex::Abort("Unexpected positional argument '" + arg + "'.");
		}
		std::string const key = arg.substr(0, equals);
		std::string const value = arg.substr(equals + 1);
		if (key == "plotfile") {
			options.plotfile = value;
		} else if (key == "cooling.hdf5_data_file") {
			options.cooling_table = value;
		} else if (key == "keep_backup") {
			options.keep_backup = parseBool(value);
		} else if (key == "tiny_profiler.enabled") {
			options.tiny_profiler_enabled = parseBool(value);
		} else if (key == "tiny_profiler.memprof_enabled") {
			options.tiny_profiler_memprof_enabled = parseBool(value);
		}
	}
	return options;
}

void setTinyProfilerDefaults(CommandLineOptions const &options)
{
	amrex::ParmParse pp("tiny_profiler");
	pp.add("enabled", options.tiny_profiler_enabled);
	pp.add("memprof_enabled", options.tiny_profiler_memprof_enabled);
}

auto findComponent(amrex::Vector<std::string> const &names, std::string const &name) -> int
{
	auto const it = std::find(names.cbegin(), names.cend(), name);
	return (it == names.cend()) ? -1 : static_cast<int>(std::distance(names.cbegin(), it));
}

auto findComponentIndices(amrex::Vector<std::string> const &names) -> ComponentIndices
{
	ComponentIndices indices;
	indices.density = findComponent(names, "gasDensity");
	indices.xmom = findComponent(names, "x-GasMomentum");
	indices.ymom = findComponent(names, "y-GasMomentum");
	indices.zmom = findComponent(names, "z-GasMomentum");
	indices.energy = findComponent(names, "gasEnergy");
	indices.bx = findComponent(names, "x-BField");
	indices.by = findComponent(names, "y-BField");
	indices.bz = findComponent(names, "z-BField");
	indices.temperature = findComponent(names, "temperature");
	indices.entropy = findComponent(names, "entropy");
	return indices;
}

void requireComponent(int index, std::string const &name)
{
	if (index < 0) {
		amrex::Abort("Input plotfile is missing required component '" + name + "'.");
	}
}

void validateComponents(ComponentIndices const &indices)
{
	requireComponent(indices.density, "gasDensity");
	requireComponent(indices.xmom, "x-GasMomentum");
	requireComponent(indices.ymom, "y-GasMomentum");
	requireComponent(indices.zmom, "z-GasMomentum");
	requireComponent(indices.energy, "gasEnergy");
	requireComponent(indices.bx, "x-BField");
	requireComponent(indices.by, "y-BField");
	requireComponent(indices.bz, "z-BField");
	requireComponent(indices.temperature, "temperature");
}

auto makeGeometry(amrex::PlotFileData const &plotfile, int lev) -> amrex::Geometry
{
	amrex::Array<int, AMREX_SPACEDIM> const is_periodic{AMREX_D_DECL(0, 0, 0)};
	amrex::RealBox const real_box(plotfile.probLo(), plotfile.probHi());
	return {plotfile.probDomain(lev), real_box, plotfile.coordSys(), is_periodic};
}

auto makeUniquePath(std::filesystem::path const &base) -> std::filesystem::path
{
	if (!std::filesystem::exists(base)) {
		return base;
	}
	for (int i = 1; i < 1000; ++i) {
		std::filesystem::path candidate = base;
		candidate += "." + std::to_string(i);
		if (!std::filesystem::exists(candidate)) {
			return candidate;
		}
	}
	amrex::Abort("Could not find an unused path based on '" + base.string() + "'.");
	return base;
}

auto normalizePlotfilePath(std::filesystem::path path) -> std::filesystem::path
{
	path = path.lexically_normal();
	while (!path.empty() && path != path.root_path() && path.filename().empty()) {
		path = path.parent_path();
	}
	if (path.filename().empty()) {
		amrex::Abort("Plotfile path '" + path.string() + "' does not name a plotfile directory.");
	}
	return path;
}

void preserveSidecarEntries(std::filesystem::path const &backup, std::filesystem::path const &rewritten)
{
	if (!amrex::ParallelDescriptor::IOProcessor()) {
		return;
	}

	for (auto const &entry : std::filesystem::directory_iterator(backup)) {
		auto const name = entry.path().filename().string();
		if (name == "Header" || name.rfind("Level_", 0) == 0) {
			continue;
		}
		auto const destination = rewritten / entry.path().filename();
		if (std::filesystem::exists(destination)) {
			continue;
		}
		std::filesystem::copy(entry.path(), destination, std::filesystem::copy_options::skip_existing | std::filesystem::copy_options::recursive);
	}
}

void overwritePlotfile(std::filesystem::path const &plotfile_path, bool const keep_backup, std::string const &cooling_table)
{
	std::filesystem::path const normalized_plotfile_path = normalizePlotfilePath(plotfile_path);
	amrex::PlotFileData plotfile(normalized_plotfile_path.string());
	auto const &varnames = plotfile.varNames();
	ComponentIndices const comps = findComponentIndices(varnames);
	validateComponents(comps);

	quokka::ResampledCooling::resampled_tables resampled_tables;
	quokka::ResampledCooling::readResampledData(cooling_table, resampled_tables);
	auto const tables = resampled_tables.const_tables();

	int const nlevs = plotfile.finestLevel() + 1;
	int const ncomp = static_cast<int>(varnames.size());
	bool const has_entropy = (comps.entropy >= 0);
	amrex::Vector<amrex::MultiFab> output(nlevs);
	amrex::Vector<amrex::Geometry> geoms;
	amrex::Vector<int> level_steps;
	amrex::Vector<amrex::IntVect> ref_ratio;
	geoms.reserve(nlevs);
	level_steps.reserve(nlevs);
	ref_ratio.reserve(std::max(0, nlevs - 1));

	for (int lev = 0; lev < nlevs; ++lev) {
		output[lev].define(plotfile.boxArray(lev), plotfile.DistributionMap(lev), ncomp, 0);
		for (int n = 0; n < ncomp; ++n) {
			amrex::MultiFab mf = plotfile.get(lev, varnames[n]);
			amrex::MultiFab::Copy(output[lev], mf, 0, n, 1, 0);
		}

		for (amrex::MFIter iter(output[lev], amrex::TilingIfNotGPU()); iter.isValid(); ++iter) {
			amrex::Box const &box = iter.tilebox();
			auto const &state = output[lev].array(iter);
			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				amrex::Real const rho = state(i, j, k, comps.density);
				amrex::Real const xmom = state(i, j, k, comps.xmom);
				amrex::Real const ymom = state(i, j, k, comps.ymom);
				amrex::Real const zmom = state(i, j, k, comps.zmom);
				amrex::Real const egas = state(i, j, k, comps.energy);
				amrex::Real const bx = state(i, j, k, comps.bx);
				amrex::Real const by = state(i, j, k, comps.by);
				amrex::Real const bz = state(i, j, k, comps.bz);
				amrex::Real const kinetic_energy = 0.5 * ((xmom * xmom) + (ymom * ymom) + (zmom * zmom)) / rho;
				amrex::Real const magnetic_energy = 0.5 * ((bx * bx) + (by * by) + (bz * bz));
				amrex::Real const internal_energy = egas - kinetic_energy - magnetic_energy;
				state(i, j, k, comps.temperature) = quokka::ResampledCooling::ComputeTgasFromEgas(rho, internal_energy, tables);
				if (has_entropy) {
					amrex::Real const K_cgs = quokka::ResampledCooling::ComputeEntropyFromRhoEint(rho, internal_energy, tables);
					state(i, j, k, comps.entropy) = K_cgs / keV_in_ergs;
				}
			});
		}

		geoms.push_back(makeGeometry(plotfile, lev));
		level_steps.push_back(plotfile.levelStep(lev));
		if (lev < plotfile.finestLevel()) {
			ref_ratio.push_back(amrex::IntVect(plotfile.refRatio(lev)));
		}
	}
	amrex::Gpu::streamSynchronize();

	std::filesystem::path const parent =
	    normalized_plotfile_path.parent_path().empty() ? std::filesystem::path(".") : normalized_plotfile_path.parent_path();
	std::filesystem::path const tmp_path = makeUniquePath(parent / (normalized_plotfile_path.filename().string() + ".tmp-rewrite"));
	std::filesystem::path const backup_path = makeUniquePath(parent / (normalized_plotfile_path.filename().string() + ".bak"));

	amrex::WriteMultiLevelPlotfile(tmp_path.string(), nlevs, amrex::GetVecOfConstPtrs(output), varnames, geoms, plotfile.time(), level_steps, ref_ratio);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::filesystem::rename(normalized_plotfile_path, backup_path);
		std::filesystem::rename(tmp_path, normalized_plotfile_path);
	}
	amrex::ParallelDescriptor::Barrier();

	preserveSidecarEntries(backup_path, normalized_plotfile_path);
	amrex::ParallelDescriptor::Barrier();

	if (!keep_backup && amrex::ParallelDescriptor::IOProcessor()) {
		std::filesystem::remove_all(backup_path);
	}
	amrex::ParallelDescriptor::Barrier();

	amrex::Print() << "Overwrote temperature" << (has_entropy ? " and entropy" : "") << " in " << normalized_plotfile_path.string() << "\n";
	if (keep_backup) {
		amrex::Print() << "Original plotfile saved as " << backup_path.string() << "\n";
	}
}

void printUsage(char const *program)
{
	amrex::Print() << "Rewrites temperature (and entropy, if present) in a plotfile using the MHD-corrected\n"
		       << "internal energy (total energy minus kinetic and magnetic energy).\n"
		       << "Usage:\n"
		       << "  " << program << " plotfile=<plt> cooling.hdf5_data_file=<table.h5> [keep_backup=1] [tiny_profiler.enabled=0]\n"
		       << "  " << program << " <plt> cooling.hdf5_data_file=<table.h5> [keep_backup=1] [tiny_profiler.enabled=0]\n";
}

void mainMain(char const *program, CommandLineOptions const &options)
{
	if (options.show_help || options.plotfile.empty()) {
		printUsage(program);
		return;
	}

	if (options.cooling_table.empty()) {
		amrex::Abort("Must specify cooling.hdf5_data_file=<table.h5>.");
	}

	overwritePlotfile(options.plotfile, options.keep_backup, options.cooling_table);
}

} // namespace

auto main(int argc, char *argv[]) -> int
{
	CommandLineOptions const options = parseCommandLine(argc, argv);
	amrex::SetVerbose(0);
	amrex::Initialize(argc, argv, false, MPI_COMM_WORLD, [&options]() { setTinyProfilerDefaults(options); });
	mainMain(argv[0], options);
	amrex::Finalize();
	return 0;
}
