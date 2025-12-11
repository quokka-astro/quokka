#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <openPMD/openPMD.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct Options
{
	std::string infile;
	std::string outfile;
	std::string engine = "BP5";
	std::string compression = "zfp";
	double tolerance = 1.0e-6;
	int maxLevel = -1;
	int iteration = -1;
	std::vector<std::string> variables;
};

void PrintUsage()
{
	amrex::Print()
	    << "\n"
	    << "Convert an AMReX plotfile to an openPMD ADIOS2 file with lossy compression.\n"
	    << "Usage:\n"
	    << "  PlotfileToOpenPMD [--infile PLOTFILE] [--outfile OUTPUT] [--engine BP5]\n"
	    << "                    [--compress zfp|sz|none] [--tolerance 1e-6]\n"
	    << "                    [--max_level N] [--iteration I] [--var name ...]\n"
	    << "Positional arguments (if flags are omitted): infile outfile\n"
	    << std::endl;
}

auto ParseOptions() -> Options
{
	Options opts;
	int const narg = amrex::command_argument_count();
	int farg = 1;
	while (farg <= narg) {
		std::string const arg = amrex::get_command_argument(farg);
		if (arg == "-h" || arg == "--help") {
			PrintUsage();
			std::exit(EXIT_SUCCESS);
		} else if (arg == "--infile") {
			opts.infile = amrex::get_command_argument(++farg);
		} else if (arg == "--outfile") {
			opts.outfile = amrex::get_command_argument(++farg);
		} else if (arg == "--engine") {
			opts.engine = amrex::get_command_argument(++farg);
		} else if (arg == "--compress") {
			opts.compression = amrex::get_command_argument(++farg);
		} else if (arg == "--tolerance" || arg == "--tol") {
			opts.tolerance = std::stod(amrex::get_command_argument(++farg));
		} else if (arg == "--max_level") {
			opts.maxLevel = std::stoi(amrex::get_command_argument(++farg));
		} else if (arg == "--iteration") {
			opts.iteration = std::stoi(amrex::get_command_argument(++farg));
		} else if (arg == "--var" || arg == "--component") {
			opts.variables.emplace_back(amrex::get_command_argument(++farg));
		} else {
			break;
		}
		++farg;
	}

	if (opts.infile.empty() && farg <= narg) {
		opts.infile = amrex::get_command_argument(farg++);
	}
	if (opts.outfile.empty() && farg <= narg) {
		opts.outfile = amrex::get_command_argument(farg++);
	}

	if (opts.infile.empty()) {
		throw std::runtime_error("Missing input plotfile path.");
	}
	if (opts.outfile.empty()) {
		opts.outfile = opts.infile + ".bp";
	}
	return opts;
}

auto ToReversedVector(const amrex::IntVect &v) -> std::vector<std::uint64_t>
{
	std::vector<std::uint64_t> u(AMREX_SPACEDIM);
	for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		u[d] = static_cast<std::uint64_t>(v[d]);
	}
	std::reverse(u.begin(), u.end());
	return u;
}

auto ToReversedVector(const amrex::Real *v) -> std::vector<double>
{
	std::vector<double> u(AMREX_SPACEDIM);
	for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		u[d] = static_cast<double>(v[d]);
	}
	std::reverse(u.begin(), u.end());
	return u;
}

auto RelativePosition(const amrex::IntVect &nodal) -> std::vector<double>
{
	std::vector<double> pos(AMREX_SPACEDIM, 0.5);
	for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		pos[d] = nodal[d] == 0 ? 0.5 : 0.0;
	}
	std::reverse(pos.begin(), pos.end());
	return pos;
}

auto MeshName(int level, const std::string &base) -> std::string
{
	std::string name = base;
	std::replace(name.begin(), name.end(), '-', '_');
	if (level > 0) {
		name += "_L" + std::to_string(level);
	}
	return name;
}

void SetupMesh(openPMD::Mesh &mesh, const amrex::Geometry &geom, const amrex::IntVect &nodal)
{
	amrex::Box const &global_box = geom.Domain();
	auto const global_size = ToReversedVector(global_box.size());
	auto const grid_spacing = ToReversedVector(geom.CellSize());
	auto const global_offset = ToReversedVector(geom.ProbLo());

	std::vector<std::string> axes{AMREX_D_DECL("x", "y", "z")};
	std::vector<std::string> grid_axes(axes.rbegin(), axes.rend());

	mesh.setGeometry("cartesian");
	mesh.setDataOrder(openPMD::Mesh::DataOrder::C);
	mesh.setGridSpacing(grid_spacing);
	mesh.setGridGlobalOffset(global_offset);
	mesh.setAxisLabels(grid_axes);
	mesh.setAttribute("fieldSmoothing", "none");

	auto mesh_comp = mesh[openPMD::MeshRecordComponent::SCALAR];
	auto const dataset = openPMD::Dataset(openPMD::determineDatatype<amrex::Real>(), global_size);
	mesh_comp.resetDataset(dataset);
	mesh_comp.setPosition(RelativePosition(nodal));
}

auto BuildAdios2Config(const Options &opts) -> std::string
{
	std::ostringstream os;
	os << "backend = \"adios2\"\n";
	os << "adios2.engine = \"" << opts.engine << "\"\n";
	if (opts.compression != "none") {
		os << "[[adios2.dataset.operators]]\n";
		os << "type = \"" << opts.compression << "\"\n";
		if (opts.compression == "zfp") {
			os << "parameters.accuracy = \"" << opts.tolerance << "\"\n";
		} else if (opts.compression == "sz" || opts.compression == "sz3") {
			os << "parameters.errorBoundMode = \"ABS\"\n";
			os << "parameters.abs = \"" << opts.tolerance << "\"\n";
		}
	}
	return os.str();
}

void WritePlotfileToOpenPMD(const Options &opts)
{
	amrex::Print() << "Reading plotfile " << opts.infile << '\n';
	amrex::PlotFileData plotfile(opts.infile);
	int const finest_level = plotfile.finestLevel();
	int const max_level = opts.maxLevel >= 0 ? std::min(opts.maxLevel, finest_level) : finest_level;
	int const nlevels = max_level + 1;

	std::vector<std::string> components;
	components.reserve(plotfile.nComp());
	auto const &available = plotfile.varNames();
	if (opts.variables.empty()) {
		components.assign(available.begin(), available.end());
	} else {
		for (auto const &name : opts.variables) {
			auto it = std::find(available.begin(), available.end(), name);
			if (it == available.end()) {
				throw std::runtime_error("Variable '" + name + "' not found in plotfile.");
			}
			components.push_back(name);
		}
	}

	amrex::Array<int, AMREX_SPACEDIM> is_per{AMREX_D_DECL(0, 0, 0)};
	amrex::RealBox const real_box(plotfile.probLo().data(), plotfile.probHi().data());

	amrex::Vector<amrex::Geometry> geom_levels;
	geom_levels.reserve(nlevels);
	for (int lev = 0; lev < nlevels; ++lev) {
		geom_levels.emplace_back(plotfile.probDomain(lev), real_box, plotfile.coordSys(), is_per);
	}

	int const iteration_index = opts.iteration >= 0 ? opts.iteration : std::max(plotfile.levelStep(0), 0);
	std::string const config = BuildAdios2Config(opts);

	openPMD::Series series(opts.outfile, openPMD::Access::CREATE, amrex::ParallelDescriptor::Communicator(), config);
	series.setSoftware("PlotfileToOpenPMD", "0.1");
	series.setIterationEncoding(openPMD::IterationEncoding::variableBased);

	auto iteration = series.iterations[iteration_index];
	iteration.setTime(plotfile.time());
	iteration.open();

	for (int lev = 0; lev < nlevels; ++lev) {
		amrex::Print() << "Converting level " << lev << " of " << max_level << '\n';
		amrex::Geometry const &geom = geom_levels[lev];
		for (auto const &comp : components) {
			amrex::MultiFab mf = plotfile.get(lev, comp);
			auto nodal_flags = mf.ixType().toIntVect();

			auto meshes = iteration.meshes;
			std::string const mesh_name = MeshName(lev, comp);
			if (!meshes.contains(mesh_name)) {
				auto mesh = meshes[mesh_name];
				SetupMesh(mesh, geom, nodal_flags);
			}

			openPMD::MeshRecordComponent mesh = iteration.meshes[mesh_name][openPMD::MeshRecordComponent::SCALAR];
			amrex::Box const global_box = geom.Domain();

			for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
				amrex::FArrayBox const &fab = mf[mfi];
				amrex::Box const &local_box = fab.box();
				auto const chunk_offset = ToReversedVector(local_box.smallEnd() - global_box.smallEnd());
				auto const chunk_size = ToReversedVector(local_box.size());
				mesh.storeChunkRaw(fab.dataPtr(0), chunk_offset, chunk_size);
			}
		}
		series.flush();
	}

	iteration.close();
	series.close();

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Wrote openPMD output to " << opts.outfile << '\n';
	}
}
} // namespace

int main(int argc, char *argv[])
{
	amrex::Initialize(argc, argv);
	int ret = 0;
	try {
		Options opts = ParseOptions();
		WritePlotfileToOpenPMD(opts);
	} catch (std::exception const &ex) {
		amrex::Print() << "PlotfileToOpenPMD failed: " << ex.what() << '\n';
		ret = 1;
	}
	amrex::Finalize();
	return ret;
}
