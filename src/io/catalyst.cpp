#include <string>

#include "AMReX.H"
#include "AMReX_ParmParse.H"

#ifdef AMREX_USE_CATALYST
#include "AMReX_Conduit_Blueprint.H"
#include "catalyst_api.h"
#include "conduit_cpp_to_c.hpp"
#endif

#include "Catalyst.H"

using namespace amrex;

FlushFormatCatalyst::FlushFormatCatalyst()
{
	ParmParse pp_catalyst("catalyst");
	std::string scriptPaths{""};
	std::string implementation{"paraview"};
	std::string searchPaths{""};
	pp_catalyst.query("script_paths", scriptPaths);
	pp_catalyst.query("implementation", implementation);
	pp_catalyst.query("implementation_search_paths", searchPaths);

#ifdef AMREX_USE_CATALYST
	conduit::Node node;

	// Loop over all given paths and load all the scripts. Delimiters are ';' and ':'
	size_t scriptNumber = 0;
	size_t pos = 0;
	std::string subpath;
	while (scriptPaths.find(':') != std::string::npos || scriptPaths.find(';') != std::string::npos) {
		pos = std::min(scriptPaths.find(':'), scriptPaths.find(';'));
		subpath = scriptPaths.substr(0, pos);

		node["catalyst/scripts/script" + std::to_string(scriptNumber)].set_string(subpath);

		scriptNumber++;
		scriptPaths.erase(0, pos + 1);
	}
	// Prevent empty end paths
	if (scriptPaths.length() != 0) {
		node["catalyst/scripts/script" + std::to_string(scriptNumber)].set_string(scriptPaths);
	}

	node["catalyst_load/implementation"].set_string(implementation);
	node["catalyst_load/search_paths/" + implementation].set_string(searchPaths);

	catalyst_status err = catalyst_initialize(conduit::c_node(&node));
	if (err != catalyst_status_ok) {
		std::string message = " Error: Failed to initialize Catalyst!\n";
		std::cerr << message << err << std::endl;
		amrex::Print() << message;
		amrex::Abort(message);
	}
#endif // AMREX_USE_CATALYST
}

void FlushFormatCatalyst::WriteToFile(amrex::Vector<std::string> const &varnames, amrex::Vector<amrex::MultiFab> const &mf,
				      amrex::Vector<amrex::Geometry> const &geom, amrex::Vector<amrex::IntVect> const &ref_ratio,
				      amrex::Vector<int> const &iteration, const double time, int nlev) const
{
#ifdef AMREX_USE_CATALYST
	amrex::Print() << "Running Catalyst pipeline scripts...";
	const BL_PROFILE("FlushFormatCatalyst::WriteToFile()");

	// Mesh data
	conduit::Node node;
	auto &state = node["catalyst/state"];
	state["timestep"].set(iteration[0]);
	state["time"].set(time);

	auto &meshChannel = node["catalyst/channels/mesh"];
	// meshChannel["type"].set_string("amrmesh");
	meshChannel["type"].set_string("multimesh");
	auto &meshData = meshChannel["data"];

	amrex::MultiLevelToBlueprint(nlev, amrex::GetVecOfConstPtrs(mf), varnames, geom, time, iteration, ref_ratio, meshData);

	// Execution
	catalyst_status err = catalyst_execute(conduit::c_node(&node));
	if (err != catalyst_status_ok) {
		std::string message = " Error: Failed to execute Catalyst!\n";
		std::cerr << message << err << std::endl;
		amrex::Print() << message;
	}
#else
	amrex::ignore_unused(varnames, mf, geom, iteration, time, nlev);
#endif // AMREX_USE_CATALYST
}

FlushFormatCatalyst::~FlushFormatCatalyst()
{
#ifdef AMREX_USE_CATALYST
	conduit::Node node;
	catalyst_status err = catalyst_finalize(conduit::c_node(&node));
	if (err != catalyst_status_ok) {
		std::string message = " Error: Failed to finalize Catalyst!\n";
		std::cerr << message << err << std::endl;
		amrex::Print() << message;
		amrex::Abort(message);
	} else {
		// Temporary, remove for final
		std::cout << "Successfully finalized Catalyst" << std::endl;
	}
#endif // AMREX_USE_CATALYST
}
