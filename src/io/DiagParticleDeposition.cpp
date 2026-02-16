#include "io/DiagParticleDeposition.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_Print.H"
#include "particles/PhysicsParticles.hpp"
#include "particles/particle_deposition_utils.hpp"
#include "particles/particle_types.hpp"
#include "simulation.hpp"
#include "yaml-cpp/yaml.h"
#include <fstream>
#include <iomanip>
#include <sstream>

namespace quokka
{

//==============================================================================
// DiagParticleDeposition Implementation
//==============================================================================

void DiagParticleDeposition::init(const std::string &a_prefix, std::string_view a_diagName)
{
	// Initialize base class
	DiagBase::init(a_prefix, a_diagName);

	// Read particle types to deposit
	amrex::ParmParse pp(a_prefix);
	std::string particleTypesStr = "CIC";
	pp.query("particle_types", particleTypesStr);

	// Parse particle types from string
	std::istringstream iss(particleTypesStr);
	std::string token;
	while (iss >> token) {
		m_particleTypes.push_back(token);
	}

	// Read fields to deposit
	std::string depositFieldsStr = "mass";
	pp.query("deposit_fields", depositFieldsStr);

	// Parse deposit fields from string
	std::istringstream iss2(depositFieldsStr);
	while (iss2 >> token) {
		m_depositFields.push_back(token);
	}

	// Read output format
	m_outputFormat = "plotfile";
	pp.query("output_format", m_outputFormat);

	// Calculate total number of components needed
	m_nComponents = 0;
	for (const auto &field : m_depositFields) {
		m_nComponents += getFieldComponents(field);
	}

	// Generate variable names
	m_varNames.clear();
	for (const auto &particleType : m_particleTypes) {
		for (const auto &field : m_depositFields) {
			auto fieldNames = getFieldNames(particleType, field);
			for (const auto &name : fieldNames) {
				m_varNames.push_back(name);
			}
		}
	}

	amrex::Print() << "DiagParticleDeposition initialized with " << m_particleTypes.size() << " particle types and " << m_depositFields.size()
		       << " deposit fields\n";
}

void DiagParticleDeposition::prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
				     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames)
{
	// Initialize deposition data structures
	initializeDepositionData(a_nlevels, a_geoms, a_grids, a_dmap);
}

void DiagParticleDeposition::addVars(amrex::Vector<std::string> &a_varList)
{
	// Add variable names to the list
	for (const auto &varName : m_varNames) {
		a_varList.push_back(varName);
	}
}

void DiagParticleDeposition::initializeDepositionData(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms,
						      const amrex::Vector<amrex::BoxArray> &a_grids, const amrex::Vector<amrex::DistributionMapping> &a_dmap)
{
	// Resize deposition data vector
	m_depositionData.resize(a_nlevels);

	// Initialize MultiFabs for each level
	for (int lev = 0; lev < a_nlevels; ++lev) {
		const int nGrow = 0;
		const int nComponents = static_cast<int>(m_varNames.size());
		m_depositionData[lev].define(a_grids[lev], a_dmap[lev], nComponents, nGrow);
		m_depositionData[lev].setVal(0.0);
	}
}

void DiagParticleDeposition::depositParticleProperties(int a_nstep, const amrex::Real &a_time)
{
	// TODO: Access particle containers from simulation
	// This would need to be implemented with access to the simulation object
	// For now, this is a placeholder that would need to be connected to the main simulation

	amrex::Print() << "DiagParticleDeposition::depositParticleProperties called at time " << a_time << "\n";
	amrex::Print() << "Note: Particle deposition implementation requires access to simulation particle containers\n";
}

void DiagParticleDeposition::writeOutput(int a_nstep, const amrex::Real &a_time, const YAML::Node &simulationMetadata)
{
	if (m_outputFormat == "plotfile") {
		// Write AMReX plotfile
		std::string plotFileName = m_diagfile + "_" + std::to_string(a_nstep);

		amrex::Vector<const amrex::MultiFab *> plotData(m_depositionData.size());
		for (int lev = 0; lev < static_cast<int>(m_depositionData.size()); ++lev) {
			plotData[lev] = &m_depositionData[lev];
		}

		// Get geometry information (would need to be passed from simulation)
		// For now, this is a placeholder
		amrex::Print() << "Would write plotfile: " << plotFileName << "\n";
	} else if (m_outputFormat == "ascii") {
		// Write ASCII output
		std::string asciiFileName = m_diagfile + "_" + std::to_string(a_nstep) + ".txt";
		std::ofstream outFile(asciiFileName);

		outFile << "# Particle deposition data at time " << a_time << "\n";
		outFile << "# Step: " << a_nstep << "\n";
		outFile << "# Variables: ";
		for (const auto &varName : m_varNames) {
			outFile << varName << " ";
		}
		outFile << "\n";

		// Write data (simplified - would need proper implementation)
		outFile << "# Data would be written here\n";
		outFile.close();

		amrex::Print() << "Would write ASCII file: " << asciiFileName << "\n";
	}
}

auto DiagParticleDeposition::getFieldComponents(const std::string &field) const -> int
{
	if (field == "mass" || field == "energy" || field == "number") {
		return 1;
	}
	if (field == "momentum") {
		return AMREX_SPACEDIM;
	}
	amrex::Abort("Unknown field type: " + field);
	return 0;
}

auto DiagParticleDeposition::getFieldNames(const std::string &particleType, const std::string &field) const -> std::vector<std::string>
{
	std::vector<std::string> names;
	const std::string prefix = particleType + "_" + field;

	if (field == "mass" || field == "energy" || field == "number") {
		names.push_back(prefix);
	} else if (field == "momentum") {
		names.push_back(prefix + "_x");
		names.push_back(prefix + "_y");
		names.push_back(prefix + "_z");
	}

	return names;
}

} // namespace quokka
