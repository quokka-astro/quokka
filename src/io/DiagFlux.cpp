#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ios>
#include <utility>
#include <vector>

#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "fundamental_constants.H"

#include "DiagFlux.H"

void DiagFlux::init(const std::string &a_prefix, std::string_view a_diagName)
{
	DiagBase::init(a_prefix, a_diagName);

	amrex::ParmParse const pp(a_prefix);
	const int n_radii = pp.countval("radii");
	const int n_radii_kpc = pp.countval("radii_kpc");

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE((n_radii > 0) != (n_radii_kpc > 0), "DiagFlux requires exactly one of 'radii' or 'radii_kpc' to be specified.");

	if (n_radii > 0) {
		pp.getarr("radii", m_radii, 0, n_radii);
		m_outputRadii = m_radii;
		m_radiusLabel = "radius";
	} else {
		pp.getarr("radii_kpc", m_outputRadii, 0, n_radii_kpc);
		m_radii.resize(m_outputRadii.size());
		for (int n = 0; n < m_outputRadii.size(); ++n) {
			m_radii[n] = m_outputRadii[n] * (1.0e3 * C::parsec);
		}
		m_radiusLabel = "radius_kpc";
	}

	for (int n = 0; n < m_radii.size(); ++n) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_radii[n] > 0.0, "DiagFlux radii must be strictly positive.");
	}

	std::vector<std::pair<amrex::Real, amrex::Real>> radii_pairs;
	radii_pairs.reserve(m_radii.size());
	for (int n = 0; n < m_radii.size(); ++n) {
		radii_pairs.emplace_back(m_radii[n], m_outputRadii[n]);
	}
	std::ranges::sort(radii_pairs, [](auto const &lhs, auto const &rhs) { return lhs.first < rhs.first; });
	for (int n = 0; n < radii_pairs.size(); ++n) {
		m_radii[n] = radii_pairs[n].first;
		m_outputRadii[n] = radii_pairs[n].second;
	}
}

void DiagFlux::prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
		       const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames)
{
	if (first_time) {
		DiagBase::prepare(a_nlevels, a_geoms, a_grids, a_dmap, a_varNames);
		first_time = false;
	}
}

void DiagFlux::addVars(amrex::Vector<std::string> &a_varList) { DiagBase::addVars(a_varList); }

void DiagFlux::writeFluxToFile(int a_nstep, const amrex::Real &a_time,
			       std::vector<std::pair<amrex::Real, quokka::diagnostics::SurfaceFluxes>> const &fluxes_by_radius) const
{
	std::string diagfile;
	if (m_per > 0.0) {
		diagfile = m_diagfile + std::to_string(a_time);
	} else {
		diagfile = amrex::Concatenate(m_diagfile, a_nstep, 6);
	}
	diagfile += ".dat";

	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::ofstream fluxFile;
		fluxFile.open(diagfile.c_str(), std::ios::out);

		const int prec = 17;
		const int width = 25;
		const int radius_width = std::max(width, static_cast<int>(m_radiusLabel.size()) + 5);

		fluxFile << "# " << std::setw(width) << "time:"
			 << " " << std::setw(width) << std::setprecision(prec) << std::scientific << a_time << "\n";
		fluxFile << "# " << std::setw(width) << "cycle:"
			 << " " << std::setw(width) << a_nstep << "\n";
		fluxFile << "# " << std::setw(width) << "variables:"
			 << " " << std::setw(radius_width) << m_radiusLabel << " " << std::setw(width) << "mass_flux"
			 << " " << std::setw(width) << "hydro_energy_flux" << " " << std::setw(width) << "mhd_energy_flux"
			 << " " << std::setw(width) << "passive_scalar_flux" << "\n";

		fluxFile << std::setw(width) << "radius_idx" << " " << std::setw(radius_width) << m_radiusLabel << " " << std::setw(width) << "mass_flux"
			 << " " << std::setw(width) << "hydro_energy_flux" << " " << std::setw(width) << "mhd_energy_flux" << " " << std::setw(width)
			 << "passive_scalar_flux" << "\n";

		for (int n = 0; n < fluxes_by_radius.size(); ++n) {
			auto const &[radius, fluxes] = fluxes_by_radius[n];
			fluxFile << std::setw(width) << n << " " << std::setw(radius_width) << std::setprecision(prec) << std::scientific << radius << " "
				 << std::setw(width) << std::setprecision(prec) << std::scientific << fluxes.mass_flux << " " << std::setw(width)
				 << std::setprecision(prec) << std::scientific << fluxes.hydro_energy_flux << " " << std::setw(width) << std::setprecision(prec)
				 << std::scientific << fluxes.mhd_energy_flux << " " << std::setw(width) << std::setprecision(prec) << std::scientific
				 << fluxes.passive_scalar_flux << "\n";
		}

		fluxFile.flush();
		fluxFile.close();
	}
}
