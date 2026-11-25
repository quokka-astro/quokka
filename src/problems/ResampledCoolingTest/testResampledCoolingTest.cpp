// ABOUTME: Test problem for cooling integrator accuracy using isochoric cooling
// ABOUTME: Uses the ResampledCooling module via runtime parameter
// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testResampledCoolingTest.cpp
/// \brief Defines a test problem for cooling integrator accuracy using the ResampledCooling module.
///

#include "QuokkaSimulation.hpp"
#include "cooling/ResampledCooling.hpp"
#include "hydro/hydro_system.hpp"
#include "util/OneZoneHarness.hpp"
#include "util/fextract.hpp"

#include "AMReX_ParmParse.H"

#include <string>
#include <vector>

struct ResampledCoolingTest {}; // dummy type to allow compile-time polymorphism via template specialization

template <> struct SimulationData<ResampledCoolingTest> : quokka::onezone::Recorder {};

template <> struct quokka::EOS_Traits<ResampledCoolingTest> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	using EOSBackend = quokka::EOSTabulated<ResampledCoolingTest>;
};

template <> struct Physics_Traits<ResampledCoolingTest> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = (AMREX_SPACEDIM == 3);
};

// Initial conditions: hot gas that will cool down
constexpr double T_initial = 1.0e7;	// K
constexpr double rho_initial = 1.0e-26; // g cm^-3 (constant density for isochoric)
constexpr double mu_initial = 0.6 * quokka::EOS_Traits<ResampledCoolingTest>::mean_molecular_weight;
constexpr double pressure_initial = rho_initial * C::k_B * T_initial / mu_initial;
constexpr double Bx_initial = 5.264491941623788e-06; // chosen so plasma beta = P_gas / (B^2 / 2) ~= 1
constexpr double magnetic_energy_initial = 0.5 * Bx_initial * Bx_initial;
constexpr double active_magnetic_energy_initial = Physics_Traits<ResampledCoolingTest>::is_mhd_enabled ? magnetic_energy_initial : 0.0;

auto computeInitialInternalEnergy() -> double
{
	constexpr double gamma = quokka::EOS_Traits<ResampledCoolingTest>::gamma;
	return rho_initial * C::k_B * T_initial / ((gamma - 1.0) * mu_initial);
}

template <> void QuokkaSimulation<ResampledCoolingTest>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// Compute initial internal energy from temperature
	const double Eint_initial = computeInitialInternalEnergy();
	const double Egas_initial = Eint_initial + active_magnetic_energy_initial;

	quokka::onezone::Cell<ResampledCoolingTest> cell{};
	cell[HydroSystem<ResampledCoolingTest>::density_index] = rho_initial;
	cell[HydroSystem<ResampledCoolingTest>::energy_index] = Egas_initial;
	cell[HydroSystem<ResampledCoolingTest>::internalEnergy_index] = Eint_initial;
	quokka::onezone::FillUniform<ResampledCoolingTest>(grid_elem, cell);
}

template <> void QuokkaSimulation<ResampledCoolingTest>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<ResampledCoolingTest>::mhdFirstIndex) = Bx_initial;
		} else {
			state_fc(i, j, k, Physics_Indices<ResampledCoolingTest>::mhdFirstIndex) = 0.0;
		}
	});
}

template <> void QuokkaSimulation<ResampledCoolingTest>::computeAfterTimestep() { userData_.record(state_new_cc_[0], Geom(0), tNew_[0]); }

auto problem_main() -> int
{
	// Read runtime parameters
	amrex::ParmParse const pp("cooling_test");
	std::string reference_file;
	pp.query("reference_solution_file", reference_file);

	std::string output_csv_file;
	pp.query("output_csv_file", output_csv_file);

	// If positive, require the final temperature at the last cell along x to exceed the temperature at the
	// first cell by this factor. Used to check that a position-dependent `heating_rate_external` is applied.
	double min_spatial_heating_T_ratio = 0.0;
	pp.query("min_spatial_heating_T_ratio", min_spatial_heating_T_ratio);

	amrex::ParmParse const ppp;
	bool use_sfh_based_pe_heating = false;
	ppp.query("use_sfh_based_pe_heating", use_sfh_based_pe_heating);

	QuokkaSimulation<ResampledCoolingTest> sim;

	sim.plotfileInterval_ = -1;

	sim.userData_.addObservable("temperature", [](quokka::onezone::CellValues const &cell) {
		// For isochoric MHD cooling with no kinetic energy and a uniform magnetic field, subtract the constant magnetic energy.
		const double Eint = cell[HydroSystem<ResampledCoolingTest>::energy_index] - active_magnetic_energy_initial;
		return quokka::EOS<ResampledCoolingTest>::ComputeTgasFromEint(cell[HydroSystem<ResampledCoolingTest>::density_index], Eint);
	});

	std::vector<quokka::onezone::Check> checks;
	if (!reference_file.empty()) {
		checks.push_back({.name = "temperature vs reference (rel. L1)",
				  .comparisons = {{.observable = "temperature", .reference = quokka::onezone::TableReference(reference_file, 1)}},
				  .tolerance = 0.02}); // should be about 1% between different cooling modules
	}

	std::vector<quokka::onezone::Predicate> predicates;
	// Check that gas has cooled significantly
	predicates.push_back({.name = "T_final / T_initial < 0.1", .passes = [](quokka::onezone::Recorder const &recorder) {
				      auto const &T = recorder.series("temperature");
				      return !T.empty() && (T.back() / T.front() <= 0.1);
			      }});
	// Check that a position-dependent external heating rate produced a position-dependent temperature
	if (min_spatial_heating_T_ratio > 0.0) {
		predicates.push_back(
		    {.name = "T(x_last) / T(x_first) > required ratio", .passes = [&sim, min_spatial_heating_T_ratio](quokka::onezone::Recorder const &) {
			     // the final profile along x (collective, so it must be called on every rank)
			     auto const final_profile = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);
			     if (!amrex::ParallelDescriptor::IOProcessor()) {
				     return true;
			     }
			     auto const &final_values = std::get<1>(final_profile);
			     auto const &Etot = final_values.at(HydroSystem<ResampledCoolingTest>::energy_index);
			     auto const &rho = final_values.at(HydroSystem<ResampledCoolingTest>::density_index);
			     const size_t i_last = Etot.size() - 1;
			     const double T_first = quokka::EOS<ResampledCoolingTest>::ComputeTgasFromEint(rho[0], Etot[0] - active_magnetic_energy_initial);
			     const double T_last =
				 quokka::EOS<ResampledCoolingTest>::ComputeTgasFromEint(rho[i_last], Etot[i_last] - active_magnetic_energy_initial);
			     const double T_ratio_x = T_last / T_first;
			     amrex::Print() << "Spatial heating check: T(x_last)/T(x_first) = " << T_ratio_x << " (required > " << min_spatial_heating_T_ratio
					    << ")\n";
			     return T_ratio_x > min_spatial_heating_T_ratio; // written so that a NaN ratio also fails
		     }});
	}

	quokka::onezone::Curve temperature_curve = {.observable = "temperature", .label = "Simulation"};
	if (!reference_file.empty()) {
		temperature_curve.reference = quokka::onezone::TableReference(reference_file, 1);
	}
	const std::vector<quokka::onezone::Figure> figures = {
	    {.filename = use_sfh_based_pe_heating ? "cooling_temperature_sfh_base_pe.pdf" : "cooling_temperature.pdf",
	     .curves = {temperature_curve},
	     .xlabel = "time (Myr)",
	     .ylabel = "Temperature (K)",
	     .log_x = true,
	     .log_y = true,
	     .title = "Isochoric Cooling Test",
	     .time_scale = 3.15576e13}}; // seconds in a Myr
	return quokka::onezone::Run(sim, checks, predicates, output_csv_file, figures);
}
