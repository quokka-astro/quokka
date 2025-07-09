#ifndef BUILTIN_FIELDS_H
#define BUILTIN_FIELDS_H

#include "DerivedFieldFactory.H"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"
#include "hydro/EOS.hpp"

// Temperature field
class TemperatureField : public DerivedFieldFactory::Register<TemperatureField>
{
public:
	static auto identifier() -> std::string { return "temperature"; }

	void init(const std::string& fieldName, const amrex::ParmParse& pp) override;
	void compute(amrex::MultiFab& mf, const amrex::MultiFab& state, 
		     const amrex::Geometry& geom, amrex::Real time, int ncomp) const override;
	
	[[nodiscard]] auto getFieldName() const -> std::string override { return fieldName_; }
	[[nodiscard]] auto getNumComponents() const -> int override { return 1; }
	[[nodiscard]] auto getComponentNames() const -> std::vector<std::string> override { return {"temperature"}; }
	[[nodiscard]] auto getRequiredStateVars() const -> std::vector<std::string> override 
	{ 
		return {"density", "x1Momentum", "x2Momentum", "x3Momentum", "energy"}; 
	}

private:
	amrex::Real gamma_{5.0/3.0};
	amrex::Real mean_molecular_weight_{2.33e-24}; // for fully ionized hydrogen
	amrex::Real boltzmann_constant_{1.380649e-16}; // erg/K
};

// Vorticity field
class VorticityField : public DerivedFieldFactory::Register<VorticityField>
{
public:
	static auto identifier() -> std::string { return "vorticity"; }

	void init(const std::string& fieldName, const amrex::ParmParse& pp) override;
	void compute(amrex::MultiFab& mf, const amrex::MultiFab& state, 
		     const amrex::Geometry& geom, amrex::Real time, int ncomp) const override;
	
	[[nodiscard]] auto getFieldName() const -> std::string override { return fieldName_; }
	[[nodiscard]] auto getNumComponents() const -> int override { return AMREX_SPACEDIM; }
	[[nodiscard]] auto getComponentNames() const -> std::vector<std::string> override;
	[[nodiscard]] auto getRequiredStateVars() const -> std::vector<std::string> override 
	{ 
		return {"density", "x1Momentum", "x2Momentum", "x3Momentum"}; 
	}
};

// B-field divergence field
class BFieldDivergenceField : public DerivedFieldFactory::Register<BFieldDivergenceField>
{
public:
	static auto identifier() -> std::string { return "bfield_divergence"; }

	void init(const std::string& fieldName, const amrex::ParmParse& pp) override;
	void compute(amrex::MultiFab& mf, const amrex::MultiFab& state, 
		     const amrex::Geometry& geom, amrex::Real time, int ncomp) const override;
	
	[[nodiscard]] auto getFieldName() const -> std::string override { return fieldName_; }
	[[nodiscard]] auto getNumComponents() const -> int override { return 1; }
	[[nodiscard]] auto getComponentNames() const -> std::vector<std::string> override { return {"bfield_divergence"}; }
	[[nodiscard]] auto getRequiredStateVars() const -> std::vector<std::string> override 
	{ 
		return {"x1BField", "x2BField", "x3BField"}; 
	}
};

// Sound speed field
class SoundSpeedField : public DerivedFieldFactory::Register<SoundSpeedField>
{
public:
	static auto identifier() -> std::string { return "sound_speed"; }

	void init(const std::string& fieldName, const amrex::ParmParse& pp) override;
	void compute(amrex::MultiFab& mf, const amrex::MultiFab& state, 
		     const amrex::Geometry& geom, amrex::Real time, int ncomp) const override;
	
	[[nodiscard]] auto getFieldName() const -> std::string override { return fieldName_; }
	[[nodiscard]] auto getNumComponents() const -> int override { return 1; }
	[[nodiscard]] auto getComponentNames() const -> std::vector<std::string> override { return {"sound_speed"}; }
	[[nodiscard]] auto getRequiredStateVars() const -> std::vector<std::string> override 
	{ 
		return {"density", "x1Momentum", "x2Momentum", "x3Momentum", "energy"}; 
	}

private:
	amrex::Real gamma_{5.0/3.0};
};

// Number density field
class NumberDensityField : public DerivedFieldFactory::Register<NumberDensityField>
{
public:
	static auto identifier() -> std::string { return "number_density"; }

	void init(const std::string& fieldName, const amrex::ParmParse& pp) override;
	void compute(amrex::MultiFab& mf, const amrex::MultiFab& state, 
		     const amrex::Geometry& geom, amrex::Real time, int ncomp) const override;
	
	[[nodiscard]] auto getFieldName() const -> std::string override { return fieldName_; }
	[[nodiscard]] auto getNumComponents() const -> int override { return 1; }
	[[nodiscard]] auto getComponentNames() const -> std::vector<std::string> override { return {"number_density"}; }
	[[nodiscard]] auto getRequiredStateVars() const -> std::vector<std::string> override 
	{ 
		return {"density"}; 
	}

private:
	amrex::Real particle_mass_{1.67e-24}; // proton mass in grams
};

#endif // BUILTIN_FIELDS_H