#ifndef EXPRESSION_FIELD_H
#define EXPRESSION_FIELD_H

#include "AMReX_Parser.H"
#include "AMReX_Print.H"
#include "DerivedFieldFactory.H"

// Expression-based derived field using AMReX parser
class ExpressionField : public DerivedFieldFactory::Register<ExpressionField>
{
      public:
	static auto identifier() -> std::string { return "expression"; }

	void init(const std::string &fieldName, const amrex::ParmParse &pp) override;
	void compute(amrex::MultiFab &mf, const amrex::MultiFab &state, const amrex::Geometry &geom, amrex::Real time, int ncomp) const override;

	[[nodiscard]] auto getFieldName() const -> std::string override { return fieldName_; }
	[[nodiscard]] auto getNumComponents() const -> int override { return 1; }
	[[nodiscard]] auto getComponentNames() const -> std::vector<std::string> override { return {fieldName_}; }
	[[nodiscard]] auto getRequiredStateVars() const -> std::vector<std::string> override;

      private:
	std::string expression_;
	std::vector<std::string> variableNames_;
	std::map<std::string, int> stateVariableMapping_;
	amrex::Parser parser_;
	amrex::ParserExecutor<7> parserExec_; // Support up to 7 variables (x,y,z,t + 3 state vars)

	// Parse the expression to extract variable names
	void parseVariableNames();

	// Map variable names to state variable indices
	void setupStateVariableMapping();
};

// User-defined field factory that creates expression fields based on configuration
class UserDefinedField : public DerivedFieldFactory::Register<UserDefinedField>
{
      public:
	static auto identifier() -> std::string { return "user_defined"; }

	void init(const std::string &fieldName, const amrex::ParmParse &pp) override;
	void compute(amrex::MultiFab &mf, const amrex::MultiFab &state, const amrex::Geometry &geom, amrex::Real time, int ncomp) const override;

	[[nodiscard]] auto getFieldName() const -> std::string override { return fieldName_; }
	[[nodiscard]] auto getNumComponents() const -> int override { return 1; }
	[[nodiscard]] auto getComponentNames() const -> std::vector<std::string> override { return {fieldName_}; }
	[[nodiscard]] auto getRequiredStateVars() const -> std::vector<std::string> override;

      private:
	std::shared_ptr<ExpressionField> expressionField_;
};

#endif // EXPRESSION_FIELD_H