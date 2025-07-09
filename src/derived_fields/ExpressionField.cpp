#include "ExpressionField.H"
#include <regex>
#include <sstream>

// ExpressionField implementation
void ExpressionField::init(const std::string &fieldName, const amrex::ParmParse &pp)
{
	fieldName_ = fieldName;

	// Read the mathematical expression
	pp.get("expression", expression_);

	// Parse variable names from expression
	parseVariableNames();

	// Set up state variable mapping
	setupStateVariableMapping();

	// Create parser
	parser_.define(expression_);

	// Create parser executor with coordinate and time variables
	std::vector<std::string> parserVars = {"x", "y", "z", "t"};
	for (const auto &var : variableNames_) {
		if (std::find(parserVars.begin(), parserVars.end(), var) == parserVars.end()) {
			parserVars.push_back(var);
		}
	}

	// Resize to maximum 7 variables
	parserVars.resize(std::min(parserVars.size(), size_t(7)));

	parserExec_ = parser_.compileHost<7>();

	amrex::Print() << "Expression field '" << fieldName_ << "' initialized with expression: " << expression_ << std::endl;
}

void ExpressionField::parseVariableNames()
{
	// Extract variable names from expression using regex
	// Look for patterns like: rho, momx, momy, momz, energy, etc.
	std::regex varPattern(R"([a-zA-Z_][a-zA-Z0-9_]*)");
	std::smatch matches;

	auto start = expression_.cbegin();
	while (std::regex_search(start, expression_.cend(), matches, varPattern)) {
		std::string var = matches[0].str();

		// Skip mathematical functions and constants
		if (var != "sin" && var != "cos" && var != "tan" && var != "exp" && var != "log" && var != "sqrt" && var != "abs" && var != "pow" &&
		    var != "pi" && var != "e" && var != "x" && var != "y" && var != "z" && var != "t") {
			if (std::find(variableNames_.begin(), variableNames_.end(), var) == variableNames_.end()) {
				variableNames_.push_back(var);
			}
		}

		start = matches.suffix().first;
	}
}

void ExpressionField::setupStateVariableMapping()
{
	// Map common variable names to state indices
	// This is a simplified mapping - in practice, this would be configured
	// based on the specific physics problem being solved
	stateVariableMapping_["rho"] = 0; // density
	stateVariableMapping_["density"] = 0;
	stateVariableMapping_["momx"] = 1; // x1Momentum
	stateVariableMapping_["x1Momentum"] = 1;
	stateVariableMapping_["momy"] = 2; // x2Momentum
	stateVariableMapping_["x2Momentum"] = 2;
	stateVariableMapping_["momz"] = 3; // x3Momentum
	stateVariableMapping_["x3Momentum"] = 3;
	stateVariableMapping_["energy"] = 4; // energy
	stateVariableMapping_["Etot"] = 4;

	// Add more mappings as needed for specific physics modules
	stateVariableMapping_["Bx"] = 5; // x1BField
	stateVariableMapping_["By"] = 6; // x2BField
	stateVariableMapping_["Bz"] = 7; // x3BField
}

auto ExpressionField::getRequiredStateVars() const -> std::vector<std::string>
{
	std::vector<std::string> required;
	for (const auto &var : variableNames_) {
		auto it = stateVariableMapping_.find(var);
		if (it != stateVariableMapping_.end()) {
			required.push_back(var);
		}
	}
	return required;
}

void ExpressionField::compute(amrex::MultiFab &mf, const amrex::MultiFab &state, const amrex::Geometry &geom, amrex::Real time, int ncomp) const
{
	auto const &stateArrays = state.const_arrays();
	auto const &outputArrays = mf.arrays();
	auto const &dx = geom.CellSizeArray();
	auto const &problo = geom.ProbLoArray();

	// Get parser executor for GPU use
	auto const &parserExec = parserExec_;
	auto const &stateVarMap = stateVariableMapping_;
	auto const &varNames = variableNames_;

	amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// Compute spatial coordinates
		amrex::Real x = problo[0] + (i + 0.5) * dx[0];
		amrex::Real y = problo[1] + (j + 0.5) * dx[1];
		amrex::Real z = problo[2] + (k + 0.5) * dx[2];

		// Prepare variables for parser
		amrex::GpuArray<amrex::Real, 7> parserVars;
		parserVars[0] = x;
		parserVars[1] = y;
		parserVars[2] = z;
		parserVars[3] = time;

		// Add state variables
		int varIndex = 4;
		for (const auto &var : varNames) {
			if (varIndex < 7) {
				auto it = stateVarMap.find(var);
				if (it != stateVarMap.end()) {
					parserVars[varIndex] = stateArrays[bx](i, j, k, it->second);
					varIndex++;
				}
			}
		}

		// Evaluate expression
		amrex::Real result = parserExec(parserVars[0], parserVars[1], parserVars[2], parserVars[3], parserVars[4], parserVars[5], parserVars[6]);

		outputArrays[bx](i, j, k, ncomp) = result;
	});
}

// UserDefinedField implementation
void UserDefinedField::init(const std::string &fieldName, const amrex::ParmParse &pp)
{
	fieldName_ = fieldName;

	// Create an expression field as the backend
	expressionField_ = std::make_shared<ExpressionField>();
	expressionField_->init(fieldName, pp);
}

void UserDefinedField::compute(amrex::MultiFab &mf, const amrex::MultiFab &state, const amrex::Geometry &geom, amrex::Real time, int ncomp) const
{
	if (expressionField_) {
		expressionField_->compute(mf, state, geom, time, ncomp);
	}
}

auto UserDefinedField::getRequiredStateVars() const -> std::vector<std::string>
{
	if (expressionField_) {
		return expressionField_->getRequiredStateVars();
	}
	return {};
}