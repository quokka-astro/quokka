#include "BuiltinFields.H"
#include "ExpressionField.H"

// This file ensures that all built-in fields are registered with the factory system
// by referencing them, which triggers their static registration.

namespace {
	// Force registration of built-in fields by referencing their static members
	void registerBuiltinFields() {
		// This function should be called at startup to ensure all built-in fields are registered
		// The act of referencing the identifier() function triggers the static registration
		volatile auto temp_id = TemperatureField::identifier();
		volatile auto vort_id = VorticityField::identifier();
		volatile auto bfield_id = BFieldDivergenceField::identifier();
		volatile auto sound_id = SoundSpeedField::identifier();
		volatile auto ndens_id = NumberDensityField::identifier();
		volatile auto expr_id = ExpressionField::identifier();
		volatile auto user_id = UserDefinedField::identifier();
		
		// Suppress unused variable warnings
		(void)temp_id;
		(void)vort_id;
		(void)bfield_id;
		(void)sound_id;
		(void)ndens_id;
		(void)expr_id;
		(void)user_id;
	}
	
	// Use a static initializer to call the registration function
	static bool registered = []() {
		registerBuiltinFields();
		return true;
	}();
}