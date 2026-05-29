#!/usr/bin/env python3
"""Evaluate the thermal Jeans mass for inputs/PopIII_production.toml."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    # CGS constants used by Quokka/Microphysics.
    G = sp.Float("6.67428e-8")
    k_B = sp.Float("1.380649e-16")
    n_A = sp.Float("6.02214076e23")
    m_p = sp.Float("1.67262192595e-24")
    m_sun = sp.Float("1.9884e33")
    parsec = sp.Float("3.085677581491367e18")

    # Cloud parameters from inputs/PopIII_production.toml.
    cloud_radius = sp.Float("3.086e18")
    cloud_numdens = sp.Float("0.90861183e4")
    cloud_temperature = sp.Float("0.26415744e3")

    # primordial_chem.primary_species_1..14 from inputs/PopIII_production.toml.
    relative_numdens = [
        sp.Float("0.88499253e-006"),
        sp.Float("0.88498062e-006"),
        sp.Float("0.99932238e+000"),
        sp.Float("0.54719550e-013"),
        sp.Float("0.21957612e-010"),
        sp.Float("0.29920413e-004"),
        sp.Float("0.58304958e-015"),
        sp.Float("0.22122496e-017"),
        sp.Float("0.38932607e-003"),
        sp.Float("0.36774691e-019"),
        sp.Float("0.79574711e-007"),
        sp.Float("0.39651766e-050"),
        sp.Float("0.24136647e-043"),
        sp.Float("0.77500001e-001"),
    ]

    # Species masses and gammas from extern/Microphysics/EOS/primordial_chem/_parameters.
    species_masses = [
        sp.Float("9.10938188e-28"),
        sp.Float("1.67262158e-24"),
        sp.Float("1.67353251819e-24"),
        sp.Float("1.67444345638e-24"),
        sp.Float("3.34512158e-24"),
        sp.Float("3.34603251819e-24"),
        sp.Float("3.34615409819e-24"),
        sp.Float("3.34694345638e-24"),
        sp.Float("3.34706503638e-24"),
        sp.Float("5.01865409819e-24"),
        sp.Float("5.01956503638e-24"),
        sp.Float("6.69024316e-24"),
        sp.Float("6.69115409819e-24"),
        sp.Float("6.69206503638e-24"),
    ]
    species_gammas = [
        sp.Rational(5, 3),
        sp.Rational(5, 3),
        sp.Rational(5, 3),
        sp.Rational(5, 3),
        sp.Rational(5, 3),
        sp.Rational(5, 3),
        sp.Float("1.4"),
        sp.Rational(5, 3),
        sp.Float("1.4"),
        sp.Float("1.4"),
        sp.Float("1.4"),
        sp.Rational(5, 3),
        sp.Rational(5, 3),
        sp.Rational(5, 3),
    ]

    number_densities = [cloud_numdens * rel for rel in relative_numdens]
    rho = sum(n_i * m_i for n_i, m_i in zip(number_densities, species_masses))

    sum_abarinv = sum(number_densities) * m_p / rho
    sum_gammasinv = sum(
        n_i * m_p / rho / (gamma_i - 1)
        for n_i, gamma_i in zip(number_densities, species_gammas)
    ) / sum_abarinv
    gamma_eff = 1 + 1 / sum_gammasinv

    gas_constant = n_A * k_B
    specific_eint = sum_gammasinv * sum_abarinv * gas_constant * cloud_temperature
    pressure = rho * specific_eint / sum_gammasinv
    sound_speed = sp.sqrt(gamma_eff * pressure / rho)

    c_s, rho_sym, G_sym = sp.symbols("c_s rho G", positive=True)
    jeans_length_expr = c_s * sp.sqrt(sp.pi / (G_sym * rho_sym))
    jeans_mass_expr = sp.Rational(4, 3) * sp.pi * rho_sym * (jeans_length_expr / 2) ** 3
    jeans_mass_closed_expr = sp.simplify(jeans_mass_expr)

    substitutions = {c_s: sound_speed, rho_sym: rho, G_sym: G}
    jeans_length = jeans_length_expr.subs(substitutions)
    jeans_mass = jeans_mass_closed_expr.subs(substitutions)
    cloud_mass = sp.Rational(4, 3) * sp.pi * rho * cloud_radius**3

    print("Jeans mass expression:")
    print(f"  M_J = {jeans_mass_closed_expr}")
    print()
    print("PopIII production cloud values:")
    print(f"  rho                 = {float(rho):.8e} g cm^-3")
    print(f"  gamma_eff           = {float(gamma_eff):.8f}")
    print(f"  c_s                 = {float(sound_speed):.8e} cm s^-1")
    print(f"  lambda_J            = {float(jeans_length):.8e} cm")
    print(f"  lambda_J            = {float(jeans_length / parsec):.8f} pc")
    print(f"  M_J                 = {float(jeans_mass):.8e} g")
    print(f"  M_J                 = {float(jeans_mass / m_sun):.8f} Msun")
    print(f"  M_cloud             = {float(cloud_mass / m_sun):.8f} Msun")
    print(f"  M_cloud / M_J       = {float(cloud_mass / jeans_mass):.8f}")
    print(f"  2 R_cloud / lambda_J = {float(2 * cloud_radius / jeans_length):.8f}")


if __name__ == "__main__":
    main()
