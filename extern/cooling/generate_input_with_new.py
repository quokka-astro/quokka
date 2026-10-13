#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a grid of Cloudy input files for Jeans-length shielded models,
with an additional cap on effective shielding column density and a minimum
cloud depth floor.

For each (n_H, T):
  1. compute Jeans length
  2. compute shielding-limited depth: L_shield = target_NH / n_H
  3. set effective depth:
         L_raw = min(L_Jeans, L_shield, max_depth_pc)
         L_eff = max(L_raw, min_depth_pc)
  4. write one Cloudy input file with:
       - hden
       - constant temperature
       - optional stop temperature control
       - stop thickness
       - optional molecule-network control
       - optional grains
       - optional grain destruction command
       - optional switch to disable gas-dust thermal exchange
       - gas metallicity scaling
       - grain abundance scaling
       - optional species-cooling save block
  5. optionally run Cloudy

New radiation-field behavior:
  - --use-ism-field writes:
        table ISM
  - --use-hm12-uvb writes:
        table HM12 redshift 0
  - using both writes both commands, so ISRF + HM12 UVB are both present.
  - if neither is used, the old --radiation-field single-line behavior is used.

Compatible with Python 3.6+
"""

from __future__ import print_function
import os
import csv
import math
import argparse
import subprocess

# =========================
# Physical constants (cgs)
# =========================
K_B = 1.380649e-16
M_P = 1.67262192369e-24
G = 6.67430e-8
PC = 3.085677581491367e18

DEFAULT_SPECIES_COOLING_LABELS = [
    "C  1c 0",
    "C  2c 0",
    "O  1c 0",
    "Si 2c 0",
    "Fe 2c 0",
    "COc 0",
    "H2Oc 0",
]

MOLECULE_MODE_FLAGS = {
    "normal": (False, False, False),
    "no_h2": (True, False, False),
    "no_co": (False, True, False),
    "no_h2_no_co": (True, True, False),
    "no_grain_molecules": (False, False, True),
    "no_h2_no_grain_molecules": (True, False, True),
    "no_co_no_grain_molecules": (False, True, True),
    "no_h2_no_co_no_grain_molecules": (True, True, True),
}

FLAGS_TO_MOLECULE_MODE = dict((v, k) for k, v in MOLECULE_MODE_FLAGS.items())


def logspace_list(log10_min, log10_max, step):
    values = []
    x = log10_min
    while x <= log10_max + 1.0e-12:
        values.append(10.0 ** x)
        x += step
    return values


def jeans_length_cm(n_h, temperature, mu=1.4):
    if n_h <= 0.0:
        raise ValueError("n_h must be > 0")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    rho = mu * M_P * n_h
    c_s_sq = K_B * temperature / (mu * M_P)
    return math.sqrt(math.pi * c_s_sq / (G * rho))


def shielding_length_cm(n_h, target_NH):
    if n_h <= 0.0:
        raise ValueError("n_h must be > 0")
    if target_NH <= 0.0:
        raise ValueError("target_NH must be > 0")
    return target_NH / n_h


def format_for_name(value):
    s = "{:.3e}".format(value)
    return s.replace(".", "p").replace("+", "")


def pick_limiting_scale(l_jeans_cm, l_shield_cm, max_depth_cm,
                        min_depth_cm, l_raw_cm, l_eff_cm):
    eps = 1.0e-12 * max(1.0, abs(l_eff_cm))

    if abs(l_eff_cm - min_depth_cm) <= eps and (min_depth_cm > l_raw_cm + eps):
        return "min_depth"

    if abs(l_raw_cm - l_jeans_cm) <= eps:
        return "jeans"
    elif abs(l_raw_cm - l_shield_cm) <= eps:
        return "shield"
    elif abs(l_raw_cm - max_depth_cm) <= eps:
        return "max_depth"
    else:
        return "unknown"


def get_radiation_field_commands(args):
    """
    Build Cloudy radiation-field commands.

    New behavior:
      --use-ism-field -> table ISM
      --use-hm12-uvb  -> table HM12 redshift 0

    If neither new switch is set, fall back to the old --radiation-field option.
    """
    commands = []

    if args.use_ism_field:
        commands.append("table ISM")

    if args.use_hm12_uvb:
        commands.append("table HM12 redshift 0")

    if not commands:
        rf = args.radiation_field.strip() if args.radiation_field is not None else ""
        if rf:
            commands.append(rf)

    return commands


def get_molecule_commands(molecule_mode):
    if molecule_mode not in MOLECULE_MODE_FLAGS:
        raise ValueError("Unknown molecule_mode: {0}".format(molecule_mode))

    disable_h2, disable_co, disable_grain_molecules = MOLECULE_MODE_FLAGS[molecule_mode]
    commands = []

    if disable_h2:
        commands.append("no H2 molecules")
    if disable_co:
        commands.append("no CO molecules")
    if disable_grain_molecules:
        commands.append("no grain molecules")

    return commands


def remove_no_co_from_mode(molecule_mode):
    if molecule_mode not in MOLECULE_MODE_FLAGS:
        raise ValueError("Unknown molecule_mode: {0}".format(molecule_mode))

    disable_h2, disable_co, disable_grain_molecules = MOLECULE_MODE_FLAGS[molecule_mode]
    new_flags = (disable_h2, False, disable_grain_molecules)
    return FLAGS_TO_MOLECULE_MODE[new_flags]


def value_in_optional_range(value, vmin=None, vmax=None):
    if (vmin is not None) and (value < vmin):
        return False
    if (vmax is not None) and (value > vmax):
        return False
    return True


def co_override_window_is_enabled(args):
    return any([
        args.co_include_logn_min is not None,
        args.co_include_logn_max is not None,
        args.co_include_logT_min is not None,
        args.co_include_logT_max is not None,
    ])


def should_include_co_for_model(log10_n_h, log10_t, args):
    if not co_override_window_is_enabled(args):
        return False

    return (
        value_in_optional_range(log10_n_h, args.co_include_logn_min, args.co_include_logn_max)
        and value_in_optional_range(log10_t, args.co_include_logT_min, args.co_include_logT_max)
    )


def resolve_molecule_mode_for_model(base_molecule_mode, log10_n_h, log10_t, args):
    if base_molecule_mode not in MOLECULE_MODE_FLAGS:
        raise ValueError("Unknown molecule_mode: {0}".format(base_molecule_mode))

    disable_h2, disable_co, disable_grain_molecules = MOLECULE_MODE_FLAGS[base_molecule_mode]

    if (not disable_co) or (not should_include_co_for_model(log10_n_h, log10_t, args)):
        return base_molecule_mode, False

    effective_mode = remove_no_co_from_mode(base_molecule_mode)
    return effective_mode, True


def resolve_stop_temperature_for_model(temperature, args):
    if args.stop_temperature_off:
        return "off", None

    if (args.stop_temperature_off_above is not None) and (temperature >= args.stop_temperature_off_above):
        return "off", None

    if args.stop_temperature is not None:
        return "linear", args.stop_temperature

    return "default", None


def parse_species_cooling_labels(raw_labels):
    import re

    labels = []

    if not raw_labels:
        return labels

    number_pattern = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')

    for item in raw_labels:
        if item is None:
            continue

        pieces = item.split(",")
        for piece in pieces:
            s = piece.strip()
            if not s:
                continue

            if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
                s = s[1:-1].strip()

            wavelength = "0"
            label = s

            match = re.match(r'^(.*?)(?:\s+([^\s]+))?$', s)
            if match is not None:
                maybe_label = match.group(1)
                maybe_last = match.group(2)

                if maybe_last is not None and number_pattern.match(maybe_last):
                    label = maybe_label.rstrip()
                    wavelength = maybe_last

            label = label.strip().strip('"')
            if not label:
                continue

            normalized = '"{0}" {1}'.format(label, wavelength)
            labels.append(normalized)

    unique = []
    seen = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            unique.append(label)

    return unique


def write_species_cooling_block(lines, species_cooling_lines, species_cooling_suffix):
    if not species_cooling_lines:
        return

    lines.append('save lines, emissivity, "{0}"'.format(species_cooling_suffix))
    for label in species_cooling_lines:
        lines.append(label)
    lines.append('end of lines')


def spcool_window_is_enabled(args):
    return any([
        args.spcool_logn_min is not None,
        args.spcool_logn_max is not None,
        args.spcool_logT_min is not None,
        args.spcool_logT_max is not None,
    ])


def should_write_species_cooling_for_model(log10_n_h, log10_t, args):
    if not args.species_cooling:
        return False

    if not spcool_window_is_enabled(args):
        return True

    return (
        value_in_optional_range(log10_n_h, args.spcool_logn_min, args.spcool_logn_max)
        and value_in_optional_range(log10_t, args.spcool_logT_min, args.spcool_logT_max)
    )


def write_cloudy_input(path,
                       title,
                       log10_n_h,
                       temperature,
                       thickness_pc,
                       radiation_fields=None,
                       use_grains=False,
                       grains_max_temp=None,
                       grain_species="ISM",
                       grain_abundance=1.0,
                       use_cosmic_rays=True,
                       abundances="abundances GASS10",
                       metallicity=1.0,
                       include_cmb=True,
                       save_prefix="model",
                       iterate=True,
                       molecule_mode="normal",
                       enable_grain_destruction=False,
                       grain_destruction_cmd="set grains sputtering on",
                       disable_gas_dust_thermal_exchange=False,
                       stop_temperature=None,
                       stop_temperature_off=False,
                       species_cooling_lines=None,
                       species_cooling_suffix=".spcool"):
    if thickness_pc <= 0.0:
        raise ValueError("thickness_pc must be > 0")
    if stop_temperature_off and (stop_temperature is not None):
        raise ValueError("Cannot use both stop_temperature_off and stop_temperature")
    if stop_temperature is not None and stop_temperature <= 0.0:
        raise ValueError("stop_temperature must be > 0")
    if metallicity <= 0.0:
        raise ValueError("metallicity must be > 0")
    if grain_abundance <= 0.0:
        raise ValueError("grain_abundance must be > 0")

    if species_cooling_lines is None:
        species_cooling_lines = []

    if radiation_fields is None:
        radiation_fields = ["table HM12"]

    log10_thickness_pc = math.log10(thickness_pc)

    grains_enabled_this_model = False
    if use_grains:
        if grains_max_temp is None:
            grains_enabled_this_model = True
        else:
            grains_enabled_this_model = (temperature <= grains_max_temp)

    lines = []
    lines.append("title {0}".format(title))
    lines.append("hden {:.8f}".format(log10_n_h))
    lines.append("constant temperature linear {:.8f}".format(temperature))

    if stop_temperature_off:
        lines.append("stop temperature off")
    elif stop_temperature is not None:
        lines.append("stop temperature linear {:.8f}".format(stop_temperature))

    if include_cmb:
        lines.append("CMB")

    for rf in radiation_fields:
        rf = rf.strip()
        if rf:
            lines.append(rf)

    lines.append(abundances)
    lines.append("metals {:.8g} linear".format(metallicity))

    if grains_enabled_this_model:
        lines.append("grains {0} {1:.8g} linear".format(grain_species, grain_abundance))

        if disable_gas_dust_thermal_exchange:
            lines.append("no grain gas collisional energy exchange")

    lines.extend(get_molecule_commands(molecule_mode))

    if enable_grain_destruction and grains_enabled_this_model:
        if grain_destruction_cmd is not None:
            cmd = grain_destruction_cmd.strip()
            if cmd:
                lines.append(cmd)

    if use_cosmic_rays:
        lines.append("cosmic ray background")

    if iterate:
        lines.append("iterate to convergence")

    lines.append("stop thickness {:.8f} parsecs".format(log10_thickness_pc))

    lines.append('set save prefix "{0}"'.format(save_prefix))
    lines.append('save cooling last ".cool"')
    lines.append('save heating last ".heat"')
    lines.append('save molecules last ".mol"')

    write_species_cooling_block(
        lines=lines,
        species_cooling_lines=species_cooling_lines,
        species_cooling_suffix=species_cooling_suffix,
    )

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    return grains_enabled_this_model


def run_cloudy(cloudy_exe, input_path, output_path):
    cmd = '{exe} < "{inp}" > "{out}"'.format(
        exe=cloudy_exe, inp=input_path, out=output_path
    )
    return subprocess.call(cmd, shell=True)


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Cloudy input files on an (n_H, T) grid with Jeans-length + column-capped stop thickness + minimum depth floor."
    )

    parser.add_argument("--logn-min", type=float, default=-6.0,
                        help="Minimum log10(n_H/cm^-3)")
    parser.add_argument("--logn-max", type=float, default=8.0,
                        help="Maximum log10(n_H/cm^-3)")
    parser.add_argument("--logn-step", type=float, default=0.5,
                        help="Step in log10(n_H)")

    parser.add_argument("--logT-min", type=float, default=1.0,
                        help="Minimum log10(T/K)")
    parser.add_argument("--logT-max", type=float, default=9.0,
                        help="Maximum log10(T/K)")
    parser.add_argument("--logT-step", type=float, default=0.5,
                        help="Step in log10(T)")

    parser.add_argument("--mu", type=float, default=1.4,
                        help="Mean molecular weight used in Jeans length")
    parser.add_argument("--max-depth-pc", type=float, default=100.0,
                        help="Maximum allowed cloud depth in pc")
    parser.add_argument("--min-depth-pc", type=float, default=1.0e-3,
                        help="Minimum allowed cloud depth in pc")
    parser.add_argument("--target-NH", type=float, default=1.0e21,
                        help="Maximum effective shielding column density in cm^-2")

    parser.add_argument("--radiation-field", type=str, default="table HM12",
                        help=("Backward-compatible single Cloudy radiation command, "
                              'e.g. "table ISM" or "table HM12 redshift 0". '
                              "Ignored if --use-ism-field or --use-hm12-uvb is set."))
    parser.add_argument("--use-ism-field", action="store_true",
                        help="Add Cloudy command: table ISM")
    parser.add_argument("--use-hm12-uvb", action="store_true",
                        help="Add Cloudy command: table HM12 redshift 0")

    parser.add_argument("--metallicity", type=float, default=1.0,
                        help="Gas metallicity relative to solar, linear Z/Zsun.")

    parser.add_argument("--use-grains", action="store_true",
                        help="Include explicit grains command in the input file")
    parser.add_argument("--grains-max-temp", type=float, default=None,
                        help="Maximum temperature [K] for including grains.")
    parser.add_argument("--grain-abundance", type=float, default=1.0,
                        help="Relative grain abundance scale factor, linear.")
    parser.add_argument("--grain-species", type=str, default="ISM",
                        help='Cloudy grains keyword, e.g. "ISM"')
    parser.add_argument("--disable-gas-dust-thermal-exchange", action="store_true",
                        help="Write 'no grain gas collisional energy exchange' while keeping PE heating.")

    parser.add_argument("--molecule-mode", type=str, default="normal",
                        choices=sorted(MOLECULE_MODE_FLAGS.keys()),
                        help="How to control molecule-related physics.")

    parser.add_argument("--co-include-logn-min", type=float, default=None,
                        help="Within this log10(n_H/cm^-3) lower bound, temporarily remove 'no CO molecules'.")
    parser.add_argument("--co-include-logn-max", type=float, default=None,
                        help="Within this log10(n_H/cm^-3) upper bound, temporarily remove 'no CO molecules'.")
    parser.add_argument("--co-include-logT-min", type=float, default=None,
                        help="Within this log10(T/K) lower bound, temporarily remove 'no CO molecules'.")
    parser.add_argument("--co-include-logT-max", type=float, default=None,
                        help="Within this log10(T/K) upper bound, temporarily remove 'no CO molecules'.")

    parser.add_argument("--enable-grain-destruction", action="store_true",
                        help="Write a grain-destruction command into the Cloudy input.")
    parser.add_argument("--grain-destruction-cmd", type=str,
                        default="set grains sputtering on",
                        help="Cloudy command to enable grain destruction.")

    parser.add_argument("--no-cosmic-rays", action="store_true",
                        help="Disable 'cosmic ray background'")
    parser.add_argument("--abundances", type=str, default="abundances GASS10",
                        help='Cloudy abundances command')
    parser.add_argument("--no-cmb", action="store_true",
                        help="Disable CMB")
    parser.add_argument("--no-iterate", action="store_true",
                        help="Disable 'iterate to convergence'")

    parser.add_argument("--stop-temperature-off", action="store_true",
                        help="Write 'stop temperature off' into every Cloudy input")
    parser.add_argument("--stop-temperature-off-above", type=float, default=None,
                        help="Write 'stop temperature off' only for models above this T [K].")
    parser.add_argument("--stop-temperature", type=float, default=None,
                        help="Write 'stop temperature linear <K>' when stop-temperature-off is not active")

    parser.add_argument("--species-cooling", action="store_true",
                        help="Write a save lines, emissivity block for species cooling labels.")
    parser.add_argument("--species-cooling-defaults", action="store_true",
                        help="Use the default species cooling labels: C I, C II, O I, Si II, Fe II, CO, H2O.")
    parser.add_argument("--species-cooling-label", action="append", default=[],
                        help=("Species cooling label to save. Can be repeated. "
                              "Examples: --species-cooling-label 'C  1c 0' "
                              "--species-cooling-label 'COc 0'."))
    parser.add_argument("--species-cooling-outfile", type=str, default=".spcool",
                        help="Suffix for the species cooling save file, default: .spcool")

    parser.add_argument("--spcool-logn-min", type=float, default=None)
    parser.add_argument("--spcool-logn-max", type=float, default=None)
    parser.add_argument("--spcool-logT-min", type=float, default=None)
    parser.add_argument("--spcool-logT-max", type=float, default=None)

    parser.add_argument("--outdir", type=str, default="cloudy_grid_models",
                        help="Directory for all generated files")
    parser.add_argument("--run-cloudy", action="store_true",
                        help="Actually run Cloudy after generating inputs")
    parser.add_argument("--cloudy-exe", type=str, default="./cloudy.exe",
                        help="Path to Cloudy executable")

    args = parser.parse_args()

    if args.max_depth_pc <= 0.0:
        raise ValueError("--max-depth-pc must be > 0")
    if args.min_depth_pc <= 0.0:
        raise ValueError("--min-depth-pc must be > 0")
    if args.min_depth_pc > args.max_depth_pc:
        raise ValueError("--min-depth-pc must be <= --max-depth-pc")
    if args.target_NH <= 0.0:
        raise ValueError("--target-NH must be > 0")
    if args.mu <= 0.0:
        raise ValueError("--mu must be > 0")
    if args.metallicity <= 0.0:
        raise ValueError("--metallicity must be > 0")
    if args.grain_abundance <= 0.0:
        raise ValueError("--grain-abundance must be > 0")

    if args.stop_temperature_off and (args.stop_temperature_off_above is not None):
        raise ValueError("Cannot use both --stop-temperature-off and --stop-temperature-off-above")
    if args.stop_temperature is not None and args.stop_temperature <= 0.0:
        raise ValueError("--stop-temperature must be > 0")
    if (args.stop_temperature_off_above is not None) and (args.stop_temperature_off_above <= 0.0):
        raise ValueError("--stop-temperature-off-above must be > 0")

    if (args.co_include_logn_min is not None) and (args.co_include_logn_max is not None):
        if args.co_include_logn_min > args.co_include_logn_max:
            raise ValueError("--co-include-logn-min must be <= --co-include-logn-max")
    if (args.co_include_logT_min is not None) and (args.co_include_logT_max is not None):
        if args.co_include_logT_min > args.co_include_logT_max:
            raise ValueError("--co-include-logT-min must be <= --co-include-logT-max")

    if (args.spcool_logn_min is not None) and (args.spcool_logn_max is not None):
        if args.spcool_logn_min > args.spcool_logn_max:
            raise ValueError("--spcool-logn-min must be <= --spcool-logn-max")
    if (args.spcool_logT_min is not None) and (args.spcool_logT_max is not None):
        if args.spcool_logT_min > args.spcool_logT_max:
            raise ValueError("--spcool-logT-min must be <= --spcool-logT-max")

    species_cooling_lines = []
    if args.species_cooling_defaults:
        species_cooling_lines.extend(DEFAULT_SPECIES_COOLING_LABELS)
    if args.species_cooling_label:
        species_cooling_lines.extend(parse_species_cooling_labels(args.species_cooling_label))

    if species_cooling_lines:
        args.species_cooling = True

    if spcool_window_is_enabled(args):
        args.species_cooling = True

    if args.species_cooling and (not species_cooling_lines):
        species_cooling_lines = list(DEFAULT_SPECIES_COOLING_LABELS)

    species_cooling_lines = parse_species_cooling_labels(species_cooling_lines)

    ensure_dir(args.outdir)

    radiation_fields = get_radiation_field_commands(args)

    n_values = logspace_list(args.logn_min, args.logn_max, args.logn_step)
    t_values = logspace_list(args.logT_min, args.logT_max, args.logT_step)

    summary_csv = os.path.join(args.outdir, "grid_summary.csv")
    max_depth_cm = args.max_depth_pc * PC
    min_depth_cm = args.min_depth_pc * PC

    with open(summary_csv, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "model_name",
            "log10_nH",
            "nH_cm^-3",
            "log10_T",
            "T_K",
            "L_Jeans_cm",
            "L_Jeans_pc",
            "L_shield_cm",
            "L_shield_pc",
            "L_raw_cm",
            "L_raw_pc",
            "min_depth_cm",
            "min_depth_pc",
            "max_depth_cm",
            "max_depth_pc",
            "L_eff_cm",
            "L_eff_pc",
            "log10_L_eff_pc_for_cloudy",
            "limiter",
            "NH_eff_cm^-2",
            "radiation_fields",
            "metallicity_Z_over_Zsun",
            "grain_species",
            "grain_abundance_scale",
            "grains_included",
            "gas_dust_thermal_exchange_disabled",
            "base_molecule_mode",
            "effective_molecule_mode",
            "co_override_active",
            "grain_destruction_enabled",
            "grain_destruction_cmd",
            "stop_temperature_mode",
            "stop_temperature_value_K",
            "stop_temperature_off_above_K",
            "species_cooling_enabled",
            "species_cooling_written_this_model",
            "species_cooling_window_enabled",
            "spcool_logn_min",
            "spcool_logn_max",
            "spcool_logT_min",
            "spcool_logT_max",
            "species_cooling_outfile",
            "species_cooling_labels",
            "input_file",
            "output_file",
        ])

        total = 0
        co_override_count = 0
        stop_off_count = 0
        spcool_written_count = 0

        for n_h in n_values:
            log10_n_h = math.log10(n_h)

            for temperature in t_values:
                log10_t = math.log10(temperature)

                l_jeans_cm = jeans_length_cm(n_h, temperature, mu=args.mu)
                l_shield_cm = shielding_length_cm(n_h, args.target_NH)

                l_raw_cm = min(l_jeans_cm, l_shield_cm, max_depth_cm)
                l_eff_cm = max(l_raw_cm, min_depth_cm)

                l_eff_pc = l_eff_cm / PC
                l_raw_pc = l_raw_cm / PC
                log10_l_eff_pc = math.log10(l_eff_pc)

                limiter = pick_limiting_scale(
                    l_jeans_cm=l_jeans_cm,
                    l_shield_cm=l_shield_cm,
                    max_depth_cm=max_depth_cm,
                    min_depth_cm=min_depth_cm,
                    l_raw_cm=l_raw_cm,
                    l_eff_cm=l_eff_cm
                )

                nh_eff = n_h * l_eff_cm

                model_name = "n_{0}_T_{1}".format(
                    format_for_name(n_h), format_for_name(temperature)
                )

                input_filename = model_name + ".in"
                output_filename = model_name + ".out"
                input_path = os.path.join(args.outdir, input_filename)
                output_path = os.path.join(args.outdir, output_filename)

                save_prefix = model_name
                title = model_name

                effective_molecule_mode, co_override_active = resolve_molecule_mode_for_model(
                    base_molecule_mode=args.molecule_mode,
                    log10_n_h=log10_n_h,
                    log10_t=log10_t,
                    args=args
                )

                stop_temperature_mode, stop_temperature_value = resolve_stop_temperature_for_model(
                    temperature=temperature,
                    args=args
                )

                write_spcool_this_model = should_write_species_cooling_for_model(
                    log10_n_h=log10_n_h,
                    log10_t=log10_t,
                    args=args
                )

                grains_included = write_cloudy_input(
                    path=input_path,
                    title=title,
                    log10_n_h=log10_n_h,
                    temperature=temperature,
                    thickness_pc=l_eff_pc,
                    radiation_fields=radiation_fields,
                    use_grains=args.use_grains,
                    grains_max_temp=args.grains_max_temp,
                    grain_species=args.grain_species,
                    grain_abundance=args.grain_abundance,
                    use_cosmic_rays=(not args.no_cosmic_rays),
                    abundances=args.abundances,
                    metallicity=args.metallicity,
                    include_cmb=(not args.no_cmb),
                    save_prefix=save_prefix,
                    iterate=(not args.no_iterate),
                    molecule_mode=effective_molecule_mode,
                    enable_grain_destruction=args.enable_grain_destruction,
                    grain_destruction_cmd=args.grain_destruction_cmd,
                    disable_gas_dust_thermal_exchange=args.disable_gas_dust_thermal_exchange,
                    stop_temperature=stop_temperature_value,
                    stop_temperature_off=(stop_temperature_mode == "off"),
                    species_cooling_lines=species_cooling_lines if write_spcool_this_model else [],
                    species_cooling_suffix=args.species_cooling_outfile,
                )

                if co_override_active:
                    co_override_count += 1
                if stop_temperature_mode == "off":
                    stop_off_count += 1
                if write_spcool_this_model:
                    spcool_written_count += 1

                grain_destruction_was_written = int(
                    args.enable_grain_destruction and grains_included
                )
                grain_destruction_cmd_written = args.grain_destruction_cmd if grain_destruction_was_written else ""

                if stop_temperature_mode == "linear":
                    stop_temperature_value_str = "{:.8e}".format(stop_temperature_value)
                else:
                    stop_temperature_value_str = ""

                gas_dust_exchange_disabled_this_model = int(
                    args.disable_gas_dust_thermal_exchange and grains_included
                )

                writer.writerow([
                    model_name,
                    "{:.8e}".format(log10_n_h),
                    "{:.8e}".format(n_h),
                    "{:.8e}".format(log10_t),
                    "{:.8e}".format(temperature),
                    "{:.8e}".format(l_jeans_cm),
                    "{:.8e}".format(l_jeans_cm / PC),
                    "{:.8e}".format(l_shield_cm),
                    "{:.8e}".format(l_shield_cm / PC),
                    "{:.8e}".format(l_raw_cm),
                    "{:.8e}".format(l_raw_pc),
                    "{:.8e}".format(min_depth_cm),
                    "{:.8e}".format(args.min_depth_pc),
                    "{:.8e}".format(max_depth_cm),
                    "{:.8e}".format(args.max_depth_pc),
                    "{:.8e}".format(l_eff_cm),
                    "{:.8e}".format(l_eff_pc),
                    "{:.8e}".format(log10_l_eff_pc),
                    limiter,
                    "{:.8e}".format(nh_eff),
                    "; ".join(radiation_fields),
                    "{:.8e}".format(args.metallicity),
                    args.grain_species,
                    "{:.8e}".format(args.grain_abundance),
                    int(grains_included),
                    gas_dust_exchange_disabled_this_model,
                    args.molecule_mode,
                    effective_molecule_mode,
                    int(co_override_active),
                    grain_destruction_was_written,
                    grain_destruction_cmd_written,
                    stop_temperature_mode,
                    stop_temperature_value_str,
                    ("{:.8e}".format(args.stop_temperature_off_above)
                     if args.stop_temperature_off_above is not None else ""),
                    int(args.species_cooling),
                    int(write_spcool_this_model),
                    int(spcool_window_is_enabled(args)),
                    ("{:.8e}".format(args.spcool_logn_min)
                     if args.spcool_logn_min is not None else ""),
                    ("{:.8e}".format(args.spcool_logn_max)
                     if args.spcool_logn_max is not None else ""),
                    ("{:.8e}".format(args.spcool_logT_min)
                     if args.spcool_logT_min is not None else ""),
                    ("{:.8e}".format(args.spcool_logT_max)
                     if args.spcool_logT_max is not None else ""),
                    args.species_cooling_outfile if write_spcool_this_model else "",
                    "; ".join(species_cooling_lines) if write_spcool_this_model else "",
                    input_filename,
                    output_filename,
                ])

                total += 1

                if args.run_cloudy:
                    ret = run_cloudy(args.cloudy_exe, input_path, output_path)
                    if ret != 0:
                        print("WARNING: Cloudy returned non-zero exit code for", input_filename)

    print("Done.")
    print("Generated {} models.".format(total))
    print("Summary written to:", summary_csv)
    print("Output directory:", args.outdir)
    print("Using target shielding column density: {:.3e} cm^-2".format(args.target_NH))
    print("Using minimum depth: {:.3e} pc".format(args.min_depth_pc))
    print("Using maximum depth: {:.3e} pc".format(args.max_depth_pc))

    print("Radiation fields:")
    for rf in radiation_fields:
        print("  ", rf)

    print("Base molecule mode:", args.molecule_mode)
    print("Gas metallicity (Z/Zsun):", args.metallicity)

    if co_override_window_is_enabled(args):
        print("CO override window enabled.")
        print("  log10(n_H) range: [{}, {}]".format(args.co_include_logn_min, args.co_include_logn_max))
        print("  log10(T)   range: [{}, {}]".format(args.co_include_logT_min, args.co_include_logT_max))
        print("  Models where CO override was applied:", co_override_count)
    else:
        print("CO override window disabled.")

    if args.use_grains:
        print("Grain species:", args.grain_species)
        print("Grain abundance scale:", args.grain_abundance)
        if args.grains_max_temp is None:
            print("Grains enabled for all temperatures.")
        else:
            print("Grains enabled only for T <= {:.3e} K".format(args.grains_max_temp))

        if args.disable_gas_dust_thermal_exchange:
            print("Gas-dust collisional thermal exchange: DISABLED")
            print("Cloudy command added: no grain gas collisional energy exchange")
            print("PE heating is kept because no 'grains ... no heating' command is written.")
        else:
            print("Gas-dust collisional thermal exchange: ENABLED")
    else:
        print("Grains disabled globally.")

    if args.enable_grain_destruction:
        print("Grain destruction command enabled:", args.grain_destruction_cmd)
    else:
        print("Grain destruction command disabled.")

    if args.stop_temperature_off:
        print("Stop temperature control: OFF for all models")
    elif args.stop_temperature_off_above is not None:
        print("Stop temperature control: OFF only for T >= {:.3e} K".format(args.stop_temperature_off_above))
        if args.stop_temperature is not None:
            print("Below that threshold: linear {:.3e} K".format(args.stop_temperature))
        else:
            print("Below that threshold: Cloudy default")
    elif args.stop_temperature is not None:
        print("Stop temperature control: linear {:.3e} K".format(args.stop_temperature))
    else:
        print("Stop temperature control: Cloudy default")

    if args.species_cooling:
        print("Species cooling block enabled.")
        print("Species cooling output suffix:", args.species_cooling_outfile)
        print("Species cooling labels:")
        for label in species_cooling_lines:
            print("  ", label)

        if spcool_window_is_enabled(args):
            print("Species cooling output window enabled.")
            print("  log10(n_H) range: [{}, {}]".format(args.spcool_logn_min, args.spcool_logn_max))
            print("  log10(T)   range: [{}, {}]".format(args.spcool_logT_min, args.spcool_logT_max))
            print("Models where .spcool block was written:", spcool_written_count)
        else:
            print("Species cooling output window disabled.")
            print("The .spcool block is written for all models.")
    else:
        print("Species cooling block disabled.")

    print("Models with stop temperature off:", stop_off_count)


if __name__ == "__main__":
    main()