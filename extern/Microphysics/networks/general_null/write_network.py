#!/usr/bin/env python3

import sys
import argparse

import network_param_file


def enum_type(size):
    dtype = "std::uint32_t"
    # always give 1 addition int for overall size
    if size < 255:
        dtype = "std::uint8_t"
    elif size < 65535:
        dtype = "std::uint16_t"
    return dtype

def abort(outfile):
    """exit when there is an error.  A dummy stub file is written out,
    which will cause a compilation failure

    """

    fout = open(outfile, "w")
    fout.write("There was an error parsing the network files")
    fout.close()
    sys.exit(1)


def write_network(header_template,
                  net_file,
                  header_file, defines):
    """read through the list of species and output the new out_file

    """

    species = []
    extra_species = []
    aux_vars = []

    # read the species defined in the net_file

    print(f"write_network.py: working on network file {net_file} ...")

    err = network_param_file.parse(species, extra_species, aux_vars, net_file, defines)

    if err:
        abort(header_file)

    # write out the C++ files based on the templates

    templates = [(header_template, header_file)]

    for tmp, out_file in templates:

        if tmp == "":
            continue

        print(f"writing {out_file}")

        # read the template
        try:
            template = open(tmp)
        except OSError:
            sys.exit(f"write_network.py: ERROR: template file {tmp} does not exist")
        else:
            template_lines = template.readlines()
            template.close()

        # output the new file, inserting the species info in between the @@...@@
        fout = open(out_file, "w")

        for line in template_lines:

            # now look for replacements of entire lines
            index = line.find("@@")

            if index >= 0:
                index2 = line.rfind("@@")

                keyword = line[index+len("@@"):index2]
                indent = index*" "

                if keyword == "NSPEC":
                    fout.write(line.replace("@@NSPEC@@", str(len(species))))

                if keyword == "NEXTRASPEC":
                    fout.write(line.replace("@@NEXTRASPEC@@", str(len(extra_species))))

                elif keyword == "NAUX":
                    fout.write(line.replace("@@NAUX@@", str(len(aux_vars))))

                elif keyword == "SPEC_NAMES":
                    for n, spec in enumerate(species):
                        fout.write(f"{indent}\"{spec.name}\",   // {n} \n")

                    for n, spec in enumerate(extra_species):
                        fout.write(f"{indent}\"{spec.name}\",   // {n + len(species)} \n")

                elif keyword == "SHORT_SPEC_NAMES":
                    for n, spec in enumerate(species):
                        fout.write(f"{indent}\"{spec.short_name}\",   // {n} \n")

                    for n, spec in enumerate(extra_species):
                        fout.write(f"{indent}\"{spec.short_name}\",   // {n + len(species)} \n")

                elif keyword == "AION":
                    for n, spec in enumerate(species):
                        fout.write(f"{indent}{spec.A},   // {n} \n")

                    for n, spec in enumerate(extra_species):
                        fout.write(f"{indent}{spec.A},   // {n + len(species)} \n")

                elif keyword == "AION_CONSTEXPR":
                    fout.write("\n")
                    for n, spec in enumerate(species):
                        fout.write(f"{indent}case {spec.short_name.capitalize()}:   // {n+1}\n")
                        fout.write(f"{indent}{{\n")
                        fout.write(f"{indent}    a = {spec.A};\n")
                        fout.write(f"{indent}    break;\n")
                        fout.write(f"{indent}}}\n\n")

                    for n, spec in enumerate(extra_species):
                        fout.write(f"{indent}case {spec.short_name.capitalize()}:   // {n + len(species) + 1}\n")
                        fout.write(f"{indent}{{\n")
                        fout.write(f"{indent}    a = {spec.A};\n")
                        fout.write(f"{indent}    break;\n")
                        fout.write(f"{indent}}}\n\n")

                elif keyword == "AION_INV":
                    for n, spec in enumerate(species):
                        if spec.A == 1:
                            fout.write(f"{indent}1.0,            // {n} \n")
                        else:
                            fout.write(f"{indent}1.0/{spec.A},   // {n} \n")

                    for n, spec in enumerate(extra_species):
                        if spec.A == 1:
                            fout.write(f"{indent}1.0,            // {n + len(species)} \n")
                        else:
                            fout.write(f"{indent}1.0/{spec.A},   // {n + len(species)} \n")

                elif keyword == "ZION":
                    for n, spec in enumerate(species):
                        fout.write(f"{indent}{spec.Z},   // {n}\n")

                    for n, spec in enumerate(extra_species):
                        fout.write(f"{indent}{spec.Z},   // {n + len(species)}\n")

                elif keyword == "ZION_CONSTEXPR":
                    fout.write("\n")
                    for n, spec in enumerate(species):
                        fout.write(f"{indent}case {spec.short_name.capitalize()}:   // {n+1}\n")
                        fout.write(f"{indent}{{\n")
                        fout.write(f"{indent}    z = {spec.Z};\n")
                        fout.write(f"{indent}    break;\n")
                        fout.write(f"{indent}}}\n\n")

                    for n, spec in enumerate(extra_species):
                        fout.write(f"{indent}case {spec.short_name.capitalize()}:   // {n + len(species) + 1}\n")
                        fout.write(f"{indent}{{\n")
                        fout.write(f"{indent}    z = {spec.Z};\n")
                        fout.write(f"{indent}    break;\n")
                        fout.write(f"{indent}}}\n\n")

                elif keyword == "AUX_NAMES":
                    for n, aux in enumerate(aux_vars):
                        fout.write(f"{indent}\"{aux.name}\",   // {n} \n")

                elif keyword == "SHORT_AUX_NAMES":
                    for n, aux in enumerate(aux_vars):
                        fout.write(f"{indent}\"{aux.name}\",   // {n} \n")

                elif keyword == "SPECIES_ENUM":
                    for n, spec in enumerate(species):
                        if n == 0:
                            fout.write(f"{indent}enum NetworkSpecies : {enum_type(len(species+extra_species))} {{\n")
                            fout.write(f"{indent + 4*' '}{spec.short_name.capitalize()}=1,\n")
                        else:
                            fout.write(f"{indent + 4*' ' }{spec.short_name.capitalize()},\n")
                    if len(extra_species) > 0:
                        fout.write(f"{indent + 4*' '}NumberSpecies={species[-1].short_name.capitalize()},\n")
                    else:
                        fout.write(f"{indent + 4*' '}NumberSpecies={species[-1].short_name.capitalize()}\n")

                    for n, spec in enumerate(extra_species):
                        fout.write(f"{indent + 4*' '}{spec.short_name.capitalize()},\n")
                    if len(extra_species) > 0:
                        fout.write("{}NumberExtraSpecies={}-{},\n".format(indent + 4*' ',
                                                                          extra_species[-1].short_name.capitalize(),
                                                                          species[-1].short_name.capitalize()))
                        fout.write(f"{indent + 4*' '}NumberTotalSpecies={extra_species[-1].short_name.capitalize()}\n")
                    fout.write(f"{indent}}};\n")

                elif keyword == "AUXZERO_ENUM":
                    if aux_vars:
                        for n, aux in enumerate(aux_vars):
                            if n == 0:
                                fout.write(f"{indent}enum AuxIndices : {enum_type(len(aux_vars))} {{\n")
                                fout.write(f"{indent + 4*' '}i{aux.name.lower()}=0,\n")
                            else:
                                fout.write(f"{indent + 4*' '}i{aux.name.lower()},\n")
                        fout.write(f"{indent + 4*' '}NumberAux=i{aux_vars[-1].name.lower()}\n")
                        fout.write(f"{indent}}};\n")

            else:
                fout.write(line)

        print(" ")
        fout.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--header_template", type=str, default="",
                        help="C++ header template file name")
    parser.add_argument("--header_output", type=str, default="",
                        help="C++ header output file name")
    parser.add_argument("-s", type=str, default="",
                        help="network file name")
    parser.add_argument("--defines", type=str, default="",
                        help="and preprocessor defines that are used in building the code")

    args = parser.parse_args()

    write_network(args.header_template,
                  args.s,
                  args.header_output, args.defines)

if __name__ == "__main__":
    main()
