"""core functions for dealing with a network file"""

import sys


class Species:
    """the species class holds the properties of a single species"""
    def __init__(self):
        self.name = ""
        self.short_name = ""
        self.A = -1
        self.Z = -1
        self.is_extra = 0

    def __str__(self):
        return f"species {self.name}, (A,Z) = {self.A},{self.Z}"


class AuxVar:
    """convenience class for an auxiliary variable"""
    def __init__(self):
        self.name = ""
        self.preprocessor = None

    def __str__(self):
        return f"auxiliary variable {self.name}"


class UnusedVar:
    """this is what we return if an Aux var doesn't meet the
    preprocessor requirements"""
    def __init__(self):
        pass


def get_next_line(fin):
    """get_next_line returns the next, non-blank line, with comments
    stripped"""
    line = fin.readline()

    pos = str.find(line, "#")

    while (pos == 0 or str.strip(line) == "") and line:
        line = fin.readline()
        pos = str.find(line, "#")

    line = line[:pos]

    return line


def get_object_index(objs, name):
    """look through the list and returns the index corresponding to the
    network object (species or auxvar) specified by name

    """

    index = -1

    for n, o in enumerate(objs):
        if o.name == name:
            index = n
            break

    return index


def parse(species, extra_species, aux_vars, net_file, defines):
    """parse read all the species listed in a given network
    inputs file and adds the valid species to the species list

    """

    err = 0

    try:
        f = open(net_file)
    except OSError:
        sys.exit(f"network_param_file.py: ERROR: file {net_file} does not exist")

    line = get_next_line(f)

    while line and not err:

        fields = line.split()

        # read the species or auxiliary variable from the line
        net_obj, err = parse_network_object(fields, defines)
        if net_obj is None:
            return err

        if isinstance(net_obj, UnusedVar):
            line = get_next_line(f)
            continue

        objs = species
        if isinstance(net_obj, AuxVar):
            objs = aux_vars
        if isinstance(net_obj, Species):
            if net_obj.is_extra == 1:
                objs = extra_species

        # check to see if this species/auxvar is defined in the current list
        index = get_object_index(objs, net_obj.name)

        if index >= 0:
            print(f"write_network.py: ERROR: {net_obj} already defined.")
            err = 1
        # add the species or auxvar to the appropriate list
        objs.append(net_obj)

        line = get_next_line(f)

    return err


def parse_network_object(fields, defines):
    """parse the fields in a line of the network file for either species
    or auxiliary variables.  Aux variables are prefixed by '__aux_' in
    the network file. Extra species (that do not participate in the actual
    RHS, but are used for constructing intermediate rates) are prefix by
    '__extra_' in the network file.

    """

    err = 0

    # check for aux variables first
    if fields[0].startswith("__aux_"):
        ret = AuxVar()
        ret.name = fields[0][6:]
        # we can put a preprocessor variable after the aux name to
        # require that it be set in order to define the auxiliary
        # variable
        try:
            ret.preprocessor = fields[1]
        except IndexError:
            ret.preprocessor = None

        # we can put a preprocessor variable after the aux name to
        # require that it be set in order to define the auxiliary
        # variable
        try:
            ret.preprocessor = fields[1]
        except IndexError:
            ret.preprocessor = None

        # if there is a preprocessor attached to this variable, then
        # we will check if we have defined that
        if ret.preprocessor is not None:
            if f"-D{ret.preprocessor }" not in defines:
                ret = UnusedVar()

        # we can put a preprocessor variable after the aux name to
        # require that it be set in order to define the auxiliary
        # variable
        try:
            ret.preprocessor = fields[1]
        except IndexError:
            ret.preprocessor = None

        # if there is a preprocessor attached to this variable, then
        # we will check if we have defined that
        if ret.preprocessor is not None:
            if f"-D{ret.preprocessor }" not in defines:
                ret = UnusedVar()

    # check for missing fields in species definition
    elif not len(fields) == 4:
        print(" ".join(fields))
        print("write_network.py: " +
              "ERROR: missing one or more fields in species definition.")
        ret = None
        err = 1
    else:
        ret = Species()

        if fields[0].startswith("__extra_"):
            ret.name = fields[0][8:]
            ret.is_extra = 1
        else:
            ret.name = fields[0]
        ret.short_name = fields[1]
        ret.A = float(fields[2])
        ret.Z = float(fields[3])

    return ret, err
