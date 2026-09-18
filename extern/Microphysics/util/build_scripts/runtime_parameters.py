#!/usr/bin/env python3

"""
A class that manages the runtime parameters.  The idea is that this
can be used by all of the parsing scripts to write the C++.
"""
import sys

class Param:
    """ the basic parameter class.  For each parameter, we hold the name,
        type, and default.  For some parameters, we also take a second
        value of the default, for use in debug mode (delimited via
        #ifdef AMREX_DEBUG)

    """

    def __init__(self, name, dtype, default,
                 cpp_var_name=None,
                 namespace=None,
                 namespace_suffix="",
                 skip_namespace_in_declare=False,
                 debug_default=None,
                 priority=0,
                 size=1,
                 in_namelist = False,
                 ifdef=None):

        self.name = name

        self.dtype = dtype
        if self.dtype == "character":
            self.dtype = "string"

        self.default = default
        self.size = size
        self.cpp_var_name = cpp_var_name
        if self.cpp_var_name is None:
            self.cpp_var_name = self.name

        self.priority = priority

        self.in_namelist = in_namelist

        if namespace is not None:
            self.namespace = namespace.strip()
        else:
            self.namespace = namespace

        self.namespace_suffix = namespace_suffix

        # if this is true, then we use the namespace when we read the var
        # (e.g., via ParmParse), but we do not declare the C++
        # parameter to be in a namespace
        self.skip_namespace_in_declare = skip_namespace_in_declare

        self.debug_default = debug_default

        if ifdef == "None":
            self.ifdef = None
        else:
            self.ifdef = ifdef

        if self.namespace is None or self.namespace == "" or self.skip_namespace_in_declare:
            self.nm_pre = ""
        else:
            self.nm_pre = f"{self.namespace}{self.namespace_suffix}::"

    def get_cxx_decl(self):
        """ get the C++ declaration """
        if self.dtype == "real":
            return "amrex::Real"
        if self.dtype == "string":
            return "std::string"
        if self.dtype == "bool":
            return "bool"

        return "int"

    def get_declare_string(self, with_extern=False):
        """this is the line that goes into, e.g., castro_declares.H included
        into Castro.cpp"""

        extern = ""
        if with_extern:
            extern = "extern "

        if self.dtype != "string":
            tstr = f"{extern}AMREX_GPU_MANAGED {self.get_cxx_decl()} {self.cpp_var_name}"
        elif self.dtype == "string":
            tstr = f"{extern}std::string {self.cpp_var_name}"
        else:
            sys.exit(f"invalid data type for parameter {self.name}")

        return f"{tstr};\n"

    def get_struct_entry(self, indent=4):
        """this is the line that goes into a struct that defines the
        runtime parameters"""

        ostr = ""

        val = self.default_format(debug=self.debug_default)

        # we can use an empty initialization list {} for empty strings
        if self.dtype == "string" and val.strip() == '""':
            val = ""

        if not self.debug_default is None:
            ostr += "#ifdef AMREX_DEBUG\n"
            ostr += f"{' '*indent}{self.get_cxx_decl()} {self.cpp_var_name}{{{val}}};\n"
            ostr += "#else\n"
            ostr += f"{' '*indent}{self.get_cxx_decl()} {self.cpp_var_name}{{{val}}};\n"
            ostr += "#endif\n"
        else:
            ostr += f"{' '*indent}{self.get_cxx_decl()} {self.cpp_var_name}{{{val}}};\n"

        return ostr

    def get_default_string(self):
        """this is the line that goes into, e.g., castro_declares.H included
        into Castro.cpp"""

        ostr = ""

        if not self.debug_default is None:
            ostr += "#ifdef AMREX_DEBUG\n"
            ostr += f"{self.nm_pre}{self.cpp_var_name} = {self.default_format(debug=True)};\n"
            ostr += "#else\n"
            ostr += f"{self.nm_pre}{self.cpp_var_name} = {self.default_format()};\n"
            ostr += "#endif\n"
        else:
            ostr += f"{self.nm_pre}{self.cpp_var_name} = {self.default_format()};\n"

        return ostr

    def get_query_string(self):
        """this is the line that queries the ParmParse object to get
        the value of the runtime parameter from the inputs file.
        This goes into, e.g., castro_queries.H included into Castro.cpp"""

        ostr = ""
        if self.is_array():
            # we need to create an amrex::Vector to read and then
            # copy into our managed array
            ostr += "\n"
            ostr += f"        amrex::Vector<{self.get_cxx_decl()}> {self.name}_tmp({self.size}, {self.default_format()});\n"
            ostr += f"        if (pp.queryarr(\"{self.name}\", {self.name}_tmp, 0, {self.size})) {{\n"
            ostr += f"            for (int n = 0; n < {self.size}; n++) {{\n"
            ostr += f"                {self.nm_pre}{self.cpp_var_name}[n] = {self.name}_tmp[n];\n"
            ostr += "            }\n\n"
            ostr += "        }\n\n"
        else:
            ostr += f"pp.query(\"{self.name}\", {self.nm_pre}{self.cpp_var_name});\n"

        return ostr

    def get_query_struct_string(self, struct_name="params", class_name=None):
        """this is the line that queries the ParmParse object to get
        the value of the runtime parameter from the inputs file.
        This is intended to use when we have a struct holding the runtime parameters,
        and will have the form class_name::struct_name.namespace.param"""

        if class_name is None:
            cname = ""
        else:
            cname = f"{class_name}::"

        ostr = ""
        if self.is_array():
            # we need to create an amrex::Vector to read and then
            # copy into our managed array
            ostr += "\n"
            ostr += f"        amrex::Vector<{self.get_cxx_decl()}> {self.name}_tmp({self.size}, {self.default_format()});\n"
            ostr += f"        if (pp.queryarr(\"{self.name}\", {self.name}_tmp, 0, {self.size})) {{\n"
            ostr += f"            for (int n = 0; n < {self.size}; n++) {{\n"
            ostr += f"                {cname}{struct_name}.{self.namespace}.{self.cpp_var_name}[n] = {self.name}_tmp[n];\n"
            ostr += "            }\n\n"
            ostr += "        }\n\n"
        else:
            ostr += f"pp.query(\"{self.name}\", {cname}{struct_name}.{self.namespace}.{self.cpp_var_name});\n"

        return ostr

    def default_format(self, *, lang="C++", debug=False):
        """return the value of the parameter in a format that it can be
        recognized in C++ code--in particular, preserve the quotes for
        strings

        """

        # note: lang is no longer used and will be removed once application
        # codes have synced

        if debug:
            val = self.debug_default
        else:
            val = self.default

        if self.dtype == "string":
            return f'{val}'
        if self.dtype == "bool":
            if val.strip() in ["1", "True", "TRUE", "true"]:
                return "true"
            return "false"
        if self.dtype == "real":
            if "d" in val:
                val = val.replace("d", "e")
            if not val.endswith("_rt"):
                val += "_rt"
        return val

    def get_job_info_test(self):
        """this is the output in C++ in the job_info writing"""

        value = self.default_format()
        if self.dtype == "string" and  value.strip() == '\"\"':
            test = f"{self.nm_pre}{self.cpp_var_name}.empty()"
        else:
            test = f"{self.nm_pre}{self.cpp_var_name} == {value}"

        ostr = (
            f'jobInfoFile << ({test} ? "    ": "[*] ") << "{self.namespace}.{self.cpp_var_name} = "' +
            f'<< {self.nm_pre}{self.cpp_var_name} << std::endl;\n')

        return ostr

    def is_array(self):
        """return true if the parameter is an array"""
        try:
            isize = int(self.size)
        except ValueError:
            return True

        if isize == 1:
            return False
        return True

    def __lt__(self, other):
        return self.priority < other.priority

    def __str__(self):
        return self.name
