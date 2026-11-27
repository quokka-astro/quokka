#!/usr/bin/env python3

import sys

import canary

canary.directives.keywords("quokka", "nightly", "regression", "gpu", "cuda", "radhydro", "shell", "radhydroshell_gpu")
canary.directives.parameterize("cpus,gpus", [(1, 1)])
canary.directives.timeout("4h")
canary.directives.link("../lib/quokka_canary.py")
canary.directives.copy(src="../../../inputs/radhydro_shell_regression.toml", dst="radhydro_shell_regression.toml")
canary.directives.link(src="../../../extern/dust_shell/initial_conditions.txt", dst="initial_conditions.txt")
canary.directives.baseline(flag="--rebaseline-benchmark")

CASE = {
    "test_name": "RadhydroShell-GPU",
    "problem_name": "RadhydroShell",
    "input_file": "radhydro_shell_regression.toml",
    "vis_var": "gasDensity",
    "ignore_return_code": False,
}


def main(argv=None) -> int:
    from quokka_canary import nightly_case_main

    return nightly_case_main(CASE, argv)


if __name__ == "__main__":
    sys.exit(main())
