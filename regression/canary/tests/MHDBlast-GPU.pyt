#!/usr/bin/env python3

import sys

import canary

canary.directives.keywords("quokka", "nightly", "regression", "gpu", "cuda", "mhd", "mhdblast", "mhdblast_gpu")
canary.directives.parameterize("cpus,gpus", [(1, 1)])
canary.directives.timeout("4h")
canary.directives.link("../lib/quokka_canary.py")
canary.directives.copy(src="../../../inputs/MHDBlast.toml", dst="MHDBlast.toml")
canary.directives.baseline(flag="--rebaseline-benchmark")

CASE = {
    "test_name": "MHDBlast-GPU",
    "problem_name": "MHDBlast",
    "input_file": "MHDBlast.toml",
    "particle_types": "tracer_particles",
    "vis_var": "gasDensity",
    "ignore_return_code": False,
}


def main(argv=None) -> int:
    from quokka_canary import nightly_case_main

    return nightly_case_main(CASE, argv)


if __name__ == "__main__":
    sys.exit(main())
