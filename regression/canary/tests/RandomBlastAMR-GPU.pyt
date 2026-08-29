#!/usr/bin/env python3

import sys

import canary

canary.directives.keywords("quokka", "nightly", "regression", "gpu", "cuda", "hydro", "randomblast", "randomblast_amr", "randomblastamr_gpu")
canary.directives.parameterize("cpus,gpus", [(1, 1)])
canary.directives.timeout("4h")
canary.directives.link("../lib/quokka_canary.py")
canary.directives.copy(src="../../../inputs/RandomBlastAMR_regression.toml", dst="RandomBlastAMR_regression.toml")
canary.directives.link(src="../../../extern/cooling/CloudyData_UVB=HM2012_resampled.h5", dst="CloudyData_UVB=HM2012_resampled.h5")
canary.directives.link(src="../../../inputs/particles_stochastic_n100.txt", dst="particles_stochastic_n100.txt")
canary.directives.baseline(flag="--rebaseline-benchmark")

CASE = {
    "test_name": "RandomBlastAMR-GPU",
    "problem_name": "RandomBlast",
    "input_file": "RandomBlastAMR_regression.toml",
    "particle_types": "StochasticStellarPop_particles",
    "vis_var": "temperature",
    "ignore_return_code": False,
}


def main(argv=None) -> int:
    from quokka_canary import nightly_case_main

    return nightly_case_main(CASE, argv)


if __name__ == "__main__":
    sys.exit(main())
