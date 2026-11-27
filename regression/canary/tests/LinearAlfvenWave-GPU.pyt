#!/usr/bin/env python3

import sys

import canary

canary.directives.keywords("quokka", "nightly", "regression", "gpu", "cuda", "mhd", "alfven", "linear_alfven_wave", "linear_alfven_wave_gpu")
canary.directives.parameterize("cpus,gpus", [(1, 1)])
canary.directives.timeout("4h")
canary.directives.link("../lib/quokka_canary.py")
canary.directives.copy(src="../../../inputs/alfven_wave_linear_regression.toml", dst="alfven_wave_linear_regression.toml")
canary.directives.baseline(flag="--rebaseline-benchmark")

CASE = {
    "test_name": "LinearAlfvenWave-GPU",
    "problem_name": "AlfvenWaveLinear",
    "input_file": "alfven_wave_linear_regression.toml",
    "vis_var": "z-BField",
    "ignore_return_code": False,
}


def main(argv=None) -> int:
    from quokka_canary import nightly_case_main

    return nightly_case_main(CASE, argv)


if __name__ == "__main__":
    sys.exit(main())
