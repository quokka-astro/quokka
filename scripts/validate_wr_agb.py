#!/usr/bin/env python3
"""
Wrapper to run the validation script placed in tests (copy) for repositories where tests/ is gitignored.
This file simply delegates to the tests/sn_mass_validation/validate_wr_agb.py script.
"""
import os
import sys

script = os.path.join(os.path.dirname(__file__), '..', 'tests', 'sn_mass_validation', 'validate_wr_agb.py')
script = os.path.abspath(script)
if not os.path.exists(script):
    print('Cannot find', script)
    sys.exit(1)

os.execv(sys.executable, [sys.executable, script] + sys.argv[1:])
