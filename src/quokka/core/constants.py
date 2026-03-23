
from __future__ import annotations

import re

SCHEMA = 1
LOCK_TYPES = ('build', 'run')
TIDY_SELECTORS = {'changed', 'previous', 'origin', 'dev'}
FORMAT_SELECTORS = {'changed', 'previous', 'origin', 'dev', 'all'}
CLANG_TIDY_FILE_EXTENSIONS = ('.cpp', '.hpp')
CLANG_FORMAT_FILE_EXTENSIONS = ('.cpp', '.hpp', '.H')
SUBMODULE_PATHS = (
    'extern/amrex',
    'extern/AMReX-Hydro',
    'extern/Microphysics',
    'extern/yaml-cpp',
    'extern/turbulence',
)
IGNORED_PROFILE_DEFINES = frozenset({'AMReX_MPI'})
ENV_OVERRIDE_KEYS = (
    'CC',
    'CXX',
    'CMAKE_GENERATOR',
    'CMAKE_PREFIX_PATH',
    'CMAKE_C_COMPILER',
    'CMAKE_CXX_COMPILER',
    'CMAKE_CUDA_COMPILER',
)
DIAGNOSTIC_CODES = {
    'OK': 0,
    'USAGE_ERROR': 10,
    'UNKNOWN_PROFILE': 11,
    'UNKNOWN_RESOURCE': 12,
    'PROFILE_UNCONFIGURED': 13,
    'INPUT_REQUIRED': 14,
    'TEST_MAPPING_UNSUPPORTED': 15,
    'RUNTIME_DIR_UNSAFE': 16,
    'TIDY_SELECTOR_INVALID': 17,
    'FORMAT_SELECTOR_INVALID': 18,
    'PRE_COMMIT_UNAVAILABLE': 19,
    'RESOURCE_LOCKED': 20,
    'STALE_ARTIFACT': 21,
    'MISSING_ARTIFACT': 22,
    'CONFIGURE_DRIFT': 23,
    'EXECUTOR_UNAVAILABLE': 24,
    'TOOL_FAILED': 25,
    'STATE_CORRUPT': 26,
    'INTERNAL_ERROR': 30,
}
MAX_DIAGNOSTIC_OUTPUT_CHARS = 4000
EXPECTATION_COMMENT_RE = re.compile(r'^\s*//[/!<]*\s*QUOKKA_EXPECT:\s*(.+?)\s*$')
NINJA_PROGRESS_RE = re.compile(r'^\[(\d+)/(\d+)\]\s+(.*)$')
PYTHON_MODULE_PACKAGES = {
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'matplotlib.pyplot': 'matplotlib',
    'matplotlib.cm': 'matplotlib',
    'PIL': 'Pillow',
}
METRIC_KEYWORDS = (
    'relative error',
    'error norm',
    'error norms',
    'l1 error',
    'l2 error',
    'rms',
    'residual',
    'final temperature',
)
AMREX_TIMESTEP_BANNER_RE = re.compile(
    r'^(?P<prefix>\d+:\s+)?(?P<label>.*?\b)?STEP\s+(?P<step>\d+)\s+at\s+t\s*=\s*(?P<time>\S+)\s+\((?P<progress>[^)]+)\)\s+starts\s+\.\.\.\s*$'
)
