from __future__ import annotations

from pathlib import Path


def cmake_cache_path(build_dir: Path) -> Path:
    return build_dir / "CMakeCache.txt"


def ctest_root_testfile_path(build_dir: Path) -> Path:
    return build_dir / "CTestTestfile.cmake"


def compile_commands_path(build_dir: Path) -> Path:
    return build_dir / "compile_commands.json"


def buildtree_state(build_dir: Path) -> dict[str, bool]:
    cmake_cache_exists = cmake_cache_path(build_dir).exists()
    ctest_metadata_exists = ctest_root_testfile_path(build_dir).exists()
    return {
        "cmake_cache_exists": cmake_cache_exists,
        "ctest_metadata_exists": ctest_metadata_exists,
        "configured": cmake_cache_exists and ctest_metadata_exists,
        "partial_configure": cmake_cache_exists and not ctest_metadata_exists,
    }
