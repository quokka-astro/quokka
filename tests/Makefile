# =============================================================================
# Code Validation Makefile
#
# AGENT WORKFLOW — to validate a problem named <JobName>:
#
#   (working directory: <RootPath>/tests/)
#
#   Step 1 — Build:
#     JOB=<JobName> make b
#     If you see "no such file or directory: ../build/**", run `make config` to configure once
#
#   Step 2 — Run:
#     JOB=<JobName> make r
#
#     Exit 0 = passed. Non-zero = failed. Prints "<JobName> Success/Failure"
#
# NOTE: JOB must match a directory under src/problems/. Input file is ../inputs/<JobName>.in
# =============================================================================

.PHONY: main config b r clean fullclean

# required on macOS; remove on Linux
SHELL := /bin/zsh

JOB     ?= ParticleSink
DIM     ?= 3
JOBINI  ?= ${JOB}.in
NP      ?= 8
ARGS    ?= tiny_profiler.enabled=0 suppress_output=1 amr.v=0

DEBUG   :=
BD      := build/clang-${DIM}d$(if $(DEBUG),-debug,)
MPI     := 2

main:
	@echo ''
	@echo '  Targets:'
	@echo '    JOB=<name> make b        build a problem'
	@echo '    JOB=<name> make r        run a problem'
	@echo '    make config              configure the CMake build (run once)'
	@echo '    make clean               remove output files'
	@echo ''
	@echo '  Variables (with defaults):'
	@echo '    JOB=$(JOB)   DIM=$(DIM)   NP=$(NP)   DEBUG=$(DEBUG)'
	@echo ''

# Configure the CMake build tree. Safe to re-run; wipes stale CMakeCache first.
config:
	mkdir -p ../$(BD) && cd ../$(BD) && (ls CMakeCache.txt && rm -rf * || true) && cmake ../.. -DCMAKE_BUILD_TYPE=$(if $(DEBUG),Debug,Release) -DAMReX_SPACEDIM=$(DIM) -G Ninja

# Build the executable for JOB. Exit code 0 = success.
b:
	cd ../$(BD) && ninja -j$(NP) $(JOB)

# Run the executable for JOB. Prints "<job> Success" or "<job> Failure".
# Exit code 0 = all jobs passed; non-zero = at least one failed.
# When JOB=<JobName>, this run `../build/clang-3d/src/problems/<JobName>/<JobName> ../inputs/<JobName>.in`
r:
	for job in ${JOB}; do ../$(BD)/src/problems/$$job/$$job ../inputs/$$job.in $(ARGS) && echo "$$job Success" || (echo "$$job Failure" && exit 1); done

# Build and run
br: b r

# Remove runtime output files produced by a test run.
clean:
	--rm -rf plt*
	--rm -rf slice*
	--rm -rf chk*
	--rm -rf part_*
	--rm -f *.csv
	--rm -f *.pdf
	--rm -f *.png
	--rm -f Backtrace.*

# Reset tests/ to a clean checkout state (removes all generated files).
fullclean:
	rm -rf *
	git checkout -- matplotlibrc
	git checkout -- Makefile
	git checkout -- test_restart_refine.sh
