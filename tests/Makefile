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
	@echo '  Code Validation Makefile  (working directory: <RootPath>/tests/)'
	@echo ''
	@echo '  Workflow — to validate a problem named <JobName>:'
	@echo '    Step 1:  make config              # configure (once, or after CMake changes)'
	@echo '    Step 2:  JOB=<JobName> make b     # build'
	@echo '    Step 3:  JOB=<JobName> make r     # run'
	@echo ''
	@echo '  Exit 0 = passed. Non-zero = failed. Prints "<JobName> Success/Failure".'
	@echo '  JobName must match a directory under src/problems/.'
	@echo ''
	@echo '  Other targets:'
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
r:
	for job in ${JOB}; do ../$(BD)/src/problems/$$job/$$job ../inputs/$$job.in $(ARGS) && echo "$$job Success" || (echo "$$job Failure" && exit 1); done

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
