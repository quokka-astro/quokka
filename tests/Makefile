.PHONY: clean

# required on macOS, remove it on Linux
SHELL := /bin/zsh

JOB ?= SN
DIM ?= 3
JOBINI ?= ${JOB}.in
NP ?= 8
ARGS ?= tiny_profiler.enabled=0 suppress_output=1 amr.v=0

DEBUG :=
BD := build/clang-${DIM}d$(if $(DEBUG),-debug,)
MPI := 2

main:
	@echo ''
	@echo '  Targets:'
	@echo '    make config              configure the CMake build (run once)'
	@echo '    JOB=<name> make b        build a problem'
	@echo '    JOB=<name> make r        run a problem'
	@echo '    make clean               remove output files'
	@echo ''
	@echo '  Variables (with defaults):'
	@echo '    JOB=$(JOB)   DIM=$(DIM)   NP=$(NP)   DEBUG=$(DEBUG)'
	@echo ''

config:
	mkdir -p ../$(BD) && cd ../$(BD) && (ls CMakeCache.txt && rm -rf * || true) && cmake ../.. -DCMAKE_BUILD_TYPE=$(if $(DEBUG),Debug,Release) -DAMReX_SPACEDIM=$(DIM) -G Ninja

b:
	cd ../$(BD) && ninja -j$(NP) $(JOB)

r:
	for job in ${JOB}; do ../$(BD)/src/problems/$$job/$$job ../inputs/$$job.in $(ARGS) && echo "$$job Success" || (echo "$$job Failure" && exit 1); done

clean:
	--rm -rf plt*
	--rm -rf slice*
	--rm -rf chk*
	--rm -rf part_*
	--rm -f *.csv
	--rm -f *.pdf
	--rm -f *.png
	--rm -f Backtrace.*

fullclean:
	rm -rf *
	git checkout -- matplotlibrc
	git checkout -- Makefile

