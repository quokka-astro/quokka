.PHONY: clean

SHELL := /bin/zsh

JOB ?= ParticleSink ParticleStar
NP ?= 9
DIM := 3
DEBUG :=
CC := clang
BD := build/$(CC)-$(DIM)d$(if $(DEBUG),-debug,)
ARGS := tiny_profiler.enabled=0 suppress_output=0 amr.v=0 ignore_return=true
# ARGS2 := tiny_profiler.enabled=0 suppress_output=0 amr.v=1 max_timesteps=4 plotfile_interval=2 checkpoint_interval=2 restartfile=chk0000002
MPI := 2

echo:
	echo "JOB: $(JOB)"

main:
	echo 'main does nothing'

rb:
	mkdir -p ../$(BD) && cd ../$(BD) && (ls CMakeCache.txt && rm -rf * || true) && cmake ../.. -DCMAKE_BUILD_TYPE=$(if $(DEBUG),Debug,Release) -DAMReX_SPACEDIM=$(DIM) -G Ninja

b:
	cd ../$(BD) && ninja -j$(NP) $(JOB)

r:
	for job in ${JOB}; do ../$(BD)/src/problems/$$job/$$job ../inputs/$$job.in $(ARGS) && echo "$$job Success" || (echo "$$job Failure" && exit 1); done

