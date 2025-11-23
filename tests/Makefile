quick_plot := /software/projects/pawsey0807/chongchong/setonix/quokka/scripts/python/quick_plot
MP := 12

t:
	@echo $(PAWSEY_HDF5_HOME)

config:
	cd ../build/gpu-3d/ && rm -rf ./* && cmake -B . -S ../.. -DCMAKE_BUILD_TYPE=Release -DAMReX_GPU_BACKEND=HIP -DAMReX_SPACEDIM=3 -DHDF5_ROOT=$(PAWSEY_HDF5_HOME) -G Ninja

com:
	cd ../build/gpu-3d/ && ninja -j${MP} ProdGalaxyTallBox
test:
	../build/gpu-3d/src/problems/ProdGalaxyTallBox/ProdGalaxyTallBox ../inputs/ProdGalaxyTallBox.in max_timesteps=2
gpu: com test

comcpu:
	cd ../build/cpu-3d/ && ninja -j12 ProdGalaxyTallBox
testcpu:
	../build/cpu-3d/src/problems/ProdGalaxyTallBox/ProdGalaxyTallBox ../inputs/ProdGalaxyTallBox.in max_timesteps=2
cpu: comcpu testcpu

pp:
	$(quick_plot) plt* -j ${MP} --dir z --width 0.98,kpc --par StochasticStellarPop_particles --kind slc \
	  && $(quick_plot) plt* -j ${MP} --dir z --width 0.98,kpc --par StochasticStellarPop_particles -f v

p:
	$(quick_plot) plt* -j ${MP} --dir x --kind slc && $(quick_plot) plt* -j ${MP} --dir x --kind slc -f v

config-cpu:
	cd ../build/cpu-3d/ && (rm -rf ./* | true) && cmake -B . -S ../.. -DCMAKE_BUILD_TYPE=Release -DAMReX_SPACEDIM=3 -G Ninja
com-cpu:
	cd ../build/cpu-3d/ && ninja -j12 ProdGalaxyTallBox
test-cpu:
	../build/cpu-3d/src/problems/ProdGalaxyTallBox/ProdGalaxyTallBox ../inputs/ProdGalaxyTallBox.in max_timesteps=10

config-d:
	cd ../build/debug-3d/ && (rm -rf ./* | true) && cmake -B . -S ../.. -DCMAKE_BUILD_TYPE=Debug -DAMReX_SPACEDIM=3 -G Ninja
com-d:
	cd ../build/debug-3d/ && ninja -j12 ProdGalaxyTallBox
test-d:
	../build/debug-3d/src/problems/ProdGalaxyTallBox/ProdGalaxyTallBox ../inputs/ProdGalaxyTallBox.in max_timesteps=2

binary-d:
	cd ../build/debug-3d/ && ninja -j12 binary_orbit && cd ../../tests/ && ../build/debug-3d/src/problems/BinaryOrbitCIC/binary_orbit ../inputs/BinaryOrbit.in max_timesteps=10


QK := /home/chongchong/softwares-setonix/quokka
JOB := spherical_collapse 

comp:
	cd $(QK)/build/gpu-20250905 && cmake --build . -j 12 -t $(JOB)

run:
	cd ../build/gpu-20250905 && cmake --build . -j 12 -t $(JOB) && cd ../../tests/ && \
		for i in {1..4}; do \
			../build/gpu-20250905-off/src/problems/SphericalCollapse/spherical_collapse ../inputs/SphericalCollapse.in; \
		done

clean:
	rm -rf plt* chk* slice* proj* part*
