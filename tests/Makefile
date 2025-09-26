config:
	cd ../build/gpu-3d/ && rm -rf ./* && cmake -B . -S ../.. -DCMAKE_BUILD_TYPE=Release -DAMReX_GPU_BACKEND=HIP -DAMReX_SPACEDIM=3 -G Ninja

com:
	cd ../build/gpu-3d/ && ninja -j16 ProdGalaxyTallBox

test:
	../build/gpu-3d/src/problems/ProdGalaxyTallBox/ProdGalaxyTallBox ../inputs/ProdGalaxyTallBox.in max_timesteps=2



QK := /home/chongchong/softwares-setonix/quokka
JOB := spherical_collapse 

comp:
	cd $(QK)/build/gpu-20250905 && cmake --build . -j 12 -t $(JOB)

run:
	cd ../build/gpu-20250905 && cmake --build . -j 12 -t $(JOB) && cd ../../tests/ && \
		for i in {1..4}; do \
			../build/gpu-20250905-off/src/problems/SphericalCollapse/spherical_collapse ../inputs/SphericalCollapse.in; \
		done

