#SHELL := /bin/bash
MK_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
QK := /home/chongchong/softwares-setonix/quokka
quick_plot := /software/projects/pawsey0807/chongchong/setonix/quokka/scripts/python/quick_plot 
JOB := ProdGalaxyTallBox
MP := 12
EXE := $(QK)/build/gpu-3d/src/problems/ProdGalaxyTallBox/ProdGalaxyTallBox
INI := $(QK)/inputs/ProdGalaxyTallBox.in


pltden:
	$(quick_plot) 00sim/plt* -j ${MP} --dir x --par StochasticStellarPop_particles --kind prj \
		--skip_existing -o ./00fig/x-colden-with-p --figsize 8 --p_size 40 --max_snapshots 200000 \
		&& $(quick_plot) 00sim/plt* -j ${MP} --dir x --kind prj \
		--skip_existing -o ./00fig/x-colden-no-p --figsize 8 --p_size 40 --max_snapshots 200000

pltslc:
	$(quick_plot) 00sim/slicex* -j ${MP} --kind slc -f v \
		--skip_existing -o ./00fig/x-v --figsize 8 --p_size 40 --max_snapshots 100000 && \
	$(quick_plot) 00sim/slicex* -j ${MP} --kind slc -f T \
		--skip_existing -o ./00fig/x-v --figsize 8 --p_size 40 --max_snapshots 100000 && \
	$(quick_plot) 00sim/slicex* -j ${MP} --kind slc -f n \
		--skip_existing -o ./00fig/x-v --figsize 8 --p_size 40 --max_snapshots 100000

pltrad:
	for i in {0..4}; do $(quick_plot) 00sim/slicex* -j ${MP} -f radEnergy-Group$${i} \
		--skip_existing -o ./00fig/x-rad$${i} --figsize 8 --p_size 40 --max_snapshots 200000; done

plt: pltslc pltrad pltden

zip:
	rm -f 00fig.zip && zip -r 00fig.zip 00fig

anim:
	for f in 00fig/x-colden-no-p 00fig/x-colden-with-p; do animateit -d $$f -o $${f}.mp4 -f 60; done
anim-30:
	for f in 00fig/x-colden-no-p 00fig/x-colden-with-p; do animateit -d $$f -o $${f}-30fps.mp4 -f 30; done
anim-small:
	for f in 00fig/x-colden-no-p 00fig/x-colden-with-p; do animateit -d $$f -o $${f}.mp4 --small -f 60; done

