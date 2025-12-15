#!/bin/bash

set -e

cd /software/projects/pawsey0807/chongchong/setonix/quokka/build/gpu-3d
ninja -j12 SphericalCollapse
cd /software/projects/pawsey0807/chongchong/setonix/quokka/tests

# Extract the value after "max_timesteps =" from ../inputs/SphericalCollapse.in
max_timesteps=$(awk '/^[[:space:]]*max_timesteps[[:space:]]*=/{print $3}' ../inputs/SphericalCollapse.in)
echo "max_timesteps: ${max_timesteps}"

# set flag to date
flag=$(date +%Y%m%d%H%M%S)
echo "flag: ${flag}"
../build/gpu-3d/src/problems/SphericalCollapse/SphericalCollapse ../inputs/SphericalCollapse.in >> log.${flag}.r1
chk=chk$(printf "%05d" ${max_timesteps}) 
plt=plt$(printf "%05d" ${max_timesteps}) 
# mv ${chk} ${chk}.${flag}.r1
mv ${plt} ${plt}.${flag}.r1
../build/gpu-3d/src/problems/SphericalCollapse/SphericalCollapse ../inputs/SphericalCollapse.in >> log.${flag}.r2
# mv ${chk} ${chk}.${flag}.r2
mv ${plt} ${plt}.${flag}.r2

if diff -q ${plt}.${flag}.r1/Level_0/Cell_H ${plt}.${flag}.r2/Level_0/Cell_H; then
    echo "test passed"
else
    echo "test failed"
		/home/chongchong/quokka/extern/amrex/Tools/Plotfile/fcompare.gnu.x86-trento.ex ${plt}.${flag}.r1 ${plt}.${flag}.r2
    exit 1
fi

exit 0