.. Running on HPC clusters

Running on HPC clusters
=====

Instructions for running on various HPC clusters are given below.

Gadi (NCI Australia)
-----------------------

Use the ``openmpi/4.1.4`` module (or newer), and build with ``gcc/system`` or ``gcc/11.1.0``, and use ``cuda/11.7.0`` (or newer).

Using VisIt
^^^^^^^^^^^^^^^^^^^^^^^

You can use VisIt in client/server mode with the following server-side `patch for the
launcher script <https://gist.github.com/BenWibking/15645ff90819f2808fdb7a04b50b4a1e>`_.

A host file is provided `here <https://gist.github.com/BenWibking/5fa4d6d419dd0adf5da0435e5057b335>`_.
You must change the username, project code, and server-side VisIt path.

Setonix (Pawsey)
-----------------------

The recommended build procedure on Setonix is: ::
  
  source scripts/setonix.profile
  mkdir build; cd build
  cmake .. -C ../cmake/setonix.cmake
  make -j16

Then a single-node test job can be run with: ::

  cd ..
  sbatch scripts/setonix-1node.submit

Workaround for interconnect issues
^^^^^^^^^^^^^^^^^^^^^^^

If interconnect issues are observed, it is recommended to add the line ::

  export FI_CXI_RX_MATCH_MODE=software

to your job scripts.
