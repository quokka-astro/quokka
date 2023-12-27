.. TwoMomentRad documentation master file, created by
   sphinx-quickstart on Fri Feb  7 18:02:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QUOKKA
========================================

**Quadrilateral, Umbra-producing, Orthogonal, Kangaroo-conserving Kode for Astrophysics!**

Quokka is a two-moment radiation hydrodynamics code that uses the piecewise-parabolic method,
with AMR and subcycling in time. Runs on CPUs (MPI+vectorized) or NVIDIA GPUs (MPI+CUDA) with a single-source codebase.
Written in C++17. (100% Fortran-free.)

.. note::
   The Quokka methods paper is now `available on arXiv <https://arxiv.org/abs/2110.01792>`_.

We use the AMReX library :cite:`AMReX_JOSS` to provide patch-based adaptive mesh functionality.
We take advantage of the C++ loop abstractions in AMReX in order to
run with high performance on either CPUs or GPUs.

Example simulation set-ups are included in the GitHub repository for many astrophysical
problems of interest related to star formation and the interstellar medium.

.. _contact:

Contact
^^^^^^^

All communication takes place on the `Quokka GitHub repository <https://github.com/quokka-astro/quokka>`_.
You can start a `Discussion <https://github.com/quokka-astro/quokka/discussions>`_ for technical support, or open an `Issue <https://github.com/quokka-astro/quokka/issues>`_ for any bug reports.


.. raw:: html

   <style>
   /* front page: hide chapter titles
    * needed for consistent HTML-PDF-EPUB chapters
    */
   section#user-guide,
   section#developer-guide {
       display:none;
   }
   </style>

.. toctree::
   :hidden:

   about
   equations
   bibliography

User Guide
------------
.. toctree::
   :caption: USER GUIDE
   :maxdepth: 1
   :hidden:

   installation
   running_on_hpc_clusters
   tests/index
   parameters
   analysis
   instability

Developer Guide
-----------
.. toctree::
   :caption: DEVELOPER GUIDE
   :maxdepth: 1
   :hidden:

   flowchart
   debugging
   error_checking
   performance
   howto_clang_tidy
   api
   
