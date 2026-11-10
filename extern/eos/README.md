# EOS

`interfaces/` and `EOS/` hold the equation of state of [AMReX-Astrophysics Microphysics](https://github.com/AMReX-Astrophysics/Microphysics), written by the Microphysics development team: the `eos()` front end and the `gamma_law`, `photoionization` and `primordial_chem` backends. They are copied verbatim from the fork <https://github.com/chongchonghe/Microphysics-fork>, branch `james471/primordial-chem-no-using-namespace-amrex`, at commit `eba42a12`.

That code is covered by Microphysics' BSD 3-clause licence, in [`LICENSE`](LICENSE).

`shim/` is written for Quokka. It supplies the files the copy includes by name, which Microphysics generates at build time: the runtime parameters of each backend and the species table of each network. The shim is transitional: it goes when the EOS is ported to Quokka.
