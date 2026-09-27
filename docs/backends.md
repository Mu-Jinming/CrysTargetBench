# Backends and installation boundaries

Core installs NumPy, ASE, spglib and JSON schema support. Native structure recognition needs no GPU or solver. `ctb doctor` reads metadata, not engines. Installed adapter discovery reads static distribution manifests; only the explicitly selected trusted plugin is imported.

The `numerical` extra pins Phonopy 2.38.2, SeeK-path 2.1.0 and SciPy 1.17.1. MatterSim uses `.[mlff]` in a CTB-owned environment, with a user-supplied model asset, exact lock and explicit bound permit. Existing relaxation/phonon/EOS recipes, serial scheduling, sample deadlines and transitive cache signatures are preserved. Do not copy an old live permit to a new batch. The disabled example in `CTB_S2_Addendum/examples/MLFF_LIVE_PERMIT.disabled.json` stays disabled.

`dft` is now an empty compatibility extra. Install an adapter independently, and install/maintain any DFT engine yourself outside CTB. The ABACUS example has no solver dependency and no bundled pseudopotentials/orbitals. Legacy `vasp_atomate2` and `qe` registry identifiers retain their planning-only meaning; they are not renamed to ABACUS and their old configurations remain readable. Deployment v3 defaults to `default_dft_provider: null`.

External site v1 (`ctb.dft.site.v1`) explicitly selects a file-exchange track. It is separate from the normal MLFF deployment. `ctb plan/evaluate --deployment site.json` uses the same SDK exchange path; strict MLFF/hybrid policies cannot be silently interpreted as this DFT site. For an external site, evaluate prepares files. For an explicitly enabled managed_local site, evaluate serially launches prepared requests and collects them, stopping when outputs are missing. Complete electronic/ionic response must be supplied by a capable plugin; absent ABACUS response is blocked_capability.

The optional `ctb dft run-local` launcher requires a bound method/adapter/owned executable hash and complete hard limits. It uses shell=False, process-group cancellation, persistent Budget v2, attempts reserved before launch, and zero retry tokens by default. A successful process return code does not establish SCF/ionic/response convergence. No queue or cluster orchestration is provided. Prepare/collect never import this launcher.

S3 tests run no real model or DFT engine. Existing historical MLFF live evidence remains in its original scope and is not rewritten or extended by parser tests. See the S3 report for software, contract, parser, live and science status separately.
