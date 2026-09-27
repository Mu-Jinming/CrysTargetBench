# DFT Python adapter API 1 / request-result wire v2

The authoritative types are `crystargetbench.dft.types`: DFTAdapter, CalculationRequest, PreparedJob, CalculationResult and AdapterManifest. Strict bounded cross-field validation is in `dft.validation`; JSON schemas alone cannot prove numeric quality. All records are JSON data. The existing CTB geometry snapshot/hash is reused.

## Independent installation

Register one entry point in `crystargetbench.dft_adapters`:

```toml
[project.entry-points."crystargetbench.dft_adapters"]
my_engine = "my_package.adapter:Adapter"
[tool.setuptools.package-data]
my_package = ["ctb_dft_adapter.json"]
```

Place `ctb_dft_adapter.json` alongside the entry-point module. The distribution name/version, EP provider_id, adapter identity/API, immutable family, per-quantity capabilities, supported engine versions, restrictions, parser/live status, and false science/eligibility flags must agree. Discovery reads installed metadata/files without importing plugins. Duplicate/reserved IDs, incompatible API and runtime describe mismatches are errors. Selection never depends on installation order. Task/site cannot grant capabilities. Trust a plugin before installing it.

Implement `describe()`, `prepare(request, directory, configuration)` and `collect(request, directory, configuration)`. Neither method may invoke an engine or access the network. Return quantities and evidence, never benchmark pass/fail/metrics. Tests demonstrate a completely separate distribution in `tests/plugins/second`; it only consumes authored synthetic data. Run the installed-plugin conformance suite with `pytest tests/adapter_conformance` after installing both example distributions.

## Requests, preparation and return records

Requests bind the complete reference geometry, method (XC, basis, grid/shift, cutoff, spin/SOC/U, occupation and convergence definitions, software and asset fingerprints), operation, requested quantities, recipe and real upstream digests. Hashes are computed by the SDK using canonical CTB identity. Do not insert thresholds or fake future relaxed geometries. The final method cannot contain default/latest/null placeholders.

A prepared job has ready, blocked_assets or unsupported status, original-atom permutation/basis mapping, and input artifact references with relative path, SHA256, byte size and role. Missing mounted assets may yield blocked_assets while still exporting INPUT/STRU/KPT. No fake asset is written. Execution readiness is distinct from file-exchange readiness.

Results repeat request/method/adapter/engine binding, include the observed final geometry, normalized original-atom mapping, explicit engine/SCF/ionic/response convergence, external or managed cost provenance, observations and errors. Every observation has quantity, value/null, unit, definition_id, calculation and quality status, evidence kind, diagnostics and hashed raw artifacts. Core validates before reducing to its existing measurement and calling assess/aggregate. Integrity validation is not independent authentication of an operator's claims.

| Quantity | Normalized representation |
| --- | --- |
| energy | scalar eV |
| forces | N × 3 eV/angstrom, original input atom order |
| stress | symmetric 3 × 3 eV/angstrom³, ASE tensile-positive Cartesian basis |
| eigenvalues / occupations | k × band, eV / unweighted per-state occupancy; paired exact k-point/weight metadata |
| electronic / ionic / total dielectric | symmetric 3 × 3 relative tensor; static fixed-strain definition, frequency 0, local-field treatment and geometry |
| born_charges | N × 3 × 3 in units of e, electronic response provenance |
| relaxed_geometry | full CTB snapshot, ionic convergence plus posterior F/stress checks |
| phonon_spectrum | THz; path: 3N of recorded Phonopy primitive; mesh: 3N of reference; ordered manifest-bound q rows, basis, scope and recipe |

Wire v2 direct mesh spectra require the full locked Gamma-centred grid with unit weights, not an unproven subset. Path is a different scope. A total tensor must explicitly identify electronic+ionic components. The trace is divided by 3, never by 9; supplied total is not summed twice. Optical/electronic-only data cannot qualify as total static response. Mixed upstream data cannot qualify a pure DFT task. Missing observations stay null.

Raw paths must be relative, within the returned request directory, not symlinks/traversal; each file and JSON record is capped at 16 MiB, arrays at one million elements, atoms at 4096, nesting at 40. Nonfinite, malformed, boolean numeric arrays and mismatched shapes/versions/methods/geometries are rejected. Portable JSON never contains executable task code.

## File exchange and CPU postprocessing

`ctb dft prepare --structures ... --task ... --site ... --output jobs` exports ready requests under jobs/requests/<request_id>. Copy each request directory, supply raw output and adapter receipt (or validated result.json), and return directories named by request ID. `ctb dft collect --run jobs --results returned` joins by ID/digest and expands only verified ready dependencies. Relaxed tasks initially export only relaxation. Idempotent batches cannot replace earlier results. Conflicts fail explicitly; missing output stays unknown.

`ctb dft reassess --run jobs --task changed-thresholds.json --output revised` uses saved measurements without engine calls. Measurements and reference definition cannot change in reassessment. For force-based external phonons, `dft.phonons.displacement_requests` exports actual Phonopy static displacement requests; `collect_phonons` validates returned forces, builds IFC with the existing pinned recipe, and samples it. It does not relax displaced cells or launch a calculator. This CPU helper is explicit and is not automatically substituted for a plugin's absent phonon_spectrum capability in high-level exchange planning.

Use `api.evaluate(..., deployment=external_site)` for the same path. The optional managed launcher is `dft.local.run_local` / `ctb dft run-local`; an enabled complete site binding is mandatory. External collect never touches its old budget. Plugins needing other engine formats implement their own parser behind this same contract, without core edits.

Input-reference phonon operations also request standardized forces/stress at that exact reference and reject a nonstationary reference as unknown. An unqualified prerequisite does not expand downstream response. Managed authorization records are excluded from portable exchange journals; they are supplied separately only to the launcher.

`sample_geometry_id` identifies the original submitted sample for persistent managed budgeting. Requests for its relaxed/derived geometries retain this identity. Standalone `make_request` defaults it to the request geometry; orchestrators must pass the original sample ID for derived requests. It is hash-bound in the request and is not a user target label.

The managed record requires a fixed absolute `ledger_root`. The optional CLI ledger-directory must match this bound site root; it is never derived from the evaluation output directory. This prevents repeated submissions with a new output path from allocating a fresh window.

## 0.3.1 wire v2 sampling requirements

The plugin method API stays version 1; request/result JSON is now **v2**. See [migration](CTB_S3_1_MIGRATION.md) for full compatibility rules. `CalculationRequest.sampling` contains authoritative, hash-bound coverage definitions. Missing v2 evidence is rejected by the common validator before reduction, for every plugin.

For electronic arrays, both diagnostics must include `sampling_manifest_digest`, `coordinate_convention`, `reciprocal_basis_Ainv`, `kpoint_indices`, actual `kpoints`, normalized `weights`, and the existing occupancy definition. The SDK's `electronic_diagnostics(manifest, actual_kpoints)` maps actual parsed points to request rows; it never fills absent bands. Row identities must form a full permutation. Irreducible sampling is explicitly unsupported until a verified expansion contract is implemented.

For `phonon_spectrum`, diagnostics must include the exact `phonon_parameters`, `sampling_manifest_digest`, `coordinate_convention` and `reciprocal_basis_Ainv`. Values contain ordered q rows/weights/frequencies matching the request manifest. Direct path branch count is 3N of its recorded **Phonopy primitive**, while direct mesh branch count remains 3N of the reference. Actual segments and the primitive-to-reference transform are recorded in the request; a scope label is not coverage evidence. Combined path+mesh tasks are split into independent scope requests after final geometry is known.

A runnable [synthetic SDK walkthrough](PREVIEW_SDK_WALKTHROUGH.md) uses the installed second plugin, staged CLI collection and the same assessor without any engine.
