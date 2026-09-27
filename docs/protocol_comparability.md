# Protocol compatibility and migration

0.3.0 keeps task v2, existing CTB geometry hashes, measurement/assessment semantics, StageCache v1, Budget v2 and serial recipe schedule v1. S1/S2 archives and wheels are not migrated in place. EOS outer points still carry the complete effective inner-relaxation signature (R3); resumed samples retain their original wall deadline including interruptions (R2); planner/executor retain the same deterministic schedule (R1).

New deployment v3 permits no selected DFT engine; v2 still requires the explicit configured default. Old `vasp_atomate2` is not an alias for `abacus_example`. Two legacy test modules now explicitly load their original disabled deployment fixture where they exercise the old VASP/QE interface. Their originals and initial failing runs are preserved under the private S3 evidence directory. No tests were deleted, skipped or xfailed.

New wire records are `ctb.dft.*.v1`; external site, run and protocol lock have distinct v1 schemas. Adapter API version is 1. Request identity hashes exact ordered geometry, scientific method, selected adapter version, recipe parameters and actual dependencies. It contains no target threshold or generator label. The protocol comparison includes CTB/adapter/parser/recipe identity and the task; input labels, output paths and submission hashes do not enter the scientific method identity. Changing a threshold changes assessment/comparison identity but can reuse saved observations. Changing XC, engine, pseudopotential, spin/U/SOC, cutoff, mesh or definitions changes the method group.

The file-exchange journal verifies stored accepted results and raw hashes on resume. Same-request different-result content is rejected, not overwritten. Repeated identical results are no-ops. Collection can read a new partial batch without any old managed permit or monotonic clock. Its external cost remains unknown/null. A core version or installed adapter manifest change requires a new exchange run/reimport, retaining the old record. Old result records without the new recipe identity are rejected; they are never silently upgraded.

`ctb.external_force_cache.v1` wraps the unchanged StageCache envelope and binds validated returned results, adapter/method, pinned numerical versions, geometry/mapping, IFC parameters and upstream digests. Sampling a different q mesh reuses the same IFC but recomputes frequencies. Core Phonopy remains on angstrom/eV-per-angstrom/amu; ABACUS-native Bohr IFCs must not be imported as this representation. Numerical NAC is still unsupported in this bridge; electron-only NAC input validation never treats total permittivity as epsilon-infinity. DFT Born + MLFF IFC is mixed, not pure DFT.

Managed local calls use the existing Budget v2 with explicit execution_kind and positive max_dft_jobs, plus the existing overall/sample/per-request/time/scratch/evaluation/retry limits. Legacy S2 callers keep the zero-DFT restriction. The permit identity includes kind, executable hash and complete binding. Old/incomplete ledgers do not receive a new window: explicit new authorization/lineage is required, as before. Deleting a ledger is not a supported resume mechanism.

Portable exchange journals store `managed: null` even when requests were prepared for managed_local execution. Enabled permits remain with the local caller and are never required for later collection. Input-reference phonons request qualified same-geometry forces/stress and enforce stationarity before response expansion.

Each new wire request also binds `sample_geometry_id`, the original submitted geometry. The managed launcher uses this stable sample identity across relaxed/derived geometries; moving to a downstream geometry does not allocate a new sample deadline. Existing wire v1 development records without this field are rejected and must be re-prepared/re-imported, never upgraded in place.

For managed_local, the owner-bound absolute `ledger_root` persists across output directories. An explicit conflicting ledger-directory is rejected. Repeat submission under the identical permit and original sample ID therefore reuses the same persistent deadline even when `--output` changes.

## 0.3.1 superseding wire rules

Request/result records and external protocol/reducer identity advance to v2; adapter method API remains 1. The detailed [S3.1 migration rules](CTB_S3_1_MIGRATION.md) supersede the v1 wire descriptions above. Old exchanges are preserved and require separate validated re-import. Physical IFC/StageCache/Budget identities are not globally invalidated. Native/external symmetry executes the same effective locked parameters, including custom tolerances; target predicates do not enter the measurement cache key.

## 0.3.2 public preview

Runtime changes are limited to the patch version and truthful CLI help text for the existing external/local paths. Property definitions, thresholds, routing, wire v2 / adapter API 1, sampling, physical caches and Budget semantics are unchanged. Existing exchange resume checks bind the exact core version: keep 0.3.1 directories intact and create a new compatible directory for raw-evidence re-import, as in the S3.1 migration. Native compatible measurement caches remain reusable; a changed target with the same protocol can use the same explicit cache and a new output directory.

## 0.3.3 repository metadata candidate

This patch changes distribution metadata, documentation and the package version only. Scientific definitions, thresholds, adapter API 1 / wire v2, native measurement cache compatibility, StageCache, persistent Budget and R1–R3/S3R1–S3R4 behavior are unchanged. Keep previous exchanges and archives intact. Because exchange resume binds the exact core version, use a new directory and reparse/revalidate original raw evidence when moving from 0.3.2; do not relabel version fields or digests. Plugin implementations remain at 0.1.1. The version denotes a locally prepared candidate, not a published release.
