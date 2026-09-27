# CTB public preview usage acceptance — 0.3.2

## Result and actual user paths

The supplied 0.3.1 closeout closes four specific repairs; its third-party subset runs are not described here as an independent full 540-test reproduction. This turn performed new local user acceptance. Candidate software is locally ready for owner-controlled trial, not a completed scientific leaderboard or a published release.

`USER_JOURNEY_RESULTS.json` records **33 real commands**, argv, dependencies, computation scope, disposable HOME/cwd, return code, stdout/stderr and checked outputs. The new non-editable environment used only supplied wheels and a fresh ordinary public-index dependency download staged outside the repository. The initial sandbox download failure and successful host-side ordinary-package download are retained; no model, solver or asset was downloaded. Core-only runtime has no plugin implementation, Torch, MatterSim or Phonopy installed/imported before plugin installation. Runtime scripts and resources were staged outside checkout; no source PYTHONPATH was used.

| Path | Actual result |
| --- | --- |
| Core wheel / version / summarize | CLI, package attribute and installed metadata agree at 0.3.2; summarize matches stored metrics. |
| Wheel-only README extraction | Original task/CIF resources copied using the actual README Python snippet, without checkout access from the runtime. |
| Real native quickstart | FCC/BCC → SG 225/229, N=2, pass=1, fail=1, unknown=0. |
| Duplicate + invalid | N=4, pass=2, fail=1, unknown=1; duplicate retained, invalid not counted as a material failure. |
| Target 225 → 229 | Item decisions flip pass/fail to fail/pass; native_calls=0, cache_hits=2 using the same compatible cache and a new output. |
| Separate plugins | Both 0.1.1 wheels install without core-byte changes; metadata discovery imports no adapter implementation. |
| ABACUS preparation | Null fingerprints block configuration. A clearly synthetic missing-asset checksum declaration exports INPUT/STRU/KPT with prepared.status=blocked_assets; no asset or engine exists/was created. |
| Existing second plugin | Authored synthetic staged relaxation→bands→metrics: N=4, pass=3, unknown=1. Missing/partial returns remain unknown, incomplete coverage and conflicting results reject, identical collection is idempotent. |
| Current High-K | blocked_capability, N=4, unknown=4, fail=0; all values null, no nominal/ML fallback. |

## Bounded changes from actual usage findings

The old README command failed in a fresh cwd because examples were assumed to exist in a checkout (recorded before exit 2). The corrected README includes executable wheel-only extraction and target/cache commands. ABACUS documentation used an obsolete core version and omitted the Cu-template versus Ar-native-fixture mapping warning; a real failed preparation caught the mismatch. It remains a valid error, with corrected documentation and an explicitly synthetic preparation fixture. The SDK main table now states the actual wire-v2 primitive-path/reference-mesh branch definitions. CLI help no longer says DFT execution is universally unavailable when existing external exchange and explicitly authorized local interfaces are present.

The only core code changes from 0.3.1 are `__version__` and CLI help text. No task definition, threshold, routing, sampling, scoring or cache/budget algorithm changed. A new patch avoids overwriting 0.3.1 runtime/distribution metadata. API 1 / wire v2 and both plugin versions remain compatible; exact core-version exchange locks require a new directory and verified raw-evidence re-import across the patch. Existing records remain unchanged.

## Tests, public source and delivery

Local public-source CI: **passed**. Public-tree suite: **544**; separately installed rebuilt-wheel suite: **544**. All **540 original nodes** are verified present; four additional public-use tests passed. No deletion, skip, xfail or lowered physical quality threshold is used. The local sequence installs from a public tree copy, tests installed core, builds core/plugins/sdist, independently installs and retests wheels, and exports/scans the public tree. No private project wheelhouse is required; fresh ordinary public-package wheels were staged only for network-independent repeatability. Hosted GitHub CI is **not_run**.

Original raw commands and private archives remain in place. The public exporter redacts sensitive paths and excludes active permissions/sessions, engines, assets and environments; its independent scan and byte checks are recorded separately. Only reviewed disabled templates and pure schema definitions are retained. Exported command logs may have private paths redacted; local originals preserve exact commands.

The initial public-tree run recorded 543 passed / 1 failed: `test_help_states_execution_boundary` still required the obsolete blanket “DFT execution remains unavailable” text. That same node now asserts external exchange and explicit local authorization, and rejects the obsolete text. Its original source and failing log are preserved under `docs/preview/initial-ci/`; no scientific assertion changed. `PRESERVATION.json` identifies this single changed original test file, the two core text/version changes, and unchanged historical archives.

`DELIVERY_INDEX.json` enumerates actual files with size, SHA256, source version and bundle membership. A path in a report is not an uploaded attachment. The delivery ZIP contains actual core/plugin wheels, sdist, public source ZIP, reports and the index; nothing has been uploaded to GitHub/PyPI. Public payload equality is checked separately from byte-for-byte reproducibility of independently rebuilt wheel metadata/timestamps.

## Capability and remaining owner decisions

See `CAPABILITY_MATRIX.json`: native is actually CPU-tested; MLFF relax/phonon/EOS retain only the recorded historical S2-B host scope, with no current live model run. ABACUS parser fragments are synthetic, not genuine reference outputs; reference/live/science flags remain false. Full electronic mesh is supported, irreducible grids are explicitly unsupported, and complete ABACUS High-K response is unimplemented. Generic response interchange does not supply absent physics.

No new MLFF/DFT/cluster invocation, model/asset download, remote creation, push, hosted CI, Release or PyPI upload occurred. `OWNER_METADATA_TODO.md` lists unconfirmed authors/citation, repository URL, security contact and publication choice. Those fields were not invented and did not stop local preparation. Threshold examples remain candidate choices, not universal material standards.
