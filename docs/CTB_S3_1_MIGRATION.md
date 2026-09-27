# 0.3.1 / S3.1 compatibility and migration

This patch repairs four software contracts. It does not validate a DFT engine or a material. Keep original 0.3.0 wheels, exchange journals, public archives, tests and reports unchanged.

| Identity | 0.3.1 rule |
| --- | --- |
| Adapter Python API / installed manifest | API 1 and `ctb.dft.adapter_manifest.v1` remain; describe/prepare/collect signatures and metadata-only discovery are unchanged. |
| Request / result wire | `ctb.dft.request.v2` and `ctb.dft.result.v2`; request now includes hash-bound `sampling`. Old v1 records are rejected, never implicitly upgraded. |
| External protocol / reduction recipe | `ctb.external_protocol.v2`, `ctb.external_reducers.v2`; requested-operation recipes created by exchange use v2. Native dependencies and actual protocol tolerances are locked. |
| External site / preparation / journal | Existing v1 envelopes remain. Journal resume already rejects a changed core/adapter identity, so 0.3.0 runs require a separate new run. |
| Electronic sampling | `ctb.electronic_sampling.v1`; exact grid/half-shift, reference reciprocal basis, actual full-grid coordinates and uniform weights. Irreducible returns are explicitly unsupported in this patch, even with self-reported expansion. |
| Direct phonon sampling | `ctb.external_phonon_sampling.v1`; exact final geometry, scope, basis, segments, ordered actual q rows, branch count, generator/settings/versions and digest. |
| StageCache / Budget / schedule | Existing versions unchanged. Persistent sample deadlines, fixed owner ledger_root and full EOS inner-relaxation signatures remain in force. No migration resets deadlines. |
| Existing MLFF IFC / sampling recipes | No identity or physical IFC cache invalidation. Path construction is extracted into a shared helper with unchanged numerical behavior. Q resampling still reuses IFC and requests no new forces. |
| External returned-force cache | Existing wrapper unchanged; v2 request/result and plugin-version digests naturally produce new external keys. No reuse of an unvalidated v1 result. |
| Public export audit | `ctb.public_export_audit.v2`, with separate `ctb.independent_public_postcheck.v1`. |

**Re-import is an explicit data operation, not an upgrade claim.** Create a new exchange directory with 0.3.1 and the compatible plugin. Supply original raw evidence to the parser or an updated external exporter, bound to the new request. The new coverage, geometry, method, artifact and quality checks must pass. Do not rewrite old request IDs/schema labels, infer missing k/q rows, or copy an old accepted flag into a new result. Missing evidence stays rejected/unknown; synthetic evidence remains synthetic. Threshold-only reassessment within a compatible run still needs no physical calculation.

ABACUS and the test-only second plugin are independently packaged at 0.1.1 with `crystargetbench>=0.3.1,<0.4`. ABACUS adds normalized sampling identity from the actual parsed k table; it has no new physical capability. The second plugin continues to consume supplied synthetic observations without returning scores or changing core. Existing third-party API-1 plugins must emit the v2 sampling evidence before their electronic/phonon data qualify.

Electronic coordinates use reciprocal row vectors including 2π, in the exact reference cell basis. Shift flag 1 means half a grid step. Each eigenvalue and occupation row supplies the same `kpoint_indices` permutation into the request's full mesh; coordinates and weights must agree. Full mesh reorderings are allowed, but missing, duplicate, shifted or inconsistently paired rows are rejected. Real fractional occupations can still produce zero gap after coverage checks; absent coverage cannot.

Direct path spectra use the actual Phonopy primitive basis recorded in the manifest and therefore 3×primitive-atom-count branches. Mesh spectra retain the full reference-cell basis and 3N reference branches. The manifest records the primitive-to-reference reciprocal transform. SeeK-path `get_path_orig_cell`, recipe hpkot and pinned dependencies generate the segments; `path_points_parameter` is the number of points **per segment**, preserving both endpoints. It is not the total number of q points. Returned rows must match the ordered manifest. Path+mesh tasks create two independent requests and cannot reuse one spectrum as the other. Numerical NAC remains unsupported.

Only the exact reviewed, unbound `ctb.s2.live_permit_template.1` disabled template is retained as a permission template, pinned by its canonical full-content SHA256. Added flags, filled bindings, changed schema/text or unknown permission records are excluded pending review. Both enabled and authorized flags, contradictory or unknown states, historical authorization forms and sessions fail closed. This is deliberately more conservative than preserving arbitrary dictionaries with enabled=false. Original private archives are never sanitized in place. Post-export inspection independently walks serialized JSON/YAML and embedded JSON, checks permission/session fields, and inventories source bytes; it is not a comprehensive secret or license audit.
