# ABACUS file-exchange adapter example

Install from this directory with `python -m pip install .` after installing CTB >=0.3.1,<0.4 (candidate core: 0.3.3; adapter: 0.1.1). This installs only CTB-authored Python conversion code and its static manifest. It does not contain, install, discover, or run ABACUS or assets. Provider ID: `abacus_example`.

The candidate profile is **ABACUS 3.7.4, PW/PBE, nspin=1, no SOC/U, fixed occupations, symmetry=0, full uniform Gamma grid**. Its authored parser fragments are informed by fixed official documentation. No real 3.7.4 output has been acquired and no live calculation is claimed. The versioned documentation itself includes older example output, so format compatibility with an actual build still requires site validation. LCAO, orbital preparation, magnetic/SOC/U methods, reduced k grids, smearing, DFPT, electronic/Born/ionic response and automatic High-K execution are unsupported. Nothing is inferred from latest documentation.

## External prepare and collect

Copy `site.disabled.json` to a private site file. Fill all scientific values, exact pseudopotential filenames and SHA256 fingerprints. Leave execution_mode=external. `path` can be null on a file-exchange host if fingerprints are known. A placeholder/null fingerprint produces a blocked plan rather than a final scientific lock.

Asset mapping keys must cover exactly the input species, and each fingerprint identifier must match `SPECIES:filename`. The template uses Cu; the bundled ideal FCC/BCC native fixtures use Ar. Copying Cu keys unchanged for those fixtures correctly raises `asset mapping must cover exactly the chemical species`. Do not invent a physical fingerprint to bypass this check. A synthetic missing-asset declaration in the preview test verifies file preparation only and remains blocked_assets.

```sh
ctb dft prepare --structures crystal.cif --task gap-task.json --site site.json --output jobs
```

Each ready frontier directory contains request.json, prepared.json, INPUT, STRU, KPT and mapping.json. Assets are never created or copied. The execution site must stage its legally held exact files at `assets/<filename>` and verify each hash. If assets are absent, preparation is `blocked_assets` even though files can be exported. Use the SDK prepare again on the execution host after staging to update readiness. Static/eigenvalue and all displacement-force requests use SCF, never relaxation; only a relaxation request uses cell-relax.

Run your own installed ABACUS outside CTB. Return the request directory containing original prepared inputs, `OUT.CTB/running_scf.log` or `running_cell-relax.log`, and final STRU (whose relative filename is recorded). Eigenvalue requests also need `OUT.CTB/istate.info` and the actual direct k-point/weight table in the log. Provide an operator receipt `collection.json`:

```json
{
  "schema_version": "ctb.abacus.receipt.v1",
  "request_digest": "COPY_EXACT_REQUEST_DIGEST",
  "effective_method": {"COPY": "the actual complete method record, identical to request or reject"},
  "input_files": [],
  "evidence_kind": "reported",
  "final_structure": "OUT.CTB/STRU_FINAL",
  "ionic_converged": true
}
```

`input_files` must be the complete hashed prepared INPUT/STRU/KPT/mapping.json references, not an empty list. The example above is explanatory, not an executable result fixture. Fill ionic_converged from the actual convergence evidence; it cannot be inferred from exit code zero. A receipt is explicitly external-reported provenance, not independent authentication. The parser checks exact prepared contents, logged version/cutoff/spin, completion/SCF records, arrays, final geometry, raw hashes and posterior force/stress tolerances. Operators must confirm actual effective settings: ABACUS OUT/INPUT alone may describe initial/default settings rather than every effective runtime choice.

```sh
ctb dft collect --run jobs --results returned --output evaluation
```

Incomplete output remains unknown. A converged metallic occupation pattern can produce gap 0; missing bands/occupations cannot. Eigenvalues are evaluated only on the bound full uniform grid. Atom grouping is reversed into CTB's original order; tensor basis stays original Cartesian. STRU lattice scale is Bohr, CTB coordinates are angstrom. Force is eV/angstrom. ABACUS compressive-positive kbar stress becomes ASE tensile-positive eV/angstrom³ (minus sign and unit conversion).

## Optional user-owned local call

`ctb evaluate --deployment site.json ...` can run the same staged loop only after you explicitly choose managed_local and replace the disabled managed record with a complete local binding. The record requires authorization_id, method_digest, adapter_id, fixed absolute argv, executable SHA256 and every Budget hard limit, including positive max_dft_jobs. The user program/wrapper must supply the same output/receipt contract. Alternatively call `ctb dft run-local --request-directory ... --site ... --ledger-directory ...` then collect. CTB does not install your program, fill unresolved assets, enable templates or dispatch a cluster job.

See [support matrix](../../../docs/ABACUS_SUPPORT_MATRIX.json), [parser evidence](../../../docs/ABACUS_PARSER_EVIDENCE.md) and [SDK](../../../docs/DFT_ADAPTER_API.md). The explicit returned-force Phonopy helper reuses CTB's angstrom/eV-per-angstrom implementation; do not substitute native ABACUS Bohr-unit IFCs. The helper is separate from high-level task exchange; unsupported electric response stays blocked_capability.

Managed sites must also bind an absolute `ledger_root`, shared across output directories. CLI `--ledger-directory`, when provided, must match it. Changing `--output` does not create a new budget; relaxed/derived requests retain the original `sample_geometry_id` deadline. Portable journals omit the enabled managed record, so later collect needs no old permit.
