# Synthetic staged SDK walkthrough

This demonstrates file exchange and scoring using the existing **test-only second plugin**. All energies, forces and bands are authored synthetic values. Nothing here runs a solver, validates a material, or qualifies a DFT task. Python adapter API is 1; request/result wire is v2. Core 0.3.2 and plugins 0.1.1 are the supplied preview artifacts.

From the supplied public source directory, after installing core and the separate `ctb_second_example-0.1.1-py3-none-any.whl`:

```sh
ctb dft adapters
python tools/preview_demo.py init --output demo
ctb dft prepare --structures demo/ideal_fcc.cif demo/ideal_bcc.cif demo/ideal_fcc.cif demo/invalid.cif --task demo/task.json --site demo/site.json --output demo/run
ctb dft collect --run demo/run
python tools/preview_demo.py return --run demo/run --output demo/returned
ctb dft collect --run demo/run --results demo/returned
python tools/preview_demo.py return --run demo/run --output demo/returned
ctb dft collect --run demo/run --results demo/returned
ctb summarize demo/run/evaluation
```

The first preparation creates only relaxation requests at actual input geometries. Collection with no return leaves N=4 and unknown=4. The first supplied synthetic batch validates final geometries and creates downstream band requests. The second batch completes full-mesh bands: N=4, pass=3, fail=0, unknown=1. The duplicated FCC input stays in N; the invalid entry stays unknown. Inspect `exchange.json`, `requests/*/{request,prepared}.json`, `received/` and `evaluation/{results,metrics,run}.json`.

Use `return --limit 1` to author only one pending return. Missing batches remain pending/unknown; no zero gap is inserted. Missing or duplicate k rows are rejected (CLI exit 2); they are not a metallic result. Recollecting identical bytes is idempotent; changing an already accepted request's result raises a conflict instead of replacing it. An external blocked plan returns exit 3. Exit 0 on a ready file-exchange plan does not mean its user-side engine or assets are installed.

For ABACUS, use its separate wheel and [site template/instructions](../examples/dft_adapters/abacus/README.md). An unresolved/null asset fingerprint blocks configuration. With exact user-provided fingerprints but absent local files, INPUT/STRU/KPT may be exported with `prepared.status=blocked_assets`; CTB does not create an asset. Actual engine installation and licensed assets belong to the user. Complete High-K remains blocked because this example implements neither electric response nor a direct phonon-spectrum capability. The generic response contract does not supply those missing quantities.

To bring old results forward, follow [wire v2 migration](CTB_S3_1_MIGRATION.md): preserve the old directory, prepare a new compatible run and re-parse/revalidate original raw evidence. Never replace schema/digest fields to pretend a missing k/q row was measured. The existing core-version guard also requires a new exchange directory when moving from 0.3.1 to preview 0.3.2; property definitions, sampling schemas, budget and physical cache rules are unchanged.
