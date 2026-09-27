# Release checks

1. Install core without GPU/DFT dependencies and verify metadata, `ctb --version`, source and wheel agree.
2. Run all original regression tests and adapter conformance without live markers. Verify R1/R2/R3 and native complete-denominator results.
3. Build and separately install each adapter wheel; compare core hashes before/after plugin install. Discovery must import neither plugin nor engine.
4. Run staged external prepare/collect and parser fixtures with honest evidence labels. Review tensor/geometry/method/budget/cache boundary tests.
5. Generate a new allowlisted public-export tree with recursive JSON/YAML/text sanitization. Keep private originals and historical archives unchanged; inspect file manifests, not just .gitignore.
6. Confirm third-party redistribution rights, owner author metadata and publication destination. CITATION.cff is intentionally draft until authors are confirmed. Do not invent a DOI or remote repository.
7. Manually choose whether to publish candidate software. No release check marks the protocol science_validated or publishes a High-K leaderboard.

`tools/s3_release_check.py` records installed-wheel/native/second-plugin and environment evidence. `tools/s3_public_export.py` builds the curated tree and compact source archive. CI performs equivalent no-engine checks. No command pushes or creates a remote.

The public exporter reads YAML with the optional `export` extra (`PyYAML==6.0.3`); JSON export does not require it. CI numerical dependencies also supply this pinned package.

S3.1 additionally requires the four preserved before/after probes, all original 497 nodes, new full/invalid k and q sampling cases, authorized/enabled export cases, independent post-export inspection, and installed public-tree regression. Archive 0.3.0 unchanged; verify request/result v2 migration notes before sharing a new exchange bundle.

For the 0.3.2 public preview, also execute README commands in a new HOME/cwd with non-editable wheels, confirm wheel-only extraction and target cache reuse, and retain all 540 original test nodes. `tools/preview_ci.py --source <public-tree> --output <evidence>` runs the no-engine local CI sequence from an independent public copy; it needs no private project wheelhouse. Optional `--dependencies` may point to a freshly downloaded ordinary public-package staging directory for offline repeatability. Hosted CI remains not_run until an owner-designated repository is separately authorized.

Use `tools/preview_export.py` for current preview outputs; the older S3 exporter keeps its historical scope. Deliver actual core/adapter wheels, sdist and public source ZIP, not only report paths. The delivery index records presence, byte size and SHA256 and distinguishes workspace files/bundled artifacts from external uploads. Owner metadata and scientific validation are separate from local-ready software status.
