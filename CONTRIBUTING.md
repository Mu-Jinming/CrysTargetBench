# Contributing

Use [Issues](https://github.com/Mu-Jinming/CrysTargetBench/issues) for ordinary bugs, feature discussions and reproducible usage questions. Propose changes through a branch or fork and a [pull request](https://github.com/Mu-Jinming/CrysTargetBench/pulls), preserving the repository history. Include the relevant tests and evidence scope in the PR. Sensitive vulnerabilities follow [SECURITY.md](SECURITY.md), not a public issue with exploit details.

Use a fresh Python environment. Install `.[dev,numerical]`, then independently install `examples/dft_adapters/abacus` and `tests/plugins/second`. Run `python -m pytest` without engines. Add an adapter through the documented entry-point API and static manifest; never add an engine import to the core registry.

Tests must distinguish synthetic/analytic, reference, reported and live evidence. Include format/version/source/license provenance for parser fixtures. Do not submit engine binaries/source bundles, model weights, pseudopotentials, orbitals, private structures, enabled permits, full sessions or credentials. No live test runs by default. Do not weaken thresholds, delete tests, skip or xfail a counterexample to obtain a pass.

Changes to scheduling, persistent deadlines and effective nested cache signatures require the R1–R3 regression tests. Changes to definitions/units/versions need compatibility notes. Include full-submission coverage and unknown cases, not only successful samples. Candidate software conformance does not establish scientific leaderboard eligibility.

Before release, build and install the wheel in isolation, test native input→space group→metrics, install an external plugin without editing core, run the public allowlist exporter, and review its audit. Owner author/DOI metadata must be confirmed before final citation metadata is published. This workspace's private history is not the public repository.
