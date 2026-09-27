# CrysTargetBench (CTB)

CTB evaluates **submitted crystal structures against explicit property tasks**. It preserves every submitted item, reports coverage and unknown results, and records the scientific method separately from the target threshold. Version 0.3.3 is candidate benchmark software; its example thresholds are not universal or scientifically validated standards.

## Install and run without a GPU or DFT engine

Project repository: [Mu-Jinming/CrysTargetBench](https://github.com/Mu-Jinming/CrysTargetBench).
Report ordinary bugs and usage questions through [Issues](https://github.com/Mu-Jinming/CrysTargetBench/issues); see [SECURITY.md](SECURITY.md) before reporting sensitive information.

Python 3.11 or newer. This 0.3.3 candidate is prepared locally; its source import and publication have not occurred. After the reviewed source import is merged, install from the repository:

```sh
git clone https://github.com/Mu-Jinming/CrysTargetBench.git
cd CrysTargetBench
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

The current local public-source tree also supports `python -m pip install .`. No PyPI availability is claimed.
For wheel-only use, place the supplied wheel in a new directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install ./crystargetbench-0.3.3-py3-none-any.whl
ctb --version
```

Copy original examples out of the installed package. This works without a checkout and invokes no backend:

```sh
python - <<'PY'
from importlib.resources import files
from pathlib import Path

root = Path('ctb-examples')
root.mkdir()  # Refuse to overwrite user files; use a fresh directory.
source = files('crystargetbench').joinpath('resources/examples')
for folder, names in {
    'structures': ['ideal_fcc.cif', 'ideal_bcc.cif'],
    'tasks': ['spacegroup_225.json', 'bandgap_measure.json', 'highk_screen.json'],
}.items():
    (root / folder).mkdir()
    for name in names:
        (root / folder / name).write_bytes(source.joinpath(folder, name).read_bytes())
print(root)
PY
```

Run real CPU coordinate analysis (ASE/spglib; no GPU or physical engine):

```sh
ctb evaluate --structures ctb-examples/structures/ideal_fcc.cif ctb-examples/structures/ideal_bcc.cif --task ctb-examples/tasks/spacegroup_225.json --cache measurements --output native-225
ctb summarize native-225
```

Expected: space groups 225 and 229; N=2, pass=1, fail=1, unknown=0 for target 225. `metrics.json`, `results.json`, `protocol.lock.json` and `run.json` record actual results. Space groups use categorical histograms. Duplicate and invalid submitted entries remain in N.

Change only the target, then reuse compatible measurements:

```sh
python - <<'PY'
import json
from pathlib import Path
task = json.loads(Path('ctb-examples/tasks/spacegroup_225.json').read_text())
task['task_id'] = 'user.sg229'
task['constraints'][0]['value'] = 229
Path('target-229.json').write_text(json.dumps(task, indent=2))
PY
ctb evaluate --structures ctb-examples/structures/ideal_fcc.cif ctb-examples/structures/ideal_bcc.cif --task target-229.json --cache measurements --output native-229
```

The accepted item changes from FCC to BCC; values remain 225/229. `native-229/run.json` should show `native_calls: 0`, `cache_hits: 2`. Use a new output directory for each target, with the same `--cache`. Changes to tolerances or measurement definitions require new compatible measurements. Advanced callers may pass the same task data to `crystargetbench.api.evaluate`; [task/metrics definitions](docs/tasks_and_metrics.md) explain the fields.

## Choose a physical method explicitly

`auto` prefers MLFF for supported physical tasks; a missing capability promotes the complete shared-geometry group to DFT. `mlff` is strict, `dft` requests DFT, and `hybrid` requires explicit routes and stays labelled mixed. The new default site does **not** preselect a DFT engine. Missing responses remain unknown/blocked; band gaps and total static dielectric response are not silently replaced by ML predictions.

Optional MatterSim uses the existing CTB-owned worker, separate environment, model fingerprint, disabled authorization template, and hard limits. Historical MatterSim phonon/EOS live evidence is limited to its recorded S2-B environment and small batch; it does not validate each new installation. See [backends](docs/backends.md). Installing `.[mlff]` alone neither downloads model weights nor authorizes calculations. The default installation does not depend on MatterSim, ABACUS, VASP, QE, MPI, atomate2, jobflow, or PYATB.

## Connect user-owned DFT software

The primary path exchanges files without installing or running an engine:

```sh
python -m pip install ./ctb_abacus_example-0.1.1-py3-none-any.whl
ctb dft adapters
# Fill your exact method and user-owned asset fingerprints in a private site file.
ctb dft prepare --structures ctb-examples/structures/ideal_fcc.cif --task ctb-examples/tasks/bandgap_measure.json --site site.json --adapter abacus_example --output jobs
# Execute prepared inputs on your own compute site and return outputs + receipt.
ctb dft collect --run jobs --results returned --output evaluation
```

Preparation exports only the ready frontier. A relaxed-reference task waits for a converged final structure before exporting dependent calculations; collection can expand the next frontier. Repeated collection is idempotent. Missing or failed results remain in the denominator. Unknown external execution costs are `null`; collection makes zero CTB engine invocations.

The [ABACUS example](examples/dft_adapters/abacus/README.md) is a separately installed thin adapter, restricted to its documented PW/PBE/nonmagnetic format profile. Parser evidence is **synthetic**, not live DFT or material validation. It does not implement electric response, so complete High-K remains blocked. No solver, pseudopotential, or orbital is distributed. An optional explicitly authorized local launcher calls a user-installed executable with fixed argv and persistent limits; it is not a cluster scheduler.

For a reproducible no-engine staged example, follow the [synthetic SDK walkthrough](docs/PREVIEW_SDK_WALKTHROUGH.md). It uses the existing second plugin and is not DFT evidence. CLI exit 3 means a blocked plan; exit 2 reports invalid data. A ready external plan (exit 0) can still contain a prepared job marked `blocked_assets`: inspect its `prepared.json` before user-side execution.

The interchange uses Python adapter API 1 and request/result wire v2. Other engines can implement [the adapter SDK](docs/DFT_ADAPTER_API.md) and register a static manifest through `crystargetbench.dft_adapters`, without changing core code. [The independently installed test plugin](tests/plugins/second) demonstrates this with synthetic evidence and the same assessor. Installing a plugin means trusting its Python code; manifests are not a security sandbox.

## Interpret and share results

- [Tasks and metrics](docs/tasks_and_metrics.md): property definitions, full denominators, pass/fail/unknown, rank versus screen.
- [Protocol compatibility](docs/protocol_comparability.md): method/geometry/cache identities and version migration.
- [Contributing](CONTRIBUTING.md), [third-party boundaries](THIRD_PARTY.md), [security](SECURITY.md).
- [Release checks](docs/release_checklist.md) and [public repository readiness](docs/PUBLIC_REPOSITORY_READINESS.md).

Compare generators within the same task and scientific profile. Keep different target batches separate. Submitted-item yield is not all-generation success rate or training cost. Share only structures/output you are entitled to disclose; use the allowlisted public exporter, which recursively removes private sessions and enabled permits. Software conformance and scientific leaderboard eligibility are separate.
