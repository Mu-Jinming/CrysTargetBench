# CrysTargetBench (CTB)

CTB evaluates **submitted crystal structures against explicit property tasks**. It preserves every submitted item, reports coverage and unknown results, and records the scientific method separately from the target threshold. Version 0.3.3 is candidate benchmark software; its example thresholds are not universal or scientifically validated standards.

## Install the environment required by your task

Use a dedicated environment rather than installing into Conda `base`. **Installing CTB core does not install the models needed for relaxation or learned property prediction.** Choose the packages for the work you intend to run:

| Intended use | What to install | CTB 0.3.3 status |
| --- | --- | --- |
| Native space group / task assessment | CTB core (`pip install .`): NumPy, ASE, spglib, jsonschema | Implemented; CPU only |
| Relaxation, energy/forces/stress | PyTorch + **MatterSim 1.1.2**, using CTB `.[mlff]` | CTB worker implemented; local model, deployment, environment lock and explicit budget required |
| MatterSim finite-displacement phonons / EOS | Same `.[mlff]` environment, including **Phonopy 2.38.2, SeeK-path 2.1.0, SciPy 1.17.1** | Implemented recipes; model evaluation still requires authorization |
| Learned properties with ALIGNN 2.0 | Separate PyTorch + **`alignn`** environment; upstream setup below | **CTB `alignn2` execution adapter is not implemented**; installation does not activate this route |
| External DFT prepare/collect | Core + separately installed adapter | User supplies engine and assets; ABACUS electric response / complete High-K remains blocked |

[Detailed backend environment instructions](docs/backends.md) include CPU/CUDA choices, MatterSim and ALIGNN package commands, installation checks, model requirements and the distinction between software installation and a runnable CTB backend.

### Core environment: install before the native quickstart

Project repository: [Mu-Jinming/CrysTargetBench](https://github.com/Mu-Jinming/CrysTargetBench).
Report ordinary bugs and usage questions through [Issues](https://github.com/Mu-Jinming/CrysTargetBench/issues); see [SECURITY.md](SECURITY.md) before reporting sensitive information.

Python 3.11 is the locally tested baseline (package metadata allows >=3.11). The candidate source is available in this repository. Create and activate an isolated environment before installing:

```sh
git clone https://github.com/Mu-Jinming/CrysTargetBench.git
cd CrysTargetBench
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -m pip check
ctb --version
```

If you use Conda instead of venv, run `conda create -n ctb-core python=3.11 -y`, then `conda activate ctb-core` and `python -m pip install .` from the checkout. Use one environment method, not both. Each new terminal must activate the chosen environment.

**Package availability:** the repository contains source. No PyPI release or GitHub Release wheel is claimed. Use the source installation above unless a maintainer has separately supplied a verified wheel. Do not expect the following filename to appear automatically after cloning. If you already have that wheel, place it in your working directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install ./crystargetbench-0.3.3-py3-none-any.whl
ctb --version
```

### MatterSim environment for relaxation, phonons and EOS

From the CTB checkout, create a separate environment, install an appropriate PyTorch build, then install the CTB MLFF extra. A minimal CPU setup is shown below; the [full instructions](docs/backends.md#mattersim-environment-relaxation-phonons-and-eos) give CUDA alternatives and version checks. These are installation steps, not a model run.

```sh
conda create -n ctb-mattersim python=3.11 -y
conda activate ctb-mattersim
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[mlff]" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"
python -m pip check
```

`.[mlff]` pins MatterSim to **1.1.2** and includes CTB/ASE/Phonopy/SeeK-path/SciPy; do not replace it with an unpinned MatterSim upgrade. The PyTorch wheel example is an installation profile, not a claim of fresh live validation. A checkpoint with verified identity and a fully bound deployment/permit are still required before relaxation. Return to `ctb-core` (or your core venv) for the native-only example below.

### ALIGNN 2.0 environment for upstream property prediction

The upstream package is named **`alignn`**, not `alignn2`. Keep it separate from the MatterSim worker environment:

```sh
conda create -n ctb-alignn python=3.11 -y
conda activate ctb-alignn
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "alignn==2026.8.11" "torch==2.6.0"
python -m pip check
python -c "from importlib.metadata import version; print('alignn', version('alignn')); print('torch', version('torch'))"
```

This prepares a standalone upstream prediction environment; **CTB 0.3.3 has no production ALIGNN2 prediction adapter**. The registered provider and example policies are planning declarations. No `.[alignn2]` extra or working CTB ALIGNN prediction command is provided. See the [ALIGNN instructions and model requirements](docs/backends.md#alignn-20-environment-property-prediction) before using upstream models; learned gaps are surrogate predictions, not DFT band gaps.

### Run the native example in the core environment

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
python -m pip install ./examples/dft_adapters/abacus
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
