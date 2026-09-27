# Backend environments and installation

Run installation commands yourself in a fresh environment on the host where the backend will run. The commands below install software, not model weights or DFT assets. They do not perform relaxation, phonons, EOS, property inference or training. Core/native installation was tested previously; the new MatterSim/ALIGNN installation profiles below were checked against package metadata and code, not freshly installed or live-tested in this documentation change.

## Choose an environment

| Environment | Required for | Packages / installation |
| --- | --- | --- |
| `ctb-core` | Native symmetry, assessment, external file exchange | Python 3.11; `python -m pip install .` |
| `ctb-mattersim` | CTB relaxation, MLFF E/F/stress, finite-displacement phonons and EOS | PyTorch; `python -m pip install ".[mlff]"` with the supported pins below |
| `ctb-alignn` | Standalone upstream ALIGNN 2.0 property prediction | PyTorch; `python -m pip install alignn==2026.8.11`; CTB prediction integration is **not implemented** |
| Developer/numerical | Existing synthetic/analytic software tests | `python -m pip install ".[dev,numerical,export]"`; separate example plugins |

Start in a CTB checkout. `.` in `pip install .` means that checkout, not an arbitrary working directory. Activate the intended environment every time you open a shell. Do not install all backends into Conda `base`. If Conda is unavailable, `python3.11 -m venv /your/chosen/env` and `. /your/chosen/env/bin/activate` are alternatives; use an actual chosen path rather than this placeholder.

## MatterSim environment: relaxation, phonons and EOS

Create a dedicated environment:

```sh
conda create -n ctb-mattersim python=3.11 -y
conda activate ctb-mattersim
python -m pip install --upgrade pip
```

Install **one** PyTorch build. This example uses the upstream compatible wheel trio 2.6.0 / 0.21.0 / 2.6.0. Choose CPU if you only need CPU execution:

```sh
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA host with a driver compatible with the CUDA 12.4 wheel, use this **instead** of the CPU command:

```sh
nvidia-smi
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Use the [official PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/#v260) to select a build supported by your host. A CUDA wheel does not install an NVIDIA driver. Availability depends on the host and visible device; the example does not establish performance, model accuracy or CTB live qualification on your machine.

Then, from the CTB checkout, install CTB and its MLFF dependencies while retaining that PyTorch version family:

```sh
python -m pip install ".[mlff]" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"
python -m pip check
```

CTB pins `mattersim==1.1.2`, `numpy==1.26.4`, `ase==3.26.0`, `phonopy==2.38.2`, `seekpath==2.1.0` and `scipy==1.17.1`. The worker explicitly checks MatterSim 1.1.2 / ASE 3.26.0. Installing a newer MatterSim is not an automatic supported upgrade. Phonopy and SeeK-path are part of this extra, so there is no separate unpinned phonon installation step.

[MatterSim 1.1.2 metadata](https://pypi.org/pypi/mattersim/1.1.2/json) also declares transitive packages including atomate2, PyTorch Geometric, torchvision and torchaudio. CTB core does not require them; selecting `.[mlff]` does bring upstream dependencies into this separate environment. An installed Python orchestration package does not supply or authorize a DFT engine. Do not bypass dependency resolution with `--no-deps` to hide conflicts. If no suitable MatterSim wheel exists for your platform, its source build may require a working C/C++ toolchain; retain the build error rather than silently changing the scientific pins.

Verify package versions and CUDA visibility without loading a model:

```sh
python -c "from importlib.metadata import version; print({p: version(p) for p in ['crystargetbench','mattersim','torch','numpy','ase','phonopy','seekpath','scipy']})"
python -c "import torch; print('torch:', torch.__version__, 'wheel CUDA:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import sys; print(sys.executable)"
ctb doctor
```

`ctb doctor` is a metadata/configuration check, not a loader or physical-backend test. In the deployment, `worker_argv` must name the absolute interpreter printed above, followed by `-m`, `crystargetbench.workers.mattersim`. The driver also needs the pinned numerical libraries: using this same `ctb-mattersim` environment for the CTB CLI is the simplest arrangement. If you keep the driver separate, install `.[numerical]` there and CTB `.[mlff]` in the worker environment.

Before any model evaluation, CTB requires all of the following to agree with the actual batch:

- A user-owned local checkpoint plus a model asset manifest: source/license review, SHA256, model configuration digest, supported package version, device and float32 precision. Merely setting a provider to configured is insufficient.
- A worker environment lock for the installed packages, and a deployment selecting the CTB-owned worker with `model_manifest`, `environment_lock`, `live_permit` and its explicit interpreter.
- An owner-authorized permit binding that model/environment, deployment, input manifest and protocol, with finite structure/evaluation/step/time/scratch/retry budgets. The supplied disabled template remains disabled; installation is not permission to turn it on.

These checks are implemented in `src/crystargetbench/assets.py`, `scientific.py` and `workers/mattersim.py`. A text package list is useful for recording installation, but is **not** the required CTB JSON environment lock:

```sh
python -m pip freeze > ctb-mattersim-packages.txt
```

No checkpoint download is part of these commands. Obtain/review model assets explicitly and keep them outside the public repository. Do not use a convenience model constructor as an installation probe: upstream loaders can download weights. Missing assets/bindings remain blocked. Historical small-batch MatterSim results do not validate a new environment.

## ALIGNN 2.0 environment: property prediction

CTB's registry identifier is `alignn2`; its upstream Python distribution is **`alignn`**. As checked on 2026-09-27, [PyPI version 2026.8.11](https://pypi.org/pypi/alignn/2026.8.11/json) declares PyTorch and jarvis-tools dependencies. CTB 0.3.3 registers `band_gap_prediction`, but `interface_implemented` is false: there is no owned ALIGNN worker, production prediction adapter or `.[alignn2]` extra. The example surrogate policies describe intended routes; they do not implement inference.

For independent upstream experiments, prepare a separate environment:

```sh
conda create -n ctb-alignn python=3.11 -y
conda activate ctb-alignn
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "alignn==2026.8.11" "torch==2.6.0"
python -m pip check
python -c "from importlib.metadata import version; print({p: version(p) for p in ['alignn','torch','jarvis-tools']})"
python -m pip freeze > ctb-alignn-packages.txt
```

A CUDA build may replace the CPU torch selection using the official matrix above. This is an upstream installation example, not a CTB-tested inference lock. The [current upstream README](https://github.com/atomgptlab/alignn) describes the pure-PyTorch ALIGNN 2.0 path; the 2026.8.11 distribution metadata does not require DGL. Older documentation/models use DGL-specific paths. Do not mix legacy DGL installation commands with the pure-PyTorch setup or assume every old checkpoint is interchangeable; follow the selected model's exact upstream instructions.

Property prediction additionally needs a chosen property/model, compatible architecture/config, explicitly obtained checkpoint, units, training target and model provenance. A band-gap property model is not automatically an energy/force/stress calculator for relaxation. ALIGNN-FF is a different upstream force-field use case and has no CTB relaxation adapter here. Installation/version checks do not load or validate those models; consult [upstream pretrained-model documentation](https://github.com/atomgptlab/alignn/tree/main/docs/pretrained) for its current API instead of assuming a legacy inference function exists.

Do not expect `ctb evaluate` to start predicting properties after `pip install alignn`. A production CTB adapter and qualified inference evidence would be separate implementation/acceptance work. Any future ML prediction must retain surrogate provenance, explicit opt-in policy and property-specific units/definition. It cannot fill missing DFT electronic/ionic response or turn a blocked complete High-K result into a pass.

## Independent DFT adapter installation

From the CTB checkout in the core environment:

```sh
python -m pip install ./examples/dft_adapters/abacus
ctb dft adapters
```

This installs the CTB-authored example adapter, not ABACUS. Installing `.[dft]` does not install a solver; that extra is intentionally empty. A user-owned DFT executable, pseudopotentials/orbitals, exact fingerprints and any execution permission are separate requirements. Prepare/collect can exchange files without an installed engine. Complete ABACUS electric response remains unsupported.

## Existing routing and execution boundaries

Core installs NumPy, ASE, spglib and JSON schema support. Native structure recognition needs no GPU or solver. `ctb doctor` reads metadata, not engines. Installed adapter discovery reads static distribution manifests; only the explicitly selected trusted plugin is imported.

The `numerical` extra pins Phonopy 2.38.2, SeeK-path 2.1.0 and SciPy 1.17.1. MatterSim uses `.[mlff]` in a CTB-owned environment, with a user-supplied model asset, exact lock and explicit bound permit. Existing relaxation/phonon/EOS recipes, serial scheduling, sample deadlines and transitive cache signatures are preserved. Do not copy an old live permit to a new batch. The disabled example in `CTB_S2_Addendum/examples/MLFF_LIVE_PERMIT.disabled.json` stays disabled.

`dft` is now an empty compatibility extra. Install an adapter independently, and install/maintain any DFT engine yourself outside CTB. The ABACUS example has no solver dependency and no bundled pseudopotentials/orbitals. Legacy `vasp_atomate2` and `qe` registry identifiers retain their planning-only meaning; they are not renamed to ABACUS and their old configurations remain readable. Deployment v3 defaults to `default_dft_provider: null`.

External site v1 (`ctb.dft.site.v1`) explicitly selects a file-exchange track. It is separate from the normal MLFF deployment. `ctb plan/evaluate --deployment site.json` uses the same SDK exchange path; strict MLFF/hybrid policies cannot be silently interpreted as this DFT site. For an external site, evaluate prepares files. For an explicitly enabled managed_local site, evaluate serially launches prepared requests and collects them, stopping when outputs are missing. Complete electronic/ionic response must be supplied by a capable plugin; absent ABACUS response is blocked_capability.

The optional `ctb dft run-local` launcher requires a bound method/adapter/owned executable hash and complete hard limits. It uses shell=False, process-group cancellation, persistent Budget v2, attempts reserved before launch, and zero retry tokens by default. A successful process return code does not establish SCF/ionic/response convergence. No queue or cluster orchestration is provided. Prepare/collect never import this launcher.

S3 tests run no real model or DFT engine. Existing historical MLFF live evidence remains in its original scope and is not rewritten or extended by parser tests. See the S3 report for software, contract, parser, live and science status separately.
