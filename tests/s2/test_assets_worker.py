"""S2 C07-C14/C46: asset gates, real subprocess contract, analytic E/F/stress.

Every executed calculator here is synthetic/analytic_test. No MLFF imports,
weights, GPU, or material-science assertions are involved.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest
from ase import Atoms
from ase.units import GPa

from crystargetbench.assets import (AssetError, model_identity, preflight,
                                    validate_permit, verify_asset)
from crystargetbench.identity import digest
from crystargetbench.workers import WorkerClient, WorkerError
from crystargetbench.workers.protocol import normalize_efs, read_npz, write_npz


def asset(tmp_path):
    path = tmp_path / "synthetic-bytes-not-a-model.pth"
    path.write_bytes(b"CTB dummy bytes; must never be loaded as torch")
    return {"schema_version": "ctb.model_asset.v1", "provider": "mattersim",
        "model_id": "not-a-real-model", "model_version": "synthetic_asset_identity_test",
        "package_version": "1.1.2", "package_source": "official-package-source-reference",
        "checkpoint_path": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_config": {}, "model_config_digest": digest({}),
        "source_reference": "CTB synthetic byte fixture", "license": "CC0-1.0",
        "license_review_status": "owner_approved", "trusted_source": True,
        "format": "torch_weights_only", "precision": "float32", "device": "cpu",
        "allow_tf32": False, "mixed_precision": False}


def permit(lock):
    # This manufactured record tests validation only. It is never passed to a
    # real loader or saved as a runtime site profile.
    return {"schema_version": "ctb.live_permit.v1", "authorized": True,
        "owner_approval_reference": "synthetic_schema_validation_only_not_owner_approval",
        "model_lock_digest": lock["model_lock_digest"], "worker_environment_lock_sha256": "1" * 64,
        "deployment_digest": "2" * 64, "input_manifest_sha256": "3" * 64,
        "protocol_digest": "4" * 64, "device": "cpu", "allowed_operations": ["energy_forces_stress"],
        "limits": {"max_unique_structures": 2, "max_supercell_atoms": 8,
            "max_displaced_structures_total": 0, "max_backend_structure_evaluations": 20,
            "max_concurrent_workers": 1, "max_relax_steps_per_structure": 50,
            "max_total_wall_seconds": 10, "max_scratch_bytes": 100000,
            "max_sample_wall_seconds": 5, "per_request_wall_seconds": 2,
            "max_retries": 0, "max_dft_jobs": 0},
        "downloads_authorized": False, "cluster_authorized": False}


def pair():
    return Atoms("Si2", positions=[[.1, .2, .3], [1.2, .9, .7]],
                 cell=[[3.1, .1, .2], [0, 3, .1], [.1, .2, 3.2]], pbc=True)


def snapshot():
    from crystargetbench.geometry import snapshot_from_atoms
    return snapshot_from_atoms(pair())


def client(tmp_path, monkeypatch, mode="normal", timeout=5):
    # Explicit test namespace; installed production worker never receives this
    # PYTHONPATH or an arbitrary import string from user task data.
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", str(root / "src") + ":" + str(root / "tests" / "s2"))
    return WorkerClient([str(Path(sys.executable).absolute()), "-m", "worker_fixtures", "--mode", mode],
        tmp_path, {"provider": "analytic_test", "evidence_kind": "synthetic", "recipe": "harmonic_pair.v1"},
        timeout_seconds=timeout, test_mode=True)


def lease(number=1):
    return {"lease_id": f"synthetic-budget-{number}", "status": "reserved"}


def test_C07_asset_missing_digest_and_preload_recheck(tmp_path):
    data = asset(tmp_path)
    lock = verify_asset(data)
    assert lock["digest_verified"] and lock["benchmark_eligible"] is False
    Path(data["checkpoint_path"]).write_bytes(b"changed")
    with pytest.raises(AssetError, match="SHA256") as exc:
        verify_asset(data)
    assert exc.value.category == "blocked_asset"
    Path(data["checkpoint_path"]).unlink()
    assert preflight(data, None)["status"] == "blocked_asset"
    assert preflight(data, None)["physics_calls"] == 0


@pytest.mark.parametrize("field,value", [("format", "pickle"), ("trusted_source", False),
    ("license_review_status", "unresolved"), ("package_version", "latest"), ("allow_tf32", True),
    ("arbitrary_import", "os.system"), ("shell", "touch danger"), ("precision", "float16")])
def test_C09_untrusted_asset_configuration_rejected(tmp_path, field, value):
    data = asset(tmp_path)
    data[field] = value
    with pytest.raises(AssetError):
        verify_asset(data)


def test_C08_model_precision_config_device_change_identity(tmp_path):
    data = asset(tmp_path)
    before = model_identity(data)
    for field, value in [("model_version", "v2"), ("precision", "float64"), ("device", "cuda:0")]:
        changed = dict(data, **{field: value})
        assert model_identity(changed) != before
    with pytest.raises(AssetError, match="float32"):
        verify_asset(dict(data, precision="float64"))
    assert model_identity(dict(data, checkpoint_path="/explicit/relocated.pth")) == before
    with pytest.raises(AssetError, match="configuration digest"):
        verify_asset(dict(data, model_config={"changed": 1}))


def test_C46_disabled_template_does_not_authorize(tmp_path):
    lock = verify_asset(asset(tmp_path))
    root = Path(__file__).resolve().parents[2]
    disabled = json.loads((root / "CTB_S2_Addendum/examples/MLFF_LIVE_PERMIT.disabled.json").read_text())
    assert preflight(lock["manifest"], disabled)["status"] == "blocked_execution"
    disabled["authorized"] = True
    with pytest.raises(AssetError):
        validate_permit(disabled, lock)
    with pytest.raises(AssetError):
        WorkerClient([str(Path(sys.executable).absolute()), "-m", "crystargetbench.workers.mattersim"],
            tmp_path, lock, permit=None)


def test_C46_authorization_is_independent_and_bindings_checked(tmp_path):
    lock = verify_asset(asset(tmp_path))
    record = permit(lock)
    assert validate_permit(record, lock)["authorized"]
    for field in ("model_lock_digest", "worker_environment_lock_sha256", "deployment_digest", "input_manifest_sha256", "protocol_digest"):
        altered = deepcopy(record)
        altered[field] = "0" * 64
        bindings = None if field == "model_lock_digest" else {field: record[field]}
        with pytest.raises(AssetError):
            validate_permit(altered, lock, bindings)
    from crystargetbench.workers.mattersim import MatterSimSession
    with pytest.raises(AssetError):
        MatterSimSession({"test_mode": True})
    assert "mattersim" not in sys.modules and "torch" not in sys.modules


def test_C12_ase_and_batch_stress_normalization_and_missing():
    tensor = np.array([[1., .2, .3], [.2, -2., .4], [.3, .4, 3.]])
    a = normalize_efs(3., [[0, 1, 2]], tensor * GPa, 1)
    b = normalize_efs(3., [[0, 1, 2]], tensor, 1, stress_unit="GPa")
    np.testing.assert_allclose(a["stress_eV_A3"], b["stress_eV_A3"], atol=0)
    assert b["pressure_GPa"] == pytest.approx(-2 / 3)
    assert b["raw_units"]["stress"] == "GPa" and len(b["conversions"]) == 1
    missing = normalize_efs(3., [[0, 1, 2]], None, 1)
    assert missing["stress_eV_A3"] is None and missing["quality_status"] == "missing_stress"
    voigt = normalize_efs(3., [[0, 1, 2]], [1, -2, 3, .4, .3, .2], 1, stress_unit="GPa")
    np.testing.assert_allclose(voigt["stress_eV_A3"], b["stress_eV_A3"])


@pytest.mark.parametrize("energy,forces,stress,options", [
    (float("nan"), [[0, 0, 0]], [0.] * 6, {}),
    (1., [[0, 0]], [0.] * 6, {}), (True, [[0, 0, 0]], [0.] * 6, {}),
    (1., [[0, 0, 0]], [[0, 1, 0], [0, 0, 0], [0, 0, 0]], {}),
    (1., [[0, 0, 0]], [0.] * 6, {"energy_unit": "eV/atom"}),
    (1., [[0, 0, 0]], [0.] * 6, {"stress_unit": "bar"}),
])
def test_C12_malformed_or_ambiguous_units_rejected(energy, forces, stress, options):
    with pytest.raises(ValueError):
        normalize_efs(energy, forces, stress, 1, **options)


def test_C13_analytic_force_energy_difference_and_translation():
    from .worker_fixtures import AnalyticPair
    atoms = pair()
    atoms.calc = AnalyticPair()
    forces = atoms.get_forces()
    delta = 1e-6
    original = atoms.positions.copy()
    for atom in range(2):
        for axis in range(3):
            atoms.positions[:] = original
            atoms.positions[atom, axis] += delta
            plus = atoms.get_potential_energy()
            atoms.positions[atom, axis] -= 2 * delta
            minus = atoms.get_potential_energy()
            assert forces[atom, axis] == pytest.approx(-(plus - minus) / (2 * delta), abs=1e-8)
    atoms.positions[:] = original
    energy = atoms.get_potential_energy()
    atoms.positions += [1.3, -2.4, .7]
    assert atoms.get_potential_energy() == pytest.approx(energy, abs=1e-12)
    np.testing.assert_allclose(atoms.get_forces(), forces, atol=1e-12)
    assert np.linalg.norm(forces) > 0


def test_C14_analytic_stress_energy_strain_and_engineering_shear():
    from .worker_fixtures import AnalyticPair
    atoms = pair()
    atoms.calc = AnalyticPair()
    stress = atoms.get_stress(voigt=False)
    initial_cell, initial_scaled = atoms.cell.copy(), atoms.get_scaled_positions()
    volume, delta = atoms.get_volume(), 1e-6
    for i, j in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]:
        strain = np.zeros((3, 3))
        strain[i, j] = 1 if i == j else .5
        strain[j, i] = strain[i, j]
        values = []
        for sign in (1, -1):
            atoms.set_cell(initial_cell @ (np.eye(3) + sign * delta * strain))
            atoms.set_scaled_positions(initial_scaled)
            values.append(atoms.get_potential_energy())
        assert (values[0] - values[1]) / (2 * delta * volume) == pytest.approx(stress[i, j], abs=2e-9)


def test_C10_C11_real_persistent_synthetic_worker_and_request_identity(tmp_path, monkeypatch):
    with client(tmp_path, monkeypatch) as worker:
        first = worker.evaluate(snapshot(), "node-1", budget_lease=lease(1))
        second = worker.evaluate(snapshot(), "node-2", budget_lease=lease(2))
        assert first["evidence_kind"] == "synthetic" and first["provider"] == "analytic_test"
        assert first["environment"]["pid"] == second["environment"]["pid"]
        assert second["environment"]["model_loads_this_session"] == 1
        assert second["environment"]["requests"] == 2
        assert len(worker.attempts) == 2
        assert first["request_id"] != second["request_id"]
        assert first["benchmark_eligible"] is False
        assert len(list(tmp_path.glob("*.attempt.json"))) == 2


@pytest.mark.parametrize("mode,category", [("bad_json", "protocol_error"), ("crash", "worker_crashed"),
    ("stdout_log", "protocol_error"), ("wrong_request", "protocol_error"),
    ("wrong_geometry", "protocol_error"), ("wrong_mapping", "protocol_error"),
    ("nonfinite", "protocol_error"), ("wrong_family", "protocol_error"),
    ("wrong_provider", "protocol_error"), ("wrong_units", "protocol_error"), ("oom", "out_of_memory")])
def test_C10_C11_worker_faults_do_not_reuse_or_switch(tmp_path, monkeypatch, mode, category):
    with client(tmp_path, monkeypatch, mode) as worker:
        with pytest.raises(WorkerError) as caught:
            worker.evaluate(snapshot(), "fault-node", budget_lease=lease())
        assert caught.value.category == category
        assert worker.attempts[-1]["status"] == category
        assert worker.attempts[-1]["retry_provider"] is None
        assert worker.process.poll() is not None


def test_C10_timeout_records_attempt_and_terminates_worker(tmp_path, monkeypatch):
    with client(tmp_path, monkeypatch, "timeout", timeout=.7) as worker:
        with pytest.raises(WorkerError) as caught:
            worker.evaluate(snapshot(), "timeout-node", budget_lease=lease())
        assert caught.value.category == "timeout"
        assert worker.process.poll() is not None
        assert worker.attempts[-1]["status"] == "timeout"


def test_C42_cancel_and_session_concurrency(tmp_path, monkeypatch):
    with client(tmp_path, monkeypatch, "timeout") as worker:
        errors = []
        def run():
            try:
                worker.evaluate(snapshot(), "cancel-node", budget_lease=lease())
            except WorkerError as exc:
                errors.append(exc.category)
        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 3
        while worker.process is None and time.monotonic() < deadline:
            time.sleep(.01)
        with pytest.raises(WorkerError, match="in-flight"):
            worker.evaluate(snapshot(), "concurrent-node", budget_lease=lease(2))
        worker.cancel()
        thread.join(timeout=3)
        assert not thread.is_alive() and errors == ["cancelled"]
        assert worker.process.poll() is not None
        assert len(worker.attempts) == 1


def test_C09_worker_task_cannot_override_asset_or_command(tmp_path, monkeypatch):
    from crystargetbench.workers.protocol import validate_request
    with client(tmp_path, monkeypatch) as worker:
        with pytest.raises(ValueError, match="budget"):
            worker.evaluate(snapshot(), "no-budget")
        assert worker.process is None
    with pytest.raises(ValueError, match="array"):
        WorkerClient("python; echo injected", tmp_path, {}, test_mode=True)
    with pytest.raises(ValueError, match="synthetic"):
        WorkerClient([str(Path(sys.executable).absolute())], tmp_path, {"provider": "mattersim"}, test_mode=True)
    with pytest.raises(ValueError):
        validate_request({"shell": "touch injected", "checkpoint_path": "arbitrary"})


def test_C09_npz_numeric_only_hash_shape_and_path_guards(tmp_path):
    data = {"forces": np.arange(12.).reshape(4, 3)}
    manifest = write_npz(tmp_path, "forces.npz", data)
    np.testing.assert_array_equal(read_npz(tmp_path, manifest)["forces"], data["forces"])
    with pytest.raises(ValueError):
        write_npz(tmp_path, "../escape.npz", data)
    with pytest.raises(ValueError):
        write_npz(tmp_path, "object.npz", {"object": np.array([{}], dtype=object)})
    changed = deepcopy(manifest)
    changed["arrays"]["forces"]["shape"] = [2, 6]
    with pytest.raises(ValueError, match="manifest"):
        read_npz(tmp_path, changed)
    (tmp_path / "forces.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_npz(tmp_path, manifest)


def test_C09_npz_object_pickle_and_zip_size_are_never_loaded(tmp_path):
    path = tmp_path / "unsafe.npz"
    np.savez(path, objects=np.array([{"must_not_unpickle": True}], dtype=object))
    manifest = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "arrays": {"objects": {"dtype": "object", "shape": [1]}}}
    with pytest.raises(ValueError, match="allow_pickle"):
        read_npz(tmp_path, manifest)
    with pytest.raises(ValueError, match="oversized"):
        read_npz(tmp_path, manifest, max_bytes=2)


def test_C09_geometry_asset_override_and_changed_mass_are_rejected(tmp_path, monkeypatch):
    with client(tmp_path, monkeypatch) as worker:
        malicious = dict(snapshot(), checkpoint_path="task-supplied.pth")
        with pytest.raises(ValueError, match="data-only"):
            worker.evaluate(malicious, "node", budget_lease=lease())
        changed_mass = deepcopy(snapshot())
        changed_mass["masses"][0] += 1
        with pytest.raises(ValueError, match="mass identity"):
            worker.evaluate(changed_mass, "node", budget_lease=lease())
        assert worker.process is None


def test_C12_synthetic_missing_stress_stays_null(tmp_path, monkeypatch):
    with client(tmp_path, monkeypatch, "missing_stress") as worker:
        observed = worker.evaluate(snapshot(), "node", budget_lease=lease())
        assert observed["stress_eV_A3"] is None
        assert observed["pressure_GPa"] is None
        assert observed["quality_status"] == "missing_stress"


def test_C41_worker_honors_remaining_lease_wall_limit(tmp_path, monkeypatch):
    with client(tmp_path, monkeypatch, "timeout", timeout=10) as worker:
        start = time.monotonic()
        with pytest.raises(WorkerError) as caught:
            worker.evaluate(snapshot(), "node", budget_lease={**lease(), "max_wall_seconds": .3})
        assert caught.value.category == "timeout"
        assert time.monotonic() - start < 3


def test_C04_worker_environment_identity_requires_actual_installed_packages(monkeypatch):
    from crystargetbench.assets import verify_worker_environment
    from importlib import metadata
    names = ("mattersim", "torch", "numpy", "ase", "crystargetbench")
    class FakeDistribution:
        version = "synthetic_env_validation"
        def read_text(self, name):
            return "synthetic-record"
    monkeypatch.setattr(metadata, "distribution", lambda _: FakeDistribution())
    lock = {"schema_version": "ctb.worker_environment.v1", "packages": {
        name: {"version": "synthetic_env_validation", "record_sha256": hashlib.sha256(b"synthetic-record").hexdigest()}
        for name in names}}
    assert verify_worker_environment(lock) == lock
    lock["packages"]["torch"]["record_sha256"] = "0" * 64
    with pytest.raises(AssetError, match="differs from lock"):
        verify_worker_environment(lock)
    del lock["packages"]["torch"]
    with pytest.raises(AssetError, match="essential"):
        verify_worker_environment(lock)


def test_C04_environment_file_hash_and_parsed_content_both_bound(tmp_path):
    from crystargetbench.assets import verify_environment_file
    lock = {"synthetic_schema_test": True}
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(lock, indent=2))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    verify_environment_file(lock, path, checksum)
    with pytest.raises(AssetError, match="identity mismatch"):
        verify_environment_file(dict(lock, changed=True), path, checksum)
    with pytest.raises(AssetError, match="identity mismatch"):
        verify_environment_file(lock, path, "0" * 64)


@pytest.mark.parametrize("name", ["max_sample_wall_seconds", "per_request_wall_seconds"])
def test_C46_runtime_permit_requires_explicit_sample_and_request_time(tmp_path, name):
    lock = verify_asset(asset(tmp_path))
    record = permit(lock)
    del record["limits"][name]
    with pytest.raises(AssetError):
        validate_permit(record, lock)
    record = permit(lock)
    record["limits"][name] = 0
    with pytest.raises(AssetError, match="positive"):
        validate_permit(record, lock)


def test_C42_parent_death_stops_synthetic_worker_process_group(tmp_path, monkeypatch):
    import os
    import subprocess
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root / "src") + ":" + str(root / "tests" / "s2"),
               OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    config = tmp_path / "driver.json"
    config.write_text(json.dumps({"geometry": snapshot()}))
    parent = subprocess.Popen([sys.executable, "-m", "worker_fixtures", "--mode", "parent_death_driver",
                               "--session-config", str(config)], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
    worker_pid = None
    try:
        pidfile = tmp_path / "child" / "watchdog.pid"
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert pidfile.exists()
        worker_pid = int(pidfile.read_text())
        parent.kill()
        parent.wait(timeout=2)
        # A dead adopted process may briefly be a zombie awaiting init's reaper.
        def running():
            status = Path(f"/proc/{worker_pid}/status")
            if not status.exists():
                return False
            try:
                return not any(line.startswith("State:") and "Z" in line for line in status.read_text().splitlines())
            except FileNotFoundError:
                return False
        deadline = time.monotonic() + 3
        while running() and time.monotonic() < deadline:
            time.sleep(.02)
        assert not running()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2)
        if worker_pid:
            try:
                os.killpg(worker_pid, 9)
            except ProcessLookupError:
                pass


def test_C41_worker_scratch_limit_stops_before_process_start(tmp_path, monkeypatch):
    with client(tmp_path, monkeypatch) as worker:
        def denied(_):
            raise RuntimeError("synthetic exhausted scratch")
        worker.scratch_check = denied
        with pytest.raises(WorkerError) as exc:
            worker.evaluate(snapshot(), "scratch", budget_lease=lease())
        assert exc.value.category == "budget_exhausted"
        assert worker.process is None
        assert worker.attempts[-1]["artifact_saved"] is False
        assert worker.attempts[-1]["physics_calls"] is None


def test_C41_large_attempt_has_small_fallback_and_terminates(tmp_path, monkeypatch):
    with client(tmp_path, monkeypatch) as worker:
        def bounded(size):
            if size > 2500:
                raise RuntimeError("synthetic artifact allowance")
        worker.scratch_check = bounded
        with pytest.raises(WorkerError) as exc:
            worker.evaluate(snapshot(), "scratch", budget_lease=lease())
        assert exc.value.category == "budget_exhausted"
        assert worker.process is not None and worker.process.poll() is not None
        assert worker.attempts[-1]["artifact_saved"] is True
        assert worker.attempts[-1]["backend_forwards"] == 1
        assert worker.attempts[-1]["status"] == "budget_exhausted"
