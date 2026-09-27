"""S2 planning/configuration tests. Manufactured permits never start a worker."""
from copy import deepcopy
from importlib.resources import files
import hashlib
import json
import sys

import pytest

from crystargetbench.assets import model_identity
from crystargetbench.backends import doctor
from crystargetbench.contracts import (default_deployment, default_policy, default_protocol,
    validate_deployment, validate_protocol)
from crystargetbench.identity import digest
from crystargetbench.planner import plan
from crystargetbench.s2_protocol import default_s2_protocol, is_complete_s2_protocol
from crystargetbench.structures import submission_digest


def example(name):
    return json.loads(files("crystargetbench").joinpath("resources", "examples", "tasks", name + ".json").read_text())


def metadata_only_site(tmp_path, task):
    checkpoint = tmp_path / "NOT_A_MODEL.synthetic"
    checkpoint.write_bytes(b"Synthetic identity fixture only; never torch.load or production execution")
    manifest = {"schema_version": "ctb.model_asset.v1", "provider": "mattersim",
        "model_id": "synthetic-planner-fixture", "model_version": "no-physical-model",
        "package_version": "1.1.2", "package_source": "synthetic-metadata-test",
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model_config": {}, "model_config_digest": digest({}), "source_reference": "synthetic_test_only",
        "license": "CC0-1.0", "license_review_status": "owner_approved", "trusted_source": True,
        "format": "torch_weights_only", "precision": "float32", "device": "cpu", "allow_tf32": False, "mixed_precision": False}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    environment = {"schema_version": "ctb.worker_environment.v1", "packages": {
        name: {"version": "synthetic-not-installed", "record_sha256": "0" * 64}
        for name in ("mattersim", "torch", "ase", "numpy", "crystargetbench")}}
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps(environment))
    permit_path = tmp_path / "permit.json"
    deployment = default_deployment()
    deployment["execution"]["mlff_authorized"] = True
    deployment["providers"]["mattersim"].update(configured=True, version="1.1.2", parameter_set_id="synthetic-test",
        model_manifest=str(manifest_path), live_permit=str(permit_path), environment_lock=str(environment_path),
        worker_argv=[sys.executable, "-m", "crystargetbench.workers.mattersim"], worker_timeout_seconds=10.0)
    items = [{"item_id": "item-0", "file_sha256": "1" * 64, "input_status": "valid", "status": "valid", "atom_count": 2,
              "input_geometry": {"geometry_id": "2" * 64, "species": ["Na", "Cl"]}}]
    permit = {"schema_version": "ctb.live_permit.v1", "authorized": True,
        "owner_approval_reference": "synthetic-schema-test-NOT-owner-authorization",
        "model_lock_digest": model_identity(manifest), "worker_environment_lock_sha256": hashlib.sha256(environment_path.read_bytes()).hexdigest(),
        "deployment_digest": digest(deployment), "input_manifest_sha256": submission_digest(items), "protocol_digest": "0" * 64,
        "device": "cpu", "allowed_operations": ["energy_forces_stress", "relaxation", "finite_displacement_phonon", "bulk_eos"],
        "limits": {"max_unique_structures": 2, "max_supercell_atoms": 128, "max_displaced_structures_total": 1000,
            "max_backend_structure_evaluations": 5000, "max_concurrent_workers": 1, "max_relax_steps_per_structure": 2000,
            "max_total_wall_seconds": 3600, "max_scratch_bytes": 100000000, "max_retries": 0, "max_dft_jobs": 0,
            "max_sample_wall_seconds": 1800, "per_request_wall_seconds": 60},
        "downloads_authorized": False, "cluster_authorized": False}
    permit_path.write_text(json.dumps(permit))
    initial = plan(task, deployment=deployment, submission=items)
    permit["protocol_digest"] = initial["protocol_digest"]
    permit_path.write_text(json.dumps(permit))
    return deployment, items, manifest, permit


def test_default_pure_mlff_gets_explicit_s2_candidate_protocol():
    result = plan(example("phonon_path"))
    protocol = result["protocol_lock"]["protocol"]
    assert protocol["protocol_id"].startswith("ctb.s2.")
    assert protocol["parameters"]["relaxation"]["filter"] == "FrechetCellFilter"
    assert protocol["parameters"]["phonon"]["primitive_matrix"] == "auto"
    assert is_complete_s2_protocol(example("phonon_path"), protocol)
    assert result["status"] == "blocked_configuration"
    assert not result["benchmark_eligible"] and result["physics_calls"] == 0


def test_s1_native_and_dft_protocols_are_preserved():
    for name in ("spacegroup_225", "highk_screen"):
        result = plan(example(name))
        assert result["protocol_lock"]["protocol"] == default_protocol(example(name))
    dft = plan(example("phonon_path"), default_policy("dft"))
    assert not dft["protocol_lock"]["protocol"]["protocol_id"].startswith("ctb.s2.")
    assert dft["physics_fidelity"] == "dft"


def test_C36_voigt_is_not_enabled_by_s2_eos_implementation():
    result = plan(example("bulk_voigt_ge100"))
    assert not result["s2_execution_supported"]
    assert "eos" not in result["protocol_lock"]["protocol"]["parameters"]


def test_seven_volume_points_and_full_quality_parameters_are_explicit():
    result = plan(example("bulk_eos_ge100"))
    eos = result["protocol_lock"]["protocol"]["parameters"]["eos"]
    assert eos["volume_factors"] == [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06]
    assert eos["fixed_cell_shape"] is True and eos["internal_relaxation"] is True
    assert eos["fit_rms_max_eV"] > 0


def test_explicit_s1_protocol_not_silently_upgraded():
    value = example("phonon_path")
    protocol = default_protocol(value)
    result = plan(value, protocol=protocol)
    assert result["protocol_lock"]["protocol"] == protocol
    assert not result["s2_execution_supported"]


def test_metadata_only_preflight_binds_every_identity_without_loading(tmp_path):
    value = example("phonon_path")
    deployment, items, manifest, _ = metadata_only_site(tmp_path, value)
    result = plan(value, deployment=deployment, submission=items)
    assert result["status"] == "ready"
    provider = result["protocol_lock"]["provider_locks"]["mattersim"]
    assert provider["model_lock_digest"] == model_identity(manifest)
    assert "checkpoint_path" not in provider["model_identity"]
    assert result["execution_preflight_required"] is True
    assert result["physics_calls"] == result["submitted_jobs"] == 0
    assert result["provider_states"]["mattersim"]["ready"] is False
    assert result["provider_states"]["mattersim"]["live_tested"] is False


def test_C46_disabled_permit_and_absent_batch_cannot_be_execution_ready(tmp_path):
    value = example("phonon_path")
    deployment, items, _, _ = metadata_only_site(tmp_path, value)
    assert plan(value, deployment=deployment)["status"] == "blocked_execution"
    from pathlib import Path
    Path(deployment["providers"]["mattersim"]["live_permit"]).write_text('{"authorized":false}')
    result = plan(value, deployment=deployment, submission=items)
    assert result["status"] == "blocked_execution"
    assert result["physics_calls"] == 0


@pytest.mark.parametrize("field", ["protocol_digest", "input_manifest_sha256", "deployment_digest", "worker_environment_lock_sha256"])
def test_permit_cannot_authorize_other_bound_inputs(tmp_path, field):
    value = example("phonon_path")
    deployment, items, _, permit = metadata_only_site(tmp_path, value)
    permit[field] = "f" * 64
    from pathlib import Path
    Path(deployment["providers"]["mattersim"]["live_permit"]).write_text(json.dumps(permit))
    assert plan(value, deployment=deployment, submission=items)["status"] == "blocked_execution"


def test_C08_model_configuration_changes_protocol_comparison_identity(tmp_path):
    value = example("phonon_path")
    deployment, items, manifest, _ = metadata_only_site(tmp_path, value)
    before = plan(value, deployment=deployment, submission=items)
    manifest["model_config"] = {"declared_configuration_variant": "different"}
    manifest["model_config_digest"] = digest(manifest["model_config"])
    from pathlib import Path
    Path(deployment["providers"]["mattersim"]["model_manifest"]).write_text(json.dumps(manifest))
    after = plan(value, deployment=deployment, submission=items)
    assert after["comparability_digest"] != before["comparability_digest"]
    assert after["status"] == "blocked_execution"


def test_C07_asset_digest_mismatch_is_not_an_implicit_dft_route(tmp_path):
    value = example("phonon_path")
    deployment, items, manifest, _ = metadata_only_site(tmp_path, value)
    from pathlib import Path
    Path(manifest["checkpoint_path"]).write_bytes(b"changed synthetic identity bytes")
    result = plan(value, deployment=deployment, submission=items)
    assert result["status"] == "missing_asset"
    assert result["physics_fidelity"] == "mlff" and result["promotion_reason"] is None


def test_untrusted_entrypoint_is_rejected_before_any_execution(tmp_path):
    value = example("phonon_path")
    deployment, items, _, _ = metadata_only_site(tmp_path, value)
    deployment["providers"]["mattersim"]["worker_argv"] = [sys.executable, "-c", "raise SystemExit"]
    assert plan(value, deployment=deployment, submission=items)["status"] == "blocked_configuration"


def test_C09_unknown_scientific_and_deployment_fields_remain_rejected():
    protocol = default_s2_protocol(example("phonon_path"))
    protocol["parameters"]["phonon"]["import"] = "os"
    with pytest.raises(ValueError):
        validate_protocol(protocol)
    deployment = default_deployment()
    deployment["providers"]["mattersim"]["shell"] = True
    with pytest.raises(ValueError):
        validate_deployment(deployment)


def test_incomplete_or_singular_s2_protocol_is_rejected():
    protocol = default_s2_protocol(example("phonon_path"))
    del protocol["parameters"]["relaxation"]["dt"]
    with pytest.raises(ValueError, match="Incomplete"):
        validate_protocol(protocol)
    protocol = default_s2_protocol(example("phonon_path"))
    protocol["parameters"]["phonon"]["primitive_matrix"] = [[0, 0, 0]] * 3
    with pytest.raises(ValueError, match="primitive"):
        validate_protocol(protocol)


def test_doctor_distinguishes_implemented_worker_from_live_validation():
    report = doctor()["providers"]["mattersim"]
    assert report["interface_implemented"] is True
    assert not report["ready"] and not report["live_tested"] and not report["physics_validated"]
