"""Synthetic routing contract tests: no test here runs a physical worker."""
from copy import deepcopy
from importlib.resources import files
import hashlib
import json

import pytest

from crystargetbench import planner
from crystargetbench.backends import REGISTRY, capabilities, doctor, record_failed_attempt, select_provider
from crystargetbench.contracts import default_deployment, default_policy, default_protocol
from crystargetbench.planner import plan


def example(group, name):
    return json.loads(files("crystargetbench").joinpath("resources", "examples", group, name + ".json").read_text())


def task(name):
    return example("tasks", name)


def policy(mode):
    return default_policy(mode)


def prepared_dft(tmp_path, highk=None):
    """Synthetic deployment metadata, with a tiny explicitly fake asset file."""
    # Explicit legacy fixture: v0.3 no longer preselects a DFT engine.
    from pathlib import Path
    deployment = json.loads((Path(__file__).parents[1]/'fixtures/legacy-deployment-v2.json').read_text())
    deployment['providers']['symmetry']['configured'] = True
    protocol = default_protocol(highk or task("highk_screen"))
    artifact = tmp_path / "synthetic-manifest.txt"
    artifact.write_bytes(b"Synthetic planner fixture. NOT a pseudopotential or physical result.")
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    for name in ("vasp_atomate2", "qe"):
        deployment["providers"][name].update(configured=True, version="synthetic-version-1", parameter_set_id="synthetic-parameters-v1")
        deployment["providers"][name]["assets"]["pseudopotential_manifest"] = {
            "path": str(artifact), "sha256": checksum, "license_status": "synthetic_fixture_only"}
    protocol["parameters"]["dft"] = {"parameter_set_id": "synthetic-parameters-v1",
        "pseudopotential_manifest_sha256": checksum, "cutoff_and_convergence_profile": "synthetic-cutoff-v1"}
    protocol["parameters"]["band_gap"].update(mesh_policy_id="synthetic-mesh-v1", spin_u_soc_policy_id="synthetic-spin-v1")
    deployment["execution"].update(max_dft_jobs=100000, per_job_cpu_cores=2, max_scratch_gb=20)
    return deployment, protocol


def test_R01_default_auto_prefers_mlff_phonon():
    result = plan(task("phonon_path"))
    assert result["physics_fidelity"] == "mlff"
    assert result["resource_preflight"]["dft_jobs_upper_bound"] == 0
    assert result["submitted_jobs"] == 0


def test_R02_dft_mode_input_symmetry_stays_native():
    result = plan(task("spacegroup_225"), policy("dft"))
    assert result["physics_fidelity"] == "native"
    assert all(node["family"] == "native" for node in result["nodes"])
    assert not any(node["kind"] == "tight_relaxation" for node in result["nodes"])


def test_R03_auto_gap_promotes_for_capability():
    result = plan(task("bandgap_gt2"))
    assert result["routes"]["gap"]["family"] == "dft"
    assert result["promotion_reason"] == "missing_required_capability"


def test_R04_strict_mlff_gap_blocks():
    result = plan(task("bandgap_gt2"), policy("mlff"))
    assert result["status"] == "blocked_capability"
    assert result["physics_calls"] == 0
    assert result["routes"]["gap"]["family"] == "mlff"


def test_R05_highk_promotes_whole_shared_geometry_group():
    result = plan(task("highk_screen"))
    assert all(route["family"] == "dft" for route in result["routes"].values())
    assert result["reference_geometry_backend"] == "dft"
    assert not any("nominal" in node["kind"] for node in result["nodes"])
    refs = [route["reference_geometry"] for route in result["routes"].values()]
    assert all(ref == refs[0] for ref in refs)


def test_R06_highk_strict_mlff_no_surrogate_substitution():
    result = plan(task("highk_screen"), policy("mlff"))
    assert result["status"] == "blocked_capability"
    assert all(route["family"] == "mlff" for route in result["routes"].values())


def test_R07_arbitrary_joint_phonon_gap_group_promotes():
    value = task("phonon_path")
    value["task_id"] = "arbitrary_user_name"
    value["measurements"]["gap"] = {"property_id": "band_gap_eV", "unit": "eV"}
    result = plan(value)
    assert all(route["family"] == "dft" for route in result["routes"].values())


def test_R08_explicit_hybrid_mixed_and_stationarity_checked():
    result = plan(task("highk_hybrid_screen"), example("policies", "hybrid_highk"))
    assert result["physics_fidelity"] == "mixed"
    assert result["routes"]["phonon"]["family"] == "mlff"
    assert result["routes"]["phonon"]["requires_stationary_reference"] is True
    nodes = {node["node_id"]: node for node in result["nodes"]}
    assert nodes["stationarity:phonon"]["on_failure"] == "measurement_unknown"
    assert nodes["stationarity:phonon"]["allow_reference_geometry_relaxation"] is False
    assert nodes["ionic_forces:kappa"]["family"] == "dft"
    assert "forces:phonon" not in nodes["ionic:kappa"]["dependencies"]


def test_R09_hybrid_without_routes_rejected():
    value = example("policies", "hybrid_highk")
    value["overrides"] = {}
    with pytest.raises(ValueError):
        plan(task("highk_hybrid_screen"), value)


def test_R10_surrogate_never_selected_implicitly():
    result = plan(task("bandgap_gt2"))
    assert result["routes"]["gap"]["family"] == "dft"
    assert result["status"] == "blocked_configuration"


def test_R11_explicit_gap_surrogate_labeled():
    result = plan(task("bandgap_gt2"), example("policies", "explicit_gap_surrogate"))
    assert result["routes"]["gap"]["physics_fidelity"] == "surrogate"
    assert "@surrogate:" in result["routes"]["gap"]["definition_id"]
    assert result["protocol_lock"]["protocol"]["parameters"]["band_gap"]["definition"] == "explicit_surrogate_prediction"


def test_R12_formal_fidelity_cannot_be_overridden():
    value = task("bandgap_gt2")
    value["measurements"]["gap"]["required_fidelity"] = "dft"
    with pytest.raises(ValueError, match="fidelity"):
        plan(value, example("policies", "explicit_gap_surrogate"))


def test_R13_missing_dft_is_configuration_not_material_failure():
    result = plan(task("highk_screen"))
    assert result["status"] == "blocked_configuration"
    assert result["physical_failures_created"] == 0
    assert len(result["nodes"]) >= 12


def test_R14_dft_without_authorization_never_submits(tmp_path):
    deployment, protocol = prepared_dft(tmp_path)
    result = plan(task("highk_screen"), protocol=protocol, deployment=deployment)
    assert result["status"] == "awaiting_authorization"
    assert result["submitted_jobs"] == 0


def test_R15_missing_weights_does_not_switch_method():
    deployment = default_deployment()
    deployment["providers"]["mattersim"].update(configured=True, version="synthetic-v1", parameter_set_id="synthetic-profile")
    result = plan(task("phonon_path"), deployment=deployment)
    assert result["status"] == "missing_asset"
    assert result["physics_fidelity"] == "mlff"


def test_R16_oom_records_attempt_without_dft_retry():
    result = record_failed_attempt("mattersim", "out_of_memory")
    assert result["calculation_status"] == "failed"
    assert result["next_action"] == "record_failure"
    assert result["retry_provider"] is None


def test_R17_nac_adds_born_and_electronic_requirements():
    value = task("phonon_path")
    protocol = default_protocol(value)
    protocol["parameters"]["phonon"]["nac"] = "from_dft_born_and_electronic"
    result = plan(value, protocol=protocol)
    assert result["physics_fidelity"] == "dft"
    requirements = set(result["routes"]["phonon"]["required_capabilities"])
    assert {"born_charges", "electronic_dielectric"} <= requirements


def test_R18_force_dft_even_when_mlff_registered():
    result = plan(task("bulk_eos_ge100"), policy("dft"))
    assert result["physics_fidelity"] == "dft"


def test_R19_eos_cannot_satisfy_voigt_recipe():
    eos = plan(task("bulk_eos_ge100"))
    value = task("bulk_voigt_ge100")
    alias = next(iter(value["measurements"]))
    value["measurements"][alias]["definition_id"] = next(iter(eos["routes"].values()))["definition_id"]
    with pytest.raises(ValueError, match="definition"):
        plan(value)


def test_R20_domain_rejection_does_not_switch_provider(monkeypatch):
    monkeypatch.setitem(REGISTRY["mattersim"], "supported_elements", ["Si"])
    result = plan(task("phonon_path"), submission=[{"item_id": "foreign", "status": "parsed", "input_geometry": {"species": ["Cu"]}}])
    assert result["item_preflight"][0]["status"] == "unsupported"
    assert result["item_preflight"][0]["implicit_itemwise_switch"] is False
    assert result["physics_fidelity"] == "mlff"


def test_R21_warm_start_preserves_final_dft_geometry_and_physics():
    value = policy("dft")
    value["warm_start"] = "mlff_pre_relax_then_dft"
    result = plan(task("highk_screen"), value)
    assert result["protocol_lock"]["warm_start"] == "mlff_pre_relax_then_dft"
    assert result["physics_fidelity"] == "dft"
    assert all(route["family"] == "dft" for alias, route in result["routes"].items() if alias != "warm_start")
    nodes = {node["node_id"]: node for node in result["nodes"]}
    assert nodes["tight_relax"]["dependencies"] == ["warm_start"]
    assert nodes["warm_start"]["final_physics_evidence"] is False


def test_R23_missing_pseudopotential_hash_blocks_configuration(tmp_path):
    deployment, protocol = prepared_dft(tmp_path)
    deployment["providers"]["vasp_atomate2"]["assets"]["pseudopotential_manifest"]["sha256"] = None
    result = plan(task("highk_screen"), protocol=protocol, deployment=deployment)
    assert result["status"] == "blocked_configuration"


def test_R24_multiple_providers_require_explicit_choice():
    selected, reason = select_provider("dft", None, default_deployment())
    assert selected is None
    assert reason == "explicit_provider_choice_required"


def test_unknown_selected_mlff_does_not_promote_due_to_configuration_error():
    value = policy("auto")
    value["preferred_mlff"] = "misspelled_provider"
    result = plan(task("phonon_path"), value)
    assert result["status"] == "blocked_configuration"
    assert result["promotion_reason"] is None
    assert not any(route["family"] == "dft" for route in result["routes"].values())


def test_R25_generator_and_submission_do_not_change_comparison():
    first = plan(task("highk_screen"), submission=[{"item_id": "one", "generator": "alpha", "atom_count": 2}])
    second = plan(task("highk_screen"), submission=[{"item_id": "two", "generator": "beta", "atom_count": 7}])
    assert first["routes"] == second["routes"]
    assert first["comparability_digest"] == second["comparability_digest"]


def test_R26_native_ready_independent_of_dft(monkeypatch):
    original = planner.doctor
    def synthetic_installation_probe(deployment):
        result = original(deployment)
        result["providers"]["symmetry"]["installed"] = True
        result["providers"]["symmetry"]["parser_dependency"]["installed"] = True
        for dependency in result["providers"]["symmetry"]["native_dependencies"].values():
            dependency["installed"] = True
        return result
    monkeypatch.setattr(planner, "doctor", synthetic_installation_probe)
    result = plan(task("spacegroup_225"))
    assert result["status"] == "ready"
    assert list(result["provider_states"]) == ["symmetry"]


def test_R27_nominal_proxy_never_satisfies_total_response():
    result = plan(task("highk_screen"), policy("mlff"))
    assert result["status"] == "blocked_capability"
    reason = next(r for r in result["reasons"] if r.get("alias") == "kappa")
    assert "electronic_dielectric" in reason["missing"]
    assert "ionic_dielectric" in reason["missing"]


def test_R28_qe_can_plan_candidate_ionic_composition_but_not_claim_validation():
    deployment = default_deployment()
    from pathlib import Path
    legacy = json.loads((Path(__file__).parents[1]/'fixtures/legacy-deployment-v2.json').read_text())
    deployment['providers']['qe'] = legacy['providers']['qe']
    deployment["default_dft_provider"] = "qe"
    result = plan(task("highk_screen"), deployment=deployment)
    assert result["routes"]["kappa"]["provider"] == "qe"
    assert "ionic_dielectric" in capabilities("qe")
    recipe = result["provider_states"]["qe"]["derived_recipes"]["ionic_dielectric"]
    assert recipe["unit_mapping_validated"] is False
    assert recipe["implemented"] is False
    assert result["provider_states"]["qe"]["live_tested"] is False


def test_highk_dag_contains_explicit_gate_and_geometry_dependencies():
    result = plan(task("highk_screen"))
    nodes = {node["node_id"]: node for node in result["nodes"]}
    assert {"tight_relax", "freeze_geometry", "scf_gap", "gap_gate", "forces:phonon", "phonon_gate:phonon", "electronic:kappa", "ionic:kappa", "tensor:kappa", "metrics"} <= nodes.keys()
    assert nodes["displacements:phonon"]["conditions"][0]["gate_node"] == "gap_gate"
    assert {condition["gate_node"] for condition in nodes["ionic:kappa"]["conditions"]} == {"gap_gate", "phonon_gate:phonon"}
    assert nodes["ionic:kappa"]["geometry_ref"] == nodes["scf_gap"]["geometry_ref"]
    assert nodes["freeze_geometry"]["identity_rule"] == "bind_actual_geometry_digest_after_completion"
    assert all(node["execution_status"] == "not_executed" for node in nodes.values())


def test_resource_estimate_counts_displacements_and_blocks_hard_limits(tmp_path):
    deployment, protocol = prepared_dft(tmp_path)
    deployment["execution"].update(dft_authorized=True, max_dft_jobs=2, max_supercell_atoms=10)
    result = plan(task("highk_screen"), protocol=protocol, deployment=deployment,
                  submission=[{"item_id": "x", "atom_count": 4}])
    assert result["status"] == "blocked_resource"
    assert result["resource_preflight"]["per_item"][0]["displacement_jobs_upper_bound"] == 192
    assert {"max_dft_jobs_exceeded", "max_supercell_atoms_exceeded"} <= set(result["resource_preflight"]["violations"])


def test_configured_authorized_metadata_does_not_make_physics_implemented(tmp_path):
    deployment, protocol = prepared_dft(tmp_path)
    deployment["execution"]["dft_authorized"] = True
    result = plan(task("highk_screen"), protocol=protocol, deployment=deployment)
    assert result["status"] == "blocked_implementation"
    assert result["resource_preflight"]["dft_jobs_upper_bound"] is None
    assert result["submitted_jobs"] == 0
    assert not result["provider_states"]["vasp_atomate2"]["ready"]


def test_deployment_cannot_augment_mlff_or_relabel_synthetic():
    deployment = default_deployment()
    deployment["providers"]["mattersim"]["reported_capabilities"].append("band_eigenvalues")
    with pytest.raises(ValueError, match="grant capabilities"):
        plan(task("bandgap_gt2"), deployment=deployment)
    deployment = default_deployment()
    from pathlib import Path
    legacy = json.loads((Path(__file__).parents[1]/'fixtures/legacy-deployment-v2.json').read_text())
    deployment["providers"]["synthetic"] = deepcopy(legacy["providers"]["vasp_atomate2"])
    with pytest.raises(ValueError, match="mismatch"):
        plan(task("highk_screen"), deployment=deployment)


def test_disordered_input_preserved_as_unsupported():
    result = plan(task("phonon_path"), submission=[{"item_id": "partial", "input_geometry": {"species": ["Si"], "occupancy": [0.5]}}])
    assert result["item_preflight"][0]["status"] == "unsupported"


def test_scientific_definition_changes_with_native_tolerance():
    value = task("spacegroup_225")
    protocol = default_protocol(value)
    before = plan(value, protocol=protocol)
    protocol["parameters"]["symprec_A"] *= 2
    after = plan(value, protocol=protocol)
    alias = next(iter(value["measurements"]))
    assert before["routes"][alias]["definition_id"] != after["routes"][alias]["definition_id"]


def test_recipe_scope_cannot_be_substituted():
    value = task("phonon_mesh")
    protocol = default_protocol(task("phonon_path"))
    with pytest.raises(ValueError, match="scope"):
        plan(value, protocol=protocol)


def test_incomplete_hybrid_map_rejected():
    value = example("policies", "hybrid_highk")
    del value["overrides"]["geometry"]
    with pytest.raises(ValueError, match="geometry"):
        plan(task("highk_hybrid_screen"), value)


def test_queue_output_and_asset_paths_not_comparison_identity(tmp_path):
    deployment, protocol = prepared_dft(tmp_path)
    before = plan(task("highk_screen"), protocol=protocol, deployment=deployment)
    deployment["execution"]["executor"] = "site_plugin"
    deployment["paths"]["runs_root"] = "elsewhere"
    deployment["execution"]["per_job_cpu_cores"] = 9
    after = plan(task("highk_screen"), protocol=protocol, deployment=deployment)
    assert before["comparability_digest"] == after["comparability_digest"]


def test_doctor_has_all_families_and_does_not_report_physics_live():
    report = doctor()
    assert {entry["family"] for entry in report["providers"].values()} >= {"native", "mlff", "dft", "surrogate", "synthetic"}
    assert all(not entry["live_tested"] and not entry["physics_validated"] for entry in report["providers"].values())
    assert report["workers_executed"] == 0
