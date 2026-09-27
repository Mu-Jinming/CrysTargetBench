"""Public data contracts: invalid data is rejected before any backend access."""

from copy import deepcopy
from importlib.resources import files
import json

import pytest

from crystargetbench.contracts import (
    default_deployment,
    default_policy,
    default_protocol,
    load_json,
    validate_deployment,
    validate_policy,
    validate_protocol,
    validate_task,
)


def example(kind, name):
    return json.loads(
        files("crystargetbench").joinpath("resources", "examples", kind, name).read_text()
    )


@pytest.mark.parametrize("filename", [
    "spacegroup_225.json", "phonon_path.json", "phonon_mesh.json", "bandgap_gt2.json",
    "bandgap_measure.json", "bulk_voigt_ge100.json", "bulk_eos_ge100.json",
    "highk_screen.json", "highk_rank.json", "highk_hybrid_screen.json", "nominal_proxy_rank.json",
])
def test_public_tasks_and_default_protocols(filename):
    task = example("tasks", filename)
    assert validate_task(task) == task
    protocol = default_protocol(task)
    assert validate_protocol(protocol) == protocol
    assert protocol["benchmark_eligible"] is False
    assert protocol["reference_geometry"] == task["reference_geometry"]


@pytest.mark.parametrize("filename", [
    "symmetry.candidate.json", "phonon_path.candidate.json", "highk_dft.candidate.json",
])
def test_kit_protocols_are_candidate_contracts_only(filename):
    protocol = example("protocols", filename)
    assert validate_protocol(protocol) == protocol
    assert protocol["validation_status"] == "not_scientifically_validated"


@pytest.mark.parametrize("filename", [
    "default_auto.json", "mlff_only.json", "dft_only.json", "hybrid_highk.json",
    "explicit_gap_surrogate.json", "explicit_nominal_proxy.json",
])
def test_all_public_method_policies(filename):
    policy = example("policies", filename)
    assert validate_policy(policy) == policy


@pytest.mark.parametrize("filename", ["local.template.json", "qe.template.json"])
def test_deployment_templates(filename):
    deployment = example("deployments", filename)
    assert validate_deployment(deployment) == deployment


@pytest.mark.parametrize("payload", [
    '{"a":1,"a":2}', '{"a":{"b":1,"b":2}}', '{"a":NaN}',
    '{"a":Infinity}', '{"a":-Infinity}', '{"a":1e999}', '[]', 'null', '{bad}',
])
def test_json_rejects_ambiguous_or_nonfinite_data(tmp_path, payload):
    target = tmp_path / "bad.json"
    target.write_text(payload)
    with pytest.raises(ValueError):
        load_json(target)


def test_json_is_data_never_an_expression(tmp_path):
    target = tmp_path / "literal.json"
    target.write_text('{"label":"__import__(\\\"os\\\").system(\\\"false\\\")"}')
    assert load_json(target)["label"] == '__import__("os").system("false")'


@pytest.mark.parametrize("value", [True, False, 225.0, 0, -1, 231, float("inf"), float("nan")])
def test_sg_targets_are_exact_integer_tokens_in_range(value):
    task = example("tasks", "spacegroup_225.json")
    task["constraints"][0]["value"] = value
    with pytest.raises(ValueError):
        validate_task(task)


def test_sg_membership_and_allowed_range():
    task = example("tasks", "spacegroup_225.json")
    task["constraints"][0].update(operator="in", values=[1, 225, 230])
    del task["constraints"][0]["value"]
    assert validate_task(task) == task
    task["constraints"][0]["values"] = [225, 225]
    with pytest.raises(ValueError):
        validate_task(task)


@pytest.mark.parametrize("alias", ["geometry", "warm_start"])
def test_measurement_aliases_cannot_overwrite_workflow_routes(alias):
    task = example("tasks", "bandgap_gt2.json")
    task["measurements"][alias] = task["measurements"].pop("gap")
    task["constraints"][0]["measurement"] = alias
    task["report"] = [alias]
    with pytest.raises(ValueError, match="reserved"):
        validate_task(task)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf")])
def test_threshold_must_be_finite_numeric_not_boolean(value):
    task = example("tasks", "bandgap_gt2.json")
    task["constraints"][0]["value"] = value
    with pytest.raises(ValueError):
        validate_task(task)


@pytest.mark.parametrize("lower,upper,closed_lower,closed_upper,valid", [
    (2, 2, True, True, True), (2, 2, False, True, False),
    (2, 2, True, False, False), (3, 2, True, True, False),
    (1, 2, False, False, True),
])
def test_nonempty_intervals(lower, upper, closed_lower, closed_upper, valid):
    task = example("tasks", "bandgap_gt2.json")
    task["constraints"] = [{
        "id": "gap_window", "measurement": "gap", "operator": "interval",
        "lower": lower, "upper": upper,
        "lower_inclusive": closed_lower, "upper_inclusive": closed_upper,
    }]
    if valid:
        assert validate_task(task) == task
    else:
        with pytest.raises(ValueError):
            validate_task(task)


@pytest.mark.parametrize("change", [
    "bad_unit", "bad_property", "bad_alias", "bad_report", "duplicate_id",
    "empty_screen", "shell", "v1", "unregistered_field", "bad_definition", "latest",
])
def test_task_semantics_and_no_execution_fields(change):
    task = example("tasks", "bandgap_gt2.json")
    if change == "bad_unit":
        task["measurements"]["gap"]["unit"] = "THz"
    elif change == "bad_property":
        task["measurements"]["gap"]["property_id"] = "arbitrary_callable"
    elif change == "bad_alias":
        task["constraints"][0]["measurement"] = "gapp"
    elif change == "bad_report":
        task["report"] = ["gapp"]
    elif change == "duplicate_id":
        task["constraints"].append(deepcopy(task["constraints"][0]))
    elif change == "empty_screen":
        task["constraints"] = []
    elif change == "shell":
        task["worker_argv"] = ["sh", "-c", "echo bad"]
    elif change == "v1":
        task["schema_version"] = "ctb.task.v1"
    elif change == "unregistered_field":
        task["measurements"]["gap"]["import"] = "evil"
    elif change == "bad_definition":
        task["measurements"]["gap"]["definition_id"] = "space_group_number@native:symmetry:abcd"
    elif change == "latest":
        task["measurements"]["gap"]["definition_id"] = "band_gap_eV@latest"
    with pytest.raises(ValueError):
        validate_task(task)


def test_measure_and_rank_mode_requirements():
    task = example("tasks", "bandgap_measure.json")
    task["constraints"] = [{"id": "gap", "measurement": "gap", "operator": "gt", "value": 2}]
    with pytest.raises(ValueError):
        validate_task(task)
    rank = example("tasks", "highk_rank.json")
    rank["ranking"]["measurement"] = "typo"
    with pytest.raises(ValueError):
        validate_task(rank)


def test_defaults_are_fresh_and_do_not_change_on_threshold_edits():
    task = example("tasks", "bandgap_gt2.json")
    protocol = default_protocol(task)
    task["task_id"] = "different-submission-task-name"
    task["constraints"][0]["value"] = 4
    assert default_protocol(task) == protocol
    first = default_policy()
    first["preferred_mlff"] = "changed"
    assert default_policy()["preferred_mlff"] == "mattersim"
    deployment = default_deployment()
    assert deployment["execution"]["per_job_cpu_cores"] >= 1
    assert deployment["execution"]["max_scratch_gb"] > 0
    assert deployment["execution"]["dft_authorized"] is False
    assert deployment["providers"]["symmetry"]["configured"] is True


def test_default_hybrid_is_never_implicit():
    with pytest.raises(ValueError, match="explicit"):
        default_policy("hybrid")


def test_validation_returns_detached_data():
    original = example("tasks", "bandgap_gt2.json")
    validated = validate_task(original)
    validated["constraints"][0]["value"] = 99
    assert original["constraints"][0]["value"] == 2


@pytest.mark.parametrize("mode", ["mlff", "dft"])
def test_strict_policy_cannot_enable_surrogate(mode):
    policy = default_policy(mode)
    policy["allow_surrogates"] = True
    with pytest.raises(ValueError):
        validate_policy(policy)


def test_dft_warm_start_is_explicit_and_keeps_mode():
    policy = default_policy("dft")
    policy["warm_start"] = "mlff_pre_relax_then_dft"
    assert validate_policy(policy)["mode"] == "dft"
    policy["mode"] = "mlff"
    with pytest.raises(ValueError):
        validate_policy(policy)


@pytest.mark.parametrize("change", ["override_in_auto", "runtime_fallback", "installation_fallback", "surrogate_without_optin"])
def test_invalid_policy_routes(change):
    policy = default_policy()
    if change == "override_in_auto":
        policy["overrides"] = {"gap": "surrogate:alignn2"}
    elif change == "runtime_fallback":
        policy["on_runtime_failure"] = "switch_to_dft"
    elif change == "installation_fallback":
        policy["promote_on_missing_installation"] = True
    else:
        policy = example("policies", "explicit_gap_surrogate.json")
        policy["allow_surrogates"] = False
    with pytest.raises(ValueError):
        validate_policy(policy)


@pytest.mark.parametrize("change", [
    "extra_parameter", "zero_symprec", "bool_symprec", "nan_symprec", "symmetrize",
    "missing_symmetry_parameter", "bad_angle",
])
def test_symmetry_protocol_has_strict_typed_parameters(change):
    protocol = example("protocols", "symmetry.candidate.json")
    params = protocol["parameters"]
    if change == "extra_parameter":
        params["execute"] = "anything"
    elif change == "zero_symprec":
        params["symprec_A"] = 0
    elif change == "bool_symprec":
        params["symprec_A"] = True
    elif change == "nan_symprec":
        params["symprec_A"] = float("nan")
    elif change == "symmetrize":
        params["target_symmetrization"] = True
    elif change == "missing_symmetry_parameter":
        del params["angle_tolerance_deg"]
    else:
        params["angle_tolerance_deg"] = -0.5
    with pytest.raises(ValueError):
        validate_protocol(protocol)


@pytest.mark.parametrize("change", [
    "zero_displacement", "singular_supercell", "bool_supercell", "fractional_supercell",
    "missing_mesh", "bad_mesh", "bool_steps", "unknown_nac", "unknown_phonon_field", "latest_dft",
])
def test_physics_protocol_parameters_are_validated_before_planning(change):
    protocol = example("protocols", "highk_dft.candidate.json")
    phonon = protocol["parameters"]["phonon"]
    if change == "zero_displacement":
        phonon["displacement_A"] = 0
    elif change == "singular_supercell":
        phonon["supercell_matrix"][0] = [0, 0, 0]
    elif change == "bool_supercell":
        phonon["supercell_matrix"][0][0] = True
    elif change == "fractional_supercell":
        phonon["supercell_matrix"][0][0] = 2.0
    elif change == "missing_mesh":
        del phonon["mesh"]
    elif change == "bad_mesh":
        phonon["mesh"] = [8, 0, 8]
    elif change == "bool_steps":
        protocol["parameters"]["relaxation"]["max_steps"] = True
    elif change == "unknown_nac":
        phonon["nac"] = "guess"
    elif change == "unknown_phonon_field":
        phonon["shell"] = "echo run"
    else:
        protocol["parameters"]["dft"]["parameter_set_id"] = "latest"
    with pytest.raises(ValueError):
        validate_protocol(protocol)


def test_combined_path_and_mesh_retains_both_sampling_definitions():
    task = example("tasks", "phonon_path.json")
    task["measurements"]["mesh"] = {"property_id": "phonon_mesh_min_frequency_THz", "unit": "THz"}
    params = default_protocol(task)["parameters"]["phonon"]
    assert params["q_scope"] == "path_and_mesh"
    assert params["mesh"] == [8, 8, 8]
    assert params["path_points_parameter"] == 101


@pytest.mark.parametrize("change", ["bool_budget", "negative_budget", "unknown_provider", "wrong_family", "nul_worker", "extra_exec"])
def test_deployment_budgets_and_provider_references(change):
    deployment = default_deployment()
    if change == "bool_budget":
        deployment["execution"]["per_job_cpu_cores"] = True
    elif change == "negative_budget":
        deployment["execution"]["max_scratch_gb"] = -1
    elif change == "unknown_provider":
        deployment["default_dft_provider"] = "missing"
    elif change == "wrong_family":
        deployment["default_dft_provider"] = "mattersim"
    elif change == "nul_worker":
        deployment["providers"]["mattersim"]["worker_argv"] = ["python\x00"]
    else:
        deployment["execution"]["shell"] = "bash"
    with pytest.raises(ValueError):
        validate_deployment(deployment)
