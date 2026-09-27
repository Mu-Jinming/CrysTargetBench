"""Synthetic observation contracts, not executions of scientific backends."""

import json
from copy import deepcopy

import pytest

from crystargetbench.assessment import assess, make_measurement, predicate, three_valued_and
from crystargetbench.metrics import aggregate


def task(mode="screen"):
    result = {
        "schema_version": "ctb.task.v2", "task_id": "synthetic_gap_test", "mode": mode,
        "reference_geometry": "input", "measurements": {"gap": {"property_id": "band_gap_eV", "unit": "eV"}},
        "constraints": [{"id": "gap_above", "measurement": "gap", "operator": "gt", "value": 2}],
    }
    if mode == "measure":
        result.pop("constraints")
        result["report"] = ["gap"]
    if mode == "constrained_rank":
        result["measurements"]["kappa"] = {"property_id": "dielectric_total_static_trace_over3", "unit": "1"}
        result["ranking"] = {"measurement": "kappa", "direction": "maximize"}
    return result


def observation(value=3, **overrides):
    fields = dict(
        value=value, reference_geometry_id="geometry-a", property_id="band_gap_eV", unit="eV", definition_id="synthetic.gap.v1",
        calculation_status="completed", quality_status="accepted", identity_status="verified",
        backend_id="test.synthetic", backend_family="synthetic", backend_version="1", physics_fidelity="synthetic",
        protocol_digest="synthetic-test-protocol", diagnostics={"test_only": True}, benchmark_eligible=False,
    )
    fields.update(overrides)
    return make_measurement(**fields)


def result(value=3, item_id="sample", current_task=None, **overrides):
    current_task = current_task or task()
    assessed = assess(current_task, {"gap": observation(value, **overrides)}, "geometry-a")
    return dict(item_id=item_id, input_status="valid", **assessed)


@pytest.mark.parametrize("constraint,value,expected", [
    ({"operator": "gt", "value": 2}, 2, False),
    ({"operator": "ge", "value": 2}, 2, True),
    ({"operator": "lt", "value": 2}, 2, False),
    ({"operator": "le", "value": 2}, 2, True),
    ({"operator": "eq", "value": 225}, 225, True),
    ({"operator": "in", "values": [1, 225]}, 225, True),
    ({"operator": "within", "value": 2, "atol": 0.1}, 2.1, True),
    ({"operator": "within", "value": 2, "atol": 0.1}, 2.10000001, False),
    ({"operator": "interval", "lower": 2, "upper": 2, "lower_inclusive": True, "upper_inclusive": True}, 2, True),
    ({"operator": "interval", "lower": 2, "upper": 3, "lower_inclusive": False, "upper_inclusive": True}, 2, False),
    ({"operator": "ge", "value": -0.001}, -0.001, True),
])
def test_predicates_boundaries(constraint, value, expected):
    assert predicate(value, constraint) is expected


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -float("inf"), "2", 10**400])
def test_predicate_rejects_nonfinite_or_non_numeric(value):
    with pytest.raises(ValueError):
        predicate(value, {"operator": "gt", "value": 2})


@pytest.mark.parametrize("decisions,expected", [
    (["pass", "pass", "pass"], "pass"), (["pass", "fail", "unknown"], "fail"),
    (["pass", "unknown"], "unknown"), (["unknown"], "unknown"), ([], "unknown"),
])
def test_three_valued_and(decisions, expected):
    assert three_valued_and(decisions) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), [1, float("nan")], 10**400])
def test_nonfinite_serializes_as_null_with_raw(value):
    measured = observation(value, diagnostics={"raw_diagnostic": float("nan")})
    assert measured["value"] is None
    assert isinstance(measured["raw_value"], str)
    assert measured["quality_status"] == "rejected"
    json.dumps(measured, allow_nan=False)
    assert assess(task(), {"gap": measured}, "geometry-a")["decision"] == "unknown"


def test_bool_observation_unknown():
    assert result(True)["decision"] == "unknown"


@pytest.mark.parametrize("overrides,reason", [
    ({"calculation_status": "failed"}, "calculation_status:failed"),
    ({"calculation_status": "skipped"}, "calculation_status:skipped"),
    ({"calculation_status": "unsupported"}, "calculation_status:unsupported"),
    ({"quality_status": "unverified"}, "quality_status:unverified"),
    ({"quality_status": "rejected"}, "quality_status:rejected"),
    ({"identity_status": "unverified"}, "identity_status:unverified"),
    ({"identity_status": "mismatch"}, "identity_status:mismatch"),
    ({"unit": "unknown"}, "unit_mismatch"),
    ({"reference_geometry_id": "different-same-formula"}, "reference_geometry_mismatch_or_unbound"),
    ({"definition_id": None}, "definition_missing"),
    ({"backend_version": None}, "backend_version_missing"),
    ({"diagnostics": {"stationary_reference_accepted": False}}, "nonstationary_reference"),
    ({"diagnostics": {"relaxation_converged": False, "termination": "max_steps"}}, "relaxation_not_converged"),
    ({"diagnostics": {"site_mapping_verified": False}}, "site_mapping_unverified"),
])
def test_ineligible_values_are_unknown_not_fail(overrides, reason):
    assessed = result(0, **overrides)
    assert assessed["decision"] == "unknown"
    assert reason in assessed["constraints"][0]["reasons"]


def test_exact_space_group_integer_not_float_or_out_of_range():
    current = task()
    current["measurements"]["gap"] = {"property_id": "space_group_number", "unit": "1"}
    current["constraints"][0].update(operator="eq", value=225)
    for value, expected in [(225, "pass"), (225.0, "unknown"), (231, "unknown"), (221, "fail")]:
        assert result(value, current_task=current, property_id="space_group_number", unit="1")["decision"] == expected


def test_definition_pin_rejects_gamma_for_mesh():
    current = task()
    current["measurements"]["gap"] = {"property_id": "phonon_mesh_min_frequency_THz", "unit": "THz", "definition_id": "mesh.v1"}
    measured = observation(1, property_id="phonon_mesh_min_frequency_THz", unit="THz", definition_id="gamma.v1")
    assessed = assess(current, {"gap": measured}, "geometry-a")
    assert assessed["decision"] == "unknown"
    assert "definition_mismatch" in assessed["constraints"][0]["reasons"]


def test_eligible_failed_gap_and_skipped_kappa_joint_fail():
    current = task("constrained_rank")
    current["mode"] = "screen"
    current.pop("ranking")
    current["constraints"].append({"id": "kappa_above", "measurement": "kappa", "operator": "gt", "value": 10})
    measured = {"gap": observation(1), "kappa": observation(None, property_id="dielectric_total_static_trace_over3", unit="1", calculation_status="skipped", reasons=["upstream_gap_failed"])}
    assessed = assess(current, measured, "geometry-a")
    assert assessed["decision"] == "fail"
    assert [entry["decision"] for entry in assessed["constraints"]] == ["fail", "unknown"]


def test_joint_cross_geometry_cannot_join_same_formula():
    current = task("constrained_rank")
    current["mode"] = "screen"
    current.pop("ranking")
    current["constraints"].append({"id": "kappa_above", "measurement": "kappa", "operator": "gt", "value": 10})
    measured = {
        "gap": observation(3, diagnostics={"formula": "AB"}),
        "kappa": observation(20, property_id="dielectric_total_static_trace_over3", unit="1", reference_geometry_id="geometry-b", diagnostics={"formula": "AB"}),
    }
    assessed = assess(current, measured, "geometry-a")
    assert assessed["decision"] == "unknown"
    assert [entry["decision"] for entry in assessed["constraints"]] == ["pass", "unknown"]


def test_R22_formal_dft_rejects_mlff_prescreen_and_synthetic_evidence():
    current = task()
    current["measurements"]["gap"]["required_fidelity"] = "dft"
    for fidelity in ("mlff", "surrogate", "synthetic"):
        measured = observation(0, physics_fidelity=fidelity, upstream_fidelity=["synthetic"])
        assessed = assess(current, {"gap": measured}, "geometry-a", {"gap": {"family": "dft"}})
        assert assessed["decision"] == "unknown"
        assert "fidelity_mismatch" in assessed["constraints"][0]["reasons"]
        assert measured["benchmark_eligible"] is False


def test_dft_tag_cannot_hide_mixed_upstream():
    current = task()
    current["measurements"]["gap"]["required_fidelity"] = "dft"
    measured = observation(4, physics_fidelity="dft", upstream_fidelity=["dft", "mlff", "synthetic"])
    assessed = assess(current, {"gap": measured}, "geometry-a")
    assert assessed["decision"] == "unknown"
    assert "mixed_upstream_not_dft" in assessed["constraints"][0]["reasons"]


def test_route_definition_and_fidelity_checked():
    assessed = assess(task(), {"gap": observation()}, "geometry-a", {"gap": {"physics_fidelity": "dft", "definition_id": "dft.gap.v1"}})
    assert assessed["decision"] == "unknown"
    assert {"definition_mismatch", "fidelity_mismatch"}.issubset(assessed["constraints"][0]["reasons"])


def test_route_provider_identity_must_match():
    assessed = assess(task(), {"gap": observation()}, "geometry-a", {"gap": {"family": "synthetic", "provider": "different.synthetic_provider"}})
    assert assessed["decision"] == "unknown"
    assert "backend_provider_mismatch" in assessed["constraints"][0]["reasons"]


def test_imported_observation_must_identify_actual_physical_provider():
    imported = observation(3, backend_family="imported", backend_id="bundle_importer")
    route = {"gap": {"family": "synthetic", "provider": "original.synthetic_provider"}}
    assert assess(task(), {"gap": imported}, "geometry-a", route)["decision"] == "unknown"
    imported["source_backend_id"] = "original.synthetic_provider"
    assert assess(task(), {"gap": imported}, "geometry-a", route)["decision"] == "pass"
    assert imported["benchmark_eligible"] is False


@pytest.mark.parametrize("stationary,expected", [(None, "unknown"), (False, "unknown"), (True, "pass")])
def test_hybrid_stationarity_requires_explicit_accepted_evidence(stationary, expected):
    diagnostics = {} if stationary is None else {"stationary_reference_accepted": stationary}
    measured = observation(3, diagnostics=diagnostics)
    route = {"gap": {"family": "synthetic", "provider": "test.synthetic", "requires_stationary_reference": True}}
    assert assess(task(), {"gap": measured}, "geometry-a", route)["decision"] == expected
    assert measured["backend_family"] == "synthetic"
    assert measured["benchmark_eligible"] is False


def test_negative_raw_gap_never_physical_fail():
    assessed = result(-1)
    assert assessed["measurements"]["gap"]["value"] == -1
    assert assessed["decision"] == "unknown"


def test_synthetic_never_benchmark_eligible_even_released():
    measured = observation(3, protocol_phase="released", benchmark_eligible=True)
    assert measured["benchmark_eligible"] is False


def test_all_unknown_and_zero_denominators():
    metrics = aggregate(task(), [result(None) for _ in range(3)])
    assert metrics["verified_yield"]["value"] == 0
    assert metrics["determined_fraction"]["value"] == 0
    assert metrics["pass_fraction_determined"]["value"] is None
    empty = aggregate(task(), [])
    assert empty["verified_yield"]["value"] is None
    assert empty["verified_yield"]["status"] == "undefined_zero_denominator"


def test_full_denominator_duplicates_and_invalid_inputs():
    invalid = dict(item_id="broken", input_status="invalid", **assess(task(), {}, None))
    metrics = aggregate(task(), [result(3, "duplicate"), result(3, "duplicate"), result(1), result(None), invalid])
    assert (metrics["n_submitted"], metrics["n_pass"], metrics["n_fail"], metrics["n_unknown"]) == (5, 2, 1, 2)
    assert metrics["verified_yield"]["value"] == 0.4
    assert metrics["coverage"]["gap"]["value"] == 0.6
    assert metrics["n_input_invalid"] == 1
    assert metrics["constraints"]["gap_above"]["n_unknown"] == 2
    assert metrics["generation_cost"]["value"] is None
    assert metrics["novelty"]["value"] is None
    assert metrics["sun"]["value"] is None


def test_invalid_input_cannot_inherit_stale_pass():
    invalid = result(3)
    invalid["input_status"] = "invalid"
    metrics = aggregate(task(), [invalid])
    assert metrics["n_unknown"] == 1
    assert metrics["n_pass"] == 0
    assert metrics["constraints"]["gap_above"]["n_unknown"] == 1


def test_measure_reports_distributions_without_pass_rates():
    current = task("measure")
    measured = [result(3, current_task=current), result(None, current_task=current)]
    assert measured[0]["decision"] is None
    metrics = aggregate(current, measured)
    assert metrics["distributions"]["gap"]["mean"] == 3
    assert metrics["distributions"]["gap"]["observed_count"] == 1
    assert metrics["coverage"]["gap"]["value"] == 0.5
    assert not {"n_pass", "verified_yield", "pass_fraction_determined"} & metrics.keys()


def test_distribution_large_finite_values_stays_strict_json():
    current = task("measure")
    metrics = aggregate(current, [result(1e308, current_task=current), result(1e308, current_task=current)])
    assert metrics["distributions"]["gap"]["mean"] == 1e308
    assert metrics["distributions"]["gap"]["median"] == 1e308
    json.dumps(metrics, allow_nan=False)


def test_rank_distinguishes_feasible_rankable_and_has_no_highk_success():
    current = task("constrained_rank")
    results = []
    for index, (gap, kappa) in enumerate([(3, 20), (3, None), (1, 1000), (3, 30)]):
        measured = {"gap": observation(gap), "kappa": observation(kappa, property_id="dielectric_total_static_trace_over3", unit="1", definition_id="synthetic.kappa.v1")}
        results.append(dict(item_id=str(index), input_status="valid", **assess(current, measured, "geometry-a")))
    metrics = aggregate(current, results)
    assert metrics["n_feasible"] == 3
    assert metrics["n_rankable"] == 2
    assert [entry["item_id"] for entry in metrics["ranking"]["items"]] == ["3", "0"]
    assert metrics["ranking"]["distribution"]["mean"] == 25
    assert "verified_yield" not in metrics
    assert "highk_success" not in metrics


def test_reassessment_changes_decision_without_changing_measurement():
    current = task()
    measured = observation(2.5)
    first = assess(current, {"gap": measured}, "geometry-a")
    stricter = deepcopy(current)
    stricter["constraints"][0]["value"] = 3
    second = assess(stricter, {"gap": measured}, "geometry-a")
    assert first["decision"] == "pass"
    assert second["decision"] == "fail"
    assert first["measurements"] == second["measurements"]
