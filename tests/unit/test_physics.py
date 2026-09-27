"""Synthetic arrays validate pure reducers, never a real phonon/DFT backend."""

import json

import pytest

from crystargetbench.physics import dielectric_response, eos_input_quality, phonon_statistics, pressure_to_gpa


@pytest.mark.parametrize("frequencies", [[], [[]], [[1, float("nan")]], [[1, float("inf")]], [[True, 1]], [[1, 2], [3]]])
def test_empty_or_nonfinite_frequencies_remain_unknown(frequencies):
    result = phonon_statistics(frequencies, scope="path")
    assert result["min_frequency_THz"] is None
    assert result["quality_status"] == "unverified"
    json.dumps(result, allow_nan=False)


def test_signed_negative_modes_not_removed_and_path_fraction_named():
    result = phonon_statistics([[-2, -0.001, 1], [0, 1, 2]], scope="path")
    assert result["min_frequency_THz"] == -2
    assert result["imaginary_fraction"] == pytest.approx(2 / 6)
    assert result["raw_frequencies_THz"][0] == [-2, -0.001, 1]
    assert result["fraction_definition"] == "path_sampling_mode_fraction"


def test_irreducible_mesh_uses_weights():
    result = phonon_statistics([[-1, 1], [1, 1]], scope="mesh", weights=[1, 3])
    assert result["imaginary_fraction"] == 1 / 8
    assert result["fraction_definition"] == "weighted_mesh_mode_fraction"
    assert phonon_statistics([[-1, 1], [1, 1]], scope="mesh")["quality_status"] == "unverified"


def test_large_finite_mesh_weights_do_not_overflow_fraction():
    result = phonon_statistics([[-1], [-1]], scope="mesh", weights=[1e308, 1e308])
    assert result["imaginary_fraction"] == 1
    json.dumps(result, allow_nan=False)


def test_first_nonzero_q_is_not_gamma():
    result = phonon_statistics([[1, 2]], scope="path", qpoints=[[0.25, 0, 0]])
    assert result["gamma_available"] is False
    assert phonon_statistics([[1, 2]], scope="gamma", qpoints=[[0.25, 0, 0]])["quality_status"] == "unverified"
    assert phonon_statistics([[1, 2]], scope="gamma", qpoints=[[0, 0, 0]])["scope"] == "gamma"


def test_frequency_regularization_must_be_explicit_and_never_changes_raw():
    assert phonon_statistics([[-1, 1]], scope="path", frequency_floor=0.1)["quality_status"] == "unverified"
    result = phonon_statistics([[-1, 1]], scope="path", frequency_floor=0.1, regularization_definition="synthetic.floor.v1")
    assert result["min_frequency_THz"] == -1
    assert result["raw_frequencies_THz"] == [[-1, 1]]


@pytest.mark.parametrize("floor", [float("nan"), float("inf"), 0, -1, True, 10**400])
def test_invalid_frequency_floor_is_unknown_and_json_safe(floor):
    result = phonon_statistics([[1]], scope="path", frequency_floor=floor, regularization_definition="explicit.test.v1")
    assert result["quality_status"] == "unverified"
    json.dumps(result, allow_nan=False)
    tensor_result = dielectric_response(total=diagonal(2), frequency_floor=floor, regularization_definition="explicit.test.v1")
    assert tensor_result["value"] is None
    json.dumps(tensor_result, allow_nan=False)


def diagonal(a, b=None, c=None):
    return [[a, 0, 0], [0, a if b is None else b, 0], [0, 0, a if c is None else c]]


def test_dielectric_missing_components_never_filled_with_identity_or_zero():
    assert dielectric_response(ionic=diagonal(10))["value"] is None
    assert dielectric_response(electronic=diagonal(2))["value"] is None


def test_dielectric_total_not_double_added_and_trace_not_nine_element_mean():
    result = dielectric_response(total=diagonal(3, 6, 9), electronic=diagonal(2), upstream_fidelity=["synthetic"])
    assert result["value"] == 6
    assert result["tensor"] == diagonal(3, 6, 9)
    assert result["principal_values"] == [3, 6, 9]
    assert result["anisotropy"] == 3
    assert result["physics_fidelity"] == "synthetic"


@pytest.mark.parametrize("options,reason", [
    ({"gamma_available": False}, "gamma_response_unavailable"),
    ({"frequency": 0.1}, "static_frequency_zero_missing"),
    ({"frequency_floor": 0.1}, "undeclared_frequency_regularization"),
    ({"ionic_definition": "nominal"}, "nominal_proxy_is_not_total_response"),
])
def test_dielectric_no_forbidden_fallbacks(options, reason):
    result = dielectric_response(electronic=diagonal(2), ionic=diagonal(10), **options)
    assert result["value"] is None
    assert reason in result["reasons"]


def test_mixed_response_lineage_cannot_be_called_dft():
    # Typed lineage reduction only. These numbers are synthetic test arrays.
    result = dielectric_response(electronic=diagonal(2), ionic=diagonal(10), upstream_fidelity=["dft", "mlff"])
    assert result["value"] == 12
    assert result["physics_fidelity"] == "mixed"


def test_invalid_tensor_not_ranked():
    assert dielectric_response(total=diagonal(-1))["value"] is None
    assert dielectric_response(total=[[1, 1, 0], [0, 1, 0], [0, 0, 1]])["value"] is None


def test_eos_too_few_points_and_no_invented_fit():
    result = eos_input_quality([1, 2], [0.1, 0.2])
    assert result["quality_status"] == "unverified"
    assert result["bulk_modulus_GPa"] is None
    result = eos_input_quality([1, 2, 3, 4, 5], [5, 2, 1, 2, 5])
    assert result["input_quality_status"] == "accepted"
    assert result["fit_status"] == "not_performed"
    assert result["bulk_modulus_GPa"] is None


def test_bulk_modulus_units_explicit():
    assert pressure_to_gpa(1, "eV/angstrom^3") == pytest.approx(160.2176634)
    assert pressure_to_gpa(20, "GPa") == 20
    with pytest.raises(ValueError):
        pressure_to_gpa(1, "unknown")
    with pytest.raises(ValueError):
        pressure_to_gpa(1e308, "eV/angstrom^3")
    with pytest.raises(ValueError):
        pressure_to_gpa(10**400, "GPa")
