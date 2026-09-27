"""Production recipe orchestration with explicitly synthetic analytic evidence."""
from copy import deepcopy

import pytest
import numpy as np

from crystargetbench.scientific import execute_item, execute_submission
from tools.generate_s2_numerical import (numerical_configuration, numerical_context,
                                         numerical_parameters, write_fixture_submission)
from .analytic_eos import AnalyticLatticeBM3
from .analytic_lattice import lattice_atoms


def test_C13_C14_combined_shared_model_energy_forces_stress_and_extensivity():
    atoms = lattice_atoms()
    atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)
    atoms.positions[1] += [0.012, -0.007, 0.009]
    atoms.calc = AnalyticLatticeBM3()
    step = 1e-6
    force = atoms.get_forces()[1, 0]
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[1, 0] += step
    minus.positions[1, 0] -= step
    plus.calc, minus.calc = AnalyticLatticeBM3(), AnalyticLatticeBM3()
    assert force == pytest.approx(-(plus.get_potential_energy() - minus.get_potential_energy()) / (2 * step), abs=1e-8)
    stress = atoms.get_stress(voigt=False)
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]:
        deformation = np.zeros((3, 3))
        deformation[i, j] = deformation[j, i] = step
        plus, minus = atoms.copy(), atoms.copy()
        plus.set_cell(atoms.cell @ (np.eye(3) + deformation), scale_atoms=True)
        minus.set_cell(atoms.cell @ (np.eye(3) - deformation), scale_atoms=True)
        plus.calc, minus.calc = AnalyticLatticeBM3(), AnalyticLatticeBM3()
        factor = 1 if i == j else 2
        derivative = (plus.get_potential_energy() - minus.get_potential_energy()) / (2 * step * atoms.get_volume() * factor)
        assert stress[i, j] == pytest.approx(derivative, abs=2e-9)
    original = lattice_atoms()
    original.set_cell(original.cell * 1.01, scale_atoms=True)
    repeated = original.repeat((2, 1, 1))
    original.calc, repeated.calc = AnalyticLatticeBM3(), AnalyticLatticeBM3()
    assert repeated.get_potential_energy() == pytest.approx(2 * original.get_potential_energy(), abs=1e-11)
    np.testing.assert_allclose(repeated.get_stress(), original.get_stress(), atol=1e-12)


def test_C19_C43_C45_shared_reference_full_denominator_and_explicit_synthetic_evidence(tmp_path):
    items = write_fixture_submission(tmp_path / "inputs")
    task, resolved = numerical_configuration()
    context = numerical_context(tmp_path / "run")
    results, metrics, stats = execute_submission(task, items, resolved, context)
    assert metrics["n_submitted"] == 3
    assert (metrics["n_pass"], metrics["n_fail"], metrics["n_unknown"]) == (2, 0, 1), results
    assert metrics["n_input_invalid"] == 1
    assert metrics["verified_yield"]["value"] == 2 / 3
    for result in results[:2]:
        reference_id = result["reference_geometry"]["geometry_id"]
        assert result["measurements"]["B"]["reference_geometry_id"] == reference_id
        assert result["measurements"]["phonon"]["reference_geometry_id"] == reference_id
        assert result["evidence_kind"] == "analytic_test"
        for measurement in result["measurements"].values():
            assert measurement["physics_fidelity"] == "synthetic"
            assert measurement["benchmark_eligible"] is False
            assert measurement["backend_id"] == "analytic_test"
    assert results[2]["reference_geometry"] is None
    assert results[2]["measurements"]["B"]["value"] is None
    assert stats["physics_calls"] == 0
    assert stats["backend_structure_evaluations"] > 0
    assert stats["stage_cache_hits"]["reference-relax"] >= 1
    assert stats["stage_cache_hits"]["eos-point"] == 7

    threshold_task, threshold_resolved = numerical_configuration(bulk_threshold=100)
    again = numerical_context(tmp_path / "run")
    _, threshold_metrics, threshold_stats = execute_submission(threshold_task, items, threshold_resolved, again)
    assert threshold_stats["backend_structure_evaluations"] == 0
    assert threshold_stats["stage_computations"] == {}
    assert (threshold_metrics["n_pass"], threshold_metrics["n_fail"], threshold_metrics["n_unknown"]) == (0, 2, 1)
    assert threshold_resolved["comparability_digest"] != resolved["comparability_digest"]

    sampling = numerical_parameters()
    sampling["phonon"]["mesh"] = [6, 6, 6]
    sampling_task, sampling_resolved = numerical_configuration(sampling)
    third = numerical_context(tmp_path / "run")
    _, _, sampling_stats = execute_submission(sampling_task, items, sampling_resolved, third)
    assert sampling_stats["backend_structure_evaluations"] == 0
    assert set(sampling_stats["stage_computations"]) == {"ctb.phonopy.sampling.v1"}


def test_C36_C45_synthetic_context_cannot_fulfill_mlff_dft_or_elastic_routes(tmp_path):
    items = write_fixture_submission(tmp_path / "inputs")
    task, resolved = numerical_configuration()
    for family in ("mlff", "dft"):
        changed = deepcopy(resolved)
        changed["routes"]["B"]["family"] = family
        context = numerical_context(tmp_path / family)
        with pytest.raises(ValueError, match="no substitution"):
            execute_item(task, items[0], changed, context)
        assert context.stats["backend_structure_evaluations"] == 0
    changed_task = deepcopy(task)
    changed_task["measurements"]["B"]["property_id"] = "bulk_modulus_voigt_GPa"
    context = numerical_context(tmp_path / "elastic")
    with pytest.raises(ValueError, match="no substitution"):
        execute_item(changed_task, items[0], resolved, context)
    assert context.stats["backend_structure_evaluations"] == 0
