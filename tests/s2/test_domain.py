"""Native geometry metamorphisms and explicit limits of the preserved S1 guard."""
import json

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from crystargetbench.budget import Budget, BudgetExceeded, DEFAULT_TEST_LIMITS
from crystargetbench.geometry import atoms_from_snapshot, snapshot_from_atoms
from crystargetbench.native import analyze_symmetry
from crystargetbench.structures import load_submission, read_structure


def write_input(path, atoms):
    path.write_text(json.dumps({"species": atoms.get_chemical_symbols(),
        "cell": atoms.cell.array.tolist(), "scaled_positions": atoms.get_scaled_positions(wrap=False).tolist(),
        "pbc": atoms.pbc.tolist()}))
    return path


def sg(snapshot):
    return analyze_symmetry(snapshot, {"symprec_A": .001, "angle_tolerance_deg": 1})["space_group_number"]


def test_C05_origin_axis_unimodular_and_replica_native_equivalence(tmp_path):
    original = bulk("Si", "diamond", a=5.43, cubic=True)
    shifted = original.copy()
    shifted.positions += [.73, -2.81, 4.31]
    permuted = original.copy()
    permuted.set_cell(original.cell.array[[1, 2, 0]], scale_atoms=False)
    basis = original.copy()
    unimodular = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    basis.set_cell(unimodular @ original.cell.array, scale_atoms=False)
    replicas = original.repeat((2, 3, 1))
    states = [read_structure(write_input(tmp_path / f"case{i}.json", atoms))
              for i, atoms in enumerate([original, shifted, permuted, basis, replicas])]
    assert [sg(state) for state in states] == [227] * 5
    for index in (1, 2, 4):
        np.testing.assert_allclose(sorted(states[0]["domain_checks"]["maximum_empty_slab_A"]),
            sorted(states[index]["domain_checks"]["maximum_empty_slab_A"]), atol=1e-12)
    assert states[0]["domain_checks"]["general_unimodular_basis_invariance_guaranteed"] is False


def test_C05_counterexample_documents_basis_dependent_admission_not_instability(tmp_path):
    long_cell = Atoms("Ar", cell=[4, 4, 24], scaled_positions=[[0, 0, 0]], pbc=True)
    equivalent = long_cell.copy()
    unimodular = np.array([[1, 0, 1], [0, 1, 1], [-1, -1, -1]])
    assert round(np.linalg.det(unimodular)) == 1
    equivalent.set_cell(unimodular @ long_cell.cell.array, scale_atoms=False)
    rejected, admitted = load_submission([write_input(tmp_path / "long.json", long_cell),
                                          write_input(tmp_path / "equivalent.json", equivalent)])
    assert rejected["input_status"] == "unsupported"
    assert "basis-dependent" in rejected["reasons"][0]
    assert admitted["input_status"] == "valid"
    assert admitted["input_geometry"]["domain_checks"]["stability_evidence"] is False
    # Native SG itself agrees when supplied the two equivalent geometry snapshots.
    assert sg(snapshot_from_atoms(long_cell)) == sg(snapshot_from_atoms(equivalent)) == 123
    assert "decision" not in rejected and "stable" not in rejected


def test_C06_derived_supercell_bypasses_input_guard_but_obeys_atom_budget(tmp_path):
    derived = Atoms("Ar", cell=[4, 4, 24], scaled_positions=[[0, 0, 0]], pbc=True).repeat((2, 1, 1))
    assert load_submission([write_input(tmp_path / "submitted.json", derived)])[0]["input_status"] == "unsupported"
    snapshot = snapshot_from_atoms(derived, parent="reference", stage="phonon_supercell")
    assert len(atoms_from_snapshot(snapshot)) == 2
    assert "domain_checks" not in snapshot
    limits = dict(DEFAULT_TEST_LIMITS, max_supercell_atoms=1)
    budget = Budget(tmp_path / "budget.json", limits, identity="synthetic_resource_guard")
    with pytest.raises(BudgetExceeded, match="max_supercell_atoms"):
        budget.check_atoms(len(derived))
    assert budget.snapshot()["reserved_evaluations"] == 0
