"""Actual ASE/Phonopy numerical tests using a test-only analytic lattice."""
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("phonopy")
pytest.importorskip("seekpath")

from crystargetbench.cache import StageCache, stage_key
from crystargetbench.geometry import atoms_from_snapshot, snapshot_from_atoms
from crystargetbench.identity import digest
from crystargetbench.recipes.phonon import (DEFAULTS, assemble_forces, build_ifc,
    sample_phonons, validate_frequencies, versions)
from .analytic_lattice import AnalyticLattice, lattice_atoms


class AnalyticBudget:
    def __init__(self, maximum_atoms=128, maximum_displacements=1000):
        self.maximum_atoms, self.maximum_displacements = maximum_atoms, maximum_displacements
        self.reserved = []

    def check_atoms(self, count):
        if count > self.maximum_atoms:
            raise RuntimeError("max_supercell_atoms")

    def reserve_displacement(self, identifier):
        if len(self.reserved) >= self.maximum_displacements:
            raise RuntimeError("displacement_budget_exhausted")
        self.reserved.append(identifier)


class AnalyticContext:
    """Inject real ASE evaluation; only accounting/storage are test scaffolding."""
    def __init__(self, root, calculator=None, budget=None):
        self.cache = StageCache(root)
        self.calculator = calculator or AnalyticLattice()
        self.budget = budget or AnalyticBudget()
        self.calls, self.hits, self.stages = 0, 0, []
        self.responses = []
        self.backend = {"id": "analytic_lattice", "family": "synthetic", "version": "test-v1",
                        "assets": {}, "spring": self.calculator.spring,
                        "same_spring": self.calculator.same_spring, "drift": self.calculator.drift.tolist()}

    def cached(self, stage, geometry, parameters, dependencies, compute):
        key = stage_key(geometry=geometry, recipe=stage, parameters=parameters,
                        dependencies=dependencies, backend=self.backend)
        result = self.cache.get(key)
        if result is None:
            result = compute()
            self.cache.put(key, result)
            self.stages.append(stage)
        else:
            self.hits += 1
        return result

    def evaluate(self, atoms, node_id, mapping=None):
        self.calls += 1
        geometry = snapshot_from_atoms(atoms)
        atoms = atoms.copy()
        atoms.calc = self.calculator
        result = {"geometry_id": geometry["geometry_id"], "node_id": node_id,
            "mapping_digest": digest(mapping), "energy_eV": atoms.get_potential_energy(),
            "forces_eV_A": atoms.get_forces().tolist(), "stress_eV_A3": atoms.get_stress(voigt=False).tolist(),
            "force_dtype": "float64", "physics_fidelity": "synthetic", "evidence_kind": "analytic_test"}
        self.responses.append(result)
        return result


def parameters(**updates):
    return {**deepcopy(DEFAULTS), "is_plusminus": True, "is_diagonal": False,
            "path_points_parameter": 5, **updates}


def artifact(tmp_path, *, model=None, atoms=None, params=None):
    model = model or AnalyticLattice()
    context = AnalyticContext(tmp_path, model)
    reference = snapshot_from_atoms(atoms if atoms is not None else lattice_atoms(), stage="reference")
    return build_ifc(reference, context, params or parameters()), context


def test_C20_actual_displacements_forces_ifc_match_independent_hessian(tmp_path):
    result, context = artifact(tmp_path)
    from ase import Atoms
    supercell = result["supercell"]
    atoms = Atoms(supercell["species"], cell=supercell["cell"], scaled_positions=supercell["positions"], pbc=True)
    expected = context.calculator.exact_ifc(atoms)
    np.testing.assert_allclose(result["raw_force_constants_eV_A2"], expected, atol=2e-10, rtol=1e-10)
    assert context.calls == 1 + len(result["displacements"])
    assert result["physics_fidelity"] == "synthetic"
    assert result["evidence_kind"] == "analytic_test"
    assert not result["benchmark_eligible"] and not result["mlff_live_tested"]


def test_C20_nonzero_q_optical_and_acoustic_spectrum_matches_formula(tmp_path):
    result, context = artifact(tmp_path)
    sampled = sample_phonons(result, parameters(mesh=[3, 3, 3], is_mesh_symmetry=False), "mesh")
    expected = context.calculator.exact_frequencies(sampled["qpoints"], result["frequency_conversion_THz"])
    np.testing.assert_allclose(sampled["frequencies_THz"], expected, atol=2e-6, rtol=1e-9)
    assert sampled["branch_count"] == 6
    gamma = sample_phonons(result, parameters(), "gamma")
    assert max(abs(value) for value in gamma["frequencies_THz"][0][:3]) < 2e-6
    assert min(gamma["frequencies_THz"][0][3:]) > 1


def test_C13_analytic_forces_equal_negative_energy_difference():
    atoms = lattice_atoms().repeat((2, 2, 2))
    atoms.positions[0] += [0.03, -0.02, 0.01]
    atoms.calc = AnalyticLattice()
    force = atoms.get_forces()[0, 1]
    h = 1e-6
    atoms.positions[0, 1] += h
    plus = atoms.get_potential_energy()
    atoms.positions[0, 1] -= 2 * h
    minus = atoms.get_potential_energy()
    assert force == pytest.approx(-(plus - minus) / (2 * h), abs=1e-9)


def test_C21_out_of_order_results_join_by_identity_and_reject_missing(tmp_path):
    result, context = artifact(tmp_path)
    responses = context.responses[1:]
    np.testing.assert_array_equal(assemble_forces(result["displacements"], list(reversed(responses))), result["raw_forces_eV_A"])
    with pytest.raises(ValueError, match="Missing"):
        assemble_forces(result["displacements"], responses[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        assemble_forces(result["displacements"], responses + [responses[0]])


def test_C22_same_species_wrong_order_is_rejected_by_mapping_identity(tmp_path):
    result, context = artifact(tmp_path)
    responses = deepcopy(context.responses[1:])
    record = result["displacements"][0]
    mapping = deepcopy(record["mapping"])
    sites = mapping["ordered_site_ids"]
    sites[0], sites[1] = sites[1], sites[0]
    responses[0]["mapping_digest"] = digest(mapping)
    responses[0]["forces_eV_A"][0], responses[0]["forces_eV_A"][1] = responses[0]["forces_eV_A"][1], responses[0]["forces_eV_A"][0]
    with pytest.raises(ValueError, match="mapping"):
        assemble_forces(result["displacements"], responses)
    responses = deepcopy(context.responses[1:])
    responses[0]["geometry_id"] = "0" * 64
    with pytest.raises(ValueError, match="geometry"):
        assemble_forces(result["displacements"], responses)


def test_C22_consistent_same_species_unitcell_permutation_preserves_spectrum(tmp_path):
    atoms = lattice_atoms().repeat((2, 1, 1))
    reordered = atoms[[2, 1, 0, 3]]
    p = parameters(supercell_matrix=[[1, 0, 0], [0, 2, 0], [0, 0, 2]])
    first, _ = artifact(tmp_path / "first", atoms=atoms, params=p)
    second, _ = artifact(tmp_path / "second", atoms=reordered, params=p)
    first_sample = sample_phonons(first, p, "mesh")
    second_sample = sample_phonons(second, p, "mesh")
    np.testing.assert_allclose(first_sample["frequencies_THz"], second_sample["frequencies_THz"], atol=2e-6)
    assert first["reference_geometry_id"] != second["reference_geometry_id"]


def test_C23_nondiagonal_supercell_mapping_and_branch_count(tmp_path):
    p = parameters(supercell_matrix=[[2, 1, 0], [0, 2, 0], [0, 0, 2]])
    result, context = artifact(tmp_path, params=p)
    sample = sample_phonons(result, p, "mesh")
    assert len(result["supercell"]["species"]) == 16
    assert len(set(result["mappings"]["ordered_site_ids"])) == 16
    assert sample["branch_count"] == 3 * len(result["primitive"]["species"]) == 6
    q_cart = np.asarray(sample["qpoints_cartesian_Ainv"])
    cubic_q = q_cart @ lattice_atoms().cell.array.T / (2 * np.pi)
    expected = context.calculator.exact_frequencies(cubic_q, result["frequency_conversion_THz"])
    np.testing.assert_allclose(sample["frequencies_THz"], expected, atol=2e-6)


def test_C24_seekpath_original_basis_matches_cartesian_analytic_spectrum(tmp_path):
    atoms = lattice_atoms()
    atoms.set_cell(np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ atoms.cell.array, scale_atoms=False)
    atoms.wrap()
    p = parameters(primitive_matrix=np.eye(3).tolist())
    result, context = artifact(tmp_path, atoms=atoms, params=p)
    sample = sample_phonons(result, p, "path")
    actual_cartesian = np.asarray(sample["qpoints"]) @ np.asarray(sample["reciprocal_basis_Ainv"])
    np.testing.assert_allclose(actual_cartesian, sample["qpoints_cartesian_Ainv"], atol=1e-12)
    cubic_q = actual_cartesian @ lattice_atoms().cell.array.T / (2 * np.pi)
    expected = context.calculator.exact_frequencies(cubic_q, result["frequency_conversion_THz"])
    np.testing.assert_allclose(sample["frequencies_THz"], expected, atol=2e-6)
    assert sample["sampling_parameters"]["path_method"] == "seekpath.get_path_orig_cell"
    assert sample["segments"] and sample["labels"]


def test_C25_negative_optical_modes_are_retained(tmp_path):
    result, _ = artifact(tmp_path, model=AnalyticLattice(spring=-0.2, same_sublattice_spring=1.0))
    sample = sample_phonons(result, parameters(), "gamma")
    frequencies = np.asarray(sample["frequencies_THz"])[0]
    assert np.count_nonzero(frequencies < -1e-3) == 3
    assert len(frequencies) == 6
    assert sample["value"] < 0
    assert sample["signed_frequencies_preserved"]


@pytest.mark.parametrize("frequencies", [[], [[float("nan")] * 6], [[float("inf")] * 6], [[0.0] * 3]])
def test_C26_bad_frequencies_are_rejected_before_reduction(frequencies):
    result = validate_frequencies(frequencies, [[0, 0, 0]], 2, scope="gamma", weights=[1])
    assert result["quality_status"] == "rejected" and result["value"] is None


def test_C27_drift_correction_and_asr_are_traced(tmp_path):
    result, _ = artifact(tmp_path, model=AnalyticLattice(drift=(0.001, -0.002, 0.003)))
    np.testing.assert_allclose(result["force_drift_eV_A"], np.tile([0.001, -0.002, 0.003], (len(result["displacements"]), 1)), atol=1e-13)
    corrected = np.asarray(result["corrected_forces_eV_A"])
    np.testing.assert_allclose(corrected.mean(axis=1), 0, atol=1e-15)
    assert result["ifc_diagnostics"]["final"]["asr_max_abs_eV_A2"] < 1e-12
    assert result["ifc_diagnostics"]["final"]["permutation_max_abs_eV_A2"] < 1e-12
    assert result["reference_force_max_eV_A"] > 0
    assert "raw_force_constants_eV_A2" in result and "drift_corrected_force_constants_eV_A2" in result


def test_C28_irreducible_weighted_mesh_matches_full_mesh(tmp_path):
    result, _ = artifact(tmp_path, model=AnalyticLattice(spring=-0.2, same_sublattice_spring=1.0))
    p = parameters(mesh=[4, 4, 4], gamma_center=False)
    reduced = sample_phonons(result, p, "mesh")
    complete = sample_phonons(result, {**p, "is_mesh_symmetry": False}, "mesh")
    assert reduced["weight_sum"] == complete["weight_sum"] == 64
    assert len(reduced["qpoints"]) < len(complete["qpoints"]) == 64
    assert 0 < reduced["statistics"]["imaginary_fraction"] < 1
    assert reduced["statistics"]["imaginary_fraction"] == pytest.approx(complete["statistics"]["imaginary_fraction"], abs=1e-14)
    expanded = np.concatenate([np.tile(row, weight) for row, weight in zip(reduced["frequencies_THz"], reduced["weights"])])
    np.testing.assert_allclose(np.sort(expanded), np.sort(np.asarray(complete["frequencies_THz"]).ravel()), atol=2e-6)


def test_C29_scopes_are_distinct_and_gamma_is_not_path(tmp_path):
    result, _ = artifact(tmp_path)
    samples = [sample_phonons(result, parameters(), scope) for scope in ("gamma", "path", "mesh")]
    assert {sample["scope"] for sample in samples} == {"gamma", "path", "mesh"}
    assert len({sample["artifact_digest"] for sample in samples}) == 3
    assert len(samples[0]["qpoints"]) == 1 < len(samples[1]["qpoints"])
    with pytest.raises(ValueError, match="scope"):
        sample_phonons(result, parameters(), "dispersion")


def test_C30_nac_refused_and_born_file_never_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "BORN").write_text("This must never be parsed or used")
    result, context = artifact(tmp_path / "cache")
    assert sample_phonons(result, parameters(), "gamma")["nac"] == "none"
    calls = context.calls
    with pytest.raises(ValueError, match="NAC"):
        build_ifc(result["reference"], context, parameters(nac="from_dft_born_and_electronic"))
    assert context.calls == calls


def test_C38_q_sampling_reuses_ifc_and_all_force_calls(tmp_path):
    result, context = artifact(tmp_path)
    calls, reservations = context.calls, len(context.budget.reserved)
    rebuilt = build_ifc(result["reference"], context, parameters(mesh=[7, 7, 7]))
    assert rebuilt["artifact_digest"] == result["artifact_digest"]
    assert context.calls == calls and len(context.budget.reserved) == reservations
    mesh = sample_phonons(result, parameters(mesh=[3, 3, 3]), "mesh")
    path = sample_phonons(result, parameters(path_points_parameter=3), "path")
    assert mesh["artifact_digest"] != path["artifact_digest"]
    assert context.calls == calls


def test_C39_mass_change_reuses_forces_and_changes_frequencies(tmp_path):
    result, context = artifact(tmp_path)
    calls = context.calls
    doubled = parameters(masses=[40.0, 80.0])
    other = build_ifc(result["reference"], context, doubled)
    assert context.calls == calls
    first = sample_phonons(result, parameters(), "gamma")
    second = sample_phonons(other, doubled, "gamma")
    np.testing.assert_allclose(np.asarray(second["frequencies_THz"])[:, 3:], np.asarray(first["frequencies_THz"])[:, 3:] / np.sqrt(2), atol=1e-8)
    assert result["artifact_digest"] != other["artifact_digest"]


@pytest.mark.parametrize("changed_input", ["supercell", "displacement", "model"])
def test_C39_supercell_displacement_and_model_invalidate_actual_force_ifc_cache(tmp_path, changed_input):
    before, _ = artifact(tmp_path)
    updates = {"supercell": {"supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 3]]},
               "displacement": {"displacement_A": 0.02}, "model": {}}[changed_input]
    model = AnalyticLattice(spring=2.0) if changed_input == "model" else AnalyticLattice()
    context = AnalyticContext(tmp_path, calculator=model)
    changed_parameters = parameters(**updates)
    after = build_ifc(before["reference"], context, changed_parameters)
    assert context.calls == 1 + len(after["displacements"])
    assert before["artifact_digest"] != after["artifact_digest"]
    assert "ctb.phonopy.finite_displacement.v1" in context.stages
    if changed_input in {"supercell", "displacement"}:
        assert {row["displacement_id"] for row in before["displacements"]}.isdisjoint(
            {row["displacement_id"] for row in after["displacements"]})
    sampled = sample_phonons(after, changed_parameters, "gamma")
    expected = model.exact_frequencies(sampled["qpoints"], after["frequency_conversion_THz"])
    np.testing.assert_allclose(sampled["frequencies_THz"], expected, atol=2e-6, rtol=1e-9)
    calls = context.calls
    repeated = build_ifc(before["reference"], context, changed_parameters)
    assert context.calls == calls
    assert repeated["artifact_digest"] == after["artifact_digest"]


def test_C41_resource_and_displacement_budgets_block_before_submission(tmp_path):
    reference = snapshot_from_atoms(lattice_atoms())
    context = AnalyticContext(tmp_path / "atoms", budget=AnalyticBudget(maximum_atoms=2))
    with pytest.raises(RuntimeError, match="max_supercell_atoms"):
        build_ifc(reference, context, parameters())
    assert context.calls == 0
    context = AnalyticContext(tmp_path / "displacements", budget=AnalyticBudget(maximum_displacements=1))
    with pytest.raises(RuntimeError, match="displacement_budget"):
        build_ifc(reference, context, parameters())
    assert context.calls == 2  # reference supercell and exactly one displacement


def test_C40_completed_displacements_survive_interruption_and_resume(tmp_path):
    reference = snapshot_from_atoms(lattice_atoms())
    context = AnalyticContext(tmp_path, budget=AnalyticBudget(maximum_displacements=1))
    with pytest.raises(RuntimeError):
        build_ifc(reference, context, parameters())
    context.budget.maximum_displacements = 1000
    result = build_ifc(reference, context, parameters())
    assert context.calls == len(result["displacements"]) + 1
    assert len(context.budget.reserved) == len(result["displacements"])


def test_versions_are_explicitly_locked():
    lock = versions()
    assert lock["phonopy"] == "2.38.2" and lock["seekpath"] == "2.1.0"
