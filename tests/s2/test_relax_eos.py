"""Real ASE optimization and BM3 fitting on test-only analytical energies."""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest
from ase import Atoms

from crystargetbench.geometry import atoms_from_snapshot, snapshot_from_atoms
from crystargetbench.identity import digest
from crystargetbench.physics import EV_PER_ANGSTROM3_TO_GPA
from crystargetbench.recipes.eos import eos, fit_birch_murnaghan
from crystargetbench.recipes.relaxation import check_stationarity, relax
from .analytic_eos import AnalyticEOSCalculator, manufactured_bm3


def relax_parameters(**changes):
    values = {"require_converged": True, "optimizer": "FIRE", "filter": "FrechetCellFilter",
              "cell_mask": [True] * 6, "fmax_eV_per_A": 1e-5, "stress_max_GPa": 0.002,
              "max_steps": 1200, "target_pressure_GPa": 0.0, "dt": 0.1, "dtmax": 1.0,
              "maxstep_A": 0.15, "max_wall_seconds": 30.0}
    values.update(changes)
    return values


def eos_parameters(**changes):
    values = {"volume_factors": [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06],
              "fit": "birch_murnaghan_3rd_order", "internal_relaxation": True,
              "fixed_cell_shape": True, "fit_rms_max_eV": 1e-6,
              "max_relative_volume_shift": 0.03, "max_relative_rms_residual_to_energy_span": 0.05}
    values.update(changes)
    return values


def atoms_fixture(*, distorted=True):
    cell = np.array([[4.08, 0.07, 0.02], [0, 4.0, 0.03], [0, 0, 4.04]]) if distorted else np.eye(3) * 4
    positions = [[0.04, 0.05, 0.02], [0.34, 0.23, 0.29]] if distorted else [[0, 0, 0], [0.25, 0.25, 0.25]]
    return Atoms("Si2", cell=cell, scaled_positions=positions, pbc=True)


class AnalyticBudget:
    def __init__(self, max_calls=10000):
        self.max_calls = max_calls
        self.calls = 0
        self.steps = 0
        self.cancelled = False

    def check(self):
        if self.cancelled:
            raise RuntimeError("synthetic_test_cancelled")

    def record_step(self, node_id):
        self.check()
        self.steps += 1

    def reserve_evaluation(self, atoms):
        self.check()
        if self.calls >= self.max_calls:
            raise RuntimeError("synthetic_test_force_budget_exhausted")
        self.calls += 1


class AnalyticContext:
    """Test namespace only; real ASE calculators, no production backend registration."""
    physics_fidelity = "synthetic"

    def __init__(self, *, max_calls=10000, calculator_options=None):
        self.budget = AnalyticBudget(max_calls)
        self.calculator_options = calculator_options or {}
        self.entries = {}
        self.cache_hits = []
        self.computations = []

    def calculator(self, node_id):
        return AnalyticEOSCalculator(before_evaluate=self.budget.reserve_evaluation, **self.calculator_options)

    def evaluate(self, atoms, node_id, mapping=None):
        self.budget.check()
        return {"energy_eV": float(atoms.get_potential_energy()), "forces_eV_A": atoms.get_forces().tolist(),
                "stress_eV_A3": atoms.get_stress(voigt=False).tolist(),
                "geometry_id": snapshot_from_atoms(atoms)["geometry_id"], "mapping": mapping,
                "backend_family": "synthetic", "physics_fidelity": "synthetic", "evidence_origin": "analytic_test"}

    def cached(self, *, stage, geometry, parameters, dependencies, compute):
        key = digest({"stage": stage, "geometry": geometry["geometry_id"], "parameters": parameters,
                      "dependencies": dependencies, "calculator_options": self.calculator_options})
        if key in self.entries:
            self.cache_hits.append(stage)
            return deepcopy(self.entries[key])
        self.computations.append(stage)
        result = compute()
        if result.get("quality_status") == "accepted":
            self.entries[key] = deepcopy(result)
        return result


def _observation(atoms):
    return {"energy_eV": float(atoms.get_potential_energy()), "forces_eV_A": atoms.get_forces().tolist(),
            "stress_eV_A3": atoms.get_stress(voigt=False).tolist()}


def test_S2_C13_analytical_forces_equal_negative_energy_derivative():
    atoms = atoms_fixture()
    atoms.calc = AnalyticEOSCalculator()
    forces = atoms.get_forces()
    step = 1e-6
    assert np.linalg.norm(forces) > 0.01
    for site in range(2):
        for axis in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[site, axis] += step
            minus.positions[site, axis] -= step
            plus.calc, minus.calc = AnalyticEOSCalculator(), AnalyticEOSCalculator()
            numerical_force = -(plus.get_potential_energy() - minus.get_potential_energy()) / (2 * step)
            assert forces[site, axis] == pytest.approx(numerical_force, abs=1e-8)
    translated = atoms.copy()
    translated.positions += [0.37, -0.29, 0.81]
    translated.calc = AnalyticEOSCalculator()
    assert translated.get_potential_energy() == pytest.approx(atoms.get_potential_energy(), abs=1e-12)
    assert np.allclose(forces.sum(axis=0), 0, atol=1e-14)


def test_S2_C14_analytical_stress_matches_strain_derivative_with_shear_factor():
    atoms = atoms_fixture()
    atoms.calc = AnalyticEOSCalculator()
    stress = atoms.get_stress(voigt=False)
    step = 1e-6
    for first, second in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]:
        strain = np.zeros((3, 3))
        strain[first, second] = step
        strain[second, first] = step
        plus, minus = atoms.copy(), atoms.copy()
        plus.set_cell(atoms.cell @ (np.eye(3) + strain), scale_atoms=True)
        minus.set_cell(atoms.cell @ (np.eye(3) - strain), scale_atoms=True)
        plus.calc, minus.calc = AnalyticEOSCalculator(), AnalyticEOSCalculator()
        factor = 1 if first == second else 2
        numerical_stress = (plus.get_potential_energy() - minus.get_potential_energy()) / (2 * step * atoms.get_volume() * factor)
        assert stress[first, second] == pytest.approx(numerical_stress, abs=2e-9)
    expanded = atoms_fixture(distorted=False)
    expanded.set_cell(expanded.cell * 1.01, scale_atoms=True)
    expanded.calc = AnalyticEOSCalculator(internal_volume_coupling=0)
    assert np.trace(expanded.get_stress(voigt=False)) > 0
    assert -np.trace(expanded.get_stress(voigt=False)) / 3 < 0


def test_S2_C15_real_fire_frechet_relaxation_retains_input_and_checks_posterior():
    reference = snapshot_from_atoms(atoms_fixture(), stage="input")
    untouched = deepcopy(reference)
    context = AnalyticContext()
    output = relax(reference, context, relax_parameters())
    assert output["quality_status"] == "accepted", output["reasons"]
    assert output["input_geometry"] == untouched == reference
    assert output["reference_geometry_id"] != reference["geometry_id"]
    assert output["reference_geometry"]["parent"] == reference["geometry_id"]
    assert output["diagnostics"]["steps"] > 0
    assert context.budget.steps == output["diagnostics"]["steps"]
    assert output["diagnostics"]["fmax_eV_per_A"] < 1e-5
    assert output["diagnostics"]["max_active_stress_GPa"] < 0.002
    assert output["diagnostics"]["volume_A3"] == pytest.approx(64, abs=0.002)
    assert output["physics_fidelity"] == "synthetic"
    assert output["benchmark_eligible"] is False
    json.dumps(output, allow_nan=False)


def test_S2_C16_optimizer_convergence_cannot_override_posterior_stress():
    reference = snapshot_from_atoms(atoms_fixture(distorted=False), stage="input")
    context = AnalyticContext(calculator_options={"stress_bias": [[1e-4, 0, 0], [0, 0, 0], [0, 0, 0]]})
    output = relax(reference, context, relax_parameters(fmax_eV_per_A=1, stress_max_GPa=1e-5))
    assert output["diagnostics"]["optimizer_converged"] is True
    assert output["quality_status"] == "rejected"
    assert "posterior_active_stress_exceeds_threshold" in output["reasons"]
    assert output["reference_geometry"] is None


@pytest.mark.parametrize("settings,termination", [({"max_steps": 0}, "max_steps"), ({"max_wall_seconds": 1e-12}, "timeout")])
def test_S2_C17_maxsteps_timeout_preserve_partial_without_accepted_reference(settings, termination):
    reference = snapshot_from_atoms(atoms_fixture(), stage="input")
    context = AnalyticContext()
    output = relax(reference, context, relax_parameters(**settings))
    assert output["diagnostics"]["termination_reason"] == termination
    assert output["quality_status"] != "accepted"
    assert output["reference_geometry"] is None
    assert output["last_geometry"] is not None
    assert output["execution_status"] == "partial"
    assert not context.entries


def test_S2_C17_budget_exhaustion_does_not_soft_succeed_or_change_method():
    context = AnalyticContext(max_calls=1)
    output = relax(snapshot_from_atoms(atoms_fixture()), context, relax_parameters())
    assert context.budget.calls == 1
    assert output["reference_geometry"] is None
    assert output["quality_status"] == "unverified"
    assert output["physics_fidelity"] == "synthetic"


def test_S2_C18_constrained_mask_and_external_pressure_check_only_declared_freedoms():
    atoms = atoms_fixture(distorted=False)
    pressure = 2.0
    tensor_gpa = np.diag([-pressure, 100.0, -70.0])
    observation = {"energy_eV": -8, "forces_eV_A": np.zeros((2, 3)).tolist(),
                   "stress_eV_A3": (tensor_gpa / EV_PER_ANGSTROM3_TO_GPA).tolist()}
    parameters = relax_parameters(cell_mask=[True, False, False, False, False, False], target_pressure_GPa=pressure)
    diagnostics, reasons = check_stationarity(atoms, observation, parameters)
    assert not reasons
    assert diagnostics["max_active_stress_GPa"] < 1e-12
    _, full_reasons = check_stationarity(atoms, observation, relax_parameters(target_pressure_GPa=pressure))
    assert "posterior_active_stress_exceeds_threshold" in full_reasons


def test_S2_C18_real_fixed_cell_relaxation_keeps_volume_at_finite_stress():
    atoms = atoms_fixture()
    reference = snapshot_from_atoms(atoms)
    result = relax(reference, AnalyticContext(), relax_parameters(), fixed_cell=True)
    assert result["quality_status"] == "accepted", result["reasons"]
    assert np.array_equal(result["reference_geometry"]["cell"], reference["cell"])
    assert abs(result["diagnostics"]["pressure_GPa"]) > 0.01
    assert result["diagnostics"]["stress_check_mask"] == [False] * 6


def test_S2_C18_real_masked_frechet_relaxation_with_nonzero_external_pressure():
    reference = snapshot_from_atoms(atoms_fixture(distorted=False))
    parameters = relax_parameters(cell_mask=[True, False, False, False, False, False], target_pressure_GPa=2)
    result = relax(reference, AnalyticContext(), parameters)
    assert result["quality_status"] == "accepted", result["reasons"]
    final = np.array(result["reference_geometry"]["cell"])
    assert final[0, 0] < 4
    assert final[1, 1] == pytest.approx(4, abs=1e-12)
    assert final[2, 2] == pytest.approx(4, abs=1e-12)
    assert abs(result["diagnostics"]["residual_stress_GPa_voigt"][0]) < 0.002
    assert abs(result["diagnostics"]["residual_stress_GPa_voigt"][1]) > 0.1
    assert result["diagnostics"]["effective_parameters"]["target_pressure_GPa"] == 2


def manufactured_points(*, v0=64, b0=0.5, bprime=4.2, noise=None):
    volumes = np.array(eos_parameters()["volume_factors"]) * 64
    energies = manufactured_bm3(volumes, v0=v0, b0=b0, bprime=bprime)
    if noise is not None:
        energies += np.asarray(noise)
    return [{"volume_A3": float(v), "energy_eV": float(e), "quality_status": "accepted"}
            for v, e in zip(volumes, energies)]


def test_S2_C31_actual_bm3_fit_recovers_independent_known_curve_and_units():
    result = fit_birch_murnaghan(manufactured_points(), 64, eos_parameters())
    assert result["quality_status"] == "accepted", result["reasons"]
    assert result["fit"]["V0_A3"] == pytest.approx(64, abs=1e-5)
    assert result["fit"]["B0_eV_A3"] == pytest.approx(0.5, abs=1e-6)
    assert result["fit"]["Bprime"] == pytest.approx(4.2, abs=1e-4)
    assert result["value"] == pytest.approx(0.5 * EV_PER_ANGSTROM3_TO_GPA, rel=1e-6)
    assert result["fit"]["rms_residual_eV"] < 1e-10


def test_S2_C19_C31_C32_C33_shared_reference_seven_fixed_volume_independent_internal_relaxations():
    context = AnalyticContext()
    original = snapshot_from_atoms(atoms_fixture(), stage="input")
    relaxed = relax(original, context, relax_parameters())
    reference = relaxed["reference_geometry"]
    assert reference is not None
    before = deepcopy(reference)
    result = eos(reference, context, eos_parameters(), relax_parameters(fmax_eV_per_A=1e-7))
    assert result["quality_status"] == "accepted", result["reasons"]
    assert reference == before
    assert result["reference_geometry_id"] == relaxed["reference_geometry_id"]
    volume = np.linalg.det(reference["cell"])
    for point, factor in zip(result["points"], eos_parameters()["volume_factors"]):
        assert point["volume_A3"] == pytest.approx(volume * factor, rel=1e-12)
        assert point["initial_geometry"]["parent"] == reference["geometry_id"]
        assert point["source_reference_geometry_id"] == reference["geometry_id"]
        assert np.allclose(point["initial_geometry"]["positions"], reference["positions"], atol=1e-14)
        assert point["relaxation"]["diagnostics"]["cell_constraint"] == "fixed_cell"
        assert point["quality_status"] == "accepted"
        assert point["final_geometry"]["cell"] == point["initial_geometry"]["cell"]
    assert any(point["relaxation"]["diagnostics"]["steps"] > 0 for point in result["points"])
    assert result["fit"]["Bprime"] == pytest.approx(4.2, abs=0.003)
    assert result["value"] == pytest.approx(0.5 * EV_PER_ANGSTROM3_TO_GPA, rel=2e-5)
    calls = context.budget.calls
    again = eos(reference, context, eos_parameters(), relax_parameters(fmax_eV_per_A=1e-7))
    assert context.budget.calls == calls
    assert again["value"] == result["value"]
    assert context.cache_hits.count("eos-point") == 7
    assert "eos-fit" in context.cache_hits


@pytest.mark.parametrize("defect", ["missing", "nonfinite", "duplicate", "unconverged"])
def test_S2_C34_bad_points_prevent_fit_without_dropping_points_or_zero(defect):
    points = manufactured_points()
    if defect == "missing":
        points.pop()
    elif defect == "nonfinite":
        points[1]["energy_eV"] = float("nan")
    elif defect == "duplicate":
        points[1]["volume_A3"] = points[0]["volume_A3"]
    else:
        points[1]["quality_status"] = "unverified"
    result = fit_birch_murnaghan(points, 64, eos_parameters())
    assert result["value"] is None
    assert result["quality_status"] != "accepted"
    assert result["reasons"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("points,parameters,reason", [
    (manufactured_points(v0=80), eos_parameters(max_relative_volume_shift=1), "fitted_minimum_outside_sampled_interior"),
    (manufactured_points(b0=-0.5), eos_parameters(), "nonpositive_curvature"),
    (manufactured_points(noise=[0, 0.01, -0.01, 0.005, -0.005, 0.01, 0]), eos_parameters(), "absolute_fit_residual_exceeds_threshold"),
    (manufactured_points(v0=65), eos_parameters(max_relative_volume_shift=0.001), "fitted_volume_shift_exceeds_reference_tolerance"),
])
def test_S2_C35_rejected_fit_retains_diagnostics(points, parameters, reason):
    result = fit_birch_murnaghan(points, 64, parameters)
    assert result["value"] is None
    assert result["quality_status"] == "rejected"
    assert reason in result["reasons"], result
    assert "B0_eV_A3" in result["fit"]
    json.dumps(result, allow_nan=False)


def test_S2_C34_unqualified_reference_or_incomplete_point_cannot_produce_B():
    reference = snapshot_from_atoms(atoms_fixture(distorted=False))
    context = AnalyticContext()
    result = eos(reference, context, eos_parameters(), relax_parameters())
    assert result["value"] is None
    assert context.budget.calls == 0
    reference["stationary_reference_accepted"] = True
    result = eos(reference, context, eos_parameters(), relax_parameters(max_steps=0))
    assert result["value"] is None
    assert len(result["points"]) == 7
    assert any(point["calculation_status"] == "skipped" for point in result["points"])


def test_S2_C36_recipe_identity_is_eos_not_voigt_or_hill():
    fitted = fit_birch_murnaghan(manufactured_points(), 64, eos_parameters())
    assert fitted["property_id"] == "bulk_modulus_eos_GPa"
    with pytest.raises(ValueError):
        fit_birch_murnaghan(manufactured_points(), 64, eos_parameters(fit="voigt"))


def test_S2_C35_relative_residual_and_flat_curve_are_independent_quality_gates():
    points = manufactured_points(noise=[0, 0.01, -0.01, 0.005, -0.005, 0.01, 0])
    result = fit_birch_murnaghan(points, 64, eos_parameters(fit_rms_max_eV=1))
    assert "absolute_fit_residual_exceeds_threshold" not in result["reasons"]
    assert "relative_fit_residual_exceeds_threshold" in result["reasons"]
    for point in points:
        point["energy_eV"] = -8.0
    assert fit_birch_murnaghan(points, 64, eos_parameters())["value"] is None


def public_context(tmp_path, calculator=None, *, max_retries=0):
    from crystargetbench.budget import Budget, DEFAULT_TEST_LIMITS
    from crystargetbench.calculation import ExecutionContext

    backend = {"id": "analytic_test", "family": "synthetic", "version": "manufactured_bm3.v1", "assets": {},
               "configuration_digest": digest({"model": "analytic_bm3", "internal_volume_coupling": 0.02}),
               "precision": "analytic_float64"}
    limits = dict(DEFAULT_TEST_LIMITS, max_total_wall_seconds=180, max_sample_wall_seconds=120, max_retries=max_retries,
                  max_relax_steps_per_structure=1200)
    budget = Budget(tmp_path / "budget.json", limits, identity=digest(backend),
                    scratch_roots=[tmp_path / "cache", tmp_path / "work"])
    return ExecutionContext(backend=backend, budget=budget, workdir=tmp_path / "work", cache_root=tmp_path / "cache",
                            analytic_calculator=calculator or AnalyticEOSCalculator(internal_volume_coupling=0.02), analytic_test=True)


def test_S2_C19_C37_public_context_budget_real_eos_and_zero_call_reassessment(tmp_path):
    from crystargetbench.assessment import assess, make_measurement

    atoms = atoms_fixture(distorted=False)
    atoms.positions[1, 0] += 0.002
    original = snapshot_from_atoms(atoms, stage="input")
    context = public_context(tmp_path)
    context.budget.start_sample(original["geometry_id"])
    parameters = relax_parameters(fmax_eV_per_A=0.001, stress_max_GPa=0.02)
    relaxed = relax(original, context, parameters)
    assert relaxed["quality_status"] == "accepted", relaxed["reasons"]
    reference = relaxed["reference_geometry"]
    measured = eos(reference, context, eos_parameters(fit_rms_max_eV=1e-5), parameters)
    assert measured["quality_status"] == "accepted", measured["reasons"]
    before = deepcopy(context.stats)
    repeated = eos(reference, context, eos_parameters(fit_rms_max_eV=1e-5), parameters)
    assert repeated["value"] == measured["value"]
    assert context.stats["backend_structure_evaluations"] == before["backend_structure_evaluations"]
    assert context.stats["stage_cache_hits"]["eos-point"] == 7
    assert context.stats["stage_cache_hits"]["eos-fit"] == 1
    observation = make_measurement(property_id="bulk_modulus_eos_GPa", definition_id="synthetic.bm3.test.v1",
        value=measured["value"], unit="GPa", reference_geometry_id=reference["geometry_id"],
        calculation_status="completed", quality_status="accepted", identity_status="verified",
        backend_id="analytic_test", backend_family="synthetic", backend_version="1", physics_fidelity="synthetic",
        protocol_digest="synthetic_public_context_test", benchmark_eligible=False)
    task = {"task_id": "synthetic_bm3_reassessment", "mode": "screen",
            "measurements": {"B": {"property_id": "bulk_modulus_eos_GPa", "unit": "GPa"}},
            "constraints": [{"id": "minimum", "measurement": "B", "operator": "gt", "value": 40}]}
    assert assess(task, {"B": observation}, reference["geometry_id"])["decision"] == "pass"
    task["constraints"][0]["value"] = 100
    assert assess(task, {"B": observation}, reference["geometry_id"])["decision"] == "fail"
    assert context.stats["backend_structure_evaluations"] == before["backend_structure_evaluations"]
    ledger = context.budget.snapshot()
    assert ledger["completed_evaluations"] == context.stats["backend_structure_evaluations"]
    assert context.stats["backend_forwards"] > 0
    assert context.stats["physics_calls"] == 0
    assert all(lease["status"] == "completed" for lease in ledger["leases"].values())


def test_S2_C40_public_context_partial_resume_reuses_completed_efs(tmp_path):
    class InterruptedAnalytic(AnalyticEOSCalculator):
        def __init__(self):
            super().__init__(internal_volume_coupling=0.02)
            self.invocations = 0

        def calculate(self, *args, **kwargs):
            self.invocations += 1
            if self.invocations == 4:
                raise RuntimeError("synthetic execution interruption; no method change")
            return super().calculate(*args, **kwargs)

    atoms = atoms_fixture(distorted=False)
    atoms.positions[1, 0] += 0.02
    original = snapshot_from_atoms(atoms, stage="input")
    context = public_context(tmp_path, InterruptedAnalytic(), max_retries=1)
    context.budget.start_sample(original["geometry_id"])
    parameters = relax_parameters(fmax_eV_per_A=0.001, stress_max_GPa=0.02)
    partial = relax(original, context, parameters, fixed_cell=True)
    assert partial["quality_status"] == "unverified"
    assert partial["reference_geometry"] is None
    assert context.stats["backend_structure_evaluations"] == 4
    resumed = public_context(tmp_path, max_retries=1)
    resumed.budget.start_sample(original["geometry_id"])
    completed = relax(original, resumed, parameters, fixed_cell=True)
    assert completed["quality_status"] == "accepted", completed["reasons"]
    assert resumed.stats["stage_cache_hits"]["energy-forces-stress"] >= 3
    assert resumed.stats["stage_computations"]["eos-point-relax"] == 1
    ledger = resumed.budget.snapshot()
    assert ledger["failed_evaluations"] == 1
    failed_operation = next(lease["operation_id"] for lease in ledger["leases"].values() if lease["status"] == "failed")
    retried = [lease for lease in ledger["leases"].values() if lease["operation_id"] == failed_operation]
    assert len(retried) == 2
    assert {lease["status"] for lease in retried} == {"failed", "completed"}
    assert not any(lease["status"] == "active" for lease in ledger["leases"].values())
