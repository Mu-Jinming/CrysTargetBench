"""Generate actual S2-A numerical artifacts using explicitly test-only calculators.

Run from the CTB workspace with the numerical extra/environment. No checkpoint,
trained model, production worker, GPU, DFT engine or cluster is used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from crystargetbench.budget import Budget, DEFAULT_TEST_LIMITS
from crystargetbench.calculation import ExecutionContext
from crystargetbench.contracts import load_json
from crystargetbench.geometry import snapshot_from_atoms
from crystargetbench.identity import digest
from crystargetbench.reporting import write_json
from crystargetbench.scientific import execute_submission
from crystargetbench.structures import load_submission, submission_digest
from tests.s2.analytic_eos import AnalyticEOSCalculator, AnalyticLatticeBM3
from tests.s2.analytic_lattice import lattice_atoms


def numerical_parameters():
    from crystargetbench.recipes.phonon import DEFAULTS

    return {
        "relaxation": {"require_converged": True, "optimizer": "FIRE", "filter": "FrechetCellFilter",
            "cell_mask": [True] * 6, "fmax_eV_per_A": 1e-5, "stress_max_GPa": 0.002,
            "max_steps": 1200, "target_pressure_GPa": 0.0, "dt": 0.1, "dtmax": 1.0,
            "maxstep_A": 0.15, "max_wall_seconds": 30.0},
        "phonon": {**deepcopy(DEFAULTS), "primitive_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "masses": [20.0, 40.0], "is_plusminus": True, "is_diagonal": False,
            "path_points_parameter": 5, "mesh": [4, 4, 4]},
        "eos": {"volume_factors": [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06],
            "fit": "birch_murnaghan_3rd_order", "internal_relaxation": True, "fixed_cell_shape": True,
            "fit_rms_max_eV": 1e-6, "max_relative_volume_shift": 0.03,
            "max_relative_rms_residual_to_energy_span": 0.05},
    }


def numerical_configuration(parameters=None, *, bulk_threshold=10):
    parameters = deepcopy(parameters or numerical_parameters())
    task = {"schema_version": "ctb.task.v2", "task_id": "synthetic_lattice_joint", "mode": "screen",
        "reference_geometry": "relaxed", "measurements": {
            "phonon": {"property_id": "phonon_mesh_min_frequency_THz", "unit": "THz"},
            "B": {"property_id": "bulk_modulus_eos_GPa", "unit": "GPa"}},
        "constraints": [{"id": "phonon_threshold", "measurement": "phonon", "operator": "ge", "value": -0.001},
                        {"id": "bulk_threshold", "measurement": "B", "operator": "ge", "value": bulk_threshold}]}
    protocol = {"protocol_id": "analytic_lattice_bm3.shared_reference.s2_test.v2", "status": "candidate",
                "parameters": parameters, "physics_fidelity": "synthetic", "evidence_kind": "analytic_test",
                "benchmark_eligible": False}
    resolved = {"protocol_digest": digest(protocol), "protocol_lock": {"protocol": protocol},
                "routes": {alias: {"provider": "analytic_test", "family": "synthetic",
                                    "physics_fidelity": "synthetic", "definition_id": "synthetic." + spec["property_id"] + ".v1",
                                    "requires_stationary_reference": True}
                           for alias, spec in task["measurements"].items()}}
    resolved["comparability_digest"] = digest({"task": task, "protocol": protocol, "metric_version": "ctb.metrics.v1"})
    return task, resolved


def numerical_context(workdir, *, calculator=None, identity="analytic_lattice_bm3_shared_reference.v2", cache_root=None):
    workdir = Path(workdir)
    backend = {"id": "analytic_test", "family": "synthetic", "version": identity, "assets": {},
               "configuration_digest": digest({"manufactured_model": identity}),
               "precision": "analytic_float64", "device": "cpu"}
    limits = dict(DEFAULT_TEST_LIMITS, max_total_wall_seconds=240, max_sample_wall_seconds=120,
                  max_relax_steps_per_structure=1200, max_backend_structure_evaluations=5000,
                  max_scratch_bytes=268435456)
    cache_root = Path(cache_root) if cache_root else workdir / "cache"
    budget = Budget(workdir / "budget.json", limits, identity=digest(backend), scratch_roots=[workdir, cache_root])
    return ExecutionContext(backend=backend, budget=budget, workdir=workdir / "execution", cache_root=cache_root,
                            analytic_calculator=calculator or AnalyticLatticeBM3(), analytic_test=True)


def write_fixture_submission(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atoms = lattice_atoms()
    structure = {"cell": atoms.cell.tolist(), "scaled_positions": atoms.get_scaled_positions().tolist(),
                 "species": atoms.get_chemical_symbols(), "pbc": [True, True, True]}
    write_json(directory / "01-lattice.json", structure)
    write_json(directory / "02-duplicate.json", structure)
    (directory / "03-broken.json").write_text('{"cell":', encoding="utf-8")
    return load_submission(directory)


def generate(output, workdir):
    output, workdir = Path(output), Path(workdir)
    output.mkdir(parents=True, exist_ok=True)
    items = write_fixture_submission(output / "structures")
    task, resolved = numerical_configuration()
    context = numerical_context(workdir / "shared")
    results, metrics, costs = execute_submission(task, items, resolved, context)
    write_json(output / "task.json", task)
    write_json(output / "protocol.lock.json", resolved)
    write_json(output / "submission.json", {"items": items, "submission_digest": submission_digest(items)})
    write_json(output / "results.json", {"results": results})
    write_json(output / "metrics.json", metrics)
    write_json(output / "costs-first.json", {**costs, "budget": context.budget.snapshot()})
    first_bundle = load_json(context.workdir / (digest(items[0]["item_id"]) + ".recipes.json"))
    for name, artifact in first_bundle["artifacts"].items():
        write_json(output / (name + ".json"), artifact)

    stricter_task, stricter_resolved = numerical_configuration(bulk_threshold=100)
    repeated_context = numerical_context(workdir / "shared")
    repeated, repeated_metrics, repeated_costs = execute_submission(stricter_task, items, stricter_resolved, repeated_context)
    write_json(output / "reassessment.json", {"results": repeated, "metrics": repeated_metrics, "costs": repeated_costs,
        "only_changed_task_threshold": True, "new_backend_evaluations": repeated_costs["backend_structure_evaluations"]})

    changed = numerical_parameters()
    changed["phonon"]["mesh"] = [6, 6, 6]
    qtask, qresolved = numerical_configuration(changed)
    sampling_context = numerical_context(workdir / "shared")
    qresults, qmetrics, qcosts = execute_submission(qtask, items, qresolved, sampling_context)
    write_json(output / "changed-sampling.json", {"results": qresults, "metrics": qmetrics, "costs": qcosts,
        "old_mesh": [4, 4, 4], "new_mesh": [6, 6, 6], "new_backend_evaluations": qcosts["backend_structure_evaluations"]})

    # A second manufactured model independently checks recovery of known BM3
    # parameters with real internal-coordinate relaxation at all seven volumes.
    from ase import Atoms
    from crystargetbench.recipes.relaxation import relax
    from crystargetbench.recipes.eos import eos
    import numpy as np

    atoms = Atoms("Si2", cell=np.eye(3) * 4.03, scaled_positions=[[0.03, 0.04, 0.02], [0.32, 0.22, 0.31]], pbc=True)
    snapshot = snapshot_from_atoms(atoms, stage="input")
    bm_context = numerical_context(workdir / "known-bm3", calculator=AnalyticEOSCalculator(), identity="analytic_known_bm3.v1")
    bm_context.budget.start_sample(snapshot["geometry_id"])
    settings = numerical_parameters()
    relaxed = relax(snapshot, bm_context, settings["relaxation"])
    bm_result = eos(relaxed["reference_geometry"], bm_context, settings["eos"], settings["relaxation"]) if relaxed["reference_geometry"] else None
    bm_context.close()
    write_json(output / "known-bm3.json", {"reference_parameters": {"E0_eV": -8, "V0_A3": 64, "B0_eV_A3": 0.5, "Bprime": 4.2},
        "relaxation": relaxed, "eos": bm_result, "costs": bm_context.stats,
        "physics_fidelity": "synthetic", "evidence_kind": "analytic_test", "benchmark_eligible": False})

    summary = {"schema_version": "ctb.s2_numerical_evidence.v1", "evidence_kind": "analytic_test",
        "physics_fidelity": "synthetic", "mlff_live_tested": False, "science_validated": False, "benchmark_eligible": False,
        "submitted": len(items), "first_counts": {name: metrics[name] for name in ("n_pass", "n_fail", "n_unknown")},
        "shared_reference_geometry_id": results[0]["reference_geometry"]["geometry_id"] if results[0]["reference_geometry"] else None,
        "first_backend_evaluations": costs["backend_structure_evaluations"],
        "reassessment_new_backend_evaluations": repeated_costs["backend_structure_evaluations"],
        "changed_sampling_new_backend_evaluations": qcosts["backend_structure_evaluations"],
        "known_bm3_quality": bm_result["quality_status"] if bm_result else "unverified",
        "versions": {name: version(name) for name in ("ase", "numpy", "scipy", "phonopy", "seekpath", "spglib")},
        "fixture_provenance": "Original manufactured two-sublattice vector-spring and BM3 internal-coordinate models; no material labels"}
    write_json(output / "SUMMARY.json", summary)
    return summary


def resume_known_bm3(output, workdir):
    """Explicit additional *synthetic test* budget; preserve the original ledger/artifact.

    No production permit or ledger is reset. Only completed scientific cache
    entries are reused, and both attempts' evaluation costs remain in the report.
    """
    from crystargetbench.recipes.relaxation import relax
    from crystargetbench.recipes.eos import eos

    output, workdir = Path(output), Path(workdir)
    previous = load_json(output / "known-bm3.json")
    preserved = output / "known-bm3-partial-before-resume.json"
    if not preserved.exists():
        write_json(preserved, previous)
    context = numerical_context(workdir / "known-bm3-explicit-numerical-resume",
        calculator=AnalyticEOSCalculator(), identity="analytic_known_bm3.v1", cache_root=workdir / "known-bm3" / "cache")
    snapshot = previous["relaxation"]["input_geometry"]
    context.budget.start_sample(snapshot["geometry_id"])
    settings = numerical_parameters()
    relaxed = relax(snapshot, context, settings["relaxation"])
    result = eos(relaxed["reference_geometry"], context, settings["eos"], settings["relaxation"]) if relaxed["reference_geometry"] else None
    context.close()
    artifact = {"reference_parameters": previous["reference_parameters"], "relaxation": relaxed, "eos": result,
        "costs": context.stats, "previous_attempt_costs": previous["costs"],
        "total_synthetic_backend_evaluations": previous["costs"]["backend_structure_evaluations"] + context.stats["backend_structure_evaluations"],
        "resume_reason": "Original 120-second synthetic sample limit reached; old partial and ledger retained",
        "resume_is_new_explicit_synthetic_test_budget": True, "numerical_parameters_changed": False,
        "physics_fidelity": "synthetic", "evidence_kind": "analytic_test", "benchmark_eligible": False}
    write_json(output / "known-bm3.json", artifact)
    summary = load_json(output / "SUMMARY.json")
    summary.update(known_bm3_quality=result["quality_status"] if result else "unverified",
                   known_bm3_first_attempt_sample_limit_reached=True,
                   known_bm3_total_synthetic_evaluations=artifact["total_synthetic_backend_evaluations"])
    write_json(output / "SUMMARY.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/s2/numerical")
    parser.add_argument("--workdir", default="runs/s2-numerical")
    parser.add_argument("--resume-known-bm3", action="store_true",
                        help="Use a separately recorded synthetic numerical budget to resume completed BM3 cache stages")
    args = parser.parse_args()
    import json
    operation = resume_known_bm3 if args.resume_known_bm3 else generate
    print(json.dumps(operation(args.output, args.workdir), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
