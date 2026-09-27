"""Controlled ASE FIRE/FrechetCellFilter recipe with independent stationarity checks.

The execution context supplies the calculator and enforces per-request budgets.
This module never constructs a production model or selects an alternative method.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from importlib.metadata import version
import math
import time

from ..identity import digest
from ..physics import EV_PER_ANGSTROM3_TO_GPA


RECIPE = "ase.fire_frechet.relaxation.v1"
IMPLEMENTATION_VERSION = "ctb.ase_relaxation.v2"
SIGNATURE_VERSION = "ctb.relaxation_signature.v1"
_FIRE_FIXED = dict(Nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99,
                   a=0.1, downhill_check=False)


def effective_relaxation_signature(parameters, fixed_cell=False):
    required = {"require_converged", "fmax_eV_per_A", "stress_max_GPa", "max_steps",
                "target_pressure_GPa", "optimizer", "filter", "cell_mask", "dt",
                "dtmax", "maxstep_A", "max_wall_seconds"}
    if not required.issubset(parameters):
        raise ValueError(f"Incomplete relaxation parameters: {sorted(required - parameters.keys())}")
    if parameters.keys() - required:
        raise ValueError(f"Unknown relaxation parameters: {sorted(parameters.keys() - required)}")
    if parameters["optimizer"] != "FIRE" or parameters["filter"] != "FrechetCellFilter":
        raise ValueError("Only the explicit FIRE/FrechetCellFilter recipe is implemented")
    if parameters["require_converged"] is not True:
        raise ValueError("S2 relaxation requires convergence; soft success is not permitted")
    for name in ("fmax_eV_per_A", "stress_max_GPa", "dt", "dtmax", "maxstep_A", "max_wall_seconds"):
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    pressure = parameters["target_pressure_GPa"]
    if isinstance(pressure, bool) or not isinstance(pressure, (int, float)) or not math.isfinite(pressure):
        raise ValueError("target_pressure_GPa must be finite")
    if type(parameters["max_steps"]) is not int or parameters["max_steps"] < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    mask = parameters["cell_mask"]
    if not isinstance(mask, list) or len(mask) != 6 or any(type(value) is not bool for value in mask):
        raise ValueError("cell_mask must contain six booleans in xx,yy,zz,yz,xz,xy order")
    if parameters["dtmax"] < parameters["dt"]:
        raise ValueError("dtmax must be at least dt")
    return {**deepcopy(parameters), "signature_schema":SIGNATURE_VERSION,
            "implementation_version":IMPLEMENTATION_VERSION, "recipe": RECIPE, "fixed_cell": bool(fixed_cell),
            "effective_cell_mask": [False] * 6 if fixed_cell else list(mask),
            "fixed_FIRE_parameters": dict(_FIRE_FIXED), "exp_cell_factor": "number_of_atoms",
            "stress_criterion": "max_abs_active_voigt_component_of_sigma_plus_pI",
            "ase_version": version("ase"), "numpy_version": version("numpy"),
            "scipy_version": version("scipy")}


def check_stationarity(atoms, observation, parameters, *, fixed_cell=False):
    """Check actual Cartesian forces and declared stress freedoms, independently of FIRE."""
    import numpy as np
    from ase.stress import full_3x3_to_voigt_6_stress

    diagnostics = {"stationary_reference_accepted": False, "fmax_eV_per_A": None,
                   "stress_GPa_voigt": None, "pressure_GPa": None,
                   "residual_pressure_GPa": None, "residual_stress_GPa_voigt": None,
                   "deviatoric_stress_GPa_voigt": None, "max_active_stress_GPa": None}
    reasons = []
    try:
        forces = np.asarray(observation["forces_eV_A"], dtype=float)
        stress = np.asarray(observation["stress_eV_A3"], dtype=float)
        energy = observation["energy_eV"]
        if isinstance(energy, bool) or not math.isfinite(energy):
            raise ValueError("Nonfinite energy")
        if forces.shape != (len(atoms), 3) or stress.shape != (3, 3):
            raise ValueError("Incorrect force or stress shape")
        if not np.isfinite(forces).all() or not np.isfinite(stress).all():
            raise ValueError("Nonfinite forces or stress")
        if not np.allclose(stress, stress.T, atol=1e-12, rtol=0):
            raise ValueError("Stress tensor is not symmetric")
        cell = np.asarray(atoms.cell)
        volume = float(np.linalg.det(cell))
        condition = float(np.linalg.cond(cell))
        if not np.isfinite(cell).all() or not np.isfinite(atoms.positions).all() or volume <= 0 or not math.isfinite(condition) or condition > 1e12:
            raise ValueError("Invalid or ill-conditioned final cell")
        if not atoms.pbc.all():
            raise ValueError("Relaxation requires fully periodic geometry")
        fmax = float(np.linalg.norm(forces, axis=1).max())
        stress_gpa = stress * EV_PER_ANGSTROM3_TO_GPA
        residual = stress_gpa + parameters["target_pressure_GPa"] * np.eye(3)
        voigt = full_3x3_to_voigt_6_stress(stress_gpa)
        residual_voigt = full_3x3_to_voigt_6_stress(residual)
        mask = np.zeros(6, dtype=bool) if fixed_cell else np.asarray(parameters["cell_mask"], dtype=bool)
        active = float(np.abs(residual_voigt[mask]).max()) if mask.any() else 0.0
        mean_stress = float(sum(stress_gpa[index, index] / 3 for index in range(3)))
        pressure = -mean_stress
        deviatoric = stress_gpa - np.eye(3) * mean_stress
        if not np.isfinite(stress_gpa).all() or not np.isfinite(residual).all() or not np.isfinite(deviatoric).all() or not math.isfinite(fmax) or not math.isfinite(pressure):
            raise ValueError("Nonfinite derived force/stress diagnostics")
        diagnostics.update(fmax_eV_per_A=fmax, stress_eV_A3_voigt=full_3x3_to_voigt_6_stress(stress).tolist(),
                           stress_GPa_voigt=voigt.tolist(), pressure_GPa=pressure,
                           residual_pressure_GPa=pressure - parameters["target_pressure_GPa"],
                           residual_stress_GPa_voigt=residual_voigt.tolist(),
                           deviatoric_stress_GPa_voigt=full_3x3_to_voigt_6_stress(deviatoric).tolist(),
                           max_active_stress_GPa=active, stress_check_mask=mask.tolist(),
                           volume_A3=volume, cell_condition_number=condition,
                           stress_convention="ASE tensile-positive; pressure=-trace(stress)/3")
        if fmax > parameters["fmax_eV_per_A"]:
            reasons.append("posterior_atomic_force_exceeds_threshold")
        if active > parameters["stress_max_GPa"]:
            reasons.append("posterior_active_stress_exceeds_threshold")
    except (KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        reasons.append(f"invalid_posterior_observation:{exc}")
    diagnostics["stationary_reference_accepted"] = not reasons
    return diagnostics, reasons


def relax(reference: dict, context, parameters: dict, *, fixed_cell=False, node_id="relax") -> dict:
    """Optimize a copied snapshot and bind a reference only after strict posterior acceptance."""
    effective = effective_relaxation_signature(parameters, fixed_cell)

    def compute():
        from ase.filters import FrechetCellFilter
        from ase.optimize import FIRE
        from ..geometry import atoms_from_snapshot, snapshot_from_atoms

        started = time.monotonic()
        atoms = atoms_from_snapshot(reference)
        original = deepcopy(reference)
        last_geometry = None
        observation = None
        optimizer = None
        completed = False
        converged = False
        termination = "not_started"
        reasons = []
        history = []
        diagnostics = {}

        def check_limits():
            context.budget.check()
            if time.monotonic() - started >= parameters["max_wall_seconds"]:
                raise TimeoutError("Per-relaxation wall limit reached")

        try:
            check_limits()
            atoms.calc = context.calculator(node_id)
            target = atoms if fixed_cell else FrechetCellFilter(
                atoms, mask=parameters["cell_mask"], exp_cell_factor=float(len(atoms)),
                scalar_pressure=parameters["target_pressure_GPa"] / EV_PER_ANGSTROM3_TO_GPA,
                hydrostatic_strain=False, constant_volume=False)
            optimizer = FIRE(target, logfile=None, trajectory=None, restart=None,
                             dt=parameters["dt"], dtmax=parameters["dtmax"],
                             maxstep=parameters["maxstep_A"], **_FIRE_FIXED)
            iterator = optimizer.irun(fmax=parameters["fmax_eV_per_A"], steps=parameters["max_steps"])
            while True:
                check_limits()
                converged = bool(next(iterator))
                last_geometry = snapshot_from_atoms(atoms, parent=reference["geometry_id"], stage="relaxation_partial",
                    transformation={"recipe": RECIPE, "fixed_cell": bool(fixed_cell), "site_mapping": list(range(len(atoms)))})
                history.append({"step": optimizer.nsteps, "geometry_id": last_geometry["geometry_id"]})
                if converged or optimizer.nsteps >= parameters["max_steps"]:
                    completed = True
                    termination = "converged" if converged else "max_steps"
                    break
                check_limits()
                context.budget.record_step(node_id)
            check_limits()
            observation = context.evaluate(atoms, node_id + ":posterior",
                mapping={"reference_geometry_id": reference["geometry_id"], "site_mapping": list(range(len(atoms)))})
            diagnostics, quality_reasons = check_stationarity(atoms, observation, parameters, fixed_cell=fixed_cell)
            reasons.extend(quality_reasons)
            if not converged:
                reasons.append("optimizer_not_converged")
        except Exception as exc:
            # Budgets, cancellation and provider errors are computational unknowns.
            # No exception authorizes a different model, precision or method.
            code = getattr(exc, "code", None)
            termination = str(code or ("timeout" if isinstance(exc, TimeoutError) else type(exc).__name__))
            reasons.append(f"execution_interrupted:{termination}:{exc}")
            completed = False
            converged = False
        try:
            last_geometry = snapshot_from_atoms(atoms, parent=reference["geometry_id"], stage="relaxation_partial",
                transformation={"recipe": RECIPE, "fixed_cell": bool(fixed_cell), "site_mapping": list(range(len(atoms)))})
        except (ValueError, TypeError):
            last_geometry = None
        accepted = completed and converged and not reasons and last_geometry is not None
        diagnostics.update(optimizer_completed=completed, optimizer_converged=converged,
                           steps=optimizer.nsteps if optimizer is not None else 0,
                           termination_reason=termination, elapsed_seconds=time.monotonic() - started,
                           stationary_reference_accepted=accepted,
                           optimizer_generalized_fmax_threshold=parameters["fmax_eV_per_A"],
                           physical_force_criterion="max_atom_cartesian_force_norm",
                           cell_constraint="fixed_cell" if fixed_cell else "declared_voigt_mask",
                           effective_parameters=effective)
        final_reference = None
        if accepted:
            final_reference = deepcopy(last_geometry)
            final_reference["stage"] = "eos_point_relaxed" if fixed_cell else "relaxed"
            final_reference["stationary_reference_accepted"] = True
            final_reference["stationary_condition"] = "fixed_cell_internal_coordinates" if fixed_cell else "declared_force_and_stress_freedoms"
            final_reference["relaxation_diagnostics_digest"] = digest(diagnostics)
        return {"schema_version": "ctb.relaxation.v1", "recipe": RECIPE,
                "input_geometry": original, "input_geometry_id": reference["geometry_id"],
                "last_geometry": last_geometry, "reference_geometry": final_reference,
                "reference_geometry_id": final_reference["geometry_id"] if final_reference else None,
                "calculation_status": "completed" if completed else "failed",
                "execution_status": "completed" if accepted else "partial",
                "quality_status": "accepted" if accepted else ("rejected" if completed else "unverified"),
                "reasons": reasons, "diagnostics": diagnostics, "observation": observation,
                "history": history, "benchmark_eligible": False,
                "physics_fidelity": getattr(context, "physics_fidelity", "unknown")}

    def compute_with_deadline():
        scope = context.budget.deadline_scope(parameters["max_wall_seconds"]) if hasattr(context.budget, "deadline_scope") else nullcontext()
        with scope:
            return compute()

    return context.cached(stage="eos-point-relax" if fixed_cell else "reference-relax", geometry=reference,
                          parameters=effective, dependencies=[], compute=compute_with_deadline)
