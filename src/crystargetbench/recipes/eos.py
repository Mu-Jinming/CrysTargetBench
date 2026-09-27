"""Seven independent fixed-shape volume points and an actual ASE BM3 fit."""
from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
import math
import warnings

from ..identity import digest
from ..physics import EV_PER_ANGSTROM3_TO_GPA
from .relaxation import relax, effective_relaxation_signature


RECIPE = "ase.eos.fixed_shape_internal_relax.bm3.v1"
POINT_SIGNATURE_VERSION = "ctb.eos_point_signature.v2"


def _validate(parameters):
    required = {"volume_factors", "fit", "internal_relaxation", "fixed_cell_shape",
                "fit_rms_max_eV", "max_relative_volume_shift", "max_relative_rms_residual_to_energy_span"}
    if not required.issubset(parameters):
        raise ValueError(f"Incomplete EOS parameters: {sorted(required - parameters.keys())}")
    if parameters.keys() - required:
        raise ValueError(f"Unknown EOS parameters: {sorted(parameters.keys() - required)}")
    factors = parameters["volume_factors"]
    if not isinstance(factors, list) or len(factors) != 7:
        raise ValueError("The S2 EOS recipe requires exactly seven volume ratios")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in factors):
        raise ValueError("EOS volume ratios must be finite and positive")
    if factors != sorted(set(factors)) or 1.0 not in factors:
        raise ValueError("EOS volume ratios must be strictly increasing, unique and include 1")
    if parameters["fit"] != "birch_murnaghan_3rd_order" or parameters["internal_relaxation"] is not True or parameters["fixed_cell_shape"] is not True:
        raise ValueError("Only BM3 with fixed shape and internal-coordinate relaxation is implemented")
    for name in ("fit_rms_max_eV", "max_relative_volume_shift", "max_relative_rms_residual_to_energy_span"):
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (name != "max_relative_volume_shift" and value == 0):
            raise ValueError(f"Invalid EOS quality parameter: {name}")


def fit_birch_murnaghan(points, reference_volume, parameters):
    """Fit qualified total-energy/total-volume points, retaining rejected fit diagnostics."""
    import numpy as np
    from ase.eos import EquationOfState, birchmurnaghan

    _validate(parameters)
    result = {"value": None, "unit": "GPa", "property_id": "bulk_modulus_eos_GPa",
              "calculation_status": "completed", "quality_status": "unverified", "reasons": [],
              "fit": {"form": "birch_murnaghan_3rd_order", "uncertainty": None,
                      "uncertainty_status": "not_returned_by_ASE_EquationOfState",
                      "ase_version": version("ase"), "scipy_version": version("scipy")}}
    if len(points) != 7 or any(point.get("quality_status") != "accepted" for point in points):
        result["reasons"].append("all_seven_prescribed_points_must_be_qualified")
        return result
    try:
        volumes = np.asarray([point["volume_A3"] for point in points], dtype=float)
        energies = np.asarray([point["energy_eV"] for point in points], dtype=float)
        if not np.isfinite(volumes).all() or not np.isfinite(energies).all() or (volumes <= 0).any():
            raise ValueError("nonfinite_or_nonpositive_point")
        if any(isinstance(point["energy_eV"], bool) or isinstance(point["volume_A3"], bool) for point in points):
            raise ValueError("boolean_is_not_energy_or_volume")
        if any(np.isclose(volumes[i], volumes[j], rtol=1e-12, atol=0) for i in range(7) for j in range(i)):
            raise ValueError("duplicate_volume")
        if isinstance(reference_volume, bool) or not math.isfinite(reference_volume) or reference_volume <= 0:
            raise ValueError("invalid_reference_volume")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reasons"].append(f"invalid_eos_points:{exc}")
        return result
    energy_span = float(np.ptp(energies))
    if not math.isfinite(energy_span):
        result["reasons"].append("nonfinite_energy_span")
        return result
    result["fit"].update(energy_span_eV=energy_span, sampled_volume_range_A3=[float(volumes.min()), float(volumes.max())],
                         reference_volume_A3=float(reference_volume), energy_convention="total_cell_energy_eV")
    if energy_span <= np.finfo(float).eps * max(1, float(np.abs(energies).max())) * 100:
        result["reasons"].append("flat_energy_curve_does_not_resolve_positive_curvature")
        return result
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = EquationOfState(volumes, energies, eos="birchmurnaghan")
            v0, e0, b0 = fitted.fit(warn=False)
        bp = float(fitted.eos_parameters[2])
        raw_parameters = [float(e0), float(v0), float(b0), bp]
        result["fit"]["warnings"] = [str(entry.message) for entry in caught]
        if not all(math.isfinite(x) for x in raw_parameters) or v0 <= 0:
            result["fit"]["raw_parameters"] = [x if math.isfinite(x) else str(x) for x in raw_parameters]
            raise ValueError("nonfinite_or_nonpositive_fit_volume")
        predicted = np.asarray(birchmurnaghan(volumes, *fitted.eos_parameters), dtype=float)
        residual = energies - predicted
        rms = float(np.linalg.norm(residual) / math.sqrt(len(residual)))
        b_gpa = float(b0) * EV_PER_ANGSTROM3_TO_GPA
        curvature = float(b0 / v0)
        shift = abs(float(v0) - reference_volume) / reference_volume
        relative_rms = rms / energy_span
        if not all(math.isfinite(value) for value in [rms, b_gpa, curvature, shift, relative_rms]) or not np.isfinite(residual).all():
            raise ValueError("nonfinite_fit_diagnostics")
        interior = bool(volumes.min() < v0 < volumes.max())
        result["fit"].update(E0_eV=float(e0), V0_A3=float(v0), B0_eV_A3=float(b0), B0_GPa=b_gpa,
                             Bprime=bp, curvature_eV_A6=curvature, rms_residual_eV=rms,
                             relative_rms_residual_to_energy_span=relative_rms,
                             max_abs_residual_eV=float(np.abs(residual).max()), residuals_eV=residual.tolist(),
                             predicted_total_energies_eV=predicted.tolist(), minimum_is_interior=interior,
                             relative_reference_volume_shift=shift)
        if b0 <= 0 or curvature <= 0:
            result["reasons"].append("nonpositive_curvature")
        if not interior:
            result["reasons"].append("fitted_minimum_outside_sampled_interior")
        if rms > parameters["fit_rms_max_eV"]:
            result["reasons"].append("absolute_fit_residual_exceeds_threshold")
        if relative_rms > parameters["max_relative_rms_residual_to_energy_span"]:
            result["reasons"].append("relative_fit_residual_exceeds_threshold")
        if shift > parameters["max_relative_volume_shift"]:
            result["reasons"].append("fitted_volume_shift_exceeds_reference_tolerance")
    except (ValueError, RuntimeError, TypeError, OverflowError, np.linalg.LinAlgError) as exc:
        result["reasons"].append(f"eos_fit_failed:{exc}")
    result["quality_status"] = "rejected" if result["reasons"] else "accepted"
    if not result["reasons"]:
        result["value"] = result["fit"]["B0_GPa"]
    return result


def eos(reference, context, parameters, relax_parameters):
    """Sample seven independent fixed-cell internal relaxations on one reference."""
    import numpy as np
    from ..geometry import atoms_from_snapshot, snapshot_from_atoms

    _validate(parameters)
    base = {"schema_version": "ctb.eos.v1", "recipe": RECIPE, "property_id": "bulk_modulus_eos_GPa",
            "reference_geometry_id": reference["geometry_id"], "value": None, "unit": "GPa",
            "quality_status": "unverified", "calculation_status": "missing", "points": [], "reasons": [],
            "physics_fidelity": getattr(context, "physics_fidelity", "unknown"), "benchmark_eligible": False}
    if reference.get("stationary_reference_accepted") is not True:
        base["reasons"].append("qualified_stationary_reference_required")
        return base
    initial_atoms = atoms_from_snapshot(reference)
    reference_volume = float(initial_atoms.get_volume())
    inner_signature=effective_relaxation_signature(relax_parameters,fixed_cell=True)
    point_dependencies = []
    interrupted = False
    for index, ratio in enumerate(parameters["volume_factors"]):
        point_id = f"eos-volume-{index}"
        if interrupted:
            base["points"].append({"point_id": point_id, "volume_ratio": ratio, "quality_status": "unverified",
                                   "calculation_status": "skipped", "reasons": ["previous_point_unqualified"]})
            continue
        # Every point starts from the same reference, never from its predecessor.
        point_atoms = initial_atoms.copy()
        point_atoms.set_cell(np.asarray(initial_atoms.cell) * ratio ** (1 / 3), scale_atoms=True)
        strained = snapshot_from_atoms(point_atoms, parent=reference["geometry_id"], stage="eos_volume_sample",
            transformation={"kind": "isotropic_fixed_shape_volume_scale", "volume_ratio": ratio,
                            "linear_scale": ratio ** (1 / 3), "site_mapping": list(range(len(point_atoms))),
                            "basis_mapping": (np.eye(3) * ratio ** (1 / 3)).tolist()})

        def compute_point(strained=strained, point_id=point_id, ratio=ratio):
            relaxed = relax(strained, context, relax_parameters, fixed_cell=True, node_id=point_id)
            observation = relaxed.get("observation") or {}
            last = relaxed.get("reference_geometry") or relaxed.get("last_geometry")
            actual_volume = float(np.linalg.det(last["cell"])) if last else None
            record = {"point_id": point_id, "volume_ratio": ratio, "source_reference_geometry_id": reference["geometry_id"],
                      "cache_signature_schema":POINT_SIGNATURE_VERSION,
                      "relaxation_signature_digest":digest(inner_signature),
                      "initial_geometry": strained, "final_geometry": relaxed.get("reference_geometry"),
                      "volume_A3": actual_volume, "energy_eV": observation.get("energy_eV"),
                      "quality_status": relaxed["quality_status"], "calculation_status": relaxed["calculation_status"],
                      "reasons": list(relaxed["reasons"]), "relaxation": relaxed}
            if actual_volume is None or not math.isclose(actual_volume, reference_volume * ratio, rel_tol=1e-10, abs_tol=1e-10):
                record["quality_status"] = "rejected"
                record["reasons"].append("fixed_cell_volume_changed_during_internal_relaxation")
            return record

        try:
            context.budget.check()
            point = context.cached(stage="eos-point", geometry=strained,
                parameters={"recipe": RECIPE,"cache_signature_schema":POINT_SIGNATURE_VERSION,
                            "volume_ratio": ratio, "internal_relaxation": inner_signature,
                            "fixed_cell_shape": True, "ase_version": version("ase")},
                dependencies=[reference["geometry_id"]], compute=compute_point)
        except Exception as exc:
            point = {"point_id": point_id, "volume_ratio": ratio, "quality_status": "unverified",
                     "calculation_status": "failed", "reasons": [f"eos_point_interrupted:{type(exc).__name__}:{exc}"]}
        base["points"].append(point)
        if point["quality_status"] != "accepted":
            interrupted = True
        else:
            observation = point["relaxation"]["observation"]
            point_dependencies.append(digest({
                "reference_geometry_id": reference["geometry_id"], "volume_ratio": point["volume_ratio"],
                "initial_geometry_id": point["initial_geometry"]["geometry_id"],
                "final_geometry_id": point["final_geometry"]["geometry_id"],
                "volume_A3": point["volume_A3"], "energy_eV": point["energy_eV"],
                "forces_eV_A": observation["forces_eV_A"], "stress_eV_A3": observation["stress_eV_A3"],
                "effective_relaxation": point["relaxation"]["diagnostics"]["effective_parameters"],
                "quality_status": point["quality_status"],
            }))
    if interrupted:
        base["calculation_status"] = "failed"
        base["reasons"].append("all_seven_prescribed_points_must_be_qualified")
        return base
    try:
        fitted = context.cached(stage="eos-fit", geometry=reference,
            parameters={"recipe": RECIPE, **deepcopy(parameters), "ase_version": version("ase"), "scipy_version": version("scipy")},
            dependencies=point_dependencies,
            compute=lambda: fit_birch_murnaghan(base["points"], reference_volume, parameters))
    except Exception as exc:
        base["calculation_status"] = "failed"
        base["reasons"].append(f"eos_fit_interrupted:{type(exc).__name__}:{exc}")
        return base
    base.update(fitted)
    return base
