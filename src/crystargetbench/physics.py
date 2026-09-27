"""Small, pure reducers and guards for already available scientific data.

These are not force, phonon, dielectric or EOS backends. S1 tests use synthetic
arrays to validate accounting and missing-data semantics only.
"""

from __future__ import annotations

import math
from decimal import Decimal


EV_PER_ANGSTROM3_TO_GPA = 160.2176634


def _finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def phonon_statistics(frequencies, *, scope, weights=None, qpoints=None, frequency_floor=None, regularization_definition=None):
    """Reduce signed q-by-mode frequencies without deleting acoustic branches.

    The imaginary fraction counts strictly negative modes. Mesh fractions use
    declared q weights; path fractions describe samples, not Brillouin-zone
    volume. A Gamma-only scope never claims path or mesh coverage.
    """
    result = {
        "scope": scope, "min_frequency_THz": None, "imaginary_fraction": None,
        "fraction_definition": "weighted_mesh_mode_fraction" if scope == "mesh" else f"{scope}_sampling_mode_fraction",
        "quality_status": "unverified", "reasons": [], "raw_frequencies_THz": _safe(frequencies),
        "gamma_available": False, "regularization_definition": _safe(regularization_definition),
    }
    if scope not in {"path", "mesh", "gamma"}:
        result["reasons"].append("unknown_sampling_scope")
        return result
    if regularization_definition is not None and (not isinstance(regularization_definition, str) or not regularization_definition):
        result["reasons"].append("invalid_regularization_definition")
        return result
    if frequency_floor is not None and (not _finite_number(frequency_floor) or frequency_floor <= 0):
        result["reasons"].append("invalid_frequency_floor")
        return result
    if frequency_floor is not None and not regularization_definition:
        result["reasons"].append("undeclared_frequency_regularization")
        return result
    if not isinstance(frequencies, (list, tuple)) or not frequencies or any(not isinstance(row, (list, tuple)) or not row for row in frequencies):
        result["reasons"].append("empty_or_malformed_frequencies")
        return result
    modes = len(frequencies[0])
    if any(len(row) != modes or any(not _finite_number(value) for value in row) for row in frequencies):
        result["reasons"].append("nonfinite_or_inconsistent_frequencies")
        return result
    if qpoints is not None:
        if len(qpoints) != len(frequencies) or any(len(q) != 3 or any(not _finite_number(x) for x in q) for q in qpoints):
            result["reasons"].append("invalid_qpoint_mapping")
            return result
        result["gamma_available"] = any(all(abs(x - round(x)) <= 1e-10 for x in q) for q in qpoints)
    if scope == "gamma" and (len(frequencies) != 1 or not result["gamma_available"]):
        result["reasons"].append("gamma_coordinate_required")
        return result
    if scope == "mesh" and weights is None:
        result["reasons"].append("mesh_weights_required")
        return result
    if weights is None:
        weights = [1] * len(frequencies)
    if len(weights) != len(frequencies) or any(not _finite_number(weight) or weight <= 0 for weight in weights):
        result["reasons"].append("invalid_qpoint_weights")
        return result
    result["min_frequency_THz"] = min(value for row in frequencies for value in row)
    decimal_weights = [Decimal(str(weight)) for weight in weights]
    numerator = sum((weight * sum(value < 0 for value in row) for weight, row in zip(decimal_weights, frequencies)), Decimal(0))
    result["imaginary_fraction"] = float(numerator / (sum(decimal_weights) * modes))
    result["quality_status"] = "accepted"
    result["qpoint_count"] = len(frequencies)
    result["mode_count"] = modes
    result["weights"] = list(weights)
    if frequency_floor is not None:
        # Explicit metadata does not authorize this reducer to modify raw modes.
        result["declared_frequency_floor_THz"] = frequency_floor
    return result


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_safe(entry) for entry in value]
    if isinstance(value, dict):
        return {key: _safe(entry) for key, entry in value.items()}
    return value


def _tensor(tensor):
    if not isinstance(tensor, (list, tuple)) or len(tensor) != 3:
        return False
    return all(isinstance(row, (list, tuple)) and len(row) == 3 and all(_finite_number(value) for value in row) for row in tensor)


def dielectric_response(*, electronic=None, ionic=None, total=None, frequency=0.0, gamma_available=True, ionic_definition="bec", upstream_fidelity=(), regularization_definition=None, frequency_floor=None):
    """Validate/reduce a supplied static dielectric tensor; never fabricate parts.

    A supplied total is already total and is never added to an electronic part.
    Component summation requires both actual tensors. Nominal-charge proxies
    cannot enter this total-response helper. No DFT execution is implied.
    """
    lineage = set(upstream_fidelity)
    fidelity = "synthetic" if "synthetic" in lineage else ("mixed" if len(lineage - {"native"}) > 1 else next(iter(lineage - {"native"}), "unknown"))
    result = {"value": None, "tensor": None, "principal_values": None, "anisotropy": None, "quality_status": "unverified", "reasons": [], "physics_fidelity": fidelity, "upstream_fidelity": list(upstream_fidelity), "scalarization": "trace_over_3", "regularization_definition": _safe(regularization_definition)}
    if regularization_definition is not None and (not isinstance(regularization_definition, str) or not regularization_definition):
        result["reasons"].append("invalid_regularization_definition")
        return result
    if frequency_floor is not None and (not _finite_number(frequency_floor) or frequency_floor <= 0):
        result["reasons"].append("invalid_frequency_floor")
        return result
    if not _finite_number(frequency) or frequency != 0:
        result["reasons"].append("static_frequency_zero_missing")
        return result
    if not gamma_available:
        result["reasons"].append("gamma_response_unavailable")
        return result
    if frequency_floor is not None and not regularization_definition:
        result["reasons"].append("undeclared_frequency_regularization")
        return result
    if ionic_definition != "bec":
        result["reasons"].append("nominal_proxy_is_not_total_response")
        return result
    if total is not None:
        if not _tensor(total):
            result["reasons"].append("invalid_total_tensor")
            return result
        tensor = [list(row) for row in total]
        result["source"] = "supplied_total"
    else:
        if not _tensor(electronic) or not _tensor(ionic):
            result["reasons"].append("missing_or_invalid_electronic_or_ionic_tensor")
            return result
        tensor = [[electronic[i][j] + ionic[i][j] for j in range(3)] for i in range(3)]
        result["source"] = "electronic_plus_ionic"
    if not _tensor(tensor):
        result["reasons"].append("nonfinite_total_tensor")
        return result
    if any(abs(tensor[i][j] - tensor[j][i]) > 1e-8 for i in range(3) for j in range(3)):
        result["reasons"].append("nonsymmetric_dielectric_tensor")
        return result
    import numpy as np
    principal = np.linalg.eigvalsh(tensor).tolist()
    if min(principal) <= 0:
        result["reasons"].append("nonpositive_total_dielectric_tensor")
        return result
    scalar = sum(tensor[i][i] / 3 for i in range(3))
    anisotropy = max(principal) / min(principal)
    if not math.isfinite(scalar) or any(not math.isfinite(value) for value in principal) or not math.isfinite(anisotropy):
        result["reasons"].append("nonfinite_tensor_derived_quantity")
        return result
    result.update(value=scalar, tensor=tensor, principal_values=principal, anisotropy=anisotropy, quality_status="accepted")
    return result


def eos_input_quality(volumes_A3, energies_eV, *, minimum_points=5):
    """Validate input points only; acceptance here does not claim an EOS fit."""
    result = {"quality_status": "unverified", "reasons": [], "bulk_modulus_GPa": None, "fit_status": "not_performed"}
    if len(volumes_A3) != len(energies_eV) or len(volumes_A3) < minimum_points:
        result["reasons"].append("insufficient_volume_energy_points")
    elif any(not _finite_number(v) or v <= 0 for v in volumes_A3) or any(not _finite_number(e) for e in energies_eV):
        result["reasons"].append("invalid_volume_energy_points")
    elif len(set(volumes_A3)) < minimum_points:
        result["reasons"].append("insufficient_distinct_volumes")
    else:
        result["input_quality_status"] = "accepted"
        result["reasons"].append("eos_fit_and_curvature_not_verified")
    return result


def pressure_to_gpa(value, unit):
    """Convert only explicitly named pressure units, with no unit inference."""
    if not _finite_number(value):
        raise ValueError("Pressure must be a finite scalar")
    if unit == "GPa":
        return value
    if unit == "eV/angstrom^3":
        converted = value * EV_PER_ANGSTROM3_TO_GPA
        if not math.isfinite(converted):
            raise ValueError("Pressure conversion produced a nonfinite result")
        return converted
    raise ValueError(f"Unsupported pressure unit: {unit}")
