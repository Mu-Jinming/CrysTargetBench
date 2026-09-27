"""Pure, three-valued assessment of typed measurements.

Nothing in this module executes a backend or repairs a measured value.  An
accepted observation can fail a scientific constraint; an unusable observation
is unknown, even if its raw number would fail that constraint.
"""

from __future__ import annotations

import math
from copy import deepcopy
from decimal import Decimal
from typing import Any

from .identity import digest


CALCULATION_STATUSES = {"completed", "failed", "skipped", "unsupported", "missing"}
QUALITY_STATUSES = {"accepted", "rejected", "unverified"}
IDENTITY_STATUSES = {"verified", "unverified", "mismatch"}
FIDELITIES = {"native", "mlff", "dft", "surrogate", "mixed", "synthetic", "unknown"}


def _finite_scalar(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _json_safe(value: Any) -> Any:
    """Retain non-finite diagnostics as strings, never invalid JSON tokens."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _has_nonfinite(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return not _finite_scalar(value)
    if isinstance(value, dict):
        return any(_has_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonfinite(item) for item in value)
    return False


def make_measurement(**kwargs: Any) -> dict:
    """Build the complete observation envelope with conservative defaults.

    A caller must explicitly attest calculation, quality and identity status.
    Synthetic observations can exercise software assessment, but never acquire
    benchmark eligibility or satisfy a route requiring real DFT evidence.
    """
    measurement = {
        "measurement_id": None,
        "reference_geometry_id": None,
        "item_links": [],
        "property_id": None,
        "definition_id": None,
        "value": None,
        "unit": None,
        "calculation_status": "missing",
        "quality_status": "unverified",
        "identity_status": "unverified",
        "reasons": [],
        "backend_id": None,
        "source_backend_id": None,
        "backend_family": None,
        "backend_version": None,
        "model_digest": None,
        "asset_digests": {},
        "physics_fidelity": "unknown",
        "upstream_fidelity": [],
        "protocol_digest": None,
        "artifact_refs": [],
        "dependency_measurement_ids": [],
        "diagnostics": {},
        "benchmark_eligible": False,
    }
    measurement.update(deepcopy(kwargs))
    for field, allowed in (
        ("calculation_status", CALCULATION_STATUSES),
        ("quality_status", QUALITY_STATUSES),
        ("identity_status", IDENTITY_STATUSES),
        ("physics_fidelity", FIDELITIES),
    ):
        if measurement[field] not in allowed:
            raise ValueError(f"Invalid {field}: {measurement[field]!r}")
    if _has_nonfinite(measurement["value"]):
        measurement["raw_value"] = str(measurement["value"])
        measurement["value"] = None
        measurement["quality_status"] = "rejected"
        measurement["reasons"].append("nonfinite_value")
    if isinstance(measurement["value"], bool):
        measurement["raw_value"] = str(measurement["value"])
        measurement["value"] = None
        measurement["quality_status"] = "rejected"
        measurement["reasons"].append("boolean_is_not_numeric")
    if (
        measurement["backend_family"] == "synthetic"
        or measurement["physics_fidelity"] == "synthetic"
        or "synthetic" in measurement["upstream_fidelity"]
        or measurement.get("protocol_phase", "candidate") != "released"
    ):
        measurement["benchmark_eligible"] = False
    measurement = _json_safe(measurement)
    if measurement["measurement_id"] is None:
        measurement["measurement_id"] = digest(
            {key: value for key, value in measurement.items() if key not in {"measurement_id", "item_links"}}
        )
    return measurement


def measurement_eligibility(
    specification: dict,
    measurement: dict | None,
    reference_geometry_id: str | None,
    route: dict | str | None = None,
) -> list[str]:
    """Return reasons that prevent this observation from being compared."""
    if not measurement:
        return ["measurement_missing"]
    reasons = []
    for key, expected in (
        ("calculation_status", "completed"),
        ("quality_status", "accepted"),
        ("identity_status", "verified"),
    ):
        if measurement.get(key) != expected:
            reasons.append(f"{key}:{measurement.get(key, 'missing')}")
    if not reference_geometry_id or measurement.get("reference_geometry_id") != reference_geometry_id:
        reasons.append("reference_geometry_mismatch_or_unbound")
    if measurement.get("property_id") != specification["property_id"]:
        reasons.append("property_mismatch")
    if measurement.get("unit") != specification["unit"]:
        reasons.append("unit_mismatch")
    required_definition = specification.get("definition_id")
    if isinstance(route, dict):
        required_definition = required_definition or route.get("definition_id")
    if not measurement.get("definition_id"):
        reasons.append("definition_missing")
    elif required_definition and measurement["definition_id"] != required_definition:
        reasons.append("definition_mismatch")

    required_fidelity = specification.get("required_fidelity", "protocol_default")
    if required_fidelity == "protocol_default":
        if isinstance(route, str):
            required_fidelity = route
        elif isinstance(route, dict):
            required_fidelity = (
                route.get("required_fidelity") or route.get("physics_fidelity")
                or route.get("backend_family") or route.get("family")
            )
        else:
            required_fidelity = None
    fidelity = measurement.get("physics_fidelity")
    family = measurement.get("backend_family")
    if isinstance(route, dict) and route.get("provider"):
        source = measurement.get("source_backend_id") if family == "imported" else measurement.get("backend_id")
        if source != route["provider"]:
            reasons.append("backend_provider_mismatch")
    if family not in {"imported", fidelity} and fidelity != "mixed":
        reasons.append("backend_fidelity_mismatch")
    if required_fidelity and required_fidelity != "protocol_default" and fidelity != required_fidelity:
        reasons.append("fidelity_mismatch")
    if fidelity not in FIDELITIES - {"unknown"}:
        reasons.append("fidelity_unverified")
    if required_fidelity == "dft":
        if measurement.get("backend_family") not in {"dft", "imported"}:
            reasons.append("dft_source_required")
        if any(value not in {"dft", "native"} for value in measurement.get("upstream_fidelity", [])):
            reasons.append("mixed_upstream_not_dft")

    for key in ("measurement_id", "backend_id", "backend_version", "protocol_digest"):
        if not measurement.get(key):
            reasons.append(f"{key}_missing")
    value = measurement.get("value")
    if not _finite_scalar(value):
        reasons.append("value_missing_or_not_finite_scalar")
    elif specification["property_id"] == "space_group_number" and (
        not isinstance(value, int) or not 1 <= value <= 230
    ):
        reasons.append("invalid_space_group_integer")
    elif specification["property_id"] == "band_gap_eV" and value < 0:
        reasons.append("negative_physical_band_gap")
    diagnostics = measurement.get("diagnostics", {})
    if isinstance(route, dict) and route.get("requires_stationary_reference") and diagnostics.get("stationary_reference_accepted") is not True:
        reasons.append("stationary_reference_evidence_missing_or_rejected")
    if diagnostics.get("stationary_reference_accepted") is False:
        reasons.append("nonstationary_reference")
    if diagnostics.get("relaxation_converged") is False:
        reasons.append("relaxation_not_converged")
    if diagnostics.get("site_mapping_verified") is False:
        reasons.append("site_mapping_unverified")
    return list(dict.fromkeys(reasons))


def predicate(value: int | float, constraint: dict) -> bool:
    """Evaluate a validated scalar constraint.

    ``within`` uses decimal forms of JSON numbers and a closed boundary.  Thus
    2.1 is within 2 +/- 0.1 without a hidden relative floating point tolerance.
    """
    if not _finite_scalar(value):
        raise ValueError("Predicates require a finite numeric scalar, not boolean")
    operator = constraint["operator"]
    if operator == "gt":
        return value > constraint["value"]
    if operator == "ge":
        return value >= constraint["value"]
    if operator == "lt":
        return value < constraint["value"]
    if operator == "le":
        return value <= constraint["value"]
    if operator == "eq":
        return value == constraint["value"]
    if operator == "in":
        return value in constraint["values"]
    if operator == "within":
        return abs(Decimal(str(value)) - Decimal(str(constraint["value"]))) <= Decimal(str(constraint["atol"]))
    if operator == "interval":
        lower = value >= constraint["lower"] if constraint["lower_inclusive"] else value > constraint["lower"]
        upper = value <= constraint["upper"] if constraint["upper_inclusive"] else value < constraint["upper"]
        return lower and upper
    raise ValueError(f"Unknown predicate operator: {operator}")


def three_valued_and(decisions: list[str]) -> str:
    if any(decision not in {"pass", "fail", "unknown"} for decision in decisions):
        raise ValueError("Invalid three-valued decision")
    if "fail" in decisions:
        return "fail"
    if decisions and all(decision == "pass" for decision in decisions):
        return "pass"
    return "unknown"


def assess(
    task: dict,
    measurements: dict[str, dict],
    reference_geometry_id: str | None,
    routes: dict | None = None,
) -> dict:
    """Assess one submitted item against a fixed shared reference geometry."""
    routes = routes or {}
    eligibility = {}
    for alias, specification in task["measurements"].items():
        reasons = measurement_eligibility(specification, measurements.get(alias), reference_geometry_id, routes.get(alias))
        eligibility[alias] = {"eligible": not reasons, "reasons": reasons}
    constraints = []
    for constraint in task.get("constraints", []):
        alias = constraint["measurement"]
        reasons = list(eligibility[alias]["reasons"])
        decision = "unknown"
        if not reasons:
            decision = "pass" if predicate(measurements[alias]["value"], constraint) else "fail"
        constraints.append({"id": constraint["id"], "measurement": alias, "decision": decision, "reasons": reasons})
    return {
        "decision": None if task["mode"] == "measure" else three_valued_and([item["decision"] for item in constraints]),
        "constraints": constraints,
        "reference_geometry_id": reference_geometry_id,
        "measurement_eligibility": eligibility,
        "measurements": deepcopy(measurements),
    }
