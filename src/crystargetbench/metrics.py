"""Submission-denominator metrics; no backend calls or scientific recomputation."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal

from .assessment import measurement_eligibility


def _fraction(numerator: int, denominator: int, definition: str, subset: str = "all_submitted") -> dict:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "status": "available" if denominator else "undefined_zero_denominator",
        "subset": subset,
        "definition": definition,
    }


def _eligible(task: dict, result: dict, alias: str) -> bool:
    if result.get("input_status", "valid") != "valid":
        return False
    eligibility = result.get("measurement_eligibility", {}).get(alias)
    if eligibility is not None:
        return bool(eligibility.get("eligible"))
    return not measurement_eligibility(
        task["measurements"][alias], result.get("measurements", {}).get(alias), result.get("reference_geometry_id")
    )


def _observed(result: dict, alias: str) -> bool:
    observation = result.get("measurements", {}).get(alias, {})
    return observation.get("value") is not None or "raw_value" in observation


def _distribution(values: list[float], observed_count: int, subset: str) -> dict:
    ordered = sorted(values)
    count = len(ordered)
    mean = float(sum((Decimal(str(value)) for value in ordered), Decimal(0)) / count) if count else None
    midpoint = (
        ordered[count // 2] if count % 2
        else float((Decimal(str(ordered[count // 2 - 1])) + Decimal(str(ordered[count // 2]))) / 2)
    ) if count else None
    return {
        "sample_count": len(values),
        "accepted_count": len(values),
        "observed_count": observed_count,
        "status": "available" if values else "unavailable",
        "subset": subset,
        "definition": "Finite accepted scalar observations; no imputation or deduplication",
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": mean,
        "median": midpoint,
    }


def _categorical_distribution(values, observed_count, subset):
    return {"kind": "categorical", "histogram": dict(sorted(Counter(str(int(x)) for x in values).items(), key=lambda pair: int(pair[0]))),
            "sample_count": len(values), "accepted_count": len(values), "observed_count": observed_count,
            "subset": subset, "status": "available" if values else "unavailable",
            "definition": "Space group categories; each submitted item counts, including duplicates"}


def aggregate(task: dict, results: list[dict]) -> dict:
    """Aggregate TaskResults returned by ``assess`` for this exact task.

    Results are consumed as an immutable assessment record. This is not an
    importer of untrusted or independently edited observation/decision bundles;
    such observations must first go through ``assess`` with their fixed routes.
    All submitted entries, including duplicates and invalid inputs, are retained.
    """
    submitted = len(results)
    counts = Counter(
        result.get("decision", "unknown") if result.get("input_status", "valid") == "valid" and result.get("decision") in {"pass", "fail", "unknown"} else "unknown"
        for result in results
    )
    coverage = {}
    distributions = {}
    for alias in task["measurements"]:
        accepted = [result for result in results if _eligible(task, result, alias)]
        values = [result["measurements"][alias]["value"] for result in accepted]
        observed = sum(_observed(result, alias) for result in results)
        coverage[alias] = _fraction(len(accepted), submitted, "Accepted observations / all submitted items")
        if task["measurements"][alias]["property_id"] == "space_group_number":
            distributions[alias] = _categorical_distribution(values, observed, "accepted_measurements")
        else:
            distributions[alias] = _distribution(values, observed, "accepted_measurements")
    complete = sum(all(_eligible(task, result, alias) for alias in task["measurements"]) for result in results)
    metrics = {
        "schema_version": "ctb.metrics.v1",
        "task_id": task["task_id"],
        "mode": task["mode"],
        "n_submitted": submitted,
        "n_input_valid": sum(result.get("input_status", "valid") == "valid" for result in results),
        "n_input_invalid": sum(result.get("input_status", "valid") == "invalid" for result in results),
        "n_input_unsupported": sum(result.get("input_status", "valid") == "unsupported" for result in results),
        "coverage": coverage,
        "distributions": distributions,
        "full_measurement_coverage": _fraction(complete, submitted, "Items with all requested observations accepted / all submitted items"),
        "generation_cost": {"value": None, "status": "unavailable", "definition": "Generator cost is outside the submitted-structure boundary"},
        "novelty": {"value": None, "status": "unavailable_reference_library", "definition": "No validated reference library was provided"},
        "sun": {"value": None, "status": "unavailable_reference_library", "definition": "SUN requires a defined, versioned reference library"},
        "benchmark_eligible": False,
    }
    if task["mode"] == "measure":
        return metrics
    passed, failed, unknown = counts["pass"], counts["fail"], counts["unknown"]
    metrics.update(n_pass=passed, n_fail=failed, n_unknown=unknown)
    metrics["joint_determined_fraction"] = _fraction(passed + failed, submitted, "(Pass + fail) / all submitted items")
    metrics["determined_fraction"] = dict(metrics["joint_determined_fraction"])
    metrics["pass_fraction_determined"] = _fraction(passed, passed + failed, "Pass / (pass + fail)", "determined_items")
    metrics["unknown_fraction"] = _fraction(unknown, submitted, "Unknown / all submitted items")
    metrics["constraints"] = {}
    for constraint in task.get("constraints", []):
        decisions = Counter(
            next((entry["decision"] for entry in result.get("constraints", []) if entry["id"] == constraint["id"]), "unknown")
            if result.get("input_status", "valid") == "valid" else "unknown"
            for result in results
        )
        metrics["constraints"][constraint["id"]] = {
            "n_pass": decisions["pass"], "n_fail": decisions["fail"], "n_unknown": decisions["unknown"],
            "coverage": _fraction(decisions["pass"] + decisions["fail"], submitted, "Determined condition evaluations / all submitted items"),
        }
    if task["mode"] == "screen":
        metrics["verified_yield"] = _fraction(passed, submitted, "Pass / all submitted items")
        return metrics
    metrics["constraint_yield"] = _fraction(passed, submitted, "Feasible items / all submitted items; not a cutoff on the ranking property")
    metrics["n_feasible"] = passed
    alias = task["ranking"]["measurement"]
    rankable = [
        (index, result) for index, result in enumerate(results)
        if result.get("decision") == "pass" and _eligible(task, result, alias)
    ]
    reverse = task["ranking"]["direction"] == "maximize"
    rankable.sort(key=lambda pair: pair[1]["measurements"][alias]["value"], reverse=reverse)
    metrics["n_rankable"] = len(rankable)
    metrics["rankable_yield"] = _fraction(len(rankable), submitted, "Feasible items with accepted ranking measurement / all submitted items")
    metrics["ranking"] = {
        "measurement": alias,
        "direction": task["ranking"]["direction"],
        "subset": "feasible_and_ranking_measurement_accepted",
        "items": [
            {"item_id": result.get("item_id"), "submission_index": index, "value": result["measurements"][alias]["value"]}
            for index, result in rankable
        ],
        "distribution": (_categorical_distribution if task["measurements"][alias]["property_id"] == "space_group_number" else _distribution)(
            [result["measurements"][alias]["value"] for _, result in rankable],
            sum(result.get("decision") == "pass" and _observed(result, alias) for result in results),
            "feasible_and_ranking_measurement_accepted",
        ),
    }
    return metrics
