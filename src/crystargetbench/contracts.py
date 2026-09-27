"""Strict, data-only contracts for tasks, science, routing, and deployment.

Candidate protocols are intentionally not scientific validation or permission
to execute.  This module reads package resources only; it never probes workers.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, validators


PROPERTY_UNITS = {
    "phonon_path_min_frequency_THz": "THz",
    "phonon_mesh_min_frequency_THz": "THz",
    "band_gap_eV": "eV",
    "space_group_number": "1",
    "bulk_modulus_voigt_GPa": "GPa",
    "bulk_modulus_reuss_GPa": "GPa",
    "bulk_modulus_hill_GPa": "GPa",
    "bulk_modulus_eos_GPa": "GPa",
    "dielectric_ionic_nominal_trace_over3": "1",
    "dielectric_electronic_trace_over3": "1",
    "dielectric_ionic_bec_trace_over3": "1",
    "dielectric_total_static_trace_over3": "1",
}

# JSON Schema calls 225.0 an integer. CTB requires an actual integer token.
_TYPES = Draft202012Validator.TYPE_CHECKER.redefine_many(
    {
        "integer": lambda checker, value: type(value) is int,
        "number": lambda checker, value: type(value) is int
        or (type(value) is float and math.isfinite(value)),
    }
)
_Validator = validators.extend(Draft202012Validator, type_checker=_TYPES)


def _finite_data(value: Any, path: str = "$") -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{path}: nonfinite number is forbidden")
    if isinstance(value, dict):
        for key, member in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: JSON object keys must be strings")
            _finite_data(member, f"{path}.{key}")
    elif isinstance(value, list):
        for index, member in enumerate(value):
            _finite_data(member, f"{path}[{index}]")
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError(f"{path}: expected JSON data, got {type(value).__name__}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON token: {value}")


def _loads(text: str) -> dict:
    try:
        result = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("JSON document must be an object")
    _finite_data(result)
    return result


def load_json(path: str | Path) -> dict:
    """Read an explicit JSON file, rejecting duplicate keys and nonfinite data."""
    try:
        return _loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc


def _resource_json(*parts: str) -> dict:
    resource = files("crystargetbench").joinpath("resources", *parts)
    return _loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _schema(name: str) -> dict:
    return _resource_json("schemas", f"{name}.schema.json")


def _validate(data: dict, schema: dict, label: str) -> dict:
    _finite_data(data)
    error = next(_Validator(schema).iter_errors(data), None)
    if error is not None:
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        raise ValueError(f"{label}.{path}: {error.message}")
    return deepcopy(data)


def validate_task(data: dict) -> dict:
    """Validate a v2 task without changing its scientific meaning."""
    task = _validate(data, _schema("task"), "task")
    measurements = task["measurements"]
    for alias, measurement in measurements.items():
        if not alias.strip():
            raise ValueError("measurement aliases must not be blank")
        if alias in {"geometry", "warm_start"}:
            raise ValueError(f"measurement alias {alias} is reserved for a workflow stage")
        prop = measurement["property_id"]
        if measurement["unit"] != PROPERTY_UNITS[prop]:
            raise ValueError(f"measurement {alias}: {prop} requires unit {PROPERTY_UNITS[prop]}")
        definition = measurement.get("definition_id")
        if definition:
            if "latest" in definition.lower().replace(":", ".").replace("@", ".").split("."):
                raise ValueError("definition_id must be pinned; latest is not a definition")
            prefix = definition.split("@", 1)[0]
            if prefix in PROPERTY_UNITS and prefix != prop:
                raise ValueError(f"measurement {alias}: incompatible definition_id")
        if prop == "space_group_number" and measurement.get("required_fidelity", "native") not in (
            "native", "protocol_default"
        ):
            raise ValueError("space_group_number is a native geometric measurement")

    identifiers = set()
    for constraint in task.get("constraints", []):
        identifier = constraint["id"]
        if identifier in identifiers:
            raise ValueError(f"duplicate constraint ID: {identifier}")
        identifiers.add(identifier)
        alias = constraint["measurement"]
        if alias not in measurements:
            raise ValueError(f"constraint {identifier}: unknown measurement alias {alias}")
        operator = constraint["operator"]
        if operator == "interval":
            lower, upper = constraint["lower"], constraint["upper"]
            if lower > upper or (
                lower == upper and not (constraint["lower_inclusive"] and constraint["upper_inclusive"])
            ):
                raise ValueError(f"constraint {identifier}: empty interval")
        if measurements[alias]["property_id"] == "space_group_number":
            if operator not in ("eq", "in"):
                raise ValueError("space_group_number supports exact eq/in constraints")
            targets = constraint.get("values", [constraint.get("value")])
            if any(type(target) is not int or not 1 <= target <= 230 for target in targets):
                raise ValueError("space-group targets must be integer tokens in 1..230")
        elif operator in ("eq", "in"):
            raise ValueError("continuous properties require inequalities, within, or interval")
    for alias in task.get("report", []):
        if alias not in measurements:
            raise ValueError(f"report references unknown measurement alias: {alias}")
    if "ranking" in task and task["ranking"]["measurement"] not in measurements:
        raise ValueError("ranking references unknown measurement alias")
    return task


def validate_policy(data: dict) -> dict:
    policy = _validate(data, _schema("method_policy"), "method_policy")
    mode = policy["mode"]
    if mode in ("mlff", "dft") and policy["allow_surrogates"]:
        raise ValueError(f"strict {mode} policy cannot allow surrogates")
    if mode != "auto" and policy["promote_on_missing_capability"]:
        raise ValueError("capability promotion is only valid in auto mode")
    for alias, route in policy["overrides"].items():
        if not alias.strip():
            raise ValueError("route aliases must not be blank")
        if route.startswith("surrogate:") and not policy["allow_surrogates"]:
            raise ValueError("surrogate routes require allow_surrogates=true")
    return policy


def _object(properties: dict, required: tuple | list | None = None) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "properties": properties,
        "required": list(properties) if required is None else list(required),
    }


_POSITIVE = {"type": "number", "exclusiveMinimum": 0}
_NONNEGATIVE = {"type": "number", "minimum": 0}
_NUMBER = {"type": "number"}
_BOOL = {"type": "boolean"}
_PIN = {"type": ["string", "null"], "minLength": 1}
_MATRIX = {
    "type": "array", "minItems": 3, "maxItems": 3,
    "items": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer"}},
}
_PARAMETERS = _object(
    {
        "symprec_A": _POSITIVE,
        "angle_tolerance_deg": {"type": "number", "minimum": -1, "maximum": 180},
        "target_symmetrization": {"const": False},
        "scope_notice": {"type": "string", "minLength": 1},
        "relaxation": _object({
            "require_converged": {"const": True},
            "fmax_eV_per_A": _POSITIVE,
            "stress_max_GPa": _NONNEGATIVE,
            "max_steps": {"type": "integer", "minimum": 1},
            "target_pressure_GPa": _NUMBER,
            "optimizer": {"const": "FIRE"}, "filter": {"const": "FrechetCellFilter"},
            "cell_mask": {"type": "array", "minItems": 6, "maxItems": 6, "items": _BOOL},
            "dt": _POSITIVE, "dtmax": _POSITIVE, "maxstep_A": _POSITIVE,
            "max_wall_seconds": _POSITIVE,
        }, required=("require_converged", "fmax_eV_per_A", "stress_max_GPa", "max_steps", "target_pressure_GPa")),
        "phonon": _object({
            "supercell_matrix": _MATRIX,
            "displacement_A": _POSITIVE,
            "is_plusminus": {"enum": ["auto", True, False]},
            "force_drift_subtraction": _BOOL,
            "force_constant_symmetrization": _BOOL,
            "nac": {"enum": ["none", "from_dft_born_and_electronic"]},
            "q_scope": {"enum": ["path", "mesh", "path_and_mesh"]},
            "path_points_parameter": {"type": "integer", "minimum": 2},
            "mesh": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": {"type": "integer", "minimum": 1},
            },
            "gamma_center": _BOOL,
            "primitive_matrix": {"oneOf": [{"const": "auto"}, {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": {"type": "array", "minItems": 3, "maxItems": 3, "items": _NUMBER}}]},
            "symprec_A": _POSITIVE,
            "masses": {"type": ["array", "null"], "minItems": 1, "items": _POSITIVE},
            "is_diagonal": _BOOL, "is_symmetry": _BOOL,
            "fc_calculator": {"const": "traditional"},
            "symmetrization_level": {"type": "integer", "minimum": 1},
            "is_time_reversal": _BOOL, "is_mesh_symmetry": _BOOL,
        }, required=(
            "supercell_matrix", "displacement_A", "is_plusminus",
            "force_drift_subtraction", "force_constant_symmetrization", "nac", "q_scope",
        )),
        "band_gap": _object({
            "definition": {"enum": ["kmesh_generalized_ks", "explicit_surrogate_prediction"]},
            "functional": {"type": "string", "minLength": 1},
            "mesh_policy_id": _PIN,
            "spin_u_soc_policy_id": _PIN,
        }),
        "dielectric": _object({
            "components": {
                "type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"enum": ["electronic", "ionic"]},
            },
            "frequency": {"const": "static"},
            "boundary": {"const": "fixed_strain"},
            "scalarization": {"const": "trace_over_3"},
            "negative_optical_mode_policy": {"const": "reject_quality"},
            "silent_frequency_floor": {"const": False},
        }),
        "dft": _object({
            "parameter_set_id": _PIN,
            "pseudopotential_manifest_sha256": {
                "type": ["string", "null"], "pattern": "^[0-9a-f]{64}$",
            },
            "cutoff_and_convergence_profile": _PIN,
        }),
        "elastic": _object({
            "strain_magnitudes": {
                "type": "array", "minItems": 4, "uniqueItems": True,
                "items": {"type": "number", "exclusiveMinimum": -1, "exclusiveMaximum": 1},
            },
            "fit": {"const": "linear_stress_strain"},
            "internal_relaxation": _BOOL,
        }),
        "eos": _object({
            "volume_factors": {
                "type": "array", "minItems": 5, "uniqueItems": True,
                "items": _POSITIVE,
            },
            "fit": {"const": "birch_murnaghan_3rd_order"},
            "internal_relaxation": _BOOL,
            "fixed_cell_shape": {"const": True}, "fit_rms_max_eV": _POSITIVE,
            "max_relative_volume_shift": _NONNEGATIVE,
            "max_relative_rms_residual_to_energy_span": _POSITIVE,
        }, required=("volume_factors", "fit", "internal_relaxation")),
        "nominal_proxy": _object({
            "charge_assignment": {"const": "explicit_species_map"},
            "nominal_charges": {"type": ["object", "null"], "additionalProperties": _NUMBER},
            "regularization": {"enum": ["none", "explicit_frequency_floor"]},
            "frequency_floor_THz": {"type": ["number", "null"], "exclusiveMinimum": 0},
        }),
    }, required=(),
)


def validate_protocol(data: dict) -> dict:
    protocol = _validate(data, _schema("protocol"), "protocol")
    params = _validate(protocol["parameters"], _PARAMETERS, "protocol.parameters")
    recipe_keys = set(params) - {"scope_notice", "dft"}
    if not recipe_keys:
        raise ValueError("protocol must specify at least one typed scientific recipe")
    symmetry_keys = {"symprec_A", "angle_tolerance_deg", "target_symmetrization"}
    if symmetry_keys & params.keys() and not symmetry_keys <= params.keys():
        raise ValueError("symmetry recipe requires symprec_A, angle_tolerance_deg, target_symmetrization")
    if "angle_tolerance_deg" in params and -1 < params["angle_tolerance_deg"] < 0:
        raise ValueError("angle_tolerance_deg must be -1 (automatic) or nonnegative")
    if "phonon" in params:
        phonon = params["phonon"]
        a, b, c = phonon["supercell_matrix"]
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        if determinant <= 0:
            raise ValueError("phonon supercell_matrix must be nonsingular and right-handed")
        scope = phonon["q_scope"]
        if scope in ("path", "path_and_mesh") and "path_points_parameter" not in phonon:
            raise ValueError("path phonon recipe requires path_points_parameter")
        if scope in ("mesh", "path_and_mesh") and not {"mesh", "gamma_center"} <= phonon.keys():
            raise ValueError("mesh phonon recipe requires mesh and gamma_center")
        if scope == "path" and {"mesh", "gamma_center"} & phonon.keys():
            raise ValueError("path-only protocol cannot carry unused mesh parameters")
        if scope == "mesh" and "path_points_parameter" in phonon:
            raise ValueError("mesh-only protocol cannot carry unused path parameters")
        primitive = phonon.get("primitive_matrix")
        if isinstance(primitive, list):
            a, b, c = primitive
            determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
            if determinant <= 0:
                raise ValueError("primitive_matrix must be nonsingular and right-handed")
    if "relaxation" in params and "optimizer" in params["relaxation"]:
        relaxation = params["relaxation"]
        required = {"optimizer", "filter", "cell_mask", "dt", "dtmax", "maxstep_A", "max_wall_seconds"}
        if not required <= relaxation.keys():
            raise ValueError("Incomplete S2 FIRE/Frechet relaxation settings")
        if relaxation["dtmax"] < relaxation["dt"] or relaxation["stress_max_GPa"] <= 0:
            raise ValueError("S2 dtmax must be at least dt and stress tolerance must be positive")
    if "elastic" in params:
        strains = params["elastic"]["strain_magnitudes"]
        if 0 in strains or not min(strains) < 0 < max(strains):
            raise ValueError("elastic strain sampling must bracket zero with nonzero strains")
    if "eos" in params:
        factors = params["eos"]["volume_factors"]
        if not min(factors) < 1 < max(factors):
            raise ValueError("EOS volume factors must bracket the reference volume")
        if "fixed_cell_shape" in params["eos"]:
            required = {"fixed_cell_shape", "fit_rms_max_eV", "max_relative_volume_shift", "max_relative_rms_residual_to_energy_span"}
            if not required <= params["eos"].keys() or len(factors) != 7 or factors != sorted(factors) or 1.0 not in factors or params["eos"]["internal_relaxation"] is not True:
                raise ValueError("S2 EOS requires seven ordered points including one and complete fit quality limits")
    if "nominal_proxy" in params:
        proxy = params["nominal_proxy"]
        has_floor = proxy["frequency_floor_THz"] is not None
        if (proxy["regularization"] == "explicit_frequency_floor") != has_floor:
            raise ValueError("nominal proxy frequency floor must match its explicit regularization")
    for group in ("dft", "band_gap"):
        for key, value in params.get(group, {}).items():
            if isinstance(value, str) and value.lower() in ("latest", "default", "auto"):
                raise ValueError(f"protocol {group}.{key} must be a pinned value or unresolved null")
    return protocol


def validate_deployment(data: dict) -> dict:
    deployment = _validate(data, _schema("deployment"), "deployment")
    providers = deployment["providers"]
    for family in ("mlff", "dft"):
        selected = deployment[f"default_{family}_provider"]
        if selected is None and deployment["schema_version"] == "ctb.deployment.v3":
            continue
        if selected not in providers or providers[selected]["family"] != family:
            raise ValueError(f"default_{family}_provider must name a registered deployment {family} provider")
    for provider in providers.values():
        if any("\x00" in argument for argument in provider["worker_argv"]):
            raise ValueError("worker argv cannot contain NUL bytes")
    return deployment


def default_policy(mode: str = "auto") -> dict:
    """Return a fresh policy; hybrid requires explicit routes from the caller."""
    if mode not in ("auto", "mlff", "dft"):
        raise ValueError("hybrid requires an explicit policy with per-property routes")
    policy = _resource_json("examples", "policies", "default_auto.json")
    policy["mode"] = mode
    policy["promote_on_missing_capability"] = mode == "auto"
    return validate_policy(policy)


def default_deployment() -> dict:
    deployment = _resource_json("examples", "deployments", "local.template.json")
    # The native adapter is part of core; actual installed versions/readiness
    # are still resolved by doctor rather than inferred from this flag.
    deployment["providers"]["symmetry"]["configured"] = True
    deployment["execution"].update(per_job_cpu_cores=1, max_scratch_gb=10.0)
    return validate_deployment(deployment)


def default_protocol(task: dict) -> dict:
    """Build a candidate protocol from properties, independently of thresholds."""
    task = validate_task(task)
    properties = {value["property_id"] for value in task["measurements"].values()}
    symmetry = _resource_json("examples", "protocols", "symmetry.candidate.json")
    path = _resource_json("examples", "protocols", "phonon_path.candidate.json")
    highk = _resource_json("examples", "protocols", "highk_dft.candidate.json")
    result = deepcopy(symmetry)
    result["reference_geometry"] = task["reference_geometry"]
    params = {}
    recipes = []
    if "space_group_number" in properties:
        params.update(symmetry["parameters"])
        recipes.append("symmetry")
    physical = properties - {"space_group_number"}
    if task["reference_geometry"] == "relaxed":
        params["relaxation"] = deepcopy(path["parameters"]["relaxation"])
        recipes.append("relaxation")
    path_phonon = "phonon_path_min_frequency_THz" in properties
    mesh_phonon = "phonon_mesh_min_frequency_THz" in properties
    real_dielectric = any(prop.startswith("dielectric_") and "nominal" not in prop for prop in properties)
    ionic_response = bool(properties & {"dielectric_ionic_bec_trace_over3", "dielectric_total_static_trace_over3"})
    nominal = "dielectric_ionic_nominal_trace_over3" in properties
    if path_phonon or mesh_phonon or nominal or ionic_response:
        phonon = deepcopy(path["parameters"]["phonon"])
        if mesh_phonon or (ionic_response and not path_phonon):
            phonon.pop("path_points_parameter")
            phonon.update(q_scope="mesh", mesh=[8, 8, 8], gamma_center=True)
        if path_phonon and mesh_phonon:
            phonon.update(q_scope="path_and_mesh", path_points_parameter=101)
        # Explicit MLFF phonons in a hybrid task do not silently acquire DFT NAC.
        mlff_phonon = any(
            value["property_id"].startswith("phonon_") and value.get("required_fidelity") == "mlff"
            for value in task["measurements"].values()
        )
        if real_dielectric and not mlff_phonon:
            phonon["nac"] = "from_dft_born_and_electronic"
        params["phonon"] = phonon
        recipes.append("phonon")
    if "band_gap_eV" in properties or real_dielectric:
        params["band_gap"] = deepcopy(highk["parameters"]["band_gap"])
        params["dft"] = deepcopy(highk["parameters"]["dft"])
        recipes.append("band_gap")
    if real_dielectric:
        params["dielectric"] = deepcopy(highk["parameters"]["dielectric"])
        if not ionic_response:
            params["dielectric"]["components"] = ["electronic"]
        recipes.append("dielectric")
    if nominal:
        params["nominal_proxy"] = {
            "charge_assignment": "explicit_species_map", "nominal_charges": None,
            "regularization": "none", "frequency_floor_THz": None,
        }
        recipes.append("nominal_proxy")
    if any(prop.startswith("bulk_modulus_") and "eos" not in prop for prop in properties):
        params["elastic"] = {
            "strain_magnitudes": [-0.02, -0.01, 0.01, 0.02],
            "fit": "linear_stress_strain", "internal_relaxation": True,
        }
        recipes.append("elastic")
    if "bulk_modulus_eos_GPa" in properties:
        params["eos"] = {
            "volume_factors": [0.94, 0.97, 1.0, 1.03, 1.06],
            "fit": "birch_murnaghan_3rd_order", "internal_relaxation": True,
        }
        recipes.append("eos")
    result["protocol_id"] = "ctb." + ".".join(sorted(recipes)) + ".candidate.v0_2"
    result["purpose"] = "S1 candidate recipe planning; numerical settings are not convergence evidence"
    result["parameters"] = params
    if physical or task["reference_geometry"] == "relaxed":
        result["required_resolutions"] = [
            "pin backend versions, model or pseudopotential content and numerical parameter sets",
            "verify common reference geometry, stationary quality and input domain",
            "validate numerical convergence and units before scientific release",
            "authorize physical execution resource limits",
        ]
    return validate_protocol(result)
