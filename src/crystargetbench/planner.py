"""Deterministic capability routing and non-executing scientific DAG planning."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from .backends import REGISTRY, capabilities, doctor, select_provider, validate_registry_bindings
from .contracts import (default_deployment, default_policy, default_protocol,
                        validate_deployment, validate_policy, validate_protocol, validate_task)
from .identity import digest
from .scheduling import measurement_schedule


def _inspect_s2_provider(config):
    """Inspect explicitly configured records offline; never imports a model."""
    from .contracts import load_json
    from .assets import preflight
    from .identity import file_digest
    manifest, permit, environment, errors = None, None, None, []
    for key in ("model_manifest", "live_permit", "environment_lock"):
        path = config.get(key)
        if not path:
            errors.append({"category": "blocked_execution" if key == "live_permit" else "blocked_configuration", "reason": f"missing_{key}"})
            continue
        try:
            value = load_json(path)
            if key == "model_manifest":
                manifest = value
            elif key == "live_permit":
                permit = value
            else:
                from .contracts import _validate
                _validate(value, {"type": "object", "additionalProperties": False,
                    "required": ["schema_version", "packages"], "properties": {
                    "schema_version": {"const": "ctb.worker_environment.v1"},
                    "packages": {"type": "object", "additionalProperties": {
                        "type": "object", "additionalProperties": False,
                        "required": ["version", "record_sha256"], "properties": {
                        "version": {"type": "string", "minLength": 1},
                        "record_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}}}}}, "worker_environment")
                if not {"mattersim", "torch", "ase", "numpy", "crystargetbench"} <= set(value["packages"]):
                    raise ValueError("environment lock missing essential distributions")
                environment = {"sha256": file_digest(Path(path).read_bytes()), "record": value}
        except ValueError as exc:
            errors.append({"category": "blocked_execution" if key == "live_permit" else "blocked_configuration", "reason": f"invalid_{key}:{exc}"})
    result = preflight(manifest, permit)
    if manifest is None:
        # Keep the specific missing-file reason rather than the resulting schema noise.
        result["missing_or_invalid"] = [error for error in result["missing_or_invalid"] if error["category"] == "blocked_execution"]
    result["missing_or_invalid"] = errors + result["missing_or_invalid"]
    result["status"] = result["missing_or_invalid"][0]["category"] if result["missing_or_invalid"] else "ready"
    if result.get("model_lock") and config.get("version") != result["model_lock"]["manifest"]["package_version"]:
        result["missing_or_invalid"].append({"category": "blocked_configuration", "reason": "provider_version_manifest_mismatch"})
        result["status"] = "blocked_configuration"
    argv = config.get("worker_argv", [])
    if len(argv) != 3 or argv[1:] != ["-m", "crystargetbench.workers.mattersim"] or not Path(argv[0]).is_absolute():
        result["missing_or_invalid"].append({"category": "blocked_configuration", "reason": "owned_worker_entrypoint_and_absolute_interpreter_required"})
        result["status"] = "blocked_configuration"
    result["environment_lock_sha256"] = environment["sha256"] if environment else None
    return result, permit


RECIPE_CAPABILITIES = {
    "space_group_number": ["space_group"],
    "phonon_path_min_frequency_THz": ["energy", "forces", "stress"],
    "phonon_mesh_min_frequency_THz": ["energy", "forces", "stress"],
    "band_gap_eV": ["band_eigenvalues"],
    "bulk_modulus_eos_GPa": ["energy", "forces"],
    "bulk_modulus_voigt_GPa": ["energy", "forces", "stress"],
    "bulk_modulus_reuss_GPa": ["energy", "forces", "stress"],
    "bulk_modulus_hill_GPa": ["energy", "forces", "stress"],
    "dielectric_electronic_trace_over3": ["electronic_dielectric", "band_eigenvalues"],
    "dielectric_ionic_bec_trace_over3": ["forces", "born_charges", "ionic_dielectric", "band_eigenvalues"],
    "dielectric_total_static_trace_over3": ["forces", "born_charges", "electronic_dielectric", "ionic_dielectric", "band_eigenvalues"],
    "dielectric_ionic_nominal_trace_over3": ["energy", "forces", "stress"],
}


def _available_capabilities(provider: str | None, deployment: dict) -> set[str]:
    if provider is None:
        return set()
    config = deployment["providers"].get(provider)
    if config is None:
        # Missing deployment is a configuration issue, never a method switch.
        return capabilities(provider)
    available = set(REGISTRY[provider]["capabilities"]) & set(config["reported_capabilities"])
    for name, recipe in REGISTRY[provider].get("derived_recipes", {}).items():
        if set(recipe["requires"]) <= available:
            available.add(name)
    return available


def recipe_capabilities(property_id: str, protocol: dict, family: str | None = None) -> list[str]:
    needed = set(RECIPE_CAPABILITIES[property_id])
    if property_id.startswith("phonon_") and protocol["parameters"].get("phonon", {}).get("nac", "none") != "none":
        needed.update(["born_charges", "electronic_dielectric"])
    if property_id == "band_gap_eV" and family == "surrogate":
        needed = {"band_gap_prediction"}
    return sorted(needed)


def _scientific_recipe(property_id: str, parameters: dict) -> dict:
    if property_id == "space_group_number":
        keys = ["symprec_A", "angle_tolerance_deg", "target_symmetrization"]
    elif property_id.startswith("phonon_"):
        keys = ["phonon", "relaxation", "dft"]
    elif property_id == "band_gap_eV":
        keys = ["band_gap", "dft"]
    elif property_id.startswith("bulk_modulus_eos"):
        keys = ["eos", "relaxation", "dft"]
    elif property_id.startswith("bulk_modulus_"):
        keys = ["elastic", "relaxation", "dft"]
    else:
        keys = ["dielectric", "nominal_proxy", "phonon", "band_gap", "dft"]
    return {key: parameters[key] for key in keys if key in parameters}


def resolved_definition(property_id: str, provider: str, protocol: dict) -> str:
    family = REGISTRY[provider]["family"]
    return f"{property_id}@{family}:{provider}:{digest(_scientific_recipe(property_id, protocol['parameters']))[:16]}"


def _validate_recipe_bindings(task, protocol):
    params = protocol["parameters"]
    if task["reference_geometry"] == "relaxed" and "relaxation" not in params:
        raise ValueError("Relaxed reference geometry requires a relaxation protocol")
    for measurement in task["measurements"].values():
        property_id = measurement["property_id"]
        if property_id == "space_group_number":
            required = {"symprec_A", "angle_tolerance_deg", "target_symmetrization"}
        elif property_id.startswith("phonon_"):
            required = {"phonon"}
            scope = "mesh" if "mesh" in property_id else "path"
            if params.get("phonon", {}).get("q_scope") not in (scope, "path_and_mesh"):
                raise ValueError(f"Requested phonon scope {scope} does not match protocol")
        elif property_id == "band_gap_eV":
            required = {"band_gap"}
        elif "nominal" in property_id:
            required = {"nominal_proxy", "phonon"}
        elif property_id.startswith("dielectric_"):
            required = {"dielectric", "band_gap"}
            if "ionic" in property_id or "total" in property_id:
                required.add("phonon")
            components = set(params.get("dielectric", {}).get("components", []))
            needed = {"electronic", "ionic"} if "total" in property_id else ({"ionic"} if "ionic" in property_id else {"electronic"})
            if not needed <= components:
                raise ValueError("Requested dielectric components do not match protocol")
        else:
            required = {"eos" if "eos" in property_id else "elastic"}
        if not required <= params.keys():
            raise ValueError(f"Missing recipe parameters for {property_id}: {sorted(required - params.keys())}")


def classify_domain(item: dict, supported_elements: list[str] | None = None) -> dict:
    """Domain rejection keeps the fixed route and the original submission item."""
    geometry = item.get("input_geometry") or {}
    if item.get("status") in {"invalid", "parse_error", "unsupported"}:
        return {"item_id": item.get("item_id"), "status": item["status"], "implicit_itemwise_switch": False}
    occupancy = geometry.get("occupancy", geometry.get("occupancies", []))
    if any(value != 1 for value in occupancy):
        status, reason = "unsupported", "ordered_full_occupancy_required"
    elif supported_elements is not None and not set(geometry.get("species", [])) <= set(supported_elements):
        status, reason = "unsupported", "outside_declared_elements"
    else:
        status, reason = "eligible_for_preflight", None
    return {"item_id": item.get("item_id"), "status": status, "reason": reason,
            "implicit_itemwise_switch": False}


def _determinant(matrix):
    a, b, c = matrix
    return abs(a[0] * (b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0]))


def _resources(task, protocol, deployment, routes, submission):
    execution = deployment["execution"]
    caps = {key: execution.get(key) for key in (
        "max_dft_jobs", "max_concurrent_jobs", "per_job_wall_seconds", "per_job_cpu_cores",
        "max_scratch_gb", "max_supercell_atoms")}
    has_dft = any(route["family"] == "dft" for route in routes.values())
    has_phonon = any(m["property_id"].startswith("phonon_") or "ionic" in m["property_id"]
                     or "total_static" in m["property_id"] for m in task["measurements"].values())
    dft_phonon = [alias for alias, value in task["measurements"].items()
                  if value["property_id"].startswith("phonon_") and routes.get(alias, {}).get("family") == "dft"]
    dft_dielectric = [alias for alias, value in task["measurements"].items()
                      if value["property_id"].startswith("dielectric_") and routes.get(alias, {}).get("family") == "dft"]
    force_groups = len(dft_phonon)
    for alias in dft_dielectric:
        prop = task["measurements"][alias]["property_id"]
        if ("ionic" in prop or "total_static" in prop) and not any(routes[a]["provider"] == routes[alias]["provider"] for a in dft_phonon):
            force_groups += 1
    multiplier = _determinant(protocol["parameters"].get("phonon", {}).get("supercell_matrix", [[1,0,0],[0,1,0],[0,0,1]]))
    item_estimates, total = [], 0
    unknown = submission is None
    for item in submission or []:
        atoms = item.get("atom_count")
        if atoms is None:
            atoms = len((item.get("input_geometry") or {}).get("species", [])) or None
        supercell = int(atoms * multiplier) if atoms is not None and has_phonon else atoms
        # An intentionally conservative bound before symmetry reduction. Actual
        # displacement expansion must reserve jobs against this hard budget.
        displacement_jobs = 6 * supercell if has_phonon and supercell is not None else (0 if not has_phonon else None)
        overhead = (1 if task["reference_geometry"] == "relaxed" else 0) + 1
        overhead += 2 if any("dielectric" in m["property_id"] for m in task["measurements"].values()) else 0
        # Count every expanded DFT force group. Extra response calculations are
        # retained in this bound even when an eventual adapter can share them.
        overhead += len(dft_phonon) if protocol["parameters"].get("phonon", {}).get("nac", "none") != "none" else 0
        overhead += max(0, 2 * len(dft_dielectric) - 2)
        for alias, value in task["measurements"].items():
            if routes.get(alias, {}).get("family") != "dft":
                continue
            if "bulk_modulus_eos" in value["property_id"]:
                overhead += len(protocol["parameters"].get("eos", {}).get("volume_factors", []))
            elif "bulk_modulus" in value["property_id"]:
                overhead += 6 * len(protocol["parameters"].get("elastic", {}).get("strain_magnitudes", []))
        jobs = force_groups * displacement_jobs + overhead if has_dft and displacement_jobs is not None else (0 if not has_dft else None)
        if jobs is None:
            unknown = True
        else:
            total += jobs
        item_estimates.append({"item_id": item.get("item_id"), "atom_count": atoms,
            "supercell_atoms_upper_bound": supercell, "displacement_jobs_upper_bound": displacement_jobs,
            "dft_force_groups": force_groups, "dft_jobs_upper_bound": jobs})
    estimate = None if unknown and has_dft else total
    violations = []
    if has_phonon and any(e["supercell_atoms_upper_bound"] is not None and e["supercell_atoms_upper_bound"] > caps["max_supercell_atoms"] for e in item_estimates):
        violations.append("max_supercell_atoms_exceeded")
    if has_dft and estimate is not None and estimate > caps["max_dft_jobs"]:
        violations.append("max_dft_jobs_exceeded")
    if has_dft and caps["max_dft_jobs"] == 0:
        violations.append("dft_job_budget_is_zero")
    for key in ("per_job_cpu_cores", "max_scratch_gb"):
        if has_dft and caps[key] is None:
            violations.append(f"missing_hard_limit:{key}")
    return {"hard_limits": caps, "estimate_kind": "conservative_pre_symmetry_upper_bound",
            "estimate_scope": "initial DAG expansion; relaxation restarts/retries require additional runtime reservation",
            "dft_jobs_upper_bound": estimate, "wall_seconds_estimate": None,
            "cpu_seconds_estimate": None, "scratch_gb_estimate": None,
            "per_item": item_estimates, "violations": sorted(set(violations)),
            "runtime_reservation_required": has_dft,
            "unknown_cost_policy": "block_execution_until_expanded_and_within_hard_limits",
            "submitted_jobs": 0}


def _build_dag(task, protocol, routes, policy):
    nodes = []
    input_ref = {"kind": "submission_geometry", "field": "input_geometry.geometry_id"}
    final_ref = {"kind": "node_output", "node_id": "freeze_geometry", "output": "reference_geometry_id"}
    def add(node_id, kind, dependencies=(), route=None, conditions=(), **extras):
        selected = routes.get(route, {}) if isinstance(route, str) else (route or {})
        if not conditions:
            conditions = [{"gate_node": dependency, "required_verdict": "pass",
                           "otherwise": "skip_with_unknown_measurement"}
                          for dependency in dependencies
                          if dependency == "gap_gate" or dependency.startswith("phonon_gate:")]
        nodes.append({"node_id": node_id, "kind": kind, "scope": "per_submission_item",
            "dependencies": list(dict.fromkeys(dependencies)), "provider": selected.get("provider"),
            "family": selected.get("family", "native"), "geometry_ref": deepcopy(final_ref),
            "required_capabilities": selected.get("required_capabilities", []),
            "conditions": list(conditions), "execution_status": "not_executed", **extras})
        return node_id
    add("validate_input", "input_validation", geometry_ref=input_ref,
        input_domain=protocol["input_domain"])
    previous = "validate_input"
    if policy["warm_start"] == "mlff_pre_relax_then_dft":
        previous = add("warm_start", "mlff_warm_start", [previous], routes.get("warm_start"),
                       geometry_ref=input_ref, final_physics_evidence=False)
    if task["reference_geometry"] == "relaxed":
        previous = add("tight_relax", "tight_relaxation", [previous], "geometry",
            geometry_ref=input_ref if previous == "validate_input" else {"kind": "node_output", "node_id": previous, "output": "geometry_id"},
            numerical_parameters=deepcopy(protocol["parameters"].get("relaxation", {})),
            acceptance=["completed", "force_converged", "stress_converged"], imposed_target_symmetry=False)
    add("freeze_geometry", "freeze_reference_geometry", [previous], geometry_ref={"kind": "node_output", "node_id": previous, "output": "geometry_id"} if previous != "validate_input" else input_ref,
        identity_rule="bind_actual_geometry_digest_after_completion", preserve_input_snapshot=True)
    gap_aliases = [a for a,m in task["measurements"].items() if m["property_id"] == "band_gap_eV"]
    phonon_aliases = [a for a,m in task["measurements"].items() if m["property_id"].startswith("phonon_")]
    dielectric_aliases = [a for a,m in task["measurements"].items() if m["property_id"].startswith("dielectric_")]
    has_formal_dielectric = any("nominal" not in task["measurements"][a]["property_id"] for a in dielectric_aliases)
    gates, measurements = [], []
    if gap_aliases or has_formal_dielectric:
        gap_route = routes[gap_aliases[0]] if gap_aliases else routes[dielectric_aliases[0]]
        add("scf_gap", "scf_kmesh_gap" if gap_route["family"] == "dft" else "explicit_gap_surrogate",
            ["freeze_geometry"], gap_route, numerical_parameters=protocol["parameters"].get("band_gap", {}),
            outputs=gap_aliases or ["internal_insulating_gap"])
        measurements.append("scf_gap")
        constraints = [c for c in task.get("constraints", []) if c["measurement"] in gap_aliases]
        if constraints or has_formal_dielectric:
            add("gap_gate", "conditional_gap_gate", ["scf_gap"], constraints=constraints,
                internal_requirements=["positive_insulating_gap"] if has_formal_dielectric else [],
                on_accepted_fail="joint_fail_skip_expensive_downstream",
                on_unknown="unknown_skip_dependent_dielectric", admissibility_required=True)
            gates.append("gap_gate")
    phonon_gate = None
    for alias in phonon_aliases:
        route = routes[alias]
        # Local S2 measurements are independent after their qualified reference.
        # Phonon verdicts gate response recipes, not another sampling scope.
        local = all(r['family'] in {'native','mlff'} for r in routes.values())
        dependencies = ["freeze_geometry", *[g for g in gates if not local or not g.startswith('phonon_gate:')]]
        if route.get("requires_stationary_reference"):
            stationarity = add(f"stationarity:{alias}", "same_geometry_stationarity_check", dependencies, alias,
                residual_force_and_stress_required=True, on_failure="measurement_unknown",
                allow_reference_geometry_relaxation=False)
            dependencies.append(stationarity)
        displacements = add(f"displacements:{alias}", "finite_displacements", dependencies,
            mapping_required=["site_mapping", "primitive_mapping", "supercell_basis", "atom_order"],
            parameters=protocol["parameters"].get("phonon", {}),
            derived_geometry={"kind": "node_output", "node_id": "freeze_geometry", "transformation": "supercell_and_displacement"})
        force_node = add(f"forces:{alias}", "displacement_forces", [displacements], alias,
            dynamic_expansion="one_force_job_per_actual_displacement", reserve_budget_before_launch=True,
            geometry_ref={"kind": "mapped_node_output", "node_id": displacements, "output": "displaced_geometry_ids"})
        dependencies = [force_node]
        nac = protocol["parameters"].get("phonon", {}).get("nac", "none") != "none"
        if nac:
            nac_node = add(f"nac:{alias}", "born_charges_and_electronic_response", ["freeze_geometry", *gates], alias,
                share_with_dielectric_if_same_provider=True, required_capabilities=["born_charges", "electronic_dielectric"])
            dependencies.append(nac_node)
        measure_node = add(f"phonon:{alias}", "phonon_sampling", dependencies, alias,
            measurement_alias=alias, q_scope="mesh" if "mesh" in task["measurements"][alias]["property_id"] else "path",
            retain_signed_frequencies=True, quality_checks=["finite_nonempty", "mapping", "units", "stationarity"])
        measurements.append(measure_node)
        constraints = [c for c in task.get("constraints", []) if c["measurement"] == alias]
        if constraints:
            phonon_gate = add(f"phonon_gate:{alias}", "conditional_phonon_gate", [measure_node],
                constraints=constraints, admissibility_required=True,
                on_accepted_fail="joint_fail_skip_dielectric", on_unknown="unknown_skip_dependent_dielectric")
            gates.append(phonon_gate)
    for alias in dielectric_aliases:
        property_id = task["measurements"][alias]["property_id"]
        route, dependencies = routes[alias], ["freeze_geometry", *gates]
        components = []
        if "nominal" in property_id:
            components.append(add(f"nominal:{alias}", "explicit_nominal_ionic_proxy", dependencies, alias,
                exploratory_only=True, requires_explicit_nominal_charge_and_regularization_protocol=True))
        else:
            if "electronic" in property_id or "total" in property_id:
                shared = [n["node_id"] for n in nodes if n["kind"] == "born_charges_and_electronic_response" and n["provider"] == route["provider"]]
                components.append(add(f"electronic:{alias}", "electronic_dielectric", [*dependencies, *shared], alias,
                    reuse_response_node=shared[0] if shared else None))
            if "ionic" in property_id or "total" in property_id:
                # A hybrid MLFF stability diagnostic is never reused as DFT IFC.
                compatible = [f"forces:{a}" for a in phonon_aliases if routes[a]["provider"] == route["provider"]]
                if compatible:
                    ifc = compatible[0]
                else:
                    displaced = add(f"ionic_displacements:{alias}", "finite_displacements", dependencies,
                        mapping_required=["site_mapping", "primitive_mapping", "supercell_basis", "atom_order"],
                        derived_geometry={"kind": "node_output", "node_id": "freeze_geometry", "transformation": "supercell_and_displacement"})
                    ifc = add(f"ionic_forces:{alias}", "displacement_forces", [displaced], alias,
                        dynamic_expansion="one_force_job_per_actual_displacement", reserve_budget_before_launch=True,
                        geometry_ref={"kind": "mapped_node_output", "node_id": displaced, "output": "displaced_geometry_ids"})
                components.append(add(f"ionic:{alias}", "ionic_dielectric", [*dependencies, ifc], alias,
                    recipe="gamma_ifc_born_effective_charges", required_capabilities=["forces", "born_charges"],
                    requires_unit_mapping_validation=True, reject_negative_optical_modes=True,
                    silent_frequency_floor=False, missing_component_policy="unknown"))
        measurements.append(add(f"tensor:{alias}", "tensor_quality_and_scalarization", components, alias,
            measurement_alias=alias, scalarization="trace_over_3", retain=["tensor", "principal_values", "anisotropy", "soft_mode_diagnostics"],
            quality_checks=["finite", "symmetric", "same_geometry", "compatible_units", "component_completeness"],
            forbid_missing_component_substitution=True))
    for alias, measurement in task["measurements"].items():
        if alias in gap_aliases + phonon_aliases + dielectric_aliases:
            continue
        property_id = measurement["property_id"]
        kind = "native_space_group" if property_id == "space_group_number" else ("eos_volume_fit" if "eos" in property_id else "elastic_strain_stress_fit")
        measurements.append(add(f"measure:{alias}", kind, ["freeze_geometry"], alias,
            measurement_alias=alias, property_id=property_id, forbid_recipe_substitution=True))
    add("joint_assessment", "three_valued_assessment", measurements, full_submission_denominator=True)
    add("metrics", "submission_metrics", ["joint_assessment"], scope="submission",
        metric_mode=task["mode"], benchmark_eligible=False)
    return nodes


def plan(task: dict, policy: dict | None = None, protocol: dict | None = None,
         deployment: dict | None = None, submission: list[dict] | None = None) -> dict:
    """Validate, route and freeze a complete plan; this function never executes it."""
    if deployment and deployment.get('schema_version') == 'ctb.dft.site.v1':
        if policy and policy.get('mode') in {'mlff','hybrid'}:raise ValueError('external site requires auto or dft policy')
        if policy and policy.get('preferred_dft') and policy['preferred_dft']!=deployment['adapter']:
            raise ValueError('explicit policy/site adapter conflict')
        from .dft.exchange import plan_external
        return plan_external(task,deployment,protocol,submission)
    task = deepcopy(task)
    validate_task(task)
    schedule=measurement_schedule(task)
    task['measurements']={alias:task['measurements'][alias] for alias in schedule['measurement_order']}
    deployment = deepcopy(deployment) if deployment is not None else default_deployment()
    validate_deployment(deployment)
    validate_registry_bindings(deployment)
    if policy is None:
        policy = default_policy()
        policy["preferred_mlff"] = deployment.get("default_mlff_provider")
        if deployment.get("default_dft_provider") is not None:
            policy["preferred_dft"] = deployment["default_dft_provider"]
    else:
        policy = deepcopy(policy)
    validate_policy(policy)
    allowed_overrides = set(task["measurements"]) | ({"geometry"} if task["reference_geometry"] == "relaxed" else set())
    if set(policy["overrides"]) - allowed_overrides:
        raise ValueError("Hybrid overrides contain unknown or unused measurement/geometry aliases")
    protocol_was_explicit = protocol is not None
    protocol = deepcopy(protocol) if protocol_was_explicit else default_protocol(task)
    surrogate_gap_aliases = [alias for alias, measurement in task["measurements"].items()
                            if measurement["property_id"] == "band_gap_eV"
                            and policy.get("overrides", {}).get(alias, "").startswith("surrogate:")]
    if surrogate_gap_aliases and not protocol_was_explicit:
        protocol["parameters"]["band_gap"] = {
            "definition": "explicit_surrogate_prediction", "functional": "not_applicable_surrogate",
            "mesh_policy_id": "not_applicable_surrogate", "spin_u_soc_policy_id": "not_applicable_surrogate"}
    validate_protocol(protocol)
    if protocol["reference_geometry"] != task["reference_geometry"]:
        raise ValueError("Task and protocol reference_geometry mismatch")
    _validate_recipe_bindings(task, protocol)
    mode, parameters = policy["mode"], protocol["parameters"]
    reasons, routes, promotion = [], {}, None
    blocked_capability = False
    mlff, mlff_error = select_provider("mlff", policy.get("preferred_mlff", deployment.get("default_mlff_provider")), deployment)
    dft, dft_error = select_provider("dft", policy.get("preferred_dft", deployment.get("default_dft_provider")), deployment)
    physical = {a: m for a,m in task["measurements"].items() if m["property_id"] != "space_group_number"}
    required = set().union(*(set(recipe_capabilities(m["property_id"], protocol)) for m in physical.values())) if physical else set()
    if task["reference_geometry"] == "relaxed":
        required.update(["energy", "forces", "stress"])
    requires_dft = any(m.get("required_fidelity") == "dft" for m in physical.values())
    group_family = "native"
    if required:
        if mode == "dft":
            group_family = "dft"
        elif mode == "mlff":
            group_family = "mlff"
        elif mode == "auto":
            missing = required - _available_capabilities(mlff, deployment) if mlff else set()
            if mlff is None:
                # A misspelled/ambiguous provider is not missing physics capability.
                group_family = "mlff"
            elif missing or requires_dft:
                group_family, promotion = "dft", "required_fidelity" if requires_dft and not missing else "missing_required_capability"
                if not policy["promote_on_missing_capability"]:
                    blocked_capability = True
                    reasons.append({"code": "promotion_disabled", "required_capabilities": sorted(required)})
            else:
                group_family = "mlff"
        else:
            group_family = "mixed"

    from .s2_protocol import default_s2_protocol, supports_s2_task, is_complete_s2_protocol, RECIPE_VERSIONS
    if not protocol_was_explicit and group_family == "mlff" and mode in {"auto", "mlff"} and supports_s2_task(task):
        protocol = default_s2_protocol(task, protocol)
        parameters = protocol["parameters"]

    def bind(alias, measurement=None):
        nonlocal blocked_capability
        native = measurement and measurement["property_id"] == "space_group_number"
        if native:
            family, provider = "native", "symmetry"
            if alias in policy["overrides"] and policy["overrides"][alias] != "native:symmetry":
                raise ValueError("Space-group identification uses the registered native adapter")
        elif mode == "hybrid":
            if alias not in policy["overrides"]:
                raise ValueError(f"Explicit hybrid route required: {alias}")
            family, provider = policy["overrides"][alias].split(":", 1)
            if provider not in REGISTRY or REGISTRY[provider]["family"] != family:
                raise ValueError(f"Invalid hybrid provider family: {alias}")
        else:
            family = group_family
            provider = mlff if family == "mlff" else dft
        if provider is None:
            reasons.append({"code": "explicit_provider_choice_required", "alias": alias, "family": family})
            routes[alias] = {"provider": None, "family": family, "physics_fidelity": family,
                "definition_id": None, "required_capabilities": recipe_capabilities(measurement["property_id"], protocol, family) if measurement else ["energy","forces","stress"],
                "reference_geometry": {"kind":"node_output","node_id":"freeze_geometry","output":"reference_geometry_id"}}
            return
        if family == "surrogate" and not policy["allow_surrogates"]:
            raise ValueError("Surrogates require explicit allow_surrogates")
        if measurement and measurement["property_id"] == "band_gap_eV":
            explicit_proxy_recipe = parameters["band_gap"]["definition"] == "explicit_surrogate_prediction"
            if (family == "surrogate") != explicit_proxy_recipe:
                raise ValueError("Gap recipe definition must match its explicit physical or surrogate route")
        if measurement and "nominal" in measurement["property_id"] and not (mode == "hybrid" and policy["allow_surrogates"]):
            raise ValueError("Nominal ionic proxy requires an explicit exploratory hybrid policy")
        caps = recipe_capabilities(measurement["property_id"], protocol, family) if measurement else ["energy", "forces", "stress"]
        mismatch = set(caps) - _available_capabilities(provider, deployment)
        fidelity = measurement.get("required_fidelity", "protocol_default") if measurement else "protocol_default"
        if fidelity != "protocol_default" and fidelity != family:
            if mode == "hybrid":
                raise ValueError(f"Task fidelity cannot be overridden: {alias}")
            mismatch.add(f"required_fidelity:{fidelity}")
        if mismatch:
            blocked_capability = True
            reasons.append({"code": "missing_required_capability", "alias": alias, "provider": provider, "missing": sorted(mismatch)})
        definition = resolved_definition(measurement["property_id"], provider, protocol) if measurement else f"geometry.relaxed@{family}:{provider}"
        if measurement and measurement.get("definition_id", definition) != definition:
            raise ValueError(f"Property recipe/definition pin mismatch: {alias}")
        routes[alias] = {"provider": provider, "family": family, "definition_id": definition,
                         "required_capabilities": caps, "physics_fidelity": family,
                         "reference_geometry": {"kind": "node_output", "node_id": "freeze_geometry", "output": "reference_geometry_id"}}
    for alias, measurement in task["measurements"].items():
        bind(alias, measurement)
    if task["reference_geometry"] == "relaxed":
        bind("geometry")
    for alias, measurement in task["measurements"].items():
        if measurement["property_id"].startswith("phonon_") and alias in routes:
            geometry_route = routes.get("geometry", {})
            routes[alias]["requires_stationary_reference"] = (
                task["reference_geometry"] == "input"
                or routes[alias]["family"] != geometry_route.get("family")
                or routes[alias]["provider"] != geometry_route.get("provider"))
    if policy["warm_start"] == "mlff_pre_relax_then_dft":
        if routes.get("geometry", {}).get("family") != "dft":
            raise ValueError("Warm start requires a final DFT relaxation")
        if mlff:
            routes["warm_start"] = {"provider": mlff, "family": "mlff", "definition_id": "warm_start.mlff",
                                   "required_capabilities": ["energy", "forces", "stress"], "final_physics": False}
    states = doctor(deployment)["providers"]
    config_issues, assets_issues = [], []
    s2_inspections, s2_permits = {}, {}
    selected = sorted({r["provider"] for r in routes.values() if r["provider"] is not None})
    for provider in selected:
        state, config = states[provider], deployment["providers"].get(provider, {})
        if not config.get("configured"):
            config_issues.append({"code": "provider_not_configured", "provider": provider})
            continue
        family = REGISTRY[provider]["family"]
        if family == "native":
            missing_dependencies = [name for name, dependency in state["native_dependencies"].items()
                                    if not dependency["installed"]]
            if missing_dependencies:
                config_issues.append({"code": "native_dependency_not_installed", "provider": provider,
                                      "missing_dependencies": missing_dependencies})
            if config.get("version") and config["version"] != state["installed_version"]:
                config_issues.append({"code": "native_version_pin_mismatch", "provider": provider})
            continue
        for key in ("version", "parameter_set_id"):
            if not config.get(key) or str(config[key]).lower() == "latest":
                config_issues.append({"code": f"unresolved_{key}", "provider": provider})
        if provider == "mattersim" and config.get("model_manifest"):
            inspection, permit = _inspect_s2_provider(config)
            s2_inspections[provider], s2_permits[provider] = inspection, permit
            for error in inspection["missing_or_invalid"]:
                if error["category"] == "blocked_configuration":
                    config_issues.append({"code": "s2_configuration", "provider": provider, "detail": error["reason"]})
                elif error["category"] == "blocked_asset":
                    assets_issues.append({"code": "missing_asset", "provider": provider, "detail": error["reason"]})
            continue
        asset_name = "pseudopotential_manifest" if family == "dft" else "checkpoint"
        asset = config.get("assets", {}).get(asset_name, {})
        if state["asset_status"].get(asset_name, {}).get("digest_verified") and asset.get("license_status") in {"site_managed_unresolved", "unresolved", "missing", "unreviewed"}:
            config_issues.append({"code": "unresolved_asset_license", "provider": provider, "asset": asset_name})
        if family == "dft" and not asset.get("sha256"):
            config_issues.append({"code": "unresolved_pseudopotential_hash", "provider": provider})
        elif not state["asset_status"].get(asset_name, {}).get("digest_verified"):
            assets_issues.append({"code": "missing_asset", "provider": provider, "asset": asset_name})
        if family == "dft":
            dft_params = parameters.get("dft", {})
            gap_params = parameters.get("band_gap", {})
            for key in ("parameter_set_id", "pseudopotential_manifest_sha256", "cutoff_and_convergence_profile"):
                if not dft_params.get(key):
                    config_issues.append({"code": f"unresolved_dft_protocol:{key}", "provider": provider})
            for key in ("mesh_policy_id", "spin_u_soc_policy_id"):
                if not gap_params.get(key):
                    config_issues.append({"code": f"unresolved_electronic_protocol:{key}", "provider": provider})
            if dft_params.get("pseudopotential_manifest_sha256") and dft_params["pseudopotential_manifest_sha256"] != asset.get("sha256"):
                config_issues.append({"code": "pseudopotential_protocol_deployment_mismatch", "provider": provider})
            if dft_params.get("parameter_set_id") and dft_params["parameter_set_id"] != config.get("parameter_set_id"):
                config_issues.append({"code": "parameter_set_protocol_deployment_mismatch", "provider": provider})
    resources = _resources(task, protocol, deployment, routes, submission)
    families = {r["family"] for a,r in routes.items() if a != "warm_start" and r["family"] != "native"}
    s2_supported = families == {"mlff"} and "warm_start" not in routes and all(
        route["family"] == "native" or route["provider"] == "mattersim" for route in routes.values()) and is_complete_s2_protocol(task, protocol)
    fidelity = "mixed" if len(families) > 1 else next(iter(families), "native")
    unauthorized = [family for family in sorted(families | ({"mlff"} if "warm_start" in routes else set()))
                    if family != "native" and not deployment["execution"].get(f"{family}_authorized", False)]
    status = "ready"
    if blocked_capability:
        status = "blocked_capability"
    elif len(routes) < len(task["measurements"]) + (task["reference_geometry"] == "relaxed") or config_issues or any(r["provider"] is None for r in routes.values()):
        status = "blocked_configuration"
    elif assets_issues:
        status = "missing_asset"
    elif unauthorized:
        status = "awaiting_authorization"
    elif resources["violations"]:
        status = "blocked_resource"
    elif families and not s2_supported:
        status = "blocked_implementation"
    reasons.extend(config_issues + assets_issues)
    if unauthorized:
        reasons.append({"code": "explicit_physics_budget_authorization_required", "families": unauthorized})
    reasons.extend({"code": code} for code in resources["violations"])
    if families and not s2_supported:
        reasons.append({"code": "requested_execution_recipe_not_implemented", "workers_started": 0})
    lock = {"schema_version": "ctb.protocol_lock.v1", "protocol": protocol,"scheduling":schedule,
        "recipe_implementation_version": "ctb.s2.recipes.v1" if s2_supported else "ctb.s1.recipes.v1",
        "parser_lock": {
            **{key: states["symmetry"]["parser_dependency"][key] for key in ("distribution", "installed_version")},
            "dependencies": {name: dependency["installed_version"] for name, dependency in states["symmetry"]["native_dependencies"].items()}},
        "routes": routes, "warm_start": policy["warm_start"],
        "provider_locks": {name: {
            "adapter_id": REGISTRY[name]["adapter_id"], "family": REGISTRY[name]["family"],
            "version": states[name]["installed_version"] if REGISTRY[name]["family"] == "native" else states[name]["configured_version"],
            "parameter_set_id": deployment["providers"].get(name, {}).get("parameter_set_id"),
            "asset_digests": {k: v.get("sha256") for k,v in deployment["providers"].get(name, {}).get("assets", {}).items()},
            "registry_capabilities": sorted(capabilities(name)),
        } for name in selected}, "physics_fidelity": fidelity,
        "benchmark_eligible": False, "scientific_validation": "not_scientifically_validated",
        "lock_resolved": not bool(config_issues or assets_issues or any(r["provider"] is None for r in routes.values())),
        "unresolved_fields": deepcopy(config_issues + assets_issues)}
    if s2_supported:
        lock["recipe_versions"] = deepcopy(RECIPE_VERSIONS)
        if any(value["property_id"].startswith("phonon_") for value in task["measurements"].values()):
            from .recipes.phonon import PHONOPY_VERSION, SEEKPATH_VERSION
            lock["numerical_adapter_package_pins"] = {"phonopy": PHONOPY_VERSION, "seekpath": SEEKPATH_VERSION}
        for provider in selected:
            inspection = s2_inspections.get(provider)
            if inspection and inspection.get("model_lock"):
                model = inspection["model_lock"]
                lock["provider_locks"][provider].update(
                    model_lock_digest=model["model_lock_digest"],
                    model_identity={key: value for key, value in model["manifest"].items() if key != "checkpoint_path"},
                    environment_lock_sha256=inspection["environment_lock_sha256"])
                lock["provider_locks"][provider]["asset_digests"]["checkpoint"] = model["manifest"]["checkpoint_sha256"]
    lock["protocol_digest"] = digest({key: value for key, value in lock.items()
                                      if key not in ("lock_resolved", "unresolved_fields")})
    if s2_supported:
        from .assets import AssetError, validate_permit
        from .structures import submission_digest
        for provider in selected:
            if REGISTRY[provider]["family"] != "mlff":
                continue
            if provider not in s2_inspections:
                s2_inspections[provider] = {"status": "blocked_configuration", "model_lock": None,
                    "missing_or_invalid": [{"category": "blocked_configuration", "reason": "missing_model_manifest"}]}
            inspection = s2_inspections[provider]
            late_errors = list(inspection["missing_or_invalid"])
            if not submission:
                late_errors.append({"category": "blocked_execution", "reason": "exact_input_batch_required_for_authorization"})
            elif inspection.get("model_lock") and not late_errors:
                bindings = {"worker_environment_lock_sha256": inspection["environment_lock_sha256"],
                    "deployment_digest": digest(deployment), "input_manifest_sha256": submission_digest(submission),
                    "protocol_digest": lock["protocol_digest"]}
                try:
                    permit = validate_permit(s2_permits[provider], inspection["model_lock"], bindings)
                    required_operations = {"energy_forces_stress"}
                    if task["reference_geometry"] == "relaxed":
                        required_operations.add("relaxation")
                    properties = {value["property_id"] for value in task["measurements"].values()}
                    if any(prop.startswith("phonon_") for prop in properties):
                        required_operations.add("finite_displacement_phonon")
                    if "bulk_modulus_eos_GPa" in properties:
                        required_operations.update(["bulk_eos", "relaxation"])
                    if not required_operations <= set(permit["allowed_operations"]):
                        raise AssetError("blocked_execution", "required scientific operation not authorized")
                    unique = {item["input_geometry"]["geometry_id"] for item in submission if item.get("input_geometry")}
                    if len(unique) > permit["limits"]["max_unique_structures"]:
                        raise AssetError("blocked_execution", "submitted batch exceeds authorized structure count")
                    if permit["limits"]["max_concurrent_workers"] != 1:
                        raise AssetError("blocked_configuration", "S2 serial execution requires exactly one authorized worker")
                    if any(item.get("atom_count", 0) and item["atom_count"] > permit["limits"]["max_supercell_atoms"] for item in submission):
                        raise AssetError("blocked_execution", "input exceeds owner-authorized atom limit")
                    if any(prop.startswith("phonon_") for prop in properties) and permit["limits"]["max_displaced_structures_total"] == 0:
                        raise AssetError("blocked_execution", "finite-displacement operation has zero displacement budget")
                except AssetError as exc:
                    late_errors.append({"category": exc.category, "reason": str(exc)})
            if late_errors:
                reasons.extend({"code": "s2_preflight", "provider": provider, **error} for error in late_errors)
                if status == "ready":
                    status = late_errors[0]["category"]
                lock["lock_resolved"] = False
                lock["unresolved_fields"].extend(late_errors)
            inspection["missing_or_invalid"] = late_errors
            inspection["status"] = late_errors[0]["category"] if late_errors else "ready"
    comparison = digest({"task_digest": digest(task), "protocol_digest": lock["protocol_digest"],
        "selection_policy": policy, "quality_version": "ctb.quality.v1", "metrics_version": "ctb.metrics.v1"})
    nodes = _build_dag(task, protocol, routes, policy) if len(routes) >= len(task["measurements"]) else []
    for node in nodes:
        node["dependency_artifact_refs"] = [
            {"kind": "node_output", "node_id": upstream, "output": "artifact_digest", "resolved_digest": None}
            for upstream in node["dependencies"]]
        node["plan_node_digest"] = digest({"node": node, "protocol_digest": lock["protocol_digest"]})
        node["computation_digest"] = None
        node["computation_digest_resolution"] = "bind_real_geometry_and_dependency_artifacts_before_cache_lookup"
    return {"schema_version": "ctb.plan.v1", "task_id": task["task_id"], "requested_backend": mode,
        "status": status, "reasons": reasons, "routes": routes, "nodes": nodes,
        "promotion_reason": promotion, "physics_fidelity": fidelity,
        "reference_geometry_backend": routes.get("geometry", {"family": "native"})["family"],
        "protocol_lock": lock, "protocol_digest": lock["protocol_digest"],
        "comparability_digest": comparison, "resource_preflight": resources,
        "provider_states": {name: states[name] for name in selected},
        "s2_execution_supported": s2_supported, "live_preflight": s2_inspections,
        "execution_preflight_required": s2_supported,
        "item_preflight": [classify_domain(item, REGISTRY.get(mlff, {}).get("supported_elements") if group_family == "mlff" else None) for item in submission or []],
        "execution_kind": "planning_only", "physics_calls": 0, "submitted_jobs": 0,
        "physical_failures_created": 0, "benchmark_eligible": False}
