"""Static, non-executing backend registry and honest availability inspection.

Registry entries describe planning interfaces.  Registration never means that a
physical adapter, installation, live calculation, or scientific validation exists.
Deployment data can restrict an entry; it cannot grant capabilities or change its
family.  No optional worker module is imported by this module.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import metadata
from pathlib import Path


def _entry(family, adapter, capabilities, distribution=None, implemented=False, **extra):
    return dict(family=family, adapter_id=adapter, capabilities=capabilities,
                distribution=distribution, registered=True,
                interface_implemented=implemented, parser_tested=False,
                live_tested=False, physics_validated=False, **extra)


REGISTRY = {
    "symmetry": _entry("native", "ctb.native.spglib", ["space_group"], "spglib", True),
    "mattersim": _entry("mlff", "ctb.ase.mattersim", ["energy", "forces", "stress"], "mattersim", True,
        implemented_recipes=["ase.fire_frechet.relaxation.v1", "ctb.phonopy.finite_displacement.v1", "ctb.phonopy.sampling.v1", "ase.eos.fixed_shape_internal_relax.bm3.v1"],
        supported_package_version="1.1.2"),
    "vasp_atomate2": _entry("dft", "ctb.dft.vasp_atomate2",
        ["energy", "forces", "stress", "band_eigenvalues", "electronic_dielectric", "born_charges", "ionic_dielectric"], "atomate2"),
    "qe": _entry("dft", "ctb.dft.qe",
        ["energy", "forces", "stress", "band_eigenvalues", "electronic_dielectric", "born_charges"],
        derived_recipes={"ionic_dielectric": {
            "requires": ["forces", "born_charges", "electronic_dielectric"],
            "recipe_id": "ctb.gamma_ifc_bec.ionic.candidate.v1",
            "implemented": False, "unit_mapping_validated": False}}),
    "alignn2": _entry("surrogate", "ctb.surrogate.alignn2", ["band_gap_prediction"], "alignn"),
    "synthetic": _entry("synthetic", "ctb.tests.synthetic", ["test_observation"], implemented=True),
}


def registry() -> dict:
    """Return copies so deployment and callers cannot mutate trusted metadata."""
    from .dft.registry import discover
    result = deepcopy(REGISTRY)
    for name, manifest in discover().items():
        if name in result:
            raise ValueError(f"Installed adapter collides with reserved provider: {name}")
        caps = [x['quantity'] for x in manifest['capabilities'] if x['implementation']=='implemented']
        if 'eigenvalues' in caps and 'occupations' in caps:
            caps.append('band_eigenvalues')
        result[name] = _entry(manifest['family'], manifest['adapter_id'], caps,
            manifest['distribution'], True, external_adapter=True, manifest=manifest)
    return result


def capabilities(provider: str) -> set[str]:
    """Capabilities include explicitly registered candidate derived recipes."""
    entry = REGISTRY.get(provider, {})
    result = set(entry.get("capabilities", []))
    for name, recipe in entry.get("derived_recipes", {}).items():
        if set(recipe["requires"]) <= result:
            result.add(name)
    return result


def select_provider(family: str, preferred: str | None, deployment: dict) -> tuple[str | None, str | None]:
    """Never choose a provider by installation/discovery order."""
    if preferred:
        if preferred not in REGISTRY or REGISTRY[preferred]["family"] != family:
            return None, "unknown_or_wrong_family_provider"
        return preferred, None
    candidates = sorted(name for name in deployment.get("providers", {})
                        if name in REGISTRY and REGISTRY[name]["family"] == family)
    if len(candidates) == 1:
        return candidates[0], None
    return None, "explicit_provider_choice_required"


def validate_registry_bindings(deployment: dict) -> None:
    """Reject attempts to relabel tests or augment a registered adapter."""
    for name, configured in deployment.get("providers", {}).items():
        if name not in REGISTRY:
            raise ValueError(f"Unregistered provider: {name}")
        trusted = REGISTRY[name]
        if configured["family"] != trusted["family"] or configured["adapter_id"] != trusted["adapter_id"]:
            raise ValueError(f"Registry family/adapter mismatch: {name}")
        if not set(configured.get("reported_capabilities", [])) <= set(trusted["capabilities"]):
            raise ValueError(f"Deployment cannot grant capabilities to {name}")


def _asset_status(asset: dict) -> dict:
    path, expected = asset.get("path"), asset.get("sha256")
    if not path or not expected:
        return {"status": "unresolved", "digest_verified": False}
    target = Path(path)
    if not target.is_file():
        return {"status": "missing_asset", "digest_verified": False}
    try:
        checksum = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        valid = checksum.hexdigest() == expected
        return {"status": "resolved" if valid else "digest_mismatch", "digest_verified": valid}
    except OSError:
        return {"status": "unreadable", "digest_verified": False}


def doctor(deployment: dict | None = None) -> dict:
    """Read metadata and explicitly configured assets only; never execute workers."""
    from .contracts import default_deployment, validate_deployment
    deployment = deepcopy(deployment) if deployment is not None else default_deployment()
    validate_deployment(deployment)
    validate_registry_bindings(deployment)
    providers = {}
    native_dependencies = {}
    for distribution in ("ase", "numpy", "scipy", "spglib"):
        dependency_version = None
        try:
            dependency_version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            pass
        native_dependencies[distribution] = {"installed_version": dependency_version,
                                             "installed": dependency_version is not None}
    parser_version = native_dependencies["ase"]["installed_version"]
    for name, entry in registry().items():
        config = deployment.get("providers", {}).get(name, {})
        installed_version = None
        if entry["distribution"]:
            try:
                installed_version = metadata.version(entry["distribution"])
            except metadata.PackageNotFoundError:
                pass
        assets = {key: _asset_status(value) for key, value in config.get("assets", {}).items()}
        providers[name] = {**deepcopy(entry),
            "installed": installed_version is not None if entry["distribution"] else None,
            "installed_version": installed_version,
            "configured": config.get("configured", False),
            "configured_version": config.get("version"),
            "version_resolved": bool(installed_version or config.get("version")),
            "asset_status": assets,
            "parser_dependency": {"distribution": "ase", "installed_version": parser_version, "installed": parser_version is not None} if name == "symmetry" else None,
            "native_dependencies": deepcopy(native_dependencies) if name == "symmetry" else None,
            "ready": bool(name == "symmetry" and all(value["installed"] for value in native_dependencies.values()) and config.get("configured", False)),
            "benchmark_eligible": False,
            "model_manifest_configured": bool(config.get("model_manifest")),
            "live_permit_configured": bool(config.get("live_permit")),
            "environment_lock_configured": bool(config.get("environment_lock")),
            "status_notice": "S2 MatterSim worker and numerical recipes implemented; readiness requires trusted asset/permit/runtime checks; no live or science validation" if name == "mattersim" else "Registration/installation is not live or scientific verification",
        }
    return {"schema_version": "ctb.backend_status.v1", "providers": providers,
            "network_accessed": False, "workers_executed": 0,
            "physics_jobs_submitted": 0}


def record_failed_attempt(provider: str, error: str) -> dict:
    """Runtime error contract: an error is not permission to select another method."""
    if provider not in REGISTRY:
        raise ValueError("Unknown provider")
    return {"provider": provider, "family": REGISTRY[provider]["family"],
            "calculation_status": "failed", "quality_status": "unverified",
            "value": None, "error": str(error), "next_action": "record_failure",
            "retry_provider": None, "physics_validated": False}
