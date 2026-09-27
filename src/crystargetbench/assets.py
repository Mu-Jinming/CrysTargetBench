"""Offline model identity and owner authorization, without importing ML software.

Only trusted deployment code supplies these records. Task/structure data never
selects a checkpoint, executable, import, or permit. Hashes are file identity,
not license approval, safe serialization guarantees, or scientific validation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import metadata
from pathlib import Path

from .contracts import _validate, _loads
from .identity import digest

MATTERSIM_VERSION = "1.1.2"
ASSET_SCHEMA = "ctb.model_asset.v1"
PERMIT_SCHEMA = "ctb.live_permit.v1"
_HASH = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_TEXT = {"type": "string", "minLength": 1}
_LIMIT_NAMES = ("max_unique_structures", "max_supercell_atoms",
    "max_displaced_structures_total", "max_backend_structure_evaluations",
    "max_concurrent_workers", "max_relax_steps_per_structure",
    "max_total_wall_seconds", "max_scratch_bytes", "max_retries", "max_dft_jobs",
    "max_sample_wall_seconds", "per_request_wall_seconds")


def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


_ASSET = _object({
    "schema_version": {"const": ASSET_SCHEMA}, "provider": {"const": "mattersim"},
    "model_id": _TEXT, "model_version": _TEXT,
    "package_version": {"const": MATTERSIM_VERSION}, "package_source": _TEXT,
    "checkpoint_path": _TEXT, "checkpoint_sha256": _HASH,
    "model_config": {"type": "object"}, "model_config_digest": _HASH,
    "source_reference": _TEXT, "license": _TEXT,
    "license_review_status": {"const": "owner_approved"},
    "trusted_source": {"const": True},
    "format": {"const": "torch_weights_only"},
    "precision": {"enum": ["float32", "float64"]},
    "device": {"type": "string", "pattern": "^(cpu|cuda:[0-9]+)$"},
    "allow_tf32": {"const": False}, "mixed_precision": {"const": False},
})
_PERMIT = _object({
    "schema_version": {"const": PERMIT_SCHEMA}, "authorized": {"const": True},
    "owner_approval_reference": _TEXT, "model_lock_digest": _HASH,
    "worker_environment_lock_sha256": _HASH, "deployment_digest": _HASH,
    "input_manifest_sha256": _HASH, "protocol_digest": _HASH,
    "device": {"type": "string", "pattern": "^(cpu|cuda:[0-9]+)$"},
    "allowed_operations": {"type": "array", "minItems": 1, "uniqueItems": True,
        "items": {"enum": ["energy_forces_stress", "relaxation", "finite_displacement_phonon", "bulk_eos"]}},
    "limits": _object({name: {"type": "integer", "minimum": 0} for name in _LIMIT_NAMES}),
    "downloads_authorized": {"const": False}, "cluster_authorized": {"const": False},
})


class AssetError(ValueError):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category


def model_identity(manifest: dict) -> str:
    """Location independent identity includes inference precision and device."""
    return digest({key: value for key, value in manifest.items() if key != "checkpoint_path"})


def verify_asset(manifest: dict) -> dict:
    """Rehash the specified regular file; never loads or acquires its content."""
    try:
        data = _validate(manifest, _ASSET, "model_asset")
    except ValueError as exc:
        raise AssetError("blocked_configuration", str(exc)) from exc
    if digest(data["model_config"]) != data["model_config_digest"]:
        raise AssetError("blocked_asset", "model configuration digest mismatch")
    path = Path(data["checkpoint_path"])
    if not path.is_absolute():
        raise AssetError("blocked_configuration", "checkpoint must be an explicit absolute path")
    if path.is_symlink() or not path.is_file():
        raise AssetError("blocked_asset", "checkpoint is missing or is a symlink")
    checksum = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
    except OSError as exc:
        raise AssetError("blocked_asset", "checkpoint unreadable") from exc
    if checksum.hexdigest() != data["checkpoint_sha256"]:
        raise AssetError("blocked_asset", "checkpoint SHA256 mismatch")
    # No precision substitution. This pinned loader is deliberately conservative.
    if data["precision"] != "float32":
        raise AssetError("blocked_configuration", "MatterSim 1.1.2 adapter supports float32 inference only")
    return {"manifest": data, "model_lock_digest": model_identity(data),
            "digest_verified": True, "checkpoint_bytes": path.stat().st_size,
            "science_validated": False, "benchmark_eligible": False}


def validate_permit(permit: dict | None, model_lock: dict, bindings: dict | None = None) -> dict:
    """A disabled template can never become a runtime permit by flipping a flag."""
    if not permit or permit.get("authorized") is not True:
        raise AssetError("blocked_execution", "explicit owner authorization is absent or disabled")
    try:
        data = _validate(permit, _PERMIT, "live_permit")
    except ValueError as exc:
        raise AssetError("blocked_execution", str(exc)) from exc
    if data["model_lock_digest"] != model_lock["model_lock_digest"]:
        raise AssetError("blocked_execution", "permit/model identity mismatch")
    if data["device"] != model_lock["manifest"]["device"]:
        raise AssetError("blocked_execution", "permit/device mismatch")
    if "energy_forces_stress" not in data["allowed_operations"]:
        raise AssetError("blocked_execution", "E/F/stress operation not authorized")
    required_positive = set(_LIMIT_NAMES) - {"max_dft_jobs", "max_retries", "max_displaced_structures_total"}
    if any(data["limits"][key] <= 0 for key in required_positive):
        raise AssetError("blocked_execution", "positive explicit hard budgets required")
    if data["limits"]["max_dft_jobs"] != 0:
        raise AssetError("blocked_execution", "S2 worker does not execute DFT")
    for key, expected in (bindings or {}).items():
        if key not in {"worker_environment_lock_sha256", "deployment_digest", "input_manifest_sha256", "protocol_digest"}:
            raise AssetError("blocked_configuration", "unrecognized authorization binding")
        if data[key] != expected:
            raise AssetError("blocked_execution", f"permit binding mismatch: {key}")
    return data


def preflight(manifest: dict | None, permit: dict | None, bindings: dict | None = None) -> dict:
    """Non-executing inspection; all missing prerequisites remain visible."""
    errors, lock = [], None
    try:
        lock = verify_asset(manifest or {})
    except AssetError as exc:
        errors.append({"category": exc.category, "reason": str(exc)})
    if not permit or permit.get("authorized") is not True:
        errors.append({"category": "blocked_execution", "reason": "owner authorization absent or disabled"})
    elif lock:
        try:
            validate_permit(permit, lock, bindings)
        except AssetError as exc:
            errors.append({"category": exc.category, "reason": str(exc)})
    return {"schema_version": "ctb.live_preflight.v1", "status": errors[0]["category"] if errors else "ready",
        "missing_or_invalid": errors, "model_lock": deepcopy(lock),
        "physics_calls": 0, "submitted_jobs": 0, "model_loaded": False,
        "downloads_attempted": 0, "mlff_live_tested": False,
        "science_validated": False, "benchmark_eligible": False}


def verify_worker_environment(lock: dict) -> dict:
    """Compare actual installed worker distributions with its reviewed lock.

    RECORD identifies installed package files; it is an installation identity,
    not a new scientific validation or a claim that all files were rehashed.
    """
    schema = _object({"schema_version": {"const": "ctb.worker_environment.v1"},
        "packages": {"type": "object", "additionalProperties": _object({"version": _TEXT, "record_sha256": _HASH})}})
    try:
        data = _validate(lock, schema, "worker_environment")
    except ValueError as exc:
        raise AssetError("blocked_configuration", str(exc)) from exc
    if not {"mattersim", "torch", "ase", "numpy", "crystargetbench"} <= set(data["packages"]):
        raise AssetError("blocked_configuration", "worker environment lock missing essential distributions")
    actual = {}
    for name, expected in data["packages"].items():
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as exc:
            raise AssetError("blocked_configuration", f"worker distribution unavailable: {name}") from exc
        record = distribution.read_text("RECORD")
        actual[name] = {"version": distribution.version,
            "record_sha256": hashlib.sha256(record.encode()).hexdigest() if record is not None else None}
        if actual[name] != expected:
            raise AssetError("blocked_configuration", f"installed worker distribution differs from lock: {name}")
    return {"schema_version": data["schema_version"], "packages": actual}


def verify_environment_file(lock: dict, path, expected_sha256: str) -> None:
    """Bind parsed lock content to the exact owner-approved file bytes."""
    if path is None:
        raise AssetError("blocked_configuration", "explicit environment lock file required")
    try:
        payload = Path(path).read_bytes()
        parsed = _loads(payload.decode("utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise AssetError("blocked_configuration", "environment lock file unreadable") from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256 or parsed != lock:
        raise AssetError("blocked_execution", "environment lock content or file identity mismatch")
