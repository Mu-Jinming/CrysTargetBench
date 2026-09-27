"""Content-addressed, atomic stage cache for JSON observation payloads."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .identity import digest


_KEY = re.compile(r"[0-9a-f]{64}\Z")


def _finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Cache identity and payload must contain finite JSON values")
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Cache JSON keys must be strings")
        for entry in value.values():
            _finite_json(entry)
    elif isinstance(value, (list, tuple)):
        for entry in value:
            _finite_json(entry)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError("Cache records require ordinary JSON values")


def stage_key(*, geometry: str | dict, recipe: str, parameters: dict, backend: dict, dependencies: list | dict) -> str:
    """Hash complete scientific inputs, independent of filenames and thresholds.

    Callers pass only the parameters of this stage: IFC construction excludes
    downstream q sampling, while the frequency stage includes that sampling and
    the parent IFC digest.  Task criteria, output paths and item labels have no
    place in this interface.  Missing key inputs are errors, never broad hits.
    """
    if not geometry or not isinstance(geometry, (str, dict)):
        raise ValueError("A complete geometry identity is required")
    if isinstance(geometry, str) and not _KEY.fullmatch(geometry):
        raise ValueError("Geometry identity must be a resolved SHA-256 digest")
    if isinstance(geometry, dict) and not {"cell", "positions", "species"}.issubset(geometry):
        raise ValueError("Inline geometry must include ordered cell, positions and species")
    if not recipe or not isinstance(recipe, str):
        raise ValueError("A versioned recipe is required")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Explicit stage parameters are required")
    if not isinstance(backend, dict) or any(field not in backend for field in ("id", "family", "version", "assets")):
        raise ValueError("Backend id, family, version and assets must be locked")
    if any(not isinstance(backend[field], str) or not backend[field] for field in ("id", "family", "version")):
        raise ValueError("Backend identity fields must be nonempty strings")
    if not isinstance(backend["assets"], dict):
        raise ValueError("Backend assets must be a digest mapping, explicitly empty if none")
    if backend["version"] in {"latest", "unknown", "unresolved"}:
        raise ValueError("Backend version must be resolved")
    if backend["family"] not in {"native", "mlff", "dft", "surrogate", "synthetic", "imported"}:
        raise ValueError("Unknown backend family")
    if any(not isinstance(value, str) or not _KEY.fullmatch(value) for value in backend["assets"].values()):
        raise ValueError("Each asset must be identified by its SHA-256 content digest")
    required_asset = {"dft": "pseudopotential_manifest", "mlff": "checkpoint", "surrogate": "checkpoint"}.get(backend["family"])
    if required_asset and required_asset not in backend["assets"]:
        raise ValueError(f"{backend['family']} cache identity requires {required_asset} digest")
    if not isinstance(dependencies, (list, dict)):
        raise ValueError("Dependencies must be explicit, even when empty")
    upstream = dependencies.values() if isinstance(dependencies, dict) else dependencies
    if any(not isinstance(value, str) or not _KEY.fullmatch(value) for value in upstream):
        raise ValueError("Dependencies require resolved stage or artifact content digests")
    if recipe == "native.space_group.v1":
        required = {"symprec_A", "angle_tolerance_deg", "target_symmetrization"}
        if not required.issubset(parameters):
            raise ValueError("Native symmetry requires both tolerances and explicit symmetrization policy")
        if isinstance(parameters["symprec_A"], bool) or not isinstance(parameters["symprec_A"], (int, float)) or parameters["symprec_A"] <= 0:
            raise ValueError("symprec_A must be positive")
        if isinstance(parameters["angle_tolerance_deg"], bool) or not isinstance(parameters["angle_tolerance_deg"], (int, float)):
            raise ValueError("angle_tolerance_deg must be numeric")
        if parameters["target_symmetrization"] is not False:
            raise ValueError("Native recognition must not impose target symmetry")
    record = {"schema_version": "ctb.stage_key.v1", "geometry": geometry, "recipe": recipe, "parameters": parameters, "backend": backend, "dependencies": dependencies}
    _finite_json(record)
    return digest(record)


class StageCache:
    """Checksummed JSON entries committed with an atomic rename.

    Incomplete temporary files, malformed envelopes, wrong keys and checksum
    mismatches are misses.  No cache payload is executable or unpickled.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("Cache key must be a lowercase SHA-256 digest")
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if path.is_symlink():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)), object_pairs_hook=_unique_object)
            if not isinstance(record, dict) or record.get("schema_version") != "ctb.stage_cache.v1" or record.get("complete") is not True or record.get("key") != key:
                return None
            payload = record.get("payload")
            if not isinstance(payload, dict):
                return None
            _finite_json(payload)
            if record.get("payload_digest") != digest(payload):
                return None
            return payload
        except (OSError, ValueError, TypeError, UnicodeError):
            return None

    def put(self, key: str, payload: dict) -> None:
        path = self._path(key)
        if not isinstance(payload, dict):
            raise ValueError("Cache payload must be an object")
        _finite_json(payload)
        record = {"schema_version": "ctb.stage_cache.v1", "complete": True, "key": key, "payload_digest": digest(payload), "payload": payload}
        encoded = json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":"))
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON cache field: {key}")
        result[key] = value
    return result
