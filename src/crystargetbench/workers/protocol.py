"""Versioned data-only E/F/stress messages and safe numeric artifact transport."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import uuid

from ..contracts import _finite_data
from ..identity import digest

PROTOCOL_VERSION = "ctb.worker.v1"
EV_A3_TO_GPA = 160.21766208  # ASE 3.26 CODATA 2014, reciprocal ase.units.GPa.
VOIGT_ORDER = ["xx", "yy", "zz", "yz", "xz", "xy"]


def normalize_efs(energy, forces, stress, atom_count, *, stress_unit="eV/angstrom^3",
                  energy_unit="eV", force_unit="eV/angstrom"):
    import numpy as np
    from ase.units import GPa

    if energy_unit != "eV" or force_unit != "eV/angstrom":
        raise ValueError("requires total energy eV and forces eV/angstrom")
    if isinstance(energy, (bool, str)) or np.asarray(energy).shape != ():
        raise ValueError("energy must be a scalar total energy")
    energy = float(energy)
    forces = np.asarray(forces)
    force_dtype = str(forces.dtype)
    if forces.dtype.kind not in "fi" or forces.shape != (atom_count, 3):
        raise ValueError("forces must be a numeric N x 3 array in original atom order")
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise ValueError("nonfinite energy or forces")
    raw_stress = None if stress is None else np.asarray(stress)
    normalized = None
    conversions = []
    if raw_stress is not None:
        if raw_stress.dtype.kind not in "fi" or raw_stress.shape not in ((6,), (3, 3)):
            raise ValueError("stress requires ASE Voigt 6 or a symmetric 3 x 3 tensor")
        if not np.isfinite(raw_stress).all():
            raise ValueError("nonfinite stress")
        normalized = raw_stress.astype(float)
        if normalized.shape == (6,):
            xx, yy, zz, yz, xz, xy = normalized
            normalized = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        if not np.allclose(normalized, normalized.T, rtol=0, atol=1e-10):
            raise ValueError("stress tensor is not symmetric")
        if stress_unit == "GPa":
            normalized *= GPa
            conversions.append({"quantity": "stress", "from": "GPa", "to": "eV/angstrom^3", "factor": GPa})
        elif stress_unit != "eV/angstrom^3":
            raise ValueError("unsupported stress unit")
    elif stress_unit not in ("GPa", "eV/angstrom^3"):
        raise ValueError("unsupported stress unit")
    return {"energy_eV": energy, "forces_eV_A": forces.astype(float).tolist(),
        "stress_eV_A3": normalized.tolist() if normalized is not None else None,
        "stress_GPa": (normalized / GPa).tolist() if normalized is not None else None,
        "pressure_GPa": float(-np.trace(normalized) / (3 * GPa)) if normalized is not None else None,
        "raw": {"energy": energy, "forces": forces.tolist(), "stress": raw_stress.tolist() if raw_stress is not None else None},
        "raw_units": {"energy": energy_unit, "forces": force_unit, "stress": stress_unit},
        "units": {"energy": "eV", "forces": "eV/angstrom", "stress": "eV/angstrom^3"},
        "conversions": conversions, "stress_convention": "ASE tensile_positive", "voigt_order": VOIGT_ORDER,
        "force_dtype": force_dtype, "postprocess_dtype": "float64",
        "quality_status": "valid" if normalized is not None else "missing_stress"}


def validate_request(request):
    _finite_data(request)
    required = {"protocol_version", "request_id", "attempt_id", "node_id", "operation",
                "geometry", "geometry_id", "mapping", "model_lock_digest", "budget_lease"}
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError("worker request has missing or unknown fields")
    if request["protocol_version"] != PROTOCOL_VERSION or request["operation"] != "energy_forces_stress":
        raise ValueError("unsupported worker protocol or operation")
    for key in ("request_id", "attempt_id", "node_id", "geometry_id", "model_lock_digest"):
        if not isinstance(request[key], str) or not request[key]:
            raise ValueError(f"invalid request identity: {key}")
    from ..geometry import atoms_from_snapshot
    geometry_allowed = {"cell", "positions", "coordinate_system", "length_unit", "species",
        "occupancy", "pbc", "magnetic_state", "geometry_id", "masses", "mass_unit", "mass_digest",
        "parent", "stage", "transformation", "domain_checks"}
    geometry_required = {"cell", "positions", "coordinate_system", "length_unit", "species",
        "occupancy", "pbc", "magnetic_state", "geometry_id"}
    if not isinstance(request["geometry"], dict) or set(request["geometry"]) - geometry_allowed or not geometry_required <= set(request["geometry"]):
        raise ValueError("geometry requires strict data-only fields")
    if "masses" in request["geometry"] and digest(request["geometry"]["masses"]) != request["geometry"].get("mass_digest"):
        raise ValueError("geometry mass identity mismatch")
    if request["geometry"].get("mass_unit", "amu") != "amu":
        raise ValueError("geometry masses require amu")
    if request["geometry"]["pbc"] != [True, True, True] or any(type(p) is not bool for p in request["geometry"]["pbc"]):
        raise ValueError("worker requires three explicit periodic dimensions")
    atoms = atoms_from_snapshot(request["geometry"])
    if any(int(number) <= 0 for number in atoms.numbers):
        raise ValueError("worker requires known chemical species")
    if any(float(moment) != 0 for moment in atoms.get_initial_magnetic_moments().flat):
        raise ValueError("MatterSim E/F/stress adapter does not model a prescribed magnetic state")
    if request["geometry"]["geometry_id"] != request["geometry_id"]:
        raise ValueError("request geometry identity mismatch")
    mapping = request["mapping"]
    n = len(request["geometry"]["species"])
    if not isinstance(mapping, dict) or not {"site_mapping", "basis_mapping"} <= set(mapping):
        raise ValueError("strict site/basis mapping required")
    if {"shell", "import", "checkpoint_path", "worker_argv", "model_lock", "permit"} & set(mapping):
        raise ValueError("mapping cannot override execution or model configuration")
    import numpy as np
    sites = mapping["site_mapping"]
    if (not isinstance(sites, list) or len(sites) != n or
            any(type(x) is not int or x < 0 for x in sites)):
        raise ValueError("mapping requires one nonnegative site index per atom")
    basis = np.asarray(mapping["basis_mapping"])
    if basis.shape != (3, 3) or basis.dtype.kind not in "fi" or not np.isfinite(basis).all() or abs(np.linalg.det(basis)) < 1e-12:
        raise ValueError("invalid basis mapping")
    lease = request["budget_lease"]
    if not isinstance(lease, dict) or not isinstance(lease.get("lease_id"), str) or not lease["lease_id"]:
        raise ValueError("reserved budget lease required")
    if lease.get("status", "reserved") != "reserved":
        raise ValueError("budget lease is not reserved")
    return request


def response_identity(request):
    return {key: request[key] for key in ("protocol_version", "request_id", "attempt_id", "node_id",
                                         "geometry_id", "model_lock_digest")}


def validate_response(response, request):
    _finite_data(response)
    for key, value in response_identity(request).items():
        if response.get(key) != value:
            raise ValueError(f"worker response identity mismatch: {key}")
    if response.get("mapping_digest") != digest(request["mapping"]):
        raise ValueError("worker response atom/basis mapping mismatch")
    if response.get("status") == "error":
        if not isinstance(response.get("error"), dict):
            raise ValueError("worker error requires structured category")
        return response
    if response.get("status") != "completed":
        raise ValueError("worker response is not completed")
    observation = response["observation"]
    normalized = normalize_efs(observation["energy_eV"], observation["forces_eV_A"],
        observation["stress_eV_A3"], len(request["geometry"]["species"]))
    raw, units = observation["raw"], observation["raw_units"]
    from_raw = normalize_efs(raw["energy"], raw["forces"], raw["stress"], len(request["geometry"]["species"]),
        energy_unit=units["energy"], force_unit=units["forces"], stress_unit=units["stress"])
    for field in ("energy_eV", "forces_eV_A", "stress_eV_A3", "stress_GPa", "pressure_GPa"):
        if observation[field] != from_raw[field]:
            raise ValueError(f"raw/normalized worker value mismatch: {field}")
    if normalized["quality_status"] != observation.get("quality_status"):
        raise ValueError("worker quality mismatch")
    if response.get("actual_geometry_id") != request["geometry_id"]:
        raise ValueError("actual calculation geometry differs from request")
    if not isinstance(response.get("environment"), dict):
        raise ValueError("execution environment required")
    cost = response.get("cost", {})
    if cost.get("backend_forwards") != 1 or cost.get("evaluated_structures") != 1:
        raise ValueError("serial worker must account for one actual forward and structure")
    return response


def write_npz(workdir, name, arrays):
    """Atomic numeric-only NPZ plus an independently hashed manifest."""
    import numpy as np
    if Path(name).name != name or not name.endswith(".npz"):
        raise ValueError("artifact must be a local .npz basename")
    root = Path(workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.is_symlink():
        raise ValueError("artifact symlink rejected")
    converted = {key: np.asarray(value) for key, value in arrays.items()}
    if any(not key or not key.replace("_", "").isalnum() for key in converted):
        raise ValueError("invalid array key")
    if any(value.dtype.kind not in "fiub" or not np.isfinite(value).all() for value in converted.values()):
        raise ValueError("NPZ accepts finite numeric arrays only")
    temp = root / ("." + uuid.uuid4().hex + ".npz")
    try:
        with temp.open("xb") as stream:
            np.savez(stream, **converted)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return {"path": name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "bytes": target.stat().st_size,
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in converted.items()}}


def read_npz(workdir, manifest, *, max_bytes=128 * 1024 * 1024):
    import numpy as np
    import zipfile
    name = manifest["path"]
    if Path(name).name != name or not name.endswith(".npz"):
        raise ValueError("artifact must be a local .npz basename")
    path = Path(workdir).resolve() / name
    if path.is_symlink() or path.stat().st_size > max_bytes:
        raise ValueError("unsafe or oversized NPZ")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("artifact checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > max_bytes:
            raise ValueError("NPZ decompressed size exceeds limit")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    actual = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in result.items()}
    if actual != manifest["arrays"]:
        raise ValueError("artifact array manifest mismatch")
    if any(value.dtype.kind not in "fiub" or not np.isfinite(value).all() for value in result.values()):
        raise ValueError("nonfinite or nonnumeric NPZ data")
    return result
