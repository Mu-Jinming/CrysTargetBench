"""Pinned Phonopy finite displacements and explicit reciprocal-space sampling.

This adapter uses the public 2.38.2 API checked against the upstream versioned
source. It never reads BORN files, discovers calculators, or changes a potential.
SeeK-path 2.1.0 is used on the *actual* Phonopy primitive cell with
``get_path_orig_cell``; its returned q coordinates therefore have the same basis
as the dynamical matrix. All arrays in persisted artifacts are ordinary JSON.
"""
from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
import warnings

from ..identity import digest


PHONOPY_VERSION = "2.38.2"
SEEKPATH_VERSION = "2.1.0"
IFC_RECIPE = "ctb.phonopy.finite_displacement.v1"
SAMPLING_RECIPE = "ctb.phonopy.sampling.v1"
DEFAULTS = {
    "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    "primitive_matrix": "auto", "symprec_A": 1e-5, "masses": None,
    "displacement_A": 0.01, "is_plusminus": "auto", "is_diagonal": True,
    "is_symmetry": True, "fc_calculator": "traditional",
    "force_drift_subtraction": True, "force_constant_symmetrization": True,
    "symmetrization_level": 1, "nac": "none", "path_points_parameter": 51,
    "mesh": [4, 4, 4], "gamma_center": True,
    "is_time_reversal": True, "is_mesh_symmetry": True,
}
IFC_FIELDS = ("supercell_matrix", "primitive_matrix", "symprec_A", "masses", "displacement_A",
              "is_plusminus", "is_diagonal", "is_symmetry", "fc_calculator",
              "force_drift_subtraction", "force_constant_symmetrization", "symmetrization_level", "nac")


def _numpy():
    import numpy as np
    return np


def versions() -> dict:
    result = {name: version(name) for name in ("phonopy", "ase", "seekpath", "spglib", "numpy")}
    if result["phonopy"] != PHONOPY_VERSION or result["seekpath"] != SEEKPATH_VERSION:
        raise ValueError("Phonopy/SeeK-path version differs from the tested 2.38.2/2.1.0 adapter")
    return result


def resolved_parameters(parameters: dict) -> dict:
    """Resolve explicit candidate defaults; task predicates never enter this API."""
    if set(parameters) - (set(DEFAULTS) | {"q_scope"}):
        raise ValueError("Unknown finite-displacement/sampling parameter")
    result = {**deepcopy(DEFAULTS), **deepcopy(parameters)}
    if result["nac"] != "none":
        raise ValueError("NAC requires independently provided Born/electronic capability; S2 supports NAC=none only")
    if result["fc_calculator"] != "traditional":
        raise ValueError("Only the pinned traditional finite-displacement IFC solver is implemented")
    for key in ("is_diagonal", "is_symmetry", "force_drift_subtraction", "force_constant_symmetrization", "gamma_center", "is_time_reversal", "is_mesh_symmetry"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if type(result["symmetrization_level"]) is not int or result["symmetrization_level"] < 1:
        raise ValueError("symmetrization_level must be a positive integer")
    for key in ("displacement_A", "symprec_A"):
        if type(result[key]) not in (int, float) or not _numpy().isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    matrix = _numpy().asarray(result["supercell_matrix"])
    if matrix.shape != (3, 3) or not _numpy().issubdtype(matrix.dtype, _numpy().integer) or _numpy().linalg.det(matrix) <= 0:
        raise ValueError("supercell_matrix must be a right-handed nonsingular integer 3x3 matrix")
    primitive = result["primitive_matrix"]
    if isinstance(primitive, str):
        if primitive != "auto":
            raise ValueError("primitive_matrix string must be 'auto'")
    else:
        primitive = _numpy().asarray(primitive, dtype=float)
        if primitive.shape != (3, 3) or not _numpy().isfinite(primitive).all() or _numpy().linalg.det(primitive) <= 0:
            raise ValueError("primitive_matrix must be right-handed and nonsingular")
    if result["is_plusminus"] != "auto" and type(result["is_plusminus"]) is not bool:
        raise ValueError("is_plusminus must be boolean or 'auto'")
    if result["masses"] is not None:
        masses = _numpy().asarray(result["masses"], dtype=float)
        if masses.ndim != 1 or not len(masses) or not _numpy().isfinite(masses).all() or (masses <= 0).any():
            raise ValueError("Primitive masses must be a nonempty finite positive array")
    return result


def _to_phonopy(reference: dict, params: dict):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.physical_units import get_physical_units
    from ..geometry import atoms_from_snapshot
    atoms = atoms_from_snapshot(reference)
    unitcell = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array,
                           scaled_positions=atoms.get_scaled_positions(wrap=False), masses=atoms.get_masses())
    phonon = Phonopy(unitcell, supercell_matrix=params["supercell_matrix"],
                     primitive_matrix=params["primitive_matrix"], symprec=params["symprec_A"],
                     is_symmetry=params["is_symmetry"], calculator="vasp",
                     factor=get_physical_units().DefaultToTHz, log_level=0)
    # The VASP unit convention here means eV/angstrom/amu only, not a DFT backend.
    if params["masses"] is not None:
        if len(params["masses"]) != len(phonon.primitive):
            raise ValueError("masses must match the actual primitive atom order")
        phonon.masses = params["masses"]
    return phonon


def _atoms(cell):
    from ase import Atoms
    return Atoms(symbols=cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions,
                 masses=cell.masses, pbc=True)


def _cell_record(cell):
    return {"cell": cell.cell.tolist(), "positions": cell.scaled_positions.tolist(),
            "species": list(cell.symbols), "masses_amu": cell.masses.tolist(),
            "coordinate_system": "fractional", "length_unit": "angstrom"}


def _mappings(phonon, reference_id):
    np = _numpy()
    supercell, primitive = phonon.supercell, phonon.primitive
    unit = phonon.unitcell
    u2s = supercell.u2s_map.tolist()
    # s2u_map values are representative *supercell* indices, not unit indices.
    representative_to_unit = {representative: index for index, representative in enumerate(u2s)}
    s2u = [representative_to_unit[int(index)] for index in supercell.s2u_map]
    primitive_representatives = primitive.p2s_map.tolist()
    representative_to_primitive = {representative: index for index, representative in enumerate(primitive_representatives)}
    s2p = [representative_to_primitive[int(index)] for index in primitive.s2p_map]
    translations = []
    for index, unit_index in enumerate(s2u):
        fractional = supercell.positions[index] @ np.linalg.inv(unit.cell) - unit.scaled_positions[unit_index]
        translation = np.rint(fractional).astype(int)
        if not np.allclose(fractional, translation, atol=1e-7, rtol=0):
            raise ValueError("Supercell site mapping would move the reference geometry")
        translations.append(translation.tolist())
    return {"reference_geometry_id": reference_id, "unitcell_to_supercell": u2s,
            "supercell_to_unitcell": s2u, "primitive_to_supercell": primitive_representatives,
            "supercell_to_primitive": s2p, "unitcell_lattice_translations": translations,
            "supercell_basis_in_unitcell": (supercell.cell @ np.linalg.inv(unit.cell)).tolist(),
            "primitive_basis_in_unitcell": (primitive.cell @ np.linalg.inv(unit.cell)).tolist(),
            "ordered_site_ids": [digest({"reference": reference_id, "site": site, "translation": translation})
                                 for site, translation in zip(s2u, translations)]}


def assemble_forces(expected: list[dict], results: list[dict]):
    """Join asynchronous results by immutable ID and validate full ordered identity."""
    np = _numpy()
    by_id = {}
    for result in results:
        identifier = result.get("node_id")
        if identifier in by_id:
            raise ValueError("Duplicate displacement result ID")
        by_id[identifier] = result
    if set(by_id) != {record["displacement_id"] for record in expected}:
        raise ValueError("Missing or unexpected displacement results")
    ordered = []
    for record in expected:
        result = by_id[record["displacement_id"]]
        if result.get("geometry_id") != record["geometry_id"]:
            raise ValueError("Displacement geometry hash mismatch")
        if result.get("mapping_digest") != record["mapping_digest"]:
            raise ValueError("Displacement site/basis mapping mismatch")
        forces = np.asarray(result.get("forces_eV_A"), dtype=float)
        if forces.shape != (record["atom_count"], 3) or not np.isfinite(forces).all():
            raise ValueError("Displacement forces must be finite ordered N x 3")
        ordered.append(forces)
    return np.asarray(ordered, dtype="double", order="C")


def ifc_residuals(force_constants) -> dict:
    np = _numpy()
    constants = np.asarray(force_constants, dtype=float)
    if constants.ndim != 4 or constants.shape[0] != constants.shape[1] or constants.shape[2:] != (3, 3) or not np.isfinite(constants).all():
        raise ValueError("Full finite force constants of shape N x N x 3 x 3 required")
    return {"asr_max_abs_eV_A2": float(np.max(np.abs(constants.sum(axis=1)))),
            "permutation_max_abs_eV_A2": float(np.max(np.abs(constants - constants.transpose(1, 0, 3, 2))))}


def _force_digest(result):
    return digest({key: result.get(key) for key in ("geometry_id", "mapping_digest", "forces_eV_A", "force_dtype")})


def build_ifc(reference: dict, context, parameters: dict) -> dict:
    """Execute cached force requests and fit real Phonopy force constants.

    The caller is responsible for an accepted stationary reference and authorized
    backend context. This function neither relaxes displaced cells nor grants
    authorization. Synthetic contexts retain their synthetic provenance.
    """
    np = _numpy()
    from ..geometry import snapshot_from_atoms
    params, lock = resolved_parameters(parameters), versions()
    phonon = _to_phonopy(reference, params)
    context.budget.check_atoms(len(phonon.supercell))
    phonon.generate_displacements(distance=params["displacement_A"], is_plusminus=params["is_plusminus"],
                                  is_diagonal=params["is_diagonal"])
    mappings = _mappings(phonon, reference["geometry_id"])
    base_atoms = _atoms(phonon.supercell)
    base_snapshot = snapshot_from_atoms(base_atoms, parent=reference["geometry_id"], stage="phonon_supercell", transformation=mappings)
    expected, results = [], []
    force_params = {key: params[key] for key in ("supercell_matrix", "primitive_matrix", "symprec_A", "displacement_A", "is_plusminus", "is_diagonal", "is_symmetry")}
    force_params["implementation"] = IFC_RECIPE
    force_params["versions"] = lock

    def compute_force(atoms, snapshot, identifier, mapping, is_displacement):
        def compute():
            if is_displacement:
                context.budget.reserve_displacement(identifier)
            return context.evaluate(atoms, identifier, mapping=mapping)
        result = context.cached("phonon.displacement_forces.v1" if is_displacement else "phonon.reference_forces.v1",
                                snapshot["geometry_id"], {**force_params, "mapping_digest": digest(mapping)}, [], compute)
        expected_record = {"displacement_id": identifier, "geometry_id": snapshot["geometry_id"],
                           "mapping_digest": digest(mapping), "atom_count": len(atoms)}
        assemble_forces([expected_record], [result])
        return expected_record, result

    base_id = "reference-supercell:" + digest({"geometry": base_snapshot["geometry_id"], "mapping": mappings})
    base_mapping = {**mappings, "displacement_id": base_id, "ordered_geometry_id": base_snapshot["geometry_id"]}
    _, base_result = compute_force(base_atoms, base_snapshot, base_id, base_mapping, False)
    for index, displaced in enumerate(phonon.supercells_with_displacements):
        atoms = _atoms(displaced)
        displacement = {"index": index, "atom": int(phonon.displacements[index][0]),
                        "cartesian_A": [float(x) for x in phonon.displacements[index][1:]]}
        snapshot = snapshot_from_atoms(atoms, parent=base_snapshot["geometry_id"], stage="phonon_displacement", transformation={"kind": "finite_displacement", **displacement})
        identifier = "displacement:" + digest({"geometry": snapshot["geometry_id"], "reference": reference["geometry_id"], "displacement": displacement, "parameters": force_params})
        mapping = {**mappings, "displacement_id": identifier, "ordered_geometry_id": snapshot["geometry_id"], "displacement": displacement}
        record, result = compute_force(atoms, snapshot, identifier, mapping, True)
        record.update(displacement=displacement, mapping=mapping, geometry=snapshot)
        expected.append(record)
        results.append(result)
    raw = assemble_forces(expected, results)
    if not len(raw):
        raise ValueError("Phonopy produced no displacement force sets")
    drift = raw.mean(axis=1)
    corrected = raw - drift[:, None, :] if params["force_drift_subtraction"] else raw.copy()
    dependencies = [_force_digest(base_result), *[_force_digest(result) for result in results]]
    ifc_params = {key: params[key] for key in IFC_FIELDS}
    ifc_params["versions"] = lock
    ifc_params["reference_masses_amu"] = phonon.unitcell.masses.tolist()

    def compute_ifc():
        phonon.forces = raw
        phonon.produce_force_constants(calculate_full_force_constants=True, fc_calculator="traditional", show_drift=False)
        raw_ifc = phonon.force_constants.copy()
        phonon.forces = corrected
        phonon.produce_force_constants(calculate_full_force_constants=True, fc_calculator="traditional", show_drift=False)
        corrected_ifc = phonon.force_constants.copy()
        if params["force_constant_symmetrization"]:
            phonon.symmetrize_force_constants(level=params["symmetrization_level"], show_drift=False)
        final_ifc = phonon.force_constants.copy()
        base_forces = np.asarray(base_result["forces_eV_A"])
        artifact = {"schema_version": "ctb.ifc.v1", "recipe": IFC_RECIPE,
            "reference_geometry_id": reference["geometry_id"], "reference": deepcopy(reference),
            "parameters": ifc_params, "versions": lock,
            "unitcell": _cell_record(phonon.unitcell), "supercell": _cell_record(phonon.supercell),
            "primitive": _cell_record(phonon.primitive), "mappings": mappings,
            "supercell_geometry_id": base_snapshot["geometry_id"], "displacements": expected,
            "raw_forces_eV_A": raw.tolist(), "corrected_forces_eV_A": corrected.tolist(),
            "force_drift_eV_A": drift.tolist(), "force_drift_subtracted": params["force_drift_subtraction"],
            "reference_supercell_forces_eV_A": base_forces.tolist(),
            "reference_force_max_eV_A": float(np.linalg.norm(base_forces, axis=1).max()),
            "reference_force_drift_eV_A": base_forces.mean(axis=0).tolist(),
            "raw_force_constants_eV_A2": raw_ifc.tolist(),
            "drift_corrected_force_constants_eV_A2": corrected_ifc.tolist(),
            "force_constants_eV_A2": final_ifc.tolist(),
            "ifc_diagnostics": {"raw": ifc_residuals(raw_ifc), "drift_corrected": ifc_residuals(corrected_ifc), "final": ifc_residuals(final_ifc),
                "correction_l2_eV_A2": float(np.linalg.norm(final_ifc - raw_ifc)),
                "drift_l2_eV_A": float(np.linalg.norm(drift))},
            "force_dtype": sorted({result.get("force_dtype", "unknown") for result in results}),
            "postprocessing_dtype": "float64", "frequency_conversion_THz": float(phonon.unit_conversion_factor),
            "frequency_unit": "THz", "force_constants_unit": "eV/angstrom^2", "nac": "none",
            "force_dependency_digests": dependencies, "calculation_status": "completed",
            "quality_status": "accepted", "physics_fidelity": base_result.get("physics_fidelity", "unknown"),
            "evidence_kind": base_result.get("evidence_kind", "unverified"),
            "mlff_live_tested": False, "science_validated": False, "benchmark_eligible": False}
        artifact["force_constants_digest"] = digest(artifact["force_constants_eV_A2"])
        artifact["artifact_digest"] = digest(artifact)
        return artifact
    return context.cached(IFC_RECIPE, reference["geometry_id"], ifc_params, dependencies, compute_ifc)


def validate_frequencies(frequencies, qpoints, primitive_atoms: int, *, scope, weights=None) -> dict:
    """Validate before reduction; preserve all signed acoustic and optical modes."""
    from ..physics import phonon_statistics
    np = _numpy()
    try:
        if np.asarray(frequencies).dtype == np.dtype(bool):
            raise ValueError("Boolean frequencies")
        array = np.asarray(frequencies, dtype=float)
    except (ValueError, TypeError):
        return {"quality_status": "rejected", "value": None, "min_frequency_THz": None,
                "reasons": ["malformed_frequency_array"], "scope": scope}
    if array.ndim != 2 or not array.shape[0] or array.shape[1] != 3 * primitive_atoms or not np.isfinite(array).all():
        return {"quality_status": "rejected", "value": None, "min_frequency_THz": None,
                "reasons": ["empty_nonfinite_or_wrong_branch_count"], "scope": scope}
    stats = phonon_statistics(array.tolist(), scope=scope, qpoints=qpoints, weights=weights)
    stats["value"] = stats["min_frequency_THz"]
    return stats


def path_sampling(primitive, params):
    """CPU-only actual path in the supplied primitive reciprocal basis."""
    np = _numpy()
    segments = []
    import seekpath
    points = params["path_points_parameter"]
    if type(points) is not int or points < 2:
        raise ValueError("path_points_parameter must be at least two points per segment")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        path = seekpath.get_path_orig_cell((primitive.cell, primitive.scaled_positions,
                                           primitive.numbers), symprec=params["symprec_A"],
                                           with_time_reversal=params["is_time_reversal"], recipe="hpkot",
                                           threshold=1e-7, angle_tolerance=-1.0)
    warnings_saved = [str(warning.message) for warning in captured]
    labels = {name: [float(value) for value in coordinate] for name, coordinate in path["point_coords"].items()}
    segments_arrays = []
    start = 0
    for left, right in path["path"]:
        segment = np.linspace(labels[left], labels[right], points)
        segments_arrays.append(segment)
        segments.append({"start_index": start, "stop_index_exclusive": start + points,
                         "labels": [left, right]})
        start += points
    if not segments_arrays:
        raise ValueError("SeeK-path returned an empty path")
    qpoints = np.concatenate(segments_arrays)
    sampling = {"path_points_parameter": points, "path_method": "seekpath.get_path_orig_cell",
                "seekpath_recipe": "hpkot", "edge_threshold": 1e-7, "angle_tolerance": -1.0,
                "is_time_reversal": params["is_time_reversal"],
                "endpoint_rule": "preserve_both_endpoints_of_every_segment",
                "is_supercell": bool(path.get("is_supercell", False)),
                "has_inversion_symmetry": bool(path["has_inversion_symmetry"])}
    return qpoints, sampling, segments, labels, warnings_saved


def sample_phonons(ifc_artifact: dict, parameters: dict, scope: str) -> dict:
    """Compute actual Phonopy frequencies, independent of any task threshold."""
    np = _numpy()
    params, lock = resolved_parameters(parameters), versions()
    if scope not in {"path", "mesh", "gamma"}:
        raise ValueError("scope must be explicitly path, mesh, or gamma")
    if ifc_artifact.get("calculation_status") != "completed" or ifc_artifact.get("quality_status") != "accepted":
        raise ValueError("A completed accepted IFC artifact is required")
    if digest(ifc_artifact["force_constants_eV_A2"]) != ifc_artifact["force_constants_digest"]:
        raise ValueError("IFC artifact checksum mismatch")
    # A q-sampling call cannot silently rebind an IFC to a different supercell,
    # displacement or primitive. Masses are the intentional exception.
    for key in IFC_FIELDS:
        if key != "masses" and params[key] != ifc_artifact["parameters"][key]:
            raise ValueError(f"IFC parameter mismatch: {key}")
    phonon = _to_phonopy(ifc_artifact["reference"], params)
    if _mappings(phonon, ifc_artifact["reference_geometry_id"]) != ifc_artifact["mappings"]:
        raise ValueError("IFC primitive/supercell/site mapping mismatch")
    phonon.force_constants = np.asarray(ifc_artifact["force_constants_eV_A2"], dtype="double", order="C")
    warnings_saved, segments, labels = [], [], {}
    if scope == "mesh":
        mesh = params["mesh"]
        if len(mesh) != 3 or any(type(value) is not int or value < 1 for value in mesh):
            raise ValueError("mesh must contain three positive integer counts")
        phonon.run_mesh(mesh, shift=[0.0, 0.0, 0.0], is_time_reversal=params["is_time_reversal"],
                       is_mesh_symmetry=params["is_mesh_symmetry"], is_gamma_center=params["gamma_center"])
        sampled = phonon.get_mesh_dict()
        qpoints = sampled["qpoints"]
        frequencies = sampled["frequencies"]
        weights = sampled["weights"].tolist()
        if sum(weights) != int(np.prod(mesh)):
            raise ValueError("Irreducible mesh weights do not sum to the full mesh size")
        sampling = {key: params[key] for key in ("mesh", "gamma_center", "is_time_reversal", "is_mesh_symmetry")}
        sampling.update(shift=[0.0, 0.0, 0.0], weight_sum=sum(weights))
    else:
        if scope == "gamma":
            qpoints = np.zeros((1, 3))
            sampling = {"qpoints": [[0.0, 0.0, 0.0]]}
        else:
            qpoints, sampling, segments, labels, warnings_saved = path_sampling(phonon.primitive, params)
        phonon.run_qpoints(qpoints)
        frequencies = phonon.get_qpoints_dict()["frequencies"]
        weights = [1] * len(qpoints)
    primitive_count = len(phonon.primitive)
    statistics = validate_frequencies(frequencies, qpoints.tolist(), primitive_count, scope=scope, weights=weights)
    reciprocal = 2 * np.pi * np.linalg.inv(phonon.primitive.cell).T
    artifact = {"schema_version": "ctb.phonon_sampling.v1", "recipe": SAMPLING_RECIPE,
        "reference_geometry_id": ifc_artifact["reference_geometry_id"], "ifc_artifact_digest": ifc_artifact["artifact_digest"],
        "force_constants_digest": ifc_artifact["force_constants_digest"], "scope": scope,
        "sampling_parameters": sampling, "masses_amu": phonon.primitive.masses.tolist(),
        "reciprocal_basis_Ainv": reciprocal.tolist(), "reciprocal_convention": "row_vectors_include_2pi",
        "qpoints": qpoints.tolist(), "qpoints_cartesian_Ainv": (qpoints @ reciprocal).tolist(),
        "labels": labels, "segments": segments, "warnings": warnings_saved,
        "frequencies_THz": frequencies.tolist(), "weights": weights, "weight_sum": sum(weights),
        "primitive_atom_count": primitive_count, "branch_count": 3 * primitive_count,
        "signed_frequencies_preserved": True, "nac": "none", "versions": lock,
        "statistics": statistics, "value": statistics.get("value"), "calculation_status": "completed",
        "quality_status": statistics["quality_status"], "reasons": statistics.get("reasons", []),
        "physics_fidelity": ifc_artifact["physics_fidelity"], "evidence_kind": ifc_artifact["evidence_kind"],
        "mlff_live_tested": False, "science_validated": False, "benchmark_eligible": False}
    artifact["artifact_digest"] = digest(artifact)
    return artifact
