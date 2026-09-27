"""Explicit S2 candidate recipes; importing these constants never loads a model."""
from copy import deepcopy

from .contracts import default_protocol, validate_protocol
from .recipes.phonon import DEFAULTS as PHONON_DEFAULTS

SUPPORTED_PROPERTIES = {"space_group_number", "phonon_path_min_frequency_THz",
                        "phonon_mesh_min_frequency_THz", "bulk_modulus_eos_GPa"}
RECIPE_VERSIONS = {"relaxation": "ase.fire_frechet.relaxation.v1",
                   "relaxation_implementation":"ctb.ase_relaxation.v2",
                   "relaxation_signature":"ctb.relaxation_signature.v1",
                   "eos_point_signature":"ctb.eos_point_signature.v2",
                   "phonon": "ctb.phonopy.finite_displacement.v1",
                   "sampling": "ctb.phonopy.sampling.v1",
                   "eos": "ase.eos.fixed_shape_internal_relax.bm3.v1"}
RELAXATION_DEFAULTS = {
    "require_converged": True, "fmax_eV_per_A": 0.005, "stress_max_GPa": 0.1,
    "max_steps": 2000, "target_pressure_GPa": 0.0,
    "optimizer": "FIRE", "filter": "FrechetCellFilter", "cell_mask": [True] * 6,
    "dt": 0.05, "dtmax": 0.5, "maxstep_A": 0.1, "max_wall_seconds": 3600.0,
}
EOS_DEFAULTS = {
    "volume_factors": [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06],
    "fit": "birch_murnaghan_3rd_order", "internal_relaxation": True, "fixed_cell_shape": True,
    "fit_rms_max_eV": 0.001, "max_relative_volume_shift": 0.02,
    "max_relative_rms_residual_to_energy_span": 0.05,
}


def supports_s2_task(task):
    return {value["property_id"] for value in task["measurements"].values()} <= SUPPORTED_PROPERTIES


def default_s2_protocol(task, base=None):
    """Promote default MLFF numerical definitions explicitly, preserving S1 DFT."""
    if not supports_s2_task(task):
        raise ValueError("The S2 execution recipe does not support every requested property")
    protocol = deepcopy(base) if base is not None else default_protocol(task)
    properties = {value["property_id"] for value in task["measurements"].values()}
    params = protocol["parameters"]
    if properties - {"space_group_number"} or task["reference_geometry"] == "relaxed":
        params["relaxation"] = {**deepcopy(RELAXATION_DEFAULTS), **params.get("relaxation", {})}
    if any(property_id.startswith("phonon_") for property_id in properties):
        params["phonon"] = {**deepcopy(PHONON_DEFAULTS), **params["phonon"]}
        if params["phonon"]["q_scope"] == "path":
            params["phonon"].pop("mesh", None)
            params["phonon"].pop("gamma_center", None)
        elif params["phonon"]["q_scope"] == "mesh":
            params["phonon"].pop("path_points_parameter", None)
    if "bulk_modulus_eos_GPa" in properties:
        params["eos"] = deepcopy(EOS_DEFAULTS)
    protocol["protocol_id"] = "ctb.s2." + ".".join(sorted(params.keys())) + ".candidate.v1"
    protocol["purpose"] = "S2 explicit FIRE/Frechet, finite-displacement and fixed-shape seven-point EOS candidate; not scientific validation"
    protocol["required_resolutions"] = ["verify trusted model manifest and actual checkpoint digest",
        "bind owner permit to the exact model, worker environment, submission and scientific protocol",
        "enforce every derived calculation budget and verify stationary reference quality",
        "independently validate convergence and physical accuracy before scientific release"]
    return validate_protocol(protocol)


def is_complete_s2_protocol(task, protocol):
    if not supports_s2_task(task):
        return False
    params = protocol["parameters"]
    properties = {value["property_id"] for value in task["measurements"].values()}
    if properties - {"space_group_number"} or task["reference_geometry"] == "relaxed":
        if not set(RELAXATION_DEFAULTS) <= params.get("relaxation", {}).keys():
            return False
    if any(prop.startswith("phonon_") for prop in properties):
        required = set(PHONON_DEFAULTS) - {"path_points_parameter", "mesh", "gamma_center"}
        if not required <= params.get("phonon", {}).keys() or params["phonon"]["nac"] != "none":
            return False
    if "bulk_modulus_eos_GPa" in properties:
        if not set(EOS_DEFAULTS) <= params.get("eos", {}).keys():
            return False
    try:
        validate_protocol(protocol)
    except ValueError:
        return False
    return True
