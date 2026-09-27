"""Real CPU symmetry analysis; never uses fixture labels or targets."""
from importlib.metadata import version
from .assessment import make_measurement


def effective_symmetry_parameters(parameters):
    """Select the executed recipe from a locked (possibly mixed) protocol."""
    keys = ('symprec_A', 'angle_tolerance_deg', 'target_symmetrization')
    if not all(key in parameters for key in keys):
        raise ValueError('locked symmetry parameters are required; no implicit defaults')
    result = {key: parameters[key] for key in keys}
    if result['target_symmetrization'] is not False:
        raise ValueError('target symmetrization is not supported')
    return result


def analyze_symmetry(geometry, parameters):
    import spglib
    from ase.data import atomic_numbers

    # The low-level geometry API historically takes the two explicit numerical
    # tolerances. It never symmetrizes toward a target. Orchestrators use the
    # strict resolver above on their complete lock before calling this function.
    parameters = effective_symmetry_parameters({'target_symmetrization': False, **parameters})
    cell = (geometry['cell'], geometry['positions'], [atomic_numbers[s] for s in geometry['species']])
    dataset = spglib.get_symmetry_dataset(cell, symprec=parameters['symprec_A'],
                                           angle_tolerance=parameters['angle_tolerance_deg'])
    if dataset is None:
        raise ValueError('spglib could not identify symmetry')
    return {'space_group_number': int(dataset.number), 'international': dataset.international,
            'hall_number': int(dataset.hall_number), 'hall_symbol': dataset.hall,
            'symprec_A': parameters['symprec_A'], 'angle_tolerance_deg': parameters['angle_tolerance_deg'],
            'spglib_version': version('spglib'), 'ase_version': version('ase'),
            'target_symmetrization': False}


def symmetry_measurement(observation, geometry, item_id, route, protocol_digest):
    return make_measurement(reference_geometry_id=geometry['geometry_id'], item_links=[item_id],
        property_id='space_group_number', definition_id=route['definition_id'],
        value=observation['space_group_number'], unit='1', calculation_status='completed',
        quality_status='accepted', identity_status='verified', backend_id='symmetry',
        backend_family='native', backend_version=observation['spglib_version'], asset_digests={},
        physics_fidelity='native', upstream_fidelity=[], protocol_digest=protocol_digest,
        diagnostics=observation, artifact_refs=[], dependency_measurement_ids=[], reasons=[])
