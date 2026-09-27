"""Request-bound sampling definitions; CPU geometry work only, never a solver.

Wire v2 supports full electronic meshes. Irreducible electronic meshes are
explicitly unsupported until symmetry expansion can be independently verified.
"""
from copy import deepcopy
from functools import lru_cache
import json
import numpy as np
from ..identity import digest

MAX_POINTS = 100_000
CONVENTION = 'fractional_reciprocal_row_vectors_include_2pi'


def full_mesh(grid, shift=(0, 0, 0)):
    if len(grid) != 3 or any(type(x) is not int or x < 1 for x in grid) or np.prod(grid, dtype=object) > MAX_POINTS:
        raise ValueError('explicit bounded full mesh required')
    if len(shift) != 3 or any(type(x) is not int or x not in (0, 1) for x in shift):
        raise ValueError('mesh shift must be three half-grid flags')
    return [[(i+shift[0]/2)/grid[0], (j+shift[1]/2)/grid[1], (k+shift[2]/2)/grid[2]]
            for i in range(grid[0]) for j in range(grid[1]) for k in range(grid[2])]


def sealed(record):
    return {**record, 'manifest_digest': digest(record)}


def electronic_manifest(geom, method):
    p = method['effective_parameters']
    return sealed({'schema_version': 'ctb.electronic_sampling.v1', 'coverage': 'full_mesh',
        'reference_geometry_id': geom['geometry_id'], 'coordinate_convention': CONVENTION,
        'reciprocal_basis_Ainv': (2*np.pi*np.linalg.inv(geom['cell']).T).tolist(),
        'grid': p['k_grid'], 'half_grid_shift': p['k_shift'],
        'kpoints': full_mesh(p['k_grid'], p['k_shift']), 'weight_definition': 'uniform_normalized',
        'irreducible_policy': 'unsupported_requires_verified_expansion'})


@lru_cache(maxsize=32)
def _phonon_manifest(geometry_json, parameters_json):
    geom, parameters = json.loads(geometry_json), json.loads(parameters_json)
    scope = parameters.get('q_scope')
    base = {'schema_version': 'ctb.external_phonon_sampling.v1',
            'scope': scope, 'reference_geometry_id': geom['geometry_id'],
            'coordinate_convention': CONVENTION}
    if scope == 'mesh':
        if parameters.get('gamma_center') is not True:
            raise ValueError('direct spectra require full Gamma-centred mesh')
        base.update(qpoints=full_mesh(parameters.get('mesh', [])), segments=[], labels={},
            reciprocal_basis_Ainv=(2*np.pi*np.linalg.inv(geom['cell']).T).tolist(),
            sampling_basis='reference', branch_count=3*len(geom['species']),
            basis_to_reference_reciprocal=np.eye(3).tolist(),
            generator={'name': 'ctb.full_gamma_mesh', 'version': 1})
    elif scope == 'path':
        from ..recipes.phonon import resolved_parameters, versions, _to_phonopy, _cell_record, path_sampling
        params = resolved_parameters(parameters)
        points = params['path_points_parameter']
        if type(points) is not int or not 2 <= points <= MAX_POINTS//100:
            raise ValueError('path points per segment exceed bounded manifest size')
        lock = versions()
        phonon = _to_phonopy(geom, params)
        q, settings, segments, labels, warnings = path_sampling(phonon.primitive, params)
        if len(q) > MAX_POINTS:
            raise ValueError('path sampling manifest too large')
        reciprocal = 2*np.pi*np.linalg.inv(phonon.primitive.cell).T
        reference_reciprocal = 2*np.pi*np.linalg.inv(geom['cell']).T
        base.update(qpoints=q.tolist(), segments=segments, labels=labels,
            reciprocal_basis_Ainv=reciprocal.tolist(), sampling_basis='phonopy_primitive',
            basis_to_reference_reciprocal=(reciprocal @ np.linalg.inv(reference_reciprocal)).tolist(),
            primitive=_cell_record(phonon.primitive), branch_count=3*len(phonon.primitive),
            generator={'name': 'seekpath.get_path_orig_cell', 'versions': lock,
                       'settings': settings, 'symprec_A': params['symprec_A']}, warnings=warnings)
    else:
        raise ValueError('direct phonon request must select one scope: path or mesh; combined tasks stage each independently')
    base['weights'] = [1]*len(base['qpoints'])
    base['qpoint_count'] = len(base['qpoints'])
    return sealed(base)


def request_sampling(geom, method, quantities, recipe):
    result = {}
    if {'eigenvalues', 'occupations'} & set(quantities):
        result['electronic'] = electronic_manifest(geom, method)
    if 'phonon_spectrum' in quantities:
        result['phonon'] = deepcopy(_phonon_manifest(json.dumps(geom, sort_keys=True),
                                                   json.dumps(recipe['parameters'], sort_keys=True)))
    return result


def electronic_diagnostics(manifest, kpoints):
    """Attach row identities only after matching actual parsed coordinates."""
    expected = np.asarray(manifest['kpoints'], float)
    indices = []
    for row in np.asarray(kpoints, float):
        delta = expected-row
        matches = np.flatnonzero(np.all(np.abs(delta-np.rint(delta)) <= 1e-8, axis=1))
        if len(matches) != 1:
            raise ValueError('k coordinate not in locked mesh')
        indices.append(int(matches[0]))
    return {'sampling_manifest_digest': manifest['manifest_digest'],
            'coordinate_convention': manifest['coordinate_convention'],
            'reciprocal_basis_Ainv': manifest['reciprocal_basis_Ainv'], 'kpoint_indices': indices}


def validate_electronic(observations, manifest):
    active = [o for o in observations if o['quantity'] in {'eigenvalues', 'occupations'} and o['value'] is not None]
    for o in active:
        info = o['diagnostics']
        if info.get('sampling') != 'uniform_mesh':
            raise ValueError('irreducible/path electronic sampling unsupported without verified full-grid reconstruction')
        for key in ('sampling_manifest_digest', 'coordinate_convention', 'reciprocal_basis_Ainv'):
            expected = manifest['manifest_digest'] if key == 'sampling_manifest_digest' else manifest[key]
            if info.get(key) != expected:
                raise ValueError('electronic sampling coverage/basis evidence missing or mismatched')
        count = len(manifest['kpoints'])
        indices = info.get('kpoint_indices', [])
        if any(type(i) is not int for i in indices) or sorted(indices) != list(range(count)):
            raise ValueError('electronic full mesh has missing/duplicate row identities')
        points = np.asarray(info.get('kpoints'), float)
        weights = np.asarray(info.get('weights'), float)
        if points.shape != (count, 3) or not np.isfinite(points).all() or len(o['value']) != count:
            raise ValueError('electronic k/array coverage mismatch')
        delta = points - np.asarray(manifest['kpoints'])[indices]
        if not np.all(np.abs(delta-np.rint(delta)) <= 1e-8):
            raise ValueError('electronic coordinate/row/shift mismatch')
        if weights.shape != (count,) or not np.allclose(weights, 1/count, atol=1e-10, rtol=1e-7):
            raise ValueError('full electronic mesh weights must be uniform and normalized')
    if len(active) == 2 and (active[0]['diagnostics'] != active[1]['diagnostics'] or
                            np.shape(active[0]['value']) != np.shape(active[1]['value'])):
        raise ValueError('eigenvalue/occupation row mapping mismatch')


def validate_phonon(value, diagnostics, manifest):
    if diagnostics.get('sampling_manifest_digest') != manifest['manifest_digest']:
        raise ValueError('phonon sampling manifest coverage evidence missing or mismatched')
    if diagnostics.get('coordinate_convention') != manifest['coordinate_convention'] or diagnostics.get('reciprocal_basis_Ainv') != manifest['reciprocal_basis_Ainv']:
        raise ValueError('phonon reciprocal basis mismatch')
    if value['scope'] != manifest['scope']:
        raise ValueError('phonon scope mismatch')
    q = np.asarray(value['qpoints'], float)
    expected = np.asarray(manifest['qpoints'], float)
    # Ordered path includes repeated segment endpoints. A set/count comparison
    # would erase segments and could assign frequencies to the wrong q rows.
    if q.shape != expected.shape or not np.allclose(q, expected, atol=1e-8, rtol=0):
        raise ValueError('phonon q rows do not cover locked sampling manifest')
    if value['weights'] != manifest['weights']:
        raise ValueError('phonon sampling weights mismatch')
    if np.shape(value['frequencies_THz']) != (len(expected), manifest['branch_count']):
        raise ValueError('phonon branch/q count differs from declared sampling basis')
