"""Explicit input parsing and conservative, auditable ordered 3D domain checks."""
from pathlib import Path
import warnings

from .identity import digest, file_digest


class UnsupportedStructure(ValueError):
    pass


def input_files(structures):
    if isinstance(structures, (str, Path)):
        structures = [structures]
    files = []
    for source in structures:
        path = Path(source)
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.is_file()))
        else:
            files.append(path)
    return files


def read_structure(path):
    import numpy as np
    from ase import Atoms
    from ase.io import read
    from .contracts import load_json

    path = Path(path)
    if path.suffix.lower() == '.json':
        data = load_json(path)
        allowed = {'cell', 'scaled_positions', 'species', 'occupancy', 'pbc', 'magnetic_moments'}
        if set(data) - allowed or not {'cell', 'scaled_positions', 'species'} <= set(data):
            raise ValueError('structure JSON requires cell, scaled_positions and species; unknown fields rejected')
        def no_bool(obj):
            if isinstance(obj, bool):
                raise ValueError('boolean is not a structural number')
            if isinstance(obj, list):
                for v in obj:
                    no_bool(v)
        for key in ('cell', 'scaled_positions', 'occupancy', 'magnetic_moments'):
            if key in data:
                no_bool(data[key])
        n = len(data['species'])
        occupancy = data.get('occupancy', [1.0] * n)
        if len(occupancy) != n or any(type(x) not in (int, float) or x != 1 for x in occupancy):
            raise UnsupportedStructure('ordered full occupancy required')
        pbc = data.get('pbc', [True, True, True])
        if not isinstance(pbc, list) or len(pbc) != 3 or any(type(x) is not bool for x in pbc):
            raise ValueError('pbc must contain exactly three booleans')
        atoms = Atoms(symbols=data['species'], cell=data['cell'], scaled_positions=data['scaled_positions'], pbc=pbc)
        if 'magnetic_moments' in data:
            atoms.set_initial_magnetic_moments(data['magnetic_moments'])
    elif path.suffix.lower() == '.cif':
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            frames = read(str(path), format='cif', index=':', store_tags=True)
        if len(frames) != 1:
            raise ValueError('one structure per submission file is required')
        atoms = frames[0]
        raw_occupancy = atoms.info.get('_atom_site_occupancy', [])
        if any(float(v) != 1 for v in raw_occupancy):
            raise UnsupportedStructure('partial occupancy or disorder is outside input domain')
        for site in atoms.info.get('occupancy', {}).values():
            if len(site) != 1 or next(iter(site.values())) != 1:
                raise UnsupportedStructure('mixed or partial occupancy is outside input domain')
    else:
        raise ValueError('supported structure formats: .cif and CTB structure .json')
    cell = np.asarray(atoms.cell, dtype=float)
    pos = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    if not len(atoms) or cell.shape != (3, 3) or pos.shape != (len(atoms), 3):
        raise ValueError('empty or malformed geometry')
    if not np.isfinite(cell).all() or not np.isfinite(pos).all() or abs(np.linalg.det(cell)) < 1e-10:
        raise ValueError('nonfinite or singular cell/coordinates')
    if not all(atoms.pbc):
        raise UnsupportedStructure('3D periodic boundary conditions required')
    if any(int(z) <= 0 for z in atoms.numbers):
        raise UnsupportedStructure('all species must be recognized elements')
    # An explicit conservative rule; PBC flags alone are insufficient.
    # Reject a periodic empty slab wider than 10 A along any cell reciprocal normal.
    heights = 1.0 / np.linalg.norm(np.linalg.inv(cell), axis=0)
    wrapped = pos % 1.0
    max_gaps = []
    for axis in range(3):
        coords = np.sort(wrapped[:, axis])
        gaps = np.diff(np.r_[coords, coords[0] + 1.0]) * heights[axis]
        max_gaps.append(float(gaps.max()))
    if max(max_gaps) > 10.0:
        raise UnsupportedStructure('vacuum_guard.v1: periodic empty slab exceeds 10 angstrom; '
            'candidate input-domain heuristic, basis-dependent; not a stability or physical dimensionality label')
    moments = atoms.get_initial_magnetic_moments()
    if not np.isfinite(moments).all():
        raise ValueError('nonfinite magnetic state')
    if np.any(moments != 0):
        raise UnsupportedStructure('magnetic space groups require a separate protocol')
    state = {
        'cell': cell.tolist(), 'positions': pos.tolist(), 'coordinate_system': 'fractional',
        'length_unit': 'angstrom', 'species': atoms.get_chemical_symbols(),
        'occupancy': [1.0] * len(atoms), 'pbc': [True, True, True],
        'magnetic_state': {'kind': 'nonmagnetic', 'moments': moments.tolist()},
    }
    return {
        **state, 'geometry_id': digest(state), 'parent': None, 'stage': 'input',
        'transformation': {'kind': 'identity', 'site_mapping': list(range(len(atoms))), 'basis_mapping': [[1,0,0],[0,1,0],[0,0,1]]},
        'domain_checks': {'rule': 'ordered_3d.v1', 'maximum_empty_slab_A': max_gaps, 'vacuum_limit_A': 10.0,
            'heuristic_id': 'vacuum_guard.v1', 'scope': 'original_submission_only',
            'physical_dimensionality_classification': False, 'stability_evidence': False,
            'origin_translation_invariant': True, 'axis_permutation_invariant': True,
            'diagonal_replication_invariant': True, 'general_unimodular_basis_invariance_guaranteed': False,
            'representation_notice': 'Projection gaps use supplied reciprocal cell normals; general basis changes '
                'can change admission. Derived calculation snapshots bypass this heuristic and use explicit resource budgets.'},
    }


def load_submission(structures):
    items = []
    for index, path in enumerate(input_files(structures)):
        item = {'item_id': f'item-{index:06d}', 'source': path.name, 'file_sha256': None,
                'input_status': 'invalid', 'status': 'invalid', 'input_geometry': None,
                'atom_count': None, 'reasons': []}
        try:
            item['file_sha256'] = file_digest(path.read_bytes())
            item['input_geometry'] = read_structure(path)
            item['atom_count'] = len(item['input_geometry']['species'])
            item['input_status'] = item['status'] = 'valid'
        except UnsupportedStructure as exc:
            item['input_status'] = item['status'] = 'unsupported'
            item['reasons'] = [str(exc)]
        except (ValueError, OSError, RuntimeError, IndexError, KeyError, AssertionError, TypeError, Warning) as exc:
            item['reasons'] = [f'{type(exc).__name__}: {exc}']
        items.append(item)
    return items


def submission_digest(items):
    return digest([{'item_id': x['item_id'], 'file_sha256': x['file_sha256'], 'status': x['input_status']} for x in items])
