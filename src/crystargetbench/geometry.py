"""Explicit calculation snapshots; derived cells never rerun input-domain heuristics."""
from .identity import digest


def snapshot_from_atoms(atoms, *, parent=None, stage='derived', transformation=None):
    import numpy as np
    cell = np.asarray(atoms.cell, float)
    positions = np.asarray(atoms.get_scaled_positions(wrap=False), float)
    masses = np.asarray(atoms.get_masses(), float)
    moments = np.asarray(atoms.get_initial_magnetic_moments(), float)
    if (not len(atoms) or cell.shape != (3,3) or positions.shape != (len(atoms),3)
            or not np.isfinite(cell).all() or not np.isfinite(positions).all()
            or not np.isfinite(masses).all() or (masses<=0).any()
            or not np.isfinite(moments).all() or abs(np.linalg.det(cell))<1e-10):
        raise ValueError('invalid calculation geometry')
    if not all(atoms.pbc):
        raise ValueError('calculation geometry requires three periodic dimensions')
    state={'cell':cell.tolist(),'positions':positions.tolist(),'coordinate_system':'fractional',
           'length_unit':'angstrom','species':atoms.get_chemical_symbols(),'occupancy':[1.0]*len(atoms),
           'pbc':[True,True,True],'magnetic_state':{'kind':'nonmagnetic','moments':moments.tolist()}}
    return {**state,'geometry_id':digest(state),'masses':masses.tolist(),'mass_unit':'amu',
            'mass_digest':digest(masses.tolist()),'parent':parent,'stage':stage,
            'transformation':transformation or {'kind':'identity','site_mapping':list(range(len(atoms))),
                                              'basis_mapping':[[1,0,0],[0,1,0],[0,0,1]]}}


def atoms_from_snapshot(snapshot):
    import numpy as np
    from ase import Atoms
    if snapshot.get('coordinate_system','fractional')!='fractional' or snapshot.get('length_unit','angstrom')!='angstrom':
        raise ValueError('unsupported geometry coordinate system or length unit')
    if any(v!=1 for v in snapshot.get('occupancy',[1]*len(snapshot['species']))):
        raise ValueError('calculation requires ordered full occupancy')
    atoms=Atoms(symbols=snapshot['species'],cell=snapshot['cell'],scaled_positions=snapshot['positions'],
                pbc=snapshot.get('pbc',[True]*3))
    if 'masses' in snapshot: atoms.set_masses(snapshot['masses'])
    moments=snapshot.get('magnetic_state',{}).get('moments')
    if moments is not None: atoms.set_initial_magnetic_moments(moments)
    state_keys=('cell','positions','coordinate_system','length_unit','species','occupancy','pbc','magnetic_state')
    if snapshot.get('geometry_id') and all(k in snapshot for k in state_keys):
        if digest({k:snapshot[k] for k in state_keys})!=snapshot['geometry_id']:
            raise ValueError('geometry content/hash mismatch')
    # Validate numeric shape without a representation-dependent vacuum cutoff.
    snapshot_from_atoms(atoms)
    return atoms
