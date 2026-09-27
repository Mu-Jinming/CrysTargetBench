"""Thin ABACUS documented-format adapter. No solver imports or process launch.

The external operator supplies collection.json as an explicit reported receipt;
this is integrity/provenance evidence, not independent proof of engine execution.
Only the narrow format/profile documented in this distribution is accepted.
"""
from importlib.resources import files
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import re
import numpy as np
from ase import Atoms
from ase.units import Bohr
from crystargetbench.geometry import atoms_from_snapshot, snapshot_from_atoms
from crystargetbench.identity import digest
from crystargetbench.physics import EV_PER_ANGSTROM3_TO_GPA
from crystargetbench.dft.validation import (validate_request, artifact, verify_artifact, safe_path,
    read_record, exact, observation, result_record, identity_mapping, MAX_BYTES)

VERSION='3.7.4'
NUMBER=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?'

def text_file(root,name):
    path=safe_path(root,name)
    if path.stat().st_size>MAX_BYTES:raise ValueError('oversized ABACUS output')
    return path.read_text()

def mapping(geom):
    species=geom['species'];order=list(dict.fromkeys(species))
    return [i for symbol in order for i,s in enumerate(species) if s==symbol]

def write_stru(geom, assets):
    atoms=atoms_from_snapshot(geom);species=geom['species'];order=list(dict.fromkeys(species))
    rows=['ATOMIC_SPECIES']
    for symbol in order:
        i=species.index(symbol)
        rows.append(f'{symbol} {atoms.get_masses()[i]:.17g} {assets[symbol]["filename"]}')
    rows+=['','LATTICE_CONSTANT',f'{1/Bohr:.17g}','','LATTICE_VECTORS']
    rows+=[' '.join(f'{x:.17g}' for x in row) for row in atoms.cell.array]
    rows+=['','ATOMIC_POSITIONS','Direct']
    for symbol in order:
        ids=[i for i,s in enumerate(species) if s==symbol]
        rows += ['',symbol,'0',str(len(ids))]
        rows+=[' '.join(f'{x:.17g}' for x in geom['positions'][i])+' 1 1 1' for i in ids]
    return '\n'.join(rows)+'\n'

def read_stru(text, original, permutation):
    rows=[x.split('#')[0].strip() for x in text.splitlines()]
    rows=[x for x in rows if x]
    if any(rows.count(k)!=1 for k in ('ATOMIC_SPECIES','LATTICE_CONSTANT','LATTICE_VECTORS','ATOMIC_POSITIONS')):
        raise ValueError('ambiguous STRU sections')
    index=rows.index('LATTICE_CONSTANT');scale=float(rows[index+1])*Bohr
    index=rows.index('LATTICE_VECTORS');cell=np.array([[float(x) for x in row.split()] for row in rows[index+1:index+4]])*scale
    index=rows.index('ATOMIC_POSITIONS');mode=rows[index+1];index+=2
    if mode not in {'Direct','Cartesian'}:raise ValueError('unsupported STRU coordinates')
    species=[];positions=[]
    while index<len(rows):
        symbol=rows[index];mag=float(rows[index+1]);n=int(rows[index+2]);index+=3
        if mag!=0 or n<1:raise ValueError('only nonmagnetic complete STRU supported')
        for row in rows[index:index+n]:
            values=row.split()
            if len(values) not in (3,6):raise ValueError('unsupported STRU position flags')
            positions.append([float(x) for x in values[:3]]);species.append(symbol)
        index+=n
    if len(species)!=len(permutation) or species!=[original['species'][i] for i in permutation]:raise ValueError('STRU atom order mismatch')
    inverse=np.argsort(permutation);pos=np.array(positions)[inverse]
    atoms=atoms_from_snapshot(original);atoms.set_cell(cell,scale_atoms=False)
    if mode=='Direct':atoms.set_scaled_positions(pos)
    else:atoms.set_positions(pos*scale)
    return snapshot_from_atoms(atoms,parent=original['geometry_id'],stage='external_relaxed',
                               transformation={'kind':'external_relaxation','site_mapping':list(range(len(atoms))),'basis_mapping':np.eye(3,dtype=int).tolist()})

def profile(request,configuration):
    validate_request(request)
    exact(configuration,'assets','ABACUS configuration')
    method=request['method'];p=method['effective_parameters']
    if method['software']!=[{'name':'ABACUS','version':VERSION}]:raise ValueError('ABACUS version outside documented profile')
    supported={'xc','basis','k_grid','k_shift','cutoff_Ry','spin','soc','hubbard_u','occupations','scf_tolerance','force_tolerance_eV_A','stress_tolerance_GPa'}
    if set(p)!=supported or (p['xc'],p['basis'],p['spin'],p['soc'],p['hubbard_u'],p['occupations'])!=('PBE','pw',1,False,{},'fixed'):
        raise ValueError('unsupported ABACUS scientific profile: PW/PBE/nspin1/noSOC/noU/fixed occupations only')
    if request['operation'] not in {'static','eigenvalues','relaxation'}:raise ValueError('ABACUS response/phonon-solver capability unavailable')
    assets=configuration['assets'];species=set(request['geometry']['species'])
    if set(assets)!=species:raise ValueError('asset mapping must cover exactly the chemical species')
    expected=[]
    for symbol,a in assets.items():
        exact(a,'path filename sha256','ABACUS pseudopotential reference')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',a['filename']):raise ValueError('unsafe pseudopotential filename')
        expected.append({'kind':'pseudopotential','identifier':symbol+':'+a['filename'],'sha256':a['sha256']})
    if sorted(expected,key=lambda x:x['identifier'])!=sorted(method['asset_fingerprints'],key=lambda x:x['identifier']):raise ValueError('method asset fingerprints do not match site references')
    return p,assets

def input_parameters(request,p,assets):
    # Explicitly disable symmetry to preserve reference basis/atom identity and
    # make k-grid weights and finite displacements unambiguous.
    result={'suffix':'CTB','calculation':'cell-relax' if request['operation']=='relaxation' else 'scf',
        'basis_type':'pw','ntype':len(assets),'dft_functional':'PBE','nspin':1,'noncolin':0,'lspinorb':0,
        'symmetry':0,'ecutwfc':p['cutoff_Ry'],'scf_thr':p['scf_tolerance'],'scf_nmax':200,
        'occupations':'fixed','cal_force':1,'cal_stress':1,'out_stru':1,
        'force_thr_ev':p['force_tolerance_eV_A'],'stress_thr':p['stress_tolerance_GPa']*10,
        'pseudo_dir':'./assets','stru_file':'STRU','kpoint_file':'KPT'}
    if request['operation']=='relaxation':result.update(relax_nmax=200,relax_method='bfgs')
    return result

class Adapter:
    def describe(self):
        return json.loads(files('ctb_abacus_example').joinpath('ctb_dft_adapter.json').read_text())

    def prepare(self,request,directory,configuration):
        p,assets=profile(request,configuration)
        root=Path(directory);root.mkdir(parents=True,exist_ok=True)
        parameters=input_parameters(request,p,assets)
        (root/'INPUT').write_text('INPUT_PARAMETERS\n'+'\n'.join(f'{k} {v}' for k,v in parameters.items())+'\n')
        (root/'STRU').write_text(write_stru(request['geometry'],assets))
        (root/'KPT').write_text('K_POINTS\n0\nGamma\n'+' '.join(str(x) for x in p['k_grid']+p['k_shift'])+'\n')
        perm=mapping(request['geometry']);mapped=identity_mapping(len(perm));mapped['output_to_input']=perm
        (root/'mapping.json').write_text(json.dumps(mapped,sort_keys=True))
        # Assets remain user-managed. Never copy, download, create or fake them.
        reasons=[]
        for symbol,a in assets.items():
            path=Path(a['path']) if a['path'] else None
            staged=root/'assets'/a['filename']
            if path is None or not path.is_file():reasons.append('missing_user_asset:'+symbol)
            elif hashlib.sha256(path.read_bytes()).hexdigest()!=a['sha256']:reasons.append('asset_digest_mismatch:'+symbol)
            if not staged.is_file() or staged.is_symlink() or hashlib.sha256(staged.read_bytes()).hexdigest()!=a['sha256']:
                reasons.append('user_must_stage_verified_asset:assets/'+a['filename'])
        return {'schema_version':'ctb.dft.prepared.v1','request_id':request['request_id'],
            'request_digest':request['request_digest'],'status':'blocked_assets' if reasons else 'ready',
            'files':[artifact(root,n,'input') for n in ['INPUT','STRU','KPT','mapping.json']],
            'mapping':mapped,'reasons':reasons}

    def collect(self,request,directory,configuration):
        p,assets=profile(request,configuration);root=Path(directory)
        # Receipt is supplied by the external operator, not invented by this
        # parser. It binds input hashes, actual effective settings and output
        # geometry. It is explicitly reported, not an authenticated execution.
        receipt=read_record(root/'collection.json')
        exact(receipt,'schema_version request_digest effective_method input_files evidence_kind final_structure ionic_converged','ABACUS collection receipt')
        if receipt['schema_version']!='ctb.abacus.receipt.v1' or receipt['request_digest']!=request['request_digest'] or receipt['effective_method']!=request['method']:
            raise ValueError('ABACUS receipt method/request mismatch')
        if receipt['evidence_kind'] not in {'synthetic','reference','reported'}:raise ValueError('receipt cannot confer live verification')
        expected={'INPUT','STRU','KPT','mapping.json'}
        if {r['relative_path'] for r in receipt['input_files']}!=expected:raise ValueError('incomplete input hash proof')
        for ref in receipt['input_files']:verify_artifact(root,ref)
        # Input contents themselves must be the deterministic request expansion.
        if text_file(root,'STRU')!=write_stru(request['geometry'],assets):raise ValueError('input STRU changed')
        parsed={}
        for line in text_file(root,'INPUT').splitlines()[1:]:
            k,v=line.split(maxsplit=1)
            if k in parsed:raise ValueError('duplicate INPUT keyword')
            parsed[k]=v
        if parsed!={k:str(v) for k,v in input_parameters(request,p,assets).items()}:raise ValueError('INPUT effective settings differ')
        expected_kpt='K_POINTS\n0\nGamma\n'+' '.join(str(x) for x in p['k_grid']+p['k_shift'])+'\n'
        if text_file(root,'KPT')!=expected_kpt:raise ValueError('KPT changed')
        perm=mapping(request['geometry']);mp=identity_mapping(len(perm));mp['output_to_input']=perm
        if read_record(root/'mapping.json')!=mp:raise ValueError('atom mapping changed')
        name='OUT.CTB/running_cell-relax.log' if request['operation']=='relaxation' else 'OUT.CTB/running_scf.log'
        log=text_file(root,name)
        versions=re.findall(r'ABACUS\s+v([\w.+-]+)',log)
        if versions!=[VERSION]:raise ValueError('missing or incompatible actual engine version')
        cutoffs=re.findall(r'energy cutoff for wavefunc \(unit:Ry\)\s*=\s*('+NUMBER+')',log)
        spins=re.findall(r'\bnspin\s*=\s*(\d+)',log)
        if not cutoffs or any(float(x)!=p['cutoff_Ry'] for x in cutoffs) or not spins or any(int(x)!=1 for x in spins):
            raise ValueError('output effective cutoff/spin differs or is missing')
        # A terminal completion record alone does not establish SCF convergence.
        completed=bool(re.search(r'^\s*Finish\s+Time\s*:',log,re.M))
        scf=completed and 'charge density convergence is achieved' in log
        if re.search(r'not converg|convergence has NOT|convergence is not',log,re.I):scf=False
        refs=[artifact(root,'collection.json'),artifact(root,name)]+receipt['input_files']
        evidence=receipt['evidence_kind'];observations=[]
        def add(q,value,diag=None):
            if q in request['requested_quantities']:
                observations.append(observation(q,value,definition_id='abacus.3.7.4.documented.'+q,raw_artifacts=refs,
                    diagnostics={'settings_evidence':'external_report_and_input_log_checks',**(diag or {})},
                    evidence_kind=evidence,qualified=scf))
        energies=re.findall(r'^\s*!FINAL_ETOT_IS\s+('+NUMBER+r')\s+eV\s*$',log,re.M)
        add('energy',float(energies[-1]) if energies and scf else None)
        force_blocks=re.findall(r'TOTAL-FORCE \(eV/Angstrom\)(.*?)(?=TOTAL-STRESS|!FINAL_ETOT_IS|\Z)',log,re.S)
        forces=None
        if force_blocks and scf:
            rows=re.findall(r'^\s*([A-Z][a-z]?\d+)\s+('+NUMBER+r')\s+('+NUMBER+r')\s+('+NUMBER+r')\s*$',force_blocks[-1],re.M)
            counters={};labels=[]
            for i in perm:
                symbol=request['geometry']['species'][i];counters[symbol]=counters.get(symbol,0)+1;labels.append(symbol+str(counters[symbol]))
            if [row[0] for row in rows]!=labels:raise ValueError('force atom labels/order/count differ')
            forces=np.asarray([[float(x) for x in row[1:]] for row in rows])[np.argsort(perm)].tolist()
        add('forces',forces)
        stresses=re.findall(r'TOTAL-STRESS \(KBAR\)(.*?)(?=TOTAL-PRESSURE|!FINAL_ETOT_IS|\Z)',log,re.S)
        stress=None
        if stresses and scf:
            rows=re.findall(r'^\s*('+NUMBER+r')\s+('+NUMBER+r')\s+('+NUMBER+r')\s*$',stresses[-1],re.M)
            if len(rows)!=3:raise ValueError('incomplete stress tensor')
            stress=(-np.asarray(rows,dtype=float)*0.1/EV_PER_ANGSTROM3_TO_GPA).tolist()
        add('stress',stress,{'sign':'ASE_tensile_positive','raw_unit':'kbar_compressive_positive'})
        observed=None;ionic=None
        if receipt['final_structure']:
            observed=read_stru(text_file(root,receipt['final_structure']),request['geometry'],perm)
            refs.append(artifact(root,receipt['final_structure']))
            if request['operation']!='relaxation':
                original=atoms_from_snapshot(request['geometry']);final=atoms_from_snapshot(observed)
                if not np.allclose(original.cell.array,final.cell.array,rtol=0,atol=1e-9) or not np.allclose(original.positions,final.positions,rtol=0,atol=1e-9):
                    raise ValueError('static calculation changed geometry')
                observed=request['geometry']
        if request['operation']=='relaxation':
            ionic=bool(receipt['ionic_converged'] is True and scf and forces is not None and stress is not None and observed is not None
                and np.max(np.linalg.norm(forces,axis=1))<=p['force_tolerance_eV_A']
                and np.max(np.abs(stress))*EV_PER_ANGSTROM3_TO_GPA<=p['stress_tolerance_GPa'])
            add('relaxed_geometry',observed if ionic else None,{'ionic_convergence_evidence':'reported_plus_posterior_force_stress'})
        elif observed is None:
            # No geometry proof means no qualified values, even after SCF.
            scf=False
            for o in observations:o['quality_status']='unverified'
        if {'eigenvalues','occupations'} & set(request['requested_quantities']):
            bands,occ,diag=parse_bands(text_file(root,'OUT.CTB/istate.info'),log,p)
            from crystargetbench.dft.sampling import electronic_diagnostics
            diag.update(electronic_diagnostics(request['sampling']['electronic'],diag['kpoints']))
            refs.append(artifact(root,'OUT.CTB/istate.info'))
            add('eigenvalues',bands if scf else None,diag);add('occupations',occ if scf else None,diag)
        return result_record(request,self.describe(),observations,observed_geometry=observed,scf=scf,ionic=ionic,
            errors=[] if scf else ['incomplete_or_unconverged_or_unbound_output'])

def parse_bands(text,log,p):
    # This narrow profile uses uniform unreduced gamma meshes (symmetry=0).
    # ABACUS istate occupations include k weights; normalize with the actual
    # direct k-point table, whose printed weights sum to two for nspin=1.
    tables=re.findall(r'K-POINTS DIRECT COORDINATES\s*\n\s*KPOINTS\s+DIRECT_X\s+DIRECT_Y\s+DIRECT_Z\s+WEIGHT\s*\n((?:\s*\d+\s+'+NUMBER+r'\s+'+NUMBER+r'\s+'+NUMBER+r'\s+'+NUMBER+r'\s*\n)+)',log)
    if not tables:raise ValueError('actual k-point weight table missing')
    rows=[line.split() for line in tables[-1].strip().splitlines()]
    k=np.array([[float(v) for v in row[1:4]] for row in rows]);w=np.array([float(row[4]) for row in rows])
    if len(rows)!=np.prod(p['k_grid']) or (w<=0).any() or not np.isclose(w.sum(),2,atol=1e-6):raise ValueError('unsupported reduced/path k-point sampling or weights')
    w=w/2
    expected=np.array([[(a+p['k_shift'][0]/2)/p['k_grid'][0],(b+p['k_shift'][1]/2)/p['k_grid'][1],(c+p['k_shift'][2]/2)/p['k_grid'][2]] for a in range(p['k_grid'][0]) for b in range(p['k_grid'][1]) for c in range(p['k_grid'][2])])
    wrap=lambda values:sorted(tuple(np.round(row%1,7)) for row in values)
    if wrap(k)!=wrap(expected) or not np.allclose(w,1/len(w),atol=1e-6):raise ValueError('uniform full mesh not proven')
    headers=list(re.finditer(r'BAND\s+Energy\(ev\)\s+Occupation\s+Kpoint\s*=\s*(\d+)\s*\(([^)]+)\)',text))
    if [int(m.group(1)) for m in headers]!=list(range(1,len(k)+1)):raise ValueError('istate k-point mapping missing')
    bands=[];occ=[]
    for i,m in enumerate(headers):
        header_k=np.array([float(v) for v in m.group(2).split()])
        if not np.allclose(header_k,k[i],atol=1e-6):raise ValueError('istate/log kpoint mismatch')
        chunk=text[m.end():headers[i+1].start() if i+1<len(headers) else len(text)]
        rows=[line.split() for line in chunk.splitlines() if line.strip()]
        if not rows or any(len(row)!=3 or int(row[0])!=j+1 for j,row in enumerate(rows)):raise ValueError('malformed istate band table')
        bands.append([float(row[1]) for row in rows]);occ.append([float(float(row[2])/w[i]) for row in rows])
    if len({len(row) for row in bands})!=1:raise ValueError('incomplete band matrix')
    return bands,occ,{'sampling':'uniform_mesh','kpoints':k.tolist(),'weights':w.tolist(),'spin_channels':1,
        'maximum_occupation':2,'occupation_tolerance':1e-6,'occupation_convention':'per_state_unweighted'}
