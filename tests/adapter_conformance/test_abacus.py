from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from crystargetbench.geometry import snapshot_from_atoms, atoms_from_snapshot
from crystargetbench.dft.registry import load_adapter
from crystargetbench.dft.validation import make_request, validate_prepared, validate_result, artifact
from crystargetbench.dft.reducers import qualified, band_gap
from crystargetbench.physics import EV_PER_ANGSTROM3_TO_GPA
from crystargetbench.reporting import write_json
from .helpers import method, task

# Authored syntax fragments, not engine output or pseudopotentials.
# No test creates, obtains, loads or runs a real engine/asset.
def request(operation='static'):
    adapter,manifest=load_adapter('abacus_example')
    geom=snapshot_from_atoms(Atoms('SiOSi',scaled_positions=[[0,0,0],[.4,.5,.6],[.8,.8,.8]],cell=[[6,0,0],[.2,7,0],[.1,.3,8]],pbc=True))
    m=method('3.7.4')
    assets={s:{'path':None,'filename':s+'.upf','sha256':hashlib.sha256(('absent-user-asset:'+s).encode()).hexdigest()} for s in ['Si','O']}
    m['asset_fingerprints']=[{'kind':'pseudopotential','identifier':s+':'+a['filename'],'sha256':a['sha256']} for s,a in assets.items()]
    qs=['energy','forces','stress']
    if operation=='relaxation':qs+=['relaxed_geometry']
    if operation=='eigenvalues':qs=['eigenvalues','occupations']
    return adapter,manifest,make_request(manifest,operation,qs,geom,m),{'assets':assets}

def outputs(root,req,prepared,*,converged=True,force=0.0,stress=0.0,ionic=True):
    (root/'OUT.CTB').mkdir(exist_ok=True)
    running='running_cell-relax.log' if req['operation']=='relaxation' else 'running_scf.log'
    text=f'''ABACUS v3.7.4
energy cutoff for wavefunc (unit:Ry) = 80
nspin = 1
K-POINTS DIRECT COORDINATES
KPOINTS DIRECT_X DIRECT_Y DIRECT_Z WEIGHT
1 0.0 0.0 0.0 2.0

{'charge density convergence is achieved' if converged else 'charge density convergence is not achieved'}
TOTAL-FORCE (eV/Angstrom)
-------------------
Si1 {force} 0.0 0.0
Si2 {force*2} 0.0 0.0
O1 {force*3} 0.0 0.0
TOTAL-STRESS (KBAR)
-------------------
{stress} 0.0 0.0
0.0 {stress*2} 0.0
0.0 0.0 {stress*3}
TOTAL-PRESSURE: 0
!FINAL_ETOT_IS -123.25 eV
Finish Time : synthetic fixture, no engine was run
'''
    (root/'OUT.CTB'/running).write_text(text)
    (root/'OUT.CTB/STRU_FINAL').write_bytes((root/'STRU').read_bytes())
    (root/'OUT.CTB/istate.info').write_text('BAND Energy(ev) Occupation Kpoint = 1 (0 0 0)\n1 -4.0 2.0\n2 -1.0 2.0\n3 2.0 0.0\n')
    receipt={'schema_version':'ctb.abacus.receipt.v1','request_digest':req['request_digest'],
        'effective_method':req['method'],'input_files':prepared['files'],'evidence_kind':'synthetic',
        'final_structure':'OUT.CTB/STRU_FINAL','ionic_converged':ionic}
    write_json(root/'collection.json',receipt)

def test_prepare_order_units_missing_assets_and_no_solver(tmp_path):
    from ctb_abacus_example.adapter import read_stru
    a,m,r,c=request();p=validate_prepared(a.prepare(r,tmp_path,c),r,tmp_path)
    assert p['status']=='blocked_assets'
    assert not (tmp_path/'assets').exists()
    assert p['mapping']['output_to_input']==[0,2,1]
    restored=read_stru((tmp_path/'STRU').read_text(),r['geometry'],[0,2,1])
    np.testing.assert_allclose(restored['cell'],r['geometry']['cell'],atol=1e-12)
    np.testing.assert_allclose(restored['positions'],r['geometry']['positions'],atol=1e-12)
    assert 'calculation scf' in (tmp_path/'INPUT').read_text()
    assert 'relax_nmax' not in (tmp_path/'INPUT').read_text()

def test_efs_actual_text_parser_normalization(tmp_path):
    a,m,r,c=request();p=a.prepare(r,tmp_path,c);outputs(tmp_path,r,p,force=.1,stress=2)
    result=validate_result(a.collect(r,tmp_path,c),r,m,tmp_path);q=qualified(result)
    assert q['energy']['value']==-123.25
    np.testing.assert_allclose(q['forces']['value'],[[.1,0,0],[.3,0,0],[.2,0,0]])
    np.testing.assert_allclose(q['stress']['value'],np.diag([2,4,6])*(-.1/EV_PER_ANGSTROM3_TO_GPA))
    assert all(o['evidence_kind']=='synthetic' for o in result['observations'])

def test_relaxation_requires_scf_ionic_and_posterior_force_stress(tmp_path):
    a,m,r,c=request('relaxation');p=a.prepare(r,tmp_path,c)
    outputs(tmp_path,r,p);result=validate_result(a.collect(r,tmp_path,c),r,m,tmp_path)
    assert 'relaxed_geometry' in qualified(result)
    outputs(tmp_path,r,p,converged=False)
    assert not qualified(validate_result(a.collect(r,tmp_path,c),r,m,tmp_path))
    outputs(tmp_path,r,p,force=.1)
    assert 'relaxed_geometry' not in qualified(validate_result(a.collect(r,tmp_path,c),r,m,tmp_path))
    outputs(tmp_path,r,p,ionic=False)
    assert 'relaxed_geometry' not in qualified(validate_result(a.collect(r,tmp_path,c),r,m,tmp_path))

def test_gap_from_actual_istate_occupations_and_missing_output(tmp_path):
    a,m,r,c=request('eigenvalues');p=a.prepare(r,tmp_path,c);outputs(tmp_path,r,p)
    result=validate_result(a.collect(r,tmp_path,c),r,m,tmp_path)
    assert band_gap(qualified(result))==(3.0,[])
    path=tmp_path/'OUT.CTB/istate.info';path.write_text(path.read_text().replace('2 -1.0 2.0','2 -1.0 1.0'))
    assert band_gap(qualified(a.collect(r,tmp_path,c)))==(0.0,[])
    path.unlink()
    with pytest.raises(FileNotFoundError):a.collect(r,tmp_path,c)

def test_parser_rejects_versions_settings_mapping_and_static_relaxation(tmp_path):
    a,m,r,c=request();p=a.prepare(r,tmp_path,c);outputs(tmp_path,r,p)
    path=tmp_path/'OUT.CTB/running_scf.log';original=path.read_text()
    for replacement in [original.replace('3.7.4','4.0.0'),original.replace('= 80','= 90'),original.replace('Si2','Si3')]:
        path.write_text(replacement)
        with pytest.raises(ValueError):a.collect(r,tmp_path,c)
    path.write_text(original)
    path=tmp_path/'OUT.CTB/STRU_FINAL';path.write_text(path.read_text().replace('0.40000000000000002','0.45'))
    with pytest.raises(ValueError,match='geometry'):a.collect(r,tmp_path,c)

def test_capabilities_do_not_claim_highk_or_latest_dfpt(tmp_path):
    from crystargetbench.dft.exchange import plan_external
    a,m,r,c=request();s={'schema_version':'ctb.dft.site.v1','adapter':'abacus_example',
        'method':r['method'],'configuration':c,'execution_mode':'external','managed':None}
    t=task()
    for spec in t['measurements'].values():spec['required_fidelity']='dft'
    p=plan_external(t,s)
    assert p['status']=='blocked_capability'
    assert {'ionic_dielectric','electronic_dielectric'}<=set(p['missing_capabilities'])
    assert all(not c['live_tested'] for c in m['capabilities'])
