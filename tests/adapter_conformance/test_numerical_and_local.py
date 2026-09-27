from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import threading
import time
import numpy as np
import pytest
from ase import Atoms
from crystargetbench.geometry import snapshot_from_atoms, atoms_from_snapshot
from crystargetbench.dft.phonons import displacement_requests, collect_phonons, nac_inputs
from crystargetbench.dft.validation import observation,result_record,artifact,make_request
from crystargetbench.dft.registry import load_adapter
from crystargetbench.dft.local import run_local
from crystargetbench.budget import Budget, BudgetExceeded, DEFAULT_TEST_LIMITS
from crystargetbench.reporting import write_json
from crystargetbench.identity import digest
from .helpers import method,site

def test_returned_analytic_forces_use_real_phonopy_and_reuse_ifc(tmp_path):
    a,m=load_adapter('second_example')
    ref=snapshot_from_atoms(Atoms('Ne2',scaled_positions=[[0,0,0],[.5,.5,.5]],cell=np.eye(3)*5,pbc=True))
    params={'supercell_matrix':np.eye(3,dtype=int).tolist(),'primitive_matrix':np.eye(3).tolist(),
        'mesh':[2,2,2],'is_symmetry':False,'is_plusminus':True,'is_diagonal':False}
    plan=displacement_requests(ref,m,method(),params)
    base=atoms_from_snapshot(plan['requests'][0]['geometry']).positions
    for req in plan['requests']:
        assert req['operation']=='static'
        delta=atoms_from_snapshot(req['geometry']).positions-base
        # Analytic two-site spring, translationally invariant; no material claim.
        forces=-2*(delta-delta.mean(axis=0))
        d=tmp_path/'returned'/req['request_id'];d.mkdir(parents=True)
        (d/'analytic.json').write_text('Authored harmonic spring forces; analytic_test, not DFT.')
        o=observation('forces',forces.tolist(),definition_id='analytic_test.pair_spring',raw_artifacts=[artifact(d,'analytic.json')],evidence_kind='synthetic')
        write_json(d/'result.json',result_record(req,m,[o],observed_geometry=req['geometry'],scf=True,evidence_mode='fixture'))
    first=collect_phonons(plan,tmp_path/'returned',tmp_path/'cache')
    second=collect_phonons(plan,tmp_path/'returned',tmp_path/'cache',sampling={'mesh':[3,3,3]})
    assert first['ctb_engine_invocations']==0
    assert first['ifc']['force_constants_digest']==second['ifc']['force_constants_digest']
    assert second['cache_hits']>len(plan['requests'])
    assert first['sampling']['artifact_digest']!=second['sampling']['artifact_digest']
    assert first['ifc']['force_constants_unit']=='eV/angstrom^2'
    assert first['ifc']['physics_fidelity']=='synthetic'
    assert np.max(first['sampling']['frequencies_THz'])>0
    assert abs(np.min(first['sampling']['frequencies_THz']))<1e-5
    write_json(tmp_path/'numerical.json',first)

def test_nac_never_accepts_total_and_mixed_ifc_is_hybrid():
    g='a'*64;diag={'reference_geometry_id':g,'upstream_fidelity':['dft']}
    b=observation('born_charges',[np.eye(3).tolist()],definition_id='born',raw_artifacts=[],diagnostics=diag)
    e=observation('electronic_dielectric',np.eye(3).tolist(),definition_id='electronic',raw_artifacts=[],diagnostics=diag)
    assert nac_inputs(b,e,ifc_fidelity='mlff',reference_geometry_id=g)['physics_fidelity']=='mixed'
    e['quantity']='total_dielectric'
    with pytest.raises(ValueError,match='electronic-only'):nac_inputs(b,e,ifc_fidelity='dft',reference_geometry_id=g)

def local_case(tmp_path,code='print("synthetic launcher test; no engine")',wall=5):
    a,m=load_adapter('second_example');s=site();s['execution_mode']='managed_local'
    limits={**DEFAULT_TEST_LIMITS,'max_dft_jobs':2,'max_total_wall_seconds':30,
            'max_sample_wall_seconds':20,'per_request_wall_seconds':wall,'max_retries':0}
    s['managed']={'enabled':True,'authorization_id':'synthetic-python-standin-only',
        'method_digest':digest(s['method']),'adapter_id':m['adapter_id'],'argv':[sys.executable,'-c',code],
        'executable_sha256':hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),'limits':limits,'ledger_root':str(tmp_path/'ledgers')}
    geom=snapshot_from_atoms(Atoms('Ne',positions=[[0,0,0]],cell=np.eye(3)*5,pbc=True))
    req=make_request(m,'static',['energy'],geom,s['method'],execution_mode='managed_local')
    d=tmp_path/'job';d.mkdir()
    write_json(d/'request.json',req);write_json(d/'prepared.json',a.prepare(req,d,{}))
    return d,s

def test_local_shell_false_attempt_and_persistent_deadline(tmp_path):
    d,s=local_case(tmp_path,code='import sys; print(sys.argv[1])')
    s['managed']['argv'].append('$(not_a_shell); literal')
    out=run_local(d,s,tmp_path/'ledgers')
    assert out['status']=='process_completed' and out['shell'] is False
    assert (d/'launcher.stdout').read_text().strip()=='$(not_a_shell); literal'
    ledger=next((tmp_path/'ledgers').glob('*/budget.json'));before=ledger.read_text()
    second=run_local(d,s,tmp_path/'ledgers')
    import json
    prior=json.loads(before);after=json.loads(ledger.read_text())
    assert after['samples']==prior['samples'] and after['reserved_evaluations']==1
    assert out==second
    # A derived geometry belongs to the same submitted sample and must not
    # receive a new deadline at the next property/relaxation stage.
    a,m=load_adapter('second_example')
    req=json.loads((d/'request.json').read_text())
    atoms=atoms_from_snapshot(req['geometry']);atoms.set_cell(atoms.cell.array*1.01,scale_atoms=True)
    derived=snapshot_from_atoms(atoms,parent=req['geometry']['geometry_id'])
    next_request=make_request(m,'static',['energy'],derived,s['method'],execution_mode='managed_local',sample_geometry_id=req['sample_geometry_id'])
    next_dir=tmp_path/'derived';next_dir.mkdir()
    write_json(next_dir/'request.json',next_request);write_json(next_dir/'prepared.json',a.prepare(next_request,next_dir,{}))
    run_local(next_dir,s,tmp_path/'ledgers')
    resumed=json.loads(ledger.read_text())
    assert resumed['samples']==prior['samples'] and resumed['reserved_evaluations']==2

def test_local_timeout_consumes_attempt_and_cannot_retry(tmp_path):
    d,s=local_case(tmp_path,code='import time; time.sleep(5)',wall=1)
    with pytest.raises(BudgetExceeded):run_local(d,s,tmp_path/'ledgers')
    with pytest.raises(BudgetExceeded,match='retries'):run_local(d,s,tmp_path/'ledgers')
    import json
    ledger=json.loads(next((tmp_path/'ledgers').glob('*/budget.json')).read_text())
    assert ledger['reserved_evaluations']==1 and ledger['failed_evaluations']==1

def test_local_cancellation_and_disabled_template(tmp_path):
    d,s=local_case(tmp_path,code='import time; time.sleep(5)')
    disabled=deepcopy(s);disabled['managed']['enabled']=False
    with pytest.raises(ValueError,match='disabled'):run_local(d,disabled,tmp_path/'ledgers')
    def cancel():
        while not list((tmp_path/'ledgers').glob('*/budget.json')):time.sleep(.01)
        path=next((tmp_path/'ledgers').glob('*/budget.json'))
        time.sleep(.1)
        budget=Budget(path,s['managed']['limits'],identity=path.parent.name,execution_kind='managed_local_dft')
        budget.cancel()
    thread=threading.Thread(target=cancel);thread.start()
    with pytest.raises(BudgetExceeded,match='cancelled'):run_local(d,s,tmp_path/'ledgers')
    thread.join(timeout=5);assert not thread.is_alive()

def test_force_return_directory_cannot_escape_via_symlink(tmp_path):
    from crystargetbench.dft.phonons import _ForceContext
    from crystargetbench.cache import StageCache
    root=tmp_path/'returned';root.mkdir();outside=tmp_path/'unrelated';outside.mkdir()
    (root/('a'*64)).symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        _ForceContext({'requests':[{'request_id':'a'*64}]},root,StageCache(tmp_path/'cache'))

def test_new_output_directory_does_not_renew_same_permit_sample(tmp_path,monkeypatch):
    import json,shutil
    import crystargetbench.budget as budget_module
    d,s=local_case(tmp_path)
    run_local(d,s)
    ledger=next((tmp_path/'ledgers').glob('*/budget.json'));before=json.loads(ledger.read_text())
    fresh=tmp_path/'other-output';fresh.mkdir()
    for name in ['request.json','prepared.json','fixture-input.json']:shutil.copyfile(d/name,fresh/name)
    with pytest.raises(ValueError,match='ledger directory'):run_local(fresh,s,tmp_path/'another-ledger')
    monkeypatch.setattr(budget_module.time,'time',lambda:before['last_observed_wall']+21)
    monkeypatch.setattr(budget_module.time,'monotonic',lambda:before['last_observed_monotonic']+21)
    with pytest.raises(BudgetExceeded,match='max_sample_wall_seconds'):run_local(fresh,s)
    after=json.loads(ledger.read_text())
    assert after['samples']==before['samples'] and after['reserved_evaluations']==1
    assert not (fresh/'launcher.stdout').exists()
