from copy import deepcopy
import json
import sys
from pathlib import Path
import pytest
from crystargetbench.dft import exchange
from crystargetbench.dft.registry import discover, load_adapter
from crystargetbench.dft.validation import validate_result, read_record, safe_path, make_request, validate_method
from crystargetbench.identity import digest
from crystargetbench.structures import load_submission
from crystargetbench.reporting import write_json
from .helpers import *

def test_installed_plugins_are_separate_distributions():
    from importlib.metadata import distribution
    for provider,dist in [('second_example','ctb-second-example'),('abacus_example','ctb-abacus-example')]:
        assert discover()[provider]['distribution']==dist
        assert distribution(dist).version=='0.1.1'
    assert 'tests' not in str(load_adapter('second_example')[0].__class__)

def test_metadata_discovery_does_not_import_plugins():
    import subprocess
    code="from crystargetbench.dft.registry import discover; import sys; d=discover(); assert len(d)>=2; assert 'ctb_abacus_example.adapter' not in sys.modules; assert 'ctb_second_example.adapter' not in sys.modules; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True)

def test_staged_relaxation_full_denominator_idempotency(tmp_path):
    bad=tmp_path/'bad.cif';bad.write_text('not a structure')
    run=tmp_path/'run'
    first=exchange.prepare([structure(),structure(),bad],task(),site=site(),output=run)
    state=read_record(run/'exchange.json')
    assert first['metrics']['n_submitted']==3 and first['metrics']['n_unknown']==3
    assert len(state['pending'])==1
    assert {r['request']['operation'] for r in state['requests'].values()}=={'relaxation'}
    assert all(x['reference_geometry'] is None for x in first['results'])
    # The final structure genuinely differs from the submitted geometry.
    from crystargetbench.geometry import snapshot_from_atoms,atoms_from_snapshot
    rid=state['pending'][0];req=state['requests'][rid]['request'];original_id=req['geometry']['geometry_id']
    write_return(req,run/'requests'/rid)
    path=run/'requests'/rid/'authored-observations.json';wire=read_record(path)
    atoms=atoms_from_snapshot(req['geometry']);atoms.set_cell(atoms.cell.array*1.01,scale_atoms=True)
    final=snapshot_from_atoms(atoms,parent=original_id)
    wire['observed_geometry']=final;wire['observations']['relaxed_geometry']['value']=final;write_json(path,wire)
    exchange.collect(run)
    result=finish(run)
    assert result['results'][0]['reference_geometry_id']==final['geometry_id']!=original_id
    assert all(e['request']['sample_geometry_id']==original_id for e in read_record(run/'exchange.json')['requests'].values())
    assert result['metrics']['n_submitted']==3
    assert result['metrics']['n_pass']==2 and result['metrics']['n_unknown']==1
    assert result['results'][0]['measurements']['kappa']['value']==30
    assert result['run']['ctb_engine_invocations']==0 and result['run']['external_wall_seconds'] is None
    repeated=exchange.collect(run)
    assert repeated['metrics']==result['metrics']
    assert all(not m['benchmark_eligible'] for m in result['results'][0]['measurements'].values())

def test_r1_order_and_fail_gate_and_threshold_reassessment(tmp_path):
    outputs=[]
    for index,reverse in enumerate([False,True]):
        t=task(relaxed=False)
        if reverse:t['measurements']=dict(reversed(list(t['measurements'].items())))
        run=tmp_path/str(index);exchange.prepare([structure()],t,site=site(),output=run)
        result=finish(run,gap=1.0);outputs.append(result)
        state=read_record(run/'exchange.json')
        assert len(state['requests'])==1
        assert result['metrics']['n_fail']==1
        assert result['results'][0]['measurements']['kappa']['value'] is None
    assert outputs[0]['metrics']==outputs[1]['metrics']
    t=task(relaxed=False);t['constraints'][1]['value']=0.5
    reass=exchange.reassess(tmp_path/'0',t,output=tmp_path/'changed')
    assert reass['metrics']['n_unknown']==1 and reass['ctb_engine_invocations']==0

def test_rank_without_kappa_cutoff(tmp_path):
    run=tmp_path/'run';exchange.prepare([structure()],task(rank=True),site=site(),output=run)
    result=finish(run,kappa=4)
    assert result['metrics']['n_rankable']==1
    assert 'verified_yield' not in result['metrics']
    assert result['metrics']['ranking']['items'][0]['value']==4

def test_incomplete_batch_and_conflict(tmp_path):
    run=tmp_path/'run';exchange.prepare([structure()],task(),site=site(),output=run)
    assert exchange.collect(run)['metrics']['n_unknown']==1
    state=read_record(run/'exchange.json');rid=state['pending'][0];request=state['requests'][rid]['request']
    write_return(request,run/'requests'/rid)
    exchange.collect(run)
    p=run/'requests'/rid/'authored-observations.json';data=read_record(p);data['observations']['energy']['value']=-2;write_json(p,data)
    with pytest.raises(ValueError,match='conflicting'):exchange.collect(run)

def test_synthetic_does_not_satisfy_dft_requirement(tmp_path):
    t=task(relaxed=False)
    for spec in t['measurements'].values():spec['required_fidelity']='dft'
    run=tmp_path/'run';exchange.prepare([structure()],t,site=site(),output=run)
    result=finish(run)
    assert result['metrics']['n_unknown']==1 and result['metrics']['n_pass']==0

def test_comparison_independent_of_input_labels_thresholds_change_only_comparison():
    a=exchange.plan_external(task(),site());t=task();t['constraints'][1]['value']=4
    b=exchange.plan_external(t,site())
    assert a['protocol_digest']==b['protocol_digest']
    assert a['comparability_digest']!=b['comparability_digest']

def test_external_collect_does_not_open_old_permit(tmp_path,monkeypatch):
    from crystargetbench.budget import Budget
    monkeypatch.setattr(Budget,'__init__',lambda *a,**k:(_ for _ in ()).throw(AssertionError('managed ledger read')))
    run=tmp_path/'run';exchange.prepare([structure()],task(),site=site(),output=run)
    assert finish(run)['metrics']['n_pass']==1

def test_path_binding_and_array_rejections(tmp_path):
    adapter,manifest=load_adapter('second_example');g=load_submission([structure()])[0]['input_geometry']
    req=make_request(manifest,'static',['energy','forces'],g,method())
    write_return(req,tmp_path);valid=adapter.collect(req,tmp_path,{})
    validate_result(valid,req,manifest,tmp_path)
    for field,value in [('request_digest','0'*64),('reference_geometry_id','0'*64),('method_digest','0'*64)]:
        bad=deepcopy(valid);bad[field]=value
        with pytest.raises(ValueError):validate_result(bad,req,manifest,tmp_path)
    for value in [float('nan'),float('inf')]:
        bad=deepcopy(valid);bad['observations'][0]['value']=value
        with pytest.raises(ValueError):validate_result(bad,req,manifest,tmp_path)
    for name in ['../evil','/tmp/evil','a/../../evil','a\\evil']:
        with pytest.raises(ValueError):safe_path(tmp_path,name)
    (tmp_path/'link').symlink_to(tmp_path/'authored-observations.json')
    with pytest.raises(ValueError):safe_path(tmp_path,'link')
    bad=deepcopy(valid);bad['observations'][0]['raw_artifacts'][0]['size_bytes']+=1
    with pytest.raises(ValueError):validate_result(bad,req,manifest,tmp_path)
    bad=deepcopy(valid);bad['mapping']['output_to_input']=[0]*len(g['species'])
    if len(g['species'])>1:
        with pytest.raises(ValueError):validate_result(bad,req,manifest,tmp_path)

def test_task_cannot_import_or_grant_capability(tmp_path):
    t=task();t['import']='evil'
    with pytest.raises(ValueError):exchange.prepare([structure()],t,site=site(),output=tmp_path/'a')
    s=site();s['capabilities']=['electric_response']
    with pytest.raises(ValueError):exchange.prepare([structure()],task(),site=s,output=tmp_path/'b')
    s=site();s['method']['effective_parameters']['xc']='auto'
    assert exchange.plan_external(task(),s)['status']=='blocked_configuration'

def test_cli_block_status_and_explicit_policy_site_conflict(tmp_path,capsys):
    from crystargetbench.cli import main
    from crystargetbench.api import evaluate
    from crystargetbench.contracts import default_policy
    from crystargetbench.planner import plan
    from .test_abacus import request
    adapter,manifest,req,config=request()
    s=site();s.update(adapter='abacus_example',method=req['method'],configuration=config)
    write_json(tmp_path/'site.json',s);write_json(tmp_path/'task.json',task())
    assert main(['dft','prepare','--structures',str(structure()),'--task',str(tmp_path/'task.json'),
                 '--site',str(tmp_path/'site.json'),'--output',str(tmp_path/'blocked')])==3
    assert json.loads(capsys.readouterr().out)['metrics']['n_unknown']==1
    policy=default_policy('dft');policy['preferred_dft']='qe'
    with pytest.raises(ValueError,match='conflict'):evaluate([structure()],task(),deployment=s,policy=policy,output=tmp_path/'bad')
    with pytest.raises(ValueError,match='conflict'):plan(task(),deployment=s,policy=policy)

def test_managed_preparation_does_not_embed_permit_and_collection_needs_none(tmp_path,monkeypatch):
    from crystargetbench.budget import Budget
    s=site();s['execution_mode']='managed_local'
    s['managed']={'enabled':True,'authorization_id':'private-test-identity-not-for-export'}
    run=tmp_path/'run';exchange.prepare([structure()],task(),site=s,output=run)
    state=read_record(run/'exchange.json')
    assert state['site']['managed'] is None
    assert 'private-test-identity-not-for-export' not in (run/'exchange.json').read_text()
    monkeypatch.setattr(Budget,'__init__',lambda *a,**k:(_ for _ in ()).throw(AssertionError('collection requested old permit')))
    result=finish(run)
    assert result['metrics']['n_pass']==1 and result['run']['managed_ledger_modified'] is False

def test_input_phonon_requires_stationary_efs(tmp_path):
    t=task(relaxed=False);run=tmp_path/'run'
    exchange.prepare([structure()],t,site=site(),output=run)
    state=read_record(run/'exchange.json');rid=state['pending'][0]
    write_return(state['requests'][rid]['request'],run/'requests'/rid)
    exchange.collect(run)
    state=read_record(run/'exchange.json');rid=state['pending'][0];r=state['requests'][rid]['request']
    assert set(r['requested_quantities'])=={'phonon_spectrum','forces','stress'}
    write_return(r,run/'requests'/rid)
    p=run/'requests'/rid/'authored-observations.json';data=read_record(p)
    data['observations']['forces']['value'][0][0]=1.0;write_json(p,data)
    result=exchange.collect(run)
    assert result['metrics']['n_unknown']==1
    assert result['results'][0]['measurements']['phonon']['value'] is None
    assert result['run']['pending_request_ids']==[]
