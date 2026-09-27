from copy import deepcopy
import json
import socket
import subprocess
import pytest
import numpy as np
from crystargetbench.dft import registry,exchange
from crystargetbench.dft.validation import *
from crystargetbench.dft.reducers import reduce_property
from crystargetbench.public_export import export_public
from .helpers import *

def test_duplicate_and_incompatible_static_manifests_rejected(tmp_path,monkeypatch):
    from importlib.metadata import distribution
    dist=distribution('ctb-second-example')
    monkeypatch.setattr(registry.metadata,'distributions',lambda:[dist,dist])
    registry.refresh()
    with pytest.raises(ValueError,match='duplicate'):registry.discover()
    manifest=read_record(dist.locate_file('ctb_second_example/ctb_dft_adapter.json'));manifest['api_version']=999
    write_json(tmp_path/'bad.json',manifest)
    class Broken:
        entry_points=dist.entry_points;files=dist.files;metadata=dist.metadata;version=dist.version
        def locate_file(self,p):return tmp_path/'bad.json'
    monkeypatch.setattr(registry.metadata,'distributions',lambda:[Broken()])
    registry.refresh()
    with pytest.raises(ValueError,match='incompatible'):registry.discover()
    registry.refresh()

def test_describe_prepare_collect_have_no_process_or_network(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('unexpected process/network')
    monkeypatch.setattr(subprocess,'Popen',forbidden);monkeypatch.setattr(socket,'socket',forbidden)
    run=tmp_path/'run';exchange.prepare([structure()],task(),site=site(),output=run)
    assert finish(run)['metrics']['n_pass']==1

def test_optical_electronic_total_geometry_and_hybrid_quality(tmp_path):
    a,m=registry.load_adapter('second_example');g=load_submission([structure()])[0]['input_geometry']
    req=make_request(m,'electric_response',['electronic_dielectric','ionic_dielectric'],g,method())
    write_return(req,tmp_path);observations=a.collect(req,tmp_path,{})['observations'];q={o['quantity']:o for o in observations}
    assert reduce_property('dielectric_total_static_trace_over3',q,g,'synthetic')==(30.0,[])
    electronic=deepcopy(q);electronic.pop('ionic_dielectric')
    assert reduce_property('dielectric_total_static_trace_over3',electronic,g,'synthetic')[0] is None
    for field,value in [('frequency_eV',1.5),('local_field_effects',False),('reference_geometry_id','0'*64),('response_converged',False)]:
        bad=deepcopy(q);bad['electronic_dielectric']['diagnostics'][field]=value
        assert reduce_property('dielectric_total_static_trace_over3',bad,g,'synthetic')[0] is None
    bad=deepcopy(q)
    for o in bad.values():o['diagnostics']['upstream_fidelity']=['dft','mlff']
    assert reduce_property('dielectric_total_static_trace_over3',bad,g,'dft')[0] is None
    # Supplied total is used once; it is not added to its components again.
    total=deepcopy(q['electronic_dielectric']);total['quantity']='total_dielectric';total['value']=(np.eye(3)*12).tolist()
    total['diagnostics']['components']=['electronic','ionic'];q['total_dielectric']=total
    assert reduce_property('dielectric_total_static_trace_over3',q,g,'synthetic')==(12.0,[])

def test_unknown_prerequisite_does_not_expand_response(tmp_path):
    run=tmp_path/'run';exchange.prepare([structure()],task(relaxed=False),site=site(),output=run)
    state=read_record(run/'exchange.json');rid=state['pending'][0]
    write_return(state['requests'][rid]['request'],run/'requests'/rid,scf=False)
    result=exchange.collect(run)
    assert result['metrics']['n_unknown']==1
    assert len(read_record(run/'exchange.json')['requests'])==1
    assert result['results'][0]['measurements']['kappa']['value'] is None

def test_band_path_not_mesh_and_missing_bands_not_metal(tmp_path):
    from crystargetbench.dft.reducers import band_gap
    a,m=registry.load_adapter('second_example');g=load_submission([structure()])[0]['input_geometry']
    req=make_request(m,'eigenvalues',['eigenvalues','occupations'],g,method())
    write_return(req,tmp_path);q={o['quantity']:o for o in a.collect(req,tmp_path,{})['observations']}
    assert band_gap(q)[0]==3
    q['eigenvalues']['diagnostics']['sampling']='path'
    assert band_gap(q)[0] is None
    q.pop('occupations');assert band_gap(q)[0] is None

def test_recursive_yaml_and_text_export(tmp_path):
    (tmp_path/'nested.yaml').write_text('data:\n  - secret_layer:\n      schema_version: ctb.live_permit.v1\n      enabled: true\n      owner: private\n  - value: 42\n')
    (tmp_path/'nested.txt').write_text('share {"outer": [{"session": {"token": "private"}}, {"value": 42}]} end')
    export_public(tmp_path,tmp_path/'public',['nested.yaml','nested.txt'])
    assert 'private' not in (tmp_path/'public/nested.yaml').read_text()
    assert 'private' not in (tmp_path/'public/nested.txt').read_text()
    assert '42' in (tmp_path/'public/nested.txt').read_text()

def test_content_hash_requires_complete_geometry_and_bounded_records(tmp_path,monkeypatch):
    import crystargetbench.dft.validation as validation
    g=load_submission([structure()])[0]['input_geometry'];del g['magnetic_state']
    with pytest.raises(ValueError,match='complete'):geometry(g)
    (tmp_path/'large.json').write_text('[0, 1, 2, 3, 4]')
    monkeypatch.setattr(validation,'MAX_BYTES',5)
    with pytest.raises(ValueError,match='oversized'):read_record(tmp_path/'large.json')
    monkeypatch.setattr(validation,'MAX_ARRAY_ELEMENTS',4)
    with pytest.raises(ValueError,match='oversized'):validation.bounded([0,1,2,3,4])

def test_blocked_highk_has_all_items_unknown(tmp_path):
    from .test_abacus import request
    a,m,r,c=request()
    s=site();s.update(adapter='abacus_example',method=r['method'],configuration=c)
    out=exchange.prepare([structure(),structure()],task(),site=s,output=tmp_path/'run')
    assert out['plan']['status']=='blocked_capability' and out['metrics']['n_submitted']==2
    assert out['metrics']['n_unknown']==2 and out['run']['external_engine_invocations'] is None

def test_export_ci_yaml_on_key_is_preserved(tmp_path):
    (tmp_path/'workflow.yml').write_text('name: test\non: [push]\npermissions:\n  contents: read\n')
    export_public(tmp_path,tmp_path/'public',['workflow.yml'])
    assert (tmp_path/'public/workflow.yml').read_text()==(tmp_path/'workflow.yml').read_text()
