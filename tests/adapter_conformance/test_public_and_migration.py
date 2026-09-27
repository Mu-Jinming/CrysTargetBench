from pathlib import Path
import json
import pytest
from crystargetbench.api import evaluate
from crystargetbench.public_export import export_public
from crystargetbench.contracts import default_deployment, default_policy, load_json
from crystargetbench.planner import plan
from .helpers import structure, RESOURCE

def test_default_has_no_engine_and_mlff_remains_first():
    d=default_deployment()
    assert d['schema_version']=='ctb.deployment.v3' and d['default_dft_provider'] is None
    assert 'preferred_dft' not in default_policy()
    assert plan(load_json(RESOURCE.joinpath('tasks/phonon_path.json')))['physics_fidelity']=='mlff'
    blocked=plan(load_json(RESOURCE.joinpath('tasks/highk_screen.json')))
    assert blocked['status']=='blocked_configuration'
    assert all(r['family']=='dft' and r['provider'] is None for r in blocked['routes'].values())
    assert blocked['protocol_lock']['lock_resolved'] is False

def test_spacegroups_are_categories_with_full_denominator(tmp_path):
    bad=tmp_path/'invalid.cif';bad.write_text('invalid')
    result=evaluate([structure(),structure(),structure('ideal_bcc.cif'),bad],load_json(RESOURCE.joinpath('tasks/spacegroup_225.json')),output=tmp_path/'run')
    dist=result['metrics']['distributions']['sg']
    assert dist['histogram']=={'225':2,'229':1}
    assert 'mean' not in dist and 'median' not in dist
    assert result['metrics']['n_submitted']==4
    t=load_json(RESOURCE.joinpath('tasks/spacegroup_225.json'))
    t['mode']='constrained_rank';t['ranking']={'measurement':'sg','direction':'maximize'}
    ranked=evaluate([structure(),structure('ideal_bcc.cif')],t,output=tmp_path/'ranked')
    assert ranked['metrics']['ranking']['distribution']['histogram']=={'225':1}
    assert 'mean' not in ranked['metrics']['ranking']['distribution']

def test_recursive_redaction_preserves_original(tmp_path):
    value={'nested':[{'unrelated':{'schema_version':'ctb.live_permit.v1','enabled':True,'owner':'private'}},
                     {'session':{'secret':'never export'},'value':42,'path':'/'+'home/'+'private/weight.bin'}],
           'more':{'session_id':'secret-id','permit':{'enabled':True,'limits':{'x':1}}},
           'disabled':json.loads(Path('CTB_S2_Addendum/examples/MLFF_LIVE_PERMIT.disabled.json').read_text())}
    p=tmp_path/'fixture.json';p.write_text(json.dumps(value));before=p.read_bytes()
    audit=export_public(tmp_path,tmp_path/'public',['fixture.json'])
    clean=json.loads((tmp_path/'public/fixture.json').read_text())
    assert 'unrelated' not in clean['nested'][0]
    assert 'session' not in clean['nested'][1]
    assert clean['nested'][1]['value']==42 and 'private' not in clean['nested'][1]['path']
    assert clean['more']=={} and clean['disabled']['authorized'] is False
    assert p.read_bytes()==before and audit['redactions']
    with pytest.raises(ValueError):export_public(tmp_path,tmp_path/'out',['../escape'])
