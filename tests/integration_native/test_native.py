from copy import deepcopy
from importlib.resources import files
import json
from pathlib import Path

import pytest

from crystargetbench.api import evaluate
from crystargetbench.contracts import load_json
from crystargetbench.structures import load_submission

pytestmark = pytest.mark.native
RESOURCE = files('crystargetbench').joinpath('resources', 'examples')


def task():
    return load_json(RESOURCE.joinpath('tasks', 'spacegroup_225.json'))


def test_real_coordinate_pipeline_duplicates_invalid_and_reassessment(tmp_path):
    fcc = RESOURCE.joinpath('structures','ideal_fcc.cif')
    bcc = RESOURCE.joinpath('structures','ideal_bcc.cif')
    broken = tmp_path/'broken.cif'
    broken.write_text('this is not a crystal')
    data = evaluate([fcc,bcc,fcc,broken], task(), output=tmp_path/'first', cache_root=tmp_path/'cache')
    assert [r['measurements']['sg']['value'] for r in data['results']] == [225,229,225,None]
    assert [r['decision'] for r in data['results']] == ['pass','fail','pass','unknown']
    assert data['metrics']['n_submitted'] == 4
    assert data['metrics']['verified_yield']['value'] == .5
    assert data['run']['native_calls'] == 2
    assert data['run']['cache_hits'] == 1
    assert data['run']['physics_calls'] == 0
    assert data['run']['input_status_counts']['invalid'] == 1
    assert data['metrics']['benchmark_eligible'] is False
    changed = task()
    changed['constraints'][0]['value'] = 229
    again = evaluate([fcc,bcc,fcc,broken], changed, output=tmp_path/'second', cache_root=tmp_path/'cache')
    assert again['run']['native_calls'] == 0
    assert again['run']['cache_hits'] == 3
    assert [r['decision'] for r in again['results']] == ['fail','pass','fail','unknown']
    assert again['plan']['comparability_digest'] != data['plan']['comparability_digest']
    assert again['run']['submission_digest'] == data['run']['submission_digest']


def test_filename_and_declared_labels_are_not_measurements(tmp_path):
    source = RESOURCE.joinpath('structures','ideal_bcc.cif').read_text()
    faked = tmp_path/'fcc_225.cif'
    faked.write_text(source.replace('data_ctb_ideal_bcc', 'data_space_group_225'))
    original = evaluate([RESOURCE.joinpath('structures','ideal_bcc.cif')], task(), output=tmp_path/'one', cache_root=tmp_path/'cache')
    renamed = evaluate([faked], task(), output=tmp_path/'two', cache_root=tmp_path/'cache')
    assert renamed['results'][0]['measurements']['sg']['value'] == 229
    assert renamed['run']['cache_hits'] == 1
    assert renamed['plan']['comparability_digest'] == original['plan']['comparability_digest']


def test_json_structure_and_domain_rejections(tmp_path):
    geometry = {'cell':[[5,0,0],[0,5,0],[0,0,5]], 'scaled_positions':[[0,0,0],[.5,.5,.5]], 'species':['Ar','Ar']}
    crystal = tmp_path/'structure.json'
    crystal.write_text(json.dumps(geometry))
    result = evaluate([crystal],task(),output=tmp_path/'ok')
    assert result['results'][0]['measurements']['sg']['value'] == 229
    geometry['occupancy'] = [1,.5]
    crystal.write_text(json.dumps(geometry))
    assert load_submission([crystal])[0]['input_status'] == 'unsupported'
    del geometry['occupancy']
    geometry['cell'][2][2] = 50
    crystal.write_text(json.dumps(geometry))
    item = load_submission([crystal])[0]
    assert item['input_status'] == 'unsupported'
    assert 'vacuum_guard' in item['reasons'][0]


def test_relaxed_sg_never_evaluates_input_as_final(tmp_path):
    changed = task()
    changed['reference_geometry'] = 'relaxed'
    data = evaluate([RESOURCE.joinpath('structures','ideal_fcc.cif')], changed, output=tmp_path/'relaxed')
    assert data['results'][0]['decision'] == 'unknown'
    assert data['results'][0]['measurements']['sg']['value'] is None
    assert data['run']['native_calls'] == 0


def test_dry_run_does_not_create_measurements(tmp_path):
    result = evaluate([RESOURCE.joinpath('structures','ideal_fcc.cif')],task(),output=tmp_path/'dry',dry_run=True)
    assert 'metrics' not in result
    assert not (tmp_path/'dry'/'metrics.json').exists()
    assert result['run']['native_calls'] == 0


def test_highk_blocked_does_not_invent_material_failure(tmp_path):
    highk = load_json(RESOURCE.joinpath('tasks','highk_screen.json'))
    result = evaluate([RESOURCE.joinpath('structures','ideal_fcc.cif')],highk,output=tmp_path/'highk')
    assert result['plan']['status'] == 'blocked_configuration'
    assert result['metrics']['n_fail'] == 0
    assert result['metrics']['n_unknown'] == 1
    assert result['run']['physics_calls'] == result['run']['submitted_jobs'] == 0
    assert all(x['value'] is None for x in result['results'][0]['measurements'].values())


def test_output_reuse_cannot_expose_stale_metrics_as_new_run(tmp_path):
    fcc=RESOURCE.joinpath('structures','ideal_fcc.cif')
    evaluate([fcc],task(),output=tmp_path/'run')
    with pytest.raises(ValueError,match='different evaluation'):
        evaluate([fcc],task(),output=tmp_path/'run',dry_run=True)
    changed=task();changed['constraints'][0]['value']=229
    with pytest.raises(ValueError,match='different evaluation'):
        evaluate([fcc],changed,output=tmp_path/'run')
    assert json.loads((tmp_path/'run/metrics.json').read_text())['n_pass'] == 1


def test_trusted_deployment_cache_location_is_honored(tmp_path):
    from crystargetbench.contracts import default_deployment
    site=default_deployment()
    site['paths']['cache_root']=str(tmp_path/'site-cache')
    fcc=RESOURCE.joinpath('structures','ideal_fcc.cif')
    first=evaluate([fcc],task(),output=tmp_path/'first',deployment=site)
    second=evaluate([fcc],task(),output=tmp_path/'second',deployment=site)
    assert first['run']['native_calls']==1
    assert second['run']['native_calls']==0
    assert second['run']['cache_hits']==1
