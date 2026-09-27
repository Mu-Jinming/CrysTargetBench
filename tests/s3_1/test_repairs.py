"""S3.1 contract attacks and positive CPU cases. No physical engine evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from crystargetbench import api
from crystargetbench.contracts import default_protocol, load_json
from crystargetbench.dft import exchange
from crystargetbench.dft.registry import load_adapter
from crystargetbench.dft.validation import make_request, validate_result, validate_request, read_record
from crystargetbench.dft.reducers import qualified, reduce_property
from crystargetbench.dft.sampling import electronic_diagnostics, request_sampling
from crystargetbench.geometry import snapshot_from_atoms
from crystargetbench.identity import digest
from crystargetbench.public_export import export_public
from crystargetbench.public_audit import audit_public_tree
from tests.adapter_conformance.helpers import RESOURCE, structure, site, method, write_return, finish
from .review_probes import PROBES


@pytest.mark.parametrize('case', list(PROBES))
def test_original_review_invariants(case, tmp_path):
    assert PROBES[case](tmp_path/case)['invariant_holds']


def test_actual_native_external_locked_tolerances_and_identity(tmp_path):
    task=load_json(RESOURCE.joinpath('tasks/spacegroup_225.json'))
    protocol=default_protocol(task)
    protocol['parameters'].update(symprec_A=0.0001, angle_tolerance_deg=0.1)
    # A perturbation tests actual tolerance behavior, not ideal fixture labels.
    from ase.build import bulk
    atoms=bulk('Cu','fcc',a=3.6,cubic=True);atoms.positions[0,0]+=0.002
    path=tmp_path/'perturbed.cif';write(path,atoms)
    native=api.evaluate([path],task,output=tmp_path/'native',protocol=protocol)
    external=exchange.prepare([path],task,output=tmp_path/'external',site=site(),protocol=protocol)
    nm=native['results'][0]['measurements']['sg'];em=external['results'][0]['measurements']['sg']
    assert nm['value']==em['value'] and em['value']!=225
    assert nm['diagnostics']==em['diagnostics']
    for key,value in protocol['parameters'].items():assert em['diagnostics'][key]==value
    lock=external['plan']['protocol_lock']
    assert lock['native_dependencies']=={'ase':em['diagnostics']['ase_version'],'spglib':em['diagnostics']['spglib_version']}
    looser=deepcopy(protocol);looser['parameters'].update(symprec_A=0.01,angle_tolerance_deg=5.)
    other=exchange.prepare([path],task,output=tmp_path/'loose',site=site(),protocol=looser)
    assert other['results'][0]['measurements']['sg']['value']==225
    assert other['plan']['protocol_digest']!=external['plan']['protocol_digest']
    target=deepcopy(task);target['constraints'][0]['value']=1
    rerun=api.evaluate([path],target,output=tmp_path/'target',protocol=protocol,cache_root=tmp_path/'native/cache')
    assert rerun['run']['native_calls']==0 and rerun['run']['cache_hits']==1
    assert exchange.plan_external(target,site(),protocol)['protocol_digest']==lock_digest(external)


def lock_digest(result):return result['plan']['protocol_digest']


def band_case(tmp_path, grid=(4,4,4), shift=(0,0,0)):
    adapter,manifest=load_adapter('second_example')
    geom=snapshot_from_atoms(Atoms('Al',scaled_positions=[[0,0,0]],cell=np.eye(3)*4,pbc=True))
    m=method();m['effective_parameters'].update(k_grid=list(grid),k_shift=list(shift))
    req=make_request(manifest,'eigenvalues',['eigenvalues','occupations'],geom,m)
    write_return(req,tmp_path)
    result=adapter.collect(req,tmp_path,{})
    return req,result,manifest


def test_complete_shifted_reordered_mesh_and_real_zero_gap(tmp_path):
    req,result,manifest=band_case(tmp_path,shift=(1,0,1))
    for obs in result['observations']:
        obs['value'].reverse()
        for key in ('kpoints','weights','kpoint_indices'):obs['diagnostics'][key].reverse()
    checked=validate_result(result,req,manifest,tmp_path)
    assert reduce_property('band_gap_eV',qualified(checked),req['geometry'],'synthetic')==(3.,[])
    # True fractional occupancy is metal only after full coverage validation.
    next(o for o in result['observations'] if o['quantity']=='occupations')['value'][0][0]=1.
    checked=validate_result(result,req,manifest,tmp_path)
    assert reduce_property('band_gap_eV',qualified(checked),req['geometry'],'synthetic')==(0.,[])


@pytest.mark.parametrize('attack',['gamma','missing','duplicate','shift','weights','row_mapping','band_occupation_mapping','basis','digest','irreducible','legacy'])
def test_generic_electronic_sampling_rejections(attack,tmp_path):
    req,result,manifest=band_case(tmp_path)
    for obs in result['observations']:
        info=obs['diagnostics']
        if attack in {'gamma','missing'}:
            stop=1 if attack=='gamma' else 63
            obs['value']=obs['value'][:stop]
            for key in ('kpoints','weights','kpoint_indices'):info[key]=info[key][:stop]
            info['weights']=[1/stop]*stop
        elif attack=='duplicate':info['kpoints'][1]=info['kpoints'][0]
        elif attack=='shift':info['kpoints'][0][0]+=.125
        elif attack=='weights':info['weights'][0]+=.01;info['weights'][1]-=.01
        elif attack=='row_mapping':info['kpoint_indices'][0],info['kpoint_indices'][1]=info['kpoint_indices'][1],info['kpoint_indices'][0]
        elif attack=='band_occupation_mapping' and obs['quantity']=='occupations':
            info['kpoints'].reverse();info['kpoint_indices'].reverse();obs['value'].reverse()
        elif attack=='basis':info['reciprocal_basis_Ainv'][0][0]+=1
        elif attack=='digest':info['sampling_manifest_digest']='0'*64
        elif attack=='irreducible':
            info['sampling']='irreducible_mesh';info['expansion']={'full_to_irreducible':[0]*64}
        elif attack=='legacy':info.pop('sampling_manifest_digest')
    with pytest.raises(ValueError):validate_result(result,req,manifest,tmp_path)


def phonon_case(tmp_path,scope='path',points=7):
    adapter,manifest=load_adapter('second_example')
    from ase.build import bulk
    geom=snapshot_from_atoms(bulk('Cu','fcc',a=3.6,cubic=True))
    task=load_json(RESOURCE.joinpath('tasks/phonon_path.json'))
    pp=default_protocol(task)['parameters']['phonon']
    pp.update(path_points_parameter=points)
    if scope=='mesh':
        pp.pop('path_points_parameter');pp.update(q_scope='mesh',mesh=[2,2,2],gamma_center=True)
    req=make_request(manifest,'phonon_spectrum',['phonon_spectrum'],geom,method(),recipe={'id':'ctb.external.phonon_spectrum.v2','parameters':pp})
    write_return(req,tmp_path)
    return req,adapter.collect(req,tmp_path,{}),manifest


def test_cpu_path_manifest_and_primitive_basis(tmp_path):
    req,result,manifest=phonon_case(tmp_path,points=101)
    sample=req['sampling']['phonon']
    assert sample['qpoint_count']==101*len(sample['segments']) and sample['qpoint_count']!=101
    assert sample['generator']['versions']['seekpath']=='2.1.0'
    assert sample['generator']['versions']['phonopy']=='2.38.2'
    assert sample['branch_count']==3 and len(req['geometry']['species'])==4
    reference_basis=2*np.pi*np.linalg.inv(req['geometry']['cell']).T
    np.testing.assert_allclose(np.asarray(sample['basis_to_reference_reciprocal']) @ reference_basis,sample['reciprocal_basis_Ainv'],atol=1e-12)
    checked=validate_result(result,req,manifest,tmp_path)
    assert reduce_property('phonon_path_min_frequency_THz',qualified(checked),req['geometry'],'synthetic')==(.1,[])
    for segment in sample['segments']:
        np.testing.assert_allclose(sample['qpoints'][segment['start_index']],sample['labels'][segment['labels'][0]])
        np.testing.assert_allclose(sample['qpoints'][segment['stop_index_exclusive']-1],sample['labels'][segment['labels'][1]])
    # This manifest is generated after the actual final geometry; another cell
    # cannot inherit it even if the same path labels are generated.
    from crystargetbench.geometry import atoms_from_snapshot
    atoms=atoms_from_snapshot(req['geometry']);cell=atoms.cell.array.copy();cell[0,0]+=.1
    atoms.set_cell(cell,scale_atoms=True);other=snapshot_from_atoms(atoms)
    assert request_sampling(other,req['method'],req['requested_quantities'],req['recipe'])['phonon']['manifest_digest']!=sample['manifest_digest']


@pytest.mark.parametrize('attack',['gamma','missing_q','reordered','scope','basis','segments','manifest_basis'])
def test_path_coverage_cannot_be_forged_by_labels(attack,tmp_path):
    req,result,manifest=phonon_case(tmp_path)
    obs=result['observations'][0];data=obs['value']
    if attack in {'gamma','missing_q'}:
        stop=1 if attack=='gamma' else len(data['qpoints'])-1
        for key in ('qpoints','weights','frequencies_THz'):data[key]=data[key][:stop]
        if attack=='gamma':data['qpoints']=[[0.,0.,0.]]
    elif attack=='reordered':data['qpoints'][0],data['qpoints'][1]=data['qpoints'][1],data['qpoints'][0]
    elif attack=='scope':data['scope']='mesh'
    elif attack=='basis':obs['diagnostics']['reciprocal_basis_Ainv'][0][0]+=.01
    else:
        changed=deepcopy(req)
        key='segments' if attack=='segments' else 'reciprocal_basis_Ainv'
        changed['sampling']['phonon'][key]=[]
        changed['sampling']['phonon']['manifest_digest']=digest({k:v for k,v in changed['sampling']['phonon'].items() if k!='manifest_digest'})
        changed['request_id']=changed['request_digest']=digest({k:v for k,v in changed.items() if k not in {'request_id','request_digest'}})
        with pytest.raises(ValueError,match='sampling manifest'):validate_request(changed)
        return
    with pytest.raises(ValueError):validate_result(result,req,manifest,tmp_path)


def test_combined_path_mesh_requests_keep_independent_scopes(tmp_path):
    task=load_json(RESOURCE.joinpath('tasks/phonon_path.json'))
    path_spec=next(iter(task['measurements'].values()))
    path_spec['required_fidelity']='protocol_default'
    task['mode']='measure';task['constraints']=[]
    task['measurements']={'path':path_spec,'mesh':{**path_spec,'property_id':'phonon_mesh_min_frequency_THz'}}
    task['report']=['path','mesh']
    protocol=default_protocol(task)
    protocol['parameters']['phonon'].update(path_points_parameter=5,mesh=[2,2,2])
    run=tmp_path/'run';exchange.prepare([structure()],task,site=site(),protocol=protocol,output=run)
    result=finish(run)
    assert result['metrics']['n_submitted']==1
    assert all(m['value']==.1 for m in result['results'][0]['measurements'].values())
    state=read_record(run/'exchange.json')
    requests=[x['request'] for x in state['requests'].values() if x['request']['operation']=='phonon_spectrum']
    assert len(requests)==2
    assert {r['sampling']['phonon']['scope'] for r in requests}=={'path','mesh'}
    assert requests[0]['request_id']!=requests[1]['request_id']


@pytest.mark.parametrize('record',[
    {'schema_version':'ctb.live_permit.v1','authorized':True},
    {'schema_version':'ctb.live_permit.v1','enabled':True},
    {'schema_version':'ctb.live_permit.v1','authorized':True,'enabled':False},
    {'schema_version':'ctb.live_permit.v1','authorized':False,'enabled':True},
    {'schema_version':'ctb.live_permit.v1','enabled':False},
    {'schema_version':'vendor.unknown_permit.v99','state':'unknown'},
    {'schema':'legacy.authorization','allowed_operations':['synthetic_test_only']},
    {'schema_version':'ctb.session.v0','state':'completed'},
    {'authorized':'false','allowed_operations':['synthetic_test_only']},
    {'enabled':False,'authorization_id':'SYNTHETIC_NOT_A_PERMISSION'},
])
def test_permission_forms_fail_closed_all_serializations(record,tmp_path):
    import yaml
    wire={'public_number':42,'nested':[record],'text':json.dumps({'deep':record})}
    sources={'sample.json':json.dumps(wire),'sample.yaml':yaml.safe_dump(wire),'sample.txt':'share '+json.dumps(wire)}
    for name,text in sources.items():(tmp_path/name).write_text(text)
    hashes={name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in sources}
    # The independent checker detects the attack in the original files.
    assert audit_public_tree(tmp_path)['findings']
    audit=export_public(tmp_path,tmp_path/'public',list(sources))
    assert not audit['independent_postcheck']['findings']
    assert audit['redactions'] and not audit_public_tree(tmp_path/'public')['findings']
    for name in sources:
        assert '42' in (tmp_path/'public'/name).read_text()
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==hashes[name]


def test_only_reviewed_disabled_template_survives_and_postcheck_is_independent(tmp_path,monkeypatch):
    template=load_json('CTB_S2_Addendum/examples/MLFF_LIVE_PERMIT.disabled.json')
    (tmp_path/'template.json').write_text(json.dumps(template))
    bad=deepcopy(template);bad['enabled']=True
    (tmp_path/'bad.json').write_text(json.dumps(bad))
    export_public(tmp_path,tmp_path/'public',['template.json','bad.json'])
    assert json.loads((tmp_path/'public/template.json').read_text())==template
    assert not (tmp_path/'public/bad.json').exists()
    # Disable the redactor deliberately: the independent scanner must still
    # catch the permission, proving this is not a repeat of the same algorithm.
    import crystargetbench.public_export as exporter
    monkeypatch.setattr(exporter,'redact',lambda value,**kwargs:value)
    with pytest.raises(ValueError,match='independent public postcheck'):
        export_public(tmp_path,tmp_path/'leaky',['bad.json'])


def test_legacy_wire_never_silently_upgrades(tmp_path):
    req,result,manifest=band_case(tmp_path,grid=(1,1,1))
    old=deepcopy(req);old['schema_version']='ctb.dft.request.v1'
    with pytest.raises(ValueError,match='legacy'):validate_request(old)
    result['schema_version']='ctb.dft.result.v1'
    with pytest.raises(ValueError,match='API'):validate_result(result,req,manifest,tmp_path)


def test_public_schema_property_is_preserved_byte_for_byte(tmp_path):
    from importlib.resources import files
    source=files('crystargetbench').joinpath('resources/schemas/deployment.schema.json').read_bytes()
    (tmp_path/'deployment.schema.json').write_bytes(source)
    audit=export_public(tmp_path,tmp_path/'public',['deployment.schema.json'])
    assert (tmp_path/'public/deployment.schema.json').read_bytes()==source
    assert audit['redactions']==[] and not audit['independent_postcheck']['findings']


@pytest.mark.parametrize('key',['permit','live_permit','authorization'])
def test_independent_postcheck_rejects_schemaless_unknown_permission(key,tmp_path):
    data={key:{'enabled':False,'unknown_state':'not_an_approved_template'}}
    (tmp_path/'unknown.json').write_text(json.dumps(data))
    assert audit_public_tree(tmp_path)['findings']
    export_public(tmp_path,tmp_path/'public',['unknown.json'])
    assert json.loads((tmp_path/'public/unknown.json').read_text())=={}


def test_path_resampling_reuses_existing_analytic_ifc_and_no_new_forces(tmp_path):
    from crystargetbench.dft.phonons import displacement_requests, collect_phonons
    from crystargetbench.dft.validation import observation,result_record,artifact
    from crystargetbench.geometry import atoms_from_snapshot
    from crystargetbench.reporting import write_json
    _,manifest=load_adapter('second_example')
    geom=snapshot_from_atoms(Atoms('Ne2',scaled_positions=[[0,0,0],[.5,.5,.5]],cell=np.eye(3)*5,pbc=True))
    params={'supercell_matrix':np.eye(3,dtype=int).tolist(),'primitive_matrix':np.eye(3).tolist(),
            'mesh':[2,2,2],'is_symmetry':False,'is_plusminus':True,'is_diagonal':False}
    plan=displacement_requests(geom,manifest,method(),params)
    base=atoms_from_snapshot(plan['requests'][0]['geometry']).positions
    for req in plan['requests']:
        delta=atoms_from_snapshot(req['geometry']).positions-base
        forces=-2*(delta-delta.mean(axis=0))
        directory=tmp_path/'returned'/req['request_id'];directory.mkdir(parents=True)
        (directory/'analytic.json').write_text('analytic_test two-site spring; not DFT')
        obs=observation('forces',forces.tolist(),definition_id='analytic_test.pair_spring',raw_artifacts=[artifact(directory,'analytic.json')],evidence_kind='synthetic')
        write_json(directory/'result.json',result_record(req,manifest,[obs],observed_geometry=req['geometry'],scf=True,evidence_mode='fixture'))
    originals={str(p):p.read_bytes() for p in (tmp_path/'returned').rglob('*') if p.is_file()}
    mesh=collect_phonons(plan,tmp_path/'returned',tmp_path/'cache')
    first=collect_phonons(plan,tmp_path/'returned',tmp_path/'cache',sampling={'path_points_parameter':5},scope='path')
    second=collect_phonons(plan,tmp_path/'returned',tmp_path/'cache',sampling={'path_points_parameter':9},scope='path')
    assert mesh['ifc']['artifact_digest']==first['ifc']['artifact_digest']==second['ifc']['artifact_digest']
    assert first['sampling']['artifact_digest']!=second['sampling']['artifact_digest']
    assert len(second['sampling']['qpoints'])==9*len(second['sampling']['segments'])
    assert second['cache_hits']>len(plan['requests']) and second['ctb_engine_invocations']==0
    assert originals=={str(p):p.read_bytes() for p in (tmp_path/'returned').rglob('*') if p.is_file()}
