"""Independent software-contract probes against the unmodified CTB 0.3.0 wheel.
No DFT, MLFF, ASE, spglib, or material calculations are run.
Geometry conversion to ASE is stubbed ONLY to avoid the unavailable dependency;
complete content-hashed CTB snapshots, validation, reducers and assessment remain
original. The symmetry probe captures arguments at the numerical boundary.
"""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import json
from importlib.resources import files
from crystargetbench.identity import digest
from crystargetbench.contracts import default_protocol,validate_protocol,load_json
from crystargetbench.dft.registry import discover
from crystargetbench.dft import validation as v, exchange as ex
from crystargetbench.dft.reducers import qualified,reduce_property
from crystargetbench.public_export import export_public

RES=files('crystargetbench').joinpath('resources/examples')
def geom():
    state={'cell':[[4.,0.,0.],[0.,4.,0.],[0.,0.,4.]],'positions':[[0.,0.,0.]],
           'coordinate_system':'fractional','length_unit':'angstrom','species':['Al'],
           'occupancy':[1.0],'pbc':[True]*3,'magnetic_state':{'kind':'nonmagnetic','moments':[0.]}}
    return {**state,'geometry_id':digest(state),'masses':[26.9815385],'mass_digest':digest([26.9815385]),'mass_unit':'amu'}
def method(grid=(1,1,1)):
    return {'method_id':'synthetic.review.v1','effective_parameters':{'xc':'PBE','basis':'pw','k_grid':list(grid),'k_shift':[0,0,0],
        'cutoff_Ry':80,'spin':1,'soc':False,'hubbard_u':{},'occupations':'fixed','scf_tolerance':1e-8,
        'force_tolerance_eV_A':0.02,'stress_tolerance_GPa':0.1},'software':[{'name':'analytic_test','version':'analytic-fixture-1'}], 'asset_fingerprints':[]}
def site(grid=(1,1,1)):
    return {'schema_version':'ctb.dft.site.v1','adapter':'second_example','method':method(grid),
            'configuration':{},'execution_mode':'external','managed':None}
def external_symmetry(root):
    task=load_json(RES.joinpath('tasks/spacegroup_225.json'))
    protocol=default_protocol(task);protocol['parameters']['symprec_A']=0.0001
    protocol['parameters']['angle_tolerance_deg']=0.1
    protocol=validate_protocol(protocol);plan=ex.plan_external(task,site(),protocol)
    g=geom();state={'task':task,'plan':plan,'items':[{'input_status':'valid','input_geometry':g,'item_id':'review-item','reasons':[]}],
        'site':site(),'manifest':discover()['second_example'],'requests':{},'accepted':{}}
    captured=[]
    def spy(geometry,parameters):
        captured.append(deepcopy(parameters))
        return {'space_group_number':225,'international':'TEST_SPY','hall_number':0,'hall_symbol':'TEST_SPY',
                'symprec_A':parameters['symprec_A'],'angle_tolerance_deg':parameters['angle_tolerance_deg'],
                'spglib_version':'test-spy-not-spglib','ase_version':'not-run','target_symmetrization':False}
    with patch.object(ex,'analyze_symmetry',spy):
        values,_,_=ex._frontier(state,root,adapter=None)
    return {'scope':'Original external frontier; numerical symmetry function replaced by argument-recording spy. No space group measured.',
        'locked':protocol['parameters'],'passed_to_native':captured[0],
        'measurement_diagnostics':values[0]['measurements']['sg']['diagnostics'],
        'invariant_holds':captured[0]==protocol['parameters']}
def band_mesh(root):
    root.mkdir(parents=True,exist_ok=True);(root/'raw.json').write_text('{"evidence":"synthetic review only"}')
    manifest=discover()['second_example'];g=geom()
    info={'sampling':'uniform_mesh','kpoints':[[0,0,0]],'weights':[1.0], 'spin_channels':1,
          'maximum_occupation':2,'occupation_tolerance':1e-6,'occupation_convention':'per_state_unweighted'}
    with patch.object(v,'atoms_from_snapshot',lambda s:None):
        request=v.make_request(manifest,'eigenvalues',['eigenvalues','occupations'],g,method((4,4,4)))
        obs=[v.observation(q,value,definition_id='synthetic.review.'+q,raw_artifacts=[v.artifact(root,'raw.json')],
             diagnostics=info,evidence_kind='synthetic') for q,value in [('eigenvalues',[[-1.,2.]]),('occupations',[[2.,0.]])]]
        record=v.result_record(request,manifest,obs,observed_geometry=g,scf=True,evidence_mode='fixture')
        validated=v.validate_result(record,request,manifest,root)
    value,why=reduce_property('band_gap_eV',qualified(validated),g,'synthetic')
    return {'scope':'Original request/result validator and reducer; only ASE geometry conversion stubbed.',
        'requested_grid':request['method']['effective_parameters']['k_grid'],'reported_kpoints':info['kpoints'],
        'reported_mapping_or_symmetry_expansion':None,'validation_accepted':True,
        'reduced_gap_eV':value,'reasons':why,'invariant_holds':value is None}
def gamma_as_path(root):
    root.mkdir(parents=True,exist_ok=True);(root/'raw.json').write_text('{"evidence":"synthetic review only"}')
    manifest=discover()['second_example'];g=geom()
    task=load_json(RES.joinpath('tasks/phonon_path.json'))
    pp=default_protocol(task)['parameters']['phonon']
    data={'frequencies_THz':[[0.1,0.2,0.3]],'qpoints':[[0.,0.,0.]],'weights':[1],
          'scope':'path','reference_geometry_id':g['geometry_id'],'upstream_fidelity':['synthetic']}
    with patch.object(v,'atoms_from_snapshot',lambda s:None):
        request=v.make_request(manifest,'phonon_spectrum',['phonon_spectrum'],g,method(),
                 recipe={'id':'ctb.external.phonon_spectrum.v1','parameters':pp})
        obs=v.observation('phonon_spectrum',data,definition_id='synthetic.review.path',
              raw_artifacts=[v.artifact(root,'raw.json')],diagnostics={'phonon_parameters':pp},evidence_kind='synthetic')
        record=v.result_record(request,manifest,[obs],observed_geometry=g,scf=True,evidence_mode='fixture')
        checked=v.validate_result(record,request,manifest,root)
    value,why=reduce_property('phonon_path_min_frequency_THz',qualified(checked),g,'synthetic')
    return {'scope':'Original request/result validator and reducer; only ASE geometry conversion stubbed.',
        'requested_phonon_parameters':pp,'reported_qpoints':data['qpoints'],'reported_scope':'path',
        'validation_accepted':True,'reduced_min_frequency_THz':value,'reasons':why,'invariant_holds':value is None}
def authorized_export(root):
    root.mkdir(parents=True,exist_ok=True)
    wire={'number':42,'nested':{'schema_version':'ctb.live_permit.v1','authorized':True,
          'approval_reference':'SYNTHETIC_REVIEW_NOT_A_REAL_APPROVAL','allowed_operations':['synthetic_test_only']}}
    p=root/'harmless-name.json';p.write_text(json.dumps(wire));before=p.read_bytes()
    audit=export_public(root,root/'public',['harmless-name.json'])
    exported=json.loads((root/'public/harmless-name.json').read_text())
    survives=isinstance(exported.get('nested'),dict) and exported['nested'].get('authorized') is True
    return {'scope':'Original public exporter on deliberately synthetic data. No real permission or credential.',
        'authorized_record_survived':survives,'unchanged_private_original':p.read_bytes()==before,
        'redactions':audit['redactions'],'invariant_holds':not survives}
def accept_safe_rejection(fn):
    def run(root):
        try:
            return fn(root)
        except ValueError as exc:
            return {'scope':'Malformed sampling evidence rejected by the real request/validator/reducer path.',
                    'validation_accepted':False,'reason':str(exc),'invariant_holds':True}
    return run

PROBES={'S3R1_symmetry_protocol':external_symmetry,'S3R2_k_sampling':accept_safe_rejection(band_mesh),
        'S3R3_q_sampling':accept_safe_rejection(gamma_as_path),'S3R4_authorized_export':authorized_export}
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    results={name:fn(root/name) for name,fn in PROBES.items()}
    (root/'PROBE_RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
