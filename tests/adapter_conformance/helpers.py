from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import json
import numpy as np
from crystargetbench.contracts import load_json
from crystargetbench.reporting import write_json
from crystargetbench.dft.registry import discover
from crystargetbench.dft.validation import make_request
from crystargetbench.dft.sampling import electronic_diagnostics
from crystargetbench.structures import load_submission

RESOURCE=files('crystargetbench').joinpath('resources/examples')
def structure(name='ideal_fcc.cif'):return RESOURCE.joinpath('structures',name)
def method(engine='analytic-fixture-1'):
    return {'method_id':'synthetic.interface.v1' if engine=='analytic-fixture-1' else 'abacus.pw.pbe.v1',
        'effective_parameters':{'xc':'PBE','basis':'pw','k_grid':[1,1,1],'k_shift':[0,0,0],
            'cutoff_Ry':80,'spin':1,'soc':False,'hubbard_u':{},'occupations':'fixed',
            'scf_tolerance':1e-8,'force_tolerance_eV_A':0.02,'stress_tolerance_GPa':0.1},
        'software':[{'name':'analytic_test' if engine=='analytic-fixture-1' else 'ABACUS','version':engine}],
        'asset_fingerprints':[]}
def site():
    return {'schema_version':'ctb.dft.site.v1','adapter':'second_example','method':method(),
        'configuration':{},'execution_mode':'external','managed':None}
def task(rank=False,relaxed=True):
    t=load_json(RESOURCE.joinpath('tasks','highk_rank.json' if rank else 'highk_screen.json'))
    t['task_id']='synthetic.'+t['task_id'];t['title']='Synthetic contract test, not material validation'
    t['reference_geometry']='relaxed' if relaxed else 'input'
    for spec in t['measurements'].values():spec['required_fidelity']='protocol_default'
    return t

def write_return(request,directory,*,gap=3.0,kappa=30.0,phonon=0.1,scf=True):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    g=request['geometry'];n=len(g['species'])
    band={'sampling':'uniform_mesh','kpoints':[[0,0,0]],'weights':[1],'spin_channels':1,
          'maximum_occupation':2,'occupation_tolerance':1e-6,'occupation_convention':'per_state_unweighted'}
    if 'electronic' in request['sampling']:
        band['kpoints']=request['sampling']['electronic']['kpoints']
        band['weights']=[1/len(band['kpoints'])]*len(band['kpoints'])
        band.update(electronic_diagnostics(request['sampling']['electronic'],band['kpoints']))
    response={'frequency_eV':0.0,'boundary':'fixed_strain','basis':'original_cartesian','local_field_effects':True,
        'reference_geometry_id':g['geometry_id'],'response_converged':True,'upstream_fidelity':['synthetic']}
    pp=request['recipe']['parameters'];mesh=pp.get('mesh',[1,1,1])
    qpoints=[[i/mesh[0],j/mesh[1],k/mesh[2]] for i in range(mesh[0]) for j in range(mesh[1]) for k in range(mesh[2])]
    pm=request['sampling'].get('phonon')
    if pm:qpoints=pm['qpoints']
    pd={'phonon_parameters':pp,'nac_dielectric_component':'electronic'}
    if pm:pd.update(sampling_manifest_digest=pm['manifest_digest'],coordinate_convention=pm['coordinate_convention'],reciprocal_basis_Ainv=pm['reciprocal_basis_Ainv'])
    v={'energy':{'value':-1.0},'forces':{'value':np.zeros((n,3)).tolist()},'stress':{'value':np.zeros((3,3)).tolist()},
       'relaxed_geometry':{'value':g},'eigenvalues':{'value':[[-1,-1+gap]]*len(band['kpoints']),'diagnostics':band},
       'occupations':{'value':[[2,0]]*len(band['kpoints']),'diagnostics':band},
       'phonon_spectrum':{'value':{'frequencies_THz':[[phonon]*(pm['branch_count'] if pm else n*3)]*len(qpoints),'qpoints':qpoints,'weights':[1]*len(qpoints),
           'scope':pm['scope'] if pm else 'mesh','reference_geometry_id':g['geometry_id'],'upstream_fidelity':['synthetic']},'diagnostics':pd},
       'electronic_dielectric':{'value':(np.eye(3)*2).tolist(),'diagnostics':response},
       'ionic_dielectric':{'value':(np.eye(3)*(kappa-2)).tolist(),'diagnostics':response},
       'total_dielectric':{'value':(np.eye(3)*kappa).tolist(),'diagnostics':{**response,'components':['electronic','ionic']}}}
    write_json(directory/'authored-observations.json',{'request_digest':request['request_digest'],'method':request['method'],
        'scf':scf,'ionic':True if request['operation']=='relaxation' else None,'response':True,
        'observed_geometry':g,'observations':{q:v[q] for q in request['requested_quantities']}})

def finish(run,**kwargs):
    from crystargetbench.dft.exchange import collect
    for _ in range(5):
        state=load_json(Path(run)/'exchange.json')
        if not state['pending']:return collect(run)
        for rid in state['pending']:write_return(state['requests'][rid]['request'],Path(run)/'requests'/rid,**kwargs)
        result=collect(run)
    raise AssertionError('frontier did not terminate')
