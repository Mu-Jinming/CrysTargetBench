"""Small synthetic walkthrough for the existing independently installed plugin.

No engine, calculator, helper from tests, network, or private input is used.
The numbers below are authored contract data, never physical measurements.
"""
import argparse
import json
from pathlib import Path
from importlib.resources import files


def write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')


def init(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    resources=files('crystargetbench').joinpath('resources/examples')
    for name in ['ideal_fcc.cif','ideal_bcc.cif']:
        (root/name).write_bytes(resources.joinpath('structures',name).read_bytes())
    (root/'invalid.cif').write_text('Authored invalid input; not a crystal.')
    task=json.loads(resources.joinpath('tasks/bandgap_measure.json').read_text())
    task.update(task_id='synthetic.preview.gap',title='Synthetic SDK demo, not DFT',mode='screen',reference_geometry='relaxed')
    alias=next(iter(task['measurements']))
    task['measurements'][alias]['required_fidelity']='protocol_default'
    task['constraints']=[{'id':'example_gap','measurement':alias,'operator':'ge','value':2.0}]
    write(root/'task.json',task)
    method={'method_id':'synthetic.preview.v1','effective_parameters':{'xc':'PBE','basis':'pw',
        'k_grid':[2,2,2],'k_shift':[0,0,0],'cutoff_Ry':80,'spin':1,'soc':False,'hubbard_u':{},
        'occupations':'fixed','scf_tolerance':1e-8,'force_tolerance_eV_A':0.02,'stress_tolerance_GPa':0.1},
        'software':[{'name':'analytic_test','version':'analytic-fixture-1'}],'asset_fingerprints':[]}
    write(root/'site.json',{'schema_version':'ctb.dft.site.v1','adapter':'second_example','method':method,
        'configuration':{},'execution_mode':'external','managed':None})
    write(root/'PROVENANCE.json',{'evidence_kind':'synthetic','engine_invocations':0,'science_validated':False,
        'description':'Authored zeros and bands for interface demonstration only; no solver or materials claim'})


def returns(run, output, limit=None):
    from crystargetbench.dft.sampling import electronic_diagnostics
    state=json.loads((Path(run)/'exchange.json').read_text());pending=state['pending']
    if limit is not None:pending=pending[:limit]
    for rid in pending:
        request=state['requests'][rid]['request'];g=request['geometry'];n=len(g['species'])
        observations={}
        for quantity in request['requested_quantities']:
            if quantity=='energy':value=-1.0
            elif quantity=='forces':value=[[0.,0.,0.] for _ in range(n)]
            elif quantity=='stress':value=[[0.,0.,0.] for _ in range(3)]
            elif quantity=='relaxed_geometry':value=g
            elif quantity in {'eigenvalues','occupations'}:
                sample=request['sampling']['electronic'];count=len(sample['kpoints'])
                value=[[-1.,2.] if quantity=='eigenvalues' else [2.,0.] for _ in range(count)]
            else:raise ValueError('demo only supplies synthetic relaxation and full-mesh bands')
            record={'value':value}
            if quantity in {'eigenvalues','occupations'}:
                record['diagnostics']={**electronic_diagnostics(sample,sample['kpoints']),
                    'sampling':'uniform_mesh','kpoints':sample['kpoints'],'weights':[1/count]*count,
                    'spin_channels':1,'maximum_occupation':2,'occupation_tolerance':1e-6,
                    'occupation_convention':'per_state_unweighted'}
            observations[quantity]=record
        write(Path(output)/rid/'authored-observations.json',{'request_digest':request['request_digest'],
            'method':request['method'],'observed_geometry':g,'scf':True,
            'ionic':True if request['operation']=='relaxation' else None,'response':None,'observations':observations})
    return {'evidence_kind':'synthetic','written_requests':pending,'engine_invocations':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='action',required=True)
    setup=subs.add_parser('init');setup.add_argument('--output',required=True)
    ret=subs.add_parser('return');ret.add_argument('--run',required=True);ret.add_argument('--output',required=True);ret.add_argument('--limit',type=int)
    args=p.parse_args()
    result=init(args.output) if args.action=='init' else returns(args.run,args.output,args.limit)
    print(json.dumps(result or {'evidence_kind':'synthetic','engine_invocations':0}))

if __name__=='__main__':main()
