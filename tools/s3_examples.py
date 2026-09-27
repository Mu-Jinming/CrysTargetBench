"""Reproducible no-engine S3 evidence. Run: python -m tools.s3_examples."""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
from crystargetbench.api import evaluate
from crystargetbench.dft.exchange import prepare,collect,reassess
from crystargetbench.dft.validation import read_record,validate_result
from crystargetbench.reporting import write_json
from tests.adapter_conformance.helpers import structure,task,site,finish,RESOURCE
from tests.adapter_conformance.test_abacus import request,outputs
from tests.adapter_conformance.test_numerical_and_local import test_returned_analytic_forces_use_real_phonopy_and_reuse_ifc


def main(root=Path('docs/s3_1/examples/synthetic')):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    broken=root/'authored-invalid.cif';broken.write_text('Authored invalid input for full denominator evidence.\n')
    # The selected plugin is the independently installed distribution, not a
    # monkeypatched core provider. No engines or model packages are imported.
    run=root/'second-plugin-screen'
    inputs=[structure(),structure('ideal_bcc.cif'),structure(),broken]
    first=prepare(inputs,task(),site=site(),output=run)
    write_json(root/'first-frontier.json',{'metrics':first['metrics'],'run':first['run']})
    completed=finish(run)
    assert completed['metrics']['n_submitted']==4 and completed['metrics']['n_pass']==3 and completed['metrics']['n_unknown']==1
    assert collect(run)['metrics']==completed['metrics']
    t=task();t['constraints'][-1]['value']=40
    reassess(run,t,output=root/'threshold-40')
    rank=root/'second-plugin-rank';prepare(inputs,task(rank=True),site=site(),output=rank);finish(rank,kappa=4)
    fail=root/'second-plugin-gated-fail';prepare(inputs,task(),site=site(),output=fail);finish(fail,gap=1)
    native=evaluate(inputs,read_record(RESOURCE.joinpath('tasks/spacegroup_225.json')),output=root/'native')
    assert native['metrics']['distributions']['sg']['histogram']=={'225':2,'229':1}
    for op in ['static','relaxation','eigenvalues']:
        d=root/'abacus'/op;d.mkdir(parents=True,exist_ok=True)
        adapter,manifest,req,config=request(op);prepared=adapter.prepare(req,d,config)
        write_json(d/'request.json',req);write_json(d/'prepared.json',prepared)
        outputs(d,req,prepared,force=0 if op=='relaxation' else .1,stress=0 if op=='relaxation' else 2)
        write_json(d/'result.json',validate_result(adapter.collect(req,d,config),req,manifest,d))
        write_json(d/'PROVENANCE.json',{'evidence_kind':'synthetic','source':'authored documented-format parser fragment',
            'engine_invocations':0,'reference_fixture':False,'live_tested':False,'science_validated':False})
    # ABACUS parser -> standardized gap -> existing assess/metrics, keeping
    # required DFT fidelity (synthetic data correctly remain unknown).
    adapter,manifest,req,config=request('eigenvalues')
    s=site();s.update(adapter='abacus_example',method=req['method'],configuration=config)
    g=req['geometry'];input_file=root/'abacus-input.json'
    write_json(input_file,{'cell':g['cell'],'scaled_positions':g['positions'],'species':g['species']})
    gap_task=read_record(RESOURCE.joinpath('tasks/bandgap_measure.json'));gap_task['reference_geometry']='input'
    abrun=root/'abacus-exchange';prepare([input_file,input_file],gap_task,site=s,output=abrun)
    state=read_record(abrun/'exchange.json')
    for rid in state['pending']:
        entry=state['requests'][rid];outputs(abrun/'requests'/rid,entry['request'],entry['prepared'])
    collect(abrun)
    highk=prepare([input_file],task(),site=s,output=root/'abacus-highk-blocked')
    assert highk['plan']['status']=='blocked_capability'
    numeric=root/'analytic-phonopy';numeric.mkdir(exist_ok=True)
    test_returned_analytic_forces_use_real_phonopy_and_reuse_ifc(numeric)
    write_json(root/'EVIDENCE_SCOPE.json',{'engine_invocations':0,'MLFF_invocations':0,'cluster_jobs':0,
        'native':'actual ASE/spglib CPU','plugin_results':'synthetic','abacus_parser':'synthetic fragments, no actual reference output',
        'phonopy':'actual CPU numerical interface with analytic_test pair spring forces',
        'live_tested':False,'science_validated':False,'benchmark_eligible':False})

if __name__=='__main__':main()
