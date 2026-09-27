"""Execute documented user paths in a disposable HOME/cwd, using installed wheels.

All dependency wheels must come from the separately recorded ordinary public
dependency download. No editable install or checkout import enters the runtime.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',default='.')
    p.add_argument('--wheel',required=True);p.add_argument('--abacus-wheel',required=True)
    p.add_argument('--second-wheel',required=True);p.add_argument('--dependencies',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();project=Path(a.project).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    base=Path(tempfile.mkdtemp(prefix='ctb-preview-user-'));work=base/'work';home=base/'home';incoming=base/'incoming'
    for folder in (work,home,incoming):folder.mkdir()
    wheels=[]
    for supplied in (a.wheel,a.abacus_wheel,a.second_wheel):
        dest=incoming/Path(supplied).name;shutil.copyfile(supplied,dest);wheels.append(dest)
    shutil.copyfile(project/'tools/preview_demo.py',work/'preview_demo.py')
    shutil.copyfile(project/'examples/dft_adapters/abacus/site.disabled.json',work/'abacus-template.json')
    # Execute the actual README Python blocks, not a separate expected mock.
    readme=(project/'README.md').read_text();blocks=[]
    for block in readme.split("python - <<'PY'\n")[1:]:blocks.append(block.split('\nPY\n',1)[0])
    assert len(blocks)>=2
    commands=[];summaries={}
    clean=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):clean.pop(key,None)
    clean.update(HOME=str(home),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',PIP_NO_CACHE_DIR='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    def run(check,argv,expected=0,computation='none',dependencies='installed CTB core',input_summary=None):
        result=subprocess.run([str(x) for x in argv],cwd=work,env=clean,text=True,capture_output=True)
        record={'check':check,'argv':[str(x) for x in argv],'cwd':str(work),'HOME':str(home),
            'dependencies':dependencies,'computation':computation,'input_summary':input_summary,
            'expected_returncode':expected,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
            'expectation_met':result.returncode==expected}
        commands.append(record)
        (out/'USER_JOURNEY_RESULTS.json').write_text(json.dumps({'status':'running','commands':commands},indent=2)+'\n')
        if result.returncode!=expected:raise AssertionError(record)
        return result.stdout
    run('environment',[sys.executable,'-m','venv',base/'venv'],dependencies='Python venv')
    python=base/'venv/bin/python';ctb=base/'venv/bin/ctb'
    install_args=['--no-index','--find-links',str(Path(a.dependencies).resolve())]
    run('U01_install',[python,'-m','pip','install',*install_args,wheels[0]],dependencies='downloaded ordinary public dependency wheels, no project wheelhouse')
    core_probe="""import json,sys,hashlib;from pathlib import Path;from importlib import metadata;import crystargetbench
root=Path(crystargetbench.__file__).parent
assert 'site-packages' in str(root)
assert crystargetbench.__version__==metadata.version('crystargetbench')
names={d.metadata['Name'].lower() for d in metadata.distributions()}
assert not names & {'mattersim','torch','phonopy','abacus','atomate2','jobflow','pyatb'}
assert all(not json.loads(d.read_text('direct_url.json') or '{}').get('dir_info',{}).get('editable') for d in metadata.distributions())
print(json.dumps({'version':crystargetbench.__version__,'core_path':str(root),'sys_path':sys.path,'packages':sorted(names),'hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.cif'}}}))"""
    core=json.loads(run('U01_isolation',[python,'-I','-c',core_probe]));summaries['core_only']=core
    assert run('U01_version',[ctb,'--version']).strip()=='ctb '+core['version']
    run('U02_extract',[python,'-I','-c',blocks[0]],input_summary='verbatim README resource extraction snippet')
    fcc='ctb-examples/structures/ideal_fcc.cif';bcc='ctb-examples/structures/ideal_bcc.cif';task='ctb-examples/tasks/spacegroup_225.json'
    def native(output,task_file=task,inputs=None):
        return json.loads(run('U01_native' if output=='native-225' else 'U04_native' if output=='native-229' else 'U03_full_denominator',
            [ctb,'evaluate','--structures',*(inputs or [fcc,bcc]),'--task',task_file,'--cache','measurements','--output',output],computation='actual native CPU; compatible cache may avoid repeat analysis'))
    m=native('native-225');assert (m['n_submitted'],m['n_pass'],m['n_fail'],m['n_unknown'])==(2,1,1,0)
    assert m['distributions']['sg']['histogram']=={'225':1,'229':1}
    summaries['native_225']=m
    summary=json.loads(run('U01_summarize',[ctb,'summarize','native-225']));assert summary==m
    run('U04_target_edit',[python,'-I','-c',blocks[1]],input_summary='verbatim README target-only edit snippet')
    summaries['native_229']=native('native-229','target-229.json')
    r=json.loads((work/'native-229/run.json').read_text());assert r['native_calls']==0 and r['cache_hits']==2
    summaries['target_reuse']={'native_calls':r['native_calls'],'cache_hits':r['cache_hits']}
    first=json.loads((work/'native-225/results.json').read_text())['results'];second=json.loads((work/'native-229/results.json').read_text())['results']
    assert [i['decision'] for i in first]==['pass','fail'] and [i['decision'] for i in second]==['fail','pass']
    run('U03_invalid_input',[python,'-I','-c',"from pathlib import Path;Path('invalid.cif').write_text('Authored invalid structure')"])
    m=native('native-full',inputs=[fcc,bcc,fcc,'invalid.cif']);assert (m['n_submitted'],m['n_pass'],m['n_fail'],m['n_unknown'])==(4,2,1,1)
    summaries['full_denominator']=m
    run('U05_install_plugins',[python,'-m','pip','install',*install_args,*wheels[1:]],dependencies='two separately built adapter wheels')
    manifests=json.loads(run('U05_metadata_discovery',[python,'-I','-c',"from crystargetbench.dft.registry import discover;import json,sys;d=discover();assert {'abacus_example','second_example'}<=d.keys();assert 'ctb_abacus_example.adapter' not in sys.modules and 'ctb_second_example.adapter' not in sys.modules;print(json.dumps(d))"]))
    assert json.loads(run('U05_core_unchanged',[python,'-I','-c',core_probe]))['hashes']==core['hashes']
    run('U05_adapters_cli',[ctb,'dft','adapters']);summaries['adapters']=manifests
    blocked=[ctb,'dft','prepare','--structures',fcc,'--task','ctb-examples/tasks/bandgap_measure.json','--site','abacus-template.json','--output','abacus-unresolved']
    b=json.loads(run('U06_unresolved_assets',blocked,expected=3));assert b['plan']['status']=='blocked_configuration'
    # A synthetic digest declaration tests missing-asset preparation. No asset
    # file is created, and this declaration cannot qualify a physical result.
    code="""import json,hashlib;from pathlib import Path
s=json.loads(Path('abacus-template.json').read_text());h=hashlib.sha256(b'SYNTHETIC_MISSING_ASSET_NOT_A_PSEUDOPOTENTIAL').hexdigest()
s['method']['asset_fingerprints'][0].update(sha256=h,identifier='Ar:USER_ASSET.upf')
s['configuration']['assets']['Ar']=s['configuration']['assets'].pop('Cu');s['configuration']['assets']['Ar']['sha256']=h
Path('abacus-declared-missing.json').write_text(json.dumps(s))"""
    run('U06_declared_missing_asset',[python,'-I','-c',code],input_summary='synthetic checksum declaration only; never an actual asset')
    cmd=blocked.copy();cmd[cmd.index('abacus-template.json')]='abacus-declared-missing.json';cmd[-1]='abacus-missing'
    ready=json.loads(run('U06_abacus_prepare',cmd))
    state=json.loads((work/'abacus-missing/exchange.json').read_text())
    assert state['requests'] and all(x['prepared']['status']=='blocked_assets' for x in state['requests'].values())
    for rid in state['requests']:
        assert all((work/'abacus-missing/requests'/rid/name).is_file() for name in ['INPUT','STRU','KPT','request.json','prepared.json'])
        assert not (work/'abacus-missing/requests'/rid/'assets').exists()
    summaries['abacus_missing_assets']={'plan':ready['plan']['status'],'prepared':'blocked_assets','assets_created':False,'engine_not_required_for_prepare':True}
    cmd=[ctb,'dft','prepare','--structures',fcc,bcc,fcc,'invalid.cif','--task','ctb-examples/tasks/highk_screen.json','--site','abacus-declared-missing.json','--output','highk-blocked']
    high=json.loads(run('U07_highk',cmd,expected=3));assert high['plan']['status']=='blocked_capability'
    assert (high['metrics']['n_submitted'],high['metrics']['n_unknown'],high['metrics']['n_fail'])==(4,4,0)
    assert all(m['value'] is None for i in high['results'] for m in i['measurements'].values())
    summaries['highk']={'status':high['plan']['status'],'missing_capabilities':high['plan']['missing_capabilities'],'metrics':high['metrics']}
    run('U06_demo_init',[python,'preview_demo.py','init','--output','demo'],computation='authored synthetic files only')
    cmd=[ctb,'dft','prepare','--structures','demo/ideal_fcc.cif','demo/ideal_bcc.cif','demo/ideal_fcc.cif','demo/invalid.cif','--task','demo/task.json','--site','demo/site.json','--output','demo/run']
    result=json.loads(run('U06_demo_prepare',cmd));assert result['metrics']['n_unknown']==4
    collect=[ctb,'dft','collect','--run','demo/run','--results','demo/returned']
    result=json.loads(run('U06_missing_return',collect));assert result['metrics']['n_unknown']==4
    run('U06_partial_return',[python,'preview_demo.py','return','--run','demo/run','--output','demo/returned','--limit','1'],computation='authored synthetic files only')
    partial=json.loads(run('U06_partial_collect',collect));assert partial['metrics']['n_submitted']==4 and partial['metrics']['n_unknown']==4
    # Fill current frontier, then prove incomplete k coverage is rejected.
    run('U06_next_return',[python,'preview_demo.py','return','--run','demo/run','--output','demo/returned'],computation='authored synthetic files only')
    run('U06_malformed_sampling',[python,'-I','-c',"import json,shutil;from pathlib import Path;shutil.copytree('demo/returned','demo/incomplete');p=next(p for p in Path('demo/incomplete').glob('*/authored-observations.json') if 'eigenvalues' in json.loads(p.read_text())['observations']);d=json.loads(p.read_text());d['observations']['eigenvalues']['value']=d['observations']['eigenvalues']['value'][:1];p.write_text(json.dumps(d))"])
    run('U06_reject_incomplete',[ctb,'dft','collect','--run','demo/run','--results','demo/incomplete'],expected=2)
    result=json.loads(run('U06_valid_collect',collect))
    for _ in range(3):
        if not result['run']['pending_request_ids']:break
        run('U06_downstream_return',[python,'preview_demo.py','return','--run','demo/run','--output','demo/returned'],computation='authored synthetic files only')
        result=json.loads(run('U06_downstream_collect',collect))
    assert (result['metrics']['n_submitted'],result['metrics']['n_pass'],result['metrics']['n_unknown'])==(4,3,1)
    again=json.loads(run('U06_idempotency',collect));assert again['metrics']==result['metrics']
    run('U06_conflicting_return',[python,'-I','-c',"import json,shutil;from pathlib import Path;shutil.copytree('demo/returned','demo/conflicting');p=next(p for p in Path('demo/conflicting').glob('*/authored-observations.json') if 'eigenvalues' in json.loads(p.read_text())['observations']);d=json.loads(p.read_text());d['observations']['eigenvalues']['value'][0][1]+=1;p.write_text(json.dumps(d))"])
    run('U06_reject_conflict',[ctb,'dft','collect','--run','demo/run','--results','demo/conflicting'],expected=2)
    summaries['synthetic_demo']={'metrics':result['metrics'],'all_scopes_synthetic':all(m['physics_fidelity']=='synthetic' for i in result['results'] if i['input_status']=='valid' for m in i['measurements'].values()),'missing_batch_retained':True,'incomplete_sampling_rejected':True,'conflict_rejected':True}
    assert summaries['synthetic_demo']['all_scopes_synthetic']
    # Only current software evidence is copied; no installed environment payload.
    for directory in ['native-225','native-229','native-full','abacus-unresolved','abacus-missing','highk-blocked','demo']:
        shutil.copytree(work/directory,out/'artifacts'/directory,ignore=shutil.ignore_patterns('cache','.exchange.lock','incomplete','conflicting'))
    report={'schema_version':'ctb.preview.journeys.v1','status':'passed','version':core['version'],'runtime_python':str(python),
        'runtime_checkout_imports':False,'editable_install':False,'dependency_source':'fresh ordinary public-index download; staged outside project',
        'commands':commands,'summaries':summaries,'physical_engine_invocations':0,'science_validated':False}
    (out/'USER_JOURNEY_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','commands':len(commands),'output':str(out),'version':core['version']}))

if __name__=='__main__':main()
