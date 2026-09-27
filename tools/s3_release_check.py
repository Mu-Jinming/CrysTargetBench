"""Build-independent isolated wheel verification; installs no physical engine.

python -m tools.s3_release_check --wheel dist/...whl --abacus-wheel ...
    --second-wheel ... --wheelhouse wheelhouse --output docs/s3/release
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--wheel',required=True);p.add_argument('--abacus-wheel',required=True)
    p.add_argument('--second-wheel',required=True);p.add_argument('--wheelhouse');p.add_argument('--output',required=True)
    a=p.parse_args();root=Path.cwd();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    wheels=[Path(x).resolve() for x in (a.wheel,a.abacus_wheel,a.second_wheel)]
    envroot=Path(tempfile.mkdtemp(prefix='ctb-s3-wheel-'));env=envroot/'venv'
    commands=[]
    def run(argv,*,cwd=envroot):
        clean=os.environ.copy();clean.pop('PYTHONPATH',None);clean['PYTHONNOUSERSITE']='1';clean['OMP_NUM_THREADS']='1';clean['PIP_DISABLE_PIP_VERSION_CHECK']='1';clean['PIP_NO_CACHE_DIR']='1'
        result=subprocess.run([str(x) for x in argv],cwd=cwd,env=clean,text=True,capture_output=True)
        commands.append({'argv':[str(x) for x in argv],'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if result.returncode:raise RuntimeError(result.stderr+'\n'+result.stdout)
        return result.stdout
    run([sys.executable,'-m','venv',env]);python=env/'bin/python'
    extra=['--no-index','--find-links',Path(a.wheelhouse).resolve()] if a.wheelhouse else []
    run([python,'-m','pip','install',*extra,wheels[0]])
    probe=r'''
from importlib import metadata,resources
from pathlib import Path
import hashlib,json,sys
import crystargetbench
from crystargetbench.api import evaluate
from crystargetbench.contracts import load_json
assert crystargetbench.__version__==metadata.version('crystargetbench')
core=Path(crystargetbench.__file__).parent
hashes={str(p.relative_to(core)):hashlib.sha256(p.read_bytes()).hexdigest() for p in core.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.cif'}}
resource=resources.files('crystargetbench').joinpath('resources/examples')
bad=Path('bad.cif');bad.write_text('authored invalid structure')
result=evaluate([resource.joinpath('structures/ideal_fcc.cif'),resource.joinpath('structures/ideal_bcc.cif'),resource.joinpath('structures/ideal_fcc.cif'),bad],load_json(resource.joinpath('tasks/spacegroup_225.json')),output='native')
assert [r['measurements']['sg']['value'] for r in result['results']]==[225,229,225,None]
assert result['metrics']['n_submitted']==4 and result['metrics']['n_pass']==2 and result['metrics']['n_fail']==1 and result['metrics']['n_unknown']==1
assert result['metrics']['distributions']['sg']['histogram']=={'225':2,'229':1}
for package in ['mattersim','torch','phonopy','abacus','atomate2','jobflow','pyatb']:
    assert package not in sys.modules
installed={d.metadata['Name']:d.version for d in metadata.distributions()}
assert all(x not in {k.lower() for k in installed} for x in ['torch','mattersim','phonopy','atomate2','jobflow','pyatb'])
print(json.dumps({'version':crystargetbench.__version__,'core_path':str(core),'hashes':hashes,'installed':installed,'native_metrics':result['metrics'],'physics_calls':result['run']['physics_calls']}))
'''
    core=json.loads(run([python,'-I','-c',probe]));version=run([env/'bin/ctb','--version']).strip();assert version=='ctb '+core['version']
    # Exact package bytes, not just matching version strings.
    with zipfile.ZipFile(wheels[0]) as z:
        for relative,checksum in core['hashes'].items():
            member='crystargetbench/'+relative
            assert hashlib.sha256(z.read(member)).hexdigest()==checksum
            assert (root/'src'/member).read_bytes()==z.read(member),relative
    run([python,'-m','pip','install',*extra,wheels[1],wheels[2]])
    plugin=json.loads(run([python,'-I','-c',r'''
from pathlib import Path
import hashlib,json,sys,crystargetbench
from crystargetbench.dft.registry import discover
core=Path(crystargetbench.__file__).parent
hashes={str(p.relative_to(core)):hashlib.sha256(p.read_bytes()).hexdigest() for p in core.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.cif'}}
manifests=discover()
assert {'abacus_example','second_example'}<=manifests.keys()
assert 'ctb_abacus_example.adapter' not in sys.modules and 'ctb_second_example.adapter' not in sys.modules
print(json.dumps({'hashes':hashes,'manifests':manifests}))
''']));assert core['hashes']==plugin['hashes']
    run([python,'-m','pip','install',*extra,'pytest==8.3.5','phonopy==2.38.2','seekpath==2.1.0','scipy==1.17.1'])
    # Current tests against installed wheels, no source PYTHONPATH.
    run([python,'-m','pytest',str(root/'tests'),'-q','--junitxml='+str(out/'installed-tests.xml')],cwd=root)
    lock=json.loads(run([python,'-I','-c',"import hashlib,json,platform,sys; from importlib import metadata; print(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':{d.metadata['Name']:{'version':d.version,'record_sha256':hashlib.sha256((d.read_text('RECORD') or '').encode()).hexdigest()} for d in metadata.distributions()}}))"]))
    data={'schema_version':'ctb.s3.release_check.v1','core_only':core,'cli_version':version,
        'core_unchanged_after_plugin_install':True,'source_wheel_installed_bytes_identical':True,
        'independent_plugins':plugin['manifests'],'installed_regression':'passed',
        'final_environment':lock,'engine_or_model_invocations':0,'engine_installations':0,
        'science_validated':False,'environment_python':str(python)}
    (out/'RELEASE_CHECK.json').write_text(json.dumps(data,indent=2)+'\n')
    (out/'environment.lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    # Evidence only, not private wheel/environment payloads.
    import shutil
    shutil.copytree(envroot/'native',out/'native',dirs_exist_ok=True)
    print(json.dumps({'status':'passed','record':str(out/'RELEASE_CHECK.json'),'python':str(python)}))

if __name__=='__main__':main()
