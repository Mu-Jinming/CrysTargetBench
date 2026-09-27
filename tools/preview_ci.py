"""Local equivalent of no-engine CI from only a public source tree.

Optional --dependencies is a fresh download of ordinary public packages, not
a required workspace/private wheelhouse. Omit it to use the normal pip index.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--dependencies')
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='ctb-preview-ci-'));project=temp/'project';shutil.copytree(a.source,project)
    home=temp/'home';home.mkdir();env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):env.pop(key,None)
    env.update(HOME=str(home),PYTHONNOUSERSITE='1',PIP_NO_CACHE_DIR='1',PIP_DISABLE_PIP_VERSION_CHECK='1',OMP_NUM_THREADS='1')
    commands=[]
    def run(argv):
        result=subprocess.run([str(x) for x in argv],cwd=project,env=env,text=True,capture_output=True)
        commands.append({'argv':[str(x) for x in argv],'cwd':str(project),'HOME':str(home),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
        return result.stdout
    run([sys.executable,'-m','venv',temp/'venv']);python=temp/'venv/bin/python'
    extra=['--no-index','--find-links',str(Path(a.dependencies).resolve())] if a.dependencies else []
    run([python,'-m','pip','install',*extra,'build==1.2.2.post1','setuptools==75.8.2','wheel==0.45.1'])
    run([python,'-m','pip','install',*extra,'--no-build-isolation','.[dev,numerical,export]','./examples/dft_adapters/abacus','./tests/plugins/second'])
    run([python,'-I','-c',"import crystargetbench;from pathlib import Path;assert 'site-packages' in str(Path(crystargetbench.__file__));print(crystargetbench.__version__)"])
    run([python,'-m','pytest','-q','--junitxml='+str(out/'public-tree-tests.xml')])
    run([python,'-m','build','--no-isolation'])
    run([python,'-m','build','--no-isolation','--wheel','--outdir','dist','examples/dft_adapters/abacus'])
    run([python,'-m','build','--no-isolation','--wheel','--outdir','dist','tests/plugins/second'])
    wheel=next((project/'dist').glob('crystargetbench-*.whl'))
    release=[python,'-m','tools.s3_release_check','--wheel',wheel,'--abacus-wheel',project/'dist/ctb_abacus_example-0.1.1-py3-none-any.whl',
        '--second-wheel',project/'dist/ctb_second_example-0.1.1-py3-none-any.whl','--output',out/'installed']
    if a.dependencies:release+=['--wheelhouse',str(Path(a.dependencies).resolve())]
    run(release)
    run([python,'-m','tools.preview_export','--output','ci-public-export','--archive','dist/CTB-preview-ci-source.zip'])
    from hashlib import sha256
    record={'status':'passed','public_source_only':True,'editable':False,'HOME':str(home),'project':str(project),
            'source':'allowlisted public tree copy','ordinary_dependency_staging':a.dependencies is not None,
            'private_project_wheelhouse_required':False,'hosted_CI':'not_run','engine_invocations':0,
            'built_distributions':[]}
    for path in (project/'dist').glob('*'):
        if path.suffix in {'.whl','.gz'}:
            dest=out/'built'/path.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(path,dest)
            record['built_distributions'].append({'path':str(dest),'size_bytes':dest.stat().st_size,'sha256':sha256(dest.read_bytes()).hexdigest()})
    (out/'CI_RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'passed','commands':len(commands),'output':str(out)}))

if __name__=='__main__':main()
