"""Optional serial launch of a user-installed executable under the durable Budget.

Process completion is not SCF convergence. Collection remains a separate,
engine-free operation. This module is never imported by discovery/prepare/collect.
"""
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time
from ..budget import Budget
from ..identity import digest
from ..reporting import write_json
from .exchange import site_config
from .validation import exact, read_record, validate_request, validate_prepared

def budget_location(config):
    root=Path(config['ledger_root'])
    if not root.is_absolute():raise ValueError('managed ledger_root must be a fixed absolute site path')
    identity=digest({'authorization':config,'execution_kind':'managed_local_dft'})
    return root/identity/'budget.json',identity


def run_local(directory,site,ledger_directory=None):
    directory=Path(directory).resolve();site,manifest=site_config(site)
    config=site['managed']
    if site['execution_mode']!='managed_local' or not isinstance(config,dict):raise ValueError('managed local authorization required')
    exact(config,'enabled authorization_id method_digest adapter_id argv executable_sha256 limits ledger_root','managed local permit')
    if config['enabled'] is not True or not config['authorization_id']:raise ValueError('disabled or unbound managed local permit')
    if config['method_digest']!=digest(site['method']) or config['adapter_id']!=manifest['adapter_id']:raise ValueError('local permit method/adapter binding mismatch')
    argv=config['argv']
    if not isinstance(argv,list) or not argv or any(not isinstance(x,str) or not x or '\x00' in x for x in argv):raise ValueError('fixed argv required')
    executable=Path(argv[0])
    if not executable.is_absolute() or not executable.is_file() or hashlib.sha256(executable.read_bytes()).hexdigest()!=config['executable_sha256']:
        raise ValueError('owned executable fingerprint mismatch')
    request=validate_request(read_record(directory/'request.json'))
    if request['execution_mode']!='managed_local' or request['method']!=site['method'] or request['adapter']!={'id':manifest['adapter_id'],'version':manifest['adapter_version']}:
        raise ValueError('local request binding mismatch')
    prepared=validate_prepared(read_record(directory/'prepared.json'),request,directory)
    if prepared['status']!='ready':raise ValueError('prepared job is not ready; assets/capability unresolved')
    ledger,identity=budget_location(config)
    if ledger_directory is not None and Path(ledger_directory).resolve()!=Path(config['ledger_root']).resolve():
        raise ValueError('ledger directory differs from the bound site ledger_root')
    budget=Budget(ledger,config['limits'],identity=identity,scratch_roots=[directory],execution_kind='managed_local_dft')
    budget.start_sample(request['sample_geometry_id'])
    record_path=directory/'local-attempt.json'
    if record_path.exists():
        previous=read_record(record_path)
        if previous['request_digest']!=request['request_digest'] or previous['budget_identity']!=identity:raise ValueError('local attempt belongs to another request/authorization')
        if previous['status']=='process_completed':return previous
    lease=budget.reserve_evaluation('external-dft',len(request['geometry']['species']),operation_id=request['request_digest'])
    record={'schema_version':'ctb.local_attempt.v1','request_digest':request['request_digest'],
        'budget_identity':identity,'lease_id':lease['lease_id'],'argv':argv,'shell':False,
        'status':'reserved','returncode':None,'engine_invocations':0,'scientific_validation':False}
    write_json(record_path,record);process=None;success=False
    try:
        with (directory/'launcher.stdout').open('wb') as out,(directory/'launcher.stderr').open('wb') as err:
            process=subprocess.Popen(argv,cwd=directory,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                     shell=False,start_new_session=True)
            record.update(status='running',engine_invocations=1);write_json(record_path,record)
            with budget.deadline_scope(lease['max_wall_seconds']):
                while process.poll() is None:
                    budget.check();time.sleep(0.025)
                budget.check()
            record['returncode']=process.returncode
            success=process.returncode==0
            record['status']='process_completed' if success else 'process_failed'
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        record.update(status='interrupted_or_failed',error=str(exc))
        raise
    finally:
        budget.release(lease,success=success,backend_forwards=record['engine_invocations'])
        write_json(record_path,record)
    return record
