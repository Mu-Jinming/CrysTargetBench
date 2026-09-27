"""Durable, serial local resource leases. No resource limit grants permission."""
from contextlib import contextmanager
import fcntl
import json
import math
import hashlib
import os
from pathlib import Path
import time
import uuid

from .identity import digest
from .reporting import write_json


class BudgetExceeded(RuntimeError):
    pass


class Cancelled(BudgetExceeded):
    pass


LEDGER_VERSION='ctb.budget.v2'
TIME_POLICY='first_sample_start_to_deadline_including_interruptions.v1'


class LedgerMigrationRequired(BudgetExceeded):
    pass


DEFAULT_TEST_LIMITS = {
    'max_unique_structures': 16, 'max_supercell_atoms': 128,
    'max_displaced_structures_total': 128, 'max_backend_structure_evaluations': 5000,
    'max_concurrent_workers': 1, 'max_relax_steps_per_structure': 600,
    'max_total_wall_seconds': 120, 'max_scratch_bytes': 134217728,
    'max_retries': 0, 'max_dft_jobs': 0,
    'max_sample_wall_seconds': 60, 'per_request_wall_seconds': 10,
}


class Budget:
    def __init__(self, path, limits, *, identity, scratch_roots=(), predecessor_ledger=None, execution_kind="mlff"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.limits=dict(limits); self.identity=identity
        required=set(DEFAULT_TEST_LIMITS)-{'max_sample_wall_seconds','per_request_wall_seconds'}
        if not required<=self.limits.keys() or set(self.limits)-set(DEFAULT_TEST_LIMITS):
            raise ValueError('budget requires all hard limits and rejects unknown fields')
        if any(type(v) is not int or v<0 for v in self.limits.values()):
            raise ValueError('resource limits must be nonnegative integer tokens')
        self.execution_kind=execution_kind
        if execution_kind not in {'mlff','managed_local_dft'}:raise ValueError('unknown budget execution kind')
        if (execution_kind=='mlff' and self.limits['max_dft_jobs']!=0) or self.limits['max_concurrent_workers']!=1:
            raise ValueError('S2 executor is serial and never authorizes DFT jobs')
        self.scratch_roots=[Path(p).resolve() for p in scratch_roots]
        self.started=time.monotonic(); self.started_wall=time.time()
        if not all(math.isfinite(v) and v>=0 for v in (self.started,self.started_wall)):
            raise BudgetExceeded('invalid_clock')
        self.last_monotonic=self.started
        self.lineage=None
        if predecessor_ledger is not None:
            from .contracts import load_json
            predecessor=Path(predecessor_ledger)
            prior=load_json(predecessor)
            if predecessor.resolve()==self.path.resolve() or prior['identity']==identity:
                raise ValueError('new authorization lineage requires a distinct identity and ledger')
            self.lineage={'previous_identity':prior['identity'],
                          'previous_ledger_sha256':hashlib.sha256(predecessor.read_bytes()).hexdigest()}
        self.deadlines=[]
        self.sample_id=None
        with self._locked() as state:
            for lease in state['leases'].values():
                if lease['status']=='active' and not self._alive(lease['pid']):
                    lease.update(status='interrupted',completed_wall=time.time())

    @staticmethod
    def _alive(pid):
        try: os.kill(pid,0); return True
        except ProcessLookupError: return False
        except PermissionError: return True

    @contextmanager
    def _locked(self):
        lock=self.path.with_suffix('.lock')
        with lock.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            if self.path.exists():
                from .contracts import load_json
                state=load_json(self.path)
                if state.get('schema_version')!=LEDGER_VERSION or state.get('time_accounting_policy')!=TIME_POLICY:
                    raise LedgerMigrationRequired('legacy/unknown ledger cannot prove sample time; preserve it and require explicit new authorization with lineage')
                if not {'samples','last_observed_wall','last_observed_monotonic','accounted_wall','clock_error'}<=state.keys():
                    raise LedgerMigrationRequired('incomplete v2 time records; no automatic time reset')
                timestamps=[state.get(k) for k in ('created_wall','last_observed_wall','last_observed_monotonic','accounted_wall')]
                if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in timestamps)
                        or not isinstance(state['samples'],dict) or set(state['structures'])!=set(state['samples'])):
                    raise LedgerMigrationRequired('invalid persisted timing records')
                for sample in state['samples'].values():
                    if (not isinstance(sample,dict) or set(sample)!={'started_wall','deadline_wall'} or
                        any(type(v) not in (int,float) or not math.isfinite(v) for v in sample.values()) or
                        sample['started_wall']<state['created_wall'] or
                        sample['deadline_wall']!=sample['started_wall']+self.limits.get('max_sample_wall_seconds',self.limits['max_total_wall_seconds'])):
                        raise LedgerMigrationRequired('invalid persisted sample deadline')
                if state['identity']!=self.identity or state['limits']!=self.limits:
                    raise ValueError('budget ledger identity or limits mismatch')
                if self.lineage is not None and self.lineage!=state.get('authorization_lineage'):
                    raise ValueError('authorization lineage mismatch')
            else:
                state={'schema_version':LEDGER_VERSION,'identity':self.identity,'limits':self.limits,
                       'created_wall':self.started_wall,'cancelled':False,'structures':[],
                       'time_accounting_policy':TIME_POLICY,'samples':{},
                       'last_observed_wall':self.started_wall,'accounted_wall':self.started_wall,
                       'last_observed_monotonic':self.started,
                       'clock_error':None,'authorization_lineage':self.lineage,
                       'displacements':[], 'steps':{}, 'leases':{}, 'reserved_evaluations':0,
                       'backend_forwards':0,'completed_evaluations':0,'failed_evaluations':0,
                       'retry_evaluations':0}
            try:
                yield state
            finally:
                write_json(self.path,state)
                fcntl.flock(stream,fcntl.LOCK_UN)

    def _time_remaining(self,state,*,check_sample=True):
        if state['clock_error']: raise BudgetExceeded(state['clock_error'])
        wall,mono=time.time(),time.monotonic()
        if (not all(math.isfinite(v) for v in (wall,mono)) or
                wall<state['last_observed_wall'] or mono<max(self.last_monotonic,state['last_observed_monotonic'])):
            state['clock_error']='clock_rollback_or_nonfinite; explicit review/new authorization required'
            raise BudgetExceeded(state['clock_error'])
        self.last_monotonic=mono
        state['last_observed_wall']=wall
        now=max(wall,state['accounted_wall']+mono-state['last_observed_monotonic'])
        state['last_observed_monotonic']=mono
        state['accounted_wall']=now
        sample=state['samples'].get(self.sample_id)
        if self.sample_id is not None and check_sample and sample is None:
            raise LedgerMigrationRequired('sample time record missing; cannot reconstruct zero elapsed')
        sample_deadline=(sample['deadline_wall'] if sample else state['created_wall']+
                         self.limits.get('max_sample_wall_seconds',self.limits['max_total_wall_seconds']))
        return {'global':state['created_wall']+self.limits['max_total_wall_seconds']-now,
                'sample':sample_deadline-now if check_sample else None,
                'node':min(self.deadlines)-mono if self.deadlines else None}

    def _check(self,state,*,check_sample=True):
        if state['cancelled']: raise Cancelled('budget cancelled')
        remaining=self._time_remaining(state,check_sample=check_sample)
        state['last_remaining_seconds']={'sample_id':self.sample_id,**remaining}
        if remaining['global']<=0:
            raise BudgetExceeded('max_total_wall_seconds exhausted')
        if check_sample and remaining['sample']<=0:
            raise BudgetExceeded('max_sample_wall_seconds exhausted')
        if remaining['node'] is not None and remaining['node']<=0:
            raise BudgetExceeded('node_wall_seconds exhausted')
        if self.scratch_bytes()>self.limits['max_scratch_bytes']:
            raise BudgetExceeded('max_scratch_bytes exceeded')
        return remaining

    def scratch_bytes(self):
        seen=set(); total=0
        for root in self.scratch_roots:
            if root.exists():
                for p in root.rglob('*'):
                    if p.is_file() and not p.is_symlink():
                        resolved=p.resolve()
                        if resolved not in seen: total+=p.stat().st_size; seen.add(resolved)
        return total

    def check(self):
        with self._locked() as state: self._check(state)

    @contextmanager
    def deadline_scope(self, seconds):
        deadline=time.monotonic()+seconds
        self.deadlines.append(deadline)
        try:
            self.check()
            yield
        finally:
            self.deadlines.remove(deadline)

    def check_scratch(self,additional_bytes):
        with self._locked() as state:
            self._check(state)
            if self.scratch_bytes()+additional_bytes>self.limits['max_scratch_bytes']:
                raise BudgetExceeded('artifact would exceed max_scratch_bytes')

    def check_atoms(self,count):
        if type(count) is not int or count<1 or count>self.limits['max_supercell_atoms']:
            raise BudgetExceeded('max_supercell_atoms exceeded')
        self.check()

    def start_sample(self,geometry_id):
        if not isinstance(geometry_id,str) or not geometry_id: raise ValueError('sample identity required')
        with self._locked() as state:
            self._check(state,check_sample=False)
            if geometry_id not in state['structures']:
                if len(state['structures'])>=self.limits['max_unique_structures']:
                    raise BudgetExceeded('max_unique_structures exhausted')
                state['structures'].append(geometry_id)
                now=state['accounted_wall']
                state['samples'][geometry_id]={'started_wall':now,'deadline_wall':now+
                    self.limits.get('max_sample_wall_seconds',self.limits['max_total_wall_seconds'])}
            self.sample_id=geometry_id
            self._check(state)

    def reserve_displacement(self,displacement_id):
        with self._locked() as state:
            self._check(state)
            if displacement_id not in state['displacements']:
                if len(state['displacements'])>=self.limits['max_displaced_structures_total']:
                    raise BudgetExceeded('max_displaced_structures_total exhausted')
                state['displacements'].append(displacement_id)

    def record_step(self,node_id):
        with self._locked() as state:
            self._check(state)
            key=f'{self.sample_id}:{node_id}'
            steps=state['steps'].get(key,0)
            if steps>=self.limits['max_relax_steps_per_structure']:
                raise BudgetExceeded('max_relax_steps_per_structure exhausted')
            state['steps'][key]=steps+1

    def reserve_evaluation(self,node_id,atom_count,operation_id=None):
        self.check_atoms(atom_count)
        with self._locked() as state:
            remaining=self._check(state)
            if any(l['status']=='active' for l in state['leases'].values()):
                raise BudgetExceeded('max_concurrent_workers lease held')
            if self.execution_kind=='managed_local_dft' and state['reserved_evaluations']>=self.limits['max_dft_jobs']:
                raise BudgetExceeded('max_dft_jobs exhausted')
            if state['reserved_evaluations']>=self.limits['max_backend_structure_evaluations']:
                raise BudgetExceeded('max_backend_structure_evaluations exhausted')
            operation_id=operation_id or node_id
            failures=sum(l.get('operation_id',l['node_id'])==operation_id and l['status'] in {'failed','interrupted'}
                         for l in state['leases'].values())
            if failures>self.limits['max_retries']:
                raise BudgetExceeded('max_retries exhausted for this calculation identity')
            if failures and state.get('retry_evaluations',0)>=self.limits['max_retries']:
                raise BudgetExceeded('max_retries aggregate tokens exhausted')
            lease={'lease_id':uuid.uuid4().hex,'budget_identity':self.identity,
                   'reservation_index':state['reserved_evaluations']+1,
                   'sample_id':self.sample_id,'remaining_seconds':remaining,
                   'operation_id':operation_id,
                   'node_id':node_id,'atom_count':atom_count,'pid':os.getpid(),'status':'active',
                   'reserved_wall':time.time(),'max_wall_seconds':min(self.limits.get('per_request_wall_seconds',60),
                       remaining['sample'],remaining['node'] if remaining['node'] is not None else float('inf'),
                       remaining['global'])}
            state['leases'][lease['lease_id']]=lease
            state['reserved_evaluations']+=1
            if failures: state['retry_evaluations']=state.get('retry_evaluations',0)+1
            return {**lease, 'status':'reserved'}

    def release(self,lease,*,success,backend_forwards=0):
        with self._locked() as state:
            record=state['leases'].get(lease['lease_id'])
            if not record or record['status']!='active': return False
            record.update(status='completed' if success else 'failed',completed_wall=time.time())
            state['completed_evaluations' if success else 'failed_evaluations']+=1
            state['backend_forwards']+=int(backend_forwards)
            return True

    def cancel(self):
        with self._locked() as state: state['cancelled']=True

    def snapshot(self):
        with self._locked() as state: return json.loads(json.dumps(state))
