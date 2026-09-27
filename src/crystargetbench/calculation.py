"""Serial scientific execution, verified stage caching, and durable resource leases.

The production constructor accepts only a preflighted CTB worker. In-process
calculators are a deliberately explicit synthetic test interface.
"""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time

from .cache import StageCache, stage_key
from .execution import JobStore
from .geometry import snapshot_from_atoms
from .identity import digest
from .reporting import write_json


class ExecutionContext:
    def __init__(self, *, backend, budget, workdir, cache_root, worker=None,
                 analytic_calculator=None, analytic_test=False):
        if (worker is None) == (analytic_calculator is None):
            raise ValueError('exactly one worker or analytic calculator is required')
        if analytic_calculator is not None and not (analytic_test is True and
                backend.get('family')=='synthetic' and backend.get('id')=='analytic_test'):
            raise ValueError('in-process calculators require explicit synthetic/analytic_test identity')
        if worker is not None and backend.get('family') != ('synthetic' if worker.test_mode else 'mlff'):
            raise ValueError('worker and backend evidence family mismatch')
        self.backend=deepcopy(backend); self.budget=budget
        self.worker=worker; self.analytic_calculator=analytic_calculator
        self.physics_fidelity=backend['family']
        self.evidence_kind='analytic_test' if self.physics_fidelity=='synthetic' else 'mlff'
        self.workdir=Path(workdir); self.workdir.mkdir(parents=True,exist_ok=True)
        self.cache=StageCache(cache_root); self.cache.root.mkdir(parents=True,exist_ok=True)
        self.store=JobStore(self.workdir/'stages.json',digest(backend))
        self.started=time.monotonic()
        self.stats={'calculator_requests':0,'backend_structure_evaluations':0,
                    'backend_forwards':0,'physics_calls':0,'cache_hits':0,
                    'backend_forwards_unknown':0,'physics_calls_confirmed':0,
                    'stage_computations':{},'stage_cache_hits':{}}

    def cached(self,stage,geometry,parameters,dependencies,compute):
        geometry_id=geometry['geometry_id'] if isinstance(geometry,dict) else geometry
        key=stage_key(geometry=geometry_id,recipe='ctb.s2:'+stage,parameters=parameters,
                      backend=self.backend,dependencies=dependencies)
        identity_details={'parameters_digest':digest(parameters),'dependency_digests':dependencies,
                          'cache_decision':'miss_absent_corrupt_or_changed_signature'}
        # The lock covers lookup, work and atomic commit. A killed process leaves
        # no valid payload; a later process can resume only completed records.
        with (self.cache.root/(key+'.lock')).open('a+') as lock:
            while True:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self.budget.check()
                    time.sleep(0.01)
            self.budget.check()
            hit=self.cache.get(key)
            if hit is not None:
                self.stats['cache_hits']+=1
                self.stats['stage_cache_hits'][stage]=self.stats['stage_cache_hits'].get(stage,0)+1
                self.store.record(key,'completed',computation_digest=key,cache_hit=True,stage=stage,
                    **{**identity_details,'cache_decision':'verified_complete_hit'})
                return hit
            self.store.record(key,'running',computation_digest=key,stage=stage,
                              **identity_details)
            try:
                result=compute()
                if not isinstance(result,dict): raise ValueError('stage result must be an object')
                self.stats['stage_computations'][stage]=self.stats['stage_computations'].get(stage,0)+1
                successful=(result.get('calculation_status','completed')=='completed' and
                            result.get('quality_status','accepted') in ('accepted','valid') and
                            result.get('execution_status','completed')=='completed')
                result=deepcopy(result)
                result.setdefault('artifact_digest',digest(result))
                encoded=json.dumps(result,allow_nan=False,sort_keys=True).encode()
                self.budget.check_scratch(len(encoded)*2+1024)
                write_json(self.workdir/(key+'.artifact.json'),result)
                if successful:
                    self.cache.put(key,result)
                    self.store.record(key,'completed',computation_digest=key,stage=stage,
                                      artifact_digest=result['artifact_digest'],**identity_details)
                else:
                    self.store.record(key,'failed',computation_digest=key,stage=stage,
                                      reason='partial or rejected artifact is not a completed cache entry',**identity_details)
                return result
            except BaseException as exc:
                self.store.record(key,'failed',computation_digest=key,stage=stage,
                                  reason=f'{type(exc).__name__}: {exc}',**identity_details)
                raise
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN)

    def evaluate(self,atoms,node_id,mapping=None):
        from .workers.protocol import normalize_efs
        self.stats['calculator_requests']+=1
        snapshot=snapshot_from_atoms(atoms)
        original_mapping=deepcopy(mapping or {'site_mapping':list(range(len(atoms))),
                                    'basis_mapping':[[1,0,0],[0,1,0],[0,0,1]]})
        def compute():
            lease=self.budget.reserve_evaluation(str(node_id),len(atoms),operation_id=digest({
                'geometry':snapshot['geometry_id'],'backend':self.backend,'operation':'energy_forces_stress'}))
            forwards=0; successful=False
            self.stats['backend_structure_evaluations']+=1
            try:
                if self.worker is not None:
                    transport=deepcopy(original_mapping)
                    transport.setdefault('site_mapping',transport.get('supercell_to_unitcell',list(range(len(atoms)))))
                    transport.setdefault('basis_mapping',transport.get('supercell_basis_in_unitcell',[[1,0,0],[0,1,0],[0,0,1]]))
                    raw=self.worker.evaluate(snapshot,node_id,mapping=transport,budget_lease=lease)
                    forwards=raw['cost']['backend_forwards']
                    observation=normalize_efs(raw['energy_eV'],raw['forces_eV_A'],raw['stress_eV_A3'],len(atoms))
                    observation.update({k:deepcopy(raw[k]) for k in
                        ('raw','raw_units','conversions','force_dtype','environment','model_lock_digest','attempt_id','request_id') if k in raw})
                else:
                    from ase.calculators.calculator import all_changes
                    calculator=self.analytic_calculator
                    calculator.calculate(atoms.copy(),['energy','forces','stress'],all_changes)
                    forwards=1
                    observation=normalize_efs(calculator.results['energy'],calculator.results['forces'],
                                              calculator.results.get('stress'),len(atoms))
                self.budget.check()
                observation.update(geometry_id=snapshot['geometry_id'],physics_fidelity=self.physics_fidelity,
                    evidence_kind=self.evidence_kind,provider=self.backend['id'],
                    backend_identity=deepcopy(self.backend),benchmark_eligible=False,science_validated=False,
                    calculation_status='completed')
                successful=True
                return observation
            finally:
                self.budget.release(lease,success=successful,backend_forwards=forwards)
                self.stats['backend_forwards']+=forwards
                if self.worker is not None and not successful and not forwards:
                    # Timeout/exit may hide a forward already started. Keep its
                    # reserved configuration token and report uncertainty.
                    self.stats['backend_forwards_unknown']+=1
                if self.physics_fidelity!='synthetic':
                    self.stats['physics_calls_confirmed']+=forwards
                    self.stats['physics_calls']=None if self.stats['backend_forwards_unknown'] else self.stats['physics_calls_confirmed']
        # Atom order is part of geometry. Metadata and masses do not change the
        # adiabatic potential; the IFC/dynamical-matrix layers bind masses.
        result=self.cached('energy-forces-stress',snapshot['geometry_id'],
                           {'operation':'total_energy_forces_stress','units':'ASE','version':1},[],compute)
        if result['geometry_id']!=snapshot['geometry_id']: raise ValueError('cached force geometry mismatch')
        result=deepcopy(result)
        result.update(node_id=str(node_id),mapping_digest=digest(original_mapping))
        return result

    def calculator(self,node_id):
        from ase.calculators.calculator import Calculator, all_changes
        from ase.stress import full_3x3_to_voigt_6_stress
        import numpy as np
        context=self
        class ContextCalculator(Calculator):
            implemented_properties=['energy','forces','stress']
            def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
                super().calculate(atoms,properties,system_changes)
                observation=context.evaluate(atoms,node_id)
                self.results={'energy':observation['energy_eV'],'forces':np.asarray(observation['forces_eV_A'])}
                if observation['stress_eV_A3'] is not None:
                    self.results['stress']=full_3x3_to_voigt_6_stress(np.asarray(observation['stress_eV_A3']))
        return ContextCalculator()

    def cancel(self):
        self.budget.cancel()
        if self.worker: self.worker.cancel()

    def close(self):
        if self.worker: self.worker.close()
        write_json(self.workdir/'costs.json',{**self.stats,'evidence_kind':self.evidence_kind,
            'elapsed_seconds':time.monotonic()-self.started,'budget':self.budget.snapshot(),
            'benchmark_eligible':False})
