"""Persistent CTB budgets and cache semantics, using no physical backend."""
from copy import deepcopy
import json
import os
import time

import pytest

from crystargetbench.budget import Budget, BudgetExceeded, Cancelled, DEFAULT_TEST_LIMITS
from crystargetbench.calculation import ExecutionContext
from crystargetbench.identity import digest
from .analytic_eos import AnalyticEOSCalculator
from .test_relax_eos import atoms_fixture


def make_budget(tmp_path,**changes):
    limits={**DEFAULT_TEST_LIMITS,**changes}
    return Budget(tmp_path/'budget.json',limits,identity=digest(limits),scratch_roots=[tmp_path/'work',tmp_path/'cache'])


def make_context(tmp_path,budget=None,**backend_changes):
    backend={'id':'analytic_test','family':'synthetic','version':'analytic_bm3.v1','assets':{},
             'calculator_parameters':{'b0':0.5,'bprime':4.2},**backend_changes}
    return ExecutionContext(backend=backend,budget=budget or make_budget(tmp_path),
        workdir=tmp_path/'work',cache_root=tmp_path/'cache',analytic_calculator=AnalyticEOSCalculator(),analytic_test=True)


def test_C41_pre_reservation_hard_evaluation_and_concurrency_limits(tmp_path):
    budget=make_budget(tmp_path,max_backend_structure_evaluations=1)
    lease=budget.reserve_evaluation('first',2)
    assert lease['status']=='reserved'
    with pytest.raises(BudgetExceeded,match='concurrent'): budget.reserve_evaluation('second',2)
    assert budget.release(lease,success=True,backend_forwards=1)
    assert not budget.release(lease,success=True,backend_forwards=1)
    with pytest.raises(BudgetExceeded,match='structure_evaluations'): budget.reserve_evaluation('second',2)
    assert budget.snapshot()['backend_forwards']==1


@pytest.mark.parametrize('kind',['unique','displaced','atoms','steps','scratch','total_wall','sample_wall','node_wall'])
def test_C41_all_resource_caps_are_enforced(tmp_path,kind):
    updates={'unique':{'max_unique_structures':0},'displaced':{'max_displaced_structures_total':0},
        'atoms':{'max_supercell_atoms':1},'steps':{'max_relax_steps_per_structure':0},
        'scratch':{'max_scratch_bytes':0},'total_wall':{'max_total_wall_seconds':0},
        'sample_wall':{'max_sample_wall_seconds':0},'node_wall':{}}
    budget=make_budget(tmp_path,**updates[kind])
    with pytest.raises(BudgetExceeded):
        if kind=='unique': budget.start_sample('a')
        elif kind=='displaced': budget.reserve_displacement('d')
        elif kind=='atoms': budget.check_atoms(2)
        elif kind=='steps': budget.record_step('relax')
        elif kind=='scratch': budget.check_scratch(1)
        elif kind=='node_wall':
            with budget.deadline_scope(0): pass
        else: budget.check()
    assert budget.snapshot()['reserved_evaluations']==0


def test_C41_wall_deadline_clamps_worker_lease(tmp_path):
    budget=make_budget(tmp_path)
    with budget.deadline_scope(0.25):
        lease=budget.reserve_evaluation('point',2)
        assert 0<lease['max_wall_seconds']<=0.25
        budget.release(lease,success=True)


def test_C42_cancel_is_persistent_and_no_double_count(tmp_path):
    budget=make_budget(tmp_path)
    budget.cancel()
    resumed=make_budget(tmp_path)
    with pytest.raises(Cancelled): resumed.reserve_evaluation('point',2)
    assert resumed.snapshot()['reserved_evaluations']==0


def test_C42_dead_process_reservation_is_not_refunded(tmp_path):
    budget=make_budget(tmp_path)
    lease=budget.reserve_evaluation('point',2)
    state=budget.snapshot()
    state['leases'][lease['lease_id']]['pid']=2147483647
    (tmp_path/'budget.json').write_text(json.dumps(state))
    resumed=make_budget(tmp_path)
    assert resumed.snapshot()['leases'][lease['lease_id']]['status']=='interrupted'
    assert resumed.snapshot()['reserved_evaluations']==1
    with pytest.raises(BudgetExceeded,match='retries'): resumed.reserve_evaluation('point',2)
    next_lease=resumed.reserve_evaluation('different-point',2)
    resumed.release(next_lease,success=True)


def test_C42_failed_requests_require_explicit_retry_tokens(tmp_path):
    budget=make_budget(tmp_path,max_retries=1)
    first=budget.reserve_evaluation('point',2); budget.release(first,success=False)
    second=budget.reserve_evaluation('point',2); budget.release(second,success=False)
    with pytest.raises(BudgetExceeded,match='retries'): budget.reserve_evaluation('point',2)
    assert budget.snapshot()['reserved_evaluations']==2


def test_C41_retry_tokens_are_aggregate_across_failed_geometries(tmp_path):
    budget=make_budget(tmp_path,max_retries=1)
    for node in ('a','b'):
        lease=budget.reserve_evaluation(node,2); budget.release(lease,success=False)
    lease=budget.reserve_evaluation('a',2); budget.release(lease,success=True)
    with pytest.raises(BudgetExceeded,match='aggregate'): budget.reserve_evaluation('b',2)
    assert budget.snapshot()['retry_evaluations']==1


def test_C37_C40_force_cache_ignores_thresholds_masses_but_not_model(tmp_path):
    context=make_context(tmp_path)
    atoms=atoms_fixture()
    first=context.evaluate(atoms,'a')
    changed=atoms.copy(); changed.set_masses([29,30])
    second=context.evaluate(changed,'b',mapping={'site_mapping':[0,1]})
    assert first['geometry_id']==second['geometry_id']
    assert context.stats['backend_structure_evaluations']==1
    assert second['node_id']=='b' and second['mapping_digest']==digest({'site_mapping':[0,1]})
    assert context.stats['physics_calls']==0 and first['physics_fidelity']=='synthetic'
    changed_model=make_context(tmp_path,version='analytic_bm3.v2')
    changed_model.evaluate(atoms,'c')
    assert changed_model.stats['backend_structure_evaluations']==1


def test_C42_completed_cache_resumes_but_corrupt_or_partial_entries_do_not(tmp_path):
    context=make_context(tmp_path)
    atoms=atoms_fixture()
    first=context.evaluate(atoms,'first')
    context.close()
    resumed=make_context(tmp_path)
    assert resumed.evaluate(atoms,'resume')['energy_eV']==first['energy_eV']
    assert resumed.stats['backend_structure_evaluations']==0
    entry=next((tmp_path/'cache').glob('*.json'))
    payload=json.loads(entry.read_text()); payload['payload']['energy_eV']=123
    entry.write_text(json.dumps(payload))
    resumed.evaluate(atoms,'corrupt')
    assert resumed.stats['backend_structure_evaluations']==1
    payload=json.loads(entry.read_text()); payload['complete']=False
    entry.write_text(json.dumps(payload))
    resumed.evaluate(atoms,'partial')
    assert resumed.stats['backend_structure_evaluations']==2


def test_C45_no_inprocess_calculator_can_impersonate_mattersim(tmp_path):
    with pytest.raises(ValueError,match='synthetic'):
        make_context(tmp_path,family='mlff',id='mattersim',assets={'checkpoint':'a'*64})


def test_C10_C11_C41_public_context_actual_synthetic_worker_and_budget(tmp_path,monkeypatch):
    from .test_assets_worker import client, pair
    worker=client(tmp_path/'worker',monkeypatch)
    budget=make_budget(tmp_path)
    backend={'id':'analytic_test','family':'synthetic','version':'harmonic_pair.v1','assets':{}}
    context=ExecutionContext(backend=backend,budget=budget,workdir=tmp_path/'work',
                             cache_root=tmp_path/'cache',worker=worker)
    mapping={'reference_geometry_id':'a'*64,'supercell_to_unitcell':[0,1],
             'supercell_basis_in_unitcell':[[1,0,0],[0,1,0],[0,0,1]],'ordered_site_ids':['a','b']}
    try:
        result=context.evaluate(pair(),'mapped',mapping=mapping)
        assert result['mapping_digest']==digest(mapping)
        assert result['node_id']=='mapped' and result['physics_fidelity']=='synthetic'
        assert budget.snapshot()['reserved_evaluations']==1
        assert budget.snapshot()['backend_forwards']==1
        repeated=context.evaluate(pair(),'repeat',mapping={'site_mapping':[0,1]})
        assert repeated['energy_eV']==result['energy_eV']
        assert context.stats['backend_structure_evaluations']==1
        assert context.stats['physics_calls']==0
    finally: context.close()


def test_C43_runtime_preflight_failure_preserves_denominator_and_route(tmp_path,monkeypatch):
    from importlib.resources import files
    from crystargetbench import api, scientific
    from crystargetbench.planner import plan
    task=json.loads(files('crystargetbench').joinpath('resources/examples/tasks/bulk_eos_ge100.json').read_text())
    inputs=files('crystargetbench').joinpath('resources/examples/structures/ideal_fcc.cif')
    bad=tmp_path/'broken.json'; bad.write_text('broken')
    planned=plan(task); planned['status']='ready'
    monkeypatch.setattr(api,'plan',lambda *args,**kwargs:deepcopy(planned))
    def denied(*args,**kwargs): raise RuntimeError('test permit changed after plan')
    monkeypatch.setattr(scientific,'production_context',denied)
    result=api.evaluate([inputs,inputs,bad],task,output=tmp_path/'evaluation')
    assert result['metrics']['n_submitted']==3 and result['metrics']['n_unknown']==3
    assert result['plan']['status']=='blocked_execution' and result['run']['physics_calls']==0
    assert {r['family'] for r in result['plan']['routes'].values()}=={'mlff'}
