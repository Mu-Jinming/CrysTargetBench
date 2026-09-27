"""R1 shared planning/execution order using actual analytic recipes and native SG."""
from copy import deepcopy
import pytest
from crystargetbench.budget import Budget,DEFAULT_TEST_LIMITS
from crystargetbench.calculation import ExecutionContext
from crystargetbench.identity import digest
from crystargetbench.planner import plan
from crystargetbench.scientific import execute_submission
from crystargetbench.scheduling import measurement_schedule
from tools.generate_s2_numerical import numerical_configuration,write_fixture_submission
from tests.s2.analytic_eos import AnalyticLatticeBM3


def context(root,maximum,cache=None):
    backend={'id':'analytic_test','family':'synthetic','version':'s2a1-analytic-bm3','assets':{}}
    limits={**DEFAULT_TEST_LIMITS,'max_backend_structure_evaluations':maximum,
            'max_sample_wall_seconds':120,'max_total_wall_seconds':240,'max_relax_steps_per_structure':1200}
    return ExecutionContext(backend=backend,budget=Budget(root/'ledger.json',limits,identity=digest(backend)),
        workdir=root/'work',cache_root=cache or root/'cache',analytic_calculator=AnalyticLatticeBM3(),analytic_test=True)


def configuration(order):
    t,r=numerical_configuration(bulk_threshold=100)
    t['measurements']['sg']={'property_id':'space_group_number','unit':'1'}
    r['routes']['sg']={'family':'native','provider':'symmetry','definition_id':'native.sg.test'}
    r['protocol_lock']['protocol']['parameters'].update(symprec_A=1e-5,angle_tolerance_deg=-1.,target_symmetrization=False)
    t['measurements']={a:t['measurements'][a] for a in order}
    r['protocol_lock']['scheduling']=measurement_schedule(t)
    return t,r


@pytest.mark.parametrize('maximum,warm',[(1,False),(1000,False),(1,True)])
def test_R1_reordering_cold_warm_tight_and_generous_budgets(tmp_path,maximum,warm):
    items=write_fixture_submission(tmp_path/'inputs'); records=[];plans=[]
    for i,order in enumerate([['B','sg','phonon'],['sg','phonon','B']]):
        task,resolved=configuration(order);plans.append(plan(task))
        root=tmp_path/str(i);cache=root/'cache'
        if warm:
            pre=context(root/'pre',1000,cache)
            execute_submission(task,items[:1],resolved,pre)
        ctx=context(root/'actual',maximum,cache)
        results,metrics,stats=execute_submission(task,items,resolved,ctx)
        ledger=ctx.budget.snapshot()
        records.append({'decisions':[x['decision'] for x in results],
            'measurement_statuses':[{a:(v['value'],v['calculation_status'],v['quality_status']) for a,v in x['measurements'].items()} for x in results],
            'leases':[(l['node_id'],l['operation_id'],l['status']) for l in
                      sorted(ledger['leases'].values(),key=lambda l:l['reservation_index'])],
            'stats':stats,'steps':ledger['steps']})
        assert metrics['n_submitted']==3
        assert len(ledger['structures'])==1  # duplicate items retain denominator, one billed sample
        assert results[0]['scheduling']==plans[-1]['protocol_lock']['scheduling']
        assert list(results[0]['measurements'])==['phonon','sg','B']
        assert results[0]['measurements']['sg']['physics_fidelity']=='native'
        if warm:assert stats['backend_structure_evaluations']==0
        expected='unknown' if maximum==1 and not warm else 'fail'
        assert results[0]['decision']==expected
    assert records[0]==records[1]
    assert plans[0]['comparability_digest']==plans[1]['comparability_digest']
    assert plans[0]['nodes']==plans[1]['nodes']
    planned_measurements=[n['measurement_alias'] for n in plans[0]['nodes'] if 'measurement_alias' in n]
    assert planned_measurements==['phonon','sg','B']


def test_R1_unqualified_reference_gates_all_branches_independently_of_keys(tmp_path):
    items=write_fixture_submission(tmp_path/'inputs')
    for i,order in enumerate([['sg','B','phonon'],['B','phonon','sg']]):
        t,r=configuration(order);ctx=context(tmp_path/str(i),0)
        results,metrics,stats=execute_submission(t,items,r,ctx)
        assert metrics['n_unknown']==3 and metrics['n_submitted']==3
        assert all(x['reference_geometry'] is None for x in results)
        assert not any(k.startswith('eos') or 'phonon' in k for k in stats['stage_computations'])
        assert stats['backend_structure_evaluations']==0


def test_R1_independent_phonon_scopes_not_short_circuited_by_each_other():
    t,_=configuration(['phonon','B','sg'])
    t['measurements']['path']={'property_id':'phonon_path_min_frequency_THz','unit':'THz'}
    t['constraints'].append({'id':'path_stability','measurement':'path','operator':'ge','value':0.})
    p=plan(t)
    displacement=[n for n in p['nodes'] if n['kind']=='finite_displacements']
    assert len(displacement)==2
    assert all(not any(d.startswith('phonon_gate:') for d in n['dependencies']) for n in displacement)


def test_R1_executor_rejects_altered_locked_schedule(tmp_path):
    items=write_fixture_submission(tmp_path/'inputs');t,r=configuration(['phonon','B','sg'])
    r['protocol_lock']['scheduling']['measurement_order'].reverse()
    ctx=context(tmp_path/'run',100)
    with pytest.raises(ValueError,match='scheduling differs'):execute_submission(t,items,r,ctx)
    assert ctx.stats['backend_structure_evaluations']==0


def test_R2_expired_duplicate_submissions_keep_full_denominator(tmp_path,monkeypatch):
    from crystargetbench import budget as module
    clock=[1000.]
    monkeypatch.setattr(module.time,'time',lambda:clock[0])
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    items=write_fixture_submission(tmp_path/'inputs');t,r=configuration(['B','sg','phonon'])
    root=tmp_path/'run';ctx=context(root,100)
    ctx.budget.start_sample(items[0]['input_geometry']['geometry_id'])
    clock[0]+=121
    restored=context(root,100)
    results,metrics,stats=execute_submission(t,items,r,restored)
    assert metrics['n_submitted']==3 and metrics['n_unknown']==3
    assert stats['backend_structure_evaluations']==0
    assert len(restored.budget.snapshot()['samples'])==1


def test_R1_legacy_production_plan_rejected_before_assets_or_workers(tmp_path):
    from crystargetbench.scientific import production_context
    t,r=configuration(['phonon','B','sg'])
    r.update(status='ready',physics_fidelity='mlff')
    del r['protocol_lock']['scheduling']
    with pytest.raises(ValueError,match='replan legacy'):
        production_context(t,[],r,{},tmp_path)
