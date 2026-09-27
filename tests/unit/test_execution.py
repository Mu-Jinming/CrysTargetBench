import pytest
from crystargetbench.execution import JobStore, execute_synthetic
from crystargetbench.contracts import load_json


class SpyBackend:
    family = 'synthetic'
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
    def execute(self, node, deps):
        self.calls.append(node['node_id'])
        if self.fail:
            raise MemoryError('out_of_memory')
        return {'calculation_status':'completed', 'value':1, 'decision':'fail'}


def test_gate_skip_and_persisted_state(tmp_path):
    backend = SpyBackend()
    store = JobStore(tmp_path/'state.json','run1')
    nodes = [{'node_id':'gap'}, {'node_id':'dielectric','dependencies':['gap'],'gate_dependencies':['gap']}]
    result = execute_synthetic(nodes,backend,store)
    assert backend.calls == ['gap']
    assert result['dielectric']['calculation_status'] == 'skipped'
    assert result['gap']['backend_family'] == 'synthetic'
    assert result['gap']['benchmark_eligible'] is False
    replay = JobStore(tmp_path/'state.json','run1')
    assert replay.state['stages']['gap']['status'] == 'completed'
    assert replay.state['stages']['gap']['attempts'] == 1


def test_failure_never_reroutes_or_chooses_best_value(tmp_path):
    backend = SpyBackend(fail=True)
    result = execute_synthetic([{'node_id':'phonon'}],backend,JobStore(tmp_path/'s.json','r'))
    assert backend.calls == ['phonon']
    assert result['phonon']['retry_provider'] is None
    assert result['phonon']['value'] is None


def test_physics_backend_cannot_use_synthetic_executor(tmp_path):
    backend = SpyBackend()
    backend.family = 'dft'
    with pytest.raises(ValueError):
        execute_synthetic([],backend,JobStore(tmp_path/'s.json','r'))


def test_planner_conditions_unknown_gate_skips_downstream(tmp_path):
    class UnknownGate(SpyBackend):
        def execute(self,node,deps):
            self.calls.append(node['node_id'])
            return {'calculation_status':'completed','decision':'unknown','value':None}
    backend = UnknownGate()
    nodes = [{'node_id':'gap_gate'}, {'node_id':'forces', 'dependencies':['gap_gate'],
            'conditions':[{'gate_node':'gap_gate','required_verdict':'pass'}]}]
    result = execute_synthetic(nodes,backend,JobStore(tmp_path/'s.json','r'))
    assert backend.calls == ['gap_gate']
    assert result['forces']['calculation_status'] == 'skipped'


def test_actual_highk_planner_dag_mock_gate(tmp_path):
    from crystargetbench.contracts import _resource_json
    from crystargetbench.planner import plan
    highk = _resource_json('examples','tasks','highk_screen.json')
    planned = plan(highk)
    backend = SpyBackend()
    result = execute_synthetic(planned['nodes'],backend,JobStore(tmp_path/'s.json','r'))
    assert 'tight_relax' in backend.calls
    assert 'scf_gap' in backend.calls
    assert 'gap_gate' in backend.calls
    assert not any(name.startswith(('forces:', 'ionic:', 'electronic:')) for name in backend.calls)
    assert result['forces:phonon']['calculation_status'] == 'skipped'
    assert all(m['physics_fidelity'] == 'synthetic' and not m['benchmark_eligible'] for m in result.values())
