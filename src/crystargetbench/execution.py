"""Persistent local stage state and test-double DAG execution.

Production S1 invokes only native analysis. This runner is for explicit injected
synthetic test adapters; no worker process, scheduler, or model is launched.
"""
from pathlib import Path
from .reporting import write_json
from .contracts import load_json


class JobStore:
    def __init__(self, path, run_digest):
        self.path = Path(path)
        self.state = {'run_digest': run_digest, 'stages': {}}
        if self.path.exists():
            old = load_json(self.path)
            if old.get('run_digest') == run_digest:
                self.state = old

    def record(self, node_id, status, **details):
        if status not in {'pending', 'running', 'completed', 'failed', 'skipped', 'blocked'}:
            raise ValueError('invalid execution status')
        previous = self.state['stages'].get(node_id, {})
        entry = {**details, 'status': status, 'attempts': previous.get('attempts', 0) + (status == 'running')}
        self.state['stages'][node_id] = entry
        write_json(self.path, self.state)


def execute_synthetic(nodes, backend, store):
    if getattr(backend, 'family', None) != 'synthetic':
        raise ValueError('S1 test executor accepts only explicitly synthetic adapters')
    results = {}
    for node in nodes:
        key = node['node_id']
        if any(dep not in results for dep in node.get('dependencies', [])):
            raise ValueError('DAG must be topologically sorted and all dependencies declared')
        deps = [results[d] for d in node.get('dependencies', [])]
        if any(d['calculation_status'] != 'completed' for d in deps):
            result = {'calculation_status': 'skipped', 'value': None, 'reason': 'dependency_not_completed'}
        elif (any(results[g].get('decision') != 'pass' for g in node.get('gate_dependencies', []))
              or any(results[c['gate_node']].get('decision') != c['required_verdict']
                     for c in node.get('conditions', []))):
            result = {'calculation_status': 'skipped', 'value': None, 'reason': 'gate_not_passed'}
        else:
            store.record(key, 'running', backend_family='synthetic')
            try:
                result = backend.execute(node, deps)
            except Exception as exc:
                result = {'calculation_status': 'failed', 'value': None,
                          'reason': type(exc).__name__, 'retry_provider': None}
        result.update(backend_family='synthetic', physics_fidelity='synthetic', benchmark_eligible=False)
        results[key] = result
        store.record(key, result['calculation_status'], result=result)
    return results
