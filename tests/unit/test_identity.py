"""Scientific comparison identity is distinct from a submitted batch."""
from crystargetbench.contracts import _resource_json, default_policy
from crystargetbench.identity import digest
from crystargetbench.planner import plan
from crystargetbench.structures import submission_digest


def test_M09_mlff_and_dft_comparison_tracks_differ():
    task = _resource_json('examples','tasks','phonon_path.json')
    mlff = plan(task,default_policy('mlff'))
    dft = plan(task,default_policy('dft'))
    assert mlff['comparability_digest'] != dft['comparability_digest']
    assert mlff['protocol_digest'] != dft['protocol_digest']
    assert mlff['physics_fidelity'] == 'mlff'
    assert dft['physics_fidelity'] == 'dft'
    assert mlff['physics_calls'] == dft['physics_calls'] == 0


def test_M10_distinct_submissions_share_scientific_comparison_identity():
    task = _resource_json('examples','tasks','spacegroup_225.json')
    first = [{'item_id':'item-000000','file_sha256':digest('synthetic-bytes-a'),
              'input_status':'valid','status':'valid','atom_count':2}]
    second = [{'item_id':'item-000000','file_sha256':digest('synthetic-bytes-b'),
               'input_status':'valid','status':'valid','atom_count':4}]
    left, right = plan(task,submission=first), plan(task,submission=second)
    assert left['comparability_digest'] == right['comparability_digest']
    assert submission_digest(first) != submission_digest(second)
    left_run=digest({'submission':submission_digest(first),'comparison':left['comparability_digest']})
    right_run=digest({'submission':submission_digest(second),'comparison':right['comparability_digest']})
    assert left_run != right_run
