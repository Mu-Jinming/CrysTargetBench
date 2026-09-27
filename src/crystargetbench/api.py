"""Independent black-box API. Physics planning never implies execution."""
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from .contracts import load_json, validate_task, default_protocol
from .identity import digest
from .planner import plan
from .structures import load_submission, submission_digest
from .reporting import write_json
from .execution import JobStore
from .assessment import assess, make_measurement
from .metrics import aggregate
from .cache import StageCache, stage_key
from .native import analyze_symmetry, symmetry_measurement, effective_symmetry_parameters


def _data(value):
    return load_json(value) if isinstance(value, (str, Path)) else value


def evaluate(structures, task, *, output, policy=None, protocol=None, deployment=None,
             dry_run=False, cache_root=None):
    task = validate_task(_data(task))
    policy, protocol, deployment = _data(policy), _data(protocol), _data(deployment)
    if deployment and deployment.get('schema_version') == 'ctb.dft.site.v1':
        if policy and policy.get('mode') in {'mlff','hybrid'}:
            raise ValueError('external DFT site requires auto or strict dft policy; MLFF/hybrid use deployment configuration')
        if policy and policy.get('preferred_dft') and policy['preferred_dft']!=deployment['adapter']:
            raise ValueError('explicit policy/site adapter conflict')
        from .dft.exchange import evaluate_external, plan_external
        if dry_run:
            return {'plan': plan_external(task, deployment, protocol), 'run': {'status': 'planned', 'physics_calls': 0}}
        return evaluate_external(structures, task, output=output, site=deployment, protocol=protocol)
    items = load_submission(structures)
    resolved = plan(task, policy=policy, protocol=protocol, deployment=deployment, submission=items)
    protocol = resolved['protocol_lock']['protocol']
    sub_digest = submission_digest(items)
    run_digest = digest({'submission': sub_digest, 'comparison': resolved['comparability_digest']})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run.json').exists():
        previous = load_json(output / 'run.json')
        if previous.get('run_digest') != run_digest or previous.get('dry_run') != dry_run:
            raise ValueError('output belongs to a different evaluation; select a new output directory')
    write_json(output / 'plan.json', resolved)
    write_json(output / 'protocol.lock.json', resolved['protocol_lock'])
    run = {'schema_version': 'ctb.run.v2', 'run_digest': run_digest, 'submission_digest': sub_digest,
           'comparability_digest': resolved['comparability_digest'],
           'requested_backend': (policy or {}).get('mode', 'auto'),
           'resolved_backends': resolved['routes'], 'plan_status': resolved['status'],
           'promotion_reason': resolved.get('promotion_reason'), 'benchmark_eligible': False,
           'protocol_status': 'candidate', 'dry_run': dry_run, 'physics_calls': 0,
           'submitted_jobs': 0, 'native_calls': 0, 'cache_hits': 0, 'generation_cost': None,
           'physics_resource_usage': None, 'dft_verified': False}
    if dry_run:
        run['status'] = 'planned'
        write_json(output / 'run.json', run)
        return {'plan': resolved, 'run': run}
    if resolved['status']=='ready' and resolved['physics_fidelity']=='mlff':
        from .scientific import production_context, execute_submission
        from .contracts import default_deployment
        try:
            context=production_context(task,items,resolved,deployment or default_deployment(),output,cache_root)
        except Exception as exc:
            # Configuration/permit changes after planning remain a computational
            # block and retain the original route and full submission denominator.
            resolved['status']='blocked_execution'
            resolved['reasons'].append({'code':'runtime_preflight_failed','reason':str(exc)})
            run['plan_status']=resolved['status']
            write_json(output/'plan.json',resolved)
        else:
            results,metrics,costs=execute_submission(task,items,resolved,context)
            metrics.update(comparability_digest=resolved['comparability_digest'],submission_digest=sub_digest)
            run.update(status='completed',physics_calls=costs['physics_calls'],cache_hits=costs['cache_hits'],
                       physics_resource_usage=costs,benchmark_eligible=False)
            run['input_status_counts']={s:sum(i['input_status']==s for i in items) for s in ('valid','invalid','unsupported')}
            for filename,payload in [('submission.json',{'items':items,'submission_digest':sub_digest}),
                                     ('results.json',{'results':results}),('metrics.json',metrics),('run.json',run)]:
                write_json(output/filename,payload)
            return {'plan':resolved,'run':run,'metrics':metrics,'results':results,'submission':items}
    cache = StageCache(cache_root or (deployment or {}).get('paths', {}).get('cache_root') or output / 'cache')
    store = JobStore(output / 'state.json', run_digest)
    results = []
    for item in items:
        geom = item['input_geometry']
        reference_id = geom['geometry_id'] if geom and task['reference_geometry'] == 'input' else None
        measurements = {}
        for alias, spec in task['measurements'].items():
            route = resolved['routes'].get(alias, {})
            state_key = f"{item['item_id']}:{alias}"
            if (item['input_status'] == 'valid' and reference_id and route.get('family') == 'native'
                    and spec['property_id'] == 'space_group_number' and resolved['status'] == 'ready'):
                params = effective_symmetry_parameters(resolved['protocol_lock']['protocol']['parameters'])
                key = stage_key(geometry=reference_id, recipe='native.space_group.v1', parameters=params,
                    backend={'id':'symmetry', 'family':'native', 'version':version('spglib'),
                             'parser_version':version('ase'), 'assets':{}}, dependencies=[])
                observation = cache.get(key)
                try:
                    if observation is None:
                        store.record(state_key, 'running', computation_digest=key)
                        observation = analyze_symmetry(geom, params)
                        run['native_calls'] += 1
                        cache.put(key, observation)
                    else:
                        run['cache_hits'] += 1
                    measurements[alias] = symmetry_measurement(observation, geom, item['item_id'], route,
                                                               resolved['protocol_digest'])
                    store.record(state_key, 'completed', computation_digest=key,
                                 measurement_id=measurements[alias]['measurement_id'])
                except (ValueError, RuntimeError) as exc:
                    measurements[alias] = make_measurement(**spec, value=None,
                        reference_geometry_id=reference_id, item_links=[item['item_id']],
                        calculation_status='failed', reasons=[str(exc)])
                    store.record(state_key, 'failed', reason=str(exc))
            else:
                calculation_status = 'unsupported' if item['input_status'] == 'unsupported' else 'missing'
                reason = item['reasons'] or resolved.get('reasons') or ['requested physics execution is unavailable']
                measurements[alias] = make_measurement(property_id=spec['property_id'], unit=spec['unit'],
                    definition_id=route.get('definition_id', spec.get('definition_id', 'unresolved')),
                    reference_geometry_id=reference_id, item_links=[item['item_id']], value=None,
                    calculation_status=calculation_status, reasons=reason, backend_id=route.get('provider'),
                    backend_family=route.get('family'), physics_fidelity=route.get('family', 'unknown'),
                    protocol_digest=resolved['protocol_digest'])
                store.record(state_key, 'blocked', reason=reason)
        result = assess(task, measurements, reference_id, routes=resolved['routes'])
        result.update(item_id=item['item_id'], input_status=item['input_status'])
        results.append(result)
    metrics = aggregate(task, results)
    metrics.update(comparability_digest=resolved['comparability_digest'], submission_digest=sub_digest,
                   benchmark_eligible=False)
    run['status'] = 'completed' if resolved['status'] == 'ready' else 'blocked'
    run['input_status_counts'] = {s: sum(i['input_status'] == s for i in items)
                                  for s in ('valid', 'invalid', 'unsupported')}
    write_json(output / 'submission.json', {'items': items, 'submission_digest': sub_digest})
    write_json(output / 'results.json', {'results': results})
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'run.json', run)
    return {'plan': resolved, 'run': run, 'metrics': metrics, 'results': results, 'submission': items}


def summarize(path):
    path = Path(path)
    return load_json(path / 'metrics.json' if path.is_dir() else path)
