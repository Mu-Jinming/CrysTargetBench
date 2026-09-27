"""Engine-free staged exchange using the existing task, assessment and metrics.

Bundles are portable data. Collection never consults or changes a managed
physics permit, and never claims to have enforced external resource usage.
"""
from copy import deepcopy
from contextlib import contextmanager
import fcntl
from pathlib import Path

from .. import __version__
from ..assessment import assess, make_measurement
from ..contracts import load_json, validate_task, validate_protocol, default_protocol
from ..identity import digest
from ..metrics import aggregate
from ..native import analyze_symmetry, symmetry_measurement, effective_symmetry_parameters
from ..reporting import write_json
from ..scheduling import measurement_schedule
from ..structures import load_submission, submission_digest
from .registry import discover, load_adapter
from .validation import (exact, read_record, validate_method, make_request, validate_result,
                         validate_prepared, capabilities, safe_path, artifact, verify_artifact)
from .reducers import qualified, reduce_property

def site_config(site,provider=None):
    site=load_json(site) if isinstance(site,(str,Path)) else deepcopy(site)
    exact(site,'schema_version adapter method configuration execution_mode managed','DFT site')
    if site['schema_version']!='ctb.dft.site.v1' or site['execution_mode'] not in {'external','managed_local'}:
        raise ValueError('invalid DFT site schema/mode')
    if provider is not None and provider!=site['adapter']:raise ValueError('explicit adapter/site conflict')
    manifests=discover()
    if site['adapter'] not in manifests:raise ValueError('explicit installed adapter selection required')
    return site,manifests[site['adapter']]

def portable_site(site):
    # File exchange never carries an enabled private permit. The caller retains
    # it separately for the optional launcher; collection only needs science.
    result=deepcopy(site);result['managed']=None
    return result

def property_quantities(prop):
    if prop=='space_group_number':return set()
    if prop=='band_gap_eV':return {'eigenvalues','occupations'}
    if prop.startswith('phonon_'):return {'phonon_spectrum'}
    if prop=='dielectric_electronic_trace_over3':return {'electronic_dielectric'}
    if prop in {'dielectric_ionic_bec_trace_over3','dielectric_total_static_trace_over3'}:
        return {'electronic_dielectric','ionic_dielectric'}
    return {'unimplemented:'+prop}

def plan_external(task,site,protocol=None,items=None):
    task=validate_task(task);site,manifest=site_config(site)
    explicit_protocol=protocol is not None
    protocol=validate_protocol(protocol) if explicit_protocol else default_protocol(task)
    if protocol['reference_geometry']!=task['reference_geometry']:raise ValueError('task/protocol reference mismatch')
    p=site['method'].get('effective_parameters',{})
    if not explicit_protocol and p:
        parameters=protocol['parameters']
        if 'dft' in parameters:
            parameters['dft']={'parameter_set_id':site['method']['method_id'],
                'pseudopotential_manifest_sha256':digest(site['method']['asset_fingerprints']),
                'cutoff_and_convergence_profile':digest(p)}
        if 'band_gap' in parameters:
            parameters['band_gap'].update(functional=p.get('xc'),mesh_policy_id=digest({k:p.get(k) for k in ('k_grid','k_shift')}),
                spin_u_soc_policy_id=digest({k:p.get(k) for k in ('spin','soc','hubbard_u')}))
    caps=capabilities(manifest);needed=set()
    for spec in task['measurements'].values():
        quantities=property_quantities(spec['property_id'])
        if spec['property_id']=='dielectric_total_static_trace_over3' and not quantities<=caps and 'total_dielectric' in caps:quantities={'total_dielectric'}
        needed|=quantities
        if spec['property_id'].startswith('phonon_') and task['reference_geometry']=='input':needed|={'forces','stress'}
    if task['reference_geometry']=='relaxed':needed|={'relaxed_geometry','forces','stress','energy'}
    missing=needed-caps
    reasons=[]
    try:validate_method(site['method'])
    except ValueError as exc:reasons.append(str(exc))
    engine=site['method'].get('software',[{}])[0].get('version')
    for cap in manifest['capabilities']:
        if cap['quantity'] in needed and engine not in cap['engine_versions']:reasons.append('engine version outside adapter capability: '+cap['quantity'])
    if explicit_protocol:
        parameters=protocol['parameters']
        if 'band_gap' in parameters and parameters['band_gap'].get('functional')!=p.get('xc'):reasons.append('protocol/method functional mismatch')
        if 'dft' in parameters and parameters['dft'].get('parameter_set_id')!=site['method']['method_id']:reasons.append('protocol/method parameter set mismatch')
    status='blocked_configuration' if reasons else ('blocked_capability' if missing else 'ready')
    if missing:reasons.append('missing capabilities: '+','.join(sorted(missing)))
    lock={'schema_version':'ctb.external_protocol.v2','ctb_version':__version__,'protocol':protocol,
          'scheduling':measurement_schedule(task),'adapter':manifest,'method':site['method'],
          'recipe_version':'ctb.external_reducers.v2','geometry_schema':'existing_ctb_snapshot',
          'science_validated':False,'benchmark_eligible':False}
    if any(spec['property_id']=='space_group_number' for spec in task['measurements'].values()):
        from importlib.metadata import version
        effective_symmetry_parameters(protocol['parameters'])
        lock['native_dependencies']={name:version(name) for name in ('spglib','ase')}
    protocol_digest=digest(lock)
    routes={a:{'provider':'symmetry' if spec['property_id']=='space_group_number' else manifest['provider_id'],
               'family':'native' if spec['property_id']=='space_group_number' else manifest['family'],
               'definition_id':f"{spec['property_id']}@external:{protocol_digest[:16]}",
               'physics_fidelity':'native' if spec['property_id']=='space_group_number' else manifest['family'],
               'requires_stationary_reference':spec['property_id'].startswith('phonon_')}
            for a,spec in task['measurements'].items()}
    return {'schema_version':'ctb.external_plan.v1','status':status,'reasons':reasons,
            'routes':routes,'protocol_lock':lock,'protocol_digest':protocol_digest,
            'comparability_digest':digest({'task':task,'protocol':protocol_digest}),
            'physics_fidelity':manifest['family'],'physics_calls':0,'submitted_jobs':0,
            'execution_kind':site['execution_mode'],'benchmark_eligible':False,
            'missing_capabilities':sorted(missing),'next_actions':['complete site method/capabilities'] if status!='ready' else ['prepare current geometry frontier']}

@contextmanager
def bundle_lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.exchange.lock').open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield root
        finally:fcntl.flock(stream,fcntl.LOCK_UN)

def _load_state(root):
    state=read_record(root/'exchange.json')
    if state['schema_version']!='ctb.external_run.v1':raise ValueError('unknown exchange state')
    if digest(state['site'])!=state['site_digest']:raise ValueError('site record changed')
    if submission_digest(state['items'])!=state['submission_digest']:raise ValueError('submission identity changed')
    if state['plan']['protocol_lock'].get('ctb_version')!=__version__:raise ValueError('core version changed; reimport into a new run')
    current=discover().get(state['site']['adapter'])
    if current!=state['manifest']:raise ValueError('installed adapter version/manifest changed; prepare a new run')
    expected=plan_external(state['task'],state['site'],state['plan']['protocol_lock']['protocol'])
    if expected!=state['plan']:raise ValueError('task/protocol/route lock changed; prepare a new run or reassess')
    for rid, result in state['accepted'].items():
        entry=state['requests'][rid]
        if digest(result)!=entry['result_digest']:raise ValueError('accepted result content changed')
        validate_result(result,entry['request'],state['manifest'],root/'received'/rid)
    return state

def _request(state,root,adapter,operation,quantities,geom,deps=(),sample_id=None,scope=None):
    parameters=state['plan']['protocol_lock']['protocol']['parameters']
    recipe_parameters=deepcopy(parameters.get('phonon',{})) if operation=='phonon_spectrum' else {}
    if operation=='phonon_spectrum' and scope:
        recipe_parameters['q_scope']=scope
        if scope=='path':
            recipe_parameters.pop('mesh',None);recipe_parameters.pop('gamma_center',None)
        else:recipe_parameters.pop('path_points_parameter',None)
    request=make_request(state['manifest'],operation,quantities,geom,state['site']['method'],deps,state['site']['execution_mode'],
        recipe={'id':'ctb.external.'+operation+'.v2','parameters':recipe_parameters},sample_geometry_id=sample_id)
    rid=request['request_id']
    if rid not in state['requests']:
        directory=root/'requests'/rid;directory.mkdir(parents=True,exist_ok=True)
        write_json(directory/'request.json',request)
        prepared=validate_prepared(adapter.prepare(request,directory,deepcopy(state['site']['configuration'])),request,directory)
        write_json(directory/'prepared.json',prepared)
        state['requests'][rid]={'request':request,'prepared':prepared,'status':'prepared','result_digest':None}
    if rid in state['accepted']:
        return state['accepted'][rid],rid
    return None,rid

def _stationary(result,method,quality=None,relaxed=True):
    import numpy as np
    from ..physics import EV_PER_ANGSTROM3_TO_GPA
    q=qualified(result);p=deepcopy(method['effective_parameters'])
    if quality:
        p['force_tolerance_eV_A']=min(p['force_tolerance_eV_A'],quality['fmax_eV_per_A'])
        p['stress_tolerance_GPa']=min(p['stress_tolerance_GPa'],quality['stress_max_GPa'])
    required={'forces','stress'}|({'relaxed_geometry'} if relaxed else set())
    if not required<=q.keys():return None,'qualified_reference_EFS_missing'
    if (np.linalg.norm(q['forces']['value'],axis=1).max()>p['force_tolerance_eV_A'] or
        np.max(np.abs(q['stress']['value']))*EV_PER_ANGSTROM3_TO_GPA>p['stress_tolerance_GPa']):
        return None,'relaxation_posterior_quality_rejected'
    return result['observed_geometry'],None

def _frontier(state,root,adapter):
    task=state['task'];plan=state['plan'];values=[];pending=set();states={}
    order=measurement_schedule(task)['measurement_order']
    for item in state['items']:
        reference=item['input_geometry'] if item['input_status']=='valid' else None
        sample_id=reference['geometry_id'] if reference else None
        reasons=list(item['reasons']);dependencies=[];available={};evidence=[];gate_block=None
        if reference and task['reference_geometry']=='relaxed':
            result,rid=_request(state,root,adapter,'relaxation',{'relaxed_geometry','energy','forces','stress'},reference,sample_id=sample_id)
            if result is None:
                pending.add(rid);reference=None;reasons=['awaiting_final_relaxed_geometry']
            else:
                reference,error=_stationary(result,state['site']['method'],plan['protocol_lock']['protocol']['parameters'].get('relaxation'))
                if error:reasons=[error]
                dependencies=[digest(result)]
                evidence.extend(o['evidence_kind'] for o in result['observations'])
        measurements={}
        for alias in order:
            spec=task['measurements'][alias];prop=spec['property_id'];route=plan['routes'][alias]
            value=None;why=list(reasons);diag={};calc='missing';quality='unverified'
            if reference and prop=='space_group_number':
                parameters=effective_symmetry_parameters(plan['protocol_lock']['protocol']['parameters'])
                native=analyze_symmetry(reference,parameters)
                measurements[alias]=symmetry_measurement(native,reference,item['item_id'],route,plan['protocol_digest'])
                continue
            if reference and (gate_block is None or prop in available):
                if prop not in available:
                    quantities=property_quantities(prop)
                    if prop.startswith('phonon_') and task['reference_geometry']=='input':quantities|={'forces','stress'}
                    if prop=='dielectric_total_static_trace_over3' and not quantities<=capabilities(state['manifest']):quantities={'total_dielectric'}
                    op='eigenvalues' if prop=='band_gap_eV' else ('phonon_spectrum' if prop.startswith('phonon_') else 'electric_response')
                    result,rid=_request(state,root,adapter,op,quantities,reference,dependencies,sample_id=sample_id,
                        scope=('mesh' if 'mesh' in prop else 'path') if prop.startswith('phonon_') else None)
                    if result is None:
                        pending.add(rid);why=['awaiting_external_result'];gate_block='upstream_pending'
                    else:
                        q=qualified(result);available[prop]=q
                        evidence.extend(o['evidence_kind'] for o in result['observations'])
                        value,why=reduce_property(prop,q,reference,state['manifest']['family'])
                        if prop.startswith('phonon_') and task['reference_geometry']=='input':
                            stationary,error=_stationary(result,state['site']['method'],plan['protocol_lock']['protocol']['parameters'].get('relaxation'),relaxed=False)
                            if error:value=None;why=[error]
                        calc='completed' if value is not None else ('failed' if result['errors'] else 'missing')
                        quality='accepted' if value is not None else 'unverified'
                        diag={'external_result_digest':digest(result),'external_evidence_kinds':sorted(set(evidence)),
                              'integrity_validated':True,'independently_authenticated':False,
                              'stationary_reference_accepted':True if task['reference_geometry']=='relaxed' or (prop.startswith('phonon_') and value is not None) else None}
                        dependencies.append(digest(result))
                        if value is None and (prop=='band_gap_eV' or prop.startswith('phonon_')):gate_block='prerequisite_unknown'
                else:
                    value,why=reduce_property(prop,available[prop],reference,state['manifest']['family'])
                    calc='completed' if value is not None else 'missing'
                    quality='accepted' if value is not None else 'unverified'
                    diag=next((deepcopy(m['diagnostics']) for m in measurements.values() if m['property_id']==prop),{})
            elif reference and gate_block:why=['downstream_not_requested:'+gate_block]
            family=state['manifest']['family']
            # Synthetic wire evidence remains synthetic even when a real adapter
            # parser is tested with authored fixture fragments.
            fidelity='synthetic' if 'synthetic' in evidence else family
            measurements[alias]=make_measurement(property_id=prop,unit=spec['unit'],definition_id=route['definition_id'],
                value=value,reference_geometry_id=reference['geometry_id'] if reference else None,item_links=[item['item_id']],
                calculation_status=calc,quality_status=quality,identity_status='verified' if value is not None else 'unverified',
                backend_id=state['manifest']['provider_id'],backend_family=fidelity,backend_version=state['manifest']['adapter_version'],
                physics_fidelity=fidelity,upstream_fidelity=[fidelity],evidence_kind='synthetic' if fidelity=='synthetic' else 'imported',
                protocol_digest=plan['protocol_digest'],diagnostics=diag,reasons=why,benchmark_eligible=False)
            # Same deterministic priority and AND semantics as the benchmark.
            partial=assess(task,measurements,reference['geometry_id'] if reference else None,routes=plan['routes'])
            if any(c['decision']=='fail' and c['measurement']==alias for c in partial['constraints']):gate_block='qualified_prerequisite_failed'
            if prop=='band_gap_eV' and value is not None and value<=0:gate_block='metal_response_not_qualified'
            if (prop=='band_gap_eV' or prop.startswith('phonon_')) and not partial['measurement_eligibility'][alias]['eligible']:
                gate_block='prerequisite_unknown'
        result=assess(task,measurements,reference['geometry_id'] if reference else None,routes=plan['routes'])
        result.update(item_id=item['item_id'],input_status=item['input_status'],reference_geometry=reference,
                      benchmark_eligible=False,evidence_scope='external_collection')
        values.append(result)
        states[item['item_id']]={'reference_geometry_id':reference['geometry_id'] if reference else None,'gate':gate_block}
    return values,sorted(pending),states

def _write_out(state,root,output,values,pending,states):
    out=Path(output or root/'evaluation');out.mkdir(parents=True,exist_ok=True)
    metrics=aggregate(state['task'],values)
    metrics.update(protocol_digest=state['plan']['protocol_digest'],comparability_digest=state['plan']['comparability_digest'],
                   submission_digest=state['submission_digest'],benchmark_eligible=False)
    run={'schema_version':'ctb.external_evaluation.v1','status':'awaiting_results' if pending else 'completed',
         'execution_mode':'external','ctb_engine_invocations':0,'external_engine_invocations':None,
         'external_wall_seconds':None,'budget_enforced_by_ctb':False,'managed_ledger_modified':False,
         'n_submitted':len(state['items']),'pending_request_ids':pending,'item_states':states,
         'next_actions':['execute prepared requests externally, then collect'] if pending else [],
         'science_validated':False,'benchmark_eligible':False}
    for name,data in [('results.json',{'results':values}),('metrics.json',metrics),('run.json',run),
                      ('protocol.lock.json',state['plan']['protocol_lock']),('submission.json',{'items':state['items']})]:
        write_json(root/'evaluation'/name,data)
        if out.resolve()!=(root/'evaluation').resolve():write_json(out/name,data)
    state['pending']=pending;write_json(root/'exchange.json',state)
    return {'plan':state['plan'],'run':run,'metrics':metrics,'results':values}

def prepare(structures=None,task=None,*,output,site=None,adapter=None,protocol=None):
    with bundle_lock(output) as root:
        if (root/'exchange.json').exists():
            state=_load_state(root)
            if site is not None and portable_site(site_config(site,adapter)[0])!=state['site']:raise ValueError('site changed: prepare a new run')
            if structures is not None and submission_digest(load_submission(structures))!=state['submission_digest']:raise ValueError('submission changed: prepare a new run')
            if task is not None and validate_task(load_json(task) if isinstance(task,(str,Path)) else task)!=state['task']:
                raise ValueError('task changed: use reassess or a new run')
        else:
            task=validate_task(load_json(task) if isinstance(task,(str,Path)) else task)
            site,manifest=site_config(site,adapter)
            site=portable_site(site)
            items=load_submission(structures)
            protocol=load_json(protocol) if isinstance(protocol,(str,Path)) else protocol
            resolved=plan_external(task,site,protocol,items)
            write_json(root/'plan.json',resolved)
            if resolved['status']!='ready':
                results=[]
                for item in items:
                    ref=item.get('input_geometry') if task['reference_geometry']=='input' else None
                    value=assess(task,{},ref['geometry_id'] if ref else None,routes=resolved['routes'])
                    value.update(item_id=item['item_id'],input_status=item['input_status'],reference_geometry=ref)
                    results.append(value)
                metrics=aggregate(task,results)
                run={'status':'blocked','ctb_engine_invocations':0,'external_engine_invocations':None,
                    'external_wall_seconds':None,'budget_enforced_by_ctb':False,'next_actions':resolved['next_actions']}
                for name,value in [('metrics.json',metrics),('results.json',{'results':results}),('run.json',run),('submission.json',{'items':items})]:
                    write_json(root/'evaluation'/name,value)
                return {'plan':resolved,'run':run,'metrics':metrics,'results':results}
            state={'schema_version':'ctb.external_run.v1','task':task,'site':site,'site_digest':digest(site),
                   'manifest':manifest,'items':items,'submission_digest':submission_digest(items),
                   'plan':resolved,'requests':{},'accepted':{},'pending':[]}
        selected,_=load_adapter(state['site']['adapter'])
        values,pending,states=_frontier(state,root,selected)
        return _write_out(state,root,None,values,pending,states)

def collect(run,results=None,*,output=None):
    with bundle_lock(run) as root:
        state=_load_state(root);selected,manifest=load_adapter(state['site']['adapter'])
        returned=Path(results) if results else root/'requests'
        # Unordered batch directories are joined solely by request identity.
        if returned.is_symlink():raise ValueError('returned bundle symlink')
        for rid,entry in sorted(state['requests'].items()):
            directory=safe_path(returned,rid)
            if not directory.exists():continue
            standard=directory/'result.json'
            if standard.exists():raw=read_record(standard)
            else:
                # Adapter knows its output filenames; missing outputs are not a
                # result and do not poison a later partial-return batch.
                try:raw=selected.collect(entry['request'],directory,deepcopy(state['site']['configuration']))
                except FileNotFoundError:continue
            checked=validate_result(raw,entry['request'],manifest,directory)
            checksum=digest(checked)
            if rid in state['accepted']:
                if digest(state['accepted'][rid])!=checksum:raise ValueError('conflicting content for existing request ID')
                continue
            # Keep portable raw artifacts with their checked identity. No
            # reference to the submitter's absolute output directory survives.
            import shutil
            for o in checked['observations']:
                for ref in o['raw_artifacts']:
                    source=safe_path(directory,ref['relative_path'])
                    target=safe_path(root/'received'/rid,ref['relative_path'])
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists() and target.read_bytes()!=source.read_bytes():raise ValueError('conflicting raw artifact')
                    if source.resolve()!=target.resolve():shutil.copyfile(source,target)
            write_json(root/'received'/rid/'result.json',checked)
            state['accepted'][rid]=checked
            entry.update(status='collected',result_digest=checksum)
        values,pending,states=_frontier(state,root,selected)
        return _write_out(state,root,output,values,pending,states)

def reassess(run,task,*,output):
    root=Path(run);state=_load_state(root)
    task=validate_task(load_json(task) if isinstance(task,(str,Path)) else task)
    original=state['task']
    for key in ('measurements','reference_geometry'):
        if task[key]!=original[key]:raise ValueError('reassessment cannot alter scientific measurement definitions')
    saved=read_record(root/'evaluation/results.json')['results'];results=[]
    for item in saved:
        value=assess(task,item['measurements'],item['reference_geometry_id'],routes=state['plan']['routes'])
        value.update(item_id=item['item_id'],input_status=item['input_status']);results.append(value)
    metrics=aggregate(task,results);metrics.update(protocol_digest=state['plan']['protocol_digest'],
        comparability_digest=digest({'task':task,'protocol':state['plan']['protocol_digest']}))
    write_json(Path(output)/'results.json',{'results':results});write_json(Path(output)/'metrics.json',metrics)
    return {'metrics':metrics,'ctb_engine_invocations':0,'external_usage':None}


def evaluate_external(structures,task,*,output,site,protocol=None):
    """Explicit external site: prepare, or opt-in serial local process/collect.

    The normal external mode never imports the launcher. The managed mode is
    entered only with an enabled, bound site record; templates remain disabled.
    """
    config,_=site_config(site)
    result=prepare(structures,task,output=output,site=config,protocol=protocol)
    if config['execution_mode']=='external' or result['plan']['status']!='ready':return result
    from .local import run_local,budget_location
    if not isinstance(config['managed'],dict) or config['managed'].get('enabled') is not True:
        raise ValueError('enabled complete managed local site required')
    root=Path(output);attempts={}
    while result['run'].get('pending_request_ids'):
        before=result['run']['pending_request_ids']
        for rid in before:
            record=run_local(root/'requests'/rid,config)
            attempts[rid]=record
        result=collect(root)
        if result['run']['pending_request_ids']==before:
            result['run']['next_actions']=['local process finished without a collectable result; inspect raw output/adapter receipt']
            break
    import json
    ledger,_=budget_location(config['managed'])
    ledgers=[read_record(ledger)] if ledger.exists() else []
    result['run'].update(execution_mode='managed_local',budget_enforced_by_ctb=bool(ledgers),
        ctb_engine_invocations=sum(x['backend_forwards'] for x in ledgers),managed_ledger_modified=bool(ledgers),
        managed_attempts=sum(x['reserved_evaluations'] for x in ledgers))
    write_json(root/'evaluation/run.json',result['run'])
    return result
