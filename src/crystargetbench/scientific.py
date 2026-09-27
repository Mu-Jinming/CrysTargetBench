"""S2 local MLFF recipe execution; no DFT substitution or implicit authorization."""
from copy import deepcopy
import hashlib
from pathlib import Path

from .assessment import assess, make_measurement
from .contracts import load_json
from .geometry import atoms_from_snapshot
from .identity import digest
from .metrics import aggregate
from .reporting import write_json
from .scheduling import measurement_schedule

SUPPORTED = {'space_group_number','phonon_path_min_frequency_THz',
             'phonon_mesh_min_frequency_THz','bulk_modulus_eos_GPa'}


def production_context(task, items, resolved, deployment, output, cache_root=None):
    """Repeat all authorization/identity checks immediately before starting a worker."""
    from .assets import verify_asset, validate_permit
    from .budget import Budget
    from .calculation import ExecutionContext
    from .structures import submission_digest
    from .workers.client import WorkerClient
    if resolved['status']!='ready' or resolved['physics_fidelity']!='mlff':
        raise ValueError('execution requires a ready, exclusively MLFF final route')
    if resolved['protocol_lock'].get('scheduling')!=measurement_schedule(task):
        raise ValueError('production requires the current explicit locked scheduling rule; replan legacy plans')
    if any(spec['property_id'] not in SUPPORTED for spec in task['measurements'].values()):
        raise ValueError('unsupported recipe cannot be substituted')
    if any(m['property_id'].startswith('phonon_') for m in task['measurements'].values()):
        from .recipes.phonon import versions
        versions()  # fail before model initialization if numerical extras are absent
    providers={route['provider'] for route in resolved['routes'].values() if route['family']=='mlff'}
    if providers!={'mattersim'}: raise ValueError('S2 supports only the owned MatterSim worker')
    config=deployment['providers']['mattersim']
    model=verify_asset(load_json(config['model_manifest']))
    permit_path=Path(config['live_permit']).resolve()
    permit=load_json(permit_path)
    environment_path=Path(config['environment_lock'])
    environment=load_json(environment_path)
    bindings={'worker_environment_lock_sha256':hashlib.sha256(environment_path.read_bytes()).hexdigest(),
              'deployment_digest':digest(deployment),'input_manifest_sha256':submission_digest(items),
              'protocol_digest':resolved['protocol_digest']}
    permit=validate_permit(permit,model,bindings)
    needed={'energy_forces_stress'}
    if task['reference_geometry']=='relaxed': needed.add('relaxation')
    if any(m['property_id'].startswith('phonon_') for m in task['measurements'].values()): needed.add('finite_displacement_phonon')
    if any(m['property_id']=='bulk_modulus_eos_GPa' for m in task['measurements'].values()): needed.add('bulk_eos')
    if not needed<=set(permit['allowed_operations']): raise ValueError('recipe operation is not owner-authorized')
    if permit['limits']['max_concurrent_workers']!=1: raise ValueError('S2 is serial; permit must specify one worker')
    argv=config['worker_argv']
    if len(argv)!=3 or argv[1:]!=['-m','crystargetbench.workers.mattersim'] or not Path(argv[0]).is_absolute():
        raise ValueError('production command must be the CTB-owned MatterSim module and explicit interpreter')
    worker_dir=Path(output)/'worker'
    cache_root=Path(cache_root or deployment['paths']['cache_root'])
    # All resumes of the same signed/owner-approved permit share durable tokens,
    # even when the caller chooses another output directory.
    budget=Budget(permit_path.parent/(digest(permit)+'.budget.json'),permit['limits'],
                  identity=digest(permit),scratch_roots=[Path(output),cache_root],
                  predecessor_ledger=config.get('budget_predecessor_ledger'))
    manifest=model['manifest']
    backend={'id':'mattersim','family':'mlff','version':manifest['package_version'],
        'assets':{'checkpoint':manifest['checkpoint_sha256']},'model_lock_digest':model['model_lock_digest'],
        'precision':manifest['precision'],'device':manifest['device'],
        'environment_lock_sha256':bindings['worker_environment_lock_sha256']}
    worker=WorkerClient(argv,worker_dir,model,config.get('worker_timeout_seconds',60),
                        permit=permit,bindings=bindings,environment_lock=environment,
                        environment_lock_path=environment_path,scratch_check=budget.check_scratch)
    return ExecutionContext(backend=backend,budget=budget,workdir=Path(output)/'execution',
                            cache_root=cache_root,worker=worker)


def execute_item(task,item,resolved,context):
    """Common recipe path. Synthetic contexts require explicitly synthetic routes."""
    from .recipes.relaxation import relax, check_stationarity
    from .recipes.eos import eos
    from .recipes.phonon import build_ifc, sample_phonons
    from .native import analyze_symmetry, symmetry_measurement, effective_symmetry_parameters
    params=resolved['protocol_lock']['protocol']['parameters']
    schedule=measurement_schedule(task)
    if resolved['protocol_lock'].get('scheduling',schedule)!=schedule:
        raise ValueError('execution scheduling differs from the locked plan')
    measurements={}; artifacts={}; reference=None; failure=[]
    for alias in schedule['measurement_order']:
        spec=task['measurements'][alias]
        route=resolved['routes'][alias]
        if spec['property_id'] not in SUPPORTED or route['family'] not in {context.physics_fidelity,'native'}:
            raise ValueError('context cannot satisfy the planned backend/recipe; no substitution permitted')
    if item['input_status']=='valid':
        try:
            context.budget.start_sample(item['input_geometry']['geometry_id'])
            if task['reference_geometry']=='relaxed':
                artifacts['relaxation']=relax(item['input_geometry'],context,params['relaxation'])
                reference=artifacts['relaxation']['reference_geometry']
                if reference is None: failure=artifacts['relaxation']['reasons'] or ['reference_relaxation_unqualified']
            else:
                reference=deepcopy(item['input_geometry'])
                if any(m['property_id']!='space_group_number' for m in task['measurements'].values()):
                    atoms=atoms_from_snapshot(reference)
                    observation=context.evaluate(atoms,'input-stationarity')
                    quality,reasons=check_stationarity(atoms,observation,params['relaxation'])
                    artifacts['input_stationarity']={'diagnostics':quality,'observation':observation,'reasons':reasons}
                    reference['stationary_reference_accepted']=not reasons
                    if reasons: failure=reasons
        except Exception as exc:
            failure=[f'{type(exc).__name__}: {exc}']
    else:
        failure=list(item['reasons'])
    reference_id=reference['geometry_id'] if reference else None
    ifc=None
    for alias in schedule['measurement_order']:
        spec=task['measurements'][alias]
        route=resolved['routes'][alias]; prop=spec['property_id']
        value=None; reasons=list(failure); calc='missing'; quality='unverified'; diagnostics={}; artifact=None
        if reference and not reasons:
            try:
                if prop=='space_group_number':
                    observation=analyze_symmetry(reference,effective_symmetry_parameters(params))
                    measurements[alias]=symmetry_measurement(observation,reference,item['item_id'],route,resolved['protocol_digest'])
                    measurements[alias]['upstream_fidelity']=[context.physics_fidelity] if task['reference_geometry']=='relaxed' else []
                    continue
                if reference.get('stationary_reference_accepted') is not True:
                    raise ValueError('qualified stationary reference required')
                if prop.startswith('phonon_'):
                    if ifc is None:
                        ifc=build_ifc(reference,context,params['phonon']); artifacts['ifc']=ifc
                    if ifc['reference_force_max_eV_A']>params['relaxation']['fmax_eV_per_A']:
                        raise ValueError('reference_supercell_not_stationary')
                    scope='mesh' if 'mesh' in prop else 'path'
                    from .recipes.phonon import SAMPLING_RECIPE, versions
                    artifact=context.cached(SAMPLING_RECIPE,reference['geometry_id'],
                        {'phonon':params['phonon'],'scope':scope,'versions':versions()},[ifc['artifact_digest']],
                        lambda:sample_phonons(ifc,params['phonon'],scope))
                elif prop=='bulk_modulus_eos_GPa':
                    artifact=eos(reference,context,params['eos'],params['relaxation'])
                artifacts[alias]=artifact
                value=artifact.get('value',artifact.get('min_frequency_THz'))
                calc=artifact.get('calculation_status','completed'); quality=artifact.get('quality_status','unverified')
                reasons=list(artifact.get('reasons',[]))
                diagnostics={'stationary_reference_accepted':True,'artifact_digest':artifact.get('artifact_digest',digest(artifact)),
                             'scope':artifact.get('scope'), 'fit':artifact.get('fit')}
            except Exception as exc:
                reasons=[f'{type(exc).__name__}: {exc}']; calc='failed'; quality='unverified'
        measurements[alias]=make_measurement(property_id=prop,unit=spec['unit'],definition_id=route['definition_id'],
            value=value,reference_geometry_id=reference_id,item_links=[item['item_id']],
            calculation_status='unsupported' if item['input_status']=='unsupported' else calc,
            quality_status=quality,identity_status='verified' if reference and calc=='completed' else 'unverified',
            backend_id=context.backend['id'],backend_family=context.physics_fidelity,
            backend_version=context.backend['version'],asset_digests=context.backend['assets'],
            physics_fidelity=context.physics_fidelity,upstream_fidelity=[context.physics_fidelity],
            evidence_kind=context.evidence_kind,protocol_digest=resolved['protocol_digest'],
            diagnostics=diagnostics,reasons=reasons,benchmark_eligible=False)
    result=assess(task,measurements,reference_id,routes=resolved['routes'])
    result.update(item_id=item['item_id'],input_status=item['input_status'],
                  input_geometry=item['input_geometry'],reference_geometry=reference,
                  evidence_kind=context.evidence_kind,benchmark_eligible=False,scheduling=schedule)
    write_json(context.workdir/(digest(item['item_id'])+'.recipes.json'),
               {'item_id':item['item_id'],'artifacts':artifacts,'evidence_kind':context.evidence_kind,'benchmark_eligible':False})
    return result


def execute_submission(task,items,resolved,context):
    results=[]
    try:
        for item in items: results.append(execute_item(task,item,resolved,context))
    finally:
        context.close()
    metrics=aggregate(task,results)
    metrics.update(benchmark_eligible=False,evidence_kind=context.evidence_kind)
    return results,metrics,deepcopy(context.stats)
