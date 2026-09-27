"""Bounded, strict external records, with one existing CTB geometry identity."""
from copy import deepcopy
import hashlib
from pathlib import Path, PurePosixPath
import re
import numpy as np

from ..contracts import _finite_data, _loads
from ..geometry import atoms_from_snapshot
from ..identity import digest
from .sampling import request_sampling, validate_electronic, validate_phonon

MAX_BYTES = 16 * 1024 * 1024
MAX_ATOMS = 4096
MAX_ARRAY_ELEMENTS = 1_000_000
QUANTITIES = {'energy','forces','stress','eigenvalues','occupations','electronic_dielectric',
              'ionic_dielectric','total_dielectric','born_charges','relaxed_geometry','phonon_spectrum'}
UNITS = {'energy':'eV','forces':'eV/angstrom','stress':'eV/angstrom^3','eigenvalues':'eV',
         'occupations':'1','electronic_dielectric':'1','ionic_dielectric':'1','total_dielectric':'1',
         'born_charges':'e','relaxed_geometry':'angstrom','phonon_spectrum':'THz'}
HASH = re.compile(r'[0-9a-f]{64}\Z')

def exact(record, fields, label):
    if not isinstance(record,dict) or set(record)!=set(fields.split()):
        raise ValueError(f'{label}: fields must be {fields}')
    _finite_data(record)
    return record

def bounded(value, depth=0):
    if depth>40: raise ValueError('record nesting too deep')
    if isinstance(value,(dict,list)):
        if len(value)>MAX_ARRAY_ELEMENTS: raise ValueError('oversized array/object')
        count=sum(bounded(x,depth+1) for x in (value.values() if isinstance(value,dict) else value))
        if count>MAX_ARRAY_ELEMENTS: raise ValueError('oversized record')
        return count
    return 1

def read_record(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>MAX_BYTES: raise ValueError('symlink or oversized record')
    try:value=_loads(path.read_text())
    except RecursionError as exc:raise ValueError('record nesting too deep') from exc
    bounded(value)
    return value

def safe_path(root, relative):
    root=Path(root).resolve()
    if not isinstance(relative,str) or not relative or '\\' in relative or '\x00' in relative:
        raise ValueError('invalid relative artifact path')
    parts=PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or '..' in parts or ':' in relative:
        raise ValueError('artifact traversal/absolute path rejected')
    path=root.joinpath(*parts)
    current=root
    for part in parts:
        current=current/part
        if current.is_symlink(): raise ValueError('artifact symlinks rejected')
    if not path.resolve().is_relative_to(root): raise ValueError('artifact escapes bundle')
    return path

def artifact(root, relative, role='raw_output'):
    path=safe_path(root,relative)
    if not path.is_file() or path.stat().st_size>MAX_BYTES: raise ValueError('missing/oversized artifact')
    return {'relative_path':relative,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes':path.stat().st_size,'role':role}

def verify_artifact(root, entry):
    exact(entry,'relative_path sha256 size_bytes role','artifact')
    if artifact(root,entry['relative_path'],entry['role'])!=entry:
        raise ValueError('artifact hash/size mismatch')

def geometry(value):
    if not isinstance(value,dict) or len(value.get('species',[]))>MAX_ATOMS:
        raise ValueError('invalid or oversized geometry')
    if not HASH.fullmatch(value.get('geometry_id','')): raise ValueError('geometry identity required')
    required={'cell','positions','coordinate_system','length_unit','species','occupancy','pbc','magnetic_state'}
    if not required<=value.keys():raise ValueError('complete CTB geometry snapshot required')
    if digest({k:value[k] for k in required})!=value['geometry_id']:raise ValueError('geometry content/hash mismatch')
    if value.get('mass_digest') and digest(value.get('masses'))!=value['mass_digest']:raise ValueError('mass digest mismatch')
    atoms_from_snapshot(value)
    return value

def validate_manifest(value):
    exact(value,'schema_version api_version provider_id adapter_id adapter_version distribution family capabilities science_validated benchmark_eligible','manifest')
    if value['schema_version']!='ctb.dft.adapter_manifest.v1' or value['api_version']!=1:
        raise ValueError('incompatible adapter API')
    if value['family'] not in {'dft','synthetic'}: raise ValueError('invalid adapter family')
    for field in ('provider_id','adapter_id','adapter_version','distribution'):
        if not isinstance(value[field],str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',value[field]):
            raise ValueError('invalid adapter identity')
    quantities=[]
    for cap in value['capabilities']:
        exact(cap,'quantity implementation parser_tested live_tested engine_versions restrictions','capability')
        if cap['quantity'] not in QUANTITIES or cap['implementation'] not in {'planned','implemented'}:
            raise ValueError('invalid quantity capability')
        if not isinstance(cap['engine_versions'],list) or not cap['engine_versions']:
            raise ValueError('pinned engine version restrictions required')
        if any(type(cap[k]) is not bool for k in ('parser_tested','live_tested')): raise ValueError('invalid status')
        quantities.append(cap['quantity'])
    if len(quantities)!=len(set(quantities)): raise ValueError('duplicate capability')
    if value['science_validated'] is not False or value['benchmark_eligible'] is not False:
        raise ValueError('API v1 manifests cannot confer scientific validation')
    return deepcopy(value)

def capabilities(manifest):
    return {x['quantity'] for x in manifest['capabilities'] if x['implementation']=='implemented'}

def validate_method(method):
    exact(method,'method_id effective_parameters software asset_fingerprints','method')
    if not isinstance(method['method_id'],str) or not method['method_id']: raise ValueError('method id required')
    p=method['effective_parameters']
    required={'xc','basis','k_grid','k_shift','cutoff_Ry','spin','soc','hubbard_u','occupations',
              'scf_tolerance','force_tolerance_eV_A','stress_tolerance_GPa'}
    if not isinstance(p,dict) or not required<=p.keys(): raise ValueError('incomplete effective scientific parameters')
    def unresolved(x):
        if x is None: return True
        if isinstance(x,str): return x.lower() in {'auto','default','latest','unknown','todo','placeholder'} or '${' in x or '{{' in x
        if isinstance(x,dict): return any(unresolved(v) for v in x.values())
        if isinstance(x,list): return any(unresolved(v) for v in x)
        return False
    if unresolved(method): raise ValueError('unresolved method cannot form final lock')
    for key in ('cutoff_Ry','scf_tolerance','force_tolerance_eV_A','stress_tolerance_GPa'):
        if type(p[key]) not in (int,float) or p[key]<=0: raise ValueError('positive scientific tolerance/cutoff required')
    if len(p['k_grid'])!=3 or any(type(x) is not int or x<1 for x in p['k_grid']): raise ValueError('invalid k grid')
    if len(p['k_shift'])!=3 or any(x not in (0,1) for x in p['k_shift']): raise ValueError('invalid k shift')
    if type(p['spin']) is not int or p['spin'] not in (1,2,4) or type(p['soc']) is not bool or not isinstance(p['hubbard_u'],dict):raise ValueError('invalid spin/SOC/U definition')
    if not isinstance(p['xc'],str) or not isinstance(p['basis'],str) or not isinstance(p['occupations'],str):raise ValueError('invalid scientific method strings')
    if not method['software']: raise ValueError('pinned engine required')
    for software in method['software']:
        exact(software,'name version','software')
        if not software['name'] or not software['version']: raise ValueError('unresolved software')
    seen=set()
    for asset in method['asset_fingerprints']:
        exact(asset,'kind identifier sha256','asset fingerprint')
        if not isinstance(asset['sha256'],str) or not HASH.fullmatch(asset['sha256']) or (asset['kind'],asset['identifier']) in seen: raise ValueError('invalid asset identity')
        seen.add((asset['kind'],asset['identifier']))
    return method

def make_request(manifest, operation, quantities, geom, method, dependencies=(), execution_mode='external', recipe=None, sample_geometry_id=None):
    validate_manifest(manifest); geometry(geom); validate_method(method)
    if not set(quantities)<=capabilities(manifest): raise ValueError('adapter lacks requested capability')
    record={'schema_version':'ctb.dft.request.v2','adapter_api_version':1,
            'adapter':{'id':manifest['adapter_id'],'version':manifest['adapter_version']},
            'sample_geometry_id':sample_geometry_id or geom['geometry_id'],
            'recipe':deepcopy(recipe or {'id':'ctb.external.quantity.v1','parameters':{}}),
            'operation':operation,'requested_quantities':sorted(quantities),'geometry':deepcopy(geom),
            'method':deepcopy(method),'method_digest':digest(method),'dependencies':sorted(dependencies),
            'execution_mode':execution_mode}
    record['sampling']=request_sampling(geom, method, quantities, record['recipe'])
    record['request_digest']=digest(record);record['request_id']=record['request_digest']
    return validate_request(record)

def validate_request(r):
    exact(r,'schema_version adapter_api_version request_id request_digest adapter sample_geometry_id recipe sampling operation requested_quantities geometry method method_digest dependencies execution_mode','request')
    bounded(r); geometry(r['geometry']);validate_method(r['method'])
    exact(r['adapter'],'id version','request adapter')
    exact(r['recipe'],'id parameters','request recipe')
    if not isinstance(r['sample_geometry_id'],str) or not HASH.fullmatch(r['sample_geometry_id']):raise ValueError('persistent original sample geometry identity required')
    if not isinstance(r['recipe']['id'],str) or not isinstance(r['recipe']['parameters'],dict):raise ValueError('invalid recipe lock')
    if r['schema_version']!='ctb.dft.request.v2' or r['adapter_api_version']!=1: raise ValueError('request wire v2 required; legacy requests need explicit new preparation/import')
    if r['operation'] not in {'static','relaxation','eigenvalues','electric_response','phonon_spectrum'}: raise ValueError('unsupported operation')
    if r['execution_mode'] not in {'external','managed_local'}: raise ValueError('invalid execution mode')
    if not r['requested_quantities'] or not set(r['requested_quantities'])<=QUANTITIES: raise ValueError('invalid requested quantities')
    if any(not isinstance(x,str) or not HASH.fullmatch(x) for x in r['dependencies']): raise ValueError('dependency digest required')
    if r['sampling']!=request_sampling(r['geometry'],r['method'],r['requested_quantities'],r['recipe']):
        raise ValueError('request sampling manifest differs from geometry/method/recipe')
    expected=digest({k:v for k,v in r.items() if k not in {'request_id','request_digest'}})
    if r['request_id']!=expected or r['request_digest']!=expected or r['method_digest']!=digest(r['method']):
        raise ValueError('request/method content hash mismatch')
    return deepcopy(r)

def validate_prepared(p,request,directory):
    exact(p,'schema_version request_id request_digest status files mapping reasons','prepared job')
    if p['schema_version']!='ctb.dft.prepared.v1' or p['status'] not in {'ready','blocked_assets','unsupported'}: raise ValueError('invalid preparation status')
    for key in ('request_id','request_digest'):
        if p[key]!=request[key]: raise ValueError('prepared request mismatch')
    validate_mapping(p['mapping'],len(request['geometry']['species']))
    for a in p['files']: verify_artifact(directory,a)
    return deepcopy(p)

def validate_mapping(mapping,n):
    exact(mapping,'output_to_input basis coordinate_convention','atom/tensor mapping')
    if sorted(mapping['output_to_input'])!=list(range(n)) or any(type(x) is not int for x in mapping['output_to_input']):
        raise ValueError('invalid atom permutation')
    if mapping['basis']!=[[1,0,0],[0,1,0],[0,0,1]] or mapping['coordinate_convention']!='cartesian_angstrom':
        raise ValueError('API v1 requires normalized original Cartesian tensor basis')

def identity_mapping(n):
    return {'output_to_input':list(range(n)),'basis':[[1,0,0],[0,1,0],[0,0,1]],'coordinate_convention':'cartesian_angstrom'}

def observation(quantity,value,*,definition_id,raw_artifacts,diagnostics=None,evidence_kind='reported',qualified=True):
    return {'quantity':quantity,'value':value,'unit':UNITS[quantity],'definition_id':definition_id,
            'calculation_status':'completed' if value is not None else 'missing',
            'quality_status':'qualified' if qualified and value is not None else 'unverified',
            'evidence_kind':evidence_kind,'diagnostics':diagnostics or {},'raw_artifacts':raw_artifacts}

def result_record(request,manifest,observations,*,observed_geometry=None,scf=False,ionic=None,response=None,evidence_mode='external',errors=()):
    return {'schema_version':'ctb.dft.result.v2','adapter_api_version':1,
            'request_id':request['request_id'],'request_digest':request['request_digest'],
            'reference_geometry_id':request['geometry']['geometry_id'],'observed_geometry':observed_geometry,
            'method_digest':request['method_digest'],'effective_method':deepcopy(request['method']),
            'adapter':request['adapter'],'engine':deepcopy(request['method']['software'][0]),
            'mapping':identity_mapping(len(request['geometry']['species'])),
            'convergence':{'engine_completed':scf,'scf':scf,'ionic':ionic,'response':response},
            'execution':{'mode':evidence_mode,'engine_invocations':None,'wall_seconds':None,'count_evidence':'unknown'},
            'observations':observations,'errors':list(errors)}

def validate_result(r,request,manifest,directory):
    validate_request(request); validate_manifest(manifest); bounded(r)
    if request['adapter']!={'id':manifest['adapter_id'],'version':manifest['adapter_version']} or not set(request['requested_quantities'])<=capabilities(manifest):
        raise ValueError('request not bound to selected manifest/capabilities')
    exact(r,'schema_version adapter_api_version request_id request_digest reference_geometry_id observed_geometry method_digest effective_method adapter engine mapping convergence execution observations errors','result')
    if r['schema_version']!='ctb.dft.result.v2' or r['adapter_api_version']!=1: raise ValueError('result API mismatch')
    for key in ('request_id','request_digest','method_digest','adapter'):
        if r[key]!=request[key]: raise ValueError(f'result {key} mismatch')
    if digest(r['effective_method'])!=request['method_digest'] or r['engine']!=request['method']['software'][0]:
        raise ValueError('effective method/engine differs from request')
    if r['reference_geometry_id']!=request['geometry']['geometry_id']: raise ValueError('result geometry mismatch')
    n=len(request['geometry']['species']);validate_mapping(r['mapping'],n)
    if r['mapping']!=identity_mapping(n): raise ValueError('adapter must normalize arrays to original atom IDs')
    if r['observed_geometry'] is not None:
        geometry(r['observed_geometry'])
        if r['observed_geometry']['species']!=request['geometry']['species']: raise ValueError('observed atom identities mismatch')
        if request['operation']!='relaxation' and r['observed_geometry']['geometry_id']!=request['geometry']['geometry_id']:
            raise ValueError('static/displacement geometry changed; relaxation is forbidden')
    exact(r['convergence'],'engine_completed scf ionic response','convergence')
    if any(x is not None and type(x) is not bool for x in r['convergence'].values()): raise ValueError('invalid convergence evidence')
    exact(r['execution'],'mode engine_invocations wall_seconds count_evidence','execution')
    e=r['execution']
    for k in ('engine_invocations','wall_seconds'):
        if e[k] is not None and (type(e[k]) not in (int,float) or e[k]<0):raise ValueError('invalid external cost')
    if e['engine_invocations'] is not None and type(e['engine_invocations']) is not int:raise ValueError('invocations must be integer')
    if e['mode'] not in {'external','managed_local','fixture'} or e['count_evidence'] not in {'unknown','external_report','managed_record','not_applicable'}:raise ValueError('invalid cost provenance')
    if e['count_evidence']=='unknown' and (e['engine_invocations'] is not None or e['wall_seconds'] is not None): raise ValueError('unknown external usage must remain null')
    if request['execution_mode']=='external' and e['count_evidence']=='managed_record':raise ValueError('external result cannot claim CTB-enforced budget')
    seen=set()
    for o in r['observations']:
        exact(o,'quantity value unit definition_id calculation_status quality_status evidence_kind diagnostics raw_artifacts','observation')
        q=o['quantity'];v=o['value']
        if q not in request['requested_quantities'] or q in seen: raise ValueError('unexpected/duplicate observation')
        seen.add(q)
        if o['unit']!=UNITS[q] or not isinstance(o['definition_id'],str) or not o['definition_id']: raise ValueError('unit/definition missing')
        if o['evidence_kind'] not in {'synthetic','reference','reported','live'}: raise ValueError('unknown evidence kind')
        if manifest['family']=='synthetic' and o['evidence_kind']!='synthetic': raise ValueError('synthetic plugin cannot relabel evidence')
        if o['calculation_status'] not in {'completed','partial','failed','missing','unsupported','skipped'} or o['quality_status'] not in {'qualified','unverified','rejected'}: raise ValueError('invalid quality status')
        if o['calculation_status'] in {'failed','missing','unsupported','skipped'} and v is not None: raise ValueError('missing quantities must be null')
        if o['quality_status']=='qualified':
            if v is None or o['calculation_status']!='completed' or not r['convergence']['engine_completed'] or not r['convergence']['scf']:
                raise ValueError('qualified value lacks completion/SCF evidence')
            if not o['raw_artifacts']: raise ValueError('qualified value requires raw artifacts')
            if r['observed_geometry'] is None:raise ValueError('qualified value requires observed geometry proof')
            if q in {'electronic_dielectric','ionic_dielectric','total_dielectric','born_charges'} and r['convergence']['response'] is not True:
                raise ValueError('qualified response lacks response convergence')
        for a in o['raw_artifacts']: verify_artifact(directory,a)
        if v is None: continue
        if q=='relaxed_geometry':
            geometry(v)
            if v!=r['observed_geometry'] or not r['convergence']['ionic']: raise ValueError('unconverged/unbound relaxed geometry')
        elif q=='phonon_spectrum':
            exact(v,'frequencies_THz qpoints weights scope reference_geometry_id upstream_fidelity','phonon spectrum')
            if v['reference_geometry_id']!=r['reference_geometry_id']:raise ValueError('phonon geometry mismatch')
            array=np.asarray(v['frequencies_THz'])
            if array.dtype.kind not in 'iuf':raise ValueError('phonon frequencies must be numeric')
            array=array.astype(float)
            if array.ndim!=2 or not np.isfinite(array).all():raise ValueError('invalid phonon frequency array')
            parameters=request['recipe']['parameters']
            if o['diagnostics'].get('phonon_parameters')!=parameters:raise ValueError('reported phonon method differs from requested recipe')
            if parameters.get('nac')!='none' and o['diagnostics'].get('nac_dielectric_component')!='electronic':raise ValueError('NAC must use electronic-only dielectric')
            validate_phonon(v,o['diagnostics'],request['sampling']['phonon'])
        else:
            a=np.asarray(v)
            if a.dtype.kind not in 'iuf':raise ValueError('numeric observation cannot contain booleans/strings')
            a=a.astype(float)
            if not np.isfinite(a).all() or a.size>MAX_ARRAY_ELEMENTS:raise ValueError('nonfinite/oversized numeric observation')
            shape={'energy':(),'forces':(n,3),'stress':(3,3),'electronic_dielectric':(3,3),'ionic_dielectric':(3,3),'total_dielectric':(3,3),'born_charges':(n,3,3)}.get(q)
            if shape is not None and a.shape!=shape:raise ValueError(f'{q} shape mismatch')
            if q=='stress' and not np.allclose(a,a.T,atol=1e-8,rtol=0):raise ValueError('stress must be symmetric')
            if q in {'eigenvalues','occupations'} and (a.ndim!=2 or min(a.shape)<1):raise ValueError('band arrays require k-by-band shape')
    if 'electronic' in request['sampling']:
        validate_electronic(r['observations'],request['sampling']['electronic'])
    return deepcopy(r)
