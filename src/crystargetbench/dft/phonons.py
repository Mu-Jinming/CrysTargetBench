"""File-exchange forces feeding the existing pinned Phonopy implementation.

Preparation enumerates actual finite displacements, including the stationary
supercell. Collection fits/samples CPU arrays only, never launches a calculator.
"""
from copy import deepcopy
from pathlib import Path
from ..identity import digest
from ..geometry import snapshot_from_atoms
from ..cache import StageCache
from ..recipes import phonon as recipe
from .validation import make_request, validate_result, safe_path
from .reducers import qualified


def displacement_requests(reference,manifest,method,parameters):
    params=recipe.resolved_parameters(parameters);versions=recipe.versions()
    phonon=recipe._to_phonopy(reference,params)
    if len(phonon.supercell)>4096:raise ValueError('external displacement atom bound exceeded')
    phonon.generate_displacements(distance=params['displacement_A'],is_plusminus=params['is_plusminus'],is_diagonal=params['is_diagonal'])
    signature=digest({'reference':reference,'ifc_parameters':{k:params[k] for k in recipe.IFC_FIELDS},'versions':versions,'recipe':recipe.IFC_RECIPE})
    requests=[]
    for index,cell in enumerate([phonon.supercell,*phonon.supercells_with_displacements]):
        geom=snapshot_from_atoms(recipe._atoms(cell),parent=reference['geometry_id'],stage='external_phonon_force')
        req=make_request(manifest,'static',['forces'],geom,method,[signature])
        requests.append(req)
    return {'schema_version':'ctb.external_displacements.v1','reference':deepcopy(reference),
        'manifest':manifest,'method':method,'parameters':params,'versions':versions,
        'signature':signature,'requests':requests,'coordinate_unit':'angstrom','force_unit':'eV/angstrom',
        'force_constants_unit':'eV/angstrom^2','engine_invocations':0}


class _ReadOnlyBounds:
    """Bounds for already returned data. This does not claim external budgeting."""
    def check_atoms(self,count):
        if count>4096:raise ValueError('external atom bound exceeded')
    def reserve_displacement(self,identifier):pass

class _ForceContext:
    def __init__(self,plan,returned,cache):
        self.budget=_ReadOnlyBounds();self.cache=cache;self.plan=plan;self.data={};self.hits=0
        if Path(returned).is_symlink():raise ValueError('returned bundle symlink')
        for request in plan['requests']:
            directory=safe_path(returned,request['request_id'])
            from .validation import read_record
            result=validate_result(read_record(directory/'result.json'),request,plan['manifest'],directory)
            obs=qualified(result).get('forces')
            if obs is None:raise ValueError('incomplete/unqualified external displacement forces')
            self.data[request['geometry']['geometry_id']]=(obs,result)
        self.evidence_digest=digest(self.data)
        self.family=plan['manifest']['family']
        if any(o['evidence_kind']=='synthetic' for o,r in self.data.values()):self.family='synthetic'
    def evaluate(self,atoms,identifier,*,mapping):
        geom=snapshot_from_atoms(atoms);obs,result=self.data[geom['geometry_id']]
        return {'node_id':identifier,'geometry_id':geom['geometry_id'],'mapping_digest':digest(mapping),
            'forces_eV_A':obs['value'],'force_dtype':'float64','physics_fidelity':self.family,
            'evidence_kind':'synthetic' if self.family=='synthetic' else 'imported'}
    def cached(self,stage,geometry,parameters,dependencies,compute):
        key=digest({'schema':'ctb.external_force_cache.v1','stage':stage,'geometry':geometry,'parameters':parameters,
            'dependencies':dependencies,'adapter':self.plan['manifest'],'method':self.plan['method'],
            'external_results':self.evidence_digest})
        value=self.cache.get(key)
        if value is None:value=compute();self.cache.put(key,value)
        else:self.hits+=1
        return value


def collect_phonons(plan,returned,cache_root,*,sampling=None,scope='mesh'):
    # Reconstruct the plan signature; q-sampling is intentionally outside IFC.
    check=displacement_requests(plan['reference'],plan['manifest'],plan['method'],plan['parameters'])
    if check!=plan:raise ValueError('external displacement plan/version changed')
    context=_ForceContext(plan,returned,StageCache(cache_root))
    ifc=recipe.build_ifc(plan['reference'],context,plan['parameters'])
    tolerance=plan['method']['effective_parameters']['force_tolerance_eV_A']
    if ifc['reference_force_max_eV_A']>tolerance:raise ValueError('external phonon reference not stationary')
    sampled=recipe.sample_phonons(ifc,{**plan['parameters'],**(sampling or {})},scope)
    return {'ifc':ifc,'sampling':sampled,'cache_hits':context.hits,'ctb_engine_invocations':0,'external_usage':None}


def nac_inputs(born,electronic,*,ifc_fidelity,reference_geometry_id):
    """Validate optional response interchange; numerical NAC stays unsupported.

    Never feed total/ionic permittivity to NAC. A mixed source is labelled mixed
    (hybrid), so it cannot be reused as a pure DFT stability measurement.
    """
    for o,q in [(born,'born_charges'),(electronic,'electronic_dielectric')]:
        if o['quantity']!=q or o['quality_status']!='qualified' or o['value'] is None:
            raise ValueError('NAC requires qualified Born and electronic-only tensors')
        if o['diagnostics'].get('reference_geometry_id')!=reference_geometry_id:raise ValueError('NAC reference mismatch')
    families=set(born['diagnostics'].get('upstream_fidelity',[])+electronic['diagnostics'].get('upstream_fidelity',[])+[ifc_fidelity])
    if not families or 'unknown' in families:raise ValueError('NAC upstream fidelity required')
    return {'born_charges':born['value'],'electronic_dielectric':electronic['value'],
        'reference_geometry_id':reference_geometry_id,'physics_fidelity':next(iter(families)) if len(families)==1 else 'mixed',
        'numerical_nac_implemented':False,'benchmark_eligible':False}
