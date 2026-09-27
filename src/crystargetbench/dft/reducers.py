"""Pure reductions of validated external data, not an alternate evaluator."""
import numpy as np
from ..physics import dielectric_response, phonon_statistics

def qualified(result):
    return {o['quantity']:o for o in result.get('observations',[]) if
            o['calculation_status']=='completed' and o['quality_status']=='qualified' and o['value'] is not None}

def band_gap(observations):
    if not {'eigenvalues','occupations'}<=observations.keys():return None,['missing_eigenvalues_or_occupations']
    e,o=observations['eigenvalues'],observations['occupations']
    a,b=np.asarray(e['value'],float),np.asarray(o['value'],float)
    info=e['diagnostics']
    required={'sampling','kpoints','weights','spin_channels','maximum_occupation','occupation_tolerance','occupation_convention'}
    if not required<=info.keys() or info['sampling']!='uniform_mesh' or info['occupation_convention']!='per_state_unweighted':
        return None,['uniform_mesh_and_normalized_occupation_evidence_required']
    if info!=o['diagnostics']:return None,['band_occupation_sampling_mismatch']
    if a.shape!=b.shape or len(info['kpoints'])!=len(a) or np.asarray(info['kpoints']).shape!=(len(a),3):return None,['band_mapping_shape_mismatch']
    weights=np.asarray(info['weights'],float)
    if weights.shape!=(len(a),) or (weights<=0).any() or not np.isfinite(weights).all() or abs(weights.sum()-1)>1e-6:return None,['invalid_k_weights']
    maximum,tol=info['maximum_occupation'],info['occupation_tolerance']
    if info['spin_channels']!=1 or maximum!=2 or type(tol) not in (int,float) or not 0<tol<0.1:return None,['unsupported_occupation_definition']
    if (b<-tol).any() or (b>maximum+tol).any():return None,['occupation_out_of_range']
    occupied=b>=maximum-tol; empty=b<=tol
    if not empty.any() or not occupied.any():return None,['insufficient_occupied_or_empty_bands']
    if ((b>tol)&(b<maximum-tol)).any():return 0.0,[]
    return max(0.0,float(a[empty].min()-a[occupied].max())),[]

def reduce_property(prop, observations, reference, family):
    if prop=='band_gap_eV': return band_gap(observations)
    if prop.startswith('phonon_'):
        observation=observations.get('phonon_spectrum')
        if not observation:return None,['missing_phonon_spectrum']
        data=observation['value'];scope='mesh' if 'mesh' in prop else 'path'
        if data['scope']!=scope:return None,['phonon_scope_mismatch']
        if data['reference_geometry_id']!=reference['geometry_id']:return None,['phonon_reference_mismatch']
        if not data['upstream_fidelity'] or (family=='dft' and any(f not in {'dft','native'} for f in data['upstream_fidelity'])):
            return None,['mixed_or_unknown_phonon_upstream']
        result=phonon_statistics(data['frequencies_THz'],scope=scope,weights=data['weights'],qpoints=data['qpoints'])
        return (result['min_frequency_THz'],[]) if result['quality_status']=='accepted' else (None,result['reasons'])
    if prop.startswith('dielectric_'):
        needed=['electronic_dielectric'] if prop=='dielectric_electronic_trace_over3' else ['electronic_dielectric','ionic_dielectric']
        total=observations.get('total_dielectric') if 'total' in prop else None
        parts=[total] if total else [observations.get(k) for k in needed]
        if any(x is None for x in parts):return None,['missing_dielectric_components']
        for o in parts:
            d=o['diagnostics']
            expected={'frequency_eV':0.0,'boundary':'fixed_strain','basis':'original_cartesian',
                      'local_field_effects':True,'reference_geometry_id':reference['geometry_id']}
            if any(d.get(k)!=v for k,v in expected.items()):return None,['unresolved_static_response_definition']
            if d.get('response_converged') is not True:return None,['response_not_converged']
            if not d.get('upstream_fidelity') or (family=='dft' and any(f not in {'dft','native'} for f in d['upstream_fidelity'])):
                return None,['mixed_upstream_not_pure_dft']
        if total and total['diagnostics'].get('components')!=['electronic','ionic']:return None,['total_response_component_provenance_missing']
        if prop=='dielectric_electronic_trace_over3':
            result=dielectric_response(total=parts[0]['value'],upstream_fidelity=[family])
        else:
            result=dielectric_response(total=total['value'] if total else None,
                electronic=observations.get('electronic_dielectric',{}).get('value'),
                ionic=observations.get('ionic_dielectric',{}).get('value'),upstream_fidelity=[family])
        if result['quality_status']!='accepted':return None,result['reasons']
        if 'ionic_bec' in prop:
            return float(np.trace(np.asarray(observations['ionic_dielectric']['value']))/3),[]
        return result['value'],[]
    return None,['external_recipe_unavailable']
