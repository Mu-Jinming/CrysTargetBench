"""R3 actual analytic ASE/BM3; dependency labels below are manufactured."""
from copy import deepcopy
import pytest
from crystargetbench.recipes import relaxation
from crystargetbench.recipes.eos import eos
from crystargetbench.geometry import snapshot_from_atoms
from tests.s2.test_relax_eos import public_context,atoms_fixture,relax_parameters,eos_parameters


@pytest.mark.parametrize('change',['numpy','scipy','ase','implementation','fixed_default','tolerance'])
def test_R3_outer_cache_tracks_full_child_signature_actual_ase(tmp_path,monkeypatch,change):
    reference=snapshot_from_atoms(atoms_fixture(distorted=False));reference['stationary_reference_accepted']=True
    ctx=public_context(tmp_path);ctx.budget.start_sample(reference['geometry_id'])
    rp=relax_parameters(fmax_eV_per_A=0.001,stress_max_GPa=0.02)
    ep=eos_parameters(fit_rms_max_eV=1e-5)
    first=eos(reference,ctx,ep,rp);assert first['quality_status']=='accepted',first['reasons']
    before=ctx.stats['backend_structure_evaluations']
    versions=relaxation.version
    if change in {'numpy','scipy','ase'}:
        monkeypatch.setattr(relaxation,'version',lambda name:'synthetic-dependency-fingerprint' if name==change else versions(name))
    elif change=='implementation':monkeypatch.setattr(relaxation,'IMPLEMENTATION_VERSION','synthetic-implementation-fingerprint')
    elif change=='fixed_default':monkeypatch.setitem(relaxation._FIRE_FIXED,'Nmin',6)
    else:rp['fmax_eV_per_A']=0.0009
    second=eos(reference,ctx,ep,rp);assert second['quality_status']=='accepted',second['reasons']
    assert ctx.stats['stage_computations']['eos-point']==14
    assert ctx.stats['stage_computations']['eos-point-relax']==14
    assert ctx.stats['stage_computations']['eos-fit']==2
    assert {p['relaxation_signature_digest'] for p in first['points']}!={p['relaxation_signature_digest'] for p in second['points']}
    if change in {'numpy','scipy','ase','implementation'}:
        assert ctx.stats['backend_structure_evaluations']==before  # exact geometries/model reused
    assert ctx.stats['physics_calls']==0


def test_R3_fit_quality_changes_reuse_points_and_forces(tmp_path):
    reference=snapshot_from_atoms(atoms_fixture(distorted=False));reference['stationary_reference_accepted']=True
    ctx=public_context(tmp_path);ctx.budget.start_sample(reference['geometry_id'])
    rp=relax_parameters(fmax_eV_per_A=0.001,stress_max_GPa=0.02)
    ep=eos_parameters(fit_rms_max_eV=1e-5)
    first=eos(reference,ctx,ep,rp);assert first['quality_status']=='accepted'
    calls=ctx.stats['backend_structure_evaluations']
    changed={**ep,'fit_rms_max_eV':1e-20}
    second=eos(reference,ctx,changed,rp)
    assert second['quality_status']=='rejected' and second['value'] is None
    assert ctx.stats['stage_cache_hits']['eos-point']==7
    assert ctx.stats['stage_computations']['eos-point-relax']==7
    assert ctx.stats['backend_structure_evaluations']==calls
