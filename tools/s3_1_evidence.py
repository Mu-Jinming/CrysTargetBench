"""Persist S3.1 actual CPU and synthetic wire evidence; never launch a solver."""
from pathlib import Path
from crystargetbench.reporting import write_json
from crystargetbench.dft.validation import validate_result
from crystargetbench.dft.reducers import qualified,reduce_property
from tests.s3_1.test_repairs import (band_case,phonon_case,
    test_actual_native_external_locked_tolerances_and_identity,
    test_path_resampling_reuses_existing_analytic_ifc_and_no_new_forces)


def main():
    root=Path('docs/s3_1/examples/synthetic/sampling-repairs')
    root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for scope in ('electronic','path','mesh'):
        directory=root/scope;directory.mkdir()
        if scope=='electronic':req,result,manifest=band_case(directory,shift=(1,0,1));prop='band_gap_eV'
        else:req,result,manifest=phonon_case(directory,scope=scope,points=101);prop=f'phonon_{scope}_min_frequency_THz'
        checked=validate_result(result,req,manifest,directory)
        value,reasons=reduce_property(prop,qualified(checked),req['geometry'],'synthetic')
        write_json(directory/'request.json',req);write_json(directory/'result.json',checked)
        rows.append({'scope':scope,'value':value,'reasons':reasons,'sampling':req['sampling'],
                     'evidence_kind':'synthetic','real_engine_invocations':0})
    native=root/'native-tolerances';native.mkdir()
    test_actual_native_external_locked_tolerances_and_identity(native)
    numerical=Path('docs/s3_1/analytic-path-resampling');numerical.mkdir(exist_ok=False)
    test_path_resampling_reuses_existing_analytic_ifc_and_no_new_forces(numerical)
    write_json(root/'NUMERICAL_EVIDENCE.json',{'cases':rows,'native':'actual ASE/spglib perturbed Cu geometry',
        'path_generator':'actual pinned Phonopy/SeeK-path CPU, no forces calculated',
        'returned_forces':'analytic_test pair spring, no material claim',
        'IFC_resampling_test':'mesh -> path(5 per segment) -> path(9 per segment); same IFC; unchanged returned force files',
        'MLFF_invocations':0,'DFT_invocations':0,'science_validated':False})

if __name__=='__main__':main()
