"""Small public-use regressions; the full recorded journeys run separately."""
import json
from pathlib import Path
import subprocess
import sys
from tools import preview_demo


def test_readme_resource_extraction_without_checkout(tmp_path):
    readme=Path('README.md').read_text()
    snippet=readme.split("python - <<'PY'\n",1)[1].split('\nPY\n',1)[0]
    script=Path('examples/public_preview/extract_examples.py').read_text().split('\n',1)[1]
    assert snippet.strip()==script.strip()
    # -I removes cwd/PYTHONPATH imports. Only installed package resources work.
    result=subprocess.run([sys.executable,'-I','-c',snippet],cwd=tmp_path,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert (tmp_path/'ctb-examples/structures/ideal_fcc.cif').is_file()
    task=json.loads((tmp_path/'ctb-examples/tasks/spacegroup_225.json').read_text())
    assert task['constraints'][0]['value']==225


def test_current_cli_help_describes_external_exchange(capsys):
    from crystargetbench.cli import main
    import pytest
    with pytest.raises(SystemExit) as exc:main(['--help'])
    assert exc.value.code==0
    text=capsys.readouterr().out
    assert 'external DFT exchange' in text and 'DFT execution remains unavailable' not in text


def test_standalone_demo_uses_existing_plugin_and_wire_v2(tmp_path):
    from crystargetbench.dft.exchange import prepare,collect
    demo=tmp_path/'demo';preview_demo.init(demo)
    structures=[demo/'ideal_fcc.cif',demo/'ideal_bcc.cif',demo/'ideal_fcc.cif',demo/'invalid.cif']
    result=prepare(structures,demo/'task.json',site=demo/'site.json',output=demo/'run')
    assert result['metrics']['n_unknown']==4
    for _ in range(3):
        state=json.loads((demo/'run/exchange.json').read_text())
        assert all(x['request']['schema_version']=='ctb.dft.request.v2' for x in state['requests'].values())
        if not state['pending']:break
        preview_demo.returns(demo/'run',demo/'returned')
        result=collect(demo/'run',demo/'returned')
    assert result['metrics']['n_submitted']==4 and result['metrics']['n_pass']==3 and result['metrics']['n_unknown']==1
    assert result['run']['ctb_engine_invocations']==0


def test_sdk_primary_description_matches_current_wire():
    sdk=Path('docs/DFT_ADAPTER_API.md').read_text()
    assert 'request-result wire v2' in sdk.splitlines()[0]
    assert 'path: 3N of recorded Phonopy primitive' in sdk
    assert 'CTB 0.3.0' not in Path('examples/dft_adapters/abacus/README.md').read_text()
