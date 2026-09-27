import json
from importlib.resources import files
from crystargetbench.cli import main

ROOT = files('crystargetbench').joinpath('resources','examples')


def test_help_is_safe():
    import pytest
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0


def test_validate_and_explicit_policy_conflict(tmp_path,capsys):
    task = str(ROOT.joinpath('tasks','spacegroup_225.json'))
    assert main(['validate-task',task]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'valid'
    code = main(['plan','--structures',str(ROOT.joinpath('structures')),'--task',task,
                 '--output',str(tmp_path/'out'),'--policy',str(ROOT.joinpath('policies','mlff_only.json')),
                 '--backend','dft'])
    assert code == 2
    assert 'conflicting' in capsys.readouterr().err


def test_cli_full_native_and_summarize(tmp_path,capsys):
    args = ['evaluate','--structures',str(ROOT.joinpath('structures')),
            '--task',str(ROOT.joinpath('tasks','spacegroup_225.json')),'--output',str(tmp_path/'out')]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['n_pass'] == 1 and result['n_fail'] == 1
    assert main(['summarize',str(tmp_path/'out')]) == 0
    assert json.loads(capsys.readouterr().out) == result


def test_highk_plan_exit_and_no_fake_metrics(tmp_path,capsys):
    assert main(['plan','--structures',str(ROOT.joinpath('structures')),
        '--task',str(ROOT.joinpath('tasks','highk_screen.json')),'--output',str(tmp_path/'out')]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'blocked_configuration'
    assert result['submitted_jobs'] == 0
    assert not (tmp_path/'out'/'metrics.json').exists()


def test_cli_mode_does_not_override_site_provider(tmp_path,capsys):
    from crystargetbench.contracts import default_deployment
    site = default_deployment()
    # The user selects and configures the legacy provider explicitly.
    from pathlib import Path
    legacy = json.loads((Path(__file__).parents[1]/'fixtures/legacy-deployment-v2.json').read_text())
    site['providers']['qe'] = legacy['providers']['qe']
    site['default_dft_provider'] = 'qe'
    path = tmp_path/'site.json'
    path.write_text(json.dumps(site))
    assert main(['plan','--structures',str(ROOT.joinpath('structures')),
        '--task',str(ROOT.joinpath('tasks','bandgap_gt2.json')),'--output',str(tmp_path/'out'),
        '--backend','dft','--deployment',str(path)]) == 3
    assert json.loads(capsys.readouterr().out)['routes']['gap']['provider'] == 'qe'
