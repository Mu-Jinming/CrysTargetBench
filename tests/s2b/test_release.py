"""Small release regressions; no model or physical execution."""
import pytest

from crystargetbench import __version__
from crystargetbench.cli import main


def test_cli_uses_package_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--version'])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f'ctb {__version__}'


def test_help_states_execution_boundary(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    help_text = ' '.join(capsys.readouterr().out.split())
    assert 'authorized MLFF' in help_text
    assert 'external DFT exchange' in help_text
    assert 'explicitly authorized local backends' in help_text
    assert 'DFT execution remains unavailable' not in help_text
    assert 'S1' not in help_text
