"""Static gates run in the regular suite; wheel verification is a real script."""
import ast
from pathlib import Path

import pytest

from tools.cleanroom_verify import _PROBE, static_scan


pytestmark = pytest.mark.cleanroom


def test_public_runtime_examples_and_tests_have_no_private_path_dependencies():
    project_root = Path(__file__).resolve().parents[2]
    result = static_scan(
        [project_root / "src", project_root / "examples", project_root / "tests"],
        project_root,
    )
    assert result["files_scanned"] > 0
    assert result["passed"], result["findings"]


def test_scanner_detects_private_absolute_paths(tmp_path):
    file = tmp_path / "example.json"
    private_path = "/" + "home/" + "researcher/private-data"
    file.write_text('{"path": "' + private_path + '"}')
    result = static_scan([tmp_path])
    assert not result["passed"]
    assert result["findings"][0]["kind"] == "personal_absolute_path"


@pytest.mark.parametrize("source,kind", [
    ("import sys\nsys.path.insert(0, 'external')\n", "sys_path_injection"),
    ("import sys\nsys.path += ['external']\n", "sys_path_assignment"),
    ("from importlib.util import spec_from_file_location\nspec_from_file_location('foreign', 'foreign.py')\n", "dynamic_path_import"),
])
def test_scanner_rejects_external_import_shortcuts(tmp_path, source, kind):
    (tmp_path / "bad.py").write_text(source)
    result = static_scan([tmp_path])
    assert not result["passed"]
    assert any(finding["kind"] == kind for finding in result["findings"])


def test_scanner_accepts_standard_package_imports_and_relative_assets(tmp_path):
    (tmp_path / "safe.py").write_text("from importlib.resources import files\nresource = files('crystargetbench')\n")
    (tmp_path / "site.json").write_text('{"assets_root": "./assets"}')
    result = static_scan([tmp_path])
    assert result["passed"]
    assert result["files_scanned"] == 2


def test_probe_is_standalone_python_and_declares_actual_native_checks():
    tree = ast.parse(_PROBE)
    assert tree.body
    assert "real input-to-spglib space groups" in _PROBE
    assert "225, 229, 225, None" in _PROBE
    assert "sys.addaudithook(audit)" in _PROBE
    assert "native C-level filesystem access is not an OS sandbox" in _PROBE
