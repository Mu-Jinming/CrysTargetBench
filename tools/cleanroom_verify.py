#!/usr/bin/env python3
"""Install a built wheel offline and exercise it outside the source workspace.

The fresh interpreter probe denies Python-level file reads outside its explicit
allowlist and denies network sockets, subprocesses, and physics-package imports.
This is a reproducible audit-hook test, not a claim of kernel-level confinement.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


_PRIVATE_PATH = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+(?:/|\b)")
_PATH_LOADERS = {"spec_from_file_location", "SourceFileLoader", "SourcelessFileLoader"}


def static_scan(roots: list[Path], project_root: Path | None = None) -> dict:
    """Read only the specified public roots; never walk their parents."""
    findings = []
    scanned = 0
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if path.is_symlink():
                label = str(path.relative_to(project_root)) if project_root and path.is_relative_to(project_root) else path.name
                findings.append({"file": label, "line": None, "kind": "symlink_in_public_tree"})
                continue
            if not path.is_file() or path.suffix not in {".py", ".json", ".cif", ".toml", ".md"}:
                continue
            if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            scanned += 1
            label = str(path.relative_to(project_root)) if project_root and path.is_relative_to(project_root) else path.name
            source = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(source.splitlines(), 1):
                if _PRIVATE_PATH.search(line):
                    findings.append({"file": label, "line": line_number, "kind": "personal_absolute_path"})
            if path.suffix != ".py":
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError as exc:
                findings.append({"file": label, "line": exc.lineno, "kind": "python_syntax_error"})
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    function = node.func
                    name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", None)
                    if name in _PATH_LOADERS:
                        findings.append({"file": label, "line": node.lineno, "kind": "dynamic_path_import"})
                    if isinstance(function, ast.Attribute) and function.attr in {"insert", "append", "extend"}:
                        owner = function.value
                        if isinstance(owner, ast.Attribute) and owner.attr == "path" and isinstance(owner.value, ast.Name) and owner.value.id == "sys":
                            findings.append({"file": label, "line": node.lineno, "kind": "sys_path_injection"})
                if isinstance(node, (ast.Assign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Attribute) and target.attr == "path" and isinstance(target.value, ast.Name) and target.value.id == "sys":
                            findings.append({"file": label, "line": node.lineno, "kind": "sys_path_assignment"})
    return {"passed": not findings, "files_scanned": scanned, "findings": findings}


# This script is copied to the clean directory. It has no imports from tools or
# the source checkout. Python -I also ignores PYTHONPATH and user site packages.
_PROBE = r'''
from pathlib import Path
from importlib import metadata, resources
import contextlib
import importlib.abc
import io
import json
import os
import socket
import sys

clean = Path(sys.argv[1]).resolve()
workspace = Path(sys.argv[2]).resolve()
result_path = clean / "probe-result.json"
allowed_roots = [clean, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
allowed_files = {Path("/dev/null"), Path("/dev/urandom")}
events = {"denied_reads": 0, "denied_network": 0, "denied_subprocess": 0, "blocked_physics_imports": 0}

def allowed_path(value):
    if isinstance(value, int):
        return True
    if not isinstance(value, (str, bytes, os.PathLike)):
        return True
    path = Path(os.fsdecode(value)).resolve()
    return path in allowed_files or any(path.is_relative_to(root) for root in allowed_roots)

def audit(event, args):
    if event in {"open", "os.listdir", "os.scandir", "os.chdir"}:
        if args and args[0] is not None and not allowed_path(args[0]):
            events["denied_reads"] += 1
            raise PermissionError("cleanroom denied access outside explicit runtime/input roots")
    if event.startswith("socket."):
        events["denied_network"] += 1
        raise PermissionError("cleanroom denied network")
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"}:
        events["denied_subprocess"] += 1
        raise PermissionError("cleanroom denied subprocess/physical execution")

class PhysicsImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"torch", "mattersim", "phonopy", "atomate2", "jobflow"}:
            events["blocked_physics_imports"] += 1
            raise AssertionError("unexpected physics-package import: " + fullname)
        return None

sys.addaudithook(audit)
sys.meta_path.insert(0, PhysicsImportGuard())
checks = []

def check(name, condition, evidence=None):
    checks.append({"name": name, "passed": bool(condition), "evidence": evidence})
    if not condition:
        raise AssertionError(name)

def cli_call(argv, allowed_codes=(None, 0)):
    from crystargetbench.cli import main
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            code = main(argv)
        except SystemExit as exc:
            code = exc.code
    check("cli " + " ".join(argv[:1]), code in allowed_codes, output.getvalue()[:1200])
    return output.getvalue()

result = {"schema_version": "ctb.portability_probe.v1", "checks": checks,
          "audit_scope": "Python audit events; native C-level filesystem access is not an OS sandbox",
          "external_private_data_access": False, "physics_calls": 0, "submitted_jobs": 0}
try:
    import crystargetbench
    package_path = Path(crystargetbench.__file__).resolve()
    check("installed wheel import", package_path.is_relative_to(Path(sys.prefix).resolve()), package_path.name)
    check("cwd outside source workspace", not Path.cwd().is_relative_to(workspace))
    check("fresh HOME", Path(os.environ["HOME"]).is_relative_to(clean))
    check("no source sys.path", all(not Path(p or ".").resolve().is_relative_to(workspace / "src") for p in sys.path))
    check("cleared PYTHONPATH", "PYTHONPATH" not in os.environ)
    cli_version = cli_call(['--version']).strip()
    check("installed distribution/package/CLI versions agree",
          metadata.version('crystargetbench') == crystargetbench.__version__ and
          cli_version == 'ctb ' + crystargetbench.__version__,
          {'distribution': metadata.version('crystargetbench'),
           'package': crystargetbench.__version__, 'cli': cli_version})
    # These controls are tested before work; later extra denials indicate an
    # attempted unexpected external dependency and cause verification to fail.
    for label, denied_path in (("workspace reads denied", workspace / "pyproject.toml"),
                               ("outside private/history reads denied", Path("/__ctb_forbidden_history__/results.json"))):
        try:
            with open(denied_path, "rb"):
                pass
        except PermissionError:
            check(label, True)
        else:
            check(label, False)
    try:
        socket.getaddrinfo("example.invalid", 443)
    except PermissionError:
        check("network denied", True)
    else:
        check("network denied", False)
    control_denials = dict(events)
    cli_call(["--help"])
    doctor_text = cli_call(["doctor"])
    examples = resources.files("crystargetbench").joinpath("resources", "examples")
    inputs = clean / "inputs"
    inputs.mkdir()
    fcc = inputs / "first.cif"
    bcc = inputs / "second.cif"
    duplicate = inputs / "third.cif"
    broken = inputs / "broken.cif"
    fcc.write_text(examples.joinpath("structures", "ideal_fcc.cif").read_text())
    bcc.write_text(examples.joinpath("structures", "ideal_bcc.cif").read_text())
    duplicate.write_bytes(fcc.read_bytes())
    broken.write_text("This is not a crystallographic structure.\n")
    task = json.loads(examples.joinpath("tasks", "spacegroup_225.json").read_text())
    task_path = inputs / "task.json"
    task_path.write_text(json.dumps(task))
    cli_call(["validate-task", str(task_path)])
    from crystargetbench.contracts import default_protocol
    from crystargetbench.planner import plan
    from crystargetbench.api import evaluate
    highk = json.loads(examples.joinpath("tasks", "highk_screen.json").read_text())
    highk_path = inputs / "highk.json"
    highk_path.write_text(json.dumps(highk))
    cli_call(["plan", "--structures", str(fcc), "--task", str(highk_path),
              "--output", str(clean / "cli-highk")], allowed_codes=(3,))
    planned = plan(highk)
    check("High-K auto plans without physical execution", planned["status"] == "blocked_configuration",
          {"status": planned["status"], "routes": {key: value["family"] for key, value in planned["routes"].items()}})
    check("High-K physical routes are DFT", all(value["family"] == "dft" for value in planned["routes"].values()))
    cli_call(["evaluate", "--structures", str(fcc), str(bcc), "--task", str(task_path),
              "--output", str(clean / "cli-native"), "--cache", str(clean / "cli-cache")])
    cli_metrics = json.loads((clean / "cli-native" / "metrics.json").read_text())
    check("installed CLI native metrics", (cli_metrics["n_submitted"], cli_metrics["n_pass"], cli_metrics["n_fail"]) == (2, 1, 1))
    summary_text = cli_call(["summarize", str(clean / "cli-native")])
    check("CLI summarize uses the measured artifact", json.loads(summary_text) == cli_metrics)
    observed = evaluate([fcc, bcc, duplicate, broken], task, output=clean / "native-first", cache_root=clean / "cache")
    groups = [entry["measurements"]["sg"]["value"] for entry in observed["results"]]
    check("real input-to-spglib space groups", groups == [225, 229, 225, None], groups)
    metrics = observed["metrics"]
    check("full submitted denominator with duplicate and invalid input",
          (metrics["n_submitted"], metrics["n_pass"], metrics["n_fail"], metrics["n_unknown"]) == (4, 2, 1, 1))
    check("native actual calls and duplicate cache reuse", observed["run"]["native_calls"] == 2 and observed["run"]["cache_hits"] == 1,
          {key: observed["run"][key] for key in ("native_calls", "cache_hits", "physics_calls", "submitted_jobs")})
    task["constraints"][0]["value"] = 229
    repeated = evaluate([fcc, bcc, duplicate, broken], task, output=clean / "native-rejudge", cache_root=clean / "cache")
    check("new target reuses actual measurements", repeated["run"]["native_calls"] == 0 and repeated["run"]["cache_hits"] == 3)
    check("new target changes assessment", repeated["metrics"]["n_pass"] == 1 and repeated["metrics"]["n_fail"] == 2)
    check("native run has no physics jobs", observed["run"]["physics_calls"] == 0 and observed["run"]["submitted_jobs"] == 0)
    check("no physics package imported", not any(name.split(".", 1)[0] in {"torch", "mattersim", "phonopy", "atomate2", "jobflow"} for name in sys.modules))
    check("no unexpected forbidden accesses", events == control_denials, events)
    result.update(passed=True, native_space_groups=groups, native_metrics=metrics,
                  rejudgment_metrics=repeated["metrics"], highk_plan_status=planned["status"],
                  import_origin="fresh_venv/site-packages/crystargetbench",
                  installed_versions={dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
                  physics_validated=False, native_live_tested=True,
                  audit_counters=events, temporary_environment_removed_after_test=True)
except BaseException as exc:
    result.update(passed=False, error=type(exc).__name__ + ": " + str(exc), audit_counters=events)
result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
raise SystemExit(0 if result.get("passed") else 1)
'''


def _uv_executable() -> Path | None:
    explicit = os.environ.get("CTB_UV_EXECUTABLE")
    candidates = [Path(explicit)] if explicit else []
    candidates.extend((Path("/tmp/ctb-bootstrap-uv"), Path("/tmp/ctb-bootstrap-uv/uv")))
    return next((path for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)


def verify(wheel: Path, wheelhouse: Path, output: Path) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    wheel, wheelhouse, output = wheel.resolve(), wheelhouse.resolve(), output.resolve()
    report = {
        "schema_version": "ctb.portability_report.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "fresh_venv": True,
        "fresh_home": True,
        "fresh_cwd": True,
        "editable_install": False,
        "install_network_allowed": False,
        "runtime_network_allowed": False,
        "uses_private_history": False,
        "static_scan": static_scan([project_root / "src", project_root / "examples", project_root / "tests"], project_root),
    }
    with tempfile.TemporaryDirectory(prefix="ctb-cleanroom-") as temporary:
        clean = Path(temporary)
        environment = clean / "venv"
        home = clean / "home"
        cwd = clean / "work"
        home.mkdir()
        cwd.mkdir()
        env = {
            "HOME": str(home), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
            "PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull, "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1", "UV_CACHE_DIR": str(clean / "uv-cache"),
            "UV_PYTHON_DOWNLOADS": "never", "UV_OFFLINE": "true", "UV_NO_PROGRESS": "1",
            "MPLCONFIGDIR": str(home / ".matplotlib"), "XDG_CACHE_HOME": str(home / ".cache"),
        }
        commands = []

        def run(command: list[str], timeout: int = 180):
            completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
            rendered = " ".join(command).replace(str(project_root), "<workspace>").replace(str(clean), "<cleanroom>")
            commands.append({"command": rendered, "returncode": completed.returncode,
                             "stdout": completed.stdout[-6000:].replace(str(clean), "<cleanroom>").replace(str(project_root), "<workspace>"),
                             "stderr": completed.stderr[-6000:].replace(str(clean), "<cleanroom>").replace(str(project_root), "<workspace>")})
            if completed.returncode:
                raise RuntimeError(f"cleanroom command failed with return code {completed.returncode}: {Path(command[0]).name}")

        try:
            uv = _uv_executable()
            run([sys.executable, "-I", "-m", "venv", "--without-pip" if uv else "--clear", str(environment)])
            python = environment / "bin" / "python"
            install = ([str(uv), "pip", "install", "--python", str(python), "--offline"] if uv else [str(python), "-I", "-m", "pip", "install"])
            run(install + ["--no-index", "--find-links", str(wheelhouse), str(wheel)])
            probe = clean / "probe.py"
            probe.write_text(_PROBE, encoding="utf-8")
            run([str(python), "-I", str(probe), str(clean), str(project_root)], timeout=180)
            report["runtime_probe"] = json.loads((clean / "probe-result.json").read_text())
            report["passed"] = report["static_scan"]["passed"] and report["runtime_probe"]["passed"]
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            report["passed"] = False
            report["error"] = str(exc)
            if (clean / "probe-result.json").exists():
                report["runtime_probe"] = json.loads((clean / "probe-result.json").read_text())
        report["commands"] = commands
    report["temporary_environment_removed"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = verify(args.wheel, args.wheelhouse, args.output)
    print(json.dumps({"passed": report["passed"], "report": str(args.output)}, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
