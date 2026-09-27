"""Pinned MatterSim 1.1.2 ASE worker. No model imports until authorized request.

Use: python -m crystargetbench.workers.mattersim --session-config TRUSTED.json
The checkpoint loader is deliberately explicit: the upstream convenience loader
creates a home cache even for explicit files and can download model aliases.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
from importlib import metadata
import io
import os
from pathlib import Path
import sys
import time

from ..assets import (AssetError, MATTERSIM_VERSION, validate_permit, verify_asset,
                      verify_worker_environment, verify_environment_file)
from ..contracts import _loads, load_json
from ..identity import canonical_json, digest
from .protocol import normalize_efs, response_identity, validate_request


class MatterSimSession:
    def __init__(self, config):
        if config.get("test_mode"):
            raise AssetError("blocked_execution", "production MatterSim worker cannot use test authorization")
        self.config = config
        self.lock = verify_asset(config["model_lock"]["manifest"])
        if self.lock["model_lock_digest"] != config["model_lock"]["model_lock_digest"]:
            raise AssetError("blocked_asset", "model lock was altered")
        self.permit = validate_permit(config.get("permit"), self.lock, config.get("bindings"))
        verify_environment_file(config.get("environment_lock"), config.get("environment_lock_path"),
                                self.permit["worker_environment_lock_sha256"])
        self.verified_environment = verify_worker_environment(config.get("environment_lock"))
        self.calculator = None
        self.environment = None
        self.forwards = 0
        self.leases = set()
        self.started = time.monotonic()

    def _load(self):
        manifest = self.lock["manifest"]
        if metadata.version("mattersim") != MATTERSIM_VERSION or metadata.version("ase") != "3.26.0":
            raise AssetError("blocked_configuration", "requires pinned MatterSim 1.1.2 / ASE 3.26.0")
        # Second identity check immediately before reading loader bytes.
        verified = verify_asset(manifest)
        if verified["model_lock_digest"] != self.lock["model_lock_digest"]:
            raise AssetError("blocked_asset", "asset changed before load")
        import torch
        from ase.units import GPa
        from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
        from mattersim.forcefield.potential import Potential, MatterSimCalculator

        device = manifest["device"]
        if device.startswith("cuda:"):
            index = int(device.split(":")[1])
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise AssetError("blocked_configuration", "authorized CUDA device unavailable")
        # Hash the very bytes passed to weights_only loader: no path TOCTOU.
        data = Path(manifest["checkpoint_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["checkpoint_sha256"]:
            raise AssetError("blocked_asset", "checkpoint changed during load")
        try:
            checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise AssetError("blocked_asset", "checkpoint is not supported by the approved weights_only loader") from exc
        if not isinstance(checkpoint, dict) or checkpoint.get("model_name") != "m3gnet":
            raise AssetError("blocked_asset", "unsupported model checkpoint structure")
        if digest(checkpoint.get("model_args")) != manifest["model_config_digest"]:
            raise AssetError("blocked_asset", "checkpoint model configuration differs from approved manifest")
        if any(value.is_floating_point() and value.dtype != torch.float32 for value in checkpoint["model"].values()):
            raise AssetError("blocked_asset", "checkpoint floating weights are not approved float32")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = M3Gnet(device=device, **checkpoint["model_args"]).to(device=device, dtype=torch.float32)
        model.load_state_dict(checkpoint["model"], strict=True)
        potential = Potential(model=model, device=device, allow_tf32=False)
        potential.model.eval()
        self.calculator = MatterSimCalculator(potential=potential, args_dict={}, compute_stress=True,
                                             stress_weight=GPa, device=device)
        parameters = list(potential.model.parameters())
        actual_devices = sorted({str(p.device) for p in parameters})
        actual_dtypes = sorted({str(p.dtype) for p in parameters if p.is_floating_point()})
        if actual_devices != [device] or actual_dtypes != ["torch.float32"]:
            raise AssetError("blocked_configuration", "actual model precision/device differs from lock")
        self.environment = {"python": sys.version.split()[0], "packages": {
            name: metadata.version(name) for name in ("mattersim", "torch", "ase", "numpy")},
            "actual_devices": actual_devices, "model_parameter_dtypes": actual_dtypes,
            "allow_tf32": False, "mixed_precision": False,
            "model_loads_this_session": 1, "adapter": "ctb.mattersim.ase.v1",
            "verified_environment_lock": self.verified_environment,
            "checkpoint_sha256": manifest["checkpoint_sha256"]}
        self.environment["digest"] = digest(self.environment)

    def evaluate(self, request):
        validate_request(request)
        if request["model_lock_digest"] != self.lock["model_lock_digest"]:
            raise AssetError("blocked_asset", "request model identity mismatch")
        limits = self.permit["limits"]
        lease = request["budget_lease"]["lease_id"]
        if lease in self.leases:
            raise AssetError("budget_exhausted", "budget lease already consumed")
        if len(request["geometry"]["species"]) > limits["max_supercell_atoms"]:
            raise AssetError("budget_exhausted", "supercell atom budget exceeded")
        if self.forwards >= limits["max_backend_structure_evaluations"]:
            raise AssetError("budget_exhausted", "backend evaluation budget exhausted")
        if time.monotonic() - self.started >= limits["max_total_wall_seconds"]:
            raise AssetError("budget_exhausted", "session wall budget exhausted")
        self.leases.add(lease)
        if self.calculator is None:
            self._load()
        from ..geometry import atoms_from_snapshot
        atoms = atoms_from_snapshot(request["geometry"])
        import numpy as np
        before = (atoms.positions.copy(), atoms.cell.array.copy(), atoms.numbers.copy(),
                  atoms.pbc.copy(), atoms.get_masses())
        self.forwards += 1  # An attempted forward consumes its reserved cost on error too.
        start = time.monotonic()
        # No global inference_mode/no_grad: E derivatives require autograd.
        self.calculator.calculate(atoms, properties=["energy", "forces", "stress"])
        raw = self.calculator.results
        observation = normalize_efs(raw["energy"], raw["forces"], raw.get("stress"), len(atoms))
        after = (atoms.positions, atoms.cell.array, atoms.numbers, atoms.pbc, atoms.get_masses())
        if any(not np.array_equal(a, b) for a, b in zip(before, after)):
            raise ValueError("calculator modified calculation geometry")
        # Identity derives from the verified input fractional snapshot; converting
        # Cartesian round trips back to fractions can introduce insignificant bits.
        actual = request["geometry_id"]
        return {**response_identity(request), "mapping_digest": digest(request["mapping"]),
            "actual_geometry_id": actual, "status": "completed", "observation": observation,
            "provider": "mattersim", "evidence_kind": "mlff", "environment": self.environment,
            "cost": {"backend_forwards": 1, "evaluated_structures": 1, "wall_seconds": time.monotonic() - start},
            "artifacts": [], "benchmark_eligible": False, "science_validated": False}


def _offline_guard(workdir):
    root = Path(workdir).resolve()
    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.bind"):
            raise PermissionError("worker network disabled")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and not Path(args[0]).resolve().is_relative_to(root):
                raise PermissionError("worker writes confined to session workdir")
        if event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename"):
            paths = args[:2] if event == "os.rename" else args[:1]
            if any(isinstance(path, (str, bytes, os.PathLike)) and not Path(path).resolve().is_relative_to(root) for path in paths):
                raise PermissionError("worker filesystem mutation outside workdir")
        if event in ("subprocess.Popen", "os.system"):
            raise PermissionError("worker cannot launch subprocesses")
    sys.addaudithook(audit)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CTB offline MatterSim E/F/stress worker")
    parser.add_argument("--session-config", required=True)
    args = parser.parse_args(argv)
    from .lifecycle import start_parent_watchdog
    start_parent_watchdog()
    config = load_json(args.session_config)
    _offline_guard(config["workdir"])
    session = None
    while True:
        line = sys.stdin.readline(32 * 1024 * 1024 + 1)
        if not line:
            break
        request = None
        try:
            if len(line) > 32 * 1024 * 1024:
                raise ValueError("worker input message size limit exceeded")
            request = _loads(line)
            validate_request(request)
            # Any provider print is captured as stderr; stdout belongs to protocol.
            with redirect_stdout(sys.stderr):
                if session is None:
                    session = MatterSimSession(config)
                result = session.evaluate(request)
        except Exception as exc:
            category = getattr(exc, "category", "backend_error")
            if "out of memory" in str(exc).lower():
                category = "out_of_memory"
            result = {**(response_identity(request) if request and all(k in request for k in
                ("protocol_version", "request_id", "attempt_id", "node_id", "geometry_id", "model_lock_digest")) else {}),
                "mapping_digest": digest(request.get("mapping")) if request else None,
                "status": "error", "error": {"category": category, "message": str(exc)},
                "retry_provider": None}
        sys.stdout.write(canonical_json(result) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
