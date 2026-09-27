"""CTB-owned analytic software fixtures; never a production provider registry.

The pair potential is a manufactured differentiable function, not a material
model. Its stress includes exact affine pair and volume derivatives.
"""
import argparse
import os
import sys
import time

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class AnalyticPair(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    evidence_kind = "synthetic"
    provider = "analytic_test"

    def __init__(self, k=1.7, bulk=0.6, volume0=24.0, **kwargs):
        super().__init__(**kwargs)
        self.k, self.bulk, self.volume0 = k, bulk, volume0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if len(atoms) != 2:
            raise ValueError("manufactured pair fixture requires exactly two atoms")
        relative = atoms.positions[1] - atoms.positions[0]
        volume = atoms.get_volume()
        energy = .5 * self.k * np.dot(relative, relative) + .5 * self.bulk / self.volume0 * (volume - self.volume0) ** 2
        force = self.k * relative
        stress = self.k * np.outer(relative, relative) / volume + self.bulk * (volume - self.volume0) / self.volume0 * np.eye(3)
        self.results = {"energy": float(energy), "forces": np.array([force, -force]), "stress": stress}


def main():
    from crystargetbench.contracts import _loads, load_json
    from crystargetbench.geometry import atoms_from_snapshot
    from crystargetbench.identity import canonical_json, digest
    from crystargetbench.workers.protocol import normalize_efs, response_identity, validate_request
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-config", required=True)
    parser.add_argument("--mode", default="normal")
    args = parser.parse_args()
    config = load_json(args.session_config)
    if args.mode == "parent_death_driver":
        from pathlib import Path
        from crystargetbench.workers import WorkerClient
        worker = WorkerClient([sys.executable, "-m", "worker_fixtures", "--mode", "timeout"],
            Path(args.session_config).parent / "child", {"provider": "analytic_test", "evidence_kind": "synthetic"},
            timeout_seconds=30, test_mode=True)
        worker.evaluate(config["geometry"], "orphan-test", budget_lease={"lease_id": "synthetic-parentdeath"})
        return
    from crystargetbench.workers.lifecycle import start_parent_watchdog
    start_parent_watchdog()
    if not config["test_mode"] or config["model_lock"].get("evidence_kind") != "synthetic":
        raise ValueError("analytic fixture requires explicit synthetic test session")
    from pathlib import Path
    Path(config["workdir"], "watchdog.pid").write_text(str(os.getpid()))
    calculator = AnalyticPair()
    requests = 0
    for line in sys.stdin:
        request = _loads(line)
        validate_request(request)
        requests += 1
        print("analytic_test log on stderr", file=sys.stderr, flush=True)
        if args.mode == "crash":
            os._exit(23)
        if args.mode == "timeout":
            time.sleep(30)
        if args.mode == "bad_json":
            print("{this is invalid", flush=True)
            continue
        if args.mode == "stdout_log":
            print("unsolicited logger output", flush=True)
        atoms = atoms_from_snapshot(request["geometry"])
        calculator.calculate(atoms)
        raw = calculator.results
        observation = normalize_efs(raw["energy"], raw["forces"],
            None if args.mode == "missing_stress" else raw["stress"], len(atoms))
        result = {**response_identity(request), "status": "completed",
            "actual_geometry_id": request["geometry_id"], "mapping_digest": digest(request["mapping"]),
            "observation": observation, "provider": "analytic_test", "evidence_kind": "synthetic",
            "environment": {"model_loads_this_session": 1, "requests": requests,
                            "pid": os.getpid(), "analytic_fixture": "harmonic_pair.v1"},
            "cost": {"backend_forwards": 1, "evaluated_structures": 1, "wall_seconds": 0.0}}
        if args.mode == "wrong_request":
            result["request_id"] = "another-request"
        if args.mode == "wrong_geometry":
            result["actual_geometry_id"] = "wrong-geometry"
        if args.mode == "wrong_mapping":
            result["mapping_digest"] = "wrong-mapping"
        if args.mode == "nonfinite":
            import json
            result["observation"]["energy_eV"] = float("nan")
            print(json.dumps(result), flush=True)
            continue
        if args.mode == "wrong_family":
            result["evidence_kind"] = "mlff"
        if args.mode == "wrong_provider":
            result["provider"] = "mattersim"
        if args.mode == "wrong_units":
            result["observation"]["raw_units"]["stress"] = "GPa"
        if args.mode == "oom":
            result.update(status="error", error={"category": "out_of_memory", "message": "synthetic OOM fault injection"})
        print(canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
