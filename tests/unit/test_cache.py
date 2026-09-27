"""Atomic cache and synthetic SpyBackend tests; no physical calculations."""

import json
from copy import deepcopy

import pytest

from crystargetbench.assessment import assess, make_measurement
from crystargetbench.cache import StageCache, stage_key
from crystargetbench.identity import digest


def arguments():
    return dict(
        geometry=digest({"cell": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "species": ["X"], "positions": [[0, 0, 0]]}),
        recipe="native.space_group.v1",
        parameters={"symprec_A": 0.01, "angle_tolerance_deg": 5.0, "target_symmetrization": False},
        backend={"id": "synthetic_spy", "family": "synthetic", "version": "1", "assets": {}},
        dependencies=[],
    )


class SpyBackend:
    def __init__(self):
        self.calls = 0

    def observe(self, geometry):
        self.calls += 1
        return make_measurement(value=225, property_id="space_group_number", unit="1", definition_id="synthetic.sg.v1", reference_geometry_id=geometry,
            backend_id="synthetic_spy", backend_family="synthetic", backend_version="1", physics_fidelity="synthetic", protocol_digest="synthetic-tests",
            calculation_status="completed", quality_status="accepted", identity_status="verified")


def test_spy_cache_reuse_for_duplicate_and_changed_threshold(tmp_path):
    cache = StageCache(tmp_path)
    args = arguments()
    key = stage_key(**args)
    spy = SpyBackend()
    task = {"task_id": "synthetic_cache", "mode": "screen", "measurements": {"sg": {"property_id": "space_group_number", "unit": "1"}}, "constraints": [{"id": "target", "measurement": "sg", "operator": "eq", "value": 225}]}
    outcomes = []
    for target in (225, 225, 221):
        observation = cache.get(key)
        if observation is None:
            observation = spy.observe(args["geometry"])
            cache.put(key, observation)
        task["constraints"][0]["value"] = target
        outcomes.append(assess(task, {"sg": observation}, args["geometry"])["decision"])
    assert outcomes == ["pass", "pass", "fail"]
    assert spy.calls == 1


def test_backend_asset_change_invalidates_key():
    args = arguments()
    old = stage_key(**args)
    args["backend"]["assets"]["checkpoint"] = digest("new-checkpoint")
    assert stage_key(**args) != old


def test_stage_parameters_split_ifc_from_frequency_sampling():
    args = arguments()
    args.update(recipe="synthetic.ifc.v1", parameters={"displacement_A": 0.01, "supercell": [2, 2, 2]})
    ifc_key = stage_key(**args)
    changed = deepcopy(args)
    changed["parameters"]["displacement_A"] = 0.02
    assert stage_key(**changed) != ifc_key
    frequencies = dict(args, recipe="synthetic.frequencies.v1", parameters={"qmesh": [4, 4, 4], "nac": False}, dependencies=[ifc_key])
    first = stage_key(**frequencies)
    frequencies["parameters"]["qmesh"] = [8, 8, 8]
    changed_sampling = stage_key(**frequencies)
    assert changed_sampling != first
    assert stage_key(**args) == ifc_key
    frequencies["parameters"]["nac"] = True
    assert stage_key(**frequencies) != changed_sampling


def test_displacement_change_invalidates_forces_and_dependent_ifc():
    forces = dict(arguments(), recipe="synthetic.displaced_forces.v1", parameters={"displacement_A": 0.01, "site_index": 0, "axis": 0})
    first_forces = stage_key(**forces)
    ifc = dict(arguments(), recipe="synthetic.ifc.v1", parameters={"supercell": [2, 2, 2], "force_constants_symmetrization": True}, dependencies=[first_forces])
    first_ifc = stage_key(**ifc)
    forces["parameters"]["displacement_A"] = 0.02
    next_forces = stage_key(**forces)
    assert next_forces != first_forces
    ifc["dependencies"] = [next_forces]
    assert stage_key(**ifc) != first_ifc


@pytest.mark.parametrize("field", ["geometry", "recipe", "parameters", "backend", "dependencies"])
def test_omitted_key_input_rejected(field):
    args = arguments()
    del args[field]
    with pytest.raises(TypeError):
        stage_key(**args)


@pytest.mark.parametrize("mutate", [
    lambda args: args["parameters"].pop("angle_tolerance_deg"),
    lambda args: args["backend"].pop("assets"),
    lambda args: args["backend"].update(version="latest"),
    lambda args: args.update(parameters={}),
    lambda args: args["parameters"].update(symprec_A=float("nan")),
    lambda args: args["parameters"].update(symprec_A=True),
    lambda args: args["parameters"].update(target_symmetrization=True),
    lambda args: args.update(geometry="formula-only"),
    lambda args: args.update(dependencies=["unresolved-node-name"]),
    lambda args: args["backend"]["assets"].update(checkpoint="filename-not-content-digest"),
])
def test_incomplete_or_invalid_semantic_key_rejected(mutate):
    args = arguments()
    mutate(args)
    with pytest.raises(ValueError):
        stage_key(**args)


def test_cache_atomic_roundtrip_and_relocation(tmp_path, monkeypatch):
    key = stage_key(**arguments())
    cache = StageCache(tmp_path / "cache")
    cache.put(key, {"value": 225, "physics_fidelity": "synthetic"})
    assert list((tmp_path / "cache").glob("*.tmp")) == []
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert stage_key(**arguments()) == key
    assert cache.get(key)["value"] == 225


def test_incomplete_and_temporary_records_are_misses(tmp_path):
    cache = StageCache(tmp_path)
    key = stage_key(**arguments())
    (tmp_path / f".{key}.tmp").write_text('{"value":225}')
    assert cache.get(key) is None
    cache.put(key, {"value": 225})
    path = tmp_path / f"{key}.json"
    envelope = json.loads(path.read_text())
    envelope["complete"] = False
    path.write_text(json.dumps(envelope))
    assert cache.get(key) is None


def test_tampered_payload_and_wrong_key_are_misses(tmp_path):
    cache = StageCache(tmp_path)
    key = stage_key(**arguments())
    cache.put(key, {"value": 225})
    path = tmp_path / f"{key}.json"
    envelope = json.loads(path.read_text())
    envelope["payload"]["value"] = 221
    path.write_text(json.dumps(envelope))
    assert cache.get(key) is None
    cache.put(key, {"value": 225})
    envelope = json.loads(path.read_text())
    envelope["key"] = "0" * 64
    path.write_text(json.dumps(envelope))
    assert cache.get(key) is None


@pytest.mark.parametrize("contents", ['{"value":NaN}', '{"unfinished":', '[]', 'null', '"unexpected"'])
def test_corrupt_json_misses(tmp_path, contents):
    key = stage_key(**arguments())
    (tmp_path / f"{key}.json").write_text(contents)
    assert StageCache(tmp_path).get(key) is None


def test_nonfinite_payload_and_path_traversal_rejected(tmp_path):
    cache = StageCache(tmp_path)
    with pytest.raises(ValueError):
        cache.put(stage_key(**arguments()), {"value": float("inf")})
    with pytest.raises(ValueError):
        cache.get("../../escape")


def test_symlink_not_followed(tmp_path):
    cache = StageCache(tmp_path / "cache")
    key = stage_key(**arguments())
    cache.put(key, {"value": 225})
    path = cache.root / f"{key}.json"
    moved = tmp_path / "elsewhere.json"
    path.rename(moved)
    path.symlink_to(moved)
    assert cache.get(key) is None


def test_interrupted_atomic_replace_preserves_previous_entry(tmp_path, monkeypatch):
    cache = StageCache(tmp_path)
    key = stage_key(**arguments())
    cache.put(key, {"value": 225})

    def interrupted(*args):
        raise OSError("synthetic interrupted rename")

    monkeypatch.setattr("crystargetbench.cache.os.replace", interrupted)
    with pytest.raises(OSError):
        cache.put(key, {"value": 221})
    assert cache.get(key) == {"value": 225}
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("family,required", [("dft", "pseudopotential_manifest"), ("mlff", "checkpoint"), ("surrogate", "checkpoint")])
def test_physics_cache_requires_locked_asset_identity(family, required):
    args = arguments()
    args["backend"]["family"] = family
    with pytest.raises(ValueError, match="digest"):
        stage_key(**args)
    args["backend"]["assets"][required] = digest("synthetic test asset identity; no actual asset loaded")
    assert len(stage_key(**args)) == 64


def test_duplicate_json_keys_never_hit_even_with_matching_final_checksum(tmp_path):
    cache = StageCache(tmp_path)
    key = stage_key(**arguments())
    cache.put(key, {"value": 229})
    path = tmp_path / f"{key}.json"
    path.write_text(path.read_text().replace('"value":229', '"value":225,"value":229'))
    assert cache.get(key) is None
