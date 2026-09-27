"""Authoritative JSON-shaped adapter API v1; validation lives in validation.py."""
from pathlib import Path
from typing import Protocol, TypedDict

API_VERSION = 1

class AdapterManifest(TypedDict):
    schema_version: str
    api_version: int
    provider_id: str
    adapter_id: str
    adapter_version: str
    distribution: str
    family: str
    capabilities: list[dict]
    science_validated: bool
    benchmark_eligible: bool

class CalculationRequest(TypedDict):
    schema_version: str
    adapter_api_version: int
    request_id: str
    request_digest: str
    adapter: dict
    sample_geometry_id: str
    recipe: dict
    sampling: dict
    operation: str
    requested_quantities: list[str]
    geometry: dict
    method: dict
    method_digest: str
    dependencies: list[str]
    execution_mode: str

class PreparedJob(TypedDict):
    schema_version: str
    request_id: str
    request_digest: str
    status: str
    files: list[dict]
    mapping: dict
    reasons: list[str]

class CalculationResult(TypedDict):
    schema_version: str
    adapter_api_version: int
    request_id: str
    request_digest: str
    reference_geometry_id: str
    observed_geometry: dict | None
    method_digest: str
    effective_method: dict
    adapter: dict
    engine: dict
    mapping: dict
    convergence: dict
    execution: dict
    observations: list[dict]
    errors: list[str]

class DFTAdapter(Protocol):
    def describe(self) -> AdapterManifest: ...
    def prepare(self, request: CalculationRequest, directory: Path, config: dict) -> PreparedJob: ...
    def collect(self, request: CalculationRequest, directory: Path, config: dict) -> CalculationResult: ...
