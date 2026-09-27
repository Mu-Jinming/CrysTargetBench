"""Persistent argv-only worker sessions with bounded protocol I/O and attempts."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import threading
import time
import uuid

from ..assets import validate_permit, verify_asset, verify_environment_file
from ..contracts import _loads
from ..identity import canonical_json, digest
from .protocol import PROTOCOL_VERSION, validate_request, validate_response


class WorkerError(RuntimeError):
    def __init__(self, category, message, attempt=None):
        super().__init__(message)
        self.category, self.attempt = category, attempt


class WorkerClient:
    """One model/session, serial requests, no implicit retry or backend change.

    ``argv`` and model records must come from trusted deployment. ``test_mode``
    accepts only an explicitly synthetic model lock and never enables MatterSim.
    Budget accounting/reservation is owned by the calling execution context.
    """
    def __init__(self, argv, workdir, model_lock, timeout_seconds=60, *, permit=None,
                 test_mode=False, bindings=None, environment_lock=None, environment_lock_path=None,
                 max_message_bytes=32 * 1024 * 1024, scratch_check=None):
        if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a or "\x00" in a for a in argv):
            raise ValueError("trusted worker argv must be a nonempty string array")
        if not Path(argv[0]).is_absolute():
            raise ValueError("trusted worker interpreter must be an explicit absolute path")
        if not 0 < timeout_seconds < float("inf"):
            raise ValueError("finite positive timeout required")
        self.argv = list(argv)
        self.workdir = Path(workdir).resolve()
        self.model_lock = deepcopy(model_lock)
        self.permit = deepcopy(permit)
        self.bindings = deepcopy(bindings)
        self.environment_lock = deepcopy(environment_lock)
        self.environment_lock_path = str(Path(environment_lock_path).resolve()) if environment_lock_path is not None else None
        self.timeout_seconds = float(timeout_seconds)
        self.max_message_bytes = max_message_bytes
        if scratch_check is not None and not callable(scratch_check):
            raise ValueError("scratch_check must be the execution budget callback")
        self.scratch_check = scratch_check
        self.test_mode = bool(test_mode)
        if self.test_mode:
            if self.model_lock.get("evidence_kind") != "synthetic" or self.model_lock.get("provider") != "analytic_test":
                raise ValueError("test worker requires synthetic/analytic_test model identity")
            if "model_lock_digest" not in self.model_lock:
                self.model_lock["model_lock_digest"] = digest(self.model_lock)
        else:
            verified = verify_asset(self.model_lock["manifest"])
            if verified["model_lock_digest"] != self.model_lock.get("model_lock_digest"):
                raise ValueError("model lock identity mismatch")
            validate_permit(self.permit, verified, self.bindings)
            if not self.environment_lock:
                raise ValueError("reviewed worker environment lock is required for production")
            verify_environment_file(self.environment_lock, self.environment_lock_path,
                                    self.permit["worker_environment_lock_sha256"])
        self.process = None
        self._mutex = threading.Lock()
        self._cancelled = threading.Event()
        self._buffer = b""
        self._stderr = bytearray()
        self._stderr_thread = None
        self.attempts = []
        self.session_id = uuid.uuid4().hex

    def _start(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        config = {"model_lock": self.model_lock, "permit": self.permit,
            "bindings": self.bindings, "test_mode": self.test_mode,
            "environment_lock": self.environment_lock,
            "environment_lock_path": self.environment_lock_path,
            "workdir": str(self.workdir), "session_id": self.session_id}
        config_path = self.workdir / (self.session_id + ".session.json")
        encoded = canonical_json(config)
        self._check_scratch(len(encoded.encode("utf-8")) + 1024)
        config_path.write_text(encoded, encoding="utf-8")
        config_path.chmod(0o600)
        env = dict(os.environ)
        env.pop("PYTHONHOME", None)
        if not self.test_mode:
            env.pop("PYTHONPATH", None)
        env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                   CTB_PARENT_PID=str(os.getpid()),
                   TORCH_HOME=str(self.workdir / "torch-cache"),
                   XDG_CACHE_HOME=str(self.workdir / "cache"),
                   MPLCONFIGDIR=str(self.workdir / "matplotlib"))
        self.process = subprocess.Popen(self.argv + ["--session-config", str(config_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self.workdir, env=env, shell=False, start_new_session=True)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _check_scratch(self, additional_bytes=0):
        if self.scratch_check is not None:
            try:
                self.scratch_check(additional_bytes)
            except Exception as exc:
                raise WorkerError("budget_exhausted", f"worker artifact scratch budget rejected: {exc}") from exc

    def _drain_stderr(self):
        # Drain without unbounded RAM/disk even if a malfunctioning worker logs forever.
        stream = self.process.stderr
        while True:
            try:
                block = os.read(stream.fileno(), 4096)
            except (OSError, ValueError):
                break
            if not block:
                break
            self._stderr.extend(block)
            if len(self._stderr) > 65536:
                del self._stderr[:-65536]

    def _read_response(self, deadline):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                if self._cancelled.is_set():
                    raise WorkerError("cancelled", "worker session cancelled")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise WorkerError("timeout", "worker request exceeded wall deadline")
                events = selector.select(min(left, 0.1))
                if events:
                    block = os.read(self.process.stdout.fileno(), 65536)
                    if not block:
                        if self._cancelled.is_set():
                            raise WorkerError("cancelled", "worker session cancelled")
                        raise WorkerError("worker_crashed", "worker exited without a complete response")
                    self._buffer += block
                    if len(self._buffer) > self.max_message_bytes:
                        raise WorkerError("protocol_error", "worker message size limit exceeded")
            line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            return _loads(line.decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise WorkerError("protocol_error", "worker stdout is not strict NDJSON") from exc

    def _send_request(self, raw, deadline):
        """A worker that stops reading cannot block a pipe write indefinitely."""
        stream = self.process.stdin
        os.set_blocking(stream.fileno(), False)
        sent = 0
        with selectors.DefaultSelector() as selector:
            selector.register(stream, selectors.EVENT_WRITE)
            while sent < len(raw):
                if self._cancelled.is_set():
                    raise WorkerError("cancelled", "worker session cancelled")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise WorkerError("timeout", "worker stdin exceeded wall deadline")
                if selector.select(min(left, 0.1)):
                    try:
                        sent += os.write(stream.fileno(), raw[sent:])
                    except BlockingIOError:
                        continue

    def evaluate(self, snapshot, node_id, mapping=None, budget_lease=None):
        if not self._mutex.acquire(blocking=False):
            raise WorkerError("concurrency_limit", "one in-flight request per worker session")
        request = None
        start = time.monotonic()
        try:
            if self._cancelled.is_set():
                raise WorkerError("cancelled", "worker session cancelled")
            if mapping is None:
                mapping = {"site_mapping": list(range(len(snapshot["species"]))),
                           "basis_mapping": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
            request = {"protocol_version": PROTOCOL_VERSION, "request_id": uuid.uuid4().hex,
                "attempt_id": uuid.uuid4().hex, "node_id": str(node_id),
                "operation": "energy_forces_stress", "geometry": deepcopy(snapshot),
                "geometry_id": snapshot["geometry_id"], "mapping": deepcopy(mapping),
                "model_lock_digest": self.model_lock["model_lock_digest"],
                "budget_lease": deepcopy(budget_lease)}
            validate_request(request)
            self._check_scratch()
            wall_limit = min(self.timeout_seconds, request["budget_lease"].get("max_wall_seconds", self.timeout_seconds))
            if not isinstance(wall_limit, (int, float)) or not 0 < wall_limit < float("inf"):
                raise WorkerError("budget_exhausted", "reserved lease has no remaining wall budget")
            if self.process is None:
                self._start()
            elif self.process.poll() is not None:
                raise WorkerError("worker_crashed", "session exited; explicit new attempt/session required")
            raw = canonical_json(request).encode() + b"\n"
            if len(raw) > self.max_message_bytes:
                raise WorkerError("protocol_error", "request exceeds message limit")
            try:
                self._send_request(raw, start + wall_limit)
            except (BrokenPipeError, OSError) as exc:
                raise WorkerError("worker_crashed", "worker stdin closed") from exc
            try:
                response = self._read_response(start + wall_limit)
            except (OSError, ValueError) as exc:
                if self._cancelled.is_set():
                    raise WorkerError("cancelled", "worker session cancelled") from exc
                raise WorkerError("worker_crashed", "worker output pipe closed") from exc
            try:
                validate_response(response, request)
            except (ValueError, KeyError, TypeError) as exc:
                raise WorkerError("protocol_error", str(exc)) from exc
            if response["status"] == "error":
                raise WorkerError(response["error"].get("category", "backend_error"),
                                  response["error"].get("message", "worker failed"))
            evidence = response.get("evidence_kind")
            if evidence != ("synthetic" if self.test_mode else "mlff"):
                raise WorkerError("protocol_error", "worker evidence family mismatch")
            if response.get("provider") != ("analytic_test" if self.test_mode else "mattersim"):
                raise WorkerError("protocol_error", "worker provider identity mismatch")
            self._check_scratch()
            attempt = self._record(request, "completed", start, response=response)
            result = deepcopy(response["observation"])
            result.update({key: deepcopy(response[key]) for key in
                ("geometry_id", "actual_geometry_id", "mapping_digest", "model_lock_digest",
                 "environment", "cost", "evidence_kind", "provider")})
            result.update(attempt_id=attempt["attempt_id"], request_id=request["request_id"],
                          benchmark_eligible=False, science_validated=False)
            return result
        except WorkerError as exc:
            if self._cancelled.is_set():
                exc = WorkerError("cancelled", "worker session cancelled")
            self._terminate()
            if request is not None and exc.attempt is None:
                try:
                    exc.attempt = self._record(request, exc.category, start, error=str(exc))
                except WorkerError as record_error:
                    exc = record_error
            raise exc
        except BaseException:
            # Invalid local data, disk errors and interrupts must not leave a
            # model process running after this call has failed.
            self._terminate()
            raise
        finally:
            self._mutex.release()

    def _record(self, request, status, start, **details):
        entry = {"request_id": request["request_id"], "attempt_id": request["attempt_id"],
            "node_id": request["node_id"], "geometry_id": request["geometry_id"],
            "model_lock_digest": request["model_lock_digest"], "status": status,
            "elapsed_seconds": time.monotonic() - start,
            "retry_provider": None, "stderr_tail": bytes(self._stderr).decode("utf-8", errors="replace"), **details}
        self.workdir.mkdir(parents=True, exist_ok=True)
        target = self.workdir / (entry["attempt_id"] + ".attempt.json")
        temp = target.with_suffix(".partial")
        encoded = canonical_json(entry)
        try:
            self._check_scratch(len(encoded.encode("utf-8")) * 2 + 1024)
        except WorkerError:
            # Preserve identity and uncertainty in memory even if no disk bytes
            # remain. Try a small status record, without claiming raw completion.
            compact = {key: entry[key] for key in ("request_id", "attempt_id", "node_id", "geometry_id", "model_lock_digest")}
            compact.update(status="budget_exhausted", prior_status=status,
                reason="attempt artifact exceeds scratch budget", physics_calls=None,
                backend_forwards=details.get("response", {}).get("cost", {}).get("backend_forwards"),
                artifact_saved=False, retry_provider=None)
            compact_text = canonical_json(compact)
            try:
                self._check_scratch(len(compact_text.encode("utf-8")) * 2 + 1024)
                compact["artifact_saved"] = True
                temp.write_text(canonical_json(compact), encoding="utf-8")
                os.replace(temp, target)
            except WorkerError:
                pass
            self.attempts.append(compact)
            raise WorkerError("budget_exhausted", "worker attempt could not be committed within scratch budget", compact)
        self.attempts.append(entry)
        temp.write_text(encoded, encoding="utf-8")
        os.replace(temp, target)
        return entry

    def _terminate(self):
        process = self.process
        if process is None:
            return
        # Kill the complete session, including any descendants, even after leader exit.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=1)
        # A child ignoring TERM can survive the leader; always close the group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self._stderr_thread:
            self._stderr_thread.join(timeout=1)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream:
                stream.close()

    def cancel(self):
        self._cancelled.set()
        self._terminate()

    def close(self):
        self._terminate()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
