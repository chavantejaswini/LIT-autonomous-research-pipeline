"""Per-source circuit breaker.

Three states per source:

  * CLOSED — calls flow through normally.
  * OPEN — calls fail fast (`CircuitOpenError`) without touching the
    network. Triggered after `failure_threshold` consecutive failures.
  * HALF_OPEN — after `recovery_seconds`, the next call probes the
    source. Success closes the circuit; failure re-opens it for another
    `recovery_seconds` window.

Threadsafe: an internal lock protects every state transition.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class CircuitOpenError(RuntimeError):
    """Returned when the circuit for a given source is OPEN.

    Surfaced as a `Failure(message="circuit-open", status_code=None)`
    by the source-client layer — callers see a typed failure, not an
    exception.
    """


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _SourceState:
    state: State = State.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0  # monotonic timestamp


class CircuitBreaker:
    """Per-source circuit breaker.

    Parameters
    ----------
    failure_threshold : int
        Number of consecutive failures (per source) that flips CLOSED → OPEN.
    recovery_seconds : float
        How long the OPEN state lasts before the next call is allowed
        through as a HALF_OPEN probe.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be ≥ 1")
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be > 0")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._states: dict[str, _SourceState] = {}
        self._lock = threading.Lock()
        # Injectable monotonic clock — tests substitute a deterministic one.
        self._now = time.monotonic

    # ---- public API used by HttpRunner ------------------------------------

    def allow_request(self, source: str) -> bool:
        """Return False if calls to `source` should be short-circuited."""
        with self._lock:
            s = self._states.setdefault(source, _SourceState())
            if s.state is State.CLOSED:
                return True
            if s.state is State.OPEN:
                if self._now() - s.opened_at >= self.recovery_seconds:
                    # Transition to HALF_OPEN: let exactly one probe through.
                    s.state = State.HALF_OPEN
                    return True
                return False
            # HALF_OPEN: only one probe is in flight at a time. We optimistically
            # allow the probe and let `record_success`/`record_failure` close
            # or re-open based on the outcome. Concurrent callers may both see
            # HALF_OPEN — that's acceptable for a sloppy breaker.
            return True

    def record_success(self, source: str) -> None:
        with self._lock:
            s = self._states.setdefault(source, _SourceState())
            s.consecutive_failures = 0
            s.state = State.CLOSED
            s.opened_at = 0.0

    def record_failure(self, source: str) -> None:
        with self._lock:
            s = self._states.setdefault(source, _SourceState())
            s.consecutive_failures += 1
            if s.state is State.HALF_OPEN:
                # Probe failed — re-open the circuit.
                s.state = State.OPEN
                s.opened_at = self._now()
                return
            if s.consecutive_failures >= self.failure_threshold:
                s.state = State.OPEN
                s.opened_at = self._now()

    # ---- introspection ----------------------------------------------------

    def state(self, source: str) -> State:
        with self._lock:
            return self._states.get(source, _SourceState()).state

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {
                src: {
                    "state": s.state.value,
                    "consecutive_failures": s.consecutive_failures,
                }
                for src, s in self._states.items()
            }
