"""Pluggable per-tick execution backend.

The scheduler decides *who* runs each tick; a `RuntimeAdapter` decides *how*
the chosen workloads' `advance()` calls actually execute. The default is
synchronous (one workload at a time, inside the scheduler thread). The
threaded variant runs them in parallel across an OS thread pool — useful
when the workloads are I/O-bound or release the GIL during heavy work
(NumPy, PyTorch, etc.).

This is the textbook "strategy" pattern: scheduler logic stays unchanged;
the runtime can be swapped without touching `scheduler.py`.

Contract
--------
`advance_all(jobs, ticks)` returns a `dict[job_id -> Exception | None]`.
The scheduler routes any non-None entry through `_mark_failed` and keeps
running — no workload exception ever bubbles out of the adapter.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol


class RuntimeAdapter(Protocol):
    """How to drive `workload.advance(ticks)` across the running jobs of a tick."""

    def advance_all(self, jobs: list, ticks: int = 1) -> dict[str, Exception | None]: ...
    def shutdown(self) -> None: ...


class SynchronousRuntime:
    """The default — execute advances sequentially in the caller thread.

    This is what every existing test exercises; behavior is byte-identical
    to a scheduler with no runtime configured.
    """

    def advance_all(self, jobs, ticks: int = 1) -> dict[str, Exception | None]:
        outcomes: dict[str, Exception | None] = {}
        for j in jobs:
            try:
                j.workload.advance(ticks)
                outcomes[j.handle.job_id] = None
            except Exception as exc:
                outcomes[j.handle.job_id] = exc
        return outcomes

    def shutdown(self) -> None:
        pass


class ThreadedRuntime:
    """Advance each running workload in parallel via a thread pool.

    Suitable when `workload.advance(...)` is I/O-bound or GIL-releasing
    (PyTorch/NumPy heavy ops, file I/O, network calls). The thread pool
    lives for the scheduler's lifetime; call `Scheduler.shutdown()` (or
    use it as a context manager) to release the threads.

    The scheduler still ticks deterministically — only the per-tick
    `advance()` calls are parallelized. Admission, pre-emption, and
    audit-log ordering remain single-threaded.
    """

    def __init__(self, max_workers: int = 8) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def advance_all(self, jobs, ticks: int = 1) -> dict[str, Exception | None]:
        if not jobs:
            return {}
        futures = {
            j.handle.job_id: self._pool.submit(_safe_advance, j.workload, ticks)
            for j in jobs
        }
        return {jid: f.result() for jid, f in futures.items()}

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


def _safe_advance(workload, ticks: int) -> Exception | None:
    """Helper run inside the pool — return any exception rather than propagating."""
    try:
        workload.advance(ticks)
        return None
    except Exception as exc:
        return exc
