"""Concurrent-write safety: parallel registrations on a file-backed SQLite
DB must converge to the right outcome — N unique predictions registered
without races, K duplicate registrations all raising `ImmutablePredictionError`.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from prediction_harness import Harness, ImmutablePredictionError
from prediction_harness.sqlite_dao import SQLitePredictionStore


def test_parallel_distinct_registrations_all_succeed(tmp_path, clock):
    url = f"sqlite:///{tmp_path / 'pred.db'}"
    h = Harness(store=SQLitePredictionStore(url=url), clock=clock)

    def register(i: int):
        return h.register_prediction("m", "ds", {"probability": 0.5, "i": i})

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(register, range(100)))

    assert len(set(ids)) == 100  # all distinct


def test_parallel_duplicate_registrations_all_but_one_raise(tmp_path, clock):
    url = f"sqlite:///{tmp_path / 'pred.db'}"
    h = Harness(store=SQLitePredictionStore(url=url), clock=clock)
    barrier = threading.Barrier(8)

    successes = 0
    failures = 0
    lock = threading.Lock()

    def attempt():
        nonlocal successes, failures
        barrier.wait()  # release all threads at once
        try:
            h.register_prediction("m", "ds", {"probability": 0.42, "key": "same"})
            with lock:
                successes += 1
        except ImmutablePredictionError:
            with lock:
                failures += 1

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The append-only invariant: exactly one register wins; the rest see
    # the unique-constraint violation translated into ImmutablePredictionError.
    assert successes == 1
    assert failures == 7
