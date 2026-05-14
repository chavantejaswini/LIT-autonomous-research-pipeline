"""Multi-seed sweep: assert the three invariants across many seeds.

This is much stronger evidence than a single-seed stress run. Hitting
100 different randomized workload mixes without breaking any invariant
demonstrates the scheduler isn't relying on lucky timing.
"""
from __future__ import annotations

from pathlib import Path

from priority_scheduler.stress import run_stress_sweep

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "priorities.yaml"


def test_stress_sweep_100_seeds(tmp_path) -> None:
    result = run_stress_sweep(
        config_path=CONFIG,
        n_workloads=50,
        slots=4,
        seeds=100,
        checkpoint_dir=tmp_path / "ckpts",
    )
    assert result.seeds_run == 100
    assert result.all_budgets_respected, f"failures: {result.seed_failures}"
    assert result.all_priority_invariant_held, f"failures: {result.seed_failures}"
    assert result.all_yields_match_resumes, f"failures: {result.seed_failures}"
    # The full sweep should have produced at least *some* pre-emptions —
    # otherwise the test isn't exercising the interesting paths.
    assert result.total_yields > 0
    assert result.total_resumes == result.total_yields
