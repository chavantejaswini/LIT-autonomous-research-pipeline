"""50-workload stress test verifying the three required invariants."""
from __future__ import annotations

from pathlib import Path

from priority_scheduler.stress import run_stress

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "priorities.yaml"


def test_stress_50_workloads_invariants(tmp_path) -> None:
    result = run_stress(
        config_path=CONFIG,
        n_workloads=50,
        slots=4,
        seed=7,
        audit_log_path=tmp_path / "audit.jsonl",
        checkpoint_dir=tmp_path / "ckpts",
    )

    # (a) Total credit usage never exceeds budget.
    assert result.credits_used <= result.credits_budget, (
        f"credits_used={result.credits_used} > budget={result.credits_budget}"
    )

    # (b) Higher priorities admit before lower — no priority inversion at any
    # admission decision (`admitted` or `resumed` events).
    assert result.priority_admission_inversions == 0, (
        f"{result.priority_admission_inversions} priority inversions detected"
    )

    # (c) No workload killed — every yield is followed by a matching resume.
    # That means total yields == total resumes (since no terminal yielded state).
    assert result.total_yields == result.total_resumes, (
        f"yields={result.total_yields} resumes={result.total_resumes}"
    )

    # And every workload reached completion (no kills / hangs).
    assert result.completed == 50
    assert result.cancelled == 0
