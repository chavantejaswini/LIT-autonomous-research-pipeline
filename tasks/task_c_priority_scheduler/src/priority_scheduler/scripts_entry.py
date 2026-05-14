"""Entry points for `scheduler-stress` and `scheduler-stress-sweep`."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .stress import run_stress, run_stress_sweep


def stress_main() -> None:
    parser = argparse.ArgumentParser(description="Run the 50-workload stress test under one seed.")
    parser.add_argument("--config", default="configs/priorities.yaml")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--audit-log", default="results/stress_audit.jsonl")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--report", default="results/stress_report.json")
    args = parser.parse_args()

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    result = run_stress(
        config_path=args.config,
        n_workloads=args.n,
        slots=args.slots,
        seed=args.seed,
        audit_log_path=args.audit_log,
        checkpoint_dir=args.checkpoint_dir,
    )

    summary = {
        "completed": result.completed,
        "cancelled": result.cancelled,
        "failed": result.failed,
        "credits_used": result.credits_used,
        "credits_budget": result.credits_budget,
        "credit_budget_respected": result.credits_used <= result.credits_budget,
        "total_yields": result.total_yields,
        "total_resumes": result.total_resumes,
        "yields_match_resumes": result.total_yields == result.total_resumes,
        "audit_entries": result.audit_entries,
        "final_tick": result.final_tick,
        "priority_admission_inversions": result.priority_admission_inversions,
        "no_priority_inversion": result.priority_admission_inversions == 0,
    }
    Path(args.report).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def sweep_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the stress test across many seeds and report aggregated invariants."
    )
    parser.add_argument("--config", default="configs/priorities.yaml")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--audit-dir", default=None)
    parser.add_argument("--checkpoint-dir", default="results/sweep_ckpts")
    parser.add_argument("--report", default="results/stress_sweep_report.json")
    args = parser.parse_args()

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    result = run_stress_sweep(
        config_path=args.config,
        n_workloads=args.n,
        slots=args.slots,
        seeds=args.seeds,
        audit_log_dir=args.audit_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    summary = {
        "seeds_run": result.seeds_run,
        "n_workloads_per_seed": args.n,
        "all_budgets_respected": result.all_budgets_respected,
        "all_priority_invariant_held": result.all_priority_invariant_held,
        "all_yields_match_resumes": result.all_yields_match_resumes,
        "completed_total": result.completed_total,
        "failed_total": result.failed_total,
        "cancelled_total": result.cancelled_total,
        "total_yields": result.total_yields,
        "total_resumes": result.total_resumes,
        "seed_failures": result.seed_failures,
    }
    Path(args.report).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
