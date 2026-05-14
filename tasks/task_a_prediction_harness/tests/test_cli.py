"""Smoke test for the `prediction-harness` CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def _cli(db: Path, *args: str) -> dict:
    """Invoke the CLI as a subprocess and return parsed stdout JSON."""
    result = subprocess.run(
        [sys.executable, "-m", "prediction_harness.cli", "--db", str(db), *args],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_cli_register_record_report_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "p.db"

    # Register
    reg = _cli(
        db, "register",
        "--model-id", "cli_test",
        "--dataset-hash", "ds_x",
        "--prediction-json", json.dumps({"probability": 0.8}),
    )
    pid = reg["prediction_id"]
    assert len(pid) == 64

    # Record an outcome strictly after registration. Use a far-future
    # ISO time so any clock skew is irrelevant.
    future = (datetime.utcnow() + timedelta(days=30)).isoformat()
    rec = _cli(
        db, "record",
        "--prediction-id", pid,
        "--outcome-json", json.dumps({"label": 1}),
        "--observed-at", future,
    )
    assert rec["status"] == "recorded"

    # Report — calibration window wide enough to include the prediction.
    rep = _cli(
        db, "report",
        "--model-id", "cli_test",
        "--start", "2020-01-01T00:00:00",
        "--end", "2030-01-01T00:00:00",
    )
    assert rep["num_registered"] == 1
    assert rep["num_with_outcomes"] == 1
    assert abs(rep["brier_score"] - 0.04) < 1e-9
    assert rep["accuracy_at_0_5"] == 1.0
    assert rep["num_realized_positives"] == 1


def test_cli_report_with_dataset_hash_filter(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    _cli(
        db, "register",
        "--model-id", "m",
        "--dataset-hash", "ds_A",
        "--prediction-json", json.dumps({"probability": 0.7}),
    )
    _cli(
        db, "register",
        "--model-id", "m",
        "--dataset-hash", "ds_B",
        "--prediction-json", json.dumps({"probability": 0.7}),
    )
    rep = _cli(
        db, "report",
        "--model-id", "m",
        "--start", "2020-01-01T00:00:00",
        "--end", "2030-01-01T00:00:00",
        "--dataset-hash", "ds_A",
    )
    assert rep["num_registered"] == 1
    assert rep["dataset_hash"] == "ds_A"
