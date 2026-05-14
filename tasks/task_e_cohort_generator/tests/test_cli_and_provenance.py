"""CLI smoke test + provenance manifest correctness."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from cohort_generator import generate, load_config, provenance

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_provenance_manifest_includes_required_fields() -> None:
    config = load_config(EXAMPLES / "metabolic_t2d_like.json")
    manifest = provenance(config, n=500, seed=7)
    assert manifest["n"] == 500
    assert manifest["seed"] == 7
    assert manifest["disease_name"] == config.disease_name
    assert len(manifest["config_sha256"]) == 64
    assert manifest["generator_version"]
    assert manifest["generated_at"]
    assert isinstance(manifest["config_inline"], dict)


def test_provenance_hash_is_stable_for_same_config() -> None:
    config = load_config(EXAMPLES / "metabolic_t2d_like.json")
    a = provenance(config, n=100, seed=1)
    b = provenance(config, n=100, seed=1)
    assert a["config_sha256"] == b["config_sha256"]


def test_cli_writes_csv_and_provenance(tmp_path) -> None:
    out_csv = tmp_path / "cohort.csv"
    out_prov = tmp_path / "cohort.provenance.json"
    config_path = EXAMPLES / "metabolic_t2d_like.json"

    result = subprocess.run(
        [
            sys.executable, "-m", "cohort_generator.cli",
            str(config_path),
            "--n", "500",
            "--seed", "42",
            "--out", str(out_csv),
            "--provenance", str(out_prov),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 500
    assert "subtype" in df.columns

    assert out_prov.exists()
    manifest = json.loads(out_prov.read_text())
    assert manifest["n"] == 500
    assert manifest["seed"] == 42


def test_cli_no_provenance_flag_skips_manifest(tmp_path) -> None:
    out_csv = tmp_path / "cohort.csv"
    config_path = EXAMPLES / "metabolic_t2d_like.json"
    subprocess.run(
        [
            sys.executable, "-m", "cohort_generator.cli",
            str(config_path), "--n", "100", "--out", str(out_csv),
            "--no-provenance",
        ],
        capture_output=True, text=True, check=True,
    )
    assert out_csv.exists()
    # Default provenance path should NOT exist.
    assert not (tmp_path / "cohort.csv.provenance.json").exists()
