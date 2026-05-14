"""`cohort-gen` console entry point.

Usage:
    cohort-gen examples/metabolic_t2d_like.json --n 1000 --seed 42 \
        --out cohort.csv --provenance cohort.provenance.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .generator import generate, provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic patient cohort from a DiseaseConfig JSON."
    )
    parser.add_argument("config", help="Path to a DiseaseConfig JSON file.")
    parser.add_argument("--n", type=int, default=1000, help="Number of rows (default: 1000).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    parser.add_argument("--out", default="cohort.csv", help="Output CSV path.")
    parser.add_argument(
        "--provenance",
        default=None,
        help="Optional sidecar JSON path. If omitted, a .provenance.json next "
             "to --out is written.",
    )
    parser.add_argument(
        "--no-provenance",
        action="store_true",
        help="Skip writing the provenance manifest entirely.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    df = generate(config, n=args.n, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows × {len(df.columns)} cols)")

    if not args.no_provenance:
        prov_path = (
            Path(args.provenance)
            if args.provenance
            else out_path.with_suffix(out_path.suffix + ".provenance.json")
        )
        manifest = provenance(config, n=args.n, seed=args.seed)
        prov_path.write_text(json.dumps(manifest, indent=2))
        print(f"wrote {prov_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
