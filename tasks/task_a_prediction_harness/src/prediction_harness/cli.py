"""`prediction-harness` console entry point.

Three subcommands mirror the public API:

  register --db PATH --model-id MID --dataset-hash DH --prediction-json JSON
  record   --db PATH --prediction-id PID --outcome-json JSON --observed-at ISO
  report   --db PATH --model-id MID --start ISO --end ISO [--dataset-hash DH]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from .api import Harness
from .sqlite_dao import SQLitePredictionStore


def _harness(db_path: str) -> Harness:
    return Harness(store=SQLitePredictionStore(url=f"sqlite:///{db_path}"))


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prediction-harness")
    parser.add_argument("--db", default="predictions.db", help="SQLite DB path.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="Register a new prediction.")
    reg.add_argument("--model-id", required=True)
    reg.add_argument("--dataset-hash", required=True)
    reg.add_argument(
        "--prediction-json",
        required=True,
        help='JSON like \'{"probability": 0.74, "subject": "S-001"}\'',
    )

    rec = sub.add_parser("record", help="Record the outcome for a registered prediction.")
    rec.add_argument("--prediction-id", required=True)
    rec.add_argument(
        "--outcome-json", required=True, help='JSON like \'{"label": 1}\''
    )
    rec.add_argument(
        "--observed-at",
        required=True,
        help="ISO-8601 datetime, strictly after the prediction's registered_at.",
    )

    rep = sub.add_parser("report", help="Print a calibration report as JSON.")
    rep.add_argument("--model-id", required=True)
    rep.add_argument("--start", required=True, help="ISO-8601 start of window.")
    rep.add_argument("--end", required=True, help="ISO-8601 end of window.")
    rep.add_argument("--dataset-hash", default=None)

    args = parser.parse_args(argv)
    h = _harness(args.db)

    if args.cmd == "register":
        prediction = json.loads(args.prediction_json)
        pid = h.register_prediction(args.model_id, args.dataset_hash, prediction)
        print(json.dumps({"prediction_id": str(pid)}))
        return 0

    if args.cmd == "record":
        outcome = json.loads(args.outcome_json)
        h.record_outcome(
            args.prediction_id, outcome, observed_at=_parse_iso(args.observed_at)
        )
        print(json.dumps({"status": "recorded", "prediction_id": args.prediction_id}))
        return 0

    if args.cmd == "report":
        report = h.calibration_report(
            args.model_id,
            time_window=(_parse_iso(args.start), _parse_iso(args.end)),
            dataset_hash=args.dataset_hash,
        )
        print(report.model_dump_json(indent=2))
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
