#!/usr/bin/env python3
"""Recover completed PySR rows from a report truncated during JSON serialization.

The audited runner writes top-level keys in this order:
    protocol -> rows -> aggregates -> detailed

The historical failure occurred inside ``detailed`` when a SymPy ``Mul`` object
was encountered. Therefore the preceding ``protocol`` and ``rows`` values are
normally complete JSON values even though the whole document is invalid.

This utility extracts those complete values, writes a valid recovered report and
CSV, and creates the checkpoint expected by
``fase_v23_g1a_vieta_full_field_audited_m1_v2.py --resume``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def extract_json_value(text: str, key: str) -> Any:
    marker = json.dumps(key) + ":"
    pos = text.find(marker)
    if pos < 0:
        raise ValueError(f"Could not locate top-level key {key!r}")
    start = pos + len(marker)
    while start < len(text) and text[start].isspace():
        start += 1
    value, _end = json.JSONDecoder().raw_decode(text, start)
    return value


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r.get("method"), r.get("label"), r.get("primitive_regime"), r.get("n"), r.get("noise"))
        groups[key].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        vals = []
        for r in rs:
            try:
                v = float(r.get("R2_oof"))
                if math.isfinite(v):
                    vals.append(v)
            except (TypeError, ValueError):
                pass
        gates = [bool(r.get("vieta_gate_pass")) for r in rs if r.get("vieta_gate_pass") is not None]
        elapsed = []
        for r in rs:
            try:
                v = float(r.get("elapsed_sec"))
                if math.isfinite(v):
                    elapsed.append(v)
            except (TypeError, ValueError):
                pass
        out.append({
            "method": key[0], "label": key[1], "primitive_regime": key[2],
            "n": key[3], "noise": key[4], "runs": len(rs),
            "R2_mean": sum(vals) / len(vals) if vals else None,
            "R2_min": min(vals) if vals else None,
            "R2_max": max(vals) if vals else None,
            "gate_pass_rate": sum(gates) / len(gates) if gates else None,
            "elapsed_total_sec": sum(elapsed) if elapsed else None,
        })
    return {"by_method_n_noise": out}


def atomic_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    # Use union of scalar top-level fields. Fold details remain embedded only in
    # the recovered report if they were present in rows (normally they are not).
    preferred = [
        "method", "label", "primitive_regime", "include_sqrt", "seed", "n", "noise", "k_folds",
        "R2_oof", "MSE_oof", "elapsed_sec", "prediction_pass", "radical_in_discovery",
        "radical_product_pass", "vieta_gate_pass", "strict_A0_not_eligible",
        "best_discovered_expression", "top_expressions_json", "error",
    ]
    keys = set()
    for row in rows:
        keys.update(k for k, v in row.items() if not isinstance(v, (dict, list)))
    fieldnames = [k for k in preferred if k in keys] + sorted(keys.difference(preferred))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Truncated audited JSON report")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    source = Path(args.input)
    text = source.read_text(encoding="utf-8", errors="replace")
    protocol = extract_json_value(text, "protocol")
    rows = extract_json_value(text, "rows")
    if not isinstance(rows, list):
        raise TypeError("Recovered rows value is not a list")

    out_dir = Path(args.out_dir)
    report_path = out_dir / "vieta_full_field_report_audited_recovered.json"
    csv_path = out_dir / "vieta_full_field_rows_audited_recovered.csv"
    checkpoint_path = out_dir / "vieta_full_field_checkpoint_audited.json"

    note = {
        "recovered_from_truncated_json": str(source),
        "recovery_time_unix": time.time(),
        "rows_recovered": len(rows),
        "detailed_fold_payload_recovered": False,
        "reason": "Original serialization failed inside detailed PySR expr_info due to a non-JSON SymPy object.",
    }
    report = {
        "protocol": protocol,
        "rows": rows,
        "aggregates": aggregate_rows(rows),
        "detailed": [],
        "recovery": note,
    }
    checkpoint = dict(report)
    checkpoint["checkpoint"] = True

    atomic_dump(report_path, report)
    atomic_dump(checkpoint_path, checkpoint)
    write_csv(csv_path, rows)

    methods = defaultdict(int)
    for row in rows:
        methods[str(row.get("label") or row.get("method"))] += 1
    print(f"Recovered {len(rows)} row(s): {dict(methods)}")
    print(f"WROTE {report_path}")
    print(f"WROTE {csv_path}")
    print(f"WROTE {checkpoint_path}")
    print("Use the v2 runner with --resume and the same --out-dir to compute only missing cases.")


if __name__ == "__main__":
    main()
