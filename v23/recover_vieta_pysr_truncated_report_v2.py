#!/usr/bin/env python3
"""Tolerantly recover PySR summary rows from a JSON report truncated mid-serialization.

This handles two failure shapes:
1. The top-level ``rows`` array contains complete row objects and then truncates.
2. Serialization fails inside the current row's nested ``folds`` payload (for
   example on a SymPy ``Mul`` in ``expr_info.sympy_format``). In that case the
   utility salvages the row's already-complete summary fields by discarding the
   incomplete ``folds`` value.

The recovered report/checkpoint can be used with
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
from typing import Any, Dict, List, Optional, Sequence, Tuple


def skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def find_key_value_start(text: str, key: str, start_at: int = 0) -> int:
    marker = json.dumps(key)
    pos = text.find(marker, start_at)
    if pos < 0:
        raise ValueError(f"Could not locate key {key!r}")
    colon = text.find(":", pos + len(marker))
    if colon < 0:
        raise ValueError(f"Could not locate ':' after key {key!r}")
    return skip_ws(text, colon + 1)


def extract_complete_value(text: str, key: str) -> Any:
    start = find_key_value_start(text, key)
    value, _end = json.JSONDecoder().raw_decode(text, start)
    return value


def find_top_level_member_start(obj_text: str, member: str) -> Optional[int]:
    """Return index of a depth-1 object member name, ignoring strings/nesting."""
    target = json.dumps(member)
    depth_obj = 0
    depth_arr = 0
    in_string = False
    escape = False
    i = 0
    while i < len(obj_text):
        ch = obj_text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            if depth_obj == 1 and depth_arr == 0 and obj_text.startswith(target, i):
                j = skip_ws(obj_text, i + len(target))
                if j < len(obj_text) and obj_text[j] == ":":
                    return i
            in_string = True
            i += 1
            continue
        if ch == "{":
            depth_obj += 1
        elif ch == "}":
            depth_obj -= 1
        elif ch == "[":
            depth_arr += 1
        elif ch == "]":
            depth_arr -= 1
        i += 1
    return None


def salvage_partial_row(fragment: str) -> Optional[Dict[str, Any]]:
    """Recover top-level summary fields from a row truncated inside ``folds``."""
    folds_pos = find_top_level_member_start(fragment, "folds")
    if folds_pos is None:
        return None

    # Remove the comma that introduced the folds member, then close the object.
    cut = folds_pos
    while cut > 0 and fragment[cut - 1].isspace():
        cut -= 1
    if cut > 0 and fragment[cut - 1] == ",":
        cut -= 1
    candidate = fragment[:cut].rstrip() + "}"
    try:
        row = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(row, dict):
        return None
    # Require enough identity/metric fields to make resume safe.
    required_identity = {"method", "label", "seed", "n", "noise"}
    if not required_identity.issubset(row):
        return None
    row.pop("folds", None)
    row["recovered_from_partial_row"] = True
    row["recovery_note"] = "Nested folds payload was truncated; summary fields were salvaged."
    return row


def recover_rows(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    start = find_key_value_start(text, "rows")
    if start >= len(text) or text[start] != "[":
        raise ValueError("The 'rows' value does not begin with '['")

    pos = skip_ws(text, start + 1)
    rows: List[Dict[str, Any]] = []
    partial_row_recovered = False
    decoder = json.JSONDecoder()

    while pos < len(text):
        pos = skip_ws(text, pos)
        if pos >= len(text):
            break
        if text[pos] == "]":
            return rows, {
                "rows_array_complete": True,
                "partial_row_recovered": partial_row_recovered,
                "stop_position": pos,
            }
        if text[pos] == ",":
            pos = skip_ws(text, pos + 1)
            continue
        try:
            value, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            partial = salvage_partial_row(text[pos:])
            if partial is not None:
                rows.append(partial)
                partial_row_recovered = True
            return rows, {
                "rows_array_complete": False,
                "partial_row_recovered": partial_row_recovered,
                "stop_position": pos,
                "decode_error": str(exc),
            }
        if not isinstance(value, dict):
            return rows, {
                "rows_array_complete": False,
                "partial_row_recovered": partial_row_recovered,
                "stop_position": pos,
                "decode_error": f"Expected row object, found {type(value).__name__}",
            }
        rows.append(value)
        pos = end

    return rows, {
        "rows_array_complete": False,
        "partial_row_recovered": partial_row_recovered,
        "stop_position": pos,
        "decode_error": "Reached end of file while reading rows array",
    }


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("method"), row.get("label"), row.get("primitive_regime"),
            row.get("n"), row.get("noise"),
        )
        groups[key].append(row)

    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        r2_values: List[float] = []
        elapsed_values: List[float] = []
        gates: List[bool] = []
        for row in group:
            try:
                value = float(row.get("R2_oof"))
                if math.isfinite(value):
                    r2_values.append(value)
            except (TypeError, ValueError):
                pass
            try:
                value = float(row.get("elapsed_sec"))
                if math.isfinite(value):
                    elapsed_values.append(value)
            except (TypeError, ValueError):
                pass
            if row.get("vieta_gate_pass") is not None:
                gates.append(bool(row.get("vieta_gate_pass")))
        output.append({
            "method": key[0],
            "label": key[1],
            "primitive_regime": key[2],
            "n": key[3],
            "noise": key[4],
            "runs": len(group),
            "R2_mean": sum(r2_values) / len(r2_values) if r2_values else None,
            "R2_min": min(r2_values) if r2_values else None,
            "R2_max": max(r2_values) if r2_values else None,
            "gate_pass_rate": sum(gates) / len(gates) if gates else None,
            "elapsed_total_sec": sum(elapsed_values) if elapsed_values else None,
        })
    return {"by_method_n_noise": output}


def atomic_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    scalar_keys = set()
    for row in rows:
        for key, value in row.items():
            if not isinstance(value, (dict, list)):
                scalar_keys.add(key)
    preferred = [
        "method", "label", "primitive_regime", "include_sqrt", "seed", "n", "noise",
        "k_folds", "R2_oof", "MSE_oof", "elapsed_sec", "prediction_pass",
        "radical_in_discovery", "radical_product_pass", "vieta_gate_pass",
        "strict_A0_not_eligible", "best_discovered_expression", "top_expressions_json",
        "recovered_from_partial_row", "recovery_note", "error",
    ]
    fieldnames = [key for key in preferred if key in scalar_keys]
    fieldnames.extend(sorted(scalar_keys.difference(fieldnames)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Truncated audited JSON report")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    text = source.read_text(encoding="utf-8", errors="replace")

    try:
        protocol = extract_complete_value(text, "protocol")
    except Exception as exc:
        protocol = {
            "protocol": "Recovered truncated Viète PySR report",
            "protocol_recovery_error": repr(exc),
        }

    rows, diagnostics = recover_rows(text)
    if not rows:
        raise RuntimeError(
            "No complete or salvageable PySR summary rows were found. The file failed "
            "inside the first row before enough summary fields were written. Re-run with "
            "the v2 runner, which checkpoints after every completed case."
        )

    out_dir = Path(args.out_dir)
    report_path = out_dir / "vieta_full_field_report_audited_recovered.json"
    csv_path = out_dir / "vieta_full_field_rows_audited_recovered.csv"
    checkpoint_path = out_dir / "vieta_full_field_checkpoint_audited.json"

    recovery = {
        "recovered_from": str(source),
        "recovery_time_unix": time.time(),
        "rows_recovered": len(rows),
        "detailed_fold_payload_recovered": False,
        "diagnostics": diagnostics,
        "warning": (
            "Only data physically written before the serialization exception can be recovered. "
            "Terminal output may show later completed cases that were never persisted."
        ),
    }
    report = {
        "protocol": protocol,
        "rows": rows,
        "aggregates": aggregate_rows(rows),
        "detailed": [],
        "recovery": recovery,
    }
    checkpoint = dict(report)
    checkpoint["checkpoint"] = True

    atomic_dump(report_path, report)
    atomic_dump(checkpoint_path, checkpoint)
    write_csv(csv_path, rows)

    counts = defaultdict(int)
    for row in rows:
        counts[str(row.get("label") or row.get("method"))] += 1

    print(f"Recovered {len(rows)} row(s): {dict(counts)}")
    print(f"Rows array complete: {diagnostics.get('rows_array_complete')}")
    print(f"Partial current row salvaged: {diagnostics.get('partial_row_recovered')}")
    print(f"WROTE {report_path}")
    print(f"WROTE {csv_path}")
    print(f"WROTE {checkpoint_path}")
    print("Run the v2 field runner with --resume and the same --out-dir.")


if __name__ == "__main__":
    main()
