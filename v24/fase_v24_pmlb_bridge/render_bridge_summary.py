#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


def fmt(x, digits=4):
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--out", default=None)
    args=ap.parse_args()
    p=Path(args.report)
    d=json.loads(p.read_text())
    out=Path(args.out) if args.out else p.with_name("V24_PMLB_BRIDGE_RESULT.md")
    lines=[]
    lines += ["# V24-PMLB Bridge Result — 529_pollen", "", f"**Horizon classification:** `{d['horizon_classification']}`", ""]
    lines += ["## Predictive layer", "", "| Model | Folds | Mean OOF R² | Median | Minimum |", "|---|---:|---:|---:|---:|"]
    for name,row in d.get('predictive_summary',{}).items():
        lines.append(f"| {name} | {row['folds']} | {fmt(row['R2_mean'])} | {fmt(row['R2_median'])} | {fmt(row['R2_min'])} |")
    b=d['bridge_summary']; c=d['destroyer_controls']
    lines += ["", "## Operator bridge", "", f"- Real promotions: **{b['real_promotions']} / {b['real_eligible_mode_trials']}**", f"- Median transformation coverage: **{fmt(b['transformation_coverage_median'])}**", f"- Stable modes: **{len(b['stable_modes'])}**", ""]
    if b['stable_modes']:
        lines += ["| Mode | Promotion rate | Median empirical R² | Median model-consistency R² | Identity improvement | Pooled improvement |", "|---|---:|---:|---:|---:|---:|"]
        for r in b['stable_modes']:
            lines.append(f"| {r['mode']} | {fmt(r['promotion_rate'])} | {fmt(r['empirical_R2_median'])} | {fmt(r['model_counterfactual_R2_median'])} | {fmt(r['identity_improvement_median'])} | {fmt(r['pooled_improvement_median'])} |")
    lines += ["", "## Destroyer controls", "", f"- Pair-shuffle false promotions: **{c['pair_shuffle']['false_promotions']} / {c['pair_shuffle']['eligible_trials']}**", f"- Target-shuffle false promotions: **{c['target_shuffle']['false_promotions']} / {c['target_shuffle']['eligible_trials']}**", f"- Combined false-promotion rate: **{fmt(c['combined_false_promotion_rate'])}**", ""]
    lines += ["## Conversion statement", "", d['interpretation'], "", "This result is a trajectory marker. The dataset is synthetic and iid; promoted operators are support-bounded empirical rewrite schemas, not causal laws.", ""]
    out.write_text("\n".join(lines), encoding='utf-8')
    print(out)
if __name__=='__main__': main()
