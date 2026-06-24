#!/usr/bin/env python3
"""
Decisive test: 1-D (shielded, one-at-a-time) refinement vs JOINT refinement on the
exact failure mode FASE-A0r exhibits on the Viete killing field.

True law:        y = p * sqrt(2 + a) / 2 + noise
Hidden coord:    sqrt(x0 + 2) * (x1 - 0)   [up to the absorbable 1/2 scale]
Wrong-shift FASE pick (off the fixed grid): sqrt(x0 + 0.5) * (x1 - 0.5)

Claim under test:
  (1) 1-D shielded refinement of the radical constant alone does NOT recover c=2,
      because the spurious x1 shift (-0.5) is coupled to the radical shift.
  (2) JOINT refinement over BOTH constants against held-out inner-val R2 recovers
      exactly (c=2, s=0) and reaches PySR-grade val R2 (~0.999998).
"""
import numpy as np
import fase_v23_g1a_coordinate_gate as base
import fase_v23_g1a_constant_refine as R

rng = np.random.default_rng(42)
n = 160
a = rng.uniform(-1.5, 8.0, n); p = rng.uniform(0.2, 2.0, n)
X = np.column_stack([a, p])
y = p * np.sqrt(2.0 + a) / 2.0 + rng.normal(0, 1e-3, n)
tr = np.arange(0, 120); va = np.arange(120, 160)
Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

context = [base.X(0), base.X(1)]
rcfg = R.RefineConfig()

def show(coord, label):
    v_full = R._val_r2_of(context + [coord], Xtr, ytr, Xva, yva, rcfg.ridge_alpha)
    print(f"   {label:<34} consts={[round(c,4) for c in R.enumerate_consts(coord)]}  val_R2={v_full:.6f}")

print("="*78)
print("STARTING COORDINATE (wrong-shift, correct form):  sqrt(x0+0.5)*(x1-0.5)")
print("="*78)
coord0 = base.Mul(base.Sqrt(base.Add(base.X(0), base.C(0.5))),
                  base.Sub(base.X(1), base.C(0.5)))
show(coord0, "start")

# ---- (1) 1-D shielded refinement (the literally-requested step) -------------------
print("\n--- (1) 1-D shielded one-at-a-time refinement ---")
shield = R.shielded_const_indices(coord0)
print(f"   shielded (refinable) indices: {shield}   (x1-shift is absorbable, excluded)")
Zc, _ = base.fit_design(context, Xtr)
resid_y = R._residualize(np.asarray(ytr, float), Zc)
ci = shield[0]
def fwl_obj(c, _ci=ci, _coord=coord0):
    trial = R.with_const(_coord, _ci, c)
    if not R.sqrt_args_valid(trial, Xtr, rcfg.max_unsafe_frac):
        return R.NEG_INF
    v = base.sanitize(base.eval_expr(trial, Xtr))
    if v.std() < 1e-10 or not np.all(np.isfinite(v)):
        return R.NEG_INF
    return R.fwl_contribution(Zc, resid_y, v)
c_1d = R.refine_one_constant(coord0, ci, fwl_obj, R.enumerate_consts(coord0)[ci], rcfg)
coord_1d = R.with_const(coord0, ci, c_1d)
print(f"   radical c recovered by 1-D  = {c_1d:.5f}   (true = 2.0)")
show(coord_1d, "after 1-D")

# ---- (2) JOINT refinement over ALL constants on held-out val ----------------------
print("\n--- (2) JOINT refinement (all constants, held-out val R2 objective) ---")
coord_j, vb, vj = R.refine_coordinate_joint(coord0, context, Xtr, ytr, Xva, yva, rcfg)
cj = R.enumerate_consts(coord_j)
print(f"   constants recovered by JOINT = {[round(c,5) for c in cj]}   (true = [2.0, 0.0])")
print(f"   val_R2 before = {vb:.6f}   ->   after = {vj:.6f}")
show(coord_j, "after JOINT")

# ---- reference: PySR-grade clean coordinate and the true hidden coordinate ---------
print("\n--- reference points ---")
clean = base.Mul(base.X(1), base.Sqrt(base.Add(base.X(0), base.C(2.0))))   # x1*sqrt(x0+2)
show(clean, "clean x1*sqrt(x0+2)")

# ---- optional MDL integer-snap (justified: val-preserving) -------------------------
cj_round = [round(c) for c in cj]
snapped = base.Mul(base.Sqrt(base.Add(base.X(0), base.C(float(cj_round[0])))),
                   base.Sub(base.X(1), base.C(float(cj_round[1]))))
v_snap = R._val_r2_of(context + [snapped], Xtr, ytr, Xva, yva, rcfg.ridge_alpha)
print(f"\n   MDL integer-snap: {[round(c,4) for c in cj]} -> {cj_round}"
      f"  val {vj:.7f} -> {v_snap:.7f}  (delta {v_snap - vj:+.2e})")

# ---- assertions (tolerance = data-supported identifiability band, ~+/-0.05) ---------
print("\n" + "="*78)
BAND = 0.05  # val-R2 is flat to <1e-6 across c in [1.95,2.05]; this is the identifiability floor
ok_joint_c = abs(cj[0] - 2.0) < BAND
ok_joint_s = abs(cj[1] - 0.0) < BAND
oned_missed = abs(c_1d - 2.0) > 0.1
ok_snap = (cj_round == [2, 0]) and (abs(v_snap - vj) < 1e-5)
print(f"[1-D MISSES c=2 (coupling)]            {oned_missed}   (got {c_1d:.4f})")
print(f"[JOINT recovers c=2 within band]       {ok_joint_c}   (got {cj[0]:.4f}, band +/-{BAND})")
print(f"[JOINT recovers s=0 within band]       {ok_joint_s}   (got {cj[1]:.4f}, band +/-{BAND})")
print(f"[JOINT reaches PySR-grade val_R2]      {vj >= 0.99999}   (got {vj:.6f})")
print(f"[MDL-snap recovers EXACT (2,0), free]  {ok_snap}   (snap {cj_round}, val delta {v_snap-vj:+.1e})")
assert oned_missed and ok_joint_c and ok_joint_s and (vj >= 0.99999) and ok_snap, "validation failed"
print("ALL CHECKS PASS")
print("="*78)
