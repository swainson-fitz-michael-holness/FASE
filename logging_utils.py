# logging_utils.py
def log_ogset_summary(res):
    print(" --- OG-SET (GLS, gauge-normalized) ---")
    line = "  GLS line: y ≈ {:.6g}".format(res.intercept)
    for op in res.ops:
        line += " {:+.6g}·[{}]".format(op.coeff, op.name)
    print(line)
    print("  Operators:")
    for r, op in enumerate(res.ops, 1):
        print(f"   [{r:02d}] {op.name:>30s}  a={op.coeff:+.6g}  SE={op.stderr:.3g}  ΔBIC(bits)={op.delta_bic_bits:+.3f}  freq={op.freq:>4.0%}")
    wi = res.W_info
    print(f"  Gauge/Σ checks: cond(W)={wi.get('cond', float('nan')):.3g}, "
          f"min_eig={wi.get('min_eig', float('nan')):.3g}, "
          f"max_eig={wi.get('max_eig', float('nan')):.3g}, "
          f"cond(XᵀWX)={wi.get('cond_XtWX', float('nan')):.3g}")

def log_ogset_diagnostics(res):
    if not res.ops:
        print("  (no OG-SET operators selected)")
        return
    print(f"  OG-only R² (train)={res.og_r2_train:.4f}" + (f" | (val)={res.og_r2_val:.4f}" if res.og_r2_val is not None else ""))
    print("  Path: k  bits    r2_train")
    last_k = len(res.ops)
    for k, (b, r2) in enumerate(zip(res.bits_path, res.r2_train_path)):
        star = "*" if k == last_k else " "
        print(f"   {k:>2d}{star} {b:>8.2f}  {r2:>8.4f}")
