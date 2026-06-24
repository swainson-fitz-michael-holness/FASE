import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import FASE_v22 as v22
from FASE_v21 import make_synthetic


class TestV22Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v22.apply_bench_config(quick=True)

    def test_nested_kfold_runs_and_reports(self):
        X, y, _ = make_synthetic(seed=11, n=180, d=5, gls_noise=False)
        res = v22.run_fase_kfold_nested(X, y, Sigma=None, K=3, seed=11)
        self.assertEqual(res["protocol"], "v22_nested_kfold")
        self.assertTrue(np.isfinite(res["R2_oof"]))
        self.assertEqual(len(res["folds"]), 3)
        for f in res["folds"]:
            # Selection data must be strictly smaller than the outer-training
            # fold: this is the structural signature of the nested protocol.
            self.assertLess(f["n_inner_select_train"], f["n_outer_train"])
            self.assertTrue(np.isfinite(f["R2"]))

    def test_oof_coverage_complete(self):
        X, y, _ = make_synthetic(seed=3, n=150, d=4, gls_noise=False)
        res = v22.run_fase_kfold_nested(X, y, Sigma=None, K=3, seed=3)
        self.assertTrue(np.all(np.isfinite(res["oof_pred"])))

    def test_results_are_json_serializable(self):
        X, y, _ = make_synthetic(seed=5, n=150, d=4, gls_noise=False)
        res = v22.run_fase_kfold_nested(X, y, Sigma=None, K=2, seed=5)
        res.pop("oof_pred")
        json.dumps(v22._json_safe(res))  # must not raise

    def test_fold_hash_is_deterministic(self):
        from FASE_v21 import kfold_indices
        f1 = kfold_indices(100, 5, seed=42)
        f2 = kfold_indices(100, 5, seed=42)
        self.assertEqual(v22.fold_index_hash(f1), v22.fold_index_hash(f2))
        f3 = kfold_indices(100, 5, seed=43)
        self.assertNotEqual(v22.fold_index_hash(f1), v22.fold_index_hash(f3))


if __name__ == "__main__":
    unittest.main()
