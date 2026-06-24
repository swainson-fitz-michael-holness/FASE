#!/usr/bin/env python3
"""Single-fold mechanism smoke test; the shell runner covers cross-fold orchestration."""
from __future__ import annotations
import json
from sklearn.model_selection import KFold
import fase_v24_pmlb_bridge_529_pollen as M


def main():
    X,y,names,_=M.load_dataset('529_pollen',None,None)
    cfg=M.matrix_config('smoke')
    X,y,_=M._subsample(X,y,cfg.subsample,42)
    tr,te=next(KFold(n_splits=2,shuffle=True,random_state=42).split(X))
    out=M.run_fold(X,y,names,tr,te,42,1,cfg,False,1000,30,180)
    assert out['predictive']['v23_1_recommended']['R2']>0.5
    assert out['bridge']['n_promoted']>=1
    assert out['controls']['pair_shuffle']['n_promoted']==0
    assert out['controls']['target_shuffle']['n_promoted']==0
    promoted=[m for m in out['bridge']['modes'] if m.get('promoted')]
    assert all(all(m['promotion_checks'].values()) for m in promoted)
    print('V24-PMLB bridge single-fold smoke: ALL CHECKS PASS')
    print(json.dumps({
        'recommended_R2':out['predictive']['v23_1_recommended']['R2'],
        'recommended_branch':out['predictive']['v23_1_recommended']['branch'],
        'promoted_modes':[m['mode'] for m in promoted],
        'false_promotions':0,
    },indent=2))
if __name__=='__main__': main()
