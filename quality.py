# arterial_stiffness/quality.py
import numpy as np

def sqi_from_residuals(rmses, thresh_high=0.12):
    """
    모델 잔차 기반 간단 SQI: 낮을수록 좋음 → 1 - clip(rmse/thresh)
    """
    vals = [r for r in rmses if r is not None and np.isfinite(r)]
    if len(vals) == 0:
        return 0.0
    rm = float(np.median(vals))
    q = 1.0 - min(1.0, rm / thresh_high)
    return max(0.0, q)

def reject_bad_beats(fits, rmse_max=0.25):
    good = []
    for f in fits:
        if f is None: continue
        if f.get("rmse", 1.0) <= rmse_max:
            good.append(f)
    return good
