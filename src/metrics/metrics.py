# arterial_stiffness/metrics.py
import numpy as np

def ai_from_amplitudes(A1, A2, eps=1e-9):
    # 진폭 기반 AIx (면적 기반은 ∫g2/∫g1로 확장 가능)
    return float(A2 / (A1 + eps))

def rt_from_mus(mu1, mu2):
    # 반사 시간 (초)
    return float(max(0.0, mu2 - mu1))

def si_from_height(height_m, rt_s, eps=1e-9):
    # Stiffness Index: 높이/RT (m/s)
    if height_m is None:
        return None
    return float(height_m / (rt_s + eps)) if rt_s is not None and rt_s > 0 else None

def beta_index_proxy(A1, A2, s1, s2):
    # 간단한 상대 지표 (파형 기울기/진폭비로 근사) — 절대치는 교정 필요
    if A1 is None or A2 is None or s1 is None or s2 is None:
        return None
    return float((A2/(A1+1e-9)) * (s1/(s2+1e-9)))

def ptt_from_rpeaks_and_feet(r_peaks, ppg_feet, fs_ecg, fs_ppg):
    """
    ECG R-peak (샘플, fs_ecg), PPG foot (샘플, fs_ppg) → 평균 지연(초)
    간단 매칭: 인덱스 순차 매칭
    """
    if r_peaks is None or ppg_feet is None or len(r_peaks)==0 or len(ppg_feet)==0:
        return None
    i = j = 0
    delays = []
    while i < len(r_peaks) and j < len(ppg_feet):
        t_r = r_peaks[i] / fs_ecg
        t_f = ppg_feet[j] / fs_ppg
        if t_f <= t_r:
            j += 1
            continue
        delays.append(t_f - t_r)
        i += 1; j += 1
    return float(np.median(delays)) if delays else None

def aggregate_robust(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if len(vals) == 0: return None
    return float(np.median(vals))
