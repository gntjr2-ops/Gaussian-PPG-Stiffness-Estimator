# arterial_stiffness/segmentation.py
import numpy as np
from scipy.signal import find_peaks
from filters import diff1, smooth_movavg

def detect_ppg_peaks(ppg, fs):
    # 상향 peak (맥파 최고점)
    height = np.percentile(ppg, 70)
    distance = int(0.3 * fs)  # ≥ 0.3 s
    peaks, _ = find_peaks(ppg, height=height, distance=distance)
    return peaks

def detect_ppg_feet(ppg, fs):
    # foot: 1차 미분 최소점 근처 (상승 시작점)
    d1 = diff1(ppg)
    inv = -smooth_movavg(d1, int(0.04*fs) or 1)
    distance = int(0.3 * fs)
    height = np.percentile(inv, 70)
    feet, _ = find_peaks(inv, height=height, distance=distance)
    # foot이 peak 이후로 잘못 잡힌 경우 peak 이전 가장 가까운 최소점으로 보정 가능 (간단화: 그대로 사용)
    return feet

def cut_beats_by_feet(x, feet, fs, pad_left=0.08, pad_right=0.40):
    """
    foot 기준으로 [foot - pad_left, foot + pad_right] 구간을 beat로 추출
    """
    beats = []
    idxs = []
    N = len(x)
    L = int(pad_left * fs)
    R = int(pad_right * fs)
    for f in feet:
        s = max(0, f - L)
        e = min(N, f + R)
        if e - s >= int(0.2*fs):  # 최소 길이
            beats.append(x[s:e])
            idxs.append((s, e))
    return beats, idxs
