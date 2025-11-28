# arterial_stiffness/preprocess.py
import numpy as np
from filters import bandpass, highpass, lowpass, smooth_movavg

def preprocess_ppg(ppg, fs):
    # 드리프트 제거 + 대역통과(0.5~8 Hz) + 약한 평활
    x = highpass(ppg, fs, 0.05, order=2)
    x = bandpass(x, fs, 0.5, 8.0, order=3)
    x = smooth_movavg(x, int(0.05*fs) or 1)
    return x

def preprocess_ecg(ecg, fs):
    # QRS 대역 5~30 Hz
    x = bandpass(ecg, fs, 5.0, 30.0, order=3)
    return x

def normalize_01(x, clip_sigma=3.0):
    x = np.asarray(x, float)
    m, s = np.mean(x), np.std(x)
    if s == 0: return np.zeros_like(x)
    x = (x - m) / s
    x = np.clip(x, -clip_sigma, clip_sigma)
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    return x
