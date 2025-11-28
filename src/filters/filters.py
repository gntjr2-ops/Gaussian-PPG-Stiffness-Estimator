# arterial_stiffness/filters.py
import numpy as np
from scipy.signal import butter, filtfilt

def _safe_filtfilt(b, a, x):
    padlen = 3 * max(len(a), len(b))
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim == 1 and x.size <= padlen:
        return x
    if x.ndim > 1:
        return np.vstack([_safe_filtfilt(b, a, x[:, i]) for i in range(x.shape[1])]).T
    return filtfilt(b, a, x)

def bandpass(x, fs, lo, hi, order=3):
    lo = max(lo, 1e-4)
    hi = min(hi, fs/2 - 1e-4)
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return _safe_filtfilt(b, a, x)

def highpass(x, fs, fc, order=3):
    fc = max(fc, 1e-4)
    b, a = butter(order, fc/(fs/2), btype="high")
    return _safe_filtfilt(b, a, x)

def lowpass(x, fs, fc, order=3):
    fc = min(fc, fs/2 - 1e-4)
    b, a = butter(order, fc/(fs/2), btype="low")
    return _safe_filtfilt(b, a, x)

def diff1(x):
    x = np.asarray(x)
    d = np.diff(x, prepend=x[:1])
    return d

def smooth_movavg(x, k):
    x = np.asarray(x, float)
    k = max(1, int(k))
    if k == 1: return x
    w = np.ones(k) / k
    y = np.convolve(x, w, mode="same")
    return y

def zscore(x, eps=1e-9):
    x = np.asarray(x, float)
    return (x - np.mean(x)) / (np.std(x) + eps)
