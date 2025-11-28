# arterial_stiffness/decomposition.py
import numpy as np
from scipy.optimize import least_squares

def _two_gauss(t, A1, mu1, s1, A2, mu2, s2):
    g1 = A1 * np.exp(-0.5 * ((t - mu1) / (s1 + 1e-9))**2)
    g2 = A2 * np.exp(-0.5 * ((t - mu2) / (s2 + 1e-9))**2)
    return g1 + g2, g1, g2

def _param_map(theta, t_max):
    """
    unconstrained theta -> constrained params
    theta = [a1, m1_raw, s1_raw, a2, dmu_raw, s2_raw]
    제약: A1,A2>=0, s1,s2>0, mu1 in [0, t_max], mu2=mu1+softplus(dmu_raw)
    """
    a1, m1r, s1r, a2, dmu_r, s2r = theta
    A1 = np.exp(a1)                      # >=0
    A2 = np.exp(a2)
    mu1 = t_max / (1.0 + np.exp(-m1r))   # (0, t_max)
    dmu = np.log1p(np.exp(dmu_r)) + 1e-3 # >0
    mu2 = np.clip(mu1 + dmu, 0, t_max)
    s1 = np.log1p(np.exp(s1r)) + 1e-3    # >0
    s2 = np.log1p(np.exp(s2r)) + 1e-3
    return A1, mu1, s1, A2, mu2, s2

def fit_two_gaussians(y, fs):
    """
    단일 beat 파형 y (길이 T samples) 를 2-가우시안으로 근사하여
    incident/reflected의 A, mu, sigma 추정
    """
    T = len(y)
    t = np.arange(T) / fs
    t_max = t[-1] if T>1 else 1.0
    y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-9)  # 0~1 정규화
    # 초기값: peak 2개 가정 (rough)
    mu1_init = 0.15 * t_max
    mu2_init = 0.30 * t_max
    s1_init = 0.05 * t_max
    s2_init = 0.07 * t_max
    A1_init = 1.0
    A2_init = 0.4

    # unconstrained theta 초기화
    def inv_softplus(x): return np.log(np.exp(x)-1.0+1e-9)
    theta0 = np.array([
        np.log(A1_init),                 # a1
        np.log(mu1_init/(t_max-mu1_init)+1e-9),  # m1_raw (sigmoid^-1)
        inv_softplus(s1_init),
        np.log(A2_init),
        inv_softplus(mu2_init-mu1_init),
        inv_softplus(s2_init)
    ], dtype=float)

    def residual(theta):
        A1, mu1, s1, A2, mu2, s2 = _param_map(theta, t_max)
        yhat, _, _ = _two_gauss(t, A1, mu1, s1, A2, mu2, s2)
        return yhat - y

    res = least_squares(residual, theta0, method="trf", max_nfev=200)
    A1, mu1, s1, A2, mu2, s2 = _param_map(res.x, t_max)
    yhat, g1, g2 = _two_gauss(t, A1, mu1, s1, A2, mu2, s2)
    out = {
        "A1": float(A1), "mu1": float(mu1), "s1": float(s1),
        "A2": float(A2), "mu2": float(mu2), "s2": float(s2),
        "t": t, "y": y, "yhat": yhat, "g1": g1, "g2": g2,
        "rmse": float(np.sqrt(np.mean((yhat - y)**2)))
    }
    return out
