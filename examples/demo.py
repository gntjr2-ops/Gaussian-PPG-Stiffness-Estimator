# arterial_stiffness/demo.py
import numpy as np
from pipeline import ArterialStiffnessPipeline

def synth_ppg_with_reflection(fs=128, dur_s=20, hr_bpm=70, rt_s=0.18, aix=0.3, noise=0.02, seed=7):
    """
    간단 합성: 2-가우시안(incident + reflected) 파형의 박동을 이어 붙여 반사파/반사지연을 가진 PPG 생성
    - rt_s: 반사 지연(초) — 짧을수록 동맥 경직 경향
    - aix : 반사 진폭 비(A2/A1) — 클수록 경직 경향
    """
    rng = np.random.default_rng(seed)
    N = fs * dur_s
    t = np.arange(N) / fs
    ppg = np.zeros(N, float)

    rr = int(round(fs * (60.0/hr_bpm)))
    # 기본 파형 파라미터
    s1 = 0.06   # incident 폭(초)
    s2 = 0.08   # reflected 폭(초)
    A1 = 1.0
    A2 = aix

    for k in range(0, N, rr):
        mu1 = k / fs + 0.12   # foot 뒤 120ms 부근 incident
        mu2 = mu1 + rt_s
        idx = np.arange(k, min(N, k + int(0.6*fs)))
        tt = idx / fs
        g1 = A1 * np.exp(-0.5 * ((tt - mu1) / (s1+1e-9))**2)
        g2 = A2 * np.exp(-0.5 * ((tt - mu2) / (s2+1e-9))**2)
        ppg[idx] += g1 + g2

    # 정규화 + 잡음
    ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min() + 1e-9)
    ppg += noise * rng.standard_normal(N)
    return ppg

if __name__ == "__main__":
    fs = 128
    pipe = ArterialStiffnessPipeline(fs_ppg=fs, fs_ecg=None, win_sec=12.0, subject_height_m=1.70)

    # 시나리오 2개: (1) 경직 낮음 (RT 길고 AIx 낮음), (2) 경직 높음 (RT 짧고 AIx 큼)
    scen = [
        ("Low stiffness", 70, 0.22, 0.20),
        ("High stiffness", 70, 0.12, 0.45),
    ]
    for name, hr, rt_s, aix in scen:
        ppg = synth_ppg_with_reflection(fs=fs, dur_s=20, hr_bpm=hr, rt_s=rt_s, aix=aix, noise=0.02, seed=7)
        res = pipe.process_window(ppg)
        print(f"\n=== {name} ===")
        print(f"AIx={res['AIx']:.3f} | RT={res['RT']:.3f}s | SI={res['SI'] if res['SI'] is not None else 'N/A'} | Beta={res['Beta'] if res['Beta'] is not None else 'N/A'} | SQI={res['SQI']:.2f} | N_beats={res['N_beats']}")
