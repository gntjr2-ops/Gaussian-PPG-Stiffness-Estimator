# arterial_stiffness/pipeline.py
import numpy as np
from preprocess import preprocess_ppg, preprocess_ecg, normalize_01
from segmentation import detect_ppg_feet, detect_ppg_peaks, cut_beats_by_feet
from decomposition import fit_two_gaussians
from metrics import ai_from_amplitudes, rt_from_mus, si_from_height, beta_index_proxy, ptt_from_rpeaks_and_feet, aggregate_robust
from quality import sqi_from_residuals, reject_bad_beats

class ArterialStiffnessPipeline:
    def __init__(self, fs_ppg=128, fs_ecg=None, win_sec=12.0, pad_left=0.10, pad_right=0.45, subject_height_m=None):
        self.fs_ppg = fs_ppg
        self.fs_ecg = fs_ecg
        self.win_sec = win_sec
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.height_m = subject_height_m

    def process_window(self, ppg_raw, ecg_raw=None, r_peaks=None):
        # 1) 전처리
        ppg = preprocess_ppg(ppg_raw, self.fs_ppg)
        ppg = normalize_01(ppg)

        # (선택) ECG 전처리
        if ecg_raw is not None and self.fs_ecg is not None:
            ecg = preprocess_ecg(ecg_raw, self.fs_ecg)
        else:
            ecg = None

        # 2) 세그멘테이션
        feet = detect_ppg_feet(ppg, self.fs_ppg)
        if feet is None or len(feet) < 2:
            return {"AIx": None, "RT": None, "SI": None, "Beta": None, "PTT": None, "SQI": 0.0, "N_beats": 0}

        beats, idxs = cut_beats_by_feet(ppg, feet, self.fs_ppg, self.pad_left, self.pad_right)

        # 3) 박동별 2-가우시안 분해
        fits = []
        aix_list, rt_list, beta_list, rmse = [], [], [], []
        for b in beats:
            if len(b) < int(0.2 * self.fs_ppg):  # 너무 짧으면 skip
                fits.append(None); continue
            f = fit_two_gaussians(b, self.fs_ppg)
            fits.append(f)
            rmse.append(f["rmse"])
            A1, A2 = f["A1"], f["A2"]
            mu1, mu2 = f["mu1"], f["mu2"]
            s1, s2 = f["s1"], f["s2"]
            aix_list.append(ai_from_amplitudes(A1, A2))
            rt_list.append(rt_from_mus(mu1, mu2))
            beta_list.append(beta_index_proxy(A1, A2, s1, s2))

        # 4) 품질 관리
        sqi = sqi_from_residuals(rmse)
        good = reject_bad_beats(fits, rmse_max=0.25)
        if len(good) < max(2, int(0.25 * len(fits))):
            # 너무 나쁘면 None 반환
            return {"AIx": None, "RT": None, "SI": None, "Beta": None, "PTT": None, "SQI": sqi, "N_beats": len(fits)}

        # 5) 윈도우 집계 (중앙값)
        AIx = aggregate_robust(aix_list)
        RT  = aggregate_robust(rt_list)
        Beta = aggregate_robust(beta_list)
        SI = si_from_height(self.height_m, RT)

        # 6) (선택) PTT
        if r_peaks is not None and len(r_peaks)>0:
            # PPG foot은 feet; ECG R-peak 샘플은 r_peaks
            PTT = ptt_from_rpeaks_and_feet(r_peaks, feet, self.fs_ecg, self.fs_ppg) if self.fs_ecg else None
        else:
            PTT = None

        return {
            "AIx": AIx, "RT": RT, "SI": SI, "Beta": Beta, "PTT": PTT,
            "SQI": sqi, "N_beats": len(fits),
            "Details": {
                "fits": fits[:10],  # 미리보기용 일부만
                "rmse_median": float(np.median(rmse)) if len(rmse)>0 else None
            }
        }
