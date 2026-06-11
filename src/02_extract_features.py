"""Step 2 — Extract handcrafted features for the XGBoost baseline.

By default this extracts the 14-dimensional PPG-only feature vector used in the
main experiments. ECG-derived features are added ONLY when --ecg-assisted is set
(paper, Section 5.8 sensitivity analysis).
"""
import argparse
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from scipy import signal as sps
from common import load_config, build_signal_path


def bandpass(x, lo, hi, fs):
    b, a = sps.butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sps.filtfilt(b, a, x)


def extract_ppg(sig, fs):
    """14-dim PPG-only feature vector."""
    f = {}
    if "PLETH" not in sig:
        return f
    ppg = sig["PLETH"]; ppg = ppg[~np.isnan(ppg)]
    if len(ppg) <= 100:
        return f
    pf = bandpass(ppg, 0.5, 8.0, fs)
    f.update(ppg_mean=np.mean(pf), ppg_std=np.std(pf), ppg_range=np.ptp(pf),
             ppg_skew=float(pd.Series(pf).skew()), ppg_kurt=float(pd.Series(pf).kurt()))
    fr, ps = sps.welch(pf, fs=fs, nperseg=min(256, len(pf)))
    for name, lo, hi in [("hr", 0.67, 3.0), ("lf", 0.04, 0.15), ("hf", 0.15, 0.4)]:
        m = (fr >= lo) & (fr <= hi)
        f[f"ppg_{name}_power"] = np.trapz(ps[m], fr[m]) if m.any() else 0.0
    f["ppg_lf_hf"] = f["ppg_lf_power"] / (f["ppg_hf_power"] + 1e-9)
    pk, _ = sps.find_peaks(pf, distance=int(fs * 0.4), height=np.mean(pf))
    f["ppg_peak_count"] = len(pk); f["ppg_hr_est"] = len(pk) * 2.0
    if len(pk) > 1:
        rri = np.diff(pk) / fs
        f.update(ppg_rri_mean=np.mean(rri), ppg_rri_std=np.std(rri),
                 ppg_rri_cv=np.std(rri) / (np.mean(rri) + 1e-9))
    return f


def extract_ecg(sig, fs):
    """Additional ECG-derived features (sensitivity analysis only)."""
    f = {}
    if "II" not in sig:
        return f
    ecg = sig["II"]; ecg = ecg[~np.isnan(ecg)]
    if len(ecg) <= 100:
        return f
    ef = bandpass(ecg, 0.5, 40.0, fs)
    f.update(ecg_mean=np.mean(ef), ecg_std=np.std(ef), ecg_range=np.ptp(ef))
    pk, _ = sps.find_peaks(ef, distance=int(fs * 0.3), height=np.percentile(ef, 75))
    f["ecg_peak_count"] = len(pk); f["ecg_hr_est"] = len(pk) * 2.0
    if len(pk) > 1:
        rri = np.diff(pk) / fs
        f.update(ecg_rri_mean=np.mean(rri), ecg_rri_std=np.std(rri),
                 ecg_rri_cv=np.std(rri) / (np.mean(rri) + 1e-9),
                 ecg_rri_iqr=float(np.percentile(rri, 75) - np.percentile(rri, 25)))
    return f


def main(cfg, ecg_assisted):
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(out / "cohort.parquet")
    fs = cfg["fs"]
    suffix = "_ecg" if ecg_assisted else ""
    rows, skip = [], 0
    for i, (_, r) in enumerate(df.iterrows()):
        if i % 2000 == 0:
            print(f"{i}/{len(df)} skipped={skip}", end="\r")
        try:
            rec = wfdb.rdrecord(str(build_signal_path(r, cfg["data_root"])))
            sig = {n: rec.p_signal[:, j].astype(float) for j, n in enumerate(rec.sig_name)}
        except Exception:
            skip += 1; continue
        feat = extract_ppg(sig, fs)
        if not feat:                      # PPG features are mandatory
            skip += 1; continue
        if ecg_assisted:
            feat.update(extract_ecg(sig, fs))
        feat.update(subject_id=r.subject_id, split=r.split,
                    clinical_group=r.clinical_group, sqi_score=float(r.sqi_score))
        rows.append(feat)
    fd = pd.DataFrame(rows).fillna(-1)
    fd.to_parquet(out / f"features{suffix}.parquet")
    n_feat = len([c for c in fd.columns
                  if c not in {"subject_id", "split", "clinical_group", "sqi_score"}])
    mode = "PPG+ECG" if ecg_assisted else "PPG-only"
    print(f"\nExtracted {len(fd):,} rows, {n_feat} {mode} features (skipped {skip}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ecg-assisted", action="store_true",
                    help="Add ECG-derived features (sensitivity analysis, Section 5.8)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    main(cfg, a.ecg_assisted or cfg.get("ecg_assisted", False))
