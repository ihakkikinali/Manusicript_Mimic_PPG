"""Step 3 — Train the XGBoost feature baseline (PPG-only by default).

Early stopping uses an inner validation subset drawn patient-disjoint from the
TRAIN split. The calibration and test splits are NOT used for early stopping.
With --ecg-assisted, the ECG-augmented feature file is used (Section 5.8).
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from common import load_config, inner_train_val_split

DROP = {"subject_id", "split", "clinical_group", "label", "sqi_score"}


def main(cfg, ecg_assisted):
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    suffix = "_ecg" if ecg_assisted else ""
    fd = pd.read_parquet(out / f"features{suffix}.parquet")
    feat_cols = [c for c in fd.columns if c not in DROP]
    le = LabelEncoder(); fd["label"] = le.fit_transform(fd.clinical_group)

    tr_full = fd[fd.split == "train"]
    ca = fd[fd.split == "calibration"]
    te = fd[fd.split == "test"]

    # patient-disjoint inner train/val for early stopping (never cal/test)
    tr_in, va_in = inner_train_val_split(
        tr_full, val_frac=cfg.get("inner_val_frac", 0.15), seed=cfg["seed"])

    Xtr, ytr = tr_in[feat_cols].values, tr_in.label.values
    Xva, yva = va_in[feat_cols].values, va_in.label.values
    cnt = np.bincount(ytr, minlength=len(le.classes_))
    sw = np.array([len(ytr) / (len(le.classes_) * max(cnt[y], 1)) for y in ytr])

    m = XGBClassifier(n_estimators=cfg["xgb"]["n_estimators"], max_depth=cfg["xgb"]["max_depth"],
                      learning_rate=cfg["xgb"]["learning_rate"], subsample=cfg["xgb"]["subsample"],
                      colsample_bytree=cfg["xgb"]["colsample_bytree"], eval_metric="mlogloss",
                      early_stopping_rounds=cfg["xgb"]["early_stopping_rounds"],
                      random_state=cfg["seed"], n_jobs=-1)
    # early stopping on the INNER VALIDATION subset only
    m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=50)

    np.savez(out / f"xgb_probs{suffix}.npz",
             prob_cal=m.predict_proba(ca[feat_cols].values), y_cal=ca.label.values,
             prob_te=m.predict_proba(te[feat_cols].values),  y_te=te.label.values,
             classes=le.classes_, sqi_cal=ca.sqi_score.values, sqi_te=te.sqi_score.values,
             sid_te=te.subject_id.values)
    mode = "PPG+ECG" if ecg_assisted else "PPG-only"
    print(f"\n[XGBoost {mode}] final test-set report:")
    print(classification_report(te.label.values, m.predict(te[feat_cols].values),
                                target_names=le.classes_, zero_division=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ecg-assisted", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    main(cfg, a.ecg_assisted or cfg.get("ecg_assisted", False))
