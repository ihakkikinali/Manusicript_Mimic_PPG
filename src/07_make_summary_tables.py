"""Step 7 (optional) — Assemble the paper's summary tables from the per-run CSVs.

Reads the CSVs produced by 05_conformal.py (PPG-only and, if present, the
ECG-assisted run) and assembles the paper-formatted summary tables:
  - base_classifier_performance.csv  (Table 2: accuracy, balanced accuracy,
    macro-F1, critical recall / FN)
  - ecg_sensitivity.csv              (Table 9: PPG-only vs PPG+ECG, with test
    accuracy and global / class-conditional critical coverage)

Both are assembled only from CSVs that exist; if the inputs for a table are not
present, the shipped copy of that table is left untouched.
"""
import argparse
from pathlib import Path
import pandas as pd
from common import load_config

MODEL_DISPLAY = {"xgboost": "XGBoost", "inceptiontime": "InceptionTime"}


def _read(rr, name):
    p = rr / name
    return pd.read_csv(p, comment="#") if p.exists() else None


def base_table(rr):
    """Table 2: assemble base-classifier metrics from base_metrics_*.csv."""
    rows = []
    for model in ["inceptiontime", "xgboost"]:
        bm = _read(rr, f"base_metrics_{model}.csv")
        if bm is None or bm.empty:
            continue
        r = bm.iloc[0].to_dict()
        r["model"] = MODEL_DISPLAY.get(model, model)
        rows.append(r)
    return pd.DataFrame(rows) if rows else None


def ecg_table(rr):
    """Table 9: merge PPG-only and PPG+ECG per-class + base metrics."""
    rows = []
    for setting, suf in [("PPG-only", ""), ("PPG+ECG", "_ecg")]:
        for model in ["xgboost", "inceptiontime"]:
            pc = _read(rr, f"per_class_{model}{suf}.csv")
            bm = _read(rr, f"base_metrics_{model}{suf}.csv")
            if pc is None:
                continue
            crit = pc[pc.cls == "Critical"]; ov = pc[pc.cls == "OVERALL"]
            if crit.empty or ov.empty:
                continue
            test_acc = float(bm.iloc[0].accuracy) if bm is not None and not bm.empty else ""
            cc = crit.iloc[0]
            critical_cc = float(cc.lac_cc_cov) if "lac_cc_cov" in crit.columns else ""
            rows.append(dict(setting=setting, model=MODEL_DISPLAY.get(model, model),
                             test_acc=test_acc,
                             global_cov=float(ov.iloc[0].lac_global_cov),
                             critical_global=float(crit.iloc[0].lac_global_cov),
                             critical_cc=critical_cc))
    return pd.DataFrame(rows) if rows else None


def main(cfg):
    out = Path(cfg["output_dir"]); rr = out / "reported_results"
    if not rr.exists():
        raise FileNotFoundError(f"{rr} not found; run 05_conformal.py first.")

    # Table 2
    bt = base_table(rr)
    if bt is not None and not bt.empty:
        bt.to_csv(rr / "base_classifier_performance.csv", index=False)
        print(f"Wrote base_classifier_performance.csv ({len(bt)} rows)")
    else:
        print("base_metrics_*.csv not found; leaving the shipped "
              "base_classifier_performance.csv untouched.")

    # Table 9 — only when an ECG-assisted run is present
    has_ecg = any((rr / f"per_class_{m}_ecg.csv").exists()
                  for m in ["xgboost", "inceptiontime"])
    if has_ecg:
        et = ecg_table(rr)
        if et is not None and not et.empty:
            et.to_csv(rr / "ecg_sensitivity.csv", index=False)
            print(f"Wrote ecg_sensitivity.csv ({len(et)} rows, PPG-only + PPG+ECG)")
    else:
        print("ECG-assisted CSVs not found; leaving the shipped ecg_sensitivity.csv "
              "untouched. Run the --ecg-assisted pipeline to regenerate Table 9.")

    print("Summary tables assembled in", rr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
