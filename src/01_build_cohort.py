"""Step 1 — Build the expanded cohort and patient-disjoint split.

Reads the full metadata.csv, selects rare/critical-rhythm patients plus a
stratified balance of common classes, and produces a patient-level
train/calibration/test split. Also emits download_urls.txt for the selected
patients (PhysioNet fetch list).
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from common import load_config, GROUP_MAP, parse_sqi_vector, sqi_score

PHYSIONET_BASE = "https://physionet.org/files/mimic-iii-ext-ppg/1.1.0"


def main(cfg):
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    meta = Path(cfg["data_root"]) / "metadata.csv"
    df = pd.read_csv(meta, low_memory=False)
    df["clinical_group"] = df["event_rhythm"].map(GROUP_MAP).fillna("other")
    df["prefix"] = df["folder_path"].str.split("/").str[0]

    # --- metadata-guided selection (rhythm-label guided, rare-first) ---
    # Selection uses only the rhythm labels and the released stratification folds;
    # no storage partition is added. This matches the paper (Section 3.2): the
    # cohort retains only the patient records needed to populate every rhythm
    # group, prioritizing rare critical rhythms.
    sel = set()
    sel |= set(df[df.clinical_group == "critical"].subject_id.unique())
    for grp in ["pacing_block", "af_aflt"]:
        sel |= set(df[(df.strat_fold == 7) & (df.clinical_group == grp)].subject_id.unique())
    sel |= set(df[(df.strat_fold.isin([8, 9])) &
                  (df.clinical_group.isin(["critical", "pacing_block", "af_aflt"]))].subject_id.unique())
    for grp, k in [("normal", 120), ("tachy_brady", 90)]:
        cand = df[(df.strat_fold == 7) & (df.clinical_group == grp)].subject_id.unique()
        sel |= set(cand[:k])

    cohort = df[df.subject_id.isin(sel)].copy()

    # SQI features
    cohort["pleth_vec"] = cohort["vector_10s_pleth_sqi"].apply(parse_sqi_vector)
    cohort["sqi_ok"]    = cohort["pleth_vec"].apply(lambda v: any(x >= 0 for x in v) if v else False)
    cohort["sqi_score"] = cohort["pleth_vec"].apply(sqi_score)
    cohort = cohort[cohort.sqi_ok].copy()

    # patient-disjoint stratified split.
    # A patient may contribute several rhythm groups; assign the patient-level
    # stratification label by clinical priority (rarest/most-critical first) so the
    # label is deterministic and independent of metadata row order.
    priority = {"critical": 0, "pacing_block": 1, "af_aflt": 2,
                "tachy_brady": 3, "normal": 4, "other": 5}
    tmp_pts = cohort[["subject_id", "clinical_group"]].copy()
    tmp_pts["prio"] = tmp_pts["clinical_group"].map(priority).fillna(99)
    pts = (tmp_pts.sort_values(["subject_id", "prio"])
                  .drop_duplicates("subject_id")[["subject_id", "clinical_group"]])
    counts = pts.clinical_group.value_counts()
    pts["strat"] = pts.clinical_group.apply(lambda g: g if counts[g] >= 6 else "other")
    seed = cfg["seed"]
    tr, tmp = train_test_split(pts.subject_id.values, test_size=1 - cfg["train_frac"],
                               random_state=seed, stratify=pts.strat.values)
    tmpdf = pts[pts.subject_id.isin(tmp)]
    cal, te = train_test_split(tmpdf.subject_id.values, test_size=0.5,
                               random_state=seed, stratify=tmpdf.strat.values)
    smap = {**{p: "train" for p in tr},
            **{p: "calibration" for p in cal},
            **{p: "test" for p in te}}
    cohort["split"] = cohort.subject_id.map(smap)

    cohort.to_parquet(out / "cohort.parquet")
    print(f"Cohort: {cohort.subject_id.nunique()} patients, {len(cohort):,} segments")
    print(cohort.groupby("split").subject_id.nunique())

    # Full-benchmark segment-level class frequency (used by 05_conformal.py for the
    # prevalence-reweighting analysis, Table 8). Computed over the full metadata,
    # not just the cohort, so the reference distribution is the benchmark's own.
    full_freq = df["clinical_group"].value_counts(normalize=True)
    full_freq.rename_axis("clinical_group").reset_index(name="freq") \
        .to_csv(out / "full_benchmark_freq.csv", index=False)
    print(f"Wrote full-benchmark class frequencies -> {out/'full_benchmark_freq.csv'}")

    # download URL list
    urls = []
    for sid in sorted(sel):
        s = str(int(sid)).zfill(6)
        urls.append(f"{PHYSIONET_BASE}/p{s[:2]}/p{s}/")
    (out / "download_urls.txt").write_text("\n".join(urls) + "\n")
    print(f"Wrote {len(urls)} download URLs -> {out/'download_urls.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    main(load_config(a.config))
