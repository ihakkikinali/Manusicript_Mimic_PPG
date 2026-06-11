# Conformal Reliability Assessment of PPG-Based ICU Arrhythmia Classification under Class Imbalance and Patient Heterogeneity

Reproducibility package for the paper *"Conformal Reliability Assessment of
PPG-Based ICU Arrhythmia Classification under Class Imbalance and Patient
Heterogeneity."*

This repository contains the code used for cohort construction, feature
extraction, model training (PPG-only **XGBoost** and **InceptionTime**), and the
conformal calibration / evaluation pipeline. Aggregate result tables used for the
figures and tables in the paper are also provided under `outputs/reported_results/`.

The **main experiments are PPG-only**. An optional ECG-assisted sensitivity
analysis (paper, Section 5.8) is available behind an explicit `--ecg-assisted`
flag and is **off by default**.

---

## 1. Data

Experiments use the **MIMIC-III-Ext-PPG** benchmark (v1.1.0), publicly available on
PhysioNet under credentialed access:

> Moulaeifard, M., Charlton, P. H., & Strodthoff, N. (2026).
> MIMIC-III-Ext-PPG: A PPG Benchmark Dataset for Cardiorespiratory Analysis (v1.1.0).
> PhysioNet. https://doi.org/10.13026/r6k1-xt76

We do **not** redistribute any patient data. Access requires completion of the
required CITI training and a signed data use agreement, per PhysioNet policy.

### Expected directory layout
```
DATA_ROOT/
├── metadata.csv
├── p00/                 # patient folders p000052/, p000107/, ...
├── p01/ ... p09/        # additional patient folders pXXYYYY/
```
Each patient folder contains WFDB records (`.hea` + `.dat`) at 125 Hz, with
channels PLETH, II (ECG), and optionally ABP / RESP. The directory layout matches
PhysioNet exactly (`p00/`, `p01/`, ..., `p09/`); no prefix remapping is performed.
The main pipeline reads **only the PLETH (PPG) channel**; the ECG channel is read
only when `--ecg-assisted` is set.

---

## 2. Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.x, CUDA 12. A single NVIDIA T4 / A100 GPU is
sufficient; InceptionTime trains in ~1–2 hours on a T4.

---

## 3. Pipeline

All steps are driven by `configs/ppg_only.yaml` (main experiments). Set
`data_root` there first. Output and figure directories are created automatically.

```bash
# Step 1 — build the expanded cohort and patient-disjoint split
python src/01_build_cohort.py        --config configs/ppg_only.yaml

# Step 2 — extract 14-dim PPG-only features (for the XGBoost baseline)
python src/02_extract_features.py    --config configs/ppg_only.yaml

# Step 3 — train the PPG-only XGBoost baseline
python src/03_train_xgboost.py       --config configs/ppg_only.yaml

# Step 4 — train the PPG-only InceptionTime model on raw waveforms
python src/04_train_inceptiontime.py --config configs/ppg_only.yaml

# Step 5 — conformal evaluation (LAC + APS, global / class-conditional,
#          patient-clustered & segment bootstrap, SQI variants,
#          prevalence reweighting)
python src/05_conformal.py           --config configs/ppg_only.yaml

# Step 6 — regenerate the paper figures from the produced CSVs
python src/06_make_figures.py        --config configs/ppg_only.yaml

# Step 7 (optional) — assemble the paper-formatted summary tables from the
#          per-run CSVs (Table 2 base-classifier metrics; Table 9 PPG-only vs
#          PPG+ECG, which also needs the ECG-assisted run below)
python src/07_make_summary_tables.py --config configs/ppg_only.yaml
```

### ECG-assisted sensitivity analysis (optional, paper Section 5.8)
```bash
python src/02_extract_features.py    --config configs/ecg_assisted.yaml --ecg-assisted
python src/03_train_xgboost.py       --config configs/ecg_assisted.yaml --ecg-assisted
python src/04_train_inceptiontime.py --config configs/ecg_assisted.yaml --ecg-assisted
python src/05_conformal.py           --config configs/ecg_assisted.yaml --ecg-assisted
```

---

## 4. Experimental Protocol (important)

To avoid information leakage, the **calibration and test splits are never used for
model training, early stopping, or checkpoint selection**:

- The 70% training split is further divided, **patient-disjoint**, into an inner
  training subset and an inner validation subset. Early stopping and checkpoint
  selection (InceptionTime) and XGBoost early stopping use the **inner validation
  subset only**.
- The 15% calibration split is used **only** to compute conformal thresholds.
- The 15% test split is used **only** for final evaluation.

This mirrors the protocol described in the paper (Section 3.5).

---

## 5. Patient-Selection Strategy

`src/01_build_cohort.py` implements the metadata-guided cohort construction
described in the paper: rather than downloading the full 100+ GB dataset, it
selects only the patient records needed to populate every rhythm group
(prioritizing rare critical rhythms). The script emits `download_urls.txt`, a list
of PhysioNet URLs for the selected patients, fetched with `src/download_patients.py`.

---

## 6. Reported Results

Aggregate CSVs for every table and figure in the paper are provided in
`outputs/reported_results/`. `src/06_make_figures.py` reads these CSVs when they
are present and falls back to the reported constants only if a CSV is absent. It
regenerates the figure PDFs used in the manuscript:
`fig1_coverage_by_class.pdf`, `fig5_model_comparison.pdf`,
`fig6_critical_ci.pdf`, `fig4_patient_coverage.pdf`,
`fig3_sqi_stratum.pdf`, and `fig2_coverage_efficiency.pdf`. These correspond
respectively to the paper's Figure 1, Figure 2, Figure 3, Figure 4, Figure 5,
and Appendix Figure A.6. (The number in each filename is an internal pipeline
index and does not necessarily match the manuscript figure number.)
`src/07_make_summary_tables.py` likewise assembles Table 2 (base classifier
metrics) and Table 9 (PPG-only vs PPG+ECG) from the per-run CSVs.

> **Note on `per_patient_inceptiontime.csv`.** The shipped copy is a *canonical
> representation* that reproduces the reported per-patient distribution
> statistics (121 patients; 43 below the 90% target; 41/116 for patients with
> at least 30 segments; mean below-target coverage about 0.79). Its `subject_id`
> values are placeholders, not real PhysioNet identifiers. Running
> `src/05_conformal.py` on the data overwrites this file with the exact
> per-patient coverages (and real patient IDs) from your own run.

| Result | Value |
|---|---|
| Cohort | 826 patients, 1,105,782 segments |
| InceptionTime accuracy / balanced-acc / macro-F1 | 0.650 / 0.388 / 0.385 |
| InceptionTime critical recall | 0.004 (267/268 missed) |
| Global CP critical coverage (overall 0.900) | 0.075 |
| APS global critical coverage (overall 0.960) | 0.231 |
| XGBoost global critical coverage | 0.163 |
| Class-conditional critical coverage (cluster CI) | 0.825 [0.578, 0.985] |
| Patients below target (n>=30 seg) | 41/116 (35.3%) |
| SQI stratum coverage (high / mixed / low) | 0.894 / 0.923 / 0.928 |

---

## 7. License

Code released under the MIT License (see `LICENSE`). The MIMIC-III-Ext-PPG data is
governed separately by the PhysioNet Credentialed Health Data License.

## 8. Citation

> **For anonymous (double-blind) review:** remove the `author` field below (or
> replace it with `author = {Anonymous}`) before sharing this repository as an
> anonymized link. Restore the author in the published version.

```bibtex
anonymous
```
