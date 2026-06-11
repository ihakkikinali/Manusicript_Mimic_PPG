"""Shared utilities: config, grouping, paths, SQI parsing."""
import re
import yaml
import numpy as np
from pathlib import Path

GROUP_MAP = {
    "SR": "normal", "SARRH": "normal",
    "STACH": "tachy_brady", "SBRAD": "tachy_brady",
    "AF": "af_aflt", "AFLT": "af_aflt",
    "VPACE": "pacing_block", "AVPACE": "pacing_block", "APACE": "pacing_block",
    "1AVB": "pacing_block", "2AVBM1": "pacing_block", "2AVBM2": "pacing_block",
    "3AVB": "pacing_block", "RBBB": "pacing_block", "LBBB": "pacing_block",
    "VTACH": "critical", "VFIB": "critical", "ASYS": "critical", "IDIOV": "critical",
    "SVTACH": "other", "JR": "other", "JTACH": "other", "MATACH": "other",
    "WAPACE": "other", "PATACH": "other", "OTHER": "other",
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_signal_path(row, data_root):
    """Resolve the WFDB record path for a metadata row.

    folder_path is e.g. 'p00/p000052/3238451_0005_0_1'. Records are stored under
    <data_root>/<prefix>/<folder>/, matching the PhysioNet directory layout
    exactly (p00/, p01/, ..., p09/). No prefix remapping is performed.
    """
    data_root = Path(data_root)
    parts = row["folder_path"].split("/")
    prefix, folder = parts[0], parts[1]
    fname = row["signal_file_name"]
    return data_root / prefix / folder / fname


def parse_sqi_vector(s):
    """Parse a stored SQI vector like '[1, 1, 0]' (possibly wrapped in
    np.float64(...)) into a list of ints."""
    try:
        s = re.sub(r"np\.float64\(([^)]+)\)", r"\1", str(s))
        s = s.strip().strip("[]").replace(",", " ")
        return [int(float(x)) for x in s.split()
                if x.strip() not in ("nan", "none", "")]
    except Exception:
        return []


def sqi_score(vec):
    """Continuous per-segment quality in [0,1] = fraction of high-quality windows."""
    if not vec:
        return 0.0
    return sum(1 for x in vec if x == 1) / len(vec)


def inner_train_val_split(train_meta, val_frac=0.15, seed=42):
    """Split the TRAIN metadata into patient-disjoint inner-train and inner-val
    subsets, used for early stopping / checkpoint selection ONLY.

    The calibration and test splits are never touched here, so conformal
    coverage guarantees are not compromised by model selection.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    patients = train_meta["subject_id"].unique()
    rng.shuffle(patients)
    n_val = max(1, int(round(len(patients) * val_frac)))
    val_patients = set(patients[:n_val])
    is_val = train_meta["subject_id"].isin(val_patients)
    return train_meta[~is_val].copy(), train_meta[is_val].copy()
