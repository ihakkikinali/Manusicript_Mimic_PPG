"""Step 5 — Conformal evaluation reproducing the paper's analyses.

Produces, for the trained PPG-only InceptionTime model (and the XGBoost baseline
if its probabilities are available):
  - LAC global & class-conditional per-class coverage / set size  (Table 3)
  - APS global per-class coverage                                  (Table 4)
  - segment-level and patient-clustered bootstrap CIs (critical)   (Table 5)
  - SQI-stratum behaviour under global CP                          (Table 6)
  - SQI calibration variants                                       (Table 7)
  - prevalence-reweighted global CP                                (Table 8)
  - per-patient coverage                                           (Fig. 4)
All outputs are written as CSVs under outputs/reported_results/.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from common import load_config
from importlib import import_module

it_mod = import_module("04_train_inceptiontime")
PPGDataset, InceptionTime = it_mod.PPGDataset, it_mod.InceptionTime


# ---------- display-name mapping (match the paper's table/figure labels) ----------
DISPLAY = {
    "critical": "Critical", "other": "Other", "normal": "Normal",
    "pacing_block": "Pacing/Block", "af_aflt": "AF/AFLT", "tachy_brady": "Tachy/Brady",
}


def disp(name):
    return DISPLAY.get(str(name), str(name))


# ---------- conformal primitives ----------
def cp_tau(scores, alpha):
    """Split-conformal threshold as the finite-sample order statistic:
    the ceil((n+1)(1-alpha))-th smallest nonconformity score."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return s[min(k - 1, n - 1)]


def metrics(psets, y):
    cov = np.mean([y[i] in psets[i] for i in range(len(y))])
    size = np.mean([len(s) for s in psets])
    return cov, size


def lac_sets(prob, tau):
    return [np.where(1 - prob[i] <= tau)[0] for i in range(len(prob))]


def aps_scores(prob, labels):
    s = np.zeros(len(labels))
    for i in range(len(labels)):
        order = np.argsort(prob[i])[::-1]; cum = 0.0
        for c in order:
            cum += prob[i, c]
            if c == labels[i]:
                s[i] = cum; break
    return s


def aps_sets(prob, tau):
    out = []
    for i in range(len(prob)):
        order = np.argsort(prob[i])[::-1]; cum = 0.0; S = []
        for c in order:
            cum += prob[i, c]; S.append(c)
            if cum >= tau:
                break
        out.append(np.array(S))
    return out


def weighted_quantile(scores, weights, q):
    o = np.argsort(scores); s = scores[o]; w = weights[o]
    cw = np.cumsum(w) / np.sum(w)
    return s[np.searchsorted(cw, min(q, 1.0))]


# ---------- main ----------
def evaluate(prob_cal, y_cal, prob_te, y_te, sqi_cal, sqi_te, sid_te,
             classes, alpha, gamma, rdir, tag):
    K = len(classes)
    crit = list(classes).index("critical") if "critical" in classes else None

    # ----- LAC global -----
    nc = 1 - prob_cal[np.arange(len(y_cal)), y_cal]
    tau_g = cp_tau(nc, alpha)
    ps_g = lac_sets(prob_te, tau_g)

    # ----- LAC class-conditional -----
    tau_cc = {k: (cp_tau(nc[y_cal == k], alpha) if (y_cal == k).sum() >= 5 else tau_g)
              for k in range(K)}
    ps_cc = [np.where(1 - prob_te[i] <= np.array([tau_cc[k] for k in range(K)]))[0]
             for i in range(len(y_te))]

    # ----- APS global -----
    aps_cal = aps_scores(prob_cal, y_cal)
    tau_aps = cp_tau(aps_cal, alpha)
    ps_aps = aps_sets(prob_te, tau_aps)

    # per-class table (Table 3/4)
    rows = []
    for k in range(K):
        m = y_te == k
        if m.sum() == 0:
            continue
        cg, sg = metrics([ps_g[i] for i in np.where(m)[0]], y_te[m])
        cc, sc = metrics([ps_cc[i] for i in np.where(m)[0]], y_te[m])
        ca, sa = metrics([ps_aps[i] for i in np.where(m)[0]], y_te[m])
        rows.append(dict(model=tag, cls=disp(classes[k]), n=int(m.sum()),
                         lac_global_cov=round(cg, 3), lac_global_size=round(sg, 3),
                         lac_cc_cov=round(cc, 3), lac_cc_size=round(sc, 3),
                         aps_global_cov=round(ca, 3)))
    og, _ = metrics(ps_g, y_te); occ, _ = metrics(ps_cc, y_te); oa, _ = metrics(ps_aps, y_te)
    rows.append(dict(model=tag, cls="OVERALL", n=len(y_te),
                     lac_global_cov=round(og, 3), lac_global_size="",
                     lac_cc_cov=round(occ, 3), lac_cc_size="", aps_global_cov=round(oa, 3)))
    pd.DataFrame(rows).to_csv(rdir / f"per_class_{tag}.csv", index=False)

    # base classifier metrics from the test-set argmax predictions (Table 2 inputs)
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
    y_pred = prob_te.argmax(1)
    bm = dict(model=tag,
              accuracy=round(accuracy_score(y_te, y_pred), 3),
              balanced_accuracy=round(balanced_accuracy_score(y_te, y_pred), 3),
              macro_f1=round(f1_score(y_te, y_pred, average="macro", zero_division=0), 3))
    if crit is not None:
        cm = y_te == crit
        bm["critical_recall"] = round(recall_score(y_te == crit, y_pred == crit, zero_division=0), 3)
        bm["critical_fn"] = f"{int((cm & (y_pred != crit)).sum())}/{int(cm.sum())}"
    pd.DataFrame([bm]).to_csv(rdir / f"base_metrics_{tag}.csv", index=False)

    if crit is None:
        return

    # ----- bootstrap CIs for critical (segment + patient-clustered) -----
    def seg_boot(ps, nb=1000, seed=42):
        rng = np.random.default_rng(seed); idx = np.where(y_te == crit)[0]
        cs = [np.mean([crit in ps[i] for i in rng.choice(idx, len(idx), replace=True)])
              for _ in range(nb)]
        return np.mean(cs), np.percentile(cs, 2.5), np.percentile(cs, 97.5)

    def clus_boot(ps, nb=1000, seed=42):
        rng = np.random.default_rng(seed)
        pats = np.unique(sid_te)
        pidx = {p: np.where((sid_te == p) & (y_te == crit))[0] for p in pats}
        pats_w = [p for p in pats if len(pidx[p]) > 0]
        cs = []
        for _ in range(nb):
            samp = rng.choice(pats_w, len(pats_w), replace=True)
            hits = [crit in ps[i] for p in samp for i in pidx[p]]
            if hits:
                cs.append(np.mean(hits))
        return np.mean(cs), np.percentile(cs, 2.5), np.percentile(cs, 97.5)

    ci_rows = []
    crit_mask = y_te == crit
    for name, ps in [("global", ps_g), ("class_conditional", ps_cc)]:
        est, _ = metrics([ps[i] for i in np.where(crit_mask)[0]], y_te[crit_mask])
        sm, sl, sh = seg_boot(ps); cm, cl, ch = clus_boot(ps)
        ci_rows.append(dict(model=tag, strategy=name, estimate=round(est, 3),
                            seg_lo=round(sl, 3), seg_hi=round(sh, 3),
                            cluster_lo=round(cl, 3), cluster_hi=round(ch, 3)))
    pd.DataFrame(ci_rows).to_csv(rdir / f"critical_ci_{tag}.csv", index=False)

    # ----- SQI stratum under global CP (Table 6) -----
    s_rows = []
    for lab, mask in [("high", sqi_te == 1.0), ("mixed", (sqi_te > 0) & (sqi_te < 1)),
                      ("low", sqi_te == 0.0)]:
        if mask.sum() == 0:
            continue
        c, s = metrics([ps_g[i] for i in np.where(mask)[0]], y_te[mask])
        s_rows.append(dict(model=tag, stratum=lab, n=int(mask.sum()),
                           coverage=round(c, 3), mean_size=round(s, 3)))
    pd.DataFrame(s_rows).to_csv(rdir / f"sqi_stratum_{tag}.csv", index=False)

    # ----- SQI calibration variants (Table 7) -----
    w = lambda q: 1.0 + gamma * (1.0 - np.clip(q, 0, 1))
    nc_w = nc / w(sqi_cal); tau_w = cp_tau(nc_w, alpha)
    ps_w = [np.where((1 - prob_te[i]) / w(sqi_te[i]) <= tau_w)[0] for i in range(len(y_te))]
    ps_gate = [np.arange(K) if sqi_te[i] == 0 else ps_g[i] for i in range(len(y_te))]
    strata = {"high": sqi_cal == 1.0, "mixed": (sqi_cal > 0) & (sqi_cal < 1), "low": sqi_cal == 0.0}
    tau_s = {kk: (cp_tau(nc[mm], alpha) if mm.sum() >= 5 else tau_g) for kk, mm in strata.items()}
    strat_of = lambda q: "high" if q == 1 else ("low" if q == 0 else "mixed")
    ps_strat = [np.where(1 - prob_te[i] <= tau_s[strat_of(sqi_te[i])])[0] for i in range(len(y_te))]
    keep = sqi_te > 0
    v_rows = []
    for name, ps, yy in [("sqi_weighted", ps_w, y_te), ("q0_full", ps_gate, y_te),
                         ("sqi_stratified", ps_strat, y_te),
                         ("q0_discard", [ps_g[i] for i in np.where(keep)[0]], y_te[keep])]:
        c, s = metrics(ps, yy)
        lm = (sqi_te == 0.0) if name != "q0_discard" else None
        low = ""
        if lm is not None and lm.sum() > 0:
            _, low = metrics([ps[i] for i in np.where(lm)[0]], yy[lm])
            low = round(low, 3)
        v_rows.append(dict(model=tag, variant=name, coverage=round(c, 3),
                           mean_size=round(s, 3), low_sqi_size=low))
    pd.DataFrame(v_rows).to_csv(rdir / f"sqi_variants_{tag}.csv", index=False)

    # ----- prevalence reweighting (Table 8) -----
    # weights = full-benchmark / cohort class frequency (segment level).
    # Cohort frequency from calibration; full-benchmark from cohort.parquet if present.
    cal_freq = pd.Series(y_cal).value_counts(normalize=True).to_dict()
    full_csv = rdir.parent / "full_benchmark_freq.csv"
    if full_csv.exists():
        ff = pd.read_csv(full_csv).set_index("clinical_group")["freq"].to_dict()
        full_freq = {list(classes).index(g): v for g, v in ff.items() if g in list(classes)}
        wc = np.array([full_freq.get(c, 1e-9) / max(cal_freq.get(c, 1e-9), 1e-9) for c in y_cal])
        tau_rw = weighted_quantile(nc, wc, min(np.ceil((len(nc) + 1) * (1 - alpha)) / len(nc), 1.0))
        ps_rw = lac_sets(prob_te, tau_rw)
        cc_rw, _ = metrics([ps_rw[i] for i in np.where(y_te == crit)[0]], y_te[y_te == crit])
        ov_rw, _ = metrics(ps_rw, y_te)
        ov_g, _ = metrics(ps_g, y_te)
        cc_g, _ = metrics([ps_g[i] for i in np.where(y_te == crit)[0]], y_te[y_te == crit])
        pd.DataFrame([
            dict(model=tag, cls="Critical", cohort=round(cc_g, 3), reweighted=round(cc_rw, 3)),
            dict(model=tag, cls="Overall",  cohort=round(ov_g, 3), reweighted=round(ov_rw, 3)),
        ]).to_csv(rdir / f"prevalence_reweight_{tag}.csv", index=False)

    # ----- per-patient coverage (Fig. 4) -----
    pr = []
    for p in np.unique(sid_te):
        m = sid_te == p
        if m.sum() < 3:
            continue
        c, _ = metrics([ps_g[i] for i in np.where(m)[0]], y_te[m])
        pr.append(dict(model=tag, subject_id=int(p), n_seg=int(m.sum()), coverage=round(c, 3)))
    pd.DataFrame(pr).to_csv(rdir / f"per_patient_{tag}.csv", index=False)


def main(cfg, ecg_assisted):
    out = Path(cfg["output_dir"]); root = cfg["data_root"]
    rdir = out / "reported_results"; rdir.mkdir(parents=True, exist_ok=True)
    alpha = cfg["conformal"]["alpha"]; gamma = cfg["conformal"].get("sqi_gamma", 0.15)
    suffix = "_ecg" if ecg_assisted else ""
    channels = cfg.get("channels", ["PLETH"])
    if ecg_assisted and "II" not in channels:
        channels = ["PLETH", "II"]

    df = pd.read_parquet(out / "cohort.parquet")
    le = LabelEncoder(); le.fit(df.clinical_group); K = len(le.classes_)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # InceptionTime probabilities
    model = InceptionTime(len(channels), K).to(dev)
    model.load_state_dict(torch.load(out / f"inception_best{suffix}.pt", map_location=dev))
    model.eval()

    def probs(meta):
        dl = DataLoader(PPGDataset(meta, le, root, channels), batch_size=cfg["model"]["batch_size"])
        P, Y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                P.append(torch.softmax(model(xb.to(dev)), 1).cpu().numpy()); Y.append(yb.numpy())
        return np.concatenate(P), np.concatenate(Y)

    ca, te = df[df.split == "calibration"], df[df.split == "test"]
    pc, yc = probs(ca); pt, yt = probs(te)
    tag = "inceptiontime" + suffix
    evaluate(pc, yc, pt, yt, ca.sqi_score.values, te.sqi_score.values,
             te.subject_id.values, le.classes_, alpha, gamma, rdir, tag)
    print(f"[{tag}] conformal CSVs written to {rdir}")

    # XGBoost probabilities, if available
    xgb_npz = out / f"xgb_probs{suffix}.npz"
    if xgb_npz.exists():
        d = np.load(xgb_npz, allow_pickle=True)
        evaluate(d["prob_cal"], d["y_cal"], d["prob_te"], d["y_te"],
                 d["sqi_cal"], d["sqi_te"], d["sid_te"], list(d["classes"]),
                 alpha, gamma, rdir, "xgboost" + suffix)
        print(f"[xgboost{suffix}] conformal CSVs written to {rdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ecg-assisted", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    main(cfg, a.ecg_assisted or cfg.get("ecg_assisted", False))
