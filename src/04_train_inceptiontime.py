"""Step 4 — Train the InceptionTime model on raw waveforms (PPG-only by default).

InceptionTime is more stable than a plain 1D-ResNet on this severely imbalanced
waveform data; the residual variant exhibited representation collapse, motivating
this choice (see paper, Methods).

Checkpoint selection / early stopping uses an inner validation subset drawn
patient-disjoint from the TRAIN split. The calibration and test splits are NOT
used during training. With --ecg-assisted, the ECG channel is added as a second
input channel (Section 5.8).
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from common import load_config, build_signal_path, inner_train_val_split

SEG = 3750


class PPGDataset(Dataset):
    """Loads the configured channels. `channels` defaults to ["PLETH"] (PPG-only);
    pass ["PLETH", "II"] for the ECG-assisted variant."""
    def __init__(self, meta, le, data_root, channels=("PLETH",), augment=False):
        self.meta = meta.reset_index(drop=True)
        self.le = le; self.root = data_root
        self.channels = list(channels); self.aug = augment

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        r = self.meta.iloc[i]
        n_ch = len(self.channels)
        try:
            rec = wfdb.rdrecord(str(build_signal_path(r, self.root)))
            names = rec.sig_name
            chans = []
            for nm in self.channels:
                if nm in names:
                    x = rec.p_signal[:, names.index(nm)].astype(np.float32)
                    x = np.nan_to_num(x); s = x.std()
                    x = (x - x.mean()) / s if s > 1e-6 else x
                else:
                    x = np.zeros(SEG, np.float32)
                chans.append(x[:SEG])
            arr = np.stack(chans, 0)
            if arr.shape[1] < SEG:                       # pad short records
                arr = np.pad(arr, ((0, 0), (0, SEG - arr.shape[1])))
        except Exception:
            arr = np.zeros((n_ch, SEG), np.float32)      # match expected channel count
        if self.aug and np.random.rand() < 0.4:
            arr = arr + np.random.normal(0, 0.02, arr.shape).astype(np.float32)
        y = self.le.transform([r.clinical_group])[0]
        return torch.tensor(arr), torch.tensor(y, dtype=torch.long)


class InceptionModule(nn.Module):
    def __init__(self, in_ch, n_filters=32, bottleneck=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False) if in_ch > 1 else None
        bn_ch = bottleneck if in_ch > 1 else in_ch
        self.convs = nn.ModuleList([
            nn.Conv1d(bn_ch, n_filters, k, padding=k // 2, bias=False) for k in [39, 19, 9]
        ])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_ch, n_filters, 1, bias=False)
        self.bn = nn.BatchNorm1d(n_filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        inp = x
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        outs = [c(x) for c in self.convs]
        outs.append(self.conv_pool(self.maxpool(inp)))
        return self.relu(self.bn(torch.cat(outs, 1)))


class InceptionTime(nn.Module):
    def __init__(self, in_ch=1, n_classes=6, n_filters=32, depth=6):
        super().__init__()
        self.blocks = nn.ModuleList(); self.shortcuts = nn.ModuleList()
        ch = in_ch
        for d in range(depth):
            self.blocks.append(InceptionModule(ch, n_filters)); ch = n_filters * 4
            if d % 3 == 2:
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(in_ch if d == 2 else n_filters * 4, n_filters * 4, 1, bias=False),
                    nn.BatchNorm1d(n_filters * 4)))
        self.gap = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(n_filters * 4, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x; sc = 0
        for d, blk in enumerate(self.blocks):
            x = blk(x)
            if d % 3 == 2:
                x = self.relu(x + self.shortcuts[sc](res)); res = x; sc += 1
        return self.fc(self.gap(x).squeeze(-1))


def balanced_sample(meta, k, seed):
    parts = [meta[meta.clinical_group == g].sample(
        min((meta.clinical_group == g).sum(), k), random_state=seed)
        for g in meta.clinical_group.unique()]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def main(cfg, ecg_assisted):
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    root = cfg["data_root"]
    channels = cfg.get("channels", ["PLETH"])
    if ecg_assisted and "II" not in channels:
        channels = ["PLETH", "II"]
    in_ch = len(channels)
    suffix = "_ecg" if ecg_assisted else ""

    df = pd.read_parquet(out / "cohort.parquet")
    le = LabelEncoder(); le.fit(df.clinical_group); n = len(le.classes_)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_full = df[df.split == "train"]
    # patient-disjoint inner train/val for checkpoint selection (never cal/test)
    tr_meta, va_meta = inner_train_val_split(
        tr_full, val_frac=cfg.get("inner_val_frac", 0.15), seed=cfg["seed"])
    va_dl = DataLoader(PPGDataset(va_meta, le, root, channels),
                       batch_size=cfg["model"]["batch_size"])

    model = InceptionTime(in_ch, n).to(dev)
    opt = AdamW(model.parameters(), lr=cfg["model"]["lr"],
                weight_decay=cfg["model"]["weight_decay"])
    crit = nn.CrossEntropyLoss(); best = 1e9
    for ep in range(1, cfg["model"]["epochs"] + 1):
        bal = balanced_sample(tr_meta, cfg["model"]["samples_per_class"], cfg["seed"] + ep)
        dl = DataLoader(PPGDataset(bal, le, root, channels, augment=True),
                        batch_size=cfg["model"]["batch_size"], shuffle=True)
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # validation on the INNER VALIDATION subset only
        model.eval(); vl = nseen = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                vl += crit(model(xb), yb).item() * len(yb); nseen += len(yb)
        vl /= max(nseen, 1)
        print(f"epoch {ep:02d}  inner_val_loss={vl:.4f}")
        if vl < best:
            best = vl; torch.save(model.state_dict(), out / f"inception_best{suffix}.pt")
    print(f"Best inner_val_loss={best:.4f}  (in_ch={in_ch}, channels={channels})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ecg-assisted", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    main(cfg, a.ecg_assisted or cfg.get("ecg_assisted", False))
