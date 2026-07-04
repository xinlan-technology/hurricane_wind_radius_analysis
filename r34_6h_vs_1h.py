"""
Hurricane R34 — 6h vs 1h masked-loss LSTM + grouped SHAP (single-file, Colab-ready).

Two models, evaluated on the SAME 6h best-track test storms:
  * 6h : r34_6h.csv  — every step is a labeled observation.
  * 1h : r34_1h.csv  — full 1-hour sequence in, but loss/eval ONLY at the
         6h "raw_data==True" points (the 5 intermediate ERA5 steps give context).

Outputs -> outputs/: scatter_{6h,1h}.png + shap_{6h,1h}.png (figures),
predictions_{6h,1h}.csv, shap_{6h,1h}.csv, metrics_summary.csv

Colab:
    !pip install torch numpy pandas scikit-learn matplotlib
    # edit the CONFIG block below (just the 3 paths), then:
    !python r34_6h_vs_1h.py
"""

# ============================================================================
# CONFIG  — the ONLY place you need to edit (e.g. point paths at your Drive)
# ============================================================================
DATA_6H    = "r34_6h.csv"
DATA_1H    = "r34_1h.csv"
OUTPUT_DIR = "outputs"

SEED       = 42
TEST_RATIO = 0.1            # fraction of storms held out for testing
N_SPLITS   = 5             # K-fold CV folds for hyperparameter search
MAX_EPOCHS = 400
PATIENCE   = 30
PARAM_GRID = {             # grid searched with K-fold CV
    "hidden_size": [64, 128, 256],
    "num_layers":  [2, 3, 4],
    "dropout":     [0.0, 0.1, 0.3, 0.5],
    "batch_size":  [8, 16],
}
# ============================================================================

import os
import random
import itertools
import tempfile
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import KFold, train_test_split

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib.pyplot as plt

# Clean grid style if available (name varies by mpl version; guard avoids a
# crash at the final plotting step after a long run).
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass

# ----------------------------------------------------------------------------
# 50 features in 3 physics groups (originally recovered from the Excel header
# colours). obs_flag (real-obs indicator) is input #50 and belongs to NO group.
# ----------------------------------------------------------------------------
GROUP_LOCATION = [
    "DIST2LAND", "USA_LAT", "USA_LON", "STORM_SPEED", "STORM_DIR",
    "era_min_lat", "era_min_lon", "distance_dev",
]
GROUP_INTENSITY = [
    "USA_WIND", "USA_PRES", "USA_SSHS",
    "uv_max", "rmax", "rmax_avg", "rmax_std",
    "u850_mean_500km", "u850_std_500km", "v850_mean_500km", "v850_std_500km",
    "warm_core_diff_200_850", "warm_core_pct_200_850", "era_min_pressure",
    "uv_max_mslp_center", "rmax_mslp_center", "rmax_avg_mslp_center", "rmax_std_mslp_center",
    "u850_mean_500km_mslp_center", "u850_std_500km_mslp_center",
    "v850_mean_500km_mslp_center", "v850_std_500km_mslp_center",
    "warm_core_diff_200_850_mslp_center", "warm_core_pct_200_850_mslp_center",
]
GROUP_ENVIRONMENT = [
    "u200_mean_500km", "u200_std_500km", "v200_mean_500km", "v200_std_500km",
    "shear_u_500km", "shear_v_500km", "shear_500km", "rh500_mean_800km", "rh500_std_800km",
    "u200_mean_500km_mslp_center", "u200_std_500km_mslp_center",
    "v200_mean_500km_mslp_center", "v200_std_500km_mslp_center",
    "shear_u_500km_mslp_center", "shear_v_500km_mslp_center", "shear_500km_mslp_center",
    "rh500_mean_800km_mslp_center", "rh500_std_800km_mslp_center",
]
FEATURE_GROUPS = OrderedDict([
    ("Location & Motion",       GROUP_LOCATION),
    ("Intensity & Inner-core",  GROUP_INTENSITY),
    ("Large-scale Environment", GROUP_ENVIRONMENT),
])
# 5-colour, colour-blind-safe (Okabe-Ito): 2 scatter models + 3 SHAP groups, all distinct.
GROUP_COLORS = {"Intensity & Inner-core":  "#D55E00",   # vermillion
                "Large-scale Environment": "#009E73",   # green
                "Location & Motion":       "#CC79A7"}   # purple
SCATTER_COLORS = {"6h": "#0072B2", "1h": "#E69F00"}     # 6h blue vs 1h amber

FEATURE_COLS  = GROUP_LOCATION + GROUP_INTENSITY + GROUP_ENVIRONMENT   # 50 physical features
ALL_INPUT_COLS = FEATURE_COLS + ["obs_flag"]                    # + obs_flag -> 51 inputs
OBS_FLAG_IDX  = len(FEATURE_COLS)                               # index 50
TARGET_COL = "R34avg_adj"

# Best-track obs: real only every 6h -> fed at the 6h points, masked to 0 at the
# intermediate steps (gated by obs_flag). DIST2LAND/USA_WIND carry hourly values too,
# but those are interpolated (not real), so they're masked as well. The other 42 are
# genuine hourly ERA5 predictors.
OBS_6H_COLS = ["DIST2LAND", "USA_LAT", "USA_LON", "USA_WIND",
               "USA_PRES", "USA_SSHS", "STORM_SPEED", "STORM_DIR"]
OBS_6H_IDX  = [ALL_INPUT_COLS.index(c) for c in OBS_6H_COLS]
HOURLY_COLS = [c for c in FEATURE_COLS if c not in OBS_6H_COLS]


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(file_path, is_1h, min_labeled=3, time_col="ISO_TIME"):
    """Build per-storm sequences with a label mask (1 at labeled 6h points)."""
    df = pd.read_csv(file_path, low_memory=False)
    df["_label"] = (df["raw_data"].astype(str).str.upper().eq("TRUE").astype(float)
                    if is_1h else 1.0)
    df["_t"] = pd.to_datetime(df[time_col])
    df = df.sort_values(["SID", "_t"]).reset_index(drop=True)

    # Hourly ERA5: fill sporadic ~0.1% gaps per storm (interpolate -> edge ffill/bfill).
    df[HOURLY_COLS] = df.groupby("SID", group_keys=False)[HOURLY_COLS].apply(
        lambda g: g.interpolate("linear").ffill().bfill())
    na_hourly = df[HOURLY_COLS].isna().sum()
    if na_hourly.any():
        raise ValueError(f"Hourly features still missing: {na_hourly[na_hourly > 0].to_dict()}")
    # Obs are NOT filled: real at 6h, NaN at intermediate (scale() uses 6h-only stats
    # and masks intermediate to 0). They MUST exist at every 6h point:
    na_obs_6h = df.loc[df["_label"] == 1, OBS_6H_COLS].isna().sum()
    if na_obs_6h.any():
        raise ValueError(f"Obs missing at 6h points: {na_obs_6h[na_obs_6h > 0].to_dict()}")

    df["obs_flag"] = df["_label"]
    df["_y"] = df[TARGET_COL].astype(float)
    df.loc[df["_label"] == 0, "_y"] = 0.0            # sentinel -1 -> 0 (masked out)

    keep = df.groupby("SID")["_label"].sum()
    df = df[df["SID"].isin(keep[keep >= min_labeled].index)].copy()

    out = {"X": [], "y": [], "m": [], "time": [], "sids": []}
    for sid, g in df.groupby("SID", sort=False):
        out["X"].append(g[ALL_INPUT_COLS].to_numpy(np.float32))
        out["y"].append(g["_y"].to_numpy(np.float32))
        out["m"].append(g["_label"].to_numpy(np.float32))
        out["time"].append(g[time_col].to_numpy())
        out["sids"].append(sid)
    print(f"[Data] {os.path.basename(file_path)}: {df['SID'].nunique()} storms, "
          f"{len(df)} rows, {int(df['_label'].sum())} labeled")
    return out


def scale(X_list, m_list, fit_idx):
    """Per-channel standardize (stats from `fit_idx` storms only, no leakage):
    obs -> mean/std from real 6h steps (m==1); ERA5 -> all steps; obs_flag -> kept 0/1.
    Then obs are masked to 0 at non-6h steps. No fill value enters the statistics.
    """
    Xfit = np.vstack([X_list[i] for i in fit_idx])
    mfit = np.concatenate([m_list[i] for i in fit_idx]).astype(bool)
    obs_set = set(OBS_6H_IDX)
    F = Xfit.shape[1]
    mean, std = np.zeros(F), np.ones(F)
    for j in range(F):
        if j == OBS_FLAG_IDX:
            continue                                   # obs_flag stays 0/1
        col = Xfit[mfit, j] if j in obs_set else Xfit[:, j]   # obs: 6h points only
        mean[j] = np.nanmean(col)
        s = np.nanstd(col)
        std[j] = s if s > 1e-8 else 1.0

    out = []
    for x, m in zip(X_list, m_list):
        z = (x - mean) / std
        z[:, OBS_FLAG_IDX] = x[:, OBS_FLAG_IDX]        # keep obs_flag raw 0/1
        inter = m == 0
        for j in OBS_6H_IDX:
            z[inter, j] = 0.0                          # mask obs at non-6h steps
        if np.isnan(z).any():
            raise ValueError("NaN remains after scaling/masking inputs.")
        out.append(z.astype(np.float32))
    return out


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class MaskedDataset(Dataset):
    def __init__(self, X, y, m):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.y = [torch.tensor(v, dtype=torch.float32) for v in y]
        self.m = [torch.tensor(v, dtype=torch.float32) for v in m]

    def __len__(self):  return len(self.X)
    def __getitem__(self, i):  return self.X[i], self.y[i], self.m[i]


def collate(batch):
    X, y, m = zip(*batch)
    Xp = pad_sequence(X, batch_first=True)
    yp = pad_sequence(y, batch_first=True)
    mp = pad_sequence(m, batch_first=True)               # label mask
    lengths = torch.tensor([len(x) for x in X], dtype=torch.long)
    pad = (torch.arange(yp.size(1))[None, :] < lengths[:, None]).float()
    return Xp, yp, mp * pad, lengths                     # loss mask = label * padding


class LSTMRegressor(nn.Module):
    """Unidirectional (causal) LSTM -> per-timestep scalar prediction."""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.fc(out).squeeze(-1)

    def forward_all(self, x):           # no packing; used for SHAP forward passes
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


def masked_mse(pred, target, mask):
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)


def train_model(model, train_loader, device, val_loader=None,
                max_epochs=None, patience=None):
    """Train with early stopping on val loss (or train loss if no val set)."""
    if max_epochs is None:
        max_epochs = MAX_EPOCHS
    if patience is None:
        patience = PATIENCE
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    best, best_state, wait = float("inf"), None, 0

    for _ in range(max_epochs):
        model.train()
        for Xb, yb, mb, lb in train_loader:
            Xb, yb, mb = Xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad()
            loss = masked_mse(model(Xb, lb), yb, mb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # monitored loss
        loader = val_loader if val_loader is not None else train_loader
        model.eval(); se = n = 0.0
        with torch.no_grad():
            for Xb, yb, mb, lb in loader:
                Xb, yb, mb = Xb.to(device), yb.to(device), mb.to(device)
                pred = model(Xb, lb)
                se += ((pred - yb) ** 2 * mb).sum().item(); n += mb.sum().item()
        cur = se / max(n, 1.0)
        sched.step(cur)
        if cur < best - 1e-6:
            best, best_state, wait = cur, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def search_hyperparams(data, trainval, device):
    """K-fold CV grid search; scaler refit per fold (no leakage)."""
    grid = list(itertools.product(PARAM_GRID["hidden_size"], PARAM_GRID["num_layers"],
                                  PARAM_GRID["dropout"], PARAM_GRID["batch_size"]))
    print(f"[Search] {len(grid)} combos x {N_SPLITS} folds")
    kf = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    best_combo, best_loss = None, float("inf")

    for ci, (h, nl, dr, bs) in enumerate(grid):
        losses = []
        for tr_rel, va_rel in kf.split(trainval):
            tr = [trainval[i] for i in tr_rel]; va = [trainval[i] for i in va_rel]
            Xs = scale(data["X"], data["m"], tr)   # obs stats from THIS fold's 6h steps
            tl = DataLoader(MaskedDataset([Xs[i] for i in tr], [data["y"][i] for i in tr],
                                          [data["m"][i] for i in tr]), bs, shuffle=True, collate_fn=collate)
            vl = DataLoader(MaskedDataset([Xs[i] for i in va], [data["y"][i] for i in va],
                                          [data["m"][i] for i in va]), bs, shuffle=False, collate_fn=collate)
            model = LSTMRegressor(len(ALL_INPUT_COLS), h, nl, dr)
            losses.append(train_model(model, tl, device, vl))
        avg = float(np.mean(losses))
        if avg < best_loss:
            best_loss, best_combo = avg, (h, nl, dr, bs)
        if (ci + 1) % 10 == 0:
            print(f"[Search] {ci + 1}/{len(grid)}")
    h, nl, dr, bs = best_combo
    print(f"[Search] best: hidden={h} layers={nl} dropout={dr} batch={bs} loss={best_loss:.3f}")
    return {"hidden_size": h, "num_layers": nl, "dropout": dr, "batch_size": bs}


def evaluate(model, Xs, data, test, device):
    """Predict and collect ONLY labeled (6h) points for the test storms."""
    model.eval(); rows = []
    with torch.no_grad():
        for i in test:
            x = torch.tensor(Xs[i], dtype=torch.float32, device=device).unsqueeze(0)
            pred = model(x, torch.tensor([x.shape[1]])).squeeze(0).cpu().numpy()
            for t in np.where(data["m"][i].astype(bool))[0]:
                rows.append({
                    "SID": data["sids"][i],
                    "ISO_TIME": data["time"][i][t],
                    "actual_R34": float(data["y"][i][t]),
                    "predicted_R34": float(pred[t]),
                })
    pred_df = pd.DataFrame(rows)
    pred_df["error"] = pred_df["predicted_R34"] - pred_df["actual_R34"]
    pred_df["abs_error"] = pred_df["error"].abs()
    pred_df["squared_error"] = pred_df["error"] ** 2
    a, p = pred_df["actual_R34"].to_numpy(), pred_df["predicted_R34"].to_numpy()
    r2 = 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae = float(np.mean(np.abs(a - p)))
    print(f"[Eval] {len(rows)} labeled test points | R2={r2:.3f} RMSE={rmse:.2f} MAE={mae:.2f}")
    return {"df": pred_df, "actual": a, "pred": p, "r2": float(r2), "rmse": rmse, "mae": mae}


# ----------------------------------------------------------------------------
# Grouped SHAP — exact permutation Shapley over the 3 groups (all 3!=6 orderings).
# ----------------------------------------------------------------------------
def grouped_shap(model, Xs, data, test, trainval, device):
    """Exact group Shapley at the labeled 6h steps: baseline = mean over labeled
    trainval steps (obs_flag held real); each of the 6 group orderings reveals a
    group to its true values and records the marginal change -> averaged Shapley.
    """
    model.eval()
    g_idx = OrderedDict((name, [ALL_INPUT_COLS.index(c) for c in cols])
                        for name, cols in FEATURE_GROUPS.items())
    names = list(g_idx)

    baseline_vec = np.vstack([Xs[i][data["m"][i].astype(bool)] for i in trainval]).mean(0)

    seqs, masks = [Xs[i] for i in test], [data["m"][i] for i in test]
    T = max(len(s) for s in seqs); N, F = len(seqs), seqs[0].shape[1]
    X = np.zeros((N, T, F), np.float32); M = np.zeros((N, T), np.float32)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        X[i, :len(s)] = s; M[i, :len(m)] = m

    Xt = torch.tensor(X, device=device)
    base = torch.tensor(np.broadcast_to(baseline_vec, (N, T, F)).copy(), device=device)
    base[:, :, OBS_FLAG_IDX] = Xt[:, :, OBS_FLAG_IDX]            # keep obs_flag real
    # obs are 0 at non-6h steps -> match in the baseline so their reveal perturbs
    # only the real 6h values.
    im = torch.tensor(M == 0, device=device)
    for j in OBS_6H_IDX:
        base[:, :, j] = torch.where(im, torch.zeros_like(base[:, :, j]), base[:, :, j])

    acc = np.zeros((N, len(names), T))
    with torch.no_grad():
        for perm in itertools.permutations(range(len(names))):
            cur = base.clone()
            f_prev = model.forward_all(cur).cpu().numpy()
            for g in perm:
                cur[:, :, g_idx[names[g]]] = Xt[:, :, g_idx[names[g]]]
                f_cur = model.forward_all(cur).cpu().numpy()
                acc[:, g, :] += f_cur - f_prev
                f_prev = f_cur
    acc /= 6
    mb = M.astype(bool)
    imp = {names[g]: float(np.abs(acc[:, g, :])[mb].mean()) for g in range(len(names))}
    print("[SHAP] " + " | ".join(f"{k}={v:.3f}" for k, v in
                                 sorted(imp.items(), key=lambda kv: -kv[1])))
    return imp


# ----------------------------------------------------------------------------
# Plots — the only deliverables: one scatter + one grouped-SHAP bar per model.
# ----------------------------------------------------------------------------
def plot_scatter(res, path, color="#4C72B0"):
    a, p = res["actual"], res["pred"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(a, p, s=25, alpha=0.6, c=color, edgecolors="none")
    lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=2, alpha=0.8)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Observed R34 [km]", fontsize=16, fontweight="bold")
    ax.set_ylabel("Predicted R34 [km]", fontsize=16, fontweight="bold")
    ax.text(0.05, 0.95, f"$R^2$={res['r2']:.3f}\nRMSE={res['rmse']:.1f} km\nMAE={res['mae']:.1f} km",
            transform=ax.transAxes, va="top", fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.tick_params(labelsize=12)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontweight("bold")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"[Plot] {path}")


def plot_shap(imp, path):
    items = sorted(imp.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    wrap = {"Intensity & Inner-core": "Intensity &\nInner-core",
            "Large-scale Environment": "Large-scale\nEnvironment",
            "Location & Motion": "Location &\nMotion"}
    disp = [wrap.get(n, n) for n in names]             # 2-line horizontal labels
    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(disp, vals, color=[GROUP_COLORS[n] for n in names],
                  edgecolor="#6b6b6b", lw=0.9, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean |SHAP| [km]", fontsize=16, fontweight="bold")
    ax.set_xlabel("Feature Group", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.tick_params(labelsize=12)
    for lb in ax.get_xticklabels():
        lb.set_fontweight("bold")
    for lb in ax.get_yticklabels():
        lb.set_fontweight("bold")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"[Plot] {path}")


# ----------------------------------------------------------------------------
def run(tag, path, is_1h, test_sids, device):
    print(f"\n{'=' * 60}\nEXPERIMENT: {tag}\n{'=' * 60}")
    set_seed()  # identical, independent RNG start per model (fair 6h-vs-1h compare)
    data = load_data(path, is_1h)
    absent = test_sids - set(data["sids"])
    if absent:  # guarantee both models are evaluated on the SAME test storms
        raise ValueError(f"[{tag}] {len(absent)} test storms dropped by the min_labeled "
                         f"filter -> test sets would differ: {sorted(absent)[:5]}")
    test = [i for i, s in enumerate(data["sids"]) if s in test_sids]
    trainval = [i for i, s in enumerate(data["sids"]) if s not in test_sids]
    print(f"[Split] trainval {len(trainval)} | test {len(test)} storms")

    Xs = scale(data["X"], data["m"], trainval)
    best = search_hyperparams(data, trainval, device)
    model = LSTMRegressor(len(ALL_INPUT_COLS), best["hidden_size"],
                          best["num_layers"], best["dropout"]).to(device)
    tl = DataLoader(MaskedDataset([Xs[i] for i in trainval], [data["y"][i] for i in trainval],
                                  [data["m"][i] for i in trainval]),
                    best["batch_size"], shuffle=True, collate_fn=collate)
    train_model(model, tl, device)

    res = evaluate(model, Xs, data, test, device)
    imp = grouped_shap(model, Xs, data, test, trainval, device)
    res["df"].assign(model=tag).to_csv(os.path.join(OUTPUT_DIR, f"predictions_{tag}.csv"), index=False)
    pd.DataFrame({"group": imp.keys(), "mean_abs_shap": imp.values()}).to_csv(
        os.path.join(OUTPUT_DIR, f"shap_{tag}.csv"), index=False
    )
    plot_scatter(res, os.path.join(OUTPUT_DIR, f"scatter_{tag}.png"),
                 color=SCATTER_COLORS.get(tag, "#4C72B0"))
    plot_shap(imp, os.path.join(OUTPUT_DIR, f"shap_{tag}.png"))
    return res


def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Shared valid test storms (both files share the same SIDs) -> identical 6h test points.
    counts = pd.read_csv(DATA_6H, usecols=["SID"]).groupby("SID").size()
    sids = sorted(counts[counts >= 3].index)
    test_sids = set(train_test_split(sids, test_size=TEST_RATIO, random_state=SEED)[1])

    results = {}
    results["6h"] = run("6h", DATA_6H, False, test_sids, device)
    results["1h"] = run("1h", DATA_1H, True, test_sids, device)

    print(f"\n{'=' * 60}\nSUMMARY (same 6h test storms)\n{'=' * 60}")
    for tag, r in results.items():
        print(f"  {tag:4s} R2={r['r2']:.3f}  RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}")
    pd.DataFrame([
        {"model": tag, "r2": r["r2"], "rmse": r["rmse"], "mae": r["mae"], "n": len(r["df"])}
        for tag, r in results.items()
    ]).to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
    d = results["1h"]["rmse"] - results["6h"]["rmse"]
    print(f"  1h vs 6h RMSE: {d:+.2f} km ({'1h better' if d < 0 else '6h better'})")


if __name__ == "__main__":
    main()
