# Hurricane wind radius (R34) — 6h vs 1h LSTM with grouped SHAP

Predicts hurricane 34-kt wind radius (R34) at 6-hour best-track points and compares
two models, then attributes each model's skill to three physics-based feature groups
via grouped SHAP.

- **6h model** (`r34_6h.csv`): every timestep is a labeled best-track observation.
- **1h model** (`r34_1h.csv`): the full 1-hour sequence is fed to the LSTM, but
  the **loss and evaluation are computed only at the 6h points** (`raw_data==True`). The
  5 intermediate hourly ERA5 steps provide sub-6h context. The 6-hourly best-track
  observations are fed only at the 6h points — intermediate steps are masked to zero
  and flagged by an `obs_flag` channel — while sporadic hourly-ERA5 gaps (~0.1%) are
  linearly interpolated within each storm. Standardization statistics come only from
  real values (best-track obs from the 6h points, ERA5 from all hours; `obs_flag`
  stays 0/1). Both models are evaluated on the **same** valid 6h test storms.

## Feature groups (for grouped SHAP)

50 features in three physics groups (**8 / 24 / 18** features; the grouping was
recovered from the Excel header colours). A separate `obs_flag` channel — the 51st
model input — gates the masked obs and is in **no** SHAP group. Group colours below
match the SHAP bar charts (colour-blind-safe):

| Group | Meaning |
|-------|---------|
| 🟪 Location & Motion | DIST2LAND, USA_LAT/LON, STORM_SPEED/DIR, era_min_lat/lon, distance_dev |
| 🟧 Intensity & Inner-core | USA_WIND/PRES/SSHS, uv_max, rmax*, u850/v850, warm_core, era_min_pressure + `*_mslp_center` twins |
| 🟩 Large-scale Environment | u200/v200, shear, rh500 + `*_mslp_center` twins |

SHAP is computed **per group** (exact permutation Shapley over all 3!=6 orderings),
because storm-center and `*_mslp_center` features are ~0.99 correlated and per-feature
SHAP would split their importance. Each group is revealed as a whole — from a baseline
(the per-feature mean over labelled 6h steps) to its real values — in every ordering, and
its averaged marginal change in predicted R34 is the group's Shapley value; importance is
the mean |SHAP| in km at the 6h test points. Only forward passes are used, so it applies
directly to the masked LSTM.

## Run

Everything is in a single self-contained script. **Edit only the `CONFIG` block at the
top** (the three data/output paths) — nothing else.

```bash
pip install -r requirements.txt        # torch, numpy, pandas, scikit-learn, matplotlib
python r34_6h_vs_1h.py
```

On Colab:
```python
!pip install torch numpy pandas scikit-learn matplotlib
# set DATA_6H / DATA_1H / OUTPUT_DIR in the CONFIG block to your Drive paths, then:
!python r34_6h_vs_1h.py
```

## Outputs (`outputs/`)

| File | Description |
|------|-------------|
| `scatter_6h.png`, `scatter_1h.png` | Observed vs predicted R34 (with R²/RMSE/MAE) |
| `shap_6h.png`, `shap_1h.png` | Grouped-SHAP importance of the three feature groups |
| `predictions_6h.csv`, `predictions_1h.csv` | Test-storm predictions at labeled 6h points |
| `shap_6h.csv`, `shap_1h.csv` | Grouped-SHAP values used in the bar plots |
| `metrics_summary.csv` | R²/RMSE/MAE comparison table |

A comparison summary (R²/RMSE/MAE for both models) is printed to the console.
