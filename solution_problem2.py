"""
Problem 2: Fill all missing NaN values in the dataset.

Pipeline:
  0.  Reverse inference: algebraically solve for missing farmers from observed
      index prices (iterative, up to 3 passes).
  1.  Remove market level (row-wise mean of observed) to get deviations.
      Compute per-column mean deviation and per-cell gap lengths.
  2.  Build temporal features on deviations + normalized time as KNN feature.
  3.  KNN ensemble (k=2,3,5,7) on deviations.
  4.  Add market level back. Override long-gap (>5 rows) cells with
      market + column mean deviation (KNN unreliable for long gaps).
  5.  Preserve original observed values and Step 0 algebraic fills.
  6.  Reconstruct index NaN values from known coefficients.
  7.  Post-KNN reverse inference (same logic as Step 0, on denser data).
  8.  Force col_34/col_52 to only use: observed, index-derived, or
      row median from the dense post-KNN matrix. Re-reconstruct affected indices.
  9.  Final safety net, clip, save.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

df = pd.read_csv('limestone_data_challenge_2026.data.csv')
cols = [c for c in df.columns if c != 'time']
data = df[cols].values.copy()
nan_mask = np.isnan(data)
n_rows, n_cols = data.shape

print(f"Dataset: {n_rows} rows x {n_cols} columns")
print(f"NaN fraction: {nan_mask.mean():.4f}")

index_coefs = {
    'col_11': {'col_28': 0.4744, 'col_20': 0.2563, 'col_26': 0.1980, 'col_07': 0.0712},
    'col_30': {'col_18': 0.4199, 'col_34': 0.1642, 'col_06': 0.1285,
               'col_19': 0.1076, 'col_40': 0.1041, 'col_03': 0.0752, 'col_01': 0.0050},
    'col_42': {'col_26': 0.6489, 'col_18': 0.1777, 'col_12': 0.1243, 'col_52': 0.0449},
    'col_46': {'col_15': 0.3479, 'col_34': 0.2442, 'col_09': 0.1473,
               'col_32': 0.1310, 'col_23': 0.1280},
    'col_48': {'col_05': 0.5892, 'col_23': 0.1936, 'col_45': 0.1396, 'col_26': 0.0751},
    'col_50': {'col_26': 0.5608, 'col_28': 0.2221, 'col_32': 0.2178},
}
index_cols = list(index_coefs.keys())
farmer_cols = sorted([c for c in cols if c not in index_cols])

MIN_COEF_FOR_REVERSE = 0.05
LONG_GAP_THRESHOLD = 5
MEDIAN_FILL_COLS = ['col_52', 'col_34']

# ============================================================
# STEP 0: Pre-KNN Reverse Inference
# ============================================================
print("\n" + "="*60)
print("STEP 0: Pre-KNN reverse inference from observed indices")
print("="*60)

pre_data = data.copy()
pre_nan = nan_mask.copy()
pre_adj = 0

for _ in range(3):
    new_adj = 0
    for idx_col, coefs in index_coefs.items():
        j_idx = cols.index(idx_col)
        farmer_info = [(cols.index(fc), c) for fc, c in coefs.items()]
        for i in range(n_rows):
            if pre_nan[i, j_idx]:
                continue
            obs_idx_val = pre_data[i, j_idx]
            nan_f = [(j_f, w) for j_f, w in farmer_info if pre_nan[i, j_f]]
            obs_f = [(j_f, w) for j_f, w in farmer_info if not pre_nan[i, j_f]]
            if len(nan_f) != 1:
                continue
            j_f, w_f = nan_f[0]
            if w_f < MIN_COEF_FOR_REVERSE:
                continue
            known_part = sum(w * pre_data[i, jf] for jf, w in obs_f)
            new_val = (obs_idx_val - known_part) / w_f
            if new_val > 0:
                col_vals = data[:, j_f]
                col_mean = np.nanmean(col_vals)
                col_std = np.nanstd(col_vals)
                if abs(new_val - col_mean) < 4 * col_std:
                    pre_data[i, j_f] = new_val
                    pre_nan[i, j_f] = False
                    new_adj += 1
    pre_adj += new_adj
    if new_adj == 0:
        break

for idx_col, coefs in index_coefs.items():
    j = cols.index(idx_col)
    farmer_info = [(cols.index(fc), c) for fc, c in coefs.items()]
    for i in range(n_rows):
        if not pre_nan[i, j]:
            continue
        if all(not pre_nan[i, jf] for jf, _ in farmer_info):
            val = sum(c * pre_data[i, jf] for jf, c in farmer_info)
            if val > 0:
                pre_data[i, j] = val
                pre_nan[i, j] = False

pre_filled_total = nan_mask.sum() - pre_nan.sum()
print(f"  Farmer adjustments: {pre_adj}")
print(f"  Total NaNs filled (farmers + indices): {pre_filled_total}")
print(f"  NaN fraction: {nan_mask.mean():.4f} -> {pre_nan.mean():.4f}")

# ============================================================
# STEP 1: Remove market level + compute gap lengths
# ============================================================
print("\n" + "="*60)
print("STEP 1: Remove market level + compute gap lengths")
print("="*60)

market_level = np.zeros(n_rows)
for i in range(n_rows):
    obs_vals = pre_data[i, ~pre_nan[i, :]]
    market_level[i] = np.mean(obs_vals) if len(obs_vals) > 0 else np.nan

ml_filled = pd.Series(market_level).interpolate(
    method='linear', limit_direction='both').ffill().bfill().values
print(f"  Market level range: [{ml_filled.min():.2f}, {ml_filled.max():.2f}]")

deviations = np.full_like(pre_data, np.nan)
for i in range(n_rows):
    for j in range(n_cols):
        if not pre_nan[i, j]:
            deviations[i, j] = pre_data[i, j] - ml_filled[i]

col_mean_dev = np.array([np.nanmean(deviations[:, j]) for j in range(n_cols)])

gap_matrix = np.zeros((n_rows, n_cols), dtype=int)
for j in range(n_cols):
    last_obs = -9999
    for i in range(n_rows):
        if not pre_nan[i, j]:
            last_obs = i
            gap_matrix[i, j] = 0
        else:
            gap_matrix[i, j] = i - last_obs if last_obs >= 0 else 9999

nan_gaps = gap_matrix[pre_nan]
short_gap = np.sum(nan_gaps <= LONG_GAP_THRESHOLD)
long_gap = np.sum(nan_gaps > LONG_GAP_THRESHOLD)
print(f"  Short-gap NaNs (<=5): {short_gap} ({100*short_gap/len(nan_gaps):.1f}%)")
print(f"  Long-gap NaNs  (>5):  {long_gap} ({100*long_gap/len(nan_gaps):.1f}%)")

# ============================================================
# STEP 2: Temporal features + time feature
# ============================================================
print("\n" + "="*60)
print("STEP 2: Temporal features + time feature on deviations")
print("="*60)

def add_temporal_features(td, windows):
    nr, nc = td.shape
    extras = []
    for w in windows:
        extra = np.full((nr, nc), np.nan)
        for i in range(nr):
            lo, hi = max(0, i - w), min(nr, i + w + 1)
            nearby = list(range(lo, i)) + list(range(i + 1, hi))
            if nearby:
                extra[i] = np.nanmean(td[nearby], axis=0)
        extras.append(extra)
    return np.column_stack([td] + extras)

windows = [1, 2, 3, 5, 7]
combined = add_temporal_features(deviations, windows)
time_feature = (np.arange(n_rows, dtype=float) / n_rows) * 6.4
combined = np.column_stack([combined, time_feature.reshape(-1, 1)])
print(f"  Feature matrix: {deviations.shape} -> {combined.shape}")

# ============================================================
# STEP 3: KNN Ensemble on deviations
# ============================================================
print("\n" + "="*60)
print("STEP 3: KNN ensemble imputation (k=2,3,5,7)")
print("="*60)

k_values = [2, 3, 5, 7]
imps = []
for k in k_values:
    print(f"  Running KNN k={k}...")
    knn = KNNImputer(n_neighbors=k)
    imp = knn.fit_transform(combined)[:, :n_cols]
    imps.append(imp)

dev_imputed = np.mean(imps, axis=0)

# ============================================================
# STEP 4: Add market level back + long-gap fallback
# ============================================================
print("\n" + "="*60)
print("STEP 4: Add market level back + long-gap fallback")
print("="*60)

final_imp = dev_imputed + ml_filled[:, None]

n_overridden = 0
for i in range(n_rows):
    for j in range(n_cols):
        if pre_nan[i, j] and gap_matrix[i, j] > LONG_GAP_THRESHOLD:
            final_imp[i, j] = ml_filled[i] + col_mean_dev[j]
            n_overridden += 1
print(f"  Long-gap overrides: {n_overridden}")

# ============================================================
# STEP 5: Preserve originals + Step 0 algebraic fills
# ============================================================
final_imp[~nan_mask] = data[~nan_mask]
step0_filled = nan_mask & ~pre_nan
final_imp[step0_filled] = pre_data[step0_filled]
print(f"  Preserved originals + {step0_filled.sum()} Step 0 fills")

# ============================================================
# STEP 6: Reconstruct index columns from coefficients
# ============================================================
print("\n" + "="*60)
print("STEP 6: Reconstruct indices + post-KNN reverse inference")
print("="*60)

for idx_col, coefs in index_coefs.items():
    j = cols.index(idx_col)
    val = np.zeros(n_rows)
    for fc, c in coefs.items():
        val += c * final_imp[:, cols.index(fc)]
    n_filled = 0
    for i in range(n_rows):
        if nan_mask[i, j]:
            final_imp[i, j] = val[i]
            n_filled += 1
    print(f"  {idx_col}: reconstructed {n_filled} NaN values")

# Post-KNN reverse inference
adj = 0
reverse_inferred = set()
for idx_col, coefs in index_coefs.items():
    j_idx = cols.index(idx_col)
    farmer_info = [(cols.index(fc), c) for fc, c in coefs.items()]
    for i in range(n_rows):
        if nan_mask[i, j_idx]:
            continue
        obs_idx_val = data[i, j_idx]
        imp_farmers = [(j_f, w) for j_f, w in farmer_info if nan_mask[i, j_f]]
        obs_farmers = [(j_f, w) for j_f, w in farmer_info if not nan_mask[i, j_f]]
        if len(imp_farmers) != 1:
            continue
        j_f, w_f = imp_farmers[0]
        if w_f < MIN_COEF_FOR_REVERSE:
            continue
        known_part = sum(w * data[i, jf] for jf, w in obs_farmers)
        new_val = (obs_idx_val - known_part) / w_f
        if new_val > 0:
            col_vals = data[:, j_f]
            col_mean = np.nanmean(col_vals)
            col_std = np.nanstd(col_vals)
            if abs(new_val - col_mean) < 4 * col_std:
                final_imp[i, j_f] = new_val
                reverse_inferred.add((i, j_f))
                adj += 1
print(f"  Post-KNN reverse inference: {adj} adjustments")

# Re-reconstruct indices after reverse inference
for idx_col, coefs in index_coefs.items():
    j = cols.index(idx_col)
    val = np.zeros(n_rows)
    for fc, c in coefs.items():
        val += c * final_imp[:, cols.index(fc)]
    for i in range(n_rows):
        if nan_mask[i, j]:
            final_imp[i, j] = val[i]

# ============================================================
# STEP 7: Force col_34/col_52 — observed / index-derived / row median
# ============================================================
print("\n" + "="*60)
print("STEP 7: Force col_34/col_52 (observed / index-derived / row median)")
print("="*60)

median_col_indices = set(cols.index(mc) for mc in MEDIAN_FILL_COLS)
other_cols_mask = np.array([j not in median_col_indices for j in range(n_cols)])

for mc in MEDIAN_FILL_COLS:
    j_mc = cols.index(mc)
    n_obs, n_rev, n_alg, n_med = 0, 0, 0, 0
    for i in range(n_rows):
        if not nan_mask[i, j_mc]:
            final_imp[i, j_mc] = data[i, j_mc]
            n_obs += 1
        elif (i, j_mc) in reverse_inferred:
            n_rev += 1
        elif nan_mask[i, j_mc] and not pre_nan[i, j_mc]:
            final_imp[i, j_mc] = pre_data[i, j_mc]
            n_alg += 1
        else:
            final_imp[i, j_mc] = np.median(final_imp[i, other_cols_mask])
            n_med += 1
    print(f"  {mc}: obs={n_obs}, algebraic={n_alg}, reverse={n_rev}, median={n_med}")

for idx_col, coefs in index_coefs.items():
    if not any(fc in MEDIAN_FILL_COLS for fc in coefs):
        continue
    j = cols.index(idx_col)
    val = np.zeros(n_rows)
    for fc, c in coefs.items():
        val += c * final_imp[:, cols.index(fc)]
    for i in range(n_rows):
        if nan_mask[i, j]:
            final_imp[i, j] = val[i]
    print(f"  Re-reconstructed {idx_col}")

# ============================================================
# STEP 8: Final checks and save
# ============================================================
print("\n" + "="*60)
print("STEP 8: Verify and save")
print("="*60)

remaining_nans = np.isnan(final_imp).sum()
if remaining_nans > 0:
    col_means = np.nanmean(final_imp, axis=0)
    for j in range(n_cols):
        mask_j = np.isnan(final_imp[:, j])
        if mask_j.any():
            final_imp[mask_j, j] = col_means[j]
    print(f"  Fallback filled {remaining_nans} remaining NaNs with column means")

final_imp = np.clip(final_imp, 0, None)

max_diff = np.max(np.abs(data[~nan_mask] - final_imp[~nan_mask]))
print(f"  Max diff on observed: {max_diff:.10f}")
print(f"  Imputed: mean={final_imp[nan_mask].mean():.2f}, std={final_imp[nan_mask].std():.2f}")
print(f"  Observed: mean={data[~nan_mask].mean():.2f}, std={data[~nan_mask].std():.2f}")
print(f"  Range: [{final_imp.min():.2f}, {final_imp.max():.2f}]")

result_df = df.copy()
for j, col in enumerate(cols):
    result_df[col] = final_imp[:, j]

result_df.to_csv('imputed_dataset.csv', index=False)
print(f"\nSaved to imputed_dataset.csv ({result_df.shape})")
 