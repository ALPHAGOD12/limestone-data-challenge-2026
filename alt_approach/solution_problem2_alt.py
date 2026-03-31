"""
Problem 2 (Alt): Fill all missing NaN values using 7-farmer classification.

Same pipeline as original problem 2, but with 7 farmers / 46 indices
from the residual variance approach. Weights found via best of
FFT-domain NNLS and greedy pair-search on raw data.
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

farmer_cols = ['col_24', 'col_31', 'col_52', 'col_32', 'col_34', 'col_12', 'col_49']

index_coefs = {
    'col_00': {'col_12': 0.4137, 'col_32': 0.3713, 'col_52': 0.2087},
    'col_01': {'col_32': 0.3956, 'col_12': 0.3668, 'col_52': 0.2408},
    'col_02': {'col_32': 0.4146, 'col_12': 0.3125, 'col_24': 0.1502, 'col_52': 0.1191},
    'col_03': {'col_12': 0.4023, 'col_32': 0.3915, 'col_52': 0.2044},
    'col_04': {'col_12': 0.4414, 'col_52': 0.2917, 'col_31': 0.2729},
    'col_05': {'col_49': 0.4019, 'col_12': 0.2398, 'col_32': 0.1854, 'col_52': 0.1729},
    'col_06': {'col_12': 0.3677, 'col_32': 0.3218, 'col_31': 0.2068, 'col_52': 0.099},
    'col_07': {'col_12': 0.3817, 'col_32': 0.3053, 'col_24': 0.2088, 'col_52': 0.097},
    'col_08': {'col_12': 0.4398, 'col_31': 0.2885, 'col_52': 0.2632},
    'col_09': {'col_32': 0.258, 'col_49': 0.2262, 'col_24': 0.2046, 'col_31': 0.1768, 'col_52': 0.1336},
    'col_10': {'col_49': 0.3891, 'col_34': 0.2248, 'col_12': 0.167, 'col_31': 0.1167, 'col_32': 0.1162},
    'col_11': {'col_32': 0.3628, 'col_52': 0.2289, 'col_24': 0.227, 'col_31': 0.1735},
    'col_13': {'col_49': 0.3197, 'col_12': 0.2416, 'col_34': 0.2136, 'col_32': 0.1781, 'col_52': 0.0502},
    'col_14': {'col_12': 0.3383, 'col_32': 0.2712, 'col_52': 0.2072, 'col_31': 0.1792},
    'col_15': {'col_49': 0.2756, 'col_32': 0.2702, 'col_12': 0.2257, 'col_34': 0.1141, 'col_52': 0.0655, 'col_31': 0.0401},
    'col_16': {'col_49': 0.3066, 'col_31': 0.197, 'col_24': 0.1745, 'col_32': 0.1415, 'col_34': 0.1227, 'col_52': 0.0543},
    'col_17': {'col_49': 0.3196, 'col_32': 0.2078, 'col_34': 0.1928, 'col_31': 0.1384, 'col_24': 0.1382},
    'col_18': {'col_12': 0.4035, 'col_32': 0.3792, 'col_52': 0.209},
    'col_19': {'col_49': 0.3555, 'col_32': 0.2481, 'col_34': 0.1901, 'col_31': 0.1616, 'col_24': 0.0414},
    'col_20': {'col_49': 0.3365, 'col_32': 0.2555, 'col_34': 0.2258, 'col_31': 0.109, 'col_52': 0.0699},
    'col_21': {'col_49': 0.3013, 'col_32': 0.2769, 'col_34': 0.2707, 'col_31': 0.1404},
    'col_22': {'col_32': 0.3271, 'col_24': 0.2922, 'col_52': 0.2053, 'col_31': 0.1808},
    'col_23': {'col_34': 0.2695, 'col_32': 0.2352, 'col_49': 0.2002, 'col_31': 0.1926, 'col_24': 0.1144},
    'col_25': {'col_49': 0.2341, 'col_31': 0.2333, 'col_32': 0.2305, 'col_24': 0.1602, 'col_52': 0.1429, 'col_12': 0.0017},
    'col_26': {'col_12': 0.4281, 'col_31': 0.3031, 'col_52': 0.2618},
    'col_27': {'col_12': 0.3341, 'col_32': 0.3289, 'col_52': 0.1848, 'col_31': 0.1497},
    'col_28': {'col_32': 0.4227, 'col_12': 0.3646, 'col_52': 0.2078},
    'col_29': {'col_32': 0.3496, 'col_49': 0.2864, 'col_24': 0.1798, 'col_31': 0.1723},
    'col_30': {'col_32': 0.2867, 'col_24': 0.2491, 'col_52': 0.2386, 'col_31': 0.2169},
    'col_33': {'col_49': 0.2641, 'col_12': 0.2294, 'col_31': 0.197, 'col_32': 0.1608, 'col_34': 0.133, 'col_24': 0.0163},
    'col_35': {'col_49': 0.3872, 'col_32': 0.2532, 'col_34': 0.1386, 'col_31': 0.1235, 'col_24': 0.0997},
    'col_36': {'col_12': 0.398, 'col_32': 0.3949, 'col_52': 0.2029},
    'col_37': {'col_12': 0.5009, 'col_31': 0.2655, 'col_52': 0.225},
    'col_38': {'col_32': 0.363, 'col_31': 0.2199, 'col_52': 0.2089, 'col_24': 0.2025},
    'col_39': {'col_12': 0.4758, 'col_52': 0.3094, 'col_31': 0.2102},
    'col_40': {'col_49': 0.2821, 'col_31': 0.233, 'col_34': 0.1454, 'col_24': 0.1219, 'col_32': 0.0962, 'col_12': 0.0691, 'col_52': 0.0505},
    'col_41': {'col_12': 0.4313, 'col_31': 0.2955, 'col_52': 0.2715},
    'col_42': {'col_32': 0.2744, 'col_12': 0.2099, 'col_52': 0.1871, 'col_31': 0.1806, 'col_24': 0.1368},
    'col_43': {'col_12': 0.4206, 'col_32': 0.3861, 'col_52': 0.1933},
    'col_44': {'col_49': 0.2391, 'col_31': 0.2381, 'col_32': 0.1932, 'col_24': 0.1748, 'col_52': 0.1467},
    'col_45': {'col_12': 0.4143, 'col_31': 0.314, 'col_52': 0.2705},
    'col_46': {'col_34': 0.3377, 'col_32': 0.2285, 'col_49': 0.1619, 'col_24': 0.1155, 'col_31': 0.09, 'col_52': 0.0654},
    'col_47': {'col_49': 0.3651, 'col_12': 0.2718, 'col_32': 0.1706, 'col_31': 0.1008, 'col_52': 0.0851},
    'col_48': {'col_12': 0.3917, 'col_32': 0.3719, 'col_52': 0.2335},
    'col_50': {'col_32': 0.5315, 'col_12': 0.2892, 'col_52': 0.175},
    'col_51': {'col_12': 0.4584, 'col_31': 0.2822, 'col_52': 0.2537},
}
index_cols = list(index_coefs.keys())
print(f"\n7 Farmers: {farmer_cols}")
print(f"46 Indices: {len(index_cols)} columns with best-of FFT/Greedy coefficients")

MIN_COEF_FOR_REVERSE = 0.05
LONG_GAP_THRESHOLD = 5

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

for idx_col, coefs in index_coefs.items():
    j = cols.index(idx_col)
    val = np.zeros(n_rows)
    for fc, c in coefs.items():
        val += c * final_imp[:, cols.index(fc)]
    for i in range(n_rows):
        if nan_mask[i, j]:
            final_imp[i, j] = val[i]

# ============================================================
# STEP 7: Final checks and save
# ============================================================
print("\n" + "="*60)
print("STEP 7: Verify and save")
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

result_df.to_csv('alt_approach/imputed_dataset_alt.csv', index=False)
print(f"\nSaved to alt_approach/imputed_dataset_alt.csv ({result_df.shape})")
