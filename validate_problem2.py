"""
Comprehensive validation of Problem 2 imputation.
Multiple tests to assess quality from different angles.
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('limestone_data_challenge_2026.data.csv')
cols = [c for c in df.columns if c != 'time']
data = df[cols].values.copy()
nan_mask = np.isnan(data)
n_rows, n_cols = data.shape

imputed = pd.read_csv('imputed_dataset.csv')
imp_data = imputed[cols].values

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

# ============================================================
# TEST 1: Observed values preserved exactly
# ============================================================
print("="*70)
print("TEST 1: Observed values preserved")
print("="*70)
obs_orig = data[~nan_mask]
obs_imp = imp_data[~nan_mask]
max_diff = np.max(np.abs(obs_orig - obs_imp))
print(f"  Max absolute diff: {max_diff:.12f}")
print(f"  PASS" if max_diff < 1e-8 else f"  FAIL")

# ============================================================
# TEST 2: No NaNs remaining
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: No NaN remaining")
print("="*70)
remaining = np.isnan(imp_data).sum()
print(f"  Remaining NaNs: {remaining}")
print(f"  PASS" if remaining == 0 else f"  FAIL")

# ============================================================
# TEST 3: All values non-negative
# ============================================================
print(f"\n{'='*70}")
print("TEST 3: All values non-negative")
print("="*70)
neg_count = (imp_data < 0).sum()
min_val = imp_data.min()
print(f"  Negative values: {neg_count}")
print(f"  Min value: {min_val:.4f}")
print(f"  PASS" if neg_count == 0 else f"  FAIL")

# ============================================================
# TEST 4: Index reconstruction accuracy
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: Index reconstruction accuracy")
print("="*70)
print("="*70)
print("  (Check if imputed index = sum(coef * imputed farmer))\n")
for idx_col, coefs in index_coefs.items():
    j = cols.index(idx_col)
    reconstructed = np.zeros(n_rows)
    for fc, c in coefs.items():
        reconstructed += c * imp_data[:, cols.index(fc)]
    
    diff = imp_data[:, j] - reconstructed
    rmse = np.sqrt(np.mean(diff**2))
    max_d = np.max(np.abs(diff))
    
    # Only check on originally NaN rows (those were reconstructed)
    nan_rows = nan_mask[:, j]
    diff_nan = imp_data[nan_rows, j] - reconstructed[nan_rows]
    rmse_nan = np.sqrt(np.mean(diff_nan**2)) if nan_rows.sum() > 0 else 0
    
    # Check on observed rows (these should differ due to noise)
    obs_rows = ~nan_mask[:, j]
    diff_obs = imp_data[obs_rows, j] - reconstructed[obs_rows]
    rmse_obs = np.sqrt(np.mean(diff_obs**2)) if obs_rows.sum() > 0 else 0
    
    print(f"  {idx_col}: NaN rows RMSE={rmse_nan:.6f}, Obs rows RMSE={rmse_obs:.4f}, Max|diff|={max_d:.4f}")

# ============================================================
# TEST 5: Random masking test (10% of observed values)
# ============================================================
print(f"\n{'='*70}")
print("TEST 5: Random masking — mask 10% of observed, re-impute, check RMSE")
print("="*70)

np.random.seed(123)

def add_temporal_features(td, windows):
    nr, nc = td.shape
    extras = []
    for w in windows:
        extra = np.full((nr, nc), np.nan)
        for i in range(nr):
            lo, hi = max(0, i - w), min(nr, i + w + 1)
            nearby = list(range(lo, i)) + list(range(i + 1, hi))
            if nearby:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    extra[i] = np.nanmean(td[nearby], axis=0)
        extras.append(extra)
    return np.column_stack([td] + extras)

obs_indices = np.argwhere(~nan_mask)
n_mask = len(obs_indices) // 10
mask_idx = obs_indices[np.random.choice(len(obs_indices), n_mask, replace=False)]

masked_data = data.copy()
true_vals = np.array([data[r, c] for r, c in mask_idx])
for r, c in mask_idx:
    masked_data[r, c] = np.nan

# Market-level detrend, then KNN on deviations (matches solution_problem2 logic)
ml = np.array([np.nanmean(masked_data[i, :]) if np.any(~np.isnan(masked_data[i, :]))
               else np.nan for i in range(n_rows)])
ml = pd.Series(ml).interpolate(method='linear', limit_direction='both').ffill().bfill().values
devs = masked_data - ml[:, None]
combined_m = add_temporal_features(devs, [1, 2, 3, 5, 7])
time_feat = (np.arange(n_rows, dtype=float) / n_rows) * 6.4
combined_m = np.column_stack([combined_m, time_feat.reshape(-1, 1)])
knn = KNNImputer(n_neighbors=5)
imp_devs = knn.fit_transform(combined_m)[:, :n_cols]
imp_m = imp_devs + ml[:, None]
pred_vals = np.array([imp_m[r, c] for r, c in mask_idx])

rmse_overall = np.sqrt(np.mean((pred_vals - true_vals)**2))
mae_overall = np.mean(np.abs(pred_vals - true_vals))
mape = np.mean(np.abs(pred_vals - true_vals) / true_vals) * 100

print(f"  Masked cells: {n_mask}")
print(f"  Overall RMSE: {rmse_overall:.4f}")
print(f"  Overall MAE:  {mae_overall:.4f}")
print(f"  Overall MAPE: {mape:.2f}%")

# Per-column breakdown for random sample
col_rmses = {}
for j in range(n_cols):
    col_mask = mask_idx[:, 1] == j
    if col_mask.sum() < 5:
        continue
    tv = true_vals[col_mask]
    pv = pred_vals[col_mask]
    col_rmses[cols[j]] = np.sqrt(np.mean((tv - pv)**2))

sorted_rmse = sorted(col_rmses.items(), key=lambda x: x[1])
print(f"\n  Per-column RMSE (best 10 / worst 10):")
print(f"  Best 10:")
for c, r in sorted_rmse[:10]:
    tag = " [INDEX]" if c in index_cols else ""
    print(f"    {c}: {r:.4f}{tag}")
print(f"  Worst 10:")
for c, r in sorted_rmse[-10:]:
    tag = " [INDEX]" if c in index_cols else ""
    print(f"    {c}: {r:.4f}{tag}")

# ============================================================
# TEST 6: Block masking (long gaps) on multiple columns
# ============================================================
print(f"\n{'='*70}")
print("TEST 6: Block masking — 100-day gaps on 10 farmer columns")
print("="*70)

np.random.seed(42)
test_farmers = np.random.choice(farmer_cols, size=10, replace=False)
block_results = []

for tc in test_farmers:
    j_tc = cols.index(tc)
    obs_rows = np.where(~nan_mask[:, j_tc])[0]
    if len(obs_rows) < 300:
        continue
    start = np.random.randint(len(obs_rows)//4, len(obs_rows)*3//4 - 100)
    block_rows = obs_rows[start:start+100]
    true_block = data[block_rows, j_tc]
    
    masked_d = data.copy()
    masked_d[block_rows, j_tc] = np.nan
    
    # Linear baseline
    s = pd.Series(masked_d[:, j_tc])
    s_lin = s.interpolate(method='linear', limit_direction='both').ffill().bfill()
    e_lin = np.sqrt(np.mean((s_lin.values[block_rows] - true_block)**2))
    
    # Re-impute with our method on masked data
    ml_b = np.array([np.nanmean(masked_d[i, :]) if np.any(~np.isnan(masked_d[i, :]))
                      else np.nan for i in range(n_rows)])
    ml_b = pd.Series(ml_b).interpolate(method='linear', limit_direction='both').ffill().bfill().values
    devs_b = masked_d - ml_b[:, None]
    comb_b = add_temporal_features(devs_b, [1, 2, 3, 5, 7])
    tf = (np.arange(n_rows, dtype=float) / n_rows) * 6.4
    comb_b = np.column_stack([comb_b, tf.reshape(-1, 1)])
    knn_b = KNNImputer(n_neighbors=5)
    imp_devs_b = knn_b.fit_transform(comb_b)[:, :n_cols]
    imp_b = imp_devs_b + ml_b[:, None]

    # Long-gap fallback for this column
    col_mean_dev_b = np.nanmean(devs_b[:, j_tc])
    gap_col = np.zeros(n_rows, dtype=int)
    last_obs = -9999
    for ii in range(n_rows):
        if not np.isnan(masked_d[ii, j_tc]):
            last_obs = ii
            gap_col[ii] = 0
        else:
            gap_col[ii] = ii - last_obs if last_obs >= 0 else 9999
    for ii in block_rows:
        if gap_col[ii] > 5:
            imp_b[ii, j_tc] = ml_b[ii] + col_mean_dev_b
    
    e_imp = np.sqrt(np.mean((imp_b[block_rows, j_tc] - true_block)**2))
    
    imp_pct = (1 - e_imp/e_lin) * 100
    block_results.append((tc, e_lin, e_imp, imp_pct))
    print(f"  {tc}: Linear={e_lin:.3f}, Imputed={e_imp:.3f}, improvement={imp_pct:+.1f}%")

avg_lin = np.mean([r[1] for r in block_results])
avg_imp = np.mean([r[2] for r in block_results])
print(f"\n  Average: Linear={avg_lin:.3f}, Imputed={avg_imp:.3f}, imp={(1-avg_imp/avg_lin)*100:+.1f}%")

# ============================================================
# TEST 7: Distribution comparison (imputed vs observed)
# ============================================================
print(f"\n{'='*70}")
print("TEST 7: Distribution comparison per column")
print("="*70)
print(f"  {'Column':<8} | {'Obs Mean':>9} | {'Imp Mean':>9} | {'Diff':>7} | {'Obs Std':>8} | {'Imp Std':>8} | {'Diff':>7}")
print(f"  " + "-"*70)

big_diffs = []
for j, col in enumerate(cols):
    obs_vals = data[~nan_mask[:, j], j]
    imp_vals = imp_data[nan_mask[:, j], j]
    if len(obs_vals) < 10 or len(imp_vals) < 10:
        continue
    om, im = obs_vals.mean(), imp_vals.mean()
    os, is_ = obs_vals.std(), imp_vals.std()
    md = abs(om - im)
    sd = abs(os - is_)
    tag = " *" if md > 5 else ""
    print(f"  {col:<8} | {om:>9.2f} | {im:>9.2f} | {md:>6.2f}{tag} | {os:>8.2f} | {is_:>8.2f} | {sd:>6.2f}")
    if md > 5:
        big_diffs.append(col)

if big_diffs:
    print(f"\n  * Columns with mean diff > 5: {big_diffs}")
else:
    print(f"\n  All columns have mean diff <= 5. Good.")

# ============================================================
# TEST 8: Convex bound check on imputed data
# ============================================================
print(f"\n{'='*70}")
print("TEST 8: Convex bounds on imputed indices")
print("="*70)
print("  (For all rows: min(constituents) <= index <= max(constituents))\n")

for idx_col, coefs in index_coefs.items():
    j = cols.index(idx_col)
    const_js = [cols.index(fc) for fc in coefs.keys()]
    
    idx_vals = imp_data[:, j]
    const_vals = imp_data[:, const_js]
    row_mins = const_vals.min(axis=1)
    row_maxs = const_vals.max(axis=1)
    
    below = idx_vals < row_mins
    above = idx_vals > row_maxs
    violations = below | above
    
    below_tol = idx_vals < (row_mins - 1.0)
    above_tol = idx_vals > (row_maxs + 1.0)
    viol_tol = below_tol | above_tol
    
    max_v = 0
    if below.any():
        max_v = max(max_v, np.max(row_mins[below] - idx_vals[below]))
    if above.any():
        max_v = max(max_v, np.max(idx_vals[above] - row_maxs[above]))
    
    print(f"  {idx_col}: {violations.sum():>4} violations / {n_rows} rows ({100*violations.mean():.1f}%), "
          f"tol>1: {viol_tol.sum():>3}, max violation: {max_v:.2f}")
