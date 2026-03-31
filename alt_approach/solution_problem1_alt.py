"""
Problem 1 (Alt): 7-farmer / 46-index classification.

Step 1: Identify 7 farmers via residual variance gap.
Step 2: For each of the 46 indices, find weights via FFT-domain NNLS:
        - Impute data (linear interpolation)
        - FFT each column
        - NNLS in frequency domain to find weights
        - Prune small coefficients
        - Validate on raw observed data
Step 3: Report weights and RMSE.
"""

import pandas as pd
import numpy as np
from scipy.optimize import nnls
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('limestone_data_challenge_2026.data.csv')
cols = [c for c in df.columns if c != 'time']
data = df[cols]
proxy = data.interpolate(method='linear', axis=0, limit_direction='both').ffill().bfill()
n_rows = len(df)

# ============================================================
# STEP 1: Identify 7 farmers via residual variance
# ============================================================
print("=" * 70)
print("STEP 1: Identify 7 farmers via residual variance")
print("=" * 70)

proxy_vals = proxy.values
market = proxy_vals.mean(axis=1)

res_vars = {}
for j, c in enumerate(cols):
    y = proxy_vals[:, j]
    x = market
    xm, ym = x.mean(), y.mean()
    slope = np.dot(x - xm, y - ym) / np.dot(x - xm, x - xm)
    intercept = ym - slope * xm
    residual = y - (slope * x + intercept)
    res_vars[c] = np.var(residual)

sorted_cols = sorted(res_vars.items(), key=lambda x: -x[1])

farmer_cols = [c for c, v in sorted_cols[:7]]
index_cols = sorted([c for c, v in sorted_cols[7:]])
gap_val = sorted_cols[6][1] - sorted_cols[7][1]

print(f"\n  7 FARMERS (by residual variance):")
for i, (c, v) in enumerate(sorted_cols[:7]):
    print(f"    {i+1}. {c}: {v:.1f}")
print(f"\n  Gap to next: {gap_val:.1f}")
print(f"  46 INDICES: {index_cols[:5]}... (and {len(index_cols)-5} more)")

# ============================================================
# STEP 2: FFT-domain NNLS for each index
# ============================================================
print(f"\n{'='*70}")
print("STEP 2: FFT-domain NNLS weight estimation")
print("=" * 70)

# Compute FFT of each farmer column (on proxy/interpolated data)
farmer_ffts = {}
for f in farmer_cols:
    farmer_ffts[f] = np.fft.rfft(proxy[f].values)

n_freq = len(farmer_ffts[farmer_cols[0]])

# Build frequency-domain design matrix: stack real and imaginary parts
# For NNLS, we need real-valued inputs, so we split complex FFT into real + imag
X_freq_parts = []
for f in farmer_cols:
    fft_f = farmer_ffts[f]
    X_freq_parts.append(np.concatenate([fft_f.real, fft_f.imag]))

X_freq = np.column_stack(X_freq_parts)  # shape: (2*n_freq, 7)


def nnls_r2_raw(target, predictors, min_rows=10):
    """NNLS on raw observed data only."""
    needed = [target] + list(predictors)
    mask = data[needed].notna().all(axis=1)
    n = mask.sum()
    if n < min_rows:
        return -1, None, n, 999
    X = data.loc[mask, list(predictors)].values
    y = data.loc[mask, target].values
    w, _ = nnls(X, y)
    pred = X @ w
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    return r2, w, n, rmse


index_definitions = {}
results_table = []

for idx_col in index_cols:
    # FFT of the index column
    idx_fft = np.fft.rfft(proxy[idx_col].values)
    y_freq = np.concatenate([idx_fft.real, idx_fft.imag])

    # NNLS in frequency domain
    w_fft, _ = nnls(X_freq, y_freq)

    # FFT-domain R2
    pred_freq = X_freq @ w_fft
    ss_res_f = np.sum((y_freq - pred_freq) ** 2)
    ss_tot_f = np.sum((y_freq - y_freq.mean()) ** 2)
    r2_fft = 1 - ss_res_f / ss_tot_f if ss_tot_f > 0 else 0

    # Get candidates (coef > 0.005)
    fft_candidates = {}
    for fi, f in enumerate(farmer_cols):
        if w_fft[fi] > 0.005:
            fft_candidates[f] = float(w_fft[fi])

    n_fft_cand = len(fft_candidates)

    # Validate on raw data using FFT-found candidates
    if fft_candidates:
        cand_list = list(fft_candidates.keys())
        r2_raw, w_raw, n_raw, rmse_raw = nnls_r2_raw(idx_col, cand_list)

        if r2_raw >= 0 and w_raw is not None:
            coef_dict = {}
            for f, wv in zip(cand_list, w_raw):
                if wv > 1e-5:
                    coef_dict[f] = float(wv)
        else:
            coef_dict = fft_candidates
            rmse_raw = 999
    else:
        coef_dict = {}
        r2_raw = -1
        n_raw = 0
        rmse_raw = 999

    # Also try greedy approach and pick the better result
    # Best pair search among 7 farmers
    best_pair_r2 = -1
    best_pair = None
    for f1, f2 in combinations(farmer_cols, 2):
        r2, w, n, _ = nnls_r2_raw(idx_col, [f1, f2])
        if r2 > best_pair_r2:
            best_pair_r2 = r2
            best_pair = [f1, f2]

    # Greedy expansion
    current_set = list(best_pair)
    current_r2 = best_pair_r2
    for _ in range(5):
        best_add = None
        best_add_r2 = current_r2
        for f in farmer_cols:
            if f in current_set:
                continue
            trial = current_set + [f]
            r2, w, n, _ = nnls_r2_raw(idx_col, trial)
            if r2 > best_add_r2 + 0.001:
                best_add_r2 = r2
                best_add = f
        if best_add is None:
            break
        current_set.append(best_add)
        current_r2 = best_add_r2

    r2_greedy, w_greedy, n_greedy, rmse_greedy = nnls_r2_raw(idx_col, current_set, min_rows=5)

    greedy_dict = {}
    if w_greedy is not None:
        for f, wv in zip(current_set, w_greedy):
            if wv > 1e-5:
                greedy_dict[f] = float(wv)

    # Pick the method with better raw R2
    if r2_raw >= r2_greedy:
        method = "FFT"
        final_dict = coef_dict
        final_r2 = r2_raw
        final_n = n_raw
        final_rmse = rmse_raw
    else:
        method = "Greedy"
        final_dict = greedy_dict
        final_r2 = r2_greedy
        final_n = n_greedy
        final_rmse = rmse_greedy

    index_definitions[idx_col] = final_dict
    results_table.append((idx_col, len(final_dict), final_r2, final_n, final_rmse,
                          r2_fft, n_fft_cand, method, final_dict))

# ============================================================
# STEP 3: Report results
# ============================================================
print(f"\n{'='*70}")
print("RESULTS: Weights and RMSE for all 46 indices")
print("=" * 70)

print(f"\n{'Col':<8} {'n_c':>4} {'R2_raw':>8} {'RMSE':>8} {'n_row':>6} {'R2_fft':>8} {'fft_c':>6} {'Method':<7}  Weights")
print("-" * 110)

for idx_col, nc, r2, nr, rmse, r2f, nfc, meth, cd in results_table:
    cs = sum(cd.values()) if cd else 0
    wt_str = ", ".join(f"{f}: {w:.4f}" for f, w in sorted(cd.items(), key=lambda x: -x[1]))
    print(f"{idx_col:<8} {nc:>4} {r2:>8.4f} {rmse:>8.2f} {nr:>6} {r2f:>8.6f} {nfc:>6} {meth:<7}  {wt_str}")

# Summary stats
r2_vals = [r[2] for r in results_table if r[2] >= 0]
rmse_vals = [r[4] for r in results_table if r[4] < 999]
fft_r2_vals = [r[5] for r in results_table]
fft_wins = sum(1 for r in results_table if r[7] == "FFT")
greedy_wins = sum(1 for r in results_table if r[7] == "Greedy")

print(f"\n  Raw R2:   mean={np.mean(r2_vals):.4f}, min={np.min(r2_vals):.4f}, max={np.max(r2_vals):.4f}")
print(f"  RMSE:     mean={np.mean(rmse_vals):.2f}, min={np.min(rmse_vals):.2f}, max={np.max(rmse_vals):.2f}")
print(f"  FFT R2:   mean={np.mean(fft_r2_vals):.6f}, min={np.min(fft_r2_vals):.6f}, max={np.max(fft_r2_vals):.6f}")
print(f"  Method wins: FFT={fft_wins}, Greedy={greedy_wins}")

# Submission
rows = []
for idx_col in sorted(index_definitions):
    for f, w in sorted(index_definitions[idx_col].items()):
        rows.append({"index_col": idx_col, "constituent_col": f, "coef": round(w, 6)})

submission_df = pd.DataFrame(rows)
submission_df.to_csv("alt_approach/problem1_alt_submission.csv", index=False)
print(f"\nSaved {len(rows)} rows to alt_approach/problem1_alt_submission.csv")
