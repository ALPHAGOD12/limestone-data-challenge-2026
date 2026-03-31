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

index_cols = ['col_11', 'col_30', 'col_42', 'col_46', 'col_48', 'col_50']
farmer_cols = sorted([c for c in cols if c not in index_cols])

def nnls_r2_raw(target, predictors, min_rows=10):
    needed = [target] + list(predictors)
    mask = data[needed].notna().all(axis=1)
    n = mask.sum()
    if n < min_rows:
        return -1, None, n
    X = data.loc[mask, list(predictors)].values
    y = data.loc[mask, target].values
    w, _ = nnls(X, y)
    pred = X @ w
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return r2, w, n


print("PROBLEM 1: Index Classification and Coefficient Estimation")
print("=" * 70)
print(f"\nClassification:")
print(f"  6 Indices: {', '.join(index_cols)}")
print(f"  47 Farmers: all other columns")

# Strategy per index:
# 1) Exhaustive pair search for all indices
# 2) Greedy expansion from best pair
# 3) At each step, keep the solution that maximizes raw_R2
#    stopping when improvement < 0.001

index_definitions = {}

for idx_col in index_cols:
    print(f"\n{'='*60}")
    print(f"  {idx_col}")
    print(f"{'='*60}")
    
    # Step 1: Find best pair
    best_pair_r2 = -1
    best_pair = None
    best_pair_w = None
    for f1, f2 in combinations(farmer_cols, 2):
        r2, w, n = nnls_r2_raw(idx_col, [f1, f2])
        if r2 > best_pair_r2:
            best_pair_r2 = r2
            best_pair = [f1, f2]
            best_pair_w = w
    
    _, w_check, n_check = nnls_r2_raw(idx_col, best_pair, min_rows=5)
    print(f"  Best pair: {best_pair}, R2={best_pair_r2:.6f}, n={n_check}")
    
    # Step 2: Greedy expansion
    current_set = list(best_pair)
    current_r2 = best_pair_r2
    
    # Track the best solution at each size
    solutions = []
    r2_c, w_c, n_c = nnls_r2_raw(idx_col, current_set, min_rows=5)
    solutions.append((list(current_set), r2_c, w_c, n_c))
    
    for round_num in range(8):
        best_add = None
        best_add_r2 = current_r2
        
        for f in farmer_cols:
            if f in current_set:
                continue
            trial = current_set + [f]
            r2, w, n = nnls_r2_raw(idx_col, trial)
            if r2 > best_add_r2 + 0.001:
                best_add_r2 = r2
                best_add = f
        
        if best_add is None:
            break
        
        current_set.append(best_add)
        current_r2 = best_add_r2
        r2_c, w_c, n_c = nnls_r2_raw(idx_col, current_set, min_rows=5)
        solutions.append((list(current_set), r2_c, w_c, n_c))
        print(f"  + {best_add}: R2={r2_c:.6f}, n={n_c}")
    
    # Pick the best solution: highest raw_R2 with at least 15 rows
    # If no solution has 15+ rows, just take highest raw_R2
    valid = [(s, r2, w, n) for s, r2, w, n in solutions if n >= 15]
    if not valid:
        valid = solutions
    
    best_sol = max(valid, key=lambda x: x[1])
    final_set, final_r2, final_w, final_n = best_sol
    
    coef_dict = {}
    for f, wv in zip(final_set, final_w):
        if wv > 1e-5:
            coef_dict[f] = float(wv)
    
    index_definitions[idx_col] = coef_dict
    
    # Proxy validation
    yp = proxy[idx_col].values
    fl = list(coef_dict.keys())
    w_arr = np.array([coef_dict[f] for f in fl])
    pred_p = proxy[fl].values @ w_arr
    r2p = 1 - np.sum((yp - pred_p)**2) / np.sum((yp - yp.mean())**2)
    
    print(f"\n  SELECTED ({len(coef_dict)} constituents):")
    print(f"    raw_R2={final_r2:.6f}, proxy_R2={r2p:.6f}, n={final_n}, sum={sum(coef_dict.values()):.4f}")
    for f, wv in sorted(coef_dict.items(), key=lambda x: -x[1]):
        print(f"      {f}: {wv:.6f}")


# SUBMISSION
print(f"\n{'='*70}")
print("SUBMISSION")
print("="*70)
rows = []
for idx_col in sorted(index_definitions):
    for f, w in sorted(index_definitions[idx_col].items()):
        rows.append({"index_col": idx_col, "constituent_col": f, "coef": round(w, 6)})

submission_df = pd.DataFrame(rows)
print(f"\n{submission_df.to_string(index=False)}")
submission_df.to_csv("problem1_submission_v3.csv", index=False)
print(f"\nSaved {len(rows)} rows to problem1_submission_v3.csv")

# FINAL VALIDATION
print(f"\n{'='*70}")
print("FINAL VALIDATION")
print("="*70)
for idx_col, cd in index_definitions.items():
    fl = list(cd.keys())
    w = np.array([cd[f] for f in fl])
    
    yp = proxy[idx_col].values
    pred_p = proxy[fl].values @ w
    r2p = 1 - np.sum((yp - pred_p)**2) / np.sum((yp - yp.mean())**2)
    
    mask = data[[idx_col] + fl].notna().all(axis=1)
    n_raw = mask.sum()
    if n_raw > 0:
        yr = data.loc[mask, idx_col].values
        pred_r = data.loc[mask, fl].values @ w
        r2r = 1 - np.sum((yr - pred_r)**2) / np.sum((yr - yr.mean())**2)
    else:
        r2r = float('nan')
    
    print(f"  {idx_col}: raw_R2={r2r:.6f} (n={n_raw}), proxy_R2={r2p:.6f}, "
          f"coef_sum={sum(cd.values()):.4f}, n_const={len(cd)}")
