"""
Problem 3: Buy 100kg of flour only from NaN-price columns, minimizing total cost.

Strategy (BlendV2):
  Decompose price = market_level + farmer_deviation.
  - Market level: estimated from today's observed prices (trivial, precise).
  - Farmer deviation: blended from temporal + cross-sectional signals.

  For each NaN column, predict its price:
    Priority 1: Index formula (exact, RMSE ~1)
    Priority 2: Blend of temporal deviation (recent history of this column)
                and cross-sectional deviation (today's observed deviations
                of correlated columns). Weighting adapts by gap length:
                short gap → temporal dominant, long gap → cross-sectional dominant.

  Go greedy: all 100kg on the predicted cheapest NaN column.

  Validated: 4.10% excess over oracle, 4.27% savings vs uniform.
  Improvement over pairwise regression: ~2% total cost reduction.
"""

import pandas as pd
import numpy as np

# ============================================================
# SETUP: runs once at import time
# ============================================================

_df = pd.read_csv('limestone_data_challenge_2026.data.csv')
_cols = [c for c in _df.columns if c != 'time']
_data = _df[_cols].values
_nan_mask = np.isnan(_data)
_n_rows, _n_cols = _data.shape
_col_to_idx = {c: i for i, c in enumerate(_cols)}

INDEX_COEFS = {
    'col_11': {'col_28': 0.4744, 'col_20': 0.2563, 'col_26': 0.1980, 'col_07': 0.0712},
    'col_30': {'col_18': 0.4199, 'col_34': 0.1642, 'col_06': 0.1285,
               'col_19': 0.1076, 'col_40': 0.1041, 'col_03': 0.0752, 'col_01': 0.0050},
    'col_42': {'col_26': 0.6489, 'col_18': 0.1777, 'col_12': 0.1243, 'col_52': 0.0449},
    'col_46': {'col_15': 0.3479, 'col_34': 0.2442, 'col_09': 0.1473,
               'col_32': 0.1310, 'col_23': 0.1280},
    'col_48': {'col_05': 0.5892, 'col_23': 0.1936, 'col_45': 0.1396, 'col_26': 0.0751},
    'col_50': {'col_26': 0.5608, 'col_28': 0.2221, 'col_32': 0.2178},
}

_col_means = np.nanmean(_data, axis=0)

# Pre-compute daily market levels and deviations
print("Problem 3: Computing market levels and deviations...")
_market_level = np.zeros(_n_rows)
for t in range(_n_rows):
    obs = _data[t, ~_nan_mask[t]]
    _market_level[t] = obs.mean() if len(obs) > 0 else np.nan

_deviation = np.full_like(_data, np.nan)
for t in range(_n_rows):
    for j in np.where(~_nan_mask[t])[0]:
        _deviation[t, j] = _data[t, j] - _market_level[t]

# Train deviation correlation models (R2 > 0.001 for max coverage)
print("Problem 3: Training deviation correlation models...")
_dev_models = {}
for i in range(_n_cols):
    models_i = []
    for j in range(_n_cols):
        if i == j:
            continue
        mask = (~np.isnan(_deviation[:, i])) & (~np.isnan(_deviation[:, j]))
        n = mask.sum()
        if n < 50:
            continue
        x = _deviation[mask, j]
        y = _deviation[mask, i]
        xm, ym = x.mean(), y.mean()
        var = np.dot(x - xm, x - xm)
        if var < 1e-10:
            continue
        slope = np.dot(x - xm, y - ym) / var
        intercept = ym - slope * xm
        ss_res = np.sum((y - slope * x - intercept)**2)
        ss_tot = np.sum((y - ym)**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        if r2 > 0.001:
            models_i.append((j, slope, intercept, r2))
    if models_i:
        _dev_models[i] = models_i

_total_dev_models = sum(len(v) for v in _dev_models.values())
print(f"  {_total_dev_models} deviation models across {len(_dev_models)} targets")

_MIN_COEF = 0.05


def trading_problem_3(row):
    """
    Trading strategy for Problem 3.

    Input: single row (pandas Series) with 'time' and col_00..col_52.
    Output: DataFrame with ['purchase_col', 'qty'], exactly 100kg from NaN columns.
    """
    nan_cols = []
    obs_cols = []
    row_values = np.zeros(_n_cols)

    for i, c in enumerate(_cols):
        val = row[c] if isinstance(row, (pd.Series, dict)) else row[c]
        if pd.isna(val):
            nan_cols.append(i)
        else:
            obs_cols.append(i)
            row_values[i] = val

    if not nan_cols:
        return pd.DataFrame({'purchase_col': [_cols[0]], 'qty': [100]})

    obs_set = set(obs_cols)
    market_today = row_values[obs_cols].mean() if obs_cols else np.nanmean(_col_means)
    obs_devs = {j: row_values[j] - market_today for j in obs_cols}

    # Figure out current time index for temporal lookback
    t = None
    if 'time' in row.index if isinstance(row, pd.Series) else 'time' in row:
        t_val = row['time']
        if not pd.isna(t_val):
            t = int(t_val)

    predictions = {}

    # Step 0: Reverse inference — solve NaN farmer prices from observed indices
    solved = {}
    for _ in range(3):
        new_solved = 0
        for idx_name, coefs in INDEX_COEFS.items():
            idx_j = _col_to_idx[idx_name]
            if idx_j not in obs_set and idx_j not in solved:
                continue
            idx_val = row_values[idx_j] if idx_j in obs_set else solved.get(idx_j)
            if idx_val is None:
                continue
            unknown = []
            known_sum = 0.0
            for f, c in coefs.items():
                fj = _col_to_idx[f]
                if fj in obs_set:
                    known_sum += c * row_values[fj]
                elif fj in solved:
                    known_sum += c * solved[fj]
                else:
                    unknown.append((fj, c))
            if len(unknown) == 1:
                fj, c = unknown[0]
                if c >= _MIN_COEF and fj not in solved:
                    val = (idx_val - known_sum) / c
                    if val > 0:
                        solved[fj] = val
                        new_solved += 1
        for idx_name, coefs in INDEX_COEFS.items():
            idx_j = _col_to_idx[idx_name]
            if idx_j in obs_set or idx_j in solved:
                continue
            all_known = True
            val = 0.0
            for f, c in coefs.items():
                fj = _col_to_idx[f]
                if fj in obs_set:
                    val += c * row_values[fj]
                elif fj in solved:
                    val += c * solved[fj]
                else:
                    all_known = False
                    break
            if all_known and val > 0:
                solved[idx_j] = val
                new_solved += 1
        if new_solved == 0:
            break

    nan_set = set(nan_cols)
    for ni in nan_cols:
        if ni in solved:
            predictions[ni] = solved[ni]
            continue
        cname = _cols[ni]

        # Priority 1: Index formula
        if cname in INDEX_COEFS:
            coefs = INDEX_COEFS[cname]
            fidxs = {_col_to_idx[f]: c for f, c in coefs.items()}
            if all(fi in obs_set for fi in fidxs):
                price = sum(c * row_values[fi] for fi, c in fidxs.items())
                if price > 0:
                    predictions[ni] = price
                    continue

        # Priority 2: Blended deviation prediction
        temporal_dev = None
        gap = 9999

        if t is not None and t > 0:
            prev = np.where(~_nan_mask[:t, ni])[0]
            if len(prev) > 0:
                gap = t - prev[-1]
                recent = prev[-3:]
                devs = []
                for rt in recent:
                    obs_rt = np.where(~_nan_mask[rt])[0]
                    if len(obs_rt) >= 5:
                        devs.append(_data[rt, ni] - _data[rt, obs_rt].mean())
                if devs:
                    temporal_dev = np.mean(devs)

        cross_dev = None
        if ni in _dev_models:
            dp, dw = [], []
            for pred_j, slope, intercept, r2 in _dev_models[ni]:
                if pred_j in obs_set:
                    dp.append(slope * obs_devs[pred_j] + intercept)
                    dw.append(r2)
            if len(dp) >= 2:
                w = np.array(dw)
                p = np.array(dp)
                cross_dev = np.dot(w, p) / w.sum()

        if temporal_dev is not None and cross_dev is not None:
            tw = max(0.77 ** gap, 0.05)
            predictions[ni] = market_today + tw * temporal_dev + (1 - tw) * cross_dev
        elif temporal_dev is not None:
            predictions[ni] = market_today + temporal_dev
        elif cross_dev is not None:
            predictions[ni] = market_today + cross_dev

    if not predictions:
        # Fallback: uniform across NaN columns
        qty_each = 100 // len(nan_cols)
        remainder = 100 - qty_each * len(nan_cols)
        trades = []
        for i, nc in enumerate(nan_cols):
            q = qty_each + (1 if i < remainder else 0)
            if q > 0:
                trades.append((_cols[nc], q))
        return pd.DataFrame(trades, columns=['purchase_col', 'qty'])

    # Greedy: all 100kg on predicted cheapest
    best_col = min(predictions, key=predictions.get)

    return pd.DataFrame({
        'purchase_col': [_cols[best_col]],
        'qty': [100]
    })


# ============================================================
# BACKTEST
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("BACKTEST: trading_problem_3 on all historical rows")
    print("="*60)

    df_imp = pd.read_csv('imputed_dataset.csv')
    imp_data = df_imp[_cols].values

    total_cost = 0
    oracle_cost = 0
    uniform_cost = 0
    n_traded = 0

    for t in range(_n_rows):
        row = _df.iloc[t]
        nan_idxs = np.where(_nan_mask[t])[0]
        if len(nan_idxs) == 0:
            continue

        result = trading_problem_3(row)
        n_traded += 1

        day_cost = 0
        for _, trade in result.iterrows():
            ci = _col_to_idx[trade['purchase_col']]
            day_cost += trade['qty'] * imp_data[t, ci]
        total_cost += day_cost

        nan_true = imp_data[t, nan_idxs]
        oracle_cost += nan_true.min() * 100
        uniform_cost += nan_true.mean() * 100

    print(f"\n  Rows traded: {n_traded}")
    print(f"\n  Total cost:    {total_cost:>14,.0f}")
    print(f"  Oracle cost:   {oracle_cost:>14,.0f}")
    print(f"  Uniform cost:  {uniform_cost:>14,.0f}")
    print(f"\n  Excess over oracle: {(total_cost/oracle_cost - 1)*100:.2f}%")
    print(f"  Savings vs uniform: {(1 - total_cost/uniform_cost)*100:.2f}%")
    print(f"\n  Avg cost/day (ours):    {total_cost/n_traded:.2f}")
    print(f"  Avg cost/day (oracle):  {oracle_cost/n_traded:.2f}")
    print(f"  Avg cost/day (uniform): {uniform_cost/n_traded:.2f}")
