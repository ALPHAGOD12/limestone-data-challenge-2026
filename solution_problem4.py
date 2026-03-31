"""
Problem 4: Arbitrage — buy from NaN columns, sell to index columns.

Strategy:
  Score = sum(qty × (dest_price - src_price)), maximize profit.
  - dest must be an index (observed or NaN), src must be NaN.
  - Up to 100kg total.

  Key insight: index price - cheapest NaN farmer ≈ 13 units on average,
  while prediction RMSE ≈ 5. The spread dwarfs the error.

  Approach:
  1. Reverse inference: use observed indices to exactly solve for NaN farmer
     prices (RMSE ~2.4, much better than BlendV2's ~5).
  2. For remaining NaN columns: BlendV2 (market + temporal/cross deviation).
  3. For dest indices: observed = exact price, NaN = formula prediction.
  4. Pick the (src, dest) pair with highest estimated profit.
  5. Put all 100kg on it.

  Validated: 4.1M profit, 66.3% of oracle (up from 58.6% without reverse inference).
"""

import pandas as pd
import numpy as np

# ============================================================
# SETUP (runs once at import time)
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
_index_idxs = [_col_to_idx[c] for c in INDEX_COEFS]
_col_means = np.nanmean(_data, axis=0)
_MIN_COEF = 0.05

# Compute market levels and deviations
print("Problem 4: Computing market levels and deviations...")
_market_level = np.zeros(_n_rows)
for t in range(_n_rows):
    obs = _data[t, ~_nan_mask[t]]
    _market_level[t] = obs.mean() if len(obs) > 0 else np.nan

_deviation = np.full_like(_data, np.nan)
for t in range(_n_rows):
    for j in np.where(~_nan_mask[t])[0]:
        _deviation[t, j] = _data[t, j] - _market_level[t]

# Train deviation correlation models
print("Problem 4: Training deviation correlation models...")
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

print(f"  {sum(len(v) for v in _dev_models.values())} deviation models ready")


def _predict_nan_price(ni, obs_cols, obs_set, obs_devs, market_today, row_values, t):
    """Predict a NaN column's price using BlendV2 (market + deviation blend)."""
    cname = _cols[ni]

    # Index formula
    if cname in INDEX_COEFS:
        coefs = INDEX_COEFS[cname]
        fidxs = {_col_to_idx[f]: c for f, c in coefs.items()}
        if all(fi in obs_set for fi in fidxs):
            price = sum(c * row_values[fi] for fi, c in fidxs.items())
            if price > 0:
                return price

    # Temporal deviation
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

    # Cross-sectional deviation
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

    # Blend
    if temporal_dev is not None and cross_dev is not None:
        tw = max(0.77 ** gap, 0.05)
        return market_today + tw * temporal_dev + (1 - tw) * cross_dev
    elif temporal_dev is not None:
        return market_today + temporal_dev
    elif cross_dev is not None:
        return market_today + cross_dev

    return _col_means[ni]


def trading_problem_4(row):
    """
    Arbitrage strategy for Problem 4.

    Input: single row (pandas Series) with 'time' and col_00..col_52.
    Output: DataFrame with ['src_col', 'dest_col', 'qty'].
            Buy from NaN src, sell to index dest. Up to 100kg.
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
        return pd.DataFrame({'src_col': [], 'dest_col': [], 'qty': []}).astype(
            {'src_col': str, 'dest_col': str, 'qty': int})

    obs_set = set(obs_cols)
    market_today = row_values[obs_cols].mean() if obs_cols else np.nanmean(_col_means)
    obs_devs = {j: row_values[j] - market_today for j in obs_cols}

    t = None
    if isinstance(row, pd.Series) and 'time' in row.index:
        t_val = row['time']
        if not pd.isna(t_val):
            t = int(t_val)
    elif isinstance(row, dict) and 'time' in row:
        t_val = row['time']
        if not pd.isna(t_val):
            t = int(t_val)

    nan_set = set(nan_cols)

    # Step 1: Reverse inference — solve NaN farmer prices from observed indices
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

    # Step 2: Predict all NaN column prices (reverse inference overrides BlendV2)
    nan_prices = {}
    for ni in nan_cols:
        if ni in solved:
            nan_prices[ni] = solved[ni]
        else:
            nan_prices[ni] = _predict_nan_price(ni, list(obs_cols), obs_set, obs_devs,
                                                 market_today, row_values, t)

    # Get dest (index) prices: observed = exact, NaN = predicted
    dest_prices = {}
    for j in _index_idxs:
        if j in obs_set:
            dest_prices[j] = (row_values[j], 'observed')
        elif j in nan_prices:
            dest_prices[j] = (nan_prices[j], 'predicted')

    if not dest_prices:
        return pd.DataFrame({'src_col': [], 'dest_col': [], 'qty': []}).astype(
            {'src_col': str, 'dest_col': str, 'qty': int})

    # Find best (src, dest) pair
    best_profit = 0
    best_src = None
    best_dest = None

    for dest_j, (dest_price, dest_type) in dest_prices.items():
        for src_j in nan_cols:
            if src_j == dest_j:
                continue
            src_price = nan_prices.get(src_j, _col_means[src_j])
            est_profit = dest_price - src_price

            if est_profit > best_profit:
                best_profit = est_profit
                best_src = src_j
                best_dest = dest_j

    if best_src is None or best_profit <= 0:
        return pd.DataFrame({'src_col': [], 'dest_col': [], 'qty': []}).astype(
            {'src_col': str, 'dest_col': str, 'qty': int})

    return pd.DataFrame({
        'src_col': [_cols[best_src]],
        'dest_col': [_cols[best_dest]],
        'qty': [100]
    })


# ============================================================
# BACKTEST
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("BACKTEST: trading_problem_4 on all historical rows")
    print("="*60)

    df_imp = pd.read_csv('imputed_dataset.csv')
    imp_data = df_imp[_cols].values

    total_profit = 0
    oracle_profit = 0
    n_traded = 0
    n_loss = 0
    n_skip = 0
    dest_types = {'observed': 0, 'predicted': 0}

    for t in range(_n_rows):
        row = _df.iloc[t]
        nan_idxs = np.where(_nan_mask[t])[0]
        if len(nan_idxs) == 0:
            n_skip += 1
            continue

        result = trading_problem_4(row)

        if len(result) == 0:
            n_skip += 1
        else:
            n_traded += 1
            for _, trade in result.iterrows():
                src_j = _col_to_idx[trade['src_col']]
                dest_j = _col_to_idx[trade['dest_col']]
                qty = trade['qty']
                true_profit = qty * (imp_data[t, dest_j] - imp_data[t, src_j])
                total_profit += true_profit
                if true_profit < 0:
                    n_loss += 1

                if not _nan_mask[t, dest_j]:
                    dest_types['observed'] += 1
                else:
                    dest_types['predicted'] += 1

        # Oracle
        best_oracle = 0
        for dj in _index_idxs:
            for sj in nan_idxs:
                if sj == dj: continue
                p = imp_data[t, dj] - imp_data[t, sj]
                if p > best_oracle:
                    best_oracle = p
        oracle_profit += best_oracle * 100

    print(f"\n  Days traded: {n_traded}, Days skipped: {n_skip}")
    print(f"  Loss days: {n_loss}")
    print(f"  Dest types: {dest_types}")
    print(f"\n  Total profit:  {total_profit:>14,.0f}")
    print(f"  Oracle profit: {oracle_profit:>14,.0f}")
    print(f"  Capture rate:  {total_profit/oracle_profit*100:.1f}%")
    print(f"\n  Avg profit/day (ours):   {total_profit/n_traded:.2f}")
    print(f"  Avg profit/day (oracle): {oracle_profit/_n_rows:.2f}")
