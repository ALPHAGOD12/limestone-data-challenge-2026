"""
Problem 5: Limit order trading — buy from NaN columns with price + quantity.

Rules:
- Each trade is a buy order on a NaN column with a price (px) and quantity (qty).
- Order is filled only if px >= true hidden price.
- Up to 100kg total. Non-negative integer sizes.
- Score: sum(qty × (median_price - px) × I{px >= true_price})
  median_price = true row-wise median of ALL columns.
- Maximize score.

Key insight: the bid price px enters the score directly (NOT true_price).
Lower bid = more profit per fill, but fewer fills. Higher bid = more fills, less profit.
Optimal margin ≈ +1 above predicted price (balances fill rate and profit).

Strategy:
1. Predict all NaN prices using BlendV2 (RI + market deviation blend).
2. Pick the single cheapest predicted NaN column.
3. Bid = predicted_price + 1 (small margin for fill safety).
4. Put all 100kg on it.
"""

import pandas as pd
import numpy as np

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
_MIN_COEF = 0.05
_BID_MARGIN = 0.0

print("Problem 5: Computing market levels and deviations...")
_market_level = np.zeros(_n_rows)
for t in range(_n_rows):
    obs = _data[t, ~_nan_mask[t]]
    _market_level[t] = obs.mean() if len(obs) > 0 else np.nan

_deviation = np.full_like(_data, np.nan)
for t in range(_n_rows):
    for j in np.where(~_nan_mask[t])[0]:
        _deviation[t, j] = _data[t, j] - _market_level[t]

print("Problem 5: Training deviation correlation models...")
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

print(f"  {sum(len(v) for v in _dev_models.values())} deviation models across {len(_dev_models)} targets")


def _predict_nan_price(ni, obs_set, obs_devs, market_today, row_values, t):
    """Predict a NaN column's price using BlendV2."""
    cname = _cols[ni]

    if cname in INDEX_COEFS:
        coefs = INDEX_COEFS[cname]
        fidxs = {_col_to_idx[f]: c for f, c in coefs.items()}
        if all(fi in obs_set for fi in fidxs):
            price = sum(c * row_values[fi] for fi, c in fidxs.items())
            if price > 0:
                return price

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
        return market_today + tw * temporal_dev + (1 - tw) * cross_dev
    elif temporal_dev is not None:
        return market_today + temporal_dev
    elif cross_dev is not None:
        return market_today + cross_dev

    return _col_means[ni]


def trading_problem_5(row):
    """
    Limit order trading strategy for Problem 5.

    Input: pandas Series with 'time' and col_00..col_52.
    Output: DataFrame with columns [order_col, px, qty].
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
        return pd.DataFrame({'order_col': [], 'px': [], 'qty': []})

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

    # Reverse inference
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

    # Predict all NaN prices
    predictions = {}
    for ni in nan_cols:
        if ni in solved:
            predictions[ni] = solved[ni]
        else:
            predictions[ni] = _predict_nan_price(ni, obs_set, obs_devs,
                                                  market_today, row_values, t)

    if not predictions:
        return pd.DataFrame({'order_col': [], 'px': [], 'qty': []})

    # Pick cheapest predicted
    best_col = min(predictions, key=predictions.get)
    pred_price = predictions[best_col]
    bid = round(pred_price + _BID_MARGIN, 2)

    return pd.DataFrame({
        'order_col': [_cols[best_col]],
        'px': [bid],
        'qty': [100]
    })


# ============================================================
# BACKTEST
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("BACKTEST: trading_problem_5 on all historical rows")
    print("=" * 60)

    df_imp = pd.read_csv('imputed_dataset.csv')
    imp_data = df_imp[_cols].values

    total_score = 0
    oracle_score = 0
    n_traded = 0
    n_fills = 0
    n_neg = 0
    n_orders = 0

    for t in range(_n_rows):
        row = _df.iloc[t]
        nan_idxs = np.where(_nan_mask[t])[0]
        if len(nan_idxs) == 0:
            continue

        n_traded += 1
        result = trading_problem_5(row)

        true_median = np.median(imp_data[t])

        for _, trade in result.iterrows():
            col_j = _col_to_idx[trade['order_col']]
            true_price = imp_data[t, col_j]
            bid_price = trade['px']
            qty = trade['qty']
            n_orders += 1

            if bid_price >= true_price:
                score = qty * (true_median - bid_price)
                total_score += score
                n_fills += 1
                if score < 0:
                    n_neg += 1

        nan_true = imp_data[t, nan_idxs]
        cheapest_true = nan_true.min()
        oracle_score += 100 * (true_median - cheapest_true)

    print(f"\n  Days traded: {n_traded}")
    print(f"  Total orders: {n_orders}")
    print(f"  Orders filled: {n_fills} ({n_fills / max(n_orders, 1) * 100:.1f}%)")
    print(f"  Negative score fills: {n_neg}")
    print(f"\n  Total score (ours):   {total_score:>14,.0f}")
    print(f"  Total score (oracle):  {oracle_score:>14,.0f}")
    print(f"  Efficiency: {total_score / max(oracle_score, 1) * 100:.1f}%")
    print(f"\n  Avg score/day (ours):  {total_score / n_traded:.2f}")
    print(f"  Avg score/day (oracle): {oracle_score / n_traded:.2f}")
