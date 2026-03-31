"""
Problem 4 (Alt): Arbitrage — buy from NaN columns, sell to index columns.
Uses 7 farmers / 46 indices from residual variance approach.
"""

import pandas as pd
import numpy as np

_df = pd.read_csv('limestone_data_challenge_2026.data.csv')
_cols = [c for c in _df.columns if c != 'time']
_data = _df[_cols].values
_nan_mask = np.isnan(_data)
_n_rows, _n_cols = _data.shape
_col_to_idx = {c: i for i, c in enumerate(_cols)}

_farmer_cols = ['col_24', 'col_31', 'col_52', 'col_32', 'col_34', 'col_12', 'col_49']

INDEX_COEFS = {
    'col_00': {'col_31': 0.2982, 'col_32': 0.2267, 'col_49': 0.2142, 'col_24': 0.1162, 'col_12': 0.086, 'col_34': 0.0435, 'col_52': 0.0037},
    'col_01': {'col_49': 0.2973, 'col_34': 0.2019, 'col_32': 0.1771, 'col_31': 0.1735, 'col_12': 0.158},
    'col_02': {'col_32': 0.2852, 'col_12': 0.2305, 'col_24': 0.1598, 'col_31': 0.1513, 'col_49': 0.0947, 'col_34': 0.0507, 'col_52': 0.0262},
    'col_03': {'col_49': 0.2697, 'col_32': 0.2261, 'col_12': 0.165, 'col_31': 0.1629, 'col_34': 0.1305, 'col_24': 0.0482},
    'col_04': {'col_49': 0.4044, 'col_34': 0.1861, 'col_32': 0.1225, 'col_12': 0.1094, 'col_31': 0.1006, 'col_24': 0.0923},
    'col_05': {'col_49': 0.29, 'col_12': 0.2476, 'col_52': 0.1806, 'col_32': 0.1448, 'col_31': 0.1412, 'col_34': 0.008},
    'col_06': {'col_12': 0.3848, 'col_31': 0.2594, 'col_32': 0.2371, 'col_24': 0.0625, 'col_49': 0.0478},
    'col_07': {'col_12': 0.3667, 'col_32': 0.2035, 'col_31': 0.1917, 'col_49': 0.1447, 'col_24': 0.0788, 'col_34': 0.0049},
    'col_08': {'col_12': 0.2211, 'col_49': 0.199, 'col_31': 0.1885, 'col_34': 0.1837, 'col_24': 0.1407, 'col_32': 0.0677},
    'col_09': {'col_24': 0.2902, 'col_32': 0.2425, 'col_31': 0.1771, 'col_49': 0.1402, 'col_52': 0.1339, 'col_34': 0.009, 'col_12': 0.0054},
    'col_10': {'col_49': 0.3891, 'col_34': 0.2248, 'col_12': 0.167, 'col_31': 0.1167, 'col_32': 0.1162},
    'col_11': {'col_49': 0.238, 'col_32': 0.2152, 'col_12': 0.1582, 'col_34': 0.148, 'col_24': 0.1187, 'col_31': 0.0962, 'col_52': 0.0202},
    'col_13': {'col_49': 0.2991, 'col_12': 0.2239, 'col_31': 0.1601, 'col_32': 0.1202, 'col_34': 0.1029, 'col_24': 0.0934},
    'col_14': {'col_12': 0.358, 'col_49': 0.2429, 'col_31': 0.1985, 'col_32': 0.1091, 'col_24': 0.0445, 'col_52': 0.0364},
    'col_15': {'col_49': 0.2744, 'col_12': 0.2381, 'col_32': 0.2147, 'col_34': 0.1121, 'col_31': 0.0816, 'col_52': 0.0716},
    'col_16': {'col_31': 0.2571, 'col_49': 0.2513, 'col_24': 0.1888, 'col_32': 0.1688, 'col_12': 0.0964, 'col_34': 0.0339},
    'col_17': {'col_49': 0.2366, 'col_32': 0.2011, 'col_12': 0.1723, 'col_24': 0.1586, 'col_31': 0.1546, 'col_34': 0.0694, 'col_52': 0.0073},
    'col_18': {'col_49': 0.2446, 'col_34': 0.1865, 'col_32': 0.1631, 'col_12': 0.1464, 'col_31': 0.1351, 'col_24': 0.1237},
    'col_19': {'col_49': 0.3245, 'col_32': 0.208, 'col_31': 0.194, 'col_12': 0.1641, 'col_34': 0.1102},
    'col_20': {'col_32': 0.2502, 'col_49': 0.1779, 'col_34': 0.1706, 'col_24': 0.1693, 'col_12': 0.1157, 'col_31': 0.0922, 'col_52': 0.025},
    'col_21': {'col_32': 0.2124, 'col_34': 0.1953, 'col_12': 0.1738, 'col_31': 0.1668, 'col_49': 0.1316, 'col_24': 0.1214},
    'col_22': {'col_49': 0.373, 'col_24': 0.1974, 'col_32': 0.1869, 'col_31': 0.1277, 'col_34': 0.0655, 'col_52': 0.0399, 'col_12': 0.0145},
    'col_23': {'col_32': 0.4033, 'col_24': 0.2019, 'col_12': 0.1443, 'col_31': 0.0972, 'col_52': 0.0887, 'col_34': 0.0623, 'col_49': 0.0048},
    'col_25': {'col_32': 0.2682, 'col_49': 0.2546, 'col_31': 0.2417, 'col_34': 0.1158, 'col_24': 0.0909, 'col_52': 0.0341},
    'col_26': {'col_49': 0.2693, 'col_31': 0.1938, 'col_34': 0.1771, 'col_24': 0.1552, 'col_32': 0.1119, 'col_12': 0.0496, 'col_52': 0.0404},
    'col_27': {'col_49': 0.3749, 'col_31': 0.1866, 'col_12': 0.1775, 'col_32': 0.1192, 'col_34': 0.0794, 'col_24': 0.0651},
    'col_28': {'col_49': 0.3615, 'col_12': 0.178, 'col_34': 0.1212, 'col_32': 0.1116, 'col_52': 0.0947, 'col_24': 0.0715, 'col_31': 0.0565},
    'col_29': {'col_12': 0.2496, 'col_49': 0.2268, 'col_32': 0.1854, 'col_31': 0.1193, 'col_24': 0.1067, 'col_34': 0.1048},
    'col_30': {'col_34': 0.2342, 'col_49': 0.2174, 'col_31': 0.164, 'col_24': 0.1409, 'col_32': 0.1179, 'col_12': 0.114, 'col_52': 0.0117},
    'col_33': {'col_49': 0.2641, 'col_12': 0.2294, 'col_31': 0.197, 'col_32': 0.1608, 'col_34': 0.133, 'col_24': 0.0163},
    'col_35': {'col_49': 0.4331, 'col_31': 0.1498, 'col_52': 0.1368, 'col_32': 0.1079, 'col_34': 0.0961, 'col_12': 0.0885},
    'col_36': {'col_49': 0.2698, 'col_32': 0.2323, 'col_12': 0.2249, 'col_31': 0.1382, 'col_34': 0.137},
    'col_37': {'col_24': 0.1881, 'col_31': 0.1748, 'col_12': 0.1658, 'col_32': 0.1655, 'col_49': 0.1548, 'col_52': 0.1166, 'col_34': 0.0258},
    'col_38': {'col_49': 0.3553, 'col_12': 0.1793, 'col_32': 0.1762, 'col_34': 0.1002, 'col_31': 0.0839, 'col_52': 0.0608, 'col_24': 0.0379},
    'col_39': {'col_12': 0.2455, 'col_31': 0.196, 'col_32': 0.1899, 'col_49': 0.1844, 'col_24': 0.1294, 'col_34': 0.0635},
    'col_40': {'col_49': 0.2821, 'col_31': 0.233, 'col_34': 0.1454, 'col_24': 0.1219, 'col_32': 0.0962, 'col_12': 0.0691, 'col_52': 0.0505},
    'col_41': {'col_49': 0.3123, 'col_32': 0.2347, 'col_31': 0.1579, 'col_34': 0.1513, 'col_12': 0.1366},
    'col_42': {'col_32': 0.2226, 'col_49': 0.206, 'col_31': 0.1972, 'col_34': 0.1212, 'col_52': 0.1124, 'col_24': 0.1087, 'col_12': 0.0249},
    'col_43': {'col_12': 0.2697, 'col_49': 0.1978, 'col_34': 0.1646, 'col_31': 0.1067, 'col_32': 0.1024, 'col_24': 0.0866, 'col_52': 0.0774},
    'col_44': {'col_49': 0.1978, 'col_12': 0.1892, 'col_31': 0.1846, 'col_24': 0.1775, 'col_32': 0.1599, 'col_52': 0.0533, 'col_34': 0.0331},
    'col_45': {'col_31': 0.231, 'col_34': 0.2205, 'col_12': 0.1656, 'col_24': 0.1414, 'col_32': 0.1357, 'col_49': 0.1081},
    'col_46': {'col_34': 0.3336, 'col_32': 0.2044, 'col_12': 0.1568, 'col_24': 0.1213, 'col_49': 0.0979, 'col_31': 0.0688, 'col_52': 0.0169},
    'col_47': {'col_49': 0.3221, 'col_12': 0.2235, 'col_31': 0.1433, 'col_34': 0.1079, 'col_32': 0.0884, 'col_24': 0.0656, 'col_52': 0.0398},
    'col_48': {'col_12': 0.239, 'col_32': 0.1943, 'col_31': 0.1902, 'col_49': 0.1457, 'col_24': 0.143, 'col_34': 0.052, 'col_52': 0.0288},
    'col_50': {'col_32': 0.4025, 'col_49': 0.1347, 'col_24': 0.1291, 'col_31': 0.1258, 'col_12': 0.1222, 'col_34': 0.0798},
    'col_51': {'col_31': 0.2271, 'col_49': 0.2068, 'col_32': 0.1827, 'col_12': 0.1795, 'col_24': 0.154, 'col_34': 0.0451},
}
_index_idxs = [_col_to_idx[c] for c in INDEX_COEFS]
_col_means = np.nanmean(_data, axis=0)
_MIN_COEF = 0.05

print("Problem 4 Alt: Loading imputed data for deviation model training...")
_imp_df = pd.read_csv('alt_approach/imputed_dataset_alt.csv')
_imp_data_train = _imp_df[_cols].values

_market_level_imp = np.mean(_imp_data_train, axis=1)
_deviation_imp = _imp_data_train - _market_level_imp[:, None]

print("Problem 4 Alt: Training deviation correlation models on imputed data...")
_dev_models = {}
for i in range(_n_cols):
    models_i = []
    for j in range(_n_cols):
        if i == j:
            continue
        x = _deviation_imp[:, j]
        y = _deviation_imp[:, i]
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


def _predict_nan_price(ni, obs_set, obs_devs, market_today, row_values, t):
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


def trading_problem_4(row):
    nan_cols, obs_cols = [], []
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

    nan_prices = {}
    for ni in nan_cols:
        if ni in solved:
            nan_prices[ni] = solved[ni]
        else:
            nan_prices[ni] = _predict_nan_price(ni, obs_set, obs_devs,
                                                 market_today, row_values, t)

    dest_prices = {}
    for j in _index_idxs:
        if j in obs_set:
            dest_prices[j] = (row_values[j], 'observed')
        elif j in nan_prices:
            dest_prices[j] = (nan_prices[j], 'predicted')

    if not dest_prices:
        return pd.DataFrame({'src_col': [], 'dest_col': [], 'qty': []}).astype(
            {'src_col': str, 'dest_col': str, 'qty': int})

    best_profit = 0
    best_src = None
    best_dest = None
    for dest_j, (dest_price, _) in dest_prices.items():
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


if __name__ == '__main__':
    print("\n" + "="*60)
    print("BACKTEST: trading_problem_4 (ALT) on all historical rows")
    print("="*60)

    df_imp = pd.read_csv('alt_approach/imputed_dataset_alt.csv')
    imp_data = df_imp[_cols].values

    total_profit = 0
    oracle_profit = 0
    n_traded = 0
    n_loss = 0
    n_skip = 0

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

        best_oracle = 0
        for dj in _index_idxs:
            for sj in nan_idxs:
                if sj == dj:
                    continue
                p = imp_data[t, dj] - imp_data[t, sj]
                if p > best_oracle:
                    best_oracle = p
        oracle_profit += best_oracle * 100

    print(f"\n  Days traded: {n_traded}, Days skipped: {n_skip}")
    print(f"  Loss days: {n_loss}")
    print(f"\n  Total profit:  {total_profit:>14,.0f}")
    print(f"  Oracle profit: {oracle_profit:>14,.0f}")
    if oracle_profit > 0:
        print(f"  Capture rate:  {total_profit/oracle_profit*100:.1f}%")
    print(f"\n  Avg profit/day (ours):   {total_profit/max(n_traded,1):.2f}")
    print(f"  Avg profit/day (oracle): {oracle_profit/_n_rows:.2f}")
