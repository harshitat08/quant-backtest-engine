from __future__ import annotations
import itertools
import pandas as pd
import numpy as np
from backtest import Backtester, BacktestConfig
from metrics import sharpe_ratio, cagr, max_drawdown


def grid_search(price_df, strategy_fn, param_grid, config=None, target_metric="sharpe", verbose=True):
    config = config or BacktestConfig()
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    results = []
    total = len(combos)
    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        try:
            signal_df = strategy_fn(price_df, **params)
            bt = Backtester(price_df, signal_df, config)
            r = bt.run()
            eq = r["equity_curve"]
            mdd, _ = max_drawdown(eq)
            ann_ret = cagr(eq)
            calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan
            results.append({**params, "sharpe": sharpe_ratio(eq), "cagr": ann_ret, "max_dd": mdd, "calmar": calmar, "n_trades": len(r["trades"])})
            if verbose and i % max(1, total // 10) == 0:
                print(f"  [{i:4d}/{total}] {params}  Sharpe={sharpe_ratio(eq):.3f}")
        except Exception as e:
            if verbose:
                print(f"  SKIP {params}: {e}")
            continue
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values(target_metric, ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def walk_forward_optimize(price_df, strategy_fn, param_grid, train_pct=0.60, n_splits=5, target_metric="sharpe", config=None, verbose=True):
    config = config or BacktestConfig()
    n = len(price_df)
    window = n // n_splits
    oos_results = []
    oos_curves = []
    best_params_list = []
    for split in range(n_splits):
        start = split * window
        end = (split + 1) * window if split < n_splits - 1 else n
        split_df = price_df.iloc[start:end].copy()
        train_end = int(len(split_df) * train_pct)
        train_df = split_df.iloc[:train_end]
        test_df = split_df.iloc[train_end:]
        if len(test_df) < 30:
            continue
        gs_df = grid_search(train_df, strategy_fn, param_grid, config, target_metric=target_metric, verbose=False)
        if gs_df.empty:
            continue
        best = gs_df.iloc[0].to_dict()
        best_params = {k: best[k] for k in param_grid.keys()}
        best_params_list.append(best_params)
        signal_df = strategy_fn(test_df, **best_params)
        bt = Backtester(test_df, signal_df, config)
        r = bt.run()
        eq = r["equity_curve"]
        oos_curves.append(eq)
        mdd, _ = max_drawdown(eq)
        oos_results.append({
            "split": split + 1,
            "train_start": train_df.index[0].date(), "train_end": train_df.index[-1].date(),
            "test_start": test_df.index[0].date(), "test_end": test_df.index[-1].date(),
            **{f"best_{k}": v for k, v in best_params.items()},
            "is_sharpe": round(float(best.get("sharpe", 0)), 3),
            "oos_sharpe": round(sharpe_ratio(eq), 3),
            "oos_cagr_pct": round(cagr(eq) * 100, 2),
            "oos_max_dd": round(mdd * 100, 2),
        })
        if verbose:
            print(f"  Split {split+1}/{n_splits} | Params: {best_params} | IS Sharpe: {best.get('sharpe', 0):.3f} | OOS Sharpe: {sharpe_ratio(eq):.3f}")
    if oos_curves:
        chained = oos_curves[0].copy()
        for seg in oos_curves[1:]:
            scale = chained.iloc[-1] / seg.iloc[0]
            chained = pd.concat([chained, seg * scale])
        oos_equity = chained
    else:
        oos_equity = pd.Series(dtype=float)
    return {"oos_results": pd.DataFrame(oos_results), "oos_equity": oos_equity, "best_params": best_params_list}