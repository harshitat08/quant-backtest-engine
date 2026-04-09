from __future__ import annotations
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

from data import fetch_ohlcv, validate_data
from strategy import ma_crossover_strategy, rsi_strategy, bollinger_strategy
from backtest import Backtester, BacktestConfig, buy_and_hold_benchmark
from metrics import performance_report, print_report, max_drawdown, sharpe_ratio, cagr
from visualize import plot_price_signals, plot_equity_and_drawdown, plot_trade_distribution, plot_strategy_comparison
from optimize import grid_search, walk_forward_optimize

CONFIG = {
    "tickers": ["AAPL", "RELIANCE.NS"],
    "start_date": "2019-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100_000.0,
    "commission_pct": 0.001,
    "slippage_pct": 0.0005,
    "output_dir": "output",
    "save_charts": True,
    "export_csv": True,
    "run_multi_strategy": True,
    "run_optimization": True,
    "run_walk_forward": False,
}


def ensure_output_dir(cfg):
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_path(out_dir, name):
    return str(out_dir / name) if CONFIG["save_charts"] else None


def run_primary_backtest(ticker, out_dir):
    print(f"\n{'='*60}\n  PRIMARY BACKTEST — {ticker}\n{'='*60}")
    df = fetch_ohlcv(ticker, CONFIG["start_date"], CONFIG["end_date"])
    validate_data(df)
    print(f"  ✓ {len(df)} trading days loaded.")
    signal_df = ma_crossover_strategy(df, short_window=20, long_window=50)
    buy_count = (signal_df["signal"] == 1).sum()
    sell_count = (signal_df["signal"] == -1).sum()
    print(f"  ✓ {buy_count} BUY signals, {sell_count} SELL signals")
    bt_config = BacktestConfig(
        initial_capital=CONFIG["initial_capital"],
        commission_pct=CONFIG["commission_pct"],
        slippage_pct=CONFIG["slippage_pct"],
    )
    bt = Backtester(df, signal_df, bt_config)
    res = bt.run()
    bh = buy_and_hold_benchmark(df, CONFIG["initial_capital"], CONFIG["commission_pct"])
    print(f"  ✓ {len(res['trades'])} trades executed.")
    _, drawdown_series = max_drawdown(res["equity_curve"])
    report = performance_report(res["equity_curve"], res["trades"], benchmark_curve=bh, strategy_name=f"MA Crossover — {ticker}")
    print_report(report)
    t = ticker.replace(".", "_")
    plot_price_signals(df, signal_df, ticker, strategy_name="MA Crossover (20/50)", save_path=save_path(out_dir, f"{t}_price_signals.png"))
    plot_equity_and_drawdown(res["equity_curve"], drawdown_series, bh, ticker=ticker, strategy_name="MA Crossover", save_path=save_path(out_dir, f"{t}_equity_drawdown.png"))
    plot_trade_distribution(res["trades"], strategy_name=f"MA Crossover — {ticker}", save_path=save_path(out_dir, f"{t}_trade_distribution.png"))
    if CONFIG["export_csv"] and not res["trade_log"].empty:
        res["trade_log"].to_csv(out_dir / f"{t}_trades.csv", index=False)
    res["equity_curve"].to_frame("portfolio_value").to_csv(out_dir / f"{t}_equity.csv")
    return {"results": res, "signal_df": signal_df, "df": df, "bh": bh}


def run_multi_strategy(ticker, df, out_dir):
    print(f"\n{'='*60}\n  MULTI-STRATEGY COMPARISON — {ticker}\n{'='*60}")
    bt_config = BacktestConfig(initial_capital=CONFIG["initial_capital"], commission_pct=CONFIG["commission_pct"], slippage_pct=CONFIG["slippage_pct"])
    strategies = {
        "MA Crossover": (ma_crossover_strategy, {}),
        "RSI Mean-Reversion": (rsi_strategy, {}),
        "Bollinger Band": (bollinger_strategy, {}),
    }
    all_results = {}
    summary_rows = []
    for name, (fn, params) in strategies.items():
        try:
            sig = fn(df, **params)
            bt = Backtester(df, sig, bt_config)
            res = bt.run()
            all_results[name] = res
            eq = res["equity_curve"]
            mdd, _ = max_drawdown(eq)
            summary_rows.append({"Strategy": name, "Total Return (%)": f"{(eq.iloc[-1]/eq.iloc[0]-1)*100:.2f}%", "CAGR (%)": f"{cagr(eq)*100:.2f}%", "Sharpe": f"{sharpe_ratio(eq):.3f}", "Max DD (%)": f"{mdd*100:.2f}%", "Trades": len(res["trades"])})
            print(f"  ✓ {name}: Sharpe={sharpe_ratio(eq):.3f}, CAGR={cagr(eq)*100:.2f}%")
        except Exception as e:
            print(f"  ✗ {name}: FAILED — {e}")
    summary = pd.DataFrame(summary_rows)
    print("\n", summary.to_string(index=False))
    t = ticker.replace(".", "_")
    plot_strategy_comparison(all_results, save_path=save_path(out_dir, f"{t}_strategy_comparison.png"))
    if CONFIG["export_csv"]:
        summary.to_csv(out_dir / f"{t}_strategy_comparison.csv", index=False)


def run_optimization(ticker, df, out_dir):
    print(f"\n{'='*60}\n  PARAMETER OPTIMIZATION — {ticker}\n{'='*60}")
    param_grid = {"short_window": [10, 15, 20, 25, 30], "long_window": [40, 50, 60, 80, 100]}
    bt_config = BacktestConfig(initial_capital=CONFIG["initial_capital"], commission_pct=CONFIG["commission_pct"], slippage_pct=CONFIG["slippage_pct"])
    gs_df = grid_search(df, ma_crossover_strategy, param_grid, config=bt_config, target_metric="sharpe", verbose=True)
    if not gs_df.empty:
        print(f"\n  Top 10 by Sharpe:\n{gs_df.head(10).to_string(index=False)}")
        t = ticker.replace(".", "_")
        if CONFIG["export_csv"]:
            gs_df.to_csv(out_dir / f"{t}_optimization.csv", index=False)


def run_walk_forward(ticker, df, out_dir):
    print(f"\n{'='*60}\n  WALK-FORWARD OPTIMIZATION — {ticker}\n{'='*60}")
    param_grid = {"short_window": [10, 15, 20, 25], "long_window": [40, 50, 60]}
    bt_config = BacktestConfig(initial_capital=CONFIG["initial_capital"], commission_pct=CONFIG["commission_pct"], slippage_pct=CONFIG["slippage_pct"])
    wf = walk_forward_optimize(df, ma_crossover_strategy, param_grid, train_pct=0.60, n_splits=5, config=bt_config, verbose=True)
    print("\n  Walk-Forward OOS Results:\n", wf["oos_results"].to_string(index=False))
    t = ticker.replace(".", "_")
    if CONFIG["export_csv"]:
        wf["oos_results"].to_csv(out_dir / f"{t}_walk_forward.csv", index=False)


def main():
    print("\n" + "=" * 60)
    print("  QUANTITATIVE BACKTESTING ENGINE  v1.0")
    print("=" * 60)
    out_dir = ensure_output_dir(CONFIG)
    for ticker in CONFIG["tickers"]:
        try:
            primary = run_primary_backtest(ticker, out_dir)
            df = primary["df"]
            if CONFIG["run_multi_strategy"]:
                run_multi_strategy(ticker, df, out_dir)
            if CONFIG["run_optimization"]:
                run_optimization(ticker, df, out_dir)
            if CONFIG["run_walk_forward"]:
                run_walk_forward(ticker, df, out_dir)
        except Exception as e:
            print(f"\n  ✗ FAILED for {ticker}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'='*60}\n  Done. See output/ for charts and CSVs.\n{'='*60}\n")


if __name__ == "__main__":
    main()
    