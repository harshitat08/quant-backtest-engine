from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.065


def total_return(equity_curve):
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0


def cagr(equity_curve):
    n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if n_days <= 0:
        return 0.0
    years = n_days / 365.25
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / years) - 1.0


def annualised_volatility(equity_curve):
    log_returns = np.log(equity_curve / equity_curve.shift(1)).dropna()
    return log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(equity_curve, risk_free_annual=RISK_FREE_RATE_ANNUAL):
    log_returns = np.log(equity_curve / equity_curve.shift(1)).dropna()
    if log_returns.std() == 0:
        return 0.0
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = log_returns - rf_daily
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(equity_curve, risk_free_annual=RISK_FREE_RATE_ANNUAL):
    log_returns = np.log(equity_curve / equity_curve.shift(1)).dropna()
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = log_returns - rf_daily
    downside = excess[excess < 0]
    if downside.std() == 0:
        return 0.0
    return (excess.mean() * TRADING_DAYS_PER_YEAR) / (downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve):
    rolling_peak = equity_curve.cummax()
    drawdown = (equity_curve - rolling_peak) / rolling_peak
    return float(drawdown.min()), drawdown


def calmar_ratio(equity_curve):
    mdd_val, _ = max_drawdown(equity_curve)
    if mdd_val == 0:
        return np.nan
    return cagr(equity_curve) / abs(mdd_val)


def trade_statistics(trades):
    if not trades:
        return {}
    returns = [t.return_pct for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    win_rate = len(wins) / len(returns) if returns else 0
    profit_factor = (sum(wins) / -sum(losses)) if losses else np.inf
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    return {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_return_pct": np.mean(returns),
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "best_trade_pct": max(returns),
        "worst_trade_pct": min(returns),
        "avg_holding_days": np.mean([t.holding_days for t in trades]),
        "expectancy_pct": win_rate * avg_win + (1 - win_rate) * avg_loss,
    }


def performance_report(equity_curve, trades, benchmark_curve=None, strategy_name="Strategy"):
    mdd_val, _ = max_drawdown(equity_curve)
    ts = trade_statistics(trades)
    records = [
        ("Period Start", equity_curve.index[0].date()),
        ("Period End", equity_curve.index[-1].date()),
        ("Initial Capital", f"${equity_curve.iloc[0]:,.2f}"),
        ("Final Value", f"${equity_curve.iloc[-1]:,.2f}"),
        ("--- Returns ---", ""),
        ("Total Return", f"{total_return(equity_curve)*100:.2f}%"),
        ("CAGR", f"{cagr(equity_curve)*100:.2f}%"),
        ("--- Risk ---", ""),
        ("Annualised Volatility", f"{annualised_volatility(equity_curve)*100:.2f}%"),
        ("Max Drawdown", f"{mdd_val*100:.2f}%"),
        ("--- Risk-Adjusted ---", ""),
        ("Sharpe Ratio", f"{sharpe_ratio(equity_curve):.3f}"),
        ("Sortino Ratio", f"{sortino_ratio(equity_curve):.3f}"),
        ("Calmar Ratio", f"{calmar_ratio(equity_curve):.3f}"),
        ("--- Trades ---", ""),
        ("Total Trades", ts.get("total_trades", 0)),
        ("Win Rate", f"{ts.get('win_rate', 0)*100:.1f}%"),
        ("Profit Factor", f"{ts.get('profit_factor', 0):.3f}"),
        ("Avg Return / Trade", f"{ts.get('avg_return_pct', 0)*100:.3f}%"),
        ("Expectancy / Trade", f"{ts.get('expectancy_pct', 0)*100:.3f}%"),
        ("Avg Holding Period", f"{ts.get('avg_holding_days', 0):.1f} days"),
        ("Best Trade", f"{ts.get('best_trade_pct', 0)*100:.2f}%"),
        ("Worst Trade", f"{ts.get('worst_trade_pct', 0)*100:.2f}%"),
    ]
    if benchmark_curve is not None:
        bh_mdd, _ = max_drawdown(benchmark_curve)
        records += [
            ("--- Benchmark (B&H) ---", ""),
            ("B&H Total Return", f"{total_return(benchmark_curve)*100:.2f}%"),
            ("B&H CAGR", f"{cagr(benchmark_curve)*100:.2f}%"),
            ("B&H Sharpe", f"{sharpe_ratio(benchmark_curve):.3f}"),
            ("B&H Max Drawdown", f"{bh_mdd*100:.2f}%"),
        ]
    df = pd.DataFrame(records, columns=["Metric", strategy_name])
    df.set_index("Metric", inplace=True)
    return df


def print_report(report_df):
    strategy_name = report_df.columns[0]
    width = 55
    print("\n" + "=" * width)
    print(f"  PERFORMANCE REPORT — {strategy_name}")
    print("=" * width)
    for metric, row in report_df.iterrows():
        val = row.iloc[0]
        if "---" in metric:
            print(f"\n  {metric}")
        else:
            print(f"  {metric:<30s}  {str(val):>18s}")
    print("=" * width + "\n")
    