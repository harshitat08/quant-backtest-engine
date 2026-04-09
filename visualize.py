from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from pathlib import Path

STYLE = {
    "bg": "#0d1117", "panel": "#161b22", "grid": "#21262d",
    "text": "#e6edf3", "subtext": "#8b949e", "green": "#3fb950",
    "red": "#f85149", "blue": "#58a6ff", "orange": "#d29922",
    "purple": "#bc8cff", "line": "#58a6ff", "benchmark": "#8b949e",
}


def _apply_dark_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(STYLE["panel"])
    ax.tick_params(colors=STYLE["subtext"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(STYLE["grid"])
    ax.grid(color=STYLE["grid"], linewidth=0.6, alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    if title:
        ax.set_title(title, color=STYLE["text"], fontsize=11, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=STYLE["subtext"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=STYLE["subtext"], fontsize=9)


def plot_price_signals(price_df, signal_df, ticker, strategy_name="Strategy", save_path=None):
    fig = plt.figure(figsize=(14, 9), facecolor=STYLE["bg"])
    has_rsi = "RSI" in signal_df.columns
    height_ratios = [3, 1] if has_rsi else [1]
    gs = gridspec.GridSpec(len(height_ratios), 1, figure=fig, height_ratios=height_ratios, hspace=0.05)
    ax_price = fig.add_subplot(gs[0])
    ax_price.plot(price_df.index, price_df["Close"], color=STYLE["line"], linewidth=1.2, label="Close", zorder=2)
    if "MA_Short" in signal_df.columns:
        ax_price.plot(signal_df.index, signal_df["MA_Short"], color=STYLE["orange"], linewidth=0.9, alpha=0.85, label="Short MA", linestyle="--")
    if "MA_Long" in signal_df.columns:
        ax_price.plot(signal_df.index, signal_df["MA_Long"], color=STYLE["purple"], linewidth=0.9, alpha=0.85, label="Long MA", linestyle="--")
    if "BB_Upper" in signal_df.columns:
        ax_price.fill_between(signal_df.index, signal_df["BB_Lower"], signal_df["BB_Upper"], alpha=0.10, color=STYLE["blue"], label="Bollinger Band")
        ax_price.plot(signal_df.index, signal_df["BB_Upper"], color=STYLE["blue"], linewidth=0.6, alpha=0.5)
        ax_price.plot(signal_df.index, signal_df["BB_Lower"], color=STYLE["blue"], linewidth=0.6, alpha=0.5)
    signals = signal_df["signal"]
    buy_dates = signals[signals == 1].index
    sell_dates = signals[signals == -1].index
    if len(buy_dates):
        ax_price.scatter(buy_dates, price_df.loc[buy_dates, "Close"], marker="^", color=STYLE["green"], s=70, zorder=5, label=f"BUY ({len(buy_dates)})", linewidths=0.5, edgecolors="white")
    if len(sell_dates):
        ax_price.scatter(sell_dates, price_df.loc[sell_dates, "Close"], marker="v", color=STYLE["red"], s=70, zorder=5, label=f"SELL ({len(sell_dates)})", linewidths=0.5, edgecolors="white")
    _apply_dark_style(ax_price, title=f"{ticker} — {strategy_name} Signals", ylabel="Price")
    ax_price.legend(loc="upper left", fontsize=8, facecolor=STYLE["panel"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"])
    if has_rsi:
        ax_rsi = fig.add_subplot(gs[1], sharex=ax_price)
        ax_rsi.plot(signal_df.index, signal_df["RSI"], color=STYLE["orange"], linewidth=0.9)
        ax_rsi.axhline(70, color=STYLE["red"], linewidth=0.8, linestyle="--", alpha=0.7)
        ax_rsi.axhline(30, color=STYLE["green"], linewidth=0.8, linestyle="--", alpha=0.7)
        ax_rsi.fill_between(signal_df.index, signal_df["RSI"], 70, where=signal_df["RSI"] >= 70, alpha=0.15, color=STYLE["red"])
        ax_rsi.fill_between(signal_df.index, signal_df["RSI"], 30, where=signal_df["RSI"] <= 30, alpha=0.15, color=STYLE["green"])
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_yticks([30, 50, 70])
        _apply_dark_style(ax_rsi, ylabel="RSI")
        plt.setp(ax_price.xaxis.get_majorticklabels(), visible=False)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_equity_and_drawdown(equity_curve, drawdown_series, benchmark_curve=None, ticker="", strategy_name="Strategy", save_path=None):
    fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.06}, facecolor=STYLE["bg"])
    initial = equity_curve.iloc[0]
    ax_eq.plot(equity_curve.index, equity_curve, color=STYLE["green"], linewidth=1.5, label=strategy_name, zorder=3)
    if benchmark_curve is not None:
        ax_eq.plot(benchmark_curve.index, benchmark_curve * (initial / benchmark_curve.iloc[0]), color=STYLE["benchmark"], linewidth=1.1, linestyle="--", label="Buy & Hold", zorder=2, alpha=0.8)
    ax_eq.fill_between(equity_curve.index, initial, equity_curve, where=equity_curve >= initial, alpha=0.06, color=STYLE["green"])
    ax_eq.fill_between(equity_curve.index, initial, equity_curve, where=equity_curve < initial, alpha=0.08, color=STYLE["red"])
    ax_eq.axhline(initial, color=STYLE["subtext"], linewidth=0.6, linestyle=":")
    _apply_dark_style(ax_eq, title=f"{ticker} — Portfolio Equity Curve", ylabel="Portfolio Value")
    ax_eq.legend(fontsize=9, facecolor=STYLE["panel"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"])
    ax_dd.fill_between(drawdown_series.index, 0, drawdown_series * 100, color=STYLE["red"], alpha=0.45)
    ax_dd.plot(drawdown_series.index, drawdown_series * 100, color=STYLE["red"], linewidth=0.8, alpha=0.8)
    ax_dd.axhline(0, color=STYLE["subtext"], linewidth=0.5)
    max_dd_val = drawdown_series.min() * 100
    max_dd_date = drawdown_series.idxmin()
    ax_dd.annotate(f"MDD: {max_dd_val:.1f}%", xy=(max_dd_date, max_dd_val), xytext=(10, -20), textcoords="offset points", color=STYLE["red"], fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="->", color=STYLE["red"], lw=0.8))
    _apply_dark_style(ax_dd, ylabel="Drawdown (%)")
    plt.setp(ax_eq.xaxis.get_majorticklabels(), visible=False)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_trade_distribution(trades, strategy_name="Strategy", save_path=None):
    if not trades:
        return
    returns_pct = [t.return_pct * 100 for t in trades]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=STYLE["bg"])
    bins = min(30, max(10, len(returns_pct) // 3))
    wins_data = [r for r in returns_pct if r >= 0]
    losses_data = [r for r in returns_pct if r < 0]
    if wins_data:
        ax.hist(wins_data, bins=bins, color=STYLE["green"], alpha=0.7, edgecolor=STYLE["bg"], linewidth=0.5, label="Wins")
    if losses_data:
        ax.hist(losses_data, bins=bins, color=STYLE["red"], alpha=0.7, edgecolor=STYLE["bg"], linewidth=0.5, label="Losses")
    ax.axvline(0, color=STYLE["text"], linewidth=0.9, linestyle="--")
    ax.axvline(np.mean(returns_pct), color=STYLE["orange"], linewidth=1.2, linestyle="--", label=f"Mean: {np.mean(returns_pct):.2f}%")
    _apply_dark_style(ax, title=f"{strategy_name} — Trade Return Distribution ({len(trades)} trades)", xlabel="Return per Trade (%)", ylabel="Frequency")
    ax.legend(fontsize=9, facecolor=STYLE["panel"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"])
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_strategy_comparison(results, save_path=None):
    from metrics import total_return, cagr, sharpe_ratio, max_drawdown, annualised_volatility
    strategies = list(results.keys())
    metrics = {
        "Total Return (%)": lambda r: total_return(r["equity_curve"]) * 100,
        "CAGR (%)": lambda r: cagr(r["equity_curve"]) * 100,
        "Sharpe Ratio": lambda r: sharpe_ratio(r["equity_curve"]),
        "Max Drawdown (%)": lambda r: max_drawdown(r["equity_curve"])[0] * 100,
        "Volatility (%)": lambda r: annualised_volatility(r["equity_curve"]) * 100,
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), facecolor=STYLE["bg"])
    colors = [STYLE["blue"], STYLE["green"], STYLE["orange"], STYLE["purple"], STYLE["red"]]
    for ax, (metric_name, fn) in zip(axes, metrics.items()):
        values = [fn(results[s]) for s in strategies]
        bars = ax.bar(strategies, values, color=colors[:len(strategies)], alpha=0.8, width=0.5, edgecolor=STYLE["bg"], linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(abs(v) for v in values) * 0.02), f"{val:.2f}", ha="center", va="bottom", color=STYLE["text"], fontsize=8)
        ax.axhline(0, color=STYLE["subtext"], linewidth=0.5)
        _apply_dark_style(ax, title=metric_name)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=15, ha="right", fontsize=8)
    fig.suptitle("Strategy Comparison", fontsize=14, fontweight="bold", color=STYLE["text"], y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Chart saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
    