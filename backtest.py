from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    integer_shares: bool = False
    allow_short: bool = False


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    direction: str = "LONG"
    entry_cost: float = 0.0
    exit_cost: float = 0.0

    @property
    def gross_pnl(self):
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def net_pnl(self):
        return self.gross_pnl - self.entry_cost - self.exit_cost

    @property
    def return_pct(self):
        invested = self.entry_price * self.shares
        return self.net_pnl / invested if invested else 0.0

    @property
    def holding_days(self):
        return (self.exit_date - self.entry_date).days


class Backtester:
    def __init__(self, price_df, signal_df, config=None):
        self.cfg = config or BacktestConfig()
        if self.cfg.allow_short:
            raise NotImplementedError("Short selling not implemented.")
        self.data = price_df[["Open", "High", "Low", "Close"]].join(
            signal_df[["signal"]], how="inner"
        ).copy()
        self.data["signal"] = self.data["signal"].fillna(0).astype(int)

    def _execution_price(self, bar, side):
        slip = self.cfg.slippage_pct
        return bar["Open"] * (1 + slip) if side == "BUY" else bar["Open"] * (1 - slip)

    def _transaction_cost(self, trade_value):
        return trade_value * self.cfg.commission_pct

    def run(self):
        cfg = self.cfg
        data = self.data
        cash = cfg.initial_capital
        shares = 0.0
        in_trade = False
        entry_bar = None
        entry_px = 0.0
        entry_cost = 0.0
        equity_curve = pd.Series(index=data.index, dtype=float)
        positions = pd.Series(index=data.index, dtype=float)
        cash_series = pd.Series(index=data.index, dtype=float)
        trades = []

        for date, bar in data.iterrows():
            sig = bar["signal"]
            close = bar["Close"]

            if sig == 1 and not in_trade:
                exec_px = self._execution_price(bar, "BUY")
                cost = self._transaction_cost(cash)
                investable = cash - cost
                if investable > 0:
                    shares = investable / exec_px
                    if cfg.integer_shares:
                        shares = np.floor(shares)
                    actual_cost = exec_px * shares
                    comm = self._transaction_cost(actual_cost)
                    cash -= (actual_cost + comm)
                    in_trade = True
                    entry_bar = date
                    entry_px = exec_px
                    entry_cost = comm

            elif sig == -1 and in_trade:
                exec_px = self._execution_price(bar, "SELL")
                proceeds = exec_px * shares
                comm = self._transaction_cost(proceeds)
                trades.append(Trade(
                    entry_date=entry_bar, exit_date=date,
                    entry_price=entry_px, exit_price=exec_px,
                    shares=shares, entry_cost=entry_cost, exit_cost=comm,
                ))
                cash += proceeds - comm
                shares = 0.0
                in_trade = False

            equity_curve[date] = cash + shares * close
            positions[date] = shares
            cash_series[date] = cash

        if in_trade and shares > 0:
            last_bar = data.iloc[-1]
            last_date = data.index[-1]
            exec_px = self._execution_price(last_bar, "SELL")
            proceeds = exec_px * shares
            comm = self._transaction_cost(proceeds)
            trades.append(Trade(
                entry_date=entry_bar, exit_date=last_date,
                entry_price=entry_px, exit_price=exec_px,
                shares=shares, entry_cost=entry_cost, exit_cost=comm,
            ))
            cash += proceeds - comm
            equity_curve[last_date] = cash
            cash_series[last_date] = cash
            positions[last_date] = 0.0

        trade_log = pd.DataFrame([{
            "Entry Date": t.entry_date.date(),
            "Exit Date": t.exit_date.date(),
            "Entry Price": round(t.entry_price, 4),
            "Exit Price": round(t.exit_price, 4),
            "Shares": round(t.shares, 4),
            "Gross P&L": round(t.gross_pnl, 2),
            "Net P&L": round(t.net_pnl, 2),
            "Return (%)": round(t.return_pct * 100, 3),
            "Holding Days": t.holding_days,
        } for t in trades]) if trades else pd.DataFrame()

        return {
            "equity_curve": equity_curve,
            "trades": trades,
            "trade_log": trade_log,
            "positions": positions,
            "cash": cash_series,
            "config": cfg,
        }


def buy_and_hold_benchmark(price_df, initial_capital=100_000.0, commission_pct=0.001):
    entry_px = price_df["Open"].iloc[0] * 1.0005
    commission = initial_capital * commission_pct
    shares = (initial_capital - commission) / entry_px
    return price_df["Close"] * shares