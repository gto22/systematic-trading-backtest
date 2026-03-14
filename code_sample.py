
from __future__ import annotations

"""
Example of a systematic FX backtesting engine written in Python.

This script:
- loads bid/ask OHLC market data on two timeframes
- builds mid-price series
- computes technical indicators
- generates rule-based signals
- simulates execution on a lower timeframe
- reports basic performance statistics

The trading strategy used here is intentionally simple and only serves as an
illustrative example. The goal of this code sample is to demonstrate how market
data, signal generation, execution logic and performance evaluation can be
structured in a systematic trading research workflow.

This repository was created as part of a job application to showcase my Python
and systematic trading development work.

Notes:
- Bid/ask prices are used so spread is implicitly taken into account.
- Commission and slippage modelling are intentionally omitted to keep the
  example concise and focused on the backtesting engine structure.
- The repository includes four sample CSV files containing Dukascopy bid/ask
  OHLC market data required to run the script.
"""


import time
import pandas as pd
import pandas_ta as ta


# =============================================================================
# Configuration
# =============================================================================

SYMBOL = "EURUSD"
CAPITAL_INITIAL = 100_000.0

SIGNAL_BID_PATH = "data/EURUSD_10 Mins_Bid_sample.csv"
SIGNAL_ASK_PATH = "data/EURUSD_10 Mins_Ask_sample.csv"
EXEC_BID_PATH = "data/EURUSD_1 Min_Bid_sample.csv"
EXEC_ASK_PATH = "data/EURUSD_1 Min_Ask_sample.csv"


ENTRY_DELAY_MINUTES = 10
RISK_PERCENT = 0.005

# Demo parameters chosen so the sample dataset generates a few trades quickly.


RSI_LEN = 5
RMA_LEN = 50
ATR_LEN = 14

RSI_LONG_MAX = 25
RSI_SHORT_MIN = 75

SL_ATR_MULT = 1
TP_ATR_MULT = 1

CONTRACT_SIZE = 100_000.0  # EURUSD standard lot
MIN_LOT = 0.01
LOT_STEP = 0.01


# =============================================================================
# Core objects
# =============================================================================

class Position:
    def __init__(
        self,
        entry_price: float,
        size: float,
        direction: str,
        sl_price: float,
        tp_price: float,
        entry_time: pd.Timestamp,
        contract_size: float,
    ):
        if direction not in ("LONG", "SHORT"):
            raise ValueError("direction must be LONG or SHORT")

        self.entry_price = entry_price
        self.size = size
        self.contract_size = contract_size
        self.direction = direction
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.entry_time = entry_time
        self.exit_time: pd.Timestamp | None = None
        self.is_open = True

    def pnl(self, exit_price: float) -> float:
        delta = exit_price - self.entry_price
        if self.direction == "SHORT":
            delta = -delta
        return delta * self.contract_size * self.size

    def close(self, exit_price: float) -> float:
        self.is_open = False
        return self.pnl(exit_price)


class Account:
    def __init__(self, capital_initial: float):
        self.capital_initial = capital_initial
        self.capital = capital_initial
        self.nb_trades = 0
        self.nb_wins = 0
        self.trades: list[dict] = []
        self.dead = False

    def apply_pnl(self, pnl: float, meta: dict | None = None) -> float:
        self.nb_trades += 1

        if pnl >= 0:
            self.nb_wins += 1

        self.capital += pnl
        if self.capital <= 0:
            self.capital = 0
            self.dead = True

        trade_info = {
            "pnl": pnl,
            "capital_after": self.capital,
        }

        if meta is not None:
            trade_info.update(meta)

        self.trades.append(trade_info)
        return self.capital

    def get_stats(self) -> dict:
        if self.nb_trades == 0:
            return {
                "capital_final": self.capital,
                "nb_trades": 0,
                "winrate": None,
                "profit_factor": None,
                "expectancy": None,
                "max_drawdown": None,
            }

        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [-p for p in pnls if p < 0]

        total_wins = sum(wins)
        total_losses = sum(losses)

        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
        expectancy = sum(pnls) / self.nb_trades

        capitals = [self.capital_initial] + [t["capital_after"] for t in self.trades]
        peak = capitals[0]
        max_drawdown = 0.0

        for c in capitals:
            if c > peak:
                peak = c
            drawdown = peak - c
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            "capital_final": self.capital,
            "nb_trades": self.nb_trades,
            "winrate": self.nb_wins / self.nb_trades,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_drawdown": max_drawdown,
        }


class Risk:
    def __init__(self, risk_percent: float):
        if risk_percent <= 0:
            raise ValueError("risk_percent must be > 0")
        self.risk_percent = risk_percent

    def get_lots(
        self,
        capital_usd: float,
        entry_price: float,
        sl_price: float,
        contract_size: float,
    ) -> float:
        if capital_usd <= 0:
            raise ValueError("capital_usd must be > 0")

        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            raise ValueError("SL distance must be > 0")

        risk_usd = capital_usd * self.risk_percent
        risk_usd_per_lot = sl_distance * contract_size

        if risk_usd_per_lot <= 0:
            raise ValueError("Invalid risk per lot")

        return risk_usd / risk_usd_per_lot


class Strategy:
    def __init__(
        self,
        rsi_len: int,
        rma_len: int,
        atr_len: int,
        rsi_long_max: float,
        rsi_short_min: float,
        sl_atr_mult: float,
        tp_atr_mult: float,
    ):
        self.rsi_len = rsi_len
        self.rma_len = rma_len
        self.atr_len = atr_len
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult

    def get_required_indicators(self) -> dict:
        return {"rsi": self.rsi_len, "rma": self.rma_len, "atr": self.atr_len}

    def add_signals(self, df_signal: pd.DataFrame) -> pd.DataFrame:
        df = df_signal.copy()

        green = df["Close_mid"] > df["Open_mid"]
        red = df["Close_mid"] < df["Open_mid"]

        slope = df["rma"].diff()

        prev_red = red.shift(1)
        prev_green = green.shift(1)
        prev_rsi = df["rsi"].shift(1)
        prev_slope = slope.shift(1)

        ok_atr_t = df["atr"].notna() & (df["atr"] > 0)
        df["signal"] = pd.Series(index=df.index, dtype="object")

        df.loc[
            ok_atr_t
            & green
            & prev_red
            & (prev_rsi < self.rsi_long_max)
            & (prev_slope > 0.00001),
            "signal",
        ] = "LONG"

        df.loc[
            ok_atr_t
            & red
            & prev_green
            & (prev_rsi > self.rsi_short_min)
            & (prev_slope < -0.00001),
            "signal",
        ] = "SHORT"

        return df

    def compute_sl_tp(
        self,
        direction: str,
        row_signal: pd.Series,
        row_exec: pd.Series,
    ) -> tuple[float, float]:
        atr = float(row_signal["atr"])
        if pd.isna(atr) or atr <= 0:
            raise ValueError(f"Invalid ATR value: {atr}")

        open_bid = float(row_exec["Open_bid"])
        open_ask = float(row_exec["Open_ask"])

        sl_distance = atr * self.sl_atr_mult
        tp_distance = atr * self.tp_atr_mult

        if direction == "LONG":
            sl = open_bid - sl_distance
            tp = open_bid + tp_distance
        elif direction == "SHORT":
            sl = open_ask + sl_distance
            tp = open_ask - tp_distance
        else:
            raise ValueError(f"Invalid direction: {direction}")

        return sl, tp

    def should_exit_exec(
        self,
        high_bid: float,
        low_bid: float,
        high_ask: float,
        low_ask: float,
        position: Position,
    ) -> str | None:
        """
        Intrabar exit logic on M1.
        Convention: if SL and TP are both hit on the same bar, SL has priority.
        """
        if position.direction == "LONG":
            sl_hit = low_bid <= position.sl_price
            tp_hit = high_bid >= position.tp_price

            if sl_hit:
                return "sl"
            if tp_hit:
                return "tp"

        else:
            sl_hit = high_ask >= position.sl_price
            tp_hit = low_ask <= position.tp_price

            if sl_hit:
                return "sl"
            if tp_hit:
                return "tp"

        return None


# =============================================================================
# Data handling
# =============================================================================

class MarketData:
    """
    Time convention:
    - CSV timestamps correspond to the START of the bar
    - Close corresponds to the END of the bar period

    Example:
    H1 02:00 covers [02:00 -> 03:00]
    Close(H1 02:00) == Close(M1 02:59) == real price at 03:00
    """

    def __init__(
        self,
        signal_bid_path: str,
        signal_ask_path: str,
        exec_bid_path: str,
        exec_ask_path: str,
    ):
        self.df_signal_bid = self._load_csv(signal_bid_path)
        self.df_signal_ask = self._load_csv(signal_ask_path)
        self.df_exec_bid = self._load_csv(exec_bid_path)
        self.df_exec_ask = self._load_csv(exec_ask_path)

        self._check_alignment(self.df_signal_bid, self.df_signal_ask, "signal")
        self._check_alignment(self.df_exec_bid, self.df_exec_ask, "exec")

        self.df_signal = self._merge_mid_bid_ask(self.df_signal_bid, self.df_signal_ask)
        self.df_exec = self._merge_mid_bid_ask(self.df_exec_bid, self.df_exec_ask)

    def _load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df["Time (UTC)"] = pd.to_datetime(df["Time (UTC)"], format="%d.%m.%Y %H:%M:%S")
        df = df.set_index("Time (UTC)")
        return df.astype(float)

    def _check_alignment(self, df_bid: pd.DataFrame, df_ask: pd.DataFrame, name: str) -> None:
        if not df_bid.index.equals(df_ask.index):
            raise ValueError(f"Bid and ask data are not aligned for '{name}'")

    def _merge_mid_bid_ask(self, df_bid: pd.DataFrame, df_ask: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=df_bid.index)

        df["Open_mid"] = (df_bid["Open"] + df_ask["Open"]) / 2
        df["High_mid"] = (df_bid["High"] + df_ask["High"]) / 2
        df["Low_mid"] = (df_bid["Low"] + df_ask["Low"]) / 2
        df["Close_mid"] = (df_bid["Close"] + df_ask["Close"]) / 2

        df["Open_bid"] = df_bid["Open"]
        df["High_bid"] = df_bid["High"]
        df["Low_bid"] = df_bid["Low"]
        df["Close_bid"] = df_bid["Close"]

        df["Open_ask"] = df_ask["Open"]
        df["High_ask"] = df_ask["High"]
        df["Low_ask"] = df_ask["Low"]
        df["Close_ask"] = df_ask["Close"]

        df["Volume"] = df_bid["Volume"]
        df["Spread"] = df["Close_ask"] - df["Close_bid"]

        return df


class Indicators:
    @staticmethod
    def add(df: pd.DataFrame, indicators: dict) -> None:
        if "rsi" in indicators:
            p = indicators["rsi"]
            df["rsi"] = ta.rsi(df["Close_mid"], length=p)

        if "ma" in indicators:
            p = indicators["ma"]
            df["ma"] = ta.sma(df["Close_mid"], length=p)

        if "rma" in indicators:
            p = indicators["rma"]
            df["rma"] = ta.rma(df["Close_mid"], length=p)

        if "atr" in indicators:
            p = indicators["atr"]
            df["atr"] = ta.atr(
                high=df["High_mid"],
                low=df["Low_mid"],
                close=df["Close_mid"],
                length=p,
            )

        
# =============================================================================
# Backtest engine
# =============================================================================

class Engine:
    def __init__(
        self,
        market_data: MarketData,
        capital_initial: float,
        strategy: Strategy,
        symbol: str,
        entry_delay_minutes: int,
        risk_percent: float,
        contract_size: float,
        min_lot: float,
        lot_step: float,
    ):
        self.market_data = market_data
        self.strategy = strategy
        self.symbol = symbol
        self.account = Account(capital_initial)
        self.entry_delay = pd.Timedelta(minutes=entry_delay_minutes)
        self.risk = Risk(risk_percent=risk_percent)

        self.contract_size = contract_size
        self.min_lot = min_lot
        self.lot_step = lot_step

        self.df_signal = self.market_data.df_signal.copy()
        self.df_signal["close_time"] = self.df_signal.index + self.entry_delay

        needed_indicators = self.strategy.get_required_indicators()
        Indicators.add(self.df_signal, needed_indicators)
        self.df_signal = self.strategy.add_signals(self.df_signal)

        self.df_exec = self.market_data.df_exec.copy()

    def _round_lot_down(self, size_raw: float) -> float:
        eps = 1e-12
        size = ((size_raw + eps) // self.lot_step) * self.lot_step
        return round(size, 10)

    def _find_first_exec_index(self, ts_entry: pd.Timestamp) -> int | None:
        idx = self.df_exec.index.searchsorted(ts_entry, side="left")
        if idx >= len(self.df_exec):
            return None
        return idx

    def _find_next_signal_index(self, ts_exit: pd.Timestamp) -> int | None:
        idx = self.df_signal.index.searchsorted(ts_exit, side="left")
        if idx >= len(self.df_signal):
            return None
        return idx

    def _book_exit(
        self,
        position: Position,
        exit_price: float,
        ts_exit: pd.Timestamp,
        signal_time: pd.Timestamp,
        reason: str,
    ) -> tuple[float, float]:
        pnl_usd = position.close(exit_price)
        position.exit_time = ts_exit

        meta = {
            "symbol": self.symbol,
            "direction": position.direction,
            "size_lots": position.size,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "sl_price": position.sl_price,
            "tp_price": position.tp_price,
            "signal_time": signal_time,
            "entry_time": position.entry_time,
            "exit_time": position.exit_time,
            "reason": reason,
        }

        capital = self.account.apply_pnl(pnl_usd, meta=meta)
        return capital, pnl_usd

    def run(self) -> None:
        signal_index = self.df_signal.index
        if len(signal_index) < 2:
            raise ValueError("Not enough signal data")

        i = 1

        while i < len(signal_index):
            ts_signal = signal_index[i]
            row_signal = self.df_signal.iloc[i]
            signal = row_signal["signal"]

            if pd.isna(signal):
                i += 1
                continue

            signal_time = ts_signal
            entry_time = row_signal["close_time"]
            direction = signal

            if entry_time not in self.df_exec.index:
                i += 1
                continue

            row_exec = self.df_exec.loc[entry_time]

            if direction == "LONG":
                entry_price = float(row_exec["Open_ask"])
            else:
                entry_price = float(row_exec["Open_bid"])

            sl_price, tp_price = self.strategy.compute_sl_tp(
                direction=direction,
                row_signal=row_signal,
                row_exec=row_exec,
            )

            size_raw = self.risk.get_lots(
                capital_usd=self.account.capital,
                entry_price=entry_price,
                sl_price=sl_price,
                contract_size=self.contract_size,
            )
            size = self._round_lot_down(size_raw)

            if size < self.min_lot:
                i += 1
                continue

            position = Position(
                entry_price=entry_price,
                size=size,
                direction=direction,
                sl_price=sl_price,
                tp_price=tp_price,
                entry_time=entry_time,
                contract_size=self.contract_size,
            )

            idx_exec = self._find_first_exec_index(entry_time)

            if idx_exec is None:
                last_row = self.df_exec.iloc[-1]
                ts_last = last_row.name
                exit_price = last_row["Close_bid"] if direction == "LONG" else last_row["Close_ask"]

                self._book_exit(
                    position=position,
                    exit_price=exit_price,
                    ts_exit=ts_last,
                    signal_time=signal_time,
                    reason="no_exec_data_after_entry",
                )
                break

            exited = False

            while idx_exec < len(self.df_exec):
                row_exec = self.df_exec.iloc[idx_exec]

                high_bid = row_exec["High_bid"]
                low_bid = row_exec["Low_bid"]
                high_ask = row_exec["High_ask"]
                low_ask = row_exec["Low_ask"]
                ts_exec = row_exec.name

                exit_event = self.strategy.should_exit_exec(
                    high_bid=high_bid,
                    low_bid=low_bid,
                    high_ask=high_ask,
                    low_ask=low_ask,
                    position=position,
                )

                if exit_event is not None:
                    exit_price = position.sl_price if exit_event == "sl" else position.tp_price

                    self._book_exit(
                        position=position,
                        exit_price=exit_price,
                        ts_exit=ts_exec,
                        signal_time=signal_time,
                        reason=exit_event,
                    )

                    if self.account.dead:
                        return

                    exited = True
                    break

                idx_exec += 1

            if not exited:
                last_row = self.df_exec.iloc[-1]
                ts_last = last_row.name
                exit_price = last_row["Close_bid"] if direction == "LONG" else last_row["Close_ask"]

                self._book_exit(
                    position=position,
                    exit_price=exit_price,
                    ts_exit=ts_last,
                    signal_time=signal_time,
                    reason="end_of_exec_data",
                )

            next_i = self._find_next_signal_index(position.exit_time)
            if next_i is None:
                break

            i = next_i


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    t1 = time.time()

    market_data = MarketData(
        signal_bid_path=SIGNAL_BID_PATH,
        signal_ask_path=SIGNAL_ASK_PATH,
        exec_bid_path=EXEC_BID_PATH,
        exec_ask_path=EXEC_ASK_PATH,
    )

    strategy = Strategy(
        rsi_len=RSI_LEN,
        rma_len=RMA_LEN,
        atr_len=ATR_LEN,
        rsi_long_max=RSI_LONG_MAX,
        rsi_short_min=RSI_SHORT_MIN,
        sl_atr_mult=SL_ATR_MULT,
        tp_atr_mult=TP_ATR_MULT,
    )

    engine = Engine(
        market_data=market_data,
        capital_initial=CAPITAL_INITIAL,
        strategy=strategy,
        symbol=SYMBOL,
        entry_delay_minutes=ENTRY_DELAY_MINUTES,
        risk_percent=RISK_PERCENT,
        contract_size=CONTRACT_SIZE,
        min_lot=MIN_LOT,
        lot_step=LOT_STEP,
    )

    engine.run()
    stats = engine.account.get_stats()

    t2 = time.time()
    runtime = t2 - t1

    print("===== BACKTEST RESULTS =====")
    print(f"Symbol: {SYMBOL}")
    print(f"Initial capital: {CAPITAL_INITIAL:.2f}")
    print(f"Final capital:   {stats['capital_final']:.2f}")
    print(f"Number of trades: {stats['nb_trades']}")

    if stats["winrate"] is None:
        print("Win rate: N/A")
    else:
        print(f"Win rate: {stats['winrate'] * 100:.2f}%")

    print(f"Profit factor: {stats['profit_factor']:.3f}")
    print(f"Expectancy per trade: {stats['expectancy']:.6f}")
    print(f"Max drawdown: {stats['max_drawdown']:.2f}")
    print(f"Runtime: {runtime:.4f} seconds")


if __name__ == "__main__":
    main()






