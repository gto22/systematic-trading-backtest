# Systematic Trading Backtest Example

This repository contains a simplified example of a systematic FX / index backtesting engine written in Python.
This example uses bid/ask prices, so spread is implicitly taken into account.
Commission and slippage modelling are intentionally omitted here to keep the example concise and focused on engine structure.

The goal of this code sample is to illustrate how market data, signals, execution logic and performance evaluation can be structured in a systematic trading research workflow.

## Run the example

Install dependencies:

pip install -r requirements.txt

Then run the script:

python backtest.py

## Features

- bid/ask OHLC market data handling
- mid-price construction
- technical indicator computation
- rule-based signal generation
- lower-timeframe execution simulation
- position sizing and risk management
- performance statistics (win rate, profit factor, drawdown, expectancy)

## Strategy

The trading strategy included in this repository is intentionally simple and serves only as an illustrative example.  
The main purpose of the code is to demonstrate the structure of the backtesting engine rather than the sophistication of the strategy itself.

## Data

The repository includes four sample CSV files located in the `data/` folder.

These files contain Dukascopy bid/ask OHLC market data used to run the example backtest.

They are provided only for demonstration purposes.

## Purpose

This repository was created as part of a job application to showcase my work on Python-based systematic trading research and backtesting.
