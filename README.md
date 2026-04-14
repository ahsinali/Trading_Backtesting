# TrdBcktest — Manual Bar-Advance Backtesting Tool

A click-to-advance chart player for manual backtesting that eliminates look-ahead bias.
Desktop app built with **Python 3.11+ / PySide6 / PyQtGraph / SQLite**.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Run tests

```bash
python -m pytest backtester/tests/ -v
```

## Loading Data

Open a CSV file via **File > Open CSV** (or `Ctrl+O`).

CSV must contain columns: `date` (or `datetime`), `open`, `high`, `low`, `close`.
Volume column is optional. Date format is auto-detected.

Constraints enforced on load:
- `high >= open`, `high >= close`
- `low <= open`, `low <= close`

## Advancing Bars

| Action | Result |
|---|---|
| Space / left-click | Reveal next bar |
| F | Fit chart to visible data |
| A | Toggle anonymization |
| I | Toggle indicators |
| X / Y | Toggle axis labels |
| Left Arrow | Step back one bar (Review Mode only) |

## Placing Orders

Use the order panel below the chart:

1. Choose **Side** (Long / Short)
2. Choose **Type** (Market / Limit / Stop)
3. For Limit and Stop orders, set the trigger/limit **Price**
4. Set **Stop** and **Target** prices for the bracket
5. Click **Place Order** — the order appears as a line on the chart

Orders fill on the next bar according to the fill semantics below.

## Fill Semantics

| Order Type | Fill Rule |
|---|---|
| Market | Next bar open |
| Limit | Fill if `Low ≤ Limit ≤ High`, at Limit price (no slippage) |
| Stop Buy | Trigger if `High ≥ Stop`; fill at `Stop + slippage` |
| Stop Sell | Trigger if `Low ≤ Stop`; fill at `Stop - slippage` |

## Ambiguity Policies

When a bar touches both the stop and target simultaneously:

| Policy | Behaviour |
|---|---|
| Default | Time-priority: bullish bar → target first; bearish → stop first |
| Conservative | Stop always wins (worst-case for the trader) |
| Optimistic | Target always wins (best-case for the trader) |
| Unknown | Defer exit to next unambiguous bar; `ambiguity_flag=1` recorded |

Select the policy in the order panel before placing the first order.
The policy is locked once an order is placed.

## Session Summary

After the first trade completes, open **Session > View Summary** for:

- **Equity Curve** — cumulative P&L line chart
- **Stats** — net P&L, win rate, profit factor, expectancy, max drawdown
- **Trade Log** — sortable full trade history

Use the **Export…** button to save trades and equity curve as CSV files.

## Review Mode

Review Mode allows stepping backward through bars to inspect past decisions.

- Only available after the last bar is reached
- Enable via **View > Review Mode**
- Use the **Left Arrow** key (or `◀ Prev` button) to step back
- Stepping back does **not** replay fills — the trade log is read-only in review mode

## Anonymization

Press **A** to anonymize prices and dates:

- Prices are transformed `P' = a·P + b`, normalising the first visible close to 1000
- Dates are shifted by a constant delta to a neutral anchor date
- Both parameters (`a`, `b`, `time_delta`, `seed`) are stored in the session manifest

This lets you share or review sessions without revealing the underlying instrument.

## Session Files

Each session produces two files in the project directory:

| File | Contents |
|---|---|
| `{session_id}.db` | SQLite database with the full trades table |
| `{session_id}.manifest.json` | Session config: symbol, timeframe, bar range, data checksum, indicator config, anonymization params, `config_hash` |

The `config_hash` (SHA-256) uniquely identifies the session configuration,
making every session reproducible from the manifest alone.
