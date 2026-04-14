# CLAUDE.md — Manual Bar-Advance Backtesting Tool

## Project Identity
A **click-to-advance chart player** for manual backtesting that eliminates look-ahead bias.
Desktop app: **Python 3.11+ / PySide6 / PyQtGraph / SQLite**.
Not a web app. Not Streamlit. PySide6 is the deliberate, non-negotiable stack choice.

---

## North Star Rules (never violate)
1. **No future data ever reaches the UI.** Indicators are precomputed but masked beyond `current_index`.
2. **Every session is reproducible.** Seeds, settings, and rule profiles are stored in the manifest.
3. **Ambiguity is recorded, not hidden.** Every fill stores `execution_rule` and `ambiguity_flag`.
4. **Review Mode is the only place Prev-bar is allowed.** Disable during live runs.
5. **Tick hygiene.** All price inputs snapped to instrument tick size before storage or comparison.

---

## Module Layout
```
backtester/
├── io/          # CSV loader, schema validators, metadata
├── engine/      # BarCursor, masking logic, indicator functions
├── sim/         # Orders, fills, ambiguity engine, slippage/commission
├── ui/          # MainWindow, ChartWidget, HUD, OrderPanel, hotkeys
├── store/       # SQLite manager, session manifest, trade log
├── tests/       # Golden series, scenario harness, property tests
└── main.py      # Entry point
```

---

## Data Contracts

### BarCursor (engine/cursor.py)
```python
cursor.current_index  # int, 0-based
cursor.advance()      # increments index by 1, emits bar_advanced signal
cursor.bars           # pd.DataFrame, full OHLCV, NEVER sliced externally
cursor.visible_bars   # property: bars.iloc[:current_index+1]
```

### Indicator masking pattern
```python
# Precompute on full series at load time
sma_full = compute_sma(bars['close'], period=20)
# Mask at render time — ALWAYS
sma_visible = sma_full.iloc[:current_index+1]
```

### Anonymization (deterministic)
```python
# Time: shift by constant delta, preserving month/day
# Price: P' = a*P + b, first visible close → 1000
# Both (a, b, time_delta, seed) stored in session manifest
```

---

## Trades Table Schema
```sql
CREATE TABLE trades (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  entry_datetime TEXT NOT NULL,
  entry_price REAL NOT NULL,
  exit_datetime TEXT NOT NULL,
  exit_price REAL NOT NULL,
  quantity REAL NOT NULL,
  side TEXT NOT NULL,           -- 'long' | 'short'
  stop_price REAL,
  target_price REAL,
  exit_reason TEXT,             -- 'stop'|'target'|'manual'|'timeout'
  pnl_currency REAL NOT NULL,
  pnl_r REAL,                   -- null if stop undefined
  commission REAL DEFAULT 0,
  slippage REAL DEFAULT 0,
  tick_size REAL NOT NULL,
  execution_rule TEXT NOT NULL, -- 'default'|'conservative'|'optimistic'|'unknown'
  ambiguity_flag INTEGER NOT NULL, -- 0/1
  notes TEXT,
  config_hash TEXT NOT NULL
);
```

---

## Fill Semantics
| Order Type | Fill Rule |
|---|---|
| Market | Next bar open |
| Limit | Fill if Low ≤ Limit ≤ High, at Limit price |
| Stop Buy | Trigger if High ≥ Stop; fill at Stop (+ slippage) |
| Stop Sell | Trigger if Low ≤ Stop; fill at Stop (- slippage) |

### Ambiguity Policies (when bar hits both stop AND target)
- **default**: time-priority path (Open→High→Low→Close if Close≥Open, else O→L→H→C)
- **conservative**: worst case (stop before target)
- **optimistic**: best case (target before stop)
- **unknown**: mark ambiguous, defer until unambiguous bar

---

## Hotkeys
| Key | Action |
|---|---|
| Space / Click | Next bar |
| F | Fit chart to visible data |
| A | Toggle anonymization |
| I | Toggle indicators |
| X / Y | Toggle axis labels |
| Arrow keys | Nudge order price ±1 tick |
| Shift+Arrow | Nudge ±10 ticks |

---

## Equity Curve Metrics
Net PnL, Avg Trade PnL, Win%, Profit Factor, Expectancy (currency + R), Max Drawdown.

---

## Testing Anchors
- **Golden series**: known OHLCV → known SMA/EMA/RSI values at each index
- **Masking test**: assert `len(indicator_visible) == current_index + 1` at all times
- **Anonymization test**: identical seed+config → identical output
- **Scenario harness**: synthetic OHLC sequences forcing each ambiguity outcome
- **Property tests**: OCO exclusivity, tick rounding, slippage non-negativity

---

## Phase Status
- [ ] Phase 1A — CSV ingest, bar cursor, masking, chart render
- [ ] Phase 1B — SMA/EMA/RSI/ATR indicators + HUD overlay
- [ ] Phase 1C — Anonymization + session manifest
- [ ] Phase 1D — UX polish, hotkeys, fit, review mode, Phase 1 tests
- [ ] Phase 2A — Market orders, bracket helper, P&L, trade log, equity curve
- [ ] Phase 2B — Limit/Stop orders, ambiguity engine, slippage/commission, exports

---

## Key Constraints
- Python 3.11+
- PySide6 for all UI (no tkinter, no matplotlib, no Streamlit)
- PyQtGraph for chart rendering (target: smooth at 50k bars)
- SQLite via stdlib `sqlite3` (no ORM)
- pandas for data; NumPy for indicator math
- No external API calls; offline-only
