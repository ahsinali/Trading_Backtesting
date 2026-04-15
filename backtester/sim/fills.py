"""FillEngine — processes a bar against pending orders and open positions."""

from __future__ import annotations

import pandas as pd

from .ambiguity import AmbiguityEngine
from .orders import OpenPosition, Order, Trade, snap_to_tick
from .pnl import compute_pnl_currency, compute_pnl_r


class FillEngine:
    """Stateful fill engine for one backtesting session.

    Maintains a list of open positions (entries that have filled but not yet
    exited).  Each call to :meth:`process_bar` does two things:

    1. Fills any pending **entry** orders from *pending_orders* whose
       conditions are met on *bar*.
    2. Checks every open position's stop / target bracket against *bar*; if
       either is triggered the position is closed and a :class:`~.orders.Trade`
       is returned.

    OCO guarantee
    -------------
    Stop and target are stored as scalar prices on the
    :class:`~.orders.OpenPosition`; there are no separate Order objects for
    them.  When one leg triggers the position is removed from
    ``_open_positions``, making it impossible for the other leg to trigger on
    a later bar — this is the OCO invariant.

    Entry order types
    -----------------
    * **Market** — filled at next bar's open; slippage deducted in P&L calc.
    * **Limit**  — filled when ``low <= price <= high``; no slippage on fill.
    * **Stop**   — long buy-stop fills when ``high >= trigger`` at
      ``trigger + slippage``; short sell-stop fills when ``low <= trigger``
      at ``trigger - slippage`` (slippage embedded in fill price).

    Ambiguity policies
    ------------------
    When both stop and target are hit on the same bar the
    :class:`~.ambiguity.AmbiguityEngine` is consulted:

    * ``'default'``      — time-priority path (OHLC order); no flag.
    * ``'conservative'`` — stop wins; ``ambiguity_flag=1``.
    * ``'optimistic'``   — target wins; ``ambiguity_flag=1``.
    * ``'unknown'``      — position deferred to next bar; no trade returned.
      The position is marked ``ambiguous=True`` and re-evaluated on the next
      call to ``process_bar``.  When it eventually closes,
      ``ambiguity_flag=1``.
    """

    def __init__(
        self,
        tick_size:      float,
        commission:     float = 0.0,
        slippage:       float = 0.0,
        execution_rule: str   = "default",
        symbol:         str   = "",
        timeframe:      str   = "",
        config_hash:    str   = "",
    ) -> None:
        self._tick_size      = tick_size
        self._commission     = commission
        self._slippage       = slippage
        self._execution_rule = execution_rule
        self._symbol         = symbol
        self._timeframe      = timeframe
        self._config_hash    = config_hash
        self._ambiguity      = AmbiguityEngine(execution_rule)
        self._open_positions: list[OpenPosition] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def open_positions(self) -> list[OpenPosition]:
        """Snapshot of currently open positions (defensive copy)."""
        return list(self._open_positions)

    def set_execution_rule(self, rule: str) -> None:
        """Change the ambiguity policy.  Locked by UI after first order."""
        self._execution_rule = rule
        self._ambiguity      = AmbiguityEngine(rule)

    def process_bar(
        self,
        bar:            pd.Series,
        pending_orders: list[Order],
    ) -> list[Trade]:
        """Process *bar* against *pending_orders* and internal open positions.

        Parameters
        ----------
        bar:
            A single OHLCV row as a ``pd.Series`` whose ``name`` attribute is
            the bar's ``Timestamp``.
        pending_orders:
            All outstanding entry orders.  Orders whose ``status`` is not
            ``'pending'`` are silently skipped.  Filled orders have their
            ``status`` set to ``'filled'`` in-place.

        Returns
        -------
        list[Trade]
            Completed trades (entry + exit) closed on this bar.
        """
        completed: list[Trade] = []
        dt = str(bar.name)

        # ── Step 1: fill pending entry orders ─────────────────────────
        for order in pending_orders:
            if order.status != "pending":
                continue
            fill_info = self._entry_fill_price(bar, order)
            if fill_info is None:
                continue
            fill_price, slip_cost = fill_info
            order.status = "filled"
            pos = OpenPosition(
                order_id          = order.id,
                session_id        = order.session_id,
                side              = order.side,
                entry_price       = fill_price,
                entry_datetime    = dt,
                quantity          = order.quantity,
                stop_price        = order.stop_price,
                target_price      = order.target_price,
                tick_size         = self._tick_size,
                slippage_pnl_cost = slip_cost,
            )
            self._open_positions.append(pos)

        # ── Step 2: check open positions for exit ─────────────────────
        remaining: list[OpenPosition] = []
        for pos in self._open_positions:
            trade = self._check_exit(bar, pos)
            if trade is not None:
                completed.append(trade)
            else:
                remaining.append(pos)
        self._open_positions = remaining

        return completed

    def close_all(self, bar: pd.Series) -> list[Trade]:
        """Manually close every open position at *bar*'s close price.

        Returns completed :class:`~.orders.Trade` objects (``exit_reason='manual'``).
        Clears ``_open_positions``.
        """
        trades     = []
        dt         = str(bar.name)
        exit_price = snap_to_tick(float(bar["close"]), self._tick_size)
        for pos in self._open_positions:
            trades.append(
                self._build_trade(pos, exit_price, dt, "manual", ambiguity_flag=0)
            )
        self._open_positions = []
        return trades

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _entry_fill_price(
        self, bar: pd.Series, order: Order
    ) -> tuple[float, float] | None:
        """Return ``(fill_price, slippage_cost)`` for *order* on *bar*, or
        ``None`` if the order condition is not met.

        Slippage semantics per order type
        ----------------------------------
        * **Market** — slippage is deducted from P&L (returned as
          ``slippage_cost = self._slippage``); fill price = bar open.
        * **Limit**  — exact limit price; no slippage (``slippage_cost = 0``).
        * **Stop**   — slippage is embedded in the fill price (adverse tick),
          recorded as ``slippage_cost = self._slippage`` for transparency;
          P&L calc receives ``slippage=0`` since it's already in the price.
        """
        ot = order.order_type

        if ot == "market":
            fill = snap_to_tick(float(bar["open"]), self._tick_size)
            return (fill, self._slippage)

        if ot == "limit":
            low   = float(bar["low"])
            high  = float(bar["high"])
            price = order.price  # already tick-snapped
            if low <= price <= high:
                return (price, 0.0)
            return None

        if ot == "stop":
            high  = float(bar["high"])
            low   = float(bar["low"])
            price = order.price  # trigger price, already tick-snapped
            if order.side == "long" and high >= price:
                fill = snap_to_tick(price + self._slippage, self._tick_size)
                return (fill, self._slippage)
            if order.side == "short" and low <= price:
                fill = snap_to_tick(price - self._slippage, self._tick_size)
                return (fill, self._slippage)
            return None

        return None

    def _check_exit(self, bar: pd.Series, pos: OpenPosition) -> Trade | None:
        """Return a completed ``Trade`` if the position exits on *bar*,
        or ``None`` if neither stop nor target was triggered."""
        high = float(bar["high"])
        low  = float(bar["low"])
        dt   = str(bar.name)

        # Capture and clear the deferred-ambiguity flag so that if this position
        # survives (e.g. 'unknown' defers again) it doesn't carry stale state.
        was_ambiguous = pos.ambiguous
        pos.ambiguous = False

        has_stop   = pos.stop_price   is not None
        has_target = pos.target_price is not None

        if pos.side == "long":
            stop_hit   = has_stop   and low  <= pos.stop_price
            target_hit = has_target and high >= pos.target_price
        else:  # short
            stop_hit   = has_stop   and high >= pos.stop_price
            target_hit = has_target and low  <= pos.target_price

        if not stop_hit and not target_hit:
            return None

        if stop_hit and target_hit:
            return self._resolve_ambiguity(bar, pos, dt, was_ambiguous)

        if target_hit:
            exit_price = snap_to_tick(pos.target_price, self._tick_size)
            return self._build_trade(
                pos, exit_price, dt, "target",
                ambiguity_flag=1 if was_ambiguous else 0,
            )

        # stop_hit
        exit_price = snap_to_tick(pos.stop_price, self._tick_size)
        return self._build_trade(
            pos, exit_price, dt, "stop",
            ambiguity_flag=1 if was_ambiguous else 0,
        )

    def _resolve_ambiguity(
        self, bar: pd.Series, pos: OpenPosition, dt: str,
        was_ambiguous: bool = False,
    ) -> Trade | None:
        """Apply the configured ambiguity policy when both stop and target are
        hit on the same bar.

        Returns ``None`` for ``'unknown'`` policy (deferred exit).
        """
        reason, flag = self._ambiguity.resolve(
            bar, pos.stop_price, pos.target_price, pos.side
        )

        if reason == "ambiguous":
            # Mark and keep; position re-evaluated next bar
            pos.ambiguous = True
            return None

        exit_price = snap_to_tick(
            pos.stop_price if reason == "stop" else pos.target_price,
            self._tick_size,
        )
        # Flag is 1 if the policy broke a tie OR if this position was previously deferred
        final_flag = 1 if (flag or was_ambiguous) else 0
        return self._build_trade(pos, exit_price, dt, reason, ambiguity_flag=final_flag)

    def _build_trade(
        self,
        pos:            OpenPosition,
        exit_price:     float,
        exit_dt:        str,
        exit_reason:    str,
        ambiguity_flag: int,
    ) -> Trade:
        # For stop-entry and limit fills the slippage is already embedded in
        # the fill price, so pass slippage=0 to avoid double-counting in pnl.
        # For market fills slippage_pnl_cost == self._slippage and is deducted here.
        pnl  = compute_pnl_currency(
            pos.entry_price, exit_price, pos.side,
            pos.quantity, self._commission, pos.slippage_pnl_cost,
        )
        pnl_r = compute_pnl_r(
            pos.entry_price, exit_price, pos.stop_price, pos.side,
            pos.quantity, self._commission, pos.slippage_pnl_cost,
        )
        return Trade(
            entry_price    = pos.entry_price,
            side           = pos.side,
            quantity       = pos.quantity,
            session_id     = pos.session_id,
            symbol         = self._symbol,
            timeframe      = self._timeframe,
            entry_datetime = pos.entry_datetime,
            exit_datetime  = exit_dt,
            exit_price     = exit_price,
            stop_price     = pos.stop_price,
            target_price   = pos.target_price,
            exit_reason    = exit_reason,
            pnl_currency   = pnl,
            pnl_r          = pnl_r,
            commission     = self._commission,
            slippage       = pos.slippage_pnl_cost,
            tick_size      = pos.tick_size,
            execution_rule = self._execution_rule,
            ambiguity_flag = ambiguity_flag,
            notes          = None,
            config_hash    = self._config_hash,
            status         = "closed",
        )


# ── Standalone flatten helper ─────────────────────────────────────────────────

def flatten_position(
    trade:       "Trade",
    current_bar: pd.Series,
    commission:  float = 0.0,
) -> "Trade":
    """Close *trade* immediately at *current_bar*'s close price.

    Mutates and returns *trade* with all exit fields populated:
    ``exit_price``, ``exit_datetime``, ``exit_reason``, ``ambiguity_flag``,
    ``execution_rule``, ``pnl_currency``, ``pnl_r``, ``commission``, and
    ``status``.

    This is a pure transformation function; it does **not** interact with
    any ``FillEngine`` or database.  The real flatten flow in
    :class:`~backtester.ui.main_window.MainWindow` uses
    :meth:`FillEngine.close_all` which handles position bookkeeping.
    """
    exit_price = snap_to_tick(float(current_bar["close"]), trade.tick_size)

    # Accept bars where datetime is stored either as the Series index (bar.name)
    # or as an explicit key (as in test fixtures).
    dt_raw = current_bar.get("datetime", None) if hasattr(current_bar, "get") else None
    trade.exit_datetime  = str(dt_raw if dt_raw is not None else current_bar.name)

    trade.exit_price     = exit_price
    trade.exit_reason    = "manual"
    trade.ambiguity_flag = 0
    trade.execution_rule = "manual"
    trade.commission    += commission
    trade.pnl_currency   = compute_pnl_currency(
        trade.entry_price, exit_price, trade.side,
        trade.quantity, trade.commission, trade.slippage,
    )
    trade.pnl_r = compute_pnl_r(
        trade.entry_price, exit_price, trade.stop_price, trade.side,
        trade.quantity, trade.commission, trade.slippage,
    )
    trade.status = "closed"
    return trade
