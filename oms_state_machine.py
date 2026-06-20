#!/usr/bin/env python
# oms_state_machine.py — Lightweight Order Management System
"""
OrderStateMachine for A-share T+1 compliance + gap-down protection.

Key Features:
  - Order states: PENDING → PARTIAL_FILL | FILLED | CANCELED | REJECTED
  - Per-symbol tracking: (total_shares, available_today) tuples
  - Gap-down stop-loss: if open < stop_loss, IMMEDIATE market sell at open
  - T+1 sell restriction: shares bought today cannot be sold today
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple


class OrderState(Enum):
    PENDING = 'pending'          # Submitted, not yet filled
    PARTIAL_FILL = 'partial'     # Partially filled
    FILLED = 'filled'            # Completely filled
    CANCELED = 'canceled'        # Canceled before fill
    REJECTED = 'rejected'        # Rejected by broker
    GAP_STOPPED = 'gap_stopped'  # 跳空击穿强制止损


class OrderType(Enum):
    BUY = 'buy'
    SELL = 'sell'


@dataclass
class Order:
    """A single order in the OMS."""
    order_id: str
    symbol: str
    order_type: OrderType
    quantity: int                # Desired shares
    filled_qty: int = 0          # Actually filled
    limit_price: Optional[float] = None  # Optional limit price
    state: OrderState = OrderState.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    notes: str = ''


@dataclass
class Position:
    """Track holdings with T+1 sell restriction."""
    symbol: str
    total_shares: int = 0        # Total shares held
    available_today: int = 0     # Shares sellable TODAY (bought before today)
    locked_until_t1: int = 0     # Shares bought today, unlockable tomorrow
    avg_cost: float = 0.0

    @property
    def is_fully_locked(self) -> bool:
        return self.total_shares > 0 and self.available_today == 0

    def can_sell(self, qty: int) -> bool:
        """Check if qty shares can be sold today (T+1 compliant)."""
        return qty <= self.available_today

    def lock_t1(self, qty: int, cost: float) -> None:
        """Lock newly bought shares until next trading day."""
        old_cost = self.avg_cost * self.total_shares
        new_cost = old_cost + cost * qty
        self.total_shares += qty
        self.locked_until_t1 += qty
        # avg_cost updated but available_today unchanged (can't sell today)
        self.avg_cost = new_cost / self.total_shares if self.total_shares > 0 else 0

    def unlock_t1(self) -> None:
        """Call at start of each trading day to release T+1 lock."""
        self.available_today += self.locked_until_t1
        self.locked_until_t1 = 0

    def reduce(self, qty: int) -> None:
        """Reduce position after a sale."""
        qty = min(qty, self.total_shares)
        self.total_shares -= qty
        self.available_today -= min(qty, self.available_today)
        if self.available_today < 0:
            self.available_today = 0


class OrderStateMachine:
    """
    Lightweight OMS for A-share T+1 compliance.

    Tracks per-symbol positions with sell availability constraints
    and handles gap-down forced stop-loss scenarios.
    """

    def __init__(self):
        self.orders: Dict[str, Order] = {}       # order_id → Order
        self.positions: Dict[str, Position] = {} # symbol → Position
        self.order_counter: int = 0

    # ---- Position Management ----

    def get_position(self, symbol: str) -> Position:
        """Return or create position for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def daily_unlock(self) -> None:
        """Call at market open: release T+1 locked shares for all positions."""
        for pos in self.positions.values():
            pos.unlock_t1()

    # ---- Order Entry ----

    def submit_buy(self, symbol: str, qty: int,
                   limit_price: Optional[float] = None) -> Order:
        """Submit a buy order. T+1 lock applied on fill."""
        self.order_counter += 1
        oid = f"BUY_{symbol}_{self.order_counter}"
        order = Order(order_id=oid, symbol=symbol,
                     order_type=OrderType.BUY, quantity=qty,
                     limit_price=limit_price)
        self.orders[oid] = order
        return order

    def submit_sell(self, symbol: str, qty: int,
                    limit_price: Optional[float] = None) -> Tuple[bool, Order]:
        """
        Submit a sell order with T+1 check.

        Returns:
            (success, order): False if sell violates T+1
        """
        pos = self.get_position(symbol)
        if not pos.can_sell(qty):
            order = Order(order_id=f"SELL_{symbol}_{self.order_counter+1}",
                         symbol=symbol, order_type=OrderType.SELL,
                         quantity=qty, limit_price=limit_price,
                         state=OrderState.REJECTED,
                         notes=f"T+1 violation: have {pos.available_today}, need {qty}")
            return False, order

        self.order_counter += 1
        oid = f"SELL_{symbol}_{self.order_counter}"
        order = Order(order_id=oid, symbol=symbol,
                     order_type=OrderType.SELL, quantity=qty,
                     limit_price=limit_price)
        self.orders[oid] = order
        return True, order

    # ---- Gap-Down Forced Stop-Loss ----

    def check_gap_down(self, symbol: str, open_price: float,
                       stop_loss: float) -> Optional[Order]:
        """
        Gap-down protection: if open_price < stop_loss, force immediate sell
        of ALL available shares at open price.

        Returns:
            Order if gap-down triggered, None otherwise
        """
        pos = self.get_position(symbol)
        if pos.available_today <= 0:
            return None  # Nothing to sell

        if open_price < stop_loss:
            self.order_counter += 1
            oid = f"GAPSTOP_{symbol}_{self.order_counter}"
            order = Order(
                order_id=oid, symbol=symbol,
                order_type=OrderType.SELL,
                quantity=pos.available_today,
                limit_price=open_price,  # Market sell at open
                state=OrderState.GAP_STOPPED,
                notes=f"Gap-down: open {open_price:.2f} < stop {stop_loss:.2f}"
            )
            self.orders[oid] = order
            # Immediately apply fill
            self._apply_fill(order, open_price)
            return order
        return None

    # ---- Fill Application ----

    def apply_fill(self, order_id: str, fill_price: float,
                   fill_qty: Optional[int] = None) -> bool:
        """Mark an order as filled (or partially filled)."""
        order = self.orders.get(order_id)
        if order is None or order.state in (OrderState.CANCELED, OrderState.REJECTED):
            return False
        return self._apply_fill(order, fill_price, fill_qty)

    def _apply_fill(self, order: Order, fill_price: float,
                    fill_qty: Optional[int] = None) -> bool:
        qty = fill_qty or (order.quantity - order.filled_qty)
        qty = min(qty, order.quantity - order.filled_qty)
        if qty <= 0:
            return False

        order.filled_qty += qty
        pos = self.get_position(order.symbol)

        if order.order_type == OrderType.BUY:
            # T+1: lock newly bought shares
            pos.lock_t1(qty, fill_price)
        else:
            pos.reduce(qty)

        # Update state
        if order.filled_qty >= order.quantity:
            order.state = OrderState.FILLED
            order.filled_at = datetime.now()
        else:
            order.state = OrderState.PARTIAL_FILL

        return True

    # ---- Query ----

    def sellable_quantity(self, symbol: str) -> int:
        return self.get_position(symbol).available_today

    def is_sellable_today(self, symbol: str, qty: int) -> bool:
        return self.get_position(symbol).can_sell(qty)

    def position_summary(self) -> str:
        lines = ["OMS Positions:"]
        for sym, pos in sorted(self.positions.items()):
            if pos.total_shares > 0:
                lines.append(f"  {sym}: total={pos.total_shares} "
                           f"avail={pos.available_today} "
                           f"locked={pos.locked_until_t1} "
                           f"cost={pos.avg_cost:.2f}")
        return '\n'.join(lines) if len(lines) > 1 else "OMS: No positions"


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    oms = OrderStateMachine()

    # Day 1: Buy 1000 shares of 600519
    oms.submit_buy('600519', 1000)
    oms.apply_fill('BUY_600519_1', 1800.0)
    print(f"After buy: sellable={oms.sellable_quantity('600519')} (should be 0, T+1)")

    # Day 2: Unlock
    oms.daily_unlock()
    print(f"After unlock: sellable={oms.sellable_quantity('600519')} (should be 1000)")

    # Gap-down test
    stop = 1700.0
    open_px = 1650.0  # Below stop!
    gap_order = oms.check_gap_down('600519', open_px, stop)
    if gap_order:
        print(f"Gap-down triggered: {gap_order.order_id} "
              f"qty={gap_order.quantity} state={gap_order.state}")
        print(f"After gap-stop: sellable={oms.sellable_quantity('600519')} (should be 0)")

    print("\n" + oms.position_summary())
    print("\nOMS state machine: OK")
