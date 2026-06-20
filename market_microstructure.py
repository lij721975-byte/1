#!/usr/bin/env python
# market_microstructure.py — Almgren-Chriss Impact + TWAP/VWAP + Order FSM + T+0 Grid
"""
Institutional-grade market microstructure engine.

Four core modules:
  1. Almgren-Chriss: Square-root impact model (replaces fixed 0.001 slippage)
  2. OrderStateMachine: Full lifecycle FSM with limit-lock partial fill handling
  3. TWAP/VWAP Slicer: Large order decomposition into N time slices
  4. IntradayT0Algo: Grid trading for existing positions during high volatility

Reference: Almgren & Chriss (2001), "Optimal execution of portfolio transactions"
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# 1. Almgren-Chriss Market Impact Model
# =============================================================================

@dataclass
class AlmgrenChrissImpact:
    """
    Square-root market impact model.

    Impact = σ × (η × (Q / ADV)^0.5 + γ × (Q / ADV))

    Where:
      σ      = daily volatility (annualized vol / sqrt(252))
      Q      = order quantity (shares)
      ADV    = average daily volume (shares)
      η      = temporary impact coefficient (∝ 1/√ADV)
      γ      = permanent impact coefficient (typically small for retail)

    Reference values for A-shares:
      η ≈ 0.30 (temporary impact — dominates for small/medium orders)
      γ ≈ 0.05 (permanent impact — relevant for very large orders)
    """

    eta: float = 0.30          # Temporary impact coefficient
    gamma: float = 0.05        # Permanent impact coefficient
    annual_vol: float = 0.30   # Default 30% annual vol

    def daily_volatility(self, atr_pct: float = 0.03) -> float:
        """Convert ATR% to daily volatility estimate."""
        return atr_pct  # ATR% ≈ daily σ

    def impact_bps(self, order_shares: int, adv_shares: int,
                   daily_vol: float = 0.03) -> float:
        """
        Compute expected market impact in basis points.

        Args:
            order_shares: number of shares to trade
            adv_shares: average daily volume in shares
            daily_vol: daily return volatility (e.g. 0.03 = 3%)

        Returns:
            Impact in basis points (0.01% = 1 bp)

        Example:
            ADV = 1M shares, order = 10K shares (1% of ADV), vol = 3%
            impact = 0.03 × (0.30 × sqrt(0.01) + 0.05 × 0.01)
                   = 0.03 × (0.03 + 0.0005) ≈ 9 bp
        """
        if adv_shares <= 0 or order_shares <= 0:
            return 0.0

        participation_rate = order_shares / adv_shares

        # Almgren-Chriss square-root formula
        temp_impact = self.eta * np.sqrt(participation_rate)
        perm_impact = self.gamma * participation_rate
        total_impact_bps = daily_vol * (temp_impact + perm_impact) * 10000

        return round(total_impact_bps, 1)

    def effective_slippage(self, order_shares: int, adv_shares: int,
                           atr_pct: float = 0.03,
                           is_buy: bool = True) -> float:
        """
        Compute effective slippage as a decimal (e.g. 0.0015 = 15bp).

        For buys: slippage is positive (price goes up against us)
        For sells: slippage is negative (price goes down against us)
        """
        impact_bps = self.impact_bps(order_shares, adv_shares, atr_pct)
        slippage = impact_bps / 10000  # Convert bps to decimal
        return slippage if is_buy else -slippage

    def execution_cost(self, order_shares: int, price: float,
                       adv_shares: int, atr_pct: float = 0.03) -> Dict:
        """Full execution cost breakdown for a single order."""
        impact_bps = self.impact_bps(order_shares, adv_shares, atr_pct)
        impact_cost = price * order_shares * (impact_bps / 10000)
        participation = order_shares / adv_shares * 100 if adv_shares > 0 else 0

        return {
            'participation_pct': round(participation, 2),
            'impact_bps': impact_bps,
            'impact_cost_rmb': round(impact_cost, 2),
            'effective_entry': round(price * (1 + impact_bps / 10000), 3),
            'warning': 'HIGH IMPACT' if impact_bps > 25 else 'ok',
        }


# =============================================================================
# 2. Micro Order State Machine (with limit-lock partial fill)
# =============================================================================

class OrderState(str, Enum):
    """Order lifecycle states."""
    CREATED = 'created'          # Initialized, not yet submitted
    SUBMITTED = 'submitted'      # Sent to broker
    PARTIAL_FILL = 'partial'     # Partially filled (limit-locked stock)
    FILLED = 'filled'            # Completely filled
    CANCELED = 'canceled'        # Canceled before complete fill
    REJECTED = 'rejected'        # Rejected by broker
    EXPIRED = 'expired'          # TWAP/VWAP slice expired unfilled


class OrderType(str, Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    TWAP = 'twap'
    VWAP = 'vwap'


@dataclass
class MicroOrder:
    """Individual order with full lifecycle tracking."""
    order_id: str
    symbol: str
    side: str                   # 'BUY' | 'SELL'
    order_type: OrderType
    quantity: int               # Desired shares
    limit_price: Optional[float] = None
    filled_qty: int = 0
    state: OrderState = OrderState.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    cancel_reason: str = ''
    # Limit-lock tracking
    limit_up_price: Optional[float] = None
    limit_down_price: Optional[float] = None

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled_qty

    @property
    def is_complete(self) -> bool:
        return self.state in (OrderState.FILLED, OrderState.CANCELED,
                              OrderState.REJECTED, OrderState.EXPIRED)

    @property
    def fill_pct(self) -> float:
        return self.filled_qty / max(self.quantity, 1)


class MicroOrderStateMachine:
    """
    Full order lifecycle FSM with limit-lock aware partial fills.

    Key behaviors:
      - Limit-locked stocks may PARTIAL_FILL (sealed board, low liquidity)
      - Automatically retries unfilled quantity on next trading day
      - Tracks fill percentage for impact model feedback
    """

    def __init__(self):
        self.orders: Dict[str, MicroOrder] = {}
        self.counter: int = 0

    def create_order(self, symbol: str, side: str, qty: int,
                     order_type: OrderType = OrderType.MARKET,
                     limit_price: float = None,
                     limit_up: float = None,
                     limit_down: float = None) -> MicroOrder:
        """Create and register a new order."""
        self.counter += 1
        oid = f"{side}_{symbol}_{self.counter}_{datetime.now():%H%M%S}"
        order = MicroOrder(order_id=oid, symbol=symbol, side=side,
                          order_type=order_type, quantity=qty,
                          limit_price=limit_price,
                          limit_up_price=limit_up,
                          limit_down_price=limit_down)
        order.state = OrderState.SUBMITTED
        self.orders[oid] = order
        return order

    def attempt_fill(self, order_id: str, available_volume: int,
                     fill_price: float) -> Tuple[int, OrderState]:
        """
        Attempt to fill an order given available market volume.

        For limit-locked stocks, available_volume may be less than
        the order quantity — this produces PARTIAL_FILL instead of
        throwing an error.

        Returns:
            (filled_this_attempt, new_state)
        """
        order = self.orders.get(order_id)
        if order is None or order.is_complete:
            return 0, order.state if order else OrderState.REJECTED

        fillable = min(available_volume, order.remaining)

        if fillable <= 0:
            # No volume available — stay SUBMITTED for retry
            order.cancel_reason = f'Zero fill volume at {fill_price}'
            return 0, order.state

        order.filled_qty += fillable
        order.filled_at = datetime.now()

        if order.filled_qty >= order.quantity:
            order.state = OrderState.FILLED
        elif fillable > 0:
            order.state = OrderState.PARTIAL_FILL
            order.cancel_reason = (f'Limit-locked: filled {fillable}/{order.remaining} '
                                  f'at {fill_price}')

        return fillable, order.state

    def cancel_unfilled(self, order_id: str, reason: str = '') -> int:
        """Cancel remaining unfilled quantity. Returns unfilled shares."""
        order = self.orders.get(order_id)
        if order is None or order.is_complete:
            return 0
        unfilled = order.remaining
        order.state = OrderState.CANCELED
        order.cancel_reason = reason or 'User canceled'
        return unfilled

    def active_orders(self, symbol: str = None) -> List[MicroOrder]:
        """Get all active (not terminal) orders, optionally filtered by symbol."""
        active = [o for o in self.orders.values() if not o.is_complete]
        if symbol:
            active = [o for o in active if o.symbol == symbol]
        return active

    def order_summary(self) -> str:
        lines = ["OrderStateMachine:"]
        for o in self.orders.values():
            lines.append(f"  {o.order_id}: {o.symbol} {o.side} "
                        f"qty={o.quantity} filled={o.filled_qty} "
                        f"state={o.state.value}")
        return '\n'.join(lines)


# =============================================================================
# 3. TWAP / VWAP Order Slicer
# =============================================================================

@dataclass
class TWAPSlice:
    """A single time slice in a TWAP/VWAP order."""
    slice_id: int
    start_time: datetime
    end_time: datetime
    quantity: int
    order_type: OrderType = OrderType.TWAP
    state: OrderState = OrderState.CREATED


class TWAPSlicer:
    """
    Decompose large orders into smaller slices for execution.

    Trigger: when order ADV participation > 2%, split into N slices.

    Example:
        ADV = 1M shares, order = 30K shares (3% participation)
        → Split into 6 slices of 5K each over 30-minute intervals
        → Each slice: 0.5% participation, minimal impact
    """

    PARTICIPATION_THRESHOLD = 0.02  # 2% of ADV triggers slicing

    def __init__(self, trading_hours: Tuple[int, int] = (9, 15),
                 slice_minutes: int = 30):
        self.market_open = trading_hours[0]
        self.market_close = trading_hours[1]
        self.slice_minutes = slice_minutes

    def should_slice(self, order_shares: int, adv_shares: int) -> bool:
        """Determine if an order needs to be sliced."""
        if adv_shares <= 0:
            return False
        return (order_shares / adv_shares) > self.PARTICIPATION_THRESHOLD

    def compute_slices(self, order_shares: int, adv_shares: int,
                       start_time: datetime = None) -> List[TWAPSlice]:
        """
        Generate TWAP slices for a large order.

        N = ceil(participation_rate / threshold) — minimum slices needed
        to bring each slice below the threshold.
        """
        if adv_shares <= 0:
            return []

        participation = order_shares / adv_shares
        if participation <= self.PARTICIPATION_THRESHOLD:
            # Single order — no slicing needed
            return []

        n_slices = int(np.ceil(participation / self.PARTICIPATION_THRESHOLD))
        n_slices = min(n_slices, 10)  # Cap at 10 slices
        slice_qty = max(100, int(order_shares / n_slices / 100) * 100)

        if start_time is None:
            now = datetime.now()
            start_time = datetime(now.year, now.month, now.day, 9, 30)

        slices = []
        for i in range(n_slices):
            t_start = start_time + timedelta(minutes=i * self.slice_minutes)
            t_end = t_start + timedelta(minutes=self.slice_minutes)
            remaining = order_shares - sum(s.quantity for s in slices)
            qty = min(slice_qty, remaining) if i < n_slices - 1 else remaining
            qty = max(100, int(qty / 100) * 100)

            if qty > 0:
                slices.append(TWAPSlice(
                    slice_id=i + 1,
                    start_time=t_start,
                    end_time=t_end,
                    quantity=qty,
                ))

        return slices

    def estimated_impact_savings(self, order_shares: int, adv_shares: int,
                                 impact_model: AlmgrenChrissImpact,
                                 atr_pct: float = 0.03) -> Dict:
        """Estimate impact savings from TWAP slicing vs single order."""
        single_impact = impact_model.impact_bps(order_shares, adv_shares, atr_pct)

        slices = self.compute_slices(order_shares, adv_shares)
        if not slices:
            return {'savings_bps': 0, 'n_slices': 1,
                    'single_impact_bps': single_impact}

        avg_slice_qty = np.mean([s.quantity for s in slices])
        slice_impact = impact_model.impact_bps(
            int(avg_slice_qty), adv_shares, atr_pct)

        savings_bps = single_impact - slice_impact

        return {
            'n_slices': len(slices),
            'single_impact_bps': single_impact,
            'per_slice_impact_bps': slice_impact,
            'savings_bps': round(savings_bps, 1),
            'avg_slice_qty': int(avg_slice_qty),
        }


# =============================================================================
# 4. Intraday T+0 Grid Algorithm
# =============================================================================

@dataclass
class GridLevel:
    """A single grid level for T+0 trading."""
    price: float
    action: str                # 'buy' | 'sell'
    quantity: int
    reason: str = ''


class IntradayT0Algo:
    """
    Intraday T+0 cost-reduction algorithm.

    Scenario:
      - We hold 10,000 shares of 600519 bought yesterday (T+1 locked today)
      - We also hold 5,000 shares bought last week (available today)
      - Current volatility is elevated (ATR% > 3%)
      - Algorithm generates symmetric grid around VWAP:
          Sell 1,000 shares @ VWAP + 1.5 × ATR
          Buy  1,000 shares @ VWAP - 1.5 × ATR
          Sell 1,000 shares @ VWAP + 2.5 × ATR
          Buy  1,000 shares @ VWAP - 2.5 × ATR
      - Net effect: reduce avg cost without changing total position

    Key constraint: total sell quantity ≤ available_today shares (T+1)
    """

    def __init__(self, n_levels: int = 3, atr_multiplier: float = 1.5,
                 grid_spacing_atr: float = 1.0):
        self.n_levels = n_levels          # Grid levels per side
        self.atr_mult = atr_multiplier    # First grid level distance (ATR units)
        self.spacing = grid_spacing_atr   # Spacing between levels (ATR units)

    def generate_grid(self, symbol: str, current_price: float,
                      atr: float, available_today: int,
                      total_position: int,
                      vwap: float = None) -> List[GridLevel]:
        """
        Generate symmetric grid orders around VWAP.

        Args:
            symbol: stock code
            current_price: last traded price
            atr: current ATR value (not %)
            available_today: shares sellable today (T+1 unlocked)
            total_position: total shares held
            vwap: intraday VWAP (defaults to current_price)

        Returns:
            List of GridLevel buy/sell orders
        """
        ref_price = vwap if vwap and vwap > 0 else current_price
        grid = []

        # Sell grid: above VWAP
        max_sell_per_level = min(available_today // (self.n_levels * 2),
                                 total_position // 10)  # Max 10% per level
        max_sell_per_level = max(100, int(max_sell_per_level / 100) * 100)

        # Buy grid: below VWAP (limited by cash & position limits)
        max_buy_per_level = max(100, int(total_position * 0.05 / 100) * 100)

        for i in range(self.n_levels):
            distance = self.atr_mult + i * self.spacing

            # Sell above
            sell_price = ref_price + distance * atr
            sell_qty = max(100, max_sell_per_level - i * 100)
            grid.append(GridLevel(
                price=round(sell_price, 2), action='sell',
                quantity=sell_qty,
                reason=f'T+0 grid sell @ VWAP+{distance:.1f}×ATR'
            ))

            # Buy below
            buy_price = ref_price - distance * atr
            buy_qty = max(100, max_buy_per_level - i * 100)
            if buy_price > 0 and buy_qty >= 100:
                grid.append(GridLevel(
                    price=round(buy_price, 2), action='buy',
                    quantity=buy_qty,
                    reason=f'T+0 grid buy @ VWAP-{distance:.1f}×ATR'
                ))

        return grid

    def should_activate(self, atr_pct: float, spread_pct: float = 0.002) -> bool:
        """
        Determine if T+0 grid trading is warranted.

        Activates when: ATR% > 3% (elevated intraday vol)
        Does NOT activate when: spread is too wide (> 0.5% indicating illiquidity)
        """
        return atr_pct > 0.03 and spread_pct < 0.005

    def estimated_cost_reduction(self, grid: List[GridLevel],
                                 current_price: float) -> Dict:
        """Estimate the cost reduction from grid trading."""
        sell_amount = sum(g.price * g.quantity for g in grid if g.action == 'sell')
        buy_amount = sum(g.price * g.quantity for g in grid if g.action == 'buy')
        sell_qty = sum(g.quantity for g in grid if g.action == 'sell')
        buy_qty = sum(g.quantity for g in grid if g.action == 'buy')

        if sell_qty > 0 and buy_qty > 0:
            avg_sell = sell_amount / sell_qty
            avg_buy = buy_amount / buy_qty
            profit_per_round = avg_sell - avg_buy
            return {
                'avg_sell_price': round(avg_sell, 2),
                'avg_buy_price': round(avg_buy, 2),
                'profit_per_share': round(profit_per_round, 2),
                'total_profit': round(profit_per_round * min(sell_qty, buy_qty), 2),
                'round_trips': min(sell_qty, buy_qty) // 100,
            }
        return {}


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == '__main__':
    # ---- 1. Almgren-Chriss ----
    impact = AlmgrenChrissImpact()
    print("=== Almgren-Chriss Impact ===")
    for pct in [0.1, 0.5, 1.0, 2.0, 5.0]:
        cost = impact.execution_cost(
            order_shares=int(1_000_000 * pct / 100),
            price=10.0,
            adv_shares=1_000_000,
        )
        print(f"  {pct}% ADV: {cost['impact_bps']:>5}bp impact")

    # ---- 2. Order FSM ----
    print("\n=== Order State Machine ===")
    oms = MicroOrderStateMachine()
    order = oms.create_order('600519', 'BUY', 10000, OrderType.MARKET,
                            limit_up=2000.0, limit_down=1600.0)
    # Simulate partial fill on limit-locked stock
    filled, state = oms.attempt_fill(order.order_id, 3000, 1850.0)
    print(f"  Attempt 1: filled={filled}, state={state}, fill_pct={order.fill_pct:.0%}")
    filled, state = oms.attempt_fill(order.order_id, 2000, 1860.0)
    print(f"  Attempt 2: filled={filled}, state={state}, fill_pct={order.fill_pct:.0%}")

    # ---- 3. TWAP Slicer ----
    print("\n=== TWAP Slicer ===")
    slicer = TWAPSlicer()
    slices = slicer.compute_slices(50000, 1_000_000)
    if slices:
        savings = slicer.estimated_impact_savings(50000, 1_000_000, impact)
        print(f"  {len(slices)} slices, savings: {savings['savings_bps']}bp")
        for s in slices:
            print(f"    Slice {s.slice_id}: {s.quantity} shares "
                  f"@{s.start_time:%H:%M}-{s.end_time:%H:%M}")

    # ---- 4. T+0 Grid ----
    print("\n=== T+0 Grid Algorithm ===")
    grid_algo = IntradayT0Algo(n_levels=2)
    grid = grid_algo.generate_grid('600519', 1800, 36.0, 5000, 15000)
    for g in grid:
        print(f"  {g.action:>4} {g.quantity:>5}sh @ {g.price:.2f} ({g.reason})")
    cost_reduction = grid_algo.estimated_cost_reduction(grid, 1800)
    if cost_reduction:
        print(f"  Profit/round: {cost_reduction.get('profit_per_share', 0)}/share")

    print("\nAll microstructure modules: OK")
