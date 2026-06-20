#!/usr/bin/env python
# execution_algos.py — VWAP U-Curve Slicing + T+0 Intraday Grid
"""
Institutional execution algorithms for A-share microstructure.

1. VWAP U-Curve: Split large orders across 4 time buckets following
   the A-share intraday volume U-pattern (high open, mid-day lull, high close).
   Each slice gets its own Almgren-Chriss impact estimate.

2. T+0 Grid: For locked positions (bought yesterday → sellable today),
   when intraday amplitude > 4%, execute symmetric buy-low/sell-high
   grid orders to reduce holding cost.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# A-Share Intraday Volume U-Curve
# =============================================================================

# Normalized volume weights per 30-min bucket (A-share empirical)
# 09:30-10:00  10:00-10:30  10:30-11:00  11:00-11:30
# 13:00-13:30  13:30-14:00  14:00-14:30  14:30-15:00
A_SHARE_VOLUME_PROFILE = np.array([
    0.20,  # 09:30-10:00  Morning burst (open auction + first trades)
    0.14,  # 10:00-10:30
    0.10,  # 10:30-11:00
    0.08,  # 11:00-11:30  Mid-day lull
    0.08,  # 13:00-13:30  Afternoon start
    0.12,  # 13:30-14:00
    0.13,  # 14:00-14:30
    0.15,  # 14:30-15:00  Closing rush
], dtype=np.float64)

# Normalize to sum=1
A_SHARE_VOLUME_PROFILE /= A_SHARE_VOLUME_PROFILE.sum()


# =============================================================================
# 1. VWAP U-Curve Order Slicer
# =============================================================================

@dataclass
class VWAPSlicer:
    """
    Split large orders into time-weighted slices following A-share U-curve.

    When order_value > 1M RMB (institutional size), slice into 4 buckets:
      - Bucket 1 (09:30-10:30): 34% of volume → 34% of order
      - Bucket 2 (10:30-13:00): 26% → 26% of order
      - Bucket 3 (13:00-14:00): 20% → 20% of order
      - Bucket 4 (14:00-15:00): 20% → 20% of order

    Each slice gets its own Almgren-Chriss impact. The VWAP price is the
    volume-weighted average of all slice execution prices.
    """

    large_order_threshold: float = 1_000_000  # 100万 RMB
    n_buckets: int = 4

    # Bucket definitions: (start_hour, end_hour, volume_weight)
    BUCKETS = [
        (9.5, 10.5, 0.34),   # 09:30-10:30
        (10.5, 13.0, 0.26),  # 10:30-13:00 (跨越午休)
        (13.0, 14.0, 0.20),  # 13:00-14:00
        (14.0, 15.0, 0.20),  # 14:00-15:00
    ]

    def should_slice(self, order_value: float) -> bool:
        """Determine if order is large enough to warrant slicing."""
        return order_value >= self.large_order_threshold

    def compute_slices(
        self,
        order_shares: int,
        entry_price: float,
        adv_shares: int,
        atr_pct: float = 0.03,
    ) -> List[Dict]:
        """
        Generate VWAP slices with per-slice impact estimates.

        Args:
            order_shares: total shares to buy
            entry_price: reference price (usually open)
            adv_shares: average daily volume
            atr_pct: daily volatility estimate

        Returns:
            List of slice dicts with qty, est_price, impact_bps, weight
        """
        from market_microstructure import AlmgrenChrissImpact

        impact_model = AlmgrenChrissImpact()
        slices = []
        remaining = order_shares

        for i, (start_h, end_h, vol_weight) in enumerate(self.BUCKETS):
            if i == self.BUCKETS.__len__() - 1:
                slice_qty = remaining
            else:
                slice_qty = int(order_shares * vol_weight / 100) * 100

            slice_qty = min(slice_qty, remaining)
            if slice_qty <= 0:
                continue

            # Per-slice Almgren-Chriss impact
            impact_bps = impact_model.impact_bps(slice_qty, adv_shares, atr_pct)
            est_price = entry_price * (1 + impact_bps / 10000)

            slices.append({
                'slice_id': i + 1,
                'start_hour': start_h,
                'end_hour': end_h,
                'quantity': slice_qty,
                'volume_weight': round(vol_weight, 2),
                'est_price': round(est_price, 3),
                'impact_bps': round(impact_bps, 1),
            })
            remaining -= slice_qty

        return slices

    def vwap_price(self, slices: List[Dict]) -> float:
        """Compute VWAP from slices: Σ(q_i * p_i) / Σq_i."""
        total_qty = sum(s['quantity'] for s in slices)
        if total_qty == 0:
            return 0.0
        vwap = sum(s['quantity'] * s['est_price'] for s in slices) / total_qty
        return round(vwap, 3)

    def total_impact_savings(
        self,
        order_shares: int,
        adv_shares: int,
        atr_pct: float = 0.03,
    ) -> Dict:
        """Compare single-order vs VWAP-sliced execution costs."""
        from market_microstructure import AlmgrenChrissImpact
        impact_model = AlmgrenChrissImpact()

        single_impact = impact_model.impact_bps(order_shares, adv_shares, atr_pct)
        slices = self.compute_slices(order_shares, 10.0, adv_shares, atr_pct)

        if not slices:
            return {'single_impact_bps': single_impact, 'vwap_impact_bps': single_impact,
                    'savings_bps': 0}

        # Weighted average impact
        total_qty = sum(s['quantity'] for s in slices)
        avg_impact = (sum(s['quantity'] * s['impact_bps'] for s in slices) /
                      max(total_qty, 1))

        return {
            'single_impact_bps': single_impact,
            'vwap_impact_bps': round(avg_impact, 1),
            'savings_bps': round(single_impact - avg_impact, 1),
            'n_slices': len(slices),
        }


# =============================================================================
# 2. Intraday T+0 Grid Algorithm
# =============================================================================

@dataclass
class T0GridAlgo:
    """
    Intraday T+0 cost reduction grid for A-shares.

    Trigger conditions (ALL must be met):
      1. Stock has sellable position (bought ≥ 1 day ago, T+1 unlocked)
      2. Intraday amplitude (high-low)/open > 4%
      3. Available cash ≥ cost of one grid buy
      4. Trend circuit breaker NOT tripped (ADX>30 & DMI->DMI+ → buy disabled)

    Logic:
      - Compute intraday VWAP as reference
      - Place buy order at VWAP - 1.5 × (amplitude/4)  → lower rail
      - Place sell order at VWAP + 1.5 × (amplitude/4) → upper rail
      - Same quantity on both sides → net position unchanged
      - Profit = (sell_price - buy_price) × qty - fees
    """

    amplitude_threshold: float = 0.04    # 4% intraday amplitude
    grid_distance_atr: float = 1.5       # Distance from VWAP in amplitude units
    fee_rate: float = 0.0015             # Round-trip cost (印花税+佣金 ≈ 0.15%)
    max_rounds: int = 2                  # Max grid rounds per day

    # ---- Trend Circuit Breaker Settings ----
    cb_adx_threshold: float = 30.0       # ADX above this → strong trend
    cb_macd_fast: int = 12               # MACD fast period for 15-min bar proxy
    cb_macd_slow: int = 26

    def _detect_strong_downtrend(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        df_15min: pd.DataFrame = None,
    ) -> bool:
        """
        Detect strong unilateral downtrend where buying is catching a falling knife.

        Returns True if:
          - ADX > 30 AND DMI- > DMI+  (strong bearish trend)
          - OR MACD histogram is negative and accelerating (proxy for 15-min bar)
        """
        if df_15min is not None and len(df_15min) >= 26:
            h, l, c = df_15min['high'], df_15min['low'], df_15min['close']

            # True Range & Directional Movement (14-period on 15-min bars)
            tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))],
                          axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]

            up_move = h.diff()
            down_move = -l.diff()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            plus_di = 100 * pd.Series(plus_dm).rolling(14).mean().iloc[-1] / max(atr, 1e-10)
            minus_di = 100 * pd.Series(minus_dm).rolling(14).mean().iloc[-1] / max(atr, 1e-10)
            dx = 100 * abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-10)
            adx = pd.Series(dx).rolling(14).mean().iloc[-1]

            if adx > self.cb_adx_threshold and minus_di > plus_di:
                return True

        # Fallback: OHLC-based heuristic for strong bearish day
        # (open ≈ high, close ≈ low, range entirely bearish)
        if open_price > 0:
            upper_wick = (high - max(open_price, close)) / max(open_price * 0.01, 0.01)
            lower_wick = (min(open_price, close) - low) / max(open_price * 0.01, 0.01)
            body_pct = (close - open_price) / max(open_price * 0.01, 0.01)
            # Strong bearish marubozu: long red body, no lower wick
            if body_pct < -0.04 and lower_wick < 0.3:
                return True

        return False

    def should_disable_buy_grid(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        df_15min: pd.DataFrame = None,
    ) -> Tuple[bool, str]:
        """
        Trend circuit breaker: disable buy-side of T+0 grid in strong downtrend.

        Returns (disabled, reason).
        """
        if self._detect_strong_downtrend(open_price, high, low, close, df_15min):
            return True, 'Strong downtrend (ADX>30, DMI->DMI+) — buying disabled to avoid catching falling knife'
        return False, ''

    def should_activate(
        self,
        open_price: float,
        high: float,
        low: float,
        sellable_shares: int,
        available_cash: float,
    ) -> bool:
        """Check if T+0 grid is warranted."""
        if sellable_shares < 100:
            return False
        amplitude = (high - low) / max(open_price, 0.01)
        if amplitude < self.amplitude_threshold:
            return False
        # Need enough cash for at least one buy
        if available_cash < low * 100:
            return False
        return True

    def generate_grid(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        sellable_shares: int,
        available_cash: float,
        current_position_cost: float,
        df_15min: pd.DataFrame = None,
    ) -> List[Dict]:
        """
        Generate symmetric T+0 grid orders.

        When trend circuit breaker is tripped:
          - Sell orders: ALLOWED (reduce exposure in downtrend)
          - Buy orders: BLOCKED (prevent catching falling knife)
        """
        amplitude = (high - low) / max(open_price, 0.01)
        ref_price = (open_price + high + low + close) / 4  # OHLC average ≈ VWAP proxy

        # Grid spacing
        half_range = open_price * amplitude / 4 * self.grid_distance_atr
        buy_price = max(low, ref_price - half_range)
        sell_price = min(high, ref_price + half_range)

        # Quantity
        grid_qty = int(sellable_shares * 0.20 / 100) * 100
        grid_qty = max(100, grid_qty)
        buy_cost = buy_price * grid_qty
        if buy_cost > available_cash:
            grid_qty = int(available_cash / buy_price / 100) * 100
        if grid_qty < 100:
            return []

        # ---- Trend circuit breaker ----
        buy_disabled, breaker_reason = self.should_disable_buy_grid(
            open_price, high, low, close, df_15min)

        orders = []
        for r in range(self.max_rounds):
            round_qty = max(100, grid_qty - r * 100)
            if round_qty < 100:
                break

            orders.append({
                'round': r + 1,
                'action': 'T+0_grid',
                'buy_price': round(buy_price, 2),
                'sell_price': round(sell_price, 2),
                'quantity': round_qty,
                'buy_disabled': buy_disabled,
                'breaker_reason': breaker_reason if buy_disabled else '',
                'gross_profit_per_share': round(sell_price - buy_price, 3),
                'net_profit_total': round((sell_price - buy_price) * round_qty *
                                         (1 - self.fee_rate), 2),
            })

        return orders

    def simulate_backtest(
        self,
        df_intraday: pd.DataFrame = None,
        open_price: float = None,
        high: float = None,
        low: float = None,
        close: float = None,
        sellable_shares: int = 0,
        available_cash: float = 0,
        current_cost: float = 0,
    ) -> Tuple[float, List[Dict]]:
        """
        Run T+0 simulation and return (net_pnl, trade_log).

        For backtest mode with only OHLC data (no intraday bars):
          - If the amplitude is large enough, assume both grid orders
            can be executed at their target prices.
          - This is an upper-bound estimate — real execution would be worse.
        """
        if open_price is None:
            return 0.0, []

        if not self.should_activate(open_price, high or open_price,
                                    low or open_price, sellable_shares,
                                    available_cash):
            return 0.0, []

        orders = self.generate_grid(
            open_price, high or open_price, low or open_price,
            close or open_price, sellable_shares, available_cash,
            current_cost,
        )

        if not orders:
            return 0.0, []

        total_pnl = sum(o['net_profit_total'] for o in orders)

        # Update position cost basis
        # Buy at low → reduces avg cost, Sell at high → realized profit
        new_cost = current_cost
        for o in orders:
            qty = o['quantity']
            # Net effect: bought qty at buy_price, sold qty at sell_price
            # Position unchanged, realized profit = (sell - buy - fees) × qty
            pass  # Cost basis update handled by caller

        return total_pnl, orders


# =============================================================================
# 3. Integration helper
# =============================================================================

def execute_with_vwap(
    order_value: float,
    order_shares: int,
    entry_price: float,
    adv_shares: int,
    atr_pct: float = 0.03,
) -> Dict:
    """
    Execute an order using VWAP if large, otherwise simple entry.

    Returns dict with:
      - execution_price: effective fill price
      - slices: VWAP slice details (if applicable)
      - is_vwap: True if order was sliced
    """
    slicer = VWAPSlicer()

    if not slicer.should_slice(order_value):
        from market_microstructure import AlmgrenChrissImpact
        impact = AlmgrenChrissImpact()
        impact_bps = impact.impact_bps(order_shares, adv_shares, atr_pct)
        exec_price = entry_price * (1 + impact_bps / 10000)
        return {
            'execution_price': round(exec_price, 3),
            'slices': [],
            'is_vwap': False,
            'impact_bps': impact_bps,
        }

    slices = slicer.compute_slices(order_shares, entry_price, adv_shares, atr_pct)
    vwap = slicer.vwap_price(slices)
    avg_impact = sum(s['impact_bps'] for s in slices) / max(len(slices), 1)

    return {
        'execution_price': vwap,
        'slices': slices,
        'is_vwap': True,
        'impact_bps': round(avg_impact, 1),
    }


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    print("=== VWAP U-Curve Slicer ===")
    slicer = VWAPSlicer()
    slices = slicer.compute_slices(50000, 10.0, 1_000_000)
    for s in slices:
        print(f"  Slice {s['slice_id']}: {s['quantity']}sh @ "
              f"{s['est_price']:.3f} ({s['impact_bps']}bp)")
    print(f"  VWAP: {slicer.vwap_price(slices):.3f}")

    savings = slicer.total_impact_savings(50000, 1_000_000)
    print(f"  Single: {savings['single_impact_bps']}bp → "
          f"VWAP: {savings['vwap_impact_bps']}bp "
          f"(save {savings['savings_bps']}bp)")

    print("\n=== T+0 Grid Algo ===")
    t0 = T0GridAlgo()
    activated = t0.should_activate(10.0, 10.6, 9.8, 5000, 200000)
    print(f"  Amplitude 8%: should_activate={activated}")

    if activated:
        orders = t0.generate_grid(10.0, 10.6, 9.8, 10.2, 5000, 200000, 10.0)
        for o in orders:
            print(f"  Round {o['round']}: buy {o['quantity']}@{o['buy_price']} "
                  f"sell @{o['sell_price']} net={o['net_profit_total']:.1f}")

    print("\nAll execution algos: OK")
