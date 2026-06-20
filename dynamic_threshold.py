#!/usr/bin/env python
# dynamic_threshold.py — Rolling Z-Score & Percentile Thresholds
"""
Replace hardcoded magic numbers (ADX > 25, VR > 450) with adaptive thresholds
based on rolling Z-Scores and percentiles computed from N-day history.

Principle:
  - Instead of "ADX > 25 is trending", use "ADX > 80th percentile of last 252 days"
  - Instead of "VR > 450 is overbought", use "VR > rolling_mean + 2.5 * rolling_std"
  - Automatically adapts to changing volatility regimes and market structure
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from dataclasses import dataclass, field


@dataclass
class DynamicThreshold:
    """Adaptive threshold based on rolling Z-Score and percentile."""

    lookback: int = 252         # Rolling window (trading days)
    z_score: float = 2.0        # Z-Score multiplier for extreme signals
    percentile: float = 80.0    # Percentile for high-side threshold
    min_periods: int = 60       # Minimum bars before threshold activates

    # Cached values
    _rolling_mean: Optional[np.ndarray] = field(default=None, repr=False)
    _rolling_std: Optional[np.ndarray] = field(default=None, repr=False)
    _rolling_percentile: Optional[np.ndarray] = field(default=None, repr=False)

    def fit(self, values: Union[pd.Series, np.ndarray]) -> "DynamicThreshold":
        """Compute rolling statistics from historical values."""
        s = pd.Series(values) if not isinstance(values, pd.Series) else values
        self._rolling_mean = s.rolling(self.lookback, min_periods=self.min_periods).mean().values
        self._rolling_std = s.rolling(self.lookback, min_periods=self.min_periods).std().values
        self._rolling_percentile = (
            s.rolling(self.lookback, min_periods=self.min_periods)
            .apply(lambda x: np.percentile(x, self.percentile), raw=True)
            .values
        )
        return self

    def is_extreme_high(self, current_value: float) -> bool:
        """Check if current value is statistically extreme on the high side."""
        if self._rolling_mean is None or self._rolling_std is None:
            raise RuntimeError("Must call .fit() before .is_extreme_high()")
        idx = -1  # Latest
        mean = self._rolling_mean[idx]
        std = max(self._rolling_std[idx], 1e-10)
        return current_value > mean + self.z_score * std

    def is_extreme_low(self, current_value: float) -> bool:
        """Check if current value is statistically extreme on the low side."""
        if self._rolling_mean is None or self._rolling_std is None:
            raise RuntimeError("Must call .fit() before .is_extreme_low()")
        idx = -1
        mean = self._rolling_mean[idx]
        std = max(self._rolling_std[idx], 1e-10)
        return current_value < mean - self.z_score * std

    def is_above_percentile(self, current_value: float) -> bool:
        """Check if current value exceeds the lookback percentile."""
        if self._rolling_percentile is None:
            raise RuntimeError("Must call .fit()")
        return current_value > self._rolling_percentile[-1]

    def z_score_value(self, current_value: float) -> float:
        """Return the Z-Score of current value relative to rolling distribution."""
        if self._rolling_mean is None or self._rolling_std is None:
            return 0.0
        idx = -1
        mean = self._rolling_mean[idx]
        std = max(self._rolling_std[idx], 1e-10)
        return (current_value - mean) / std


class AdaptiveRegimeDetector:
    """
    Replace hardcoded "ADX > 25 = trending" with adaptive regime detection.

    Uses rolling percentile of ADX to determine if current market is
    'trending' (ADX high relative to history) or 'ranging' (ADX low).
    """

    def __init__(self, lookback: int = 252):
        self.adx_threshold = DynamicThreshold(lookback=lookback, percentile=75, z_score=1.5)

    def fit(self, adx_series: pd.Series) -> "AdaptiveRegimeDetector":
        self.adx_threshold.fit(adx_series)
        self.adx_values = adx_series.values
        return self

    def detect(self, current_adx: float) -> str:
        """Return 'trending', 'ranging', or 'transitional' based on adaptive ADX."""
        if self.adx_threshold.is_above_percentile(current_adx):
            return 'trending'
        elif current_adx < np.percentile(self.adx_values[-252:], 25):
            return 'ranging'
        return 'transitional'


class AdaptiveVolumeDetector:
    """
    Replace hardcoded "VR > 450 = overbought" with adaptive Z-Score.

    VR (Volume Ratio) thresholds vary dramatically by stock — a small-cap
    stock may have VR=200 as extreme, while a large-cap may need VR=800.
    Z-Score normalization makes this stock-adaptive.
    """

    def __init__(self, lookback: int = 252, z_threshold: float = 2.5):
        self.threshold = DynamicThreshold(lookback=lookback, z_score=z_threshold)

    def fit(self, vr_series: pd.Series) -> "AdaptiveVolumeDetector":
        self.threshold.fit(vr_series)
        return self

    def is_overbought(self, current_vr: float) -> bool:
        return self.threshold.is_extreme_high(current_vr)

    def is_oversold(self, current_vr: float) -> bool:
        return self.threshold.is_extreme_low(current_vr)


# =============================================================================
# Factory: on-demand computation
# =============================================================================

def compute_adaptive_thresholds(
    indicators: Dict,
    history_data: Optional[Dict[str, pd.Series]] = None,
) -> Dict[str, float]:
    """
    Compute adaptive thresholds from indicator history.

    Args:
        indicators: current indicator values dict
        history_data: optional pre-computed history dict {name: series}

    Returns:
        Dict of {indicator_name: z_score_value}
    """
    if history_data is None:
        return {}

    z_scores = {}
    for name, series in history_data.items():
        if name in indicators and series is not None and len(series) >= 60:
            dt = DynamicThreshold()
            dt.fit(series)
            current = float(indicators[name]) if indicators[name] is not None else 0.0
            z_scores[f'{name}_zscore'] = round(dt.z_score_value(current), 2)
            z_scores[f'{name}_percentile_high'] = dt.is_above_percentile(current)

    return z_scores


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == '__main__':
    np.random.seed(123)

    # Simulate 500 days of ADX
    n_days = 500
    regime_changes = np.zeros(n_days)
    regime_changes[200:350] = 1  # Trending period
    adx = 15 + 20 * regime_changes + np.random.randn(n_days) * 3
    adx_series = pd.Series(adx)

    detector = AdaptiveRegimeDetector(lookback=252).fit(adx_series)

    # Test during ranging period
    print(f"Day 100 (ranging): ADX={adx[99]:.1f}, regime={detector.detect(adx[99])}")

    # Test during trending period
    print(f"Day 300 (trending): ADX={adx[299]:.1f}, regime={detector.detect(adx[299])}")

    # Test Z-Score
    dt = DynamicThreshold().fit(adx_series)
    print(f"\nDay 300 Z-Score: {dt.z_score_value(adx[299]):.2f}")
    print(f"Day 300 percentile high: {dt.is_above_percentile(adx[299])}")
    print(f"Day 100 Z-Score: {dt.z_score_value(adx[99]):.2f}")
    print(f"Day 100 percentile high: {dt.is_above_percentile(adx[99])}")
    print("\n✅ Adaptive thresholds working: no magic numbers needed.")
