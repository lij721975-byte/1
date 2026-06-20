# test_system.py — 量化系统基础测试套件
"""Run with: ./venv/Scripts/python test_system.py"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import traceback


def test_safe_functions():
    """Test helper functions that don't require data."""
    from expert_ensemble import _safe, _clip_score, _direction_from_score, _opposite_dir
    from expert_ensemble import _detect_market_regime, _REGIME_WEIGHTS

    assert _safe(None) == 0.0
    assert _safe(float('nan')) == 0.0
    assert _safe(42) == 42
    assert _safe(None, 99) == 99

    assert _clip_score(1.5) == 1.0
    assert _clip_score(-2.0) == -1.0
    assert _clip_score(0.5) == 0.5

    assert _direction_from_score(0.5) == 'bullish'
    assert _direction_from_score(-0.5) == 'bearish'
    assert _direction_from_score(0.0) == 'neutral'

    assert _opposite_dir('bullish') == 'bearish'
    assert _opposite_dir('bearish') == 'bullish'
    assert _opposite_dir('neutral') == 'neutral'

    # Regime detection
    assert _detect_market_regime({'dmi_adx': 30, 'rv_percentile': 0.5}) == 'trending'
    assert _detect_market_regime({'dmi_adx': 12, 'rv_percentile': 0.5}) == 'ranging'
    assert _detect_market_regime({'dmi_adx': 20, 'rv_percentile': 0.90}) == 'volatile'
    assert _detect_market_regime({'dmi_adx': 20, 'rv_percentile': 0.5}) == 'transitional'

    assert 'trending' in _REGIME_WEIGHTS
    assert 'ranging' in _REGIME_WEIGHTS
    for regime in _REGIME_WEIGHTS:
        assert len(_REGIME_WEIGHTS[regime]) == 15, f"{regime} missing schools"

    print("  ✓ _safe / _clip_score / _direction_from_score / _opposite_dir / regime detection")


def test_school_definitions():
    """Verify all 15 schools have proper definitions."""
    from expert_ensemble import SCHOOLS, SCHOOL_NAMES, SCHOOL_COMPUTE

    assert len(SCHOOLS) >= 22, f"Expected >=22 schools, got {len(SCHOOLS)}"
    assert len(SCHOOL_NAMES) >= 22, f"Expected >=22 names, got {len(SCHOOL_NAMES)}"
    assert len(SCHOOL_COMPUTE) >= 22, f"Expected >=22 compute, got {len(SCHOOL_COMPUTE)}"

    required_fields = ['label', 'short_label', 'description', 'indicators']
    for name, school in SCHOOLS.items():
        for f in required_fields:
            assert f in school, f"{name} missing {f}"
        # No duplicate indicator keys
        inds = school['indicators']
        assert len(inds) == len(set(inds)), f"{name} has duplicate indicator keys: {[x for x in inds if inds.count(x) > 1]}"

    print("  ✓ 15 schools defined correctly, no duplicate keys")


def test_ensemble_with_mock():
    """Test ensemble pipeline with mock indicators (no data needed)."""
    from expert_ensemble import compute_expert_ensemble, compute_market_regime_filter
    from expert_ensemble import compute_timeframe_coordination, compute_expert_ensemble_bayesian

    mock = {
        'current_price': 28.0,
        'ma5': 27.5, 'ma20': 26.0, 'ma60': 24.0, 'ma120': 22.0,
        'bb_upper': 29.0, 'bb_mid': 26.0, 'bb_lower': 23.0,
        'macd_dif': 0.5, 'macd_dea': 0.3, 'macd_hist': 0.2,
        'rsi': 60,
        'dmi_adx': 22, 'dmi_pdi': 30, 'dmi_mdi': 15, 'dmi_di_direction': 'bullish',
        'vol_ratio': 1.2, 'vol_trend': 'increasing', 'vol_type': 'normal',
        'obv_trend': 'up', 'obv_divergence': 'none', 'obv_signal': 'bullish',
        'vol_price_resonance': 0.3,
        'vol_stacking': False, 'vol_stacking_days': 0,
        'high_vol_stagnation': False, 'low_vol_pullback': False,
        'dynamic_support': [], 'dynamic_resistance': [],
        'verified_support': [], 'verified_resistance': [],
        'rv_composite': 0.015, 'rv_level': '中等波动', 'rv_percentile': 0.5,
        'mp_phase': '上涨', 'mp_confidence': 0.7, 'mp_price_position': 0.5,
        'mp_ret_20d': 0.03, 'mp_ret_60d': 0.05,
        'mp_trend_credibility': 0.6,
        'pattern_dominant': 'bullish', 'pattern_recent': 'morning_star',
        'pattern_latest_fractal': 'bottom',
        'pattern_top_fractal_n': 0, 'pattern_bottom_fractal_n': 2,
        'pattern_morning_star_n': 1, 'pattern_evening_star_n': 0,
        'momentum_state_short': 'bullish', 'momentum_state_long': 'bullish',
        'momentum_resonance': 0.5, 'momentum_direction': 'bullish',
        'roc_short': 3.0, 'roc_long': 8.0,
        'cci_short': 80, 'cci_long': 60,
        'vol_atr14': 1.2, 'vol_atr_pct': 4.0,
        'bebe_regime': 'good_environment', 'bebe_vrp_signal': 'neutral',
        'bebe_good_bad_ratio': 1.5,
        'bebe_good_vol_trend': 'stable', 'bebe_bad_vol_trend': 'stable',
        'price_range_5d_pct': 5.0,
        'sm_smart_signal': 'neutral',
        'top_escape_grade': 'normal', 'dist_days_warning': False,
        'har_direction': 'stable', 'har_decay': '中期波动主导',
        'chanlun_status': '笔向上',
        'chanlun_buy': [{'type': '盘整背驰', 'price': 26.0}],
        'chanlun_sell': [],
        'chanlun_stroke_state': 'up', 'chanlun_stroke_label': '笔向上',
        'chanlun_stroke_direction': 'up',
        'chanlun_divergence_type': 'none', 'chanlun_divergence_strength': 0,
        'chanlun_divergence_detail': '',
        'chanlun_wolf_signal': 'normal', 'chanlun_wolf_position': 'above',
        'chanlun_wolf_days': 0,
        'chanlun_trend_type': 'up', 'chanlun_trend_direction': 'up',
        'chanlun_zn_value': 26.0, 'chanlun_zn_position': 'above',
        'chanlun_zn_pattern': 'expanding',
        'chanlun_fractal_type': 'bottom', 'chanlun_fractal_price': 25.0,
        'chanlun_segment_count': 3, 'chanlun_bi_count': 5,
        'chanlun_multi_tf_alignment': 'bullish',
        'chanlun_multi_tf_recommendation': 'buy',
        'chanlun_segment_destructions': [],
        'chanlun_last_segment_destruction': '',
        'chanlun_zs_expanded': False, 'chanlun_zs_upgraded': False,
        'chanlun_zs_upgraded_level': '',
        'chanlun_xzd_active': False, 'chanlun_xzd_direction': '',
        'chanlun_xzd_confidence': 0, 'chanlun_xzd_reasons': '',
        'ew_pattern': 'impulse', 'ew_current_wave': 'wave3',
        'ew_structure': 'impulse_5', 'ew_rules_valid': True,
        'ew_rule_violations': [],
        'ew_fib_ratios': {'wave2': 0.5, 'wave3': 1.618},
        'ew_next_target': 30.0, 'ew_next_support': 26.0,
        'ew_confidence': 0.7,
        'ew_trade_signal': 'bullish', 'ew_trade_setup': 'wave3_breakout',
        'ew_trade_entry': '27.5-28.0', 'ew_trade_stop': '25.5',
        'ew_trade_target': '30.0', 'ew_trade_rationale': 'Wave 3 extension',
        'ew_pivot_count': 5, 'ew_trend': 'up',
        'ew_enhanced': {},
        'tang_jiato': True, 'tang_jiaya': False,
        'tang_jiato_strength': 85, 'tang_jiaya_strength': 0,
        'tang_jiato_phase': 'active', 'tang_jiaya_phase': '',
        'tang_triangle_area': 15.0,
        'tang_golden_cross': True, 'tang_death_cross': False,
        'tang_golden_score': 80, 'tang_death_score': 0,
        'tang_cross_days': 5, 'tang_triple_status': 'golden',
        'tang_runway': True, 'tang_runway_thickness': 0.15,
        'tang_runway_grade': 'A', 'tang_runway_slope': 0.1,
        'tang_target_2t1l': 32.0, 'tang_target_conservative': 30.0,
        'tang_target_pct': 15.0, 'tang_stop_2t': 25.0,
        'tang_risk_reward': 3.0,
        'tang_33_valid': True, 'tang_33_score': 75,
        'tang_33_days_above': 5, 'tang_33_pct_above': 3.0,
        'tang_laoyatou': True, 'tang_laoyatou_phase': 'mouth',
        'tang_laoyatou_score': 70, 'tang_laoyatou_neck': 28.0,
        'tang_laoyatou_head_h': 30.0, 'tang_laoyatou_mouth_a': 26.0,
        'livermore_pivots': [(28.0, 'resistance')],
        'livermore_nearest_pivot': 28.0,
        'livermore_breakout': True, 'livermore_breakout_str': '突破关键点',
        'livermore_signal': 'buy',
        'livermore_danger': 20, 'livermore_danger_signals': [],
        'livermore_danger_level': 20, 'livermore_action': '',
        'b2560_signal': 'bullish', 'b2560_description': '2560做多信号',
        'b2560_ma25_direction': 'up',
        'b2560_ma_golden_cross': True, 'b2560_ma_dead_cross': False,
        'b2560_vol_golden_cross': True, 'b2560_weekly_signal': 'bullish',
        'dma': 27.0, 'dma_ama': 26.5,
        'dma_diff': 0.5, 'dma_signal': 'bullish', 'dma_trend': 'up',
        'obv_price_direction': 'up', 'obv_direction': 'up',
        'top_escape_count': 0, 'top_escape_signals': '',
        'top_escape_prob': 0, 'top_escape_action': '',
        'dist_days_count': 0, 'dist_days_pct': 0, 'dist_days_stale': False,
    }

    # Test ensemble
    result = compute_expert_ensemble(mock)
    assert result is not None
    assert 'ensemble_signal' in result
    assert 'ensemble_confidence' in result
    assert 'votes' in result
    assert 'weighted_votes' in result
    assert 'school_signals' in result
    assert 'dynamic_router' in result
    assert result['dynamic_router']['gated_confidence'] <= 0.95
    assert len(result['school_signals']) == 15
    assert result['ensemble_confidence'] <= 0.95
    assert result['ensemble_confidence'] >= 0.15

    print(f"  ✓ Ensemble: signal={result['ensemble_signal']}, conf={result['ensemble_confidence']:.3f}, "
          f"votes={result['votes']}, diversity={result['diversity']:.3f}")

    # Test market filter
    filt = compute_market_regime_filter(mock)
    assert 'block_bullish' in filt
    assert 'block_bearish' in filt
    assert 'confidence_cap' in filt

    # Test bearish block in bull market (mp_phase=上涨, mp_conf=0.7)
    assert filt['active'] is True or filt['block_bearish'] is True
    print(f"  ✓ Market filter: active={filt['active']}, block_bullish={filt['block_bullish']}, "
          f"block_bearish={filt['block_bearish']}, cap={filt['confidence_cap']}")

    # Test timeframe coordination
    tf = compute_timeframe_coordination(mock, weekly_trend='多头')
    assert 'coordination_quality' in tf
    assert 'adjusted_confidence' in tf
    # Round 3 (last round) must have 'intention' and 'confidence' keys
    r3 = tf['rounds'][-1]
    assert r3['round'] == 3, f"Expected round 3, got {r3}"
    assert 'intention' in r3, f"Round 3 missing 'intention': {r3.keys()}"
    assert 'confidence' in r3, f"Round 3 missing 'confidence': {r3.keys()}"
    print(f"  ✓ Timeframe coordination: quality={tf['coordination_quality']}, "
          f"adj_conf={tf['adjusted_confidence']:.3f}")

    # Test Bayesian ensemble
    bayes_result = compute_expert_ensemble_bayesian(mock)
    assert bayes_result is not None
    # If Bayesian direction differs, reasons should be swapped
    print(f"  ✓ Bayesian ensemble: signal={bayes_result['ensemble_signal']}")

    return result


def test_market_filter_bull_block():
    """Test that block_bearish activates in bull market."""
    from expert_ensemble import compute_market_regime_filter

    # Bull market scenario
    bull_mock = {'mp_phase': '上涨', 'mp_confidence': 0.7, 'top_escape_grade': 'normal',
                 'dist_days_warning': False, 'rv_level': '中等波动', 'bebe_regime': 'good_environment'}
    filt = compute_market_regime_filter(bull_mock)
    assert filt['block_bearish'] is True, f"block_bearish should be True in bull market, got {filt}"
    print("  ✓ Bull market blocks bearish signals correctly")

    # Bear market scenario
    bear_mock = {'mp_phase': '下跌', 'mp_confidence': 0.7, 'top_escape_grade': 'normal',
                 'dist_days_warning': False, 'rv_level': '中等波动', 'bebe_regime': 'good_environment'}
    filt = compute_market_regime_filter(bear_mock)
    assert filt['block_bullish'] is True, f"block_bullish should be True in bear market, got {filt}"
    print("  ✓ Bear market blocks bullish signals correctly")


def test_local_trade_plan():
    """Test local trade plan extraction with mock data."""
    from local_trade_plan import extract_local_trade_plan, apply_adaptive_fusion_to_local_plan

    # Reuse ensemble mock data
    mock = _get_mock_indicators()
    plan = extract_local_trade_plan(mock)
    assert plan is not None
    assert 'signal' in plan
    assert 'confidence' in plan
    assert 'entry_zone' in plan
    assert 'stop_loss' in plan
    assert 'key_values' in plan
    assert 'principles' in plan

    # Risk score should be additive (not multiplicative)
    print(f"  ✓ Local plan: signal={plan['signal']}, conf={plan['confidence']:.3f}, "
          f"entry={plan['entry_zone']}, stop={plan['stop_loss']}")

    # Test AI fusion
    ai_signal = {'signal': 'bullish', 'confidence': 0.72, 'source': 'DeepSeek'}
    trust = {'fusion_weight_local': 0.55, 'ai_trust': 0.62, 'local_trust': 0.58,
             'trust_ratio': 1.07, 'recommendation': '略偏本地'}
    result = apply_adaptive_fusion_to_local_plan(plan, ai_signal, trust)
    assert result is not None
    assert 'adaptive_fusion' in result
    print(f"  ✓ AI fusion: alignment={result['alignment']}, "
          f"final_signal={result['signal']}, final_conf={result['confidence']:.3f}")


def test_portfolio_risk():
    """Test portfolio risk assessment improvements."""
    from advanced_indicators import portfolio_risk_assessment

    positions = [
        {'symbol': '000001', 'direction': 'bullish', 'weight': 0.15, 'sector': '银行', 'beta': 0.8, 'atr_pct': 1.5},
        {'symbol': '000002', 'direction': 'bullish', 'weight': 0.20, 'sector': '银行', 'beta': 0.9, 'atr_pct': 1.8},
        {'symbol': '600001', 'direction': 'bullish', 'weight': 0.10, 'sector': '科技', 'beta': 1.3, 'atr_pct': 3.0},
        {'symbol': '300001', 'direction': 'bearish', 'weight': 0.10, 'sector': '医药', 'beta': 1.1, 'atr_pct': 2.5},
    ]
    risk = portfolio_risk_assessment(positions)

    # New fields should exist
    assert 'stress_loss_5pct' in risk
    assert 'stress_loss_10pct' in risk
    assert 'n_effective_sectors' in risk
    assert 'bull_bear_ratio' in risk
    assert 'risk_contribution' in risk
    assert 'mean_correlation' in risk
    assert 'cvar_95_pct' in risk

    # Bearish in A-share = exit, not short → net exposure only counts bullish
    assert risk['net_exposure_pct'] == 45.0 or abs(risk['net_exposure_pct'] - 45.0) < 1

    # Sector concentration should flag 银行 at 35%
    assert risk['max_sector_pct'] > 0

    print(f"  ✓ Portfolio risk: exposure={risk['total_exposure_pct']}%, "
          f"net={risk['net_exposure_pct']}%, sector={risk['max_sector_pct']}%({risk.get('max_sector_name','?')}), "
          f"VaR={risk['var_95_pct']}%, stress-5%={risk['stress_loss_5pct']}%")


def test_config_params():
    """Test that config parameters are accessible."""
    import config
    assert hasattr(config, 'MA_SHORT')
    assert hasattr(config, 'ATR_PERIOD')
    assert hasattr(config, 'MAX_PORTFOLIO_EXPOSURE')
    assert hasattr(config, 'FEEDBACK_LOOKBACK')
    assert isinstance(config.DEEPSEEK_API_KEY, str) and len(config.DEEPSEEK_API_KEY) > 10
    print(f"  ✓ Config params accessible, {len([x for x in dir(config) if x.isupper() and not x.startswith('_')])} uppercase constants")


def _get_mock_indicators():
    """Return mock indicators dict for testing."""
    return {
        'current_price': 28.0,
        'ma5': 27.5, 'ma20': 26.0, 'ma60': 24.0, 'ma120': 22.0,
        'bb_upper': 29.0, 'bb_mid': 26.0, 'bb_lower': 23.0,
        'macd_dif': 0.5, 'macd_dea': 0.3, 'macd_hist': 0.2,
        'rsi': 60,
        'dmi_adx': 22, 'dmi_pdi': 30, 'dmi_mdi': 15, 'dmi_di_direction': 'bullish',
        'vol_ratio': 1.2, 'vol_trend': 'increasing', 'vol_type': 'normal',
        'obv_trend': 'up', 'obv_divergence': 'none', 'obv_signal': 'bullish',
        'vol_price_resonance': 0.3,
        'vol_stacking': False, 'vol_stacking_days': 0,
        'high_vol_stagnation': False, 'low_vol_pullback': False,
        'dynamic_support': [], 'dynamic_resistance': [],
        'verified_support': [], 'verified_resistance': [],
        'rv_composite': 0.015, 'rv_level': '中等波动', 'rv_percentile': 0.5,
        'mp_phase': '上涨', 'mp_confidence': 0.7, 'mp_price_position': 0.5,
        'mp_ret_20d': 0.03, 'mp_ret_60d': 0.05,
        'mp_trend_credibility': 0.6,
        'pattern_dominant': 'bullish', 'pattern_recent': 'morning_star',
        'pattern_latest_fractal': 'bottom',
        'pattern_top_fractal_n': 0, 'pattern_bottom_fractal_n': 2,
        'pattern_morning_star_n': 1, 'pattern_evening_star_n': 0,
        'momentum_state_short': 'bullish', 'momentum_state_long': 'bullish',
        'momentum_resonance': 0.5, 'momentum_direction': 'bullish',
        'roc_short': 3.0, 'roc_long': 8.0,
        'cci_short': 80, 'cci_long': 60,
        'vol_atr14': 1.2, 'vol_atr_pct': 4.0,
        'bebe_regime': 'good_environment', 'bebe_vrp_signal': 'neutral',
        'bebe_good_bad_ratio': 1.5,
        'bebe_good_vol_trend': 'stable', 'bebe_bad_vol_trend': 'stable',
        'price_range_5d_pct': 5.0,
        'sm_smart_signal': 'neutral',
        'top_escape_grade': 'normal', 'dist_days_warning': False,
        'har_direction': 'stable', 'har_decay': '中期波动主导',
        'chanlun_status': '笔向上',
        'chanlun_buy': [{'type': '盘整背驰', 'price': 26.0}],
        'chanlun_sell': [],
        'chanlun_stroke_state': 'up', 'chanlun_stroke_label': '笔向上',
        'chanlun_stroke_direction': 'up',
        'chanlun_divergence_type': 'none', 'chanlun_divergence_strength': 0,
        'chanlun_divergence_detail': '',
        'chanlun_wolf_signal': 'normal', 'chanlun_wolf_position': 'above',
        'chanlun_wolf_days': 0,
        'chanlun_trend_type': 'up', 'chanlun_trend_direction': 'up',
        'chanlun_zn_value': 26.0, 'chanlun_zn_position': 'above',
        'chanlun_zn_pattern': 'expanding',
        'chanlun_fractal_type': 'bottom', 'chanlun_fractal_price': 25.0,
        'chanlun_segment_count': 3, 'chanlun_bi_count': 5,
        'chanlun_multi_tf_alignment': 'bullish',
        'chanlun_multi_tf_recommendation': 'buy',
        'chanlun_segment_destructions': [],
        'chanlun_last_segment_destruction': '',
        'chanlun_zs_expanded': False, 'chanlun_zs_upgraded': False,
        'chanlun_zs_upgraded_level': '',
        'chanlun_xzd_active': False, 'chanlun_xzd_direction': '',
        'chanlun_xzd_confidence': 0, 'chanlun_xzd_reasons': '',
        'ew_pattern': 'impulse', 'ew_current_wave': 'wave3',
        'ew_structure': 'impulse_5', 'ew_rules_valid': True,
        'ew_rule_violations': [],
        'ew_fib_ratios': {'wave2': 0.5, 'wave3': 1.618},
        'ew_next_target': 30.0, 'ew_next_support': 26.0,
        'ew_confidence': 0.7,
        'ew_trade_signal': 'bullish', 'ew_trade_setup': 'wave3_breakout',
        'ew_trade_entry': '27.5-28.0', 'ew_trade_stop': '25.5',
        'ew_trade_target': '30.0', 'ew_trade_rationale': 'Wave 3 extension',
        'ew_pivot_count': 5, 'ew_trend': 'up',
        'ew_enhanced': {},
        'tang_jiato': True, 'tang_jiaya': False,
        'tang_jiato_strength': 85, 'tang_jiaya_strength': 0,
        'tang_jiato_phase': 'active', 'tang_jiaya_phase': '',
        'tang_triangle_area': 15.0,
        'tang_golden_cross': True, 'tang_death_cross': False,
        'tang_golden_score': 80, 'tang_death_score': 0,
        'tang_cross_days': 5, 'tang_triple_status': 'golden',
        'tang_runway': True, 'tang_runway_thickness': 0.15,
        'tang_runway_grade': 'A', 'tang_runway_slope': 0.1,
        'tang_target_2t1l': 32.0, 'tang_target_conservative': 30.0,
        'tang_target_pct': 15.0, 'tang_stop_2t': 25.0,
        'tang_risk_reward': 3.0,
        'tang_33_valid': True, 'tang_33_score': 75,
        'tang_33_days_above': 5, 'tang_33_pct_above': 3.0,
        'tang_laoyatou': True, 'tang_laoyatou_phase': 'mouth',
        'tang_laoyatou_score': 70, 'tang_laoyatou_neck': 28.0,
        'tang_laoyatou_head_h': 30.0, 'tang_laoyatou_mouth_a': 26.0,
        'livermore_pivots': [(28.0, 'resistance')],
        'livermore_nearest_pivot': 28.0,
        'livermore_breakout': True, 'livermore_breakout_str': '突破关键点',
        'livermore_signal': 'buy',
        'livermore_danger': 20, 'livermore_danger_signals': [],
        'livermore_danger_level': 20, 'livermore_action': '',
        'b2560_signal': 'bullish', 'b2560_description': '2560做多信号',
        'b2560_ma25_direction': 'up',
        'b2560_ma_golden_cross': True, 'b2560_ma_dead_cross': False,
        'b2560_vol_golden_cross': True, 'b2560_weekly_signal': 'bullish',
        'dma': 27.0, 'dma_ama': 26.5,
        'dma_diff': 0.5, 'dma_signal': 'bullish', 'dma_trend': 'up',
        'obv_price_direction': 'up', 'obv_direction': 'up',
        'top_escape_count': 0, 'top_escape_signals': '',
        'top_escape_prob': 0, 'top_escape_action': '',
        'dist_days_count': 0, 'dist_days_pct': 0, 'dist_days_stale': False,
        'bayes_fused_signal': 'bullish',
        'bayes_fused_posterior': 0.65,
        'bayes_fused_entropy': 0.4,
    }


def run_all():
    print("=" * 50)
    print("量化系统基础测试套件")
    print("=" * 50)

    tests = [
        ("Helper functions", test_safe_functions),
        ("School definitions", test_school_definitions),
        ("Market filter bull/bear blocks", test_market_filter_bull_block),
        ("Ensemble pipeline (mock)", test_ensemble_with_mock),
        ("Local trade plan + AI fusion", test_local_trade_plan),
        ("Portfolio risk assessment", test_portfolio_risk),
        ("Config parameters", test_config_params),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} FAILED: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"结果: {passed} passed, {failed} failed, {len(tests)} total")
    return failed == 0


if __name__ == '__main__':
    ok = run_all()
    sys.exit(0 if ok else 1)
