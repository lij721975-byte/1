# expert_ensemble.py — School-based expert ensemble with Nüwa learning router
# Architecture: 15 trading-theory schools, each internally coherent,
# combined via performance-weighted ensemble with Nüwa adaptive routing.
#
# Schools (学派):
#   1. 缠论学派 — Chanlun + 解缠论 unified structure analysis
#   2. 艾略特波浪 — Elliott Wave theory
#   3. 唐能通体系 — Tang Nengtong short-term system
#   4. 利弗莫尔战法 — Livermore pivotal points + danger signals
#   5. Busch量化体系 — Busch quantitative frameworks (2560, DMI, VPA, Granville)
#   6. 经典技术分析 — Classical TA (MA, MACD, RSI, Bollinger, etc.)
#   7. 风险与市场环境 — Risk/volatility regime detection
#   8. 江恩理论 — Gann Theory (time cycles, % retracement, resonance, 12 rules)
#   9. 威科夫量价 — Wyckoff VSA (supply/demand, accumulation/distribution, effort vs result)
#  10. 道氏理论 — Dow Theory (primary/secondary/minor trend, mutual confirmation)
#  11. 欧奈尔CANSLIM — O'Neil growth stock system (EPS, RS rating, cup-handle)
#  12. 海龟交易法 — Turtle Trading (Donchian channels, ATR position sizing, pyramid)
#  13. 谐波形态 — Harmonic Patterns (Gartley, Bat, Butterfly, Crab, Cypher, PRZ)
#  14. 市场轮廓 — Market Profile/TPO (value area, POC, auction imbalance)
#  15. 混沌交易法 — Chaos Theory (fractal, alligator, AO, AC, MFI, 5D resonance)
#
# Nüwa Router: tracks per-school historical accuracy, adapts voting weights.
# Each school is independently calibratable — its track record determines
# its influence in the ensemble.

import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from advanced_indicators import (compute_hdi, rope_decision,
                                      james_stein_shrinkage, empirical_bayes_shrinkage)
    _BAYES_REFINEMENT_AVAILABLE = True
except ImportError:
    _BAYES_REFINEMENT_AVAILABLE = False
    compute_hdi = rope_decision = james_stein_shrinkage = empirical_bayes_shrinkage = None


# ============================================================
# School definitions — each school is a coherent trading theory
# ============================================================

SCHOOLS = {
    'school_chanlun': {
        'label': '缠论学派',
        'short_label': '缠论',
        'description': '缠中说禅+解缠论：笔段中枢、走势类型、背驰、买卖点、多级别联立',
        'indicators': [
            'chanlun_status', 'chanlun_buy', 'chanlun_sell',
            'chanlun_stroke_state', 'chanlun_stroke_direction',
            'chanlun_divergence_type', 'chanlun_divergence_strength',
            'chanlun_wolf_signal', 'chanlun_wolf_days',
            'chanlun_trend_type', 'chanlun_trend_direction',
            'chanlun_zn_position', 'chanlun_zn_pattern',
            'chanlun_multi_tf_alignment',
            'chanlun_last_segment_destruction', 'chanlun_zs_expanded',
            'chanlun_xzd_active', 'chanlun_xzd_direction', 'chanlun_xzd_confidence',
            'cl_power_ratio', 'cl_divergence_type', 'cl_divergence_warning',
            'cl_cg_direction', 'cl_cg_trend_str',
            'cl_turn_score', 'cl_turn_direction', 'cl_turn_confidence', 'cl_turn_signal',
            'cl_intra_divergence', 'cl_intra_div_type', 'cl_intra_div_severity',
        ],
    },
    'school_tang': {
        'label': '唐能通体系',
        'short_label': '唐能通',
        'description': '短线是银：价托价压、三金叉三死叉、跑道厚度、老鸭头、三三过滤',
        'indicators': [
            'tang_jiato', 'tang_jiaya', 'tang_jiato_strength', 'tang_jiaya_strength',
            'tang_jiato_phase', 'tang_jiaya_phase',
            'tang_golden_cross', 'tang_death_cross', 'tang_triple_status',
            'tang_runway', 'tang_runway_thickness', 'tang_runway_grade',
            'tang_laoyatou', 'tang_laoyatou_score', 'tang_laoyatou_phase',
            'tang_golden_score', 'tang_death_score',
            'tang_33_valid', 'tang_33_score',
        ],
    },
    'school_livermore': {
        'label': '利弗莫尔战法',
        'short_label': '利弗莫尔',
        'description': '股票大作手：关键点突破、危险信号、金字塔加仓',
        'indicators': [
            'livermore_signal', 'livermore_breakout', 'livermore_breakout_str',
            'livermore_danger', 'livermore_danger_level', 'livermore_action',
        ],
    },
    'school_busch': {
        'label': 'Busch量化体系',
        'short_label': 'Busch',
        'description': 'Busch量化框架：2560战法、DMI增强、量价代数、量能形态、九大关系、格兰维尔',
        'indicators': [
            'b2560_signal', 'b2560_description', 'b2560_ma25_direction',
            'b2560_ma_golden_cross', 'b2560_vol_golden_cross', 'b2560_weekly_signal',
            'bdmi_signal', 'bdmi_description', 'bdmi_mdi_peaking', 'bdmi_pdi_rising', 'bdmi_big_momentum',
            'vpa_formula', 'vpa_signal', 'vpa_interpretation', 'vpa_action', 'vpa_is_extreme_vol',
            'vc_pattern', 'vc_description', 'vc_signal',
            'v9r_strength', 'v9r_id',
            'g8_active_rule', 'g8_rule_category', 'g8_land_volume_zone',
            'cva_composite_signal', 'cva_bull_count', 'cva_bear_count',
        ],
    },
    'school_classical': {
        'label': '经典技术分析',
        'short_label': '经典TA',
        'description': '传统技术指标：均线、MACD、RSI、布林带、DMA、DMI、TD、K线形态、OBV、VR',
        'indicators': [
            'ma5', 'ma20', 'ma60',
            'macd_dif', 'macd_dea', 'macd_hist',
            'rsi', 'roc_short', 'roc_long', 'cci_short', 'cci_long',
            'bb_upper', 'bb_mid', 'bb_lower',
            'dma', 'dma_ama', 'dma_diff', 'dma_signal', 'dma_trend',
            'dmi_adx', 'dmi_pdi', 'dmi_mdi', 'dmi_di_direction', 'dmi_adx_trend',
            'td_count', 'td_direction', 'td_completed',
            'pattern_dominant', 'pattern_latest_fractal', 'pattern_latest_reversal', 'pattern_recent',
            'pattern_top_fractal_n', 'pattern_bottom_fractal_n',
            'pattern_evening_star_n', 'pattern_morning_star_n',
            'dynamic_support', 'dynamic_resistance', 'verified_support', 'verified_resistance',
            'current_price',
            'obv_trend', 'obv_divergence', 'obv_signal',
            'vr_value', 'vr_zone', 'vr_signal', 'vr_divergence',
            'momentum_state_short', 'momentum_state_long', 'momentum_resonance', 'momentum_direction',
            'vol_ratio', 'vol_price_resonance', 'vol_type', 'vol_trend',
            'vol_stacking', 'high_vol_stagnation', 'low_vol_pullback',
            'volume_breakout', 'granville_signal',
            'vol_verified_support', 'vol_verified_resistance', 'price_range_5d_pct',
            'sm_smart_signal', 'sm_ad_divergence', 'vol_stacking_days',
        ],
    },
    'school_risk': {
        'label': '风险与市场环境',
        'short_label': '风险环境',
        'description': '波动率建模+市场阶段+风险预警：RV、HAR、BEGE、逃顶、分配日、NRB',
        'indicators': [
            'rv_parkinson', 'rv_garman_klass', 'rv_composite',
            'rv_trend', 'rv_percentile', 'rv_level',
            'har_rv_daily', 'har_rv_weekly', 'har_rv_monthly',
            'har_forecast', 'har_direction', 'har_r_squared', 'har_decay',
            'har_beta_d', 'har_beta_w', 'har_beta_m',
            'vol_signal', 'vol_advice', 'vol_position_mult', 'vol_position_advice',
            'vol_adaptive_stop_pct', 'vol_adaptive_stop_price',
            'vol_entry_quality', 'vol_atr14', 'vol_atr_pct',
            'nrb_signal', 'nrb_type',
            'mp_phase', 'mp_confidence', 'mp_big_trend', 'mp_mid_trend', 'mp_small_trend',
            'mp_trend_credibility', 'mp_price_position', 'mp_ret_20d', 'mp_ret_60d',
            'top_escape_prob', 'top_escape_grade', 'top_escape_count',
            'top_escape_action', 'top_escape_signals',
            'dist_days_count', 'dist_days_warning', 'dist_days_pct',
            'kmeans_cluster',
            'bayes_fused_posterior', 'bayes_fused_signal', 'bayes_fused_entropy',
            'bayes_dimensions_active', 'bayes_entropy_high',
            'bebe_regime', 'bebe_vrp_signal', 'bebe_good_bad_ratio',
            'bebe_vol_asymmetry', 'bebe_good_vol_trend', 'bebe_bad_vol_trend',
        ],
    },
    'school_gann': {
        'label': '江恩理论',
        'short_label': '江恩',
        'description': '江恩理论：时间周期+百分比回调+三天图九点图+波动共振+12条买卖法则+21条守则',
        'indicators': [
            # Trend & price indicators (shared — Gann's 3-day chart, % retracement, new highs/lows)
            'current_price', 'ma5', 'ma20', 'ma60',
            'macd_dif', 'macd_dea', 'macd_hist',
            'dmi_adx', 'dmi_pdi', 'dmi_mdi', 'dmi_di_direction',
            # Volume (Gann's rule 7)
            'vol_ratio', 'vol_trend', 'vol_type',
            'obv_trend', 'obv_divergence',
            # Support/resistance (Gann's rule 2: single/double/triple bottoms)
            'dynamic_support', 'dynamic_resistance',
            'verified_support', 'verified_resistance',
            'bb_upper', 'bb_mid', 'bb_lower',
            # Momentum (Gann's rule 6: 5-7 point swings)
            'rsi', 'roc_short', 'roc_long', 'cci_short', 'cci_long',
            'momentum_state_short', 'momentum_state_long', 'momentum_resonance',
            # Volatility (Gann's 1x1, 1x2 lines)
            'vol_atr14', 'vol_atr_pct',
            'rv_composite', 'rv_level', 'rv_percentile',
            # Market phase (Gann's segmentation rule 5)
            'mp_phase', 'mp_confidence',
            # Pattern (Gann's single/double/triple bottom)
            'pattern_dominant', 'pattern_latest_fractal',
            'pattern_top_fractal_n', 'pattern_bottom_fractal_n',
            # Price range for % calculations
            'price_range_5d_pct',
            # Volume stacking (Gann accumulation/distribution)
            'vol_stacking', 'vol_price_resonance',
            # Other
            'bebe_regime', 'bebe_vrp_signal',
        ],
    },
    'school_wyckoff': {
        'label': '威科夫量价',
        'short_label': '威科夫',
        'description': '威科夫VSA：供需定律+因果定律+努力vs结果+吸筹/派发周期+Spring/UTAD+SOS/SOW',
        'indicators': [
            # Price structure (TR identification, spring/UTAD)
            'current_price', 'ma5', 'ma20', 'ma60',
            'bb_upper', 'bb_mid', 'bb_lower',
            # Volume (the key to Wyckoff — effort vs result)
            'vol_ratio', 'vol_trend', 'vol_type',
            'vol_price_resonance', 'vol_stacking',
            'high_vol_stagnation', 'low_vol_pullback',
            'vol_stacking_days',
            # OBV (accumulation/distribution detection)
            'obv_trend', 'obv_divergence', 'obv_signal',
            # Support/resistance (TR boundaries)
            'dynamic_support', 'dynamic_resistance',
            'verified_support', 'verified_resistance',
            # Pattern (spring=bottom fractal, UTAD=top fractal)
            'pattern_dominant', 'pattern_latest_fractal',
            'pattern_top_fractal_n', 'pattern_bottom_fractal_n',
            'pattern_evening_star_n', 'pattern_morning_star_n',
            'pattern_recent',
            # Momentum (effort confirmation)
            'rsi', 'roc_short', 'roc_long', 'cci_short', 'cci_long',
            'momentum_state_short', 'momentum_state_long', 'momentum_resonance',
            # Market phase (accumulation/markup/distribution/markdown)
            'mp_phase', 'mp_confidence', 'mp_price_position',
            'mp_ret_20d', 'mp_ret_60d',
            # Volume-price resonance
            'dmi_adx', 'dmi_pdi', 'dmi_mdi', 'dmi_di_direction',
            # ATR (for TR measurement)
            'vol_atr14', 'vol_atr_pct',
            # Price range
            'price_range_5d_pct',
            # Smart money
            'sm_smart_signal', 'sm_ad_divergence',
            # Risk context
            'rv_level', 'rv_percentile', 'bebe_regime',
            # VR
            'vr_value', 'vr_zone', 'vr_signal',
        ],
    },
    'school_harmonic': {
        'label': '谐波形态',
        'short_label': '谐波',
        'description': '谐波形态：Gartley/Bat/Butterfly/Crab/Cypher五形态+PRZ斐波汇聚+反转K线确认',
        'indicators': [
            'current_price', 'ma5', 'ma20', 'ma60',
            'pattern_dominant', 'pattern_latest_fractal', 'pattern_recent',
            'pattern_top_fractal_n', 'pattern_bottom_fractal_n',
            'pattern_morning_star_n', 'pattern_evening_star_n',
            'rsi', 'cci_short', 'cci_long',
            'vol_atr14', 'vol_atr_pct',
            'vol_ratio', 'vol_trend',
            'bb_upper', 'bb_mid', 'bb_lower',
            'dynamic_support', 'dynamic_resistance',
            'verified_support', 'verified_resistance',
            'mp_phase', 'mp_confidence', 'mp_price_position',
            'macd_dif', 'macd_dea', 'macd_hist',
            'obv_divergence',
        ],
    },
    'school_roc_breakout': {
        'label': 'ROC动量突破策略',
        'short_label': 'ROC-Momentum',
        'description': 'ROC+RSI+ATR+RVOL动量突破：ROC方向+RSI超买超卖+相对放量+ATR扩张+趋势过滤 5条件打分制',
        'indicators': [
            'rsi', 'current_price', 'ma5', 'ma20',
            'vol_ratio', 'vol_trend',
            'momentum_state_short', 'momentum_state_long',
            'vol_atr14', 'vol_atr_pct',
            'roc_short', 'roc_long',
        ],
    },
    'school_volume_profile': {
        'label': '成交量分布(Volume Profile)策略',
        'short_label': 'VolProfile',
        'description': 'Fixed Range Volume Profile：POC控制点+VAH/VAL价值区+HVN/LVN高/低量节点+VA突破/回踩 5条件打分制',
        'indicators': [
            'rsi', 'current_price', 'ma5', 'ma20',
            'vol_ratio', 'vol_trend',
            'momentum_state_short', 'momentum_state_long',
            'vol_atr14', 'vol_atr_pct',
        ],
    },
    'school_fusion': {
        'label': '蒸馏融合策略(Fusion)',
        'short_label': 'Fusion',
        'description': 'ADX趋势判别+FRVP筹码峰+缠论分型+Delta订单流+ROC动量 6条件≥4共振',
        'indicators': [
            'rsi', 'current_price', 'ma5', 'ma20',
            'dmi_adx', 'dmi_pdi', 'dmi_mdi',
            'vol_ratio', 'vol_trend',
            'obv_trend', 'obv_divergence',
            'roc_short', 'macd_hist', 'bb_upper',
            'momentum_state_short', 'momentum_state_long',
            'pattern_bottom_fractal_n', 'mp_phase',
        ],
    },
    'school_mean_reversion': {
        'label': '均值回归(震荡市)',
        'short_label': '均值回归',
        'description': '布林带均值回归：下轨超卖买入+RSI<35+RVOL放量+中轨止盈+上轨做空。专攻ADX<22震荡市。',
        'indicators': [
            'current_price', 'ma20', 'ma60',
            'bb_upper', 'bb_mid', 'bb_lower', 'bb_width',
            'rsi', 'cci_short',
            'vol_ratio', 'vol_trend',
            'dmi_adx', 'dmi_pdi', 'dmi_mdi',
            'vol_atr14', 'vol_atr_pct',
            'momentum_state_short',
            'dynamic_support', 'dynamic_resistance',
        ],
    },
    'school_capital_flow': {
        'label': '资金流向学派',
        'short_label': '资金流向',
        'description': '北向资金+主力资金+大单占比+融资变化+资金共振 5维打分',
        'indicators': [
            'current_price', 'ma5', 'ma20',
            'vol_ratio', 'vol_trend', 'vol_stacking',
            'rsi',
            'dmi_adx', 'dmi_pdi', 'dmi_mdi',
            'vol_atr14', 'vol_atr_pct',
            'obv_trend', 'obv_divergence',
        ],
    },
    'school_pattern_features': {
        'label': '经典形态特征因子库',
        'short_label': '形态因子',
        'description': '42维连续形态特征:单K几何/双K关系/三K结构/图表形态/合成压力 → 树模型原生输入',
        'indicators': [
            'current_price', 'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma20', 'ma60', 'vol_atr14', 'vol_atr_pct', 'rsi',
        ],
    },
    'school_mean_reversion': {
        'label': '均值回归学派(MR)',
        'short_label': '均值回归',
        'description': 'BB+RSI+BIAS+KC 多因子均值回归：7条件超卖买入/4条件超买卖出，ADX趋势过滤+阳线防飞刀',
        'indicators': [
            'current_price', 'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma20', 'vol_atr14', 'rsi',
        ],
    },
    'school_ml': {
        'label': 'ML模型(XGBoost)',
        'short_label': 'ML',
        'description': 'XGBoost分类器: 15学派投票→15维特征→bullish/bearish/neutral预测',
        'indicators': [
            'current_price', 'open', 'high', 'low', 'close', 'volume',
        ],
    },
    'school_nd': {
        'label': '正态分布学派(ND)',
        'short_label': 'ND',
        'description': 'Z-Score: 超卖(-2σ)企稳→均值回归做多; 放量突破(+2σ)→动量做多',
        'indicators': ['current_price', 'open', 'high', 'low', 'close', 'volume'],
    },
    'school_vp': {
        'label': '成交量分布学派(VP)',
        'short_label': 'VP',
        'description': 'Volume Profile POC: 放量突破POC→做多; 回踩POC获支撑→做多; POC受阻→看空',
        'indicators': ['current_price', 'open', 'high', 'low', 'close', 'volume'],
    },
    'school_roc_factor': {
        'label': 'ROC动量因子(定制)',
        'short_label': 'ROC-Factor',
        'description': 'AX/BX ROC变体: (P_t-P_{t-20})/P_{t-60}×100, 动量正向因子',
        'indicators': ['current_price', 'open', 'high', 'low', 'close', 'volume'],
    },
    'school_beta': {
        'label': '历史Beta学派',
        'short_label': 'Beta',
        'description': 'Historical Beta: 高Beta牛市做多/熊市回避, 低Beta防御',
        'indicators': ['current_price', 'open', 'high', 'low', 'close', 'volume'],
    },
    'school_ewm_beta': {
        'label': 'EWM Beta学派',
        'short_label': 'EWM-Beta',
        'description': 'EWM Beta(hl=63): 指数衰减加权协方差/方差, 正向因子',
        'indicators': ['current_price', 'open', 'high', 'low', 'close', 'volume'],
    },
    'school_brooks_pa': {
        'label': 'Brooks价格行为学派',
        'short_label': 'Brooks PA',
        'description': 'Al Brooks PA三部曲：趋势K线/H1-H4计数/信号入场/多空陷阱/EMA引力',
        'indicators': [
            'current_price', 'open', 'high', 'low', 'close', 'volume',
            'vol_ratio', 'vol_atr14', 'vol_atr_pct',
            'dmi_adx', 'rv_level',
            'bebe_regime',
        ],
    },
}

# School order for consistent iteration
SCHOOL_NAMES = list(SCHOOLS.keys())


# ============================================================
# Helpers
# ============================================================

def _safe(v, default=0.0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return default
        # Extract last element
        last = v[-1]
        # Handle (price, strength) tuple format
        if isinstance(last, (list, tuple)):
            if len(last) > 0:
                return float(last[0])
            return default
        if last is None or (isinstance(last, float) and np.isnan(last)):
            return default
        return float(last)
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def _clip_score(score, lo=-1.0, hi=1.0):
    return float(np.clip(score, lo, hi))


def _direction_from_score(score, threshold=0.08):
    if score > threshold:
        return 'bullish'
    elif score < -threshold:
        return 'bearish'
    return 'neutral'


def _opposite_dir(direction):
    """Return the opposite direction."""
    if direction == 'bullish':
        return 'bearish'
    elif direction == 'bearish':
        return 'bullish'
    return 'neutral'


# ============================================================
# School compute functions — each resolves internal contradictions
# ============================================================

def _compute_school_chanlun(indicators: Dict) -> Dict:
    """缠论学派 V2.0 — 八段工业级防脆量化拓扑

    Topology:
      Step 1: Polarity-grouped accumulation (20 original indicators → bull/bear)
      Step 2: Advanced dynamics stream (C2/C4/C8 → 5-tuple split)
      Step 3: Advanced structural stream (C1/C3/C5/C7 → 6-tuple split)
      Step 4: Nonlinear squashing (1.0 - exp(-x * 1.5))
      Step 5: Resonance interception (bull > 0.40 & bear > 0.40 → 0)
      Step 6: Trend-completion asymmetric modifier (走势必完美 saturation gate)
      Step 7: Bardo-stage vacuum noise filter (中阴滤噪器)
      Step 8: Final assembly (raw * dynamics * bardo → clip → direction/confidence)
    """
    import numpy as np
    score_bull = 0.0; score_bear = 0.0
    reasons_bull = []; reasons_bear = []
    def _add_bull(w, label):
        nonlocal score_bull; score_bull += w; reasons_bull.append(label)
    def _add_bear(w, label):
        nonlocal score_bear; score_bear += w; reasons_bear.append(label)

    # ════════════════════════════════════════════════════════════════
    # STEP 1 — Polarity-grouped accumulation (20 original indicators)
    # ════════════════════════════════════════════════════════════════

    # 1a) 笔状态机
    stroke_st = str(indicators.get('chanlun_stroke_state', ''))
    if stroke_st == '(1,1)':         _add_bull(0.20, '笔(1,1)向上确认')
    elif stroke_st == '(1,0)':       _add_bull(0.08, '笔(1,0)向上延伸')
    elif stroke_st == '(-1,1)':      _add_bear(0.20, '笔(-1,1)向下确认')
    elif stroke_st == '(-1,0)':      _add_bear(0.08, '笔(-1,0)向下延伸')

    # 1b) 买卖点
    chan_buy = indicators.get('chanlun_buy', [])
    chan_sell = indicators.get('chanlun_sell', [])
    if chan_buy and len(chan_buy) > 0:
        bp_types = ','.join([b.get('type', '?') if isinstance(b, dict) else str(b) for b in chan_buy])
        _add_bull(0.18, f'买点:{bp_types}')
    if chan_sell and len(chan_sell) > 0:
        sp_types = ','.join([s.get('type', '?') if isinstance(s, dict) else str(s) for s in chan_sell])
        _add_bear(0.18, f'卖点:{sp_types}')

    # 1c) 背驰信号
    div_type = str(indicators.get('chanlun_divergence_type', ''))
    div_strength = _safe(indicators.get('chanlun_divergence_strength'), 0)
    if div_type == '底背驰' and div_strength > 0.4:
        _add_bull(0.22, f'底背驰(强度{div_strength:.0%})')
    elif div_type == '顶背驰' and div_strength > 0.4:
        _add_bear(0.22, f'顶背驰(强度{div_strength:.0%})')

    # 1d) 防狼术
    wolf = str(indicators.get('chanlun_wolf_signal', ''))
    wolf_days = _safe(indicators.get('chanlun_wolf_days'), 0)
    if wolf == 'danger' and wolf_days > 10:
        _add_bear(0.28, f'防狼术:MACD零轴下{wolf_days}天')

    # 1e) 走势类型
    chan_trend = str(indicators.get('chanlun_trend_type', ''))
    if '下跌趋势' in chan_trend:
        _add_bear(0.12, f'{chan_trend}')
    elif '上涨趋势' in chan_trend:
        _add_bull(0.14, f'{chan_trend}')

    # 1f) 多级别联立
    tf_align = str(indicators.get('chanlun_multi_tf_alignment', ''))
    if '共振看多' in tf_align:
        _add_bull(0.20, '多级别共振看多')
    elif '共振看空' in tf_align:
        _add_bear(0.20, '多级别共振看空')

    # 1g) 中枢位置
    zn_pos = str(indicators.get('chanlun_zn_position', ''))
    if '超卖' in zn_pos:
        _add_bull(0.10, 'Zn中枢下沿超卖')
    elif '超买' in zn_pos:
        _add_bear(0.10, 'Zn中枢上沿超买')

    # 1h) 线段破坏
    seg_d = str(indicators.get('chanlun_last_segment_destruction', ''))
    if '标准破坏' in seg_d:
        if '新高' in seg_d:
            _add_bull(0.15, '线段标准破坏→新高确认')
        elif '新低' in seg_d:
            _add_bear(0.15, '线段标准破坏→新低确认')
    elif '背驰完成' in seg_d:
        if '顶分型' in seg_d:
            _add_bear(0.12, '线段背驰完成→顶分型确认')
        elif '底分型' in seg_d:
            _add_bull(0.12, '线段背驰完成→底分型确认')

    # 1i) 小转大
    xzd = indicators.get('chanlun_xzd_active', False)
    xzd_dir = str(indicators.get('chanlun_xzd_direction', ''))
    xzd_conf = _safe(indicators.get('chanlun_xzd_confidence'), 0)
    if xzd and xzd_conf > 0.6:
        if xzd_dir == 'bullish':
            _add_bull(0.22, f'小转大看多(置信{xzd_conf:.0%})')
        elif xzd_dir == 'bearish':
            _add_bear(0.22, f'小转大看空(置信{xzd_conf:.0%})')

    # 1j) 中枢扩展 — conf_mod note (not scored, recorded for Step 7)
    zs_expanded = indicators.get('chanlun_zs_expanded', False)

    # 1k) 解缠论：力度比背驰
    cl_div_type2 = str(indicators.get('cl_divergence_type', ''))
    if 'bottom_divergence' in cl_div_type2:
        _add_bull(0.16, '力度比底背驰→下跌衰竭')
    elif 'top_divergence' in cl_div_type2:
        _add_bear(0.16, '力度比顶背驰→上涨衰竭')

    # 1l) 解缠论：中枢重心偏移
    cl_cg_dir = str(indicators.get('cl_cg_direction', ''))
    cl_cg_str = _safe(indicators.get('cl_cg_trend_str'), 0)
    if cl_cg_dir == 'up' and cl_cg_str > 40:
        _add_bull(0.10, '中枢重心连续上移')
    elif cl_cg_dir == 'down' and cl_cg_str > 40:
        _add_bear(0.10, '中枢重心连续下移')

    # 1m) 解缠论：5条件拐点评分
    cl_ts = _safe(indicators.get('cl_turn_score'), 0)
    cl_ts_sig = str(indicators.get('cl_turn_signal', ''))
    if cl_ts_sig == 'strong_buy' and cl_ts >= 80:
        _add_bull(0.25, f'拐点评分{cl_ts:.0f}→高概率底部')
    elif cl_ts_sig == 'buy_watch' and cl_ts >= 60:
        _add_bull(0.12, f'拐点评分{cl_ts:.0f}→中概率拐点')
    elif cl_ts_sig == 'strong_sell' and cl_ts >= 80:
        _add_bear(0.25, f'拐点评分{cl_ts:.0f}→高概率顶部')
    elif cl_ts_sig == 'sell_watch' and cl_ts >= 60:
        _add_bear(0.12, f'拐点评分{cl_ts:.0f}→警惕反转')

    # 1n) 解缠论：笔内背离
    cl_intra = indicators.get('cl_intra_divergence', False)
    cl_intra_type = str(indicators.get('cl_intra_div_type', ''))
    if cl_intra and 'bottom' in cl_intra_type:
        _add_bull(0.14, '笔内底背离→早期反转')
    elif cl_intra and 'top' in cl_intra_type:
        _add_bear(0.14, '笔内顶背离→早期见顶')

    # ════════════════════════════════════════════════════════════════
    # STEP 2 — Advanced dynamics stream (C2/C4/C8)
    # ════════════════════════════════════════════════════════════════
    completion_prob = 0.0
    try:
        from chanlun_dynamics_features import compute_chanlun_dynamics_score_split
        sb1, sbe1, rb1, rbe1, completion_prob = \
            compute_chanlun_dynamics_score_split(indicators)
        score_bull += sb1; score_bear += sbe1
        reasons_bull.extend(rb1); reasons_bear.extend(rbe1)
    except Exception:
        pass

    # ════════════════════════════════════════════════════════════════
    # STEP 3 — Advanced structural stream (C1/C3/C5/C7)
    # ════════════════════════════════════════════════════════════════
    new_zs_prob = 0.0; old_zs_expand_prob = 0.0
    try:
        from chanlun_structural_features import compute_chanlun_structural_score_split
        sb2, sbe2, rb2, rbe2, new_zs_prob, old_zs_expand_prob = \
            compute_chanlun_structural_score_split(indicators)
        score_bull += sb2; score_bear += sbe2
        reasons_bull.extend(rb2); reasons_bear.extend(rbe2)
    except Exception:
        pass

    # ════════════════════════════════════════════════════════════════
    # STEP 4 — Nonlinear squashing (指数包络线压缩)
    # ════════════════════════════════════════════════════════════════
    score_bull = 1.0 - np.exp(-score_bull * 1.5)
    score_bear = 1.0 - np.exp(-score_bear * 1.5)

    # ════════════════════════════════════════════════════════════════
    # STEP 5 — Resonance interception (多空冲突拦截门)
    # ════════════════════════════════════════════════════════════════
    if score_bull > 0.40 and score_bear > 0.40:
        final_score = 0.0; confidence = 0.10; direction = 'neutral'
        reasons_final = ['缠论多空形态强烈共振→中阴观望']
    else:
        raw_final = np.clip(score_bull - score_bear, -1.0, 1.0)

        # ════════════════════════════════════════════════════════════
        # STEP 6 — "走势必完美" saturation asymmetric modifier
        # ════════════════════════════════════════════════════════════
        if raw_final > 0 and completion_prob > 0.75:
            # Bullish but trend is over-saturated → discount (追高风险)
            dynamics_multiplier = 0.60
            reasons_bull.append(f'走势高度饱和(compl={completion_prob:.0%})→折价防追高')
        elif raw_final < 0 and completion_prob > 0.75:
            # Bearish but trend over-saturated → premium (背驰反转溢价)
            dynamics_multiplier = 1.35
            reasons_bull.append(f'下跌走势饱和(compl={completion_prob:.0%})→背驰反转溢价')
        else:
            dynamics_multiplier = 1.0

        # ════════════════════════════════════════════════════════════
        # STEP 7 — Bardo-stage vacuum noise filter (中阴滤噪器)
        # ════════════════════════════════════════════════════════════
        if old_zs_expand_prob > 0.60:
            bardo_multiplier = 0.70
            reasons_bull.append(f'原中枢扩展(prob={old_zs_expand_prob:.0%})→宽幅震荡折价')
        else:
            bardo_multiplier = 1.0

        # 中枢扩展额外折价 (from Step 1j)
        zs_expand_mod = 0.80 if zs_expanded else 1.0
        if zs_expanded:
            reasons_bull.append('中枢扩展→级别升级')

        # ════════════════════════════════════════════════════════════
        # STEP 8 — Final assembly
        # ════════════════════════════════════════════════════════════
        final_score = raw_final * dynamics_multiplier * bardo_multiplier * zs_expand_mod
        final_score = np.clip(final_score, -1.0, 1.0)

        confidence = min(abs(final_score) * 1.20, 0.95)
        direction = 'bullish' if final_score > 0.06 else \
                    ('bearish' if final_score < -0.06 else 'neutral')
        reasons_final = (reasons_bull if final_score > 0 else
                         (reasons_bear if final_score < 0 else
                          reasons_bull + reasons_bear))[:6]

    return {
        'direction': direction,
        'score': round(float(final_score), 3),
        'confidence': round(float(confidence), 3),
        'reasons': reasons_final,
    }


def _compute_school_elliott(indicators: Dict) -> Dict:
    """艾略特波浪学派：浪型+交易信号+规则验证 → 单一判断"""
    score = 0.0
    reasons = []

    ew_trade = str(indicators.get('ew_trade_signal', ''))
    ew_conf = _safe(indicators.get('ew_confidence'), 0)
    ew_rules = indicators.get('ew_rules_valid', True)

    if not ew_rules:
        score *= 0.50; reasons.append('波浪规则违规→结构存疑')

    if ew_conf > 0.5:
        if '做多' in ew_trade or 'buy' in ew_trade.lower():
            score += 0.22; reasons.append(f'波浪做多信号(置信{ew_conf:.2f})')
        elif '做空' in ew_trade or 'sell' in ew_trade.lower():
            score -= 0.22; reasons.append(f'波浪做空信号(置信{ew_conf:.2f})')
    elif ew_conf > 0.3:
        if '做多' in ew_trade or 'buy' in ew_trade.lower():
            score += 0.10; reasons.append(f'波浪偏多(置信{ew_conf:.2f})')
        elif '做空' in ew_trade or 'sell' in ew_trade.lower():
            score -= 0.10; reasons.append(f'波浪偏空(置信{ew_conf:.2f})')

    # 交替规则验证
    if indicators.get('ew_alternation_valid', False):
        score += 0.06; reasons.append('交替规则验证通过')

    # 延长浪
    ew_ext = indicators.get('ew_extension_wave')
    if ew_ext:
        reasons.append(f'浪{ew_ext}延长')

    # 修正浪类型
    ew_corr = str(indicators.get('ew_corrective_type', ''))
    if ew_corr in ['zigzag', 'expanded_flat', 'regular_flat']:
        reasons.append(f'{ew_corr}修正')

    # 倾斜三角形→反转预警
    if indicators.get('ew_is_diagonal', False):
        score -= 0.15; reasons.append('倾斜三角形→楔形反转预警')

    # 浪个性评分
    ew_pers = _safe(indicators.get('ew_personality_score'), 0.5)
    if ew_pers > 0.7:
        score += 0.06; reasons.append('浪个性匹配度高')
    elif ew_pers < 0.3:
        score -= 0.08; reasons.append('浪个性不匹配→结构存疑')

    # 通道位置
    ew_ch_upper = _safe(indicators.get('ew_channel_upper'), 0)
    ew_ch_lower = _safe(indicators.get('ew_channel_lower'), 0)
    cp = _safe(indicators.get('current_price'), 0)
    if ew_ch_upper > 0 and cp > 0:
        if cp >= ew_ch_upper * 0.98:
            score -= 0.06; reasons.append('接近通道上轨')
        elif cp <= ew_ch_lower * 1.02:
            score += 0.06; reasons.append('接近通道下轨')

    score = _clip_score(score)
    return {
        'direction': _direction_from_score(score, 0.08),
        'score': round(score, 3),
        'confidence': round(min(abs(score) * 1.2, 0.95), 3),
        'reasons': reasons[:5],
    }


def _compute_school_tang(indicators: Dict) -> Dict:
    """唐能通体系 — 极性分组+非线性压缩评分 (19 signals across 5 original + 14 advanced)"""
    import numpy as np
    score_bull = 0.0
    score_bear = 0.0
    reasons_bull = []
    reasons_bear = []

    def _add_bull(w, label):
        nonlocal score_bull; score_bull += w; reasons_bull.append(label)
    def _add_bear(w, label):
        nonlocal score_bear; score_bear += w; reasons_bear.append(label)

    # ── Original 5 signals ──
    jt_str = _safe(indicators.get('tang_jiato_strength'), 0)
    jy_str = _safe(indicators.get('tang_jiaya_strength'), 0)
    if indicators.get('tang_jiato', False) and jt_str > 40:
        _add_bull(0.20, f'价托({jt_str:.0f})')
    if indicators.get('tang_jiaya', False) and jy_str > 40:
        _add_bear(0.20, f'价压({jy_str:.0f})')

    tang_triple = str(indicators.get('tang_triple_status', ''))
    if tang_triple == 'triple_golden':    _add_bull(0.25, '三金叉')
    elif tang_triple == 'triple_death':   _add_bear(0.25, '三死叉')
    elif tang_triple == 'partial_golden': _add_bull(0.08, '部分金叉')
    elif tang_triple == 'partial_death':  _add_bear(0.08, '部分死叉')

    runway_grade = str(indicators.get('tang_runway_grade', ''))
    runway_thick = _safe(indicators.get('tang_runway_thickness'), 0)
    if runway_grade == 'thick':           _add_bull(0.15, f'跑道厚({runway_thick:.1%})')
    elif runway_grade == 'moderate':      _add_bull(0.08, '跑道适中')
    elif runway_grade in ('thin','too_thin'): _add_bear(0.08, '跑道偏薄')

    tang_lyt = indicators.get('tang_laoyatou', False)
    lyt_s = _safe(indicators.get('tang_laoyatou_score'), 0)
    lyt_phase = str(indicators.get('tang_laoyatou_phase', ''))
    if tang_lyt and lyt_s > 60:           _add_bull(0.22, f'老鸭头({lyt_phase})')
    elif tang_lyt and lyt_s > 30:         _add_bull(0.08, '老鸭头雏形')

    tang_33 = indicators.get('tang_33_valid', False)
    tang_33_s = _safe(indicators.get('tang_33_score'), 0)
    if tang_33 and tang_33_s > 60:        _add_bull(0.10, '三三过滤')
    elif not tang_33 and tang_33_s > 30:  score_bull *= 0.85; reasons_bull.append('三三未过')

    tang_gs = _safe(indicators.get('tang_golden_score'), 0)
    tang_ds = _safe(indicators.get('tang_death_score'), 0)
    if tang_gs > 60: _add_bull(0.08, f'金叉({tang_gs:.0f})')
    if tang_ds > 60: _add_bear(0.08, f'死叉({tang_ds:.0f})')

    # ── 5 Advanced patterns ──
    try:
        from tang_advanced_features import generate_tnt_features
        import pandas as pd
        cp = _safe(indicators.get('current_price'), 0)
        o_v = _safe(indicators.get('open'), cp); h_v = _safe(indicators.get('high'), cp)
        l_v = _safe(indicators.get('low'), cp);  v_v = _safe(indicators.get('volume'), 0)
        o_df = pd.DataFrame({'S': [o_v]}); h_df = pd.DataFrame({'S': [h_v]})
        l_df = pd.DataFrame({'S': [l_v]}); c_df = pd.DataFrame({'S': [cp]})
        v_df = pd.DataFrame({'S': [v_v]})
        feats = generate_tnt_features(o_df, h_df, l_df, c_df, v_df)
        for key, w, label in [('water_lotus',0.30,'出水芙蓉'),('easy_breakthrough',0.25,'轻松过头'),
                              ('deep_v',0.35,'海底捞月'),('eastern_red_sun',0.20,'东方红大阳升')]:
            if feats[key]['S'].iloc[0] > 0: _add_bull(w, label)
        if feats['poison_spider']['S'].iloc[0] < 0: _add_bear(0.30, '毒蜘蛛')
    except Exception:
        pass

    # ── 9 Remaining patterns ──
    try:
        from tang_remaining_features import tang_remaining_signal_score_split
        bull_sub, bear_sub, reasons_bull_sub, reasons_bear_sub = tang_remaining_signal_score_split(indicators)
        score_bull += bull_sub; score_bear += bear_sub
        reasons_bull.extend(reasons_bull_sub); reasons_bear.extend(reasons_bear_sub)
    except Exception:
        pass

    # ── Nonlinear squashing: 1 - exp(-x) keeps each polarity in [0,0.95] ──
    score_bull = 1.0 - np.exp(-score_bull * 1.5)
    score_bear = 1.0 - np.exp(-score_bear * 1.5)

    # ── Conflict detection: both > 0.40 → chaotic, kill signal ──
    if score_bull > 0.40 and score_bear > 0.40:
        final_score = 0.0
        confidence = 0.10
        direction = 'neutral'
        reasons_final = ['信号矛盾(多空共振)→观望']
    else:
        final_score = np.clip(score_bull - score_bear, -1.0, 1.0)
        confidence = min(abs(final_score) * 1.25, 0.92)
        direction = 'bullish' if final_score > 0.06 else ('bearish' if final_score < -0.06 else 'neutral')
        reasons_final = (reasons_bull if final_score > 0 else (reasons_bear if final_score < 0 else reasons_bull + reasons_bear))[:5]

    return {
        'direction': direction,
        'score': round(float(final_score), 3),
        'confidence': round(float(confidence), 3),
        'reasons': reasons_final,
    }


def _compute_school_livermore(indicators: Dict) -> Dict:
    """利弗莫尔战法 — 极性分组+非线性压缩 (3 original + 6 advanced signals)"""
    import numpy as np
    score_bull = 0.0
    score_bear = 0.0
    reasons_bull = []
    reasons_bear = []

    def _add_bull(w, label):
        nonlocal score_bull; score_bull += w; reasons_bull.append(label)
    def _add_bear(w, label):
        nonlocal score_bear; score_bear += w; reasons_bear.append(label)

    # ── Original 3 signals ──
    liv_sig = str(indicators.get('livermore_signal', ''))
    if liv_sig == 'strong_buy':   _add_bull(0.25, '关键点强势突破→强买入')
    elif liv_sig == 'buy':        _add_bull(0.12, '关键点附近→偏多')
    elif liv_sig == 'strong_sell': _add_bear(0.25, '关键点跌破→强卖出')
    elif liv_sig == 'sell':       _add_bear(0.12, '关键点下方→偏空')

    liv_danger = indicators.get('livermore_danger', False)
    liv_dl = _safe(indicators.get('livermore_danger_level'), 0)
    if liv_danger and liv_dl >= 70:   _add_bear(0.30, f'危险信号Lv{liv_dl:.0f}→离场')
    elif liv_danger and liv_dl >= 40: _add_bear(0.18, f'危险信号Lv{liv_dl:.0f}→减仓')
    elif liv_danger and liv_dl >= 20: score_bear *= 0.75; reasons_bear.append('危险预警')

    liv_action = str(indicators.get('livermore_action', ''))
    if liv_action and 'buy' in liv_action.lower():
        reasons_bull.append(liv_action[:30])
    elif liv_action:
        reasons_bear.append(liv_action[:30])

    # ── 6 Advanced patterns (split bull/bear) ──
    try:
        from livermore_features import compute_livermore_signal_score_split
        s_bull, s_bear, r_bull, r_bear = compute_livermore_signal_score_split(indicators)
        score_bull += s_bull; score_bear += s_bear
        reasons_bull.extend(r_bull); reasons_bear.extend(r_bear)
    except Exception:
        pass

    # ── Nonlinear squashing ──
    score_bull = 1.0 - np.exp(-score_bull * 1.5)
    score_bear = 1.0 - np.exp(-score_bear * 1.5)

    # ── Conflict detection ──
    if score_bull > 0.40 and score_bear > 0.40:
        final_score = 0.0; confidence = 0.10; direction = 'neutral'
        reasons_final = ['多空共振→观望']
    else:
        final_score = np.clip(score_bull - score_bear, -1.0, 1.0)
        confidence = min(abs(final_score) * 1.30, 0.95)
        direction = 'bullish' if final_score > 0.08 else ('bearish' if final_score < -0.08 else 'neutral')
        reasons_final = (reasons_bull if final_score > 0 else (reasons_bear if final_score < 0 else reasons_bull + reasons_bear))[:5]

    return {
        'direction': direction,
        'score': round(float(final_score), 3),
        'confidence': round(float(confidence), 3),
        'reasons': reasons_final,
    }


def _compute_school_busch(indicators: Dict) -> Dict:
    """Busch量化体系 — 极性分组+非线性压缩 (8 original + 6 advanced signals)"""
    import numpy as np
    score_bull = 0.0; score_bear = 0.0
    reasons_bull = []; reasons_bear = []
    def _add_bull(w, label):
        nonlocal score_bull; score_bull += w; reasons_bull.append(label)
    def _add_bear(w, label):
        nonlocal score_bear; score_bear += w; reasons_bear.append(label)

    # ── 2560战法 ──
    b2560 = str(indicators.get('b2560_signal', ''))
    if b2560 == 'strong_buy':   _add_bull(0.22, '2560强买入')
    elif b2560 == 'sell':       _add_bear(0.18, '2560卖出')
    elif b2560 == 'hold_long':  _add_bull(0.08, '2560持仓偏多')
    elif b2560 == 'hold_short': _add_bear(0.08, '2560观望偏空')

    b2560_w = str(indicators.get('b2560_weekly_signal', ''))
    if b2560_w == 'weekly_golden_cross': _add_bull(0.10, '2560周线金叉')
    elif b2560_w == 'weekly_dead_cross':  _add_bear(0.10, '2560周线死叉')

    # ── DMI增强 ──
    bdmi = str(indicators.get('bdmi_signal', ''))
    if bdmi == 'bullish_reversal': _add_bull(0.18, 'DMI转折买入')
    elif bdmi == 'bullish_trend':  _add_bull(0.10, 'DMI趋势做多')
    elif bdmi == 'bearish_trend':  _add_bear(0.12, 'DMI趋势做空')
    elif bdmi == 'weak_market':    score_bear *= 0.60; reasons_bear.append('弱势市场')
    if indicators.get('bdmi_big_momentum'): reasons_bull.append('大动量信号')

    # ── 量价代数 ──
    vpa_sig = str(indicators.get('vpa_signal', ''))
    vpa_f = str(indicators.get('vpa_formula', ''))
    if vpa_sig == 'bullish':         _add_bull(0.12, f'量价:{vpa_f}')
    elif vpa_sig == 'bearish':       _add_bear(0.15, f'量价:{vpa_f}')
    elif vpa_sig == 'bearish_warning': _add_bear(0.08, '价升量缩')
    elif vpa_sig == 'bullish_warning': _add_bull(0.06, '价跌量缩')
    if indicators.get('vpa_is_extreme_vol'): score_bear += 0.10; reasons_bear.append('天量')

    # ── 量能形态 ──
    vc_sig = str(indicators.get('vc_signal', ''))
    if vc_sig == 'breakout':           _add_bull(0.12, '聚量突破')
    elif vc_sig == 'distribution':     _add_bear(0.12, '放量出货')
    elif vc_sig == 'contrarian_bullish': _add_bull(0.10, '地量见底')
    elif vc_sig == 'trend_weakening':  score_bear *= 0.85; reasons_bear.append('量能消散')

    # ── 综合量价 ──
    cva = str(indicators.get('cva_composite_signal', ''))
    if '强烈看多' in cva:   _add_bull(0.15, 'CVA强烈看多')
    elif '偏多' in cva:     _add_bull(0.08, 'CVA偏多')
    elif '强烈看空' in cva: _add_bear(0.15, 'CVA强烈看空')
    elif '偏空' in cva:     _add_bear(0.08, 'CVA偏空')

    # ── 九大量价关系 ──
    v9r = _safe(indicators.get('v9r_id'), 0)
    if v9r in [1,9]:   _add_bull(0.12, f'量价关系({int(v9r)})')
    elif v9r == 2:     _add_bear(0.10, '量价关系(2)顶背离')
    elif v9r == 4:     _add_bear(0.14, '量价关系(4)空头')
    elif v9r == 5:     _add_bull(0.10, '量价关系(5)底背离')

    # ── 格兰维尔 ──
    g8_cat = str(indicators.get('g8_rule_category', ''))
    g8_rule = indicators.get('g8_active_rule')
    if 'buy' in g8_cat and 'contrarian' not in g8_cat:    _add_bull(0.14, f'格兰维尔买{g8_rule}')
    elif 'sell' in g8_cat and 'contrarian' not in g8_cat:  _add_bear(0.14, f'格兰维尔卖{g8_rule}')
    elif 'buy_contrarian' in g8_cat:  _add_bull(0.06, f'格兰维尔超跌{g8_rule}')
    elif 'sell_contrarian' in g8_cat: _add_bear(0.06, f'格兰维尔超涨{g8_rule}')
    if '地量密集' in str(indicators.get('g8_land_volume_zone', '')): _add_bull(0.08, '地量密集')

    # ── 6 Advanced Busch patterns ──
    ts_filter = 1.0
    try:
        from busch_advanced_features import compute_busch_signal_score_split
        sb, sbe, rb, rbe, tsf = compute_busch_signal_score_split(indicators)
        score_bull += sb; score_bear += sbe
        reasons_bull.extend(rb); reasons_bear.extend(rbe)
        ts_filter = tsf
    except Exception:
        pass

    # ── Nonlinear squashing ──
    score_bull = 1.0 - np.exp(-score_bull * 1.5)
    score_bear = 1.0 - np.exp(-score_bear * 1.5)

    # ── Conflict detection (MUST precede filter — filter must not mask conflict) ──
    if score_bull > 0.40 and score_bear > 0.40:
        final_score = 0.0; confidence = 0.10; direction = 'neutral'
        reasons_final = ['多空共振→观望']
    else:
        raw_final = np.clip(score_bull - score_bear, -1.0, 1.0)
        # Apply trend strength filter to net score only (not to raw polarity)
        final_score = raw_final * ts_filter
        confidence = min(abs(final_score) * 1.20, 0.95)
        direction = 'bullish' if final_score > 0.06 else ('bearish' if final_score < -0.06 else 'neutral')
        reasons_final = (reasons_bull if final_score > 0 else (reasons_bear if final_score < 0 else reasons_bull + reasons_bear))[:5]

    return {
        'direction': direction,
        'score': round(float(final_score), 3),
        'confidence': round(float(confidence), 3),
        'reasons': reasons_final,
    }


def _compute_school_classical(indicators: Dict) -> Dict:
    """经典技术分析：均线+MACD+RSI+布林+DMA+DMI+TD+K线+OBV+VR+动量 → 综合判断"""
    score = 0.0
    reasons = []

    # MA排列
    ma5 = _safe(indicators.get('ma5'))
    ma20 = _safe(indicators.get('ma20'))
    ma60 = _safe(indicators.get('ma60'))
    if ma5 > ma20 > ma60:
        score += 0.18; reasons.append('MA多头排列')
    elif ma5 < ma20 < ma60:
        score -= 0.18; reasons.append('MA空头排列')
    elif ma5 > ma20:
        score += 0.06; reasons.append('短均金叉')
    elif ma5 < ma20:
        score -= 0.06; reasons.append('短均死叉')

    cp = _safe(indicators.get('current_price'))
    if cp > 0 and ma20 > 0:
        if cp > ma20:
            score += 0.05; reasons.append('价格>MA20')
        else:
            score -= 0.05; reasons.append('价格<MA20')

    # MACD
    dif = _safe(indicators.get('macd_dif'))
    dea = _safe(indicators.get('macd_dea'))
    hist = _safe(indicators.get('macd_hist'))
    if dif > dea and hist > 0:
        score += 0.12; reasons.append('MACD金叉+红柱')
    elif dif < dea and hist < 0:
        score -= 0.12; reasons.append('MACD死叉+绿柱')
    elif dif > dea:
        score += 0.05; reasons.append('MACD弱金叉')
    elif dif < dea:
        score -= 0.05; reasons.append('MACD弱死叉')

    # RSI
    rsi = _safe(indicators.get('rsi'), 50)
    if rsi > 70:
        score -= 0.06; reasons.append(f'RSI={rsi:.1f}超买')
    elif rsi < 30:
        score += 0.06; reasons.append(f'RSI={rsi:.1f}超卖')
    elif rsi > 60:
        score += 0.06; reasons.append(f'RSI={rsi:.1f}偏强')
    elif rsi < 40:
        score -= 0.06; reasons.append(f'RSI={rsi:.1f}偏弱')

    # 布林带
    bb_u = _safe(indicators.get('bb_upper'))
    bb_l = _safe(indicators.get('bb_lower'))
    bb_m = _safe(indicators.get('bb_mid'))
    if cp > 0 and bb_u > 0:
        if cp >= bb_u * 0.98:
            score -= 0.05; reasons.append('触及布林上轨')
        elif cp <= bb_l * 1.02:
            score += 0.05; reasons.append('触及布林下轨')
        elif cp > bb_m:
            score += 0.03
        else:
            score -= 0.03

    # DMA
    dma_sig = str(indicators.get('dma_signal', ''))
    if '金叉' in dma_sig:
        score += 0.10; reasons.append('DMA金叉')
    elif '死叉' in dma_sig:
        score -= 0.10; reasons.append('DMA死叉')

    # DMI
    adx = _safe(indicators.get('dmi_adx'))
    di_dir = str(indicators.get('dmi_di_direction', ''))
    if adx > 25:
        if '多' in di_dir or '+DI' in di_dir:
            score += 0.12; reasons.append(f'ADX={adx:.1f}>25多方')
        elif '空' in di_dir or '-DI' in di_dir:
            score -= 0.12; reasons.append(f'ADX={adx:.1f}>25空方')
    elif adx < 20:
        score *= 0.70; reasons.append(f'ADX={adx:.1f}<20震荡市')

    # TD Sequential
    td_comp = indicators.get('td_completed', False)
    td_dir = str(indicators.get('td_direction', ''))
    td_cnt = _safe(indicators.get('td_count'))
    if td_comp:
        if td_dir == 'buy' or '买' in td_dir:
            score += 0.12; reasons.append('TD买入结构完成(9-13)')
        elif td_dir == 'sell' or '卖' in td_dir:
            score -= 0.12; reasons.append('TD卖出结构完成(9-13)')
    elif td_cnt >= 8:
        if td_dir == 'buy':
            score += 0.05; reasons.append(f'TD买入计数{int(td_cnt)}/9')
        elif td_dir == 'sell':
            score -= 0.05; reasons.append(f'TD卖出计数{int(td_cnt)}/9')

    # K线形态
    pat_rec = str(indicators.get('pattern_recent', ''))
    pat_mor = _safe(indicators.get('pattern_morning_star_n'))
    pat_eve = _safe(indicators.get('pattern_evening_star_n'))
    pat_bot = _safe(indicators.get('pattern_bottom_fractal_n'))
    pat_top = _safe(indicators.get('pattern_top_fractal_n'))
    if pat_mor > 0:
        score += 0.08 * min(pat_mor, 3); reasons.append(f'晨星×{int(pat_mor)}')
    if pat_eve > 0:
        score -= 0.08 * min(pat_eve, 3); reasons.append(f'暮星×{int(pat_eve)}')
    if pat_bot > 0:
        score += 0.05 * min(pat_bot, 3)
    if pat_top > 0:
        score -= 0.05 * min(pat_top, 3)

    # K线形态关键词（排除否定前缀）
    for kw, kw_cn, sign in [
        ('晨星', '晨星', 0.10), ('暮星', '暮星', -0.10),
        ('吞没看涨', '吞没看涨', 0.10), ('吞没看跌', '吞没看跌', -0.10),
        ('刺透', '刺透', 0.08), ('乌云盖顶', '乌云盖顶', -0.08),
    ]:
        idx = pat_rec.find(kw)
        if idx >= 0 and kw_cn not in str(reasons):
            if idx > 0 and pat_rec[idx - 1] in ('无', '非', '不'):
                continue
            score += sign
            reasons.append(kw_cn)

    # 支撑阻力
    ver_sup = indicators.get('verified_support', [])
    ver_res = indicators.get('verified_resistance', [])
    if ver_sup and len(ver_sup) > 0:
        score += 0.04; reasons.append(f'多指标支撑:{ver_sup[:2]}')
    if ver_res and len(ver_res) > 0:
        score -= 0.04; reasons.append(f'多指标阻力:{ver_res[:2]}')

    # OBV
    obv_div = str(indicators.get('obv_divergence', ''))
    if '底背离' in obv_div:
        score += 0.12; reasons.append('OBV底背离')
    elif '顶背离' in obv_div:
        score -= 0.12; reasons.append('OBV顶背离')

    # VR
    vr_val = _safe(indicators.get('vr_value'), 100)
    vr_sig = str(indicators.get('vr_signal', ''))
    if vr_sig == 'bearish' and vr_val > 300:
        score -= 0.10; reasons.append(f'VR超买({vr_val:.0f})')
    elif vr_sig == 'contrarian_bullish':
        score += 0.10; reasons.append(f'VR底部({vr_val:.0f})→超卖反弹')
    elif vr_sig == 'bullish':
        score += 0.05; reasons.append(f'VR强势({vr_val:.0f})')

    # 动量共振
    mom_res = indicators.get('momentum_resonance', False)
    mom_dir = str(indicators.get('momentum_direction', ''))
    if mom_res:
        if 'bull' in mom_dir or '多' in mom_dir:
            score += 0.10; reasons.append('动量共振偏多')
        elif 'bear' in mom_dir or '空' in mom_dir:
            score -= 0.10; reasons.append('动量共振偏空')

    # ROC
    roc_s = _safe(indicators.get('roc_short'))
    roc_l = _safe(indicators.get('roc_long'))
    if roc_s > 0 and roc_l > 0:
        score += 0.04
    elif roc_s < 0 and roc_l < 0:
        score -= 0.04

    # 量价配合
    if indicators.get('vol_price_resonance', False):
        vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)
        if vol_ratio > 1.5:
            score += 0.12; reasons.append(f'放量配合({vol_ratio:.1f}x)')
        else:
            score += 0.06; reasons.append('量价配合')
    if indicators.get('vol_stacking', False):
        score += 0.10; reasons.append('堆量→主力介入')
    if indicators.get('high_vol_stagnation', False):
        score -= 0.12; reasons.append('放量滞涨→警惕出货')
    if indicators.get('low_vol_pullback', False):
        score += 0.08; reasons.append('缩量回踩→洗盘特征')

    # 主力资金
    sm_sig = str(indicators.get('sm_smart_signal', ''))
    if sm_sig == 'strong_accumulation':
        score += 0.12; reasons.append('主力强烈收集')
    elif sm_sig == 'strong_distribution':
        score -= 0.12; reasons.append('主力强烈派发')
    elif sm_sig == 'accumulation':
        score += 0.06
    elif sm_sig == 'distribution':
        score -= 0.06

    # CCI
    cci_s = _safe(indicators.get('cci_short'))
    if cci_s > 100:
        score += 0.04
    elif cci_s < -100:
        score -= 0.04

    score = _clip_score(score)
    conf_mult = 0.85 if adx < 20 else 1.15  # 震荡市降低经典TA置信度
    return {
        'direction': _direction_from_score(score, 0.08),
        'score': round(score, 3),
        'confidence': round(min(abs(score) * conf_mult, 0.95), 3),
        'reasons': reasons[:6],
    }


def _compute_school_risk(indicators: Dict) -> Dict:
    """风险与市场环境：波动率+市场阶段+风险预警 → 风险评分（正=低风险偏多，负=高风险偏空）"""
    score = 0.0
    reasons = []

    # 贝叶斯融合后验（来自9维/7维证据融合的结果）
    bayes_post = _safe(indicators.get('bayes_fused_posterior'), 0.5)
    bayes_sig = str(indicators.get('bayes_fused_signal', 'neutral'))
    bayes_ent = _safe(indicators.get('bayes_fused_entropy'), 0.5)
    dims_active = _safe(indicators.get('bayes_dimensions_active'), 0)

    if bayes_ent > 0.85:
        reasons.append(f'贝叶斯高熵({bayes_ent:.3f})→信号不可靠')
        score *= 0.30
    elif dims_active >= 5 and bayes_post > 0.65:
        if bayes_sig == 'bullish':
            score += 0.20; reasons.append(f'贝叶斯强看多({int(dims_active)}维共振)')
        elif bayes_sig == 'bearish':
            score -= 0.20; reasons.append(f'贝叶斯强看空({int(dims_active)}维共振)')
    elif bayes_post > 0.55:
        if bayes_sig == 'bullish':
            score += 0.08; reasons.append(f'贝叶斯偏多')
        elif bayes_sig == 'bearish':
            score -= 0.08; reasons.append(f'贝叶斯偏空')

    # 波动率状态
    rv_lvl = str(indicators.get('rv_level', ''))
    rv_pct = _safe(indicators.get('rv_percentile'), 0.5)
    har_dir = str(indicators.get('har_direction', ''))

    if '极高' in rv_lvl:
        score *= 0.50; reasons.append(f'极高波动(分位{rv_pct:.0%})→避险')
    elif '高' in rv_lvl:
        score *= 0.70; reasons.append(f'高波动(分位{rv_pct:.0%})→降仓')
    elif '极低' in rv_lvl and har_dir == '扩张':
        score += 0.06; reasons.append('低波扩张→酝酿突破')

    if har_dir == '扩张' and rv_pct > 0.7:
        score *= 0.80; reasons.append('HAR扩张+高波位→警惕')
    elif har_dir == '收缩' and rv_pct > 0.8:
        score += 0.04; reasons.append('HAR收缩→恐慌消退')

    # BEGE
    bebe_reg = str(indicators.get('bebe_regime', ''))
    bebe_vrp = str(indicators.get('bebe_vrp_signal', 'neutral'))
    bebe_ratio = _safe(indicators.get('bebe_good_bad_ratio'), 1.0)
    bebe_asym = _safe(indicators.get('bebe_vol_asymmetry'), 0.5)
    bebe_gt = str(indicators.get('bebe_good_vol_trend', 'stable'))
    bebe_bt = str(indicators.get('bebe_bad_vol_trend', 'stable'))

    if bebe_reg == 'bad_environment':
        score -= 0.18; reasons.append(f'BEGE坏环境(不对称{bebe_asym:.0%})→下行风险')
    elif bebe_reg == 'good_environment':
        score += 0.10; reasons.append(f'BEGE好环境(比率{bebe_ratio:.2f})→风险偏好强')

    if bebe_bt == 'rising' and bebe_gt == 'falling':
        score -= 0.12; reasons.append('BEGE分布偏移→坏波升+好波降')

    if bebe_vrp == 'high_premium':
        score -= 0.08; reasons.append('VRP高溢价→市场恐慌')
    elif bebe_vrp == 'negative_premium':
        score -= 0.10; reasons.append('VRP负溢价→异常信号')

    # 市场四阶段
    mphase = str(indicators.get('mp_phase', ''))
    mp_conf = _safe(indicators.get('mp_confidence'), 0)
    mp_cred = _safe(indicators.get('mp_trend_credibility'), 0.5)
    if mphase == '拉升' and mp_conf > 0.6:
        score += 0.12; reasons.append(f'拉升阶段(可信{mp_cred:.0%})')
    elif mphase == '盘头' and mp_conf > 0.5:
        score -= 0.10; reasons.append('盘头阶段→减仓')
    elif mphase == '下跌' and mp_conf > 0.5:
        score -= 0.15; reasons.append('下跌阶段→空仓')
    elif mphase == '筑底' and mp_conf > 0.5:
        score += 0.04; reasons.append('筑底阶段→关注反转')

    # 逃顶信号
    top_grade = str(indicators.get('top_escape_grade', ''))
    top_prob = _safe(indicators.get('top_escape_prob'), 0)
    top_cnt = _safe(indicators.get('top_escape_count'), 0)
    if top_grade == 'critical':
        score -= 0.22; reasons.append(f'逃顶临界({int(top_cnt)}个,{top_prob:.0%})→立即离场')
    elif top_grade == 'high_risk':
        score -= 0.15; reasons.append(f'逃顶高风险({int(top_cnt)}个)→减仓')
    elif top_grade == 'elevated_risk':
        score -= 0.06; reasons.append('逃顶风险上升')

    # 分配日
    if indicators.get('dist_days_warning', False):
        dist_cnt = _safe(indicators.get('dist_days_count'), 0)
        score -= 0.10; reasons.append(f'分配日{int(dist_cnt)}个→顶部预警')

    # NRB
    nrb_sig = str(indicators.get('nrb_signal', ''))
    if nrb_sig == 'bullish_reversal':
        score += 0.06; reasons.append('NRB底部反转')
    elif nrb_sig == 'bearish_reversal':
        score -= 0.06; reasons.append('NRB顶部反转')

    # 波动率仓位建议
    vol_mult = _safe(indicators.get('vol_position_mult'), 1.0)
    if vol_mult < 0.8:
        reasons.append(f'波动仓位乘数={vol_mult:.0%}→建议减仓')

    # KMeans cluster
    cluster = _safe(indicators.get('kmeans_cluster'), -1)
    if cluster >= 0:
        reasons.append(f'KMeans模式#{int(cluster)}')

    score = _clip_score(score)
    return {
        'direction': _direction_from_score(score, 0.06),
        'score': round(score, 3),
        'confidence': round(min(abs(score) * 1.25, 0.95), 3),
        'reasons': reasons[:5],
    }


def _compute_school_gann(indicators: Dict) -> Dict:
    """江恩理论学派 — 极性分组 + 非线性压缩 + 双重确认安全拓扑

    Architecture (Busch-equivalent topology):
      1. Polarity-grouped accumulation: ALL 18+ original indicators → score_bull / score_bear
      2. Gann advanced features (price-directional only, time signals stripped)
      3. Gann final features (3d_swing + gann_fan are price; seasonal_windows is time-only)
      4. Nonlinear squashing: 1.0 - exp(-x * 1.5) on both polarities
      5. Resonance interception: if bull > 0.40 and bear > 0.40 → final_score = 0
      6. Net score: raw_final = clip(score_bull - score_bear, -1.0, 1.0)
      7. Double confirmation multiplier (applied AFTER conflict gate):
         multiplier = 1.50 if (has_price AND has_time)
                    = 0.70 if (has_price only)
                    = 0.0  otherwise
      8. Confidence modifiers (volatility / ADX / discipline) → conf_mod
      9. final_score = raw_final * multiplier * conf_mod
    """
    import numpy as np
    score_bull = 0.0; score_bear = 0.0
    reasons_bull = []; reasons_bear = []
    def _add_bull(w, label):
        nonlocal score_bull; score_bull += w; reasons_bull.append(label)
    def _add_bear(w, label):
        nonlocal score_bear; score_bear += w; reasons_bear.append(label)

    cp = _safe(indicators.get('current_price'), 0)
    atr14 = _safe(indicators.get('vol_atr14'), 0)
    atr_pct = _safe(indicators.get('vol_atr_pct'), 0.02)

    # ════════════════════════════════════════════════════════════════
    # SECTION A — Original 13 Gann rule indicators (polarity-grouped)
    # ════════════════════════════════════════════════════════════════

    # ── 法则1: 趋势判断（均线排列）──
    ma5 = _safe(indicators.get('ma5'), 0)
    ma20 = _safe(indicators.get('ma20'), 0)
    ma60 = _safe(indicators.get('ma60'), 0)

    if ma5 > ma20 > ma60 and cp > 0:
        _add_bull(0.15, '均线多头排列→趋势向上')
    elif ma5 < ma20 < ma60 and cp > 0:
        _add_bear(0.15, '均线空头排列→趋势向下')
    elif ma5 > ma20 and cp > 0:
        _add_bull(0.06, '短期均线金叉')
    elif ma5 < ma20 and cp > 0:
        _add_bear(0.06, '短期均线死叉')

    # MACD 辅助确认
    dif = _safe(indicators.get('macd_dif'), 0)
    dea = _safe(indicators.get('macd_dea'), 0)
    hist = _safe(indicators.get('macd_hist'), 0)
    if dif > dea and hist > 0:
        _add_bull(0.06, 'MACD金叉+红柱')
    elif dif < dea and hist < 0:
        _add_bear(0.06, 'MACD死叉+绿柱')

    # ── 法则3: 百分比回调（布林带位置）──
    bb_u = _safe(indicators.get('bb_upper'), 0)
    bb_l = _safe(indicators.get('bb_lower'), 0)
    if cp > 0 and bb_u > 0:
        bb_range = bb_u - bb_l
        if bb_range > 0:
            bb_pos = (cp - bb_l) / bb_range
            if bb_pos < 0.15:
                _add_bull(0.08, f'价格近布林下轨(位置{bb_pos:.0%})→超卖区')
            elif bb_pos > 0.85:
                _add_bear(0.06, f'价格近布林上轨(位置{bb_pos:.0%})→超买区')
            elif 0.45 <= bb_pos <= 0.55:
                reasons_bull.append('价格在布林50%中位')  # neutral, no score

    # 价格相对MA20 — 50%回调核心定位
    if cp > 0 and ma20 > 0:
        price_vs_ma20 = (cp - ma20) / ma20
        if -0.03 <= price_vs_ma20 <= 0.03:
            _add_bull(0.05, '价格在MA20(≈50%回调带)→关键平衡点')
        elif -0.05 <= price_vs_ma20 < -0.03:
            _add_bull(0.06, '价格在MA20下方(63%回调带)→潜在支撑')

    # ── 法则2: 底/顶分形形态 ──
    pat_top_n = _safe(indicators.get('pattern_top_fractal_n'), 0)
    pat_bot_n = _safe(indicators.get('pattern_bottom_fractal_n'), 0)
    if pat_bot_n >= 2:
        _add_bull(0.10, f'双底/三底形态(底分型×{int(pat_bot_n)})→强支撑')
    elif pat_bot_n == 1:
        _add_bull(0.04, '单底分型')
    if pat_top_n >= 2:
        _add_bear(0.10, f'双顶/三顶形态(顶分型×{int(pat_top_n)})→强阻力')
    elif pat_top_n == 1:
        _add_bear(0.04, '单顶分型')

    # 验证支撑/阻力
    ver_sup = indicators.get('verified_support', [])
    ver_res = indicators.get('verified_resistance', [])
    if ver_sup:
        _add_bull(0.06, f'多指标验证支撑:{ver_sup[:2]}')
    if ver_res:
        _add_bear(0.06, f'多指标验证阻力:{ver_res[:2]}')

    # ── 法则6: 5-7点波动（RSI + ROC + CCI）──
    rsi = _safe(indicators.get('rsi'), 50)
    roc_s = _safe(indicators.get('roc_short'), 0)
    roc_l = _safe(indicators.get('roc_long'), 0)

    if rsi < 30:
        _add_bull(0.08, f'RSI={rsi:.0f}超卖→5-7点回调完成')
    elif rsi > 70:
        _add_bear(0.06, f'RSI={rsi:.0f}超买→5-7点反弹完成')
    elif rsi > 60:
        _add_bull(0.04, f'RSI={rsi:.0f}偏多')
    elif rsi < 40:
        _add_bear(0.04, f'RSI={rsi:.0f}偏空')

    if roc_s > 0 and roc_l > 0:
        _add_bull(0.04, 'ROC双周期正向')
    elif roc_s < 0 and roc_l < 0:
        _add_bear(0.04, 'ROC双周期负向')

    cci_s = _safe(indicators.get('cci_short'), 0)
    if cci_s < -150:
        _add_bull(0.06, 'CCI极端超卖→报复反弹概率高')
    elif cci_s > 150:
        _add_bear(0.04, 'CCI极端超买→回调概率高')

    # ── 法则7: 成交量验证 ──
    vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)
    obv_div = str(indicators.get('obv_divergence', ''))

    if indicators.get('vol_price_resonance', False) and vol_ratio > 1.5:
        _add_bull(0.08, f'放量共振(vol_ratio={vol_ratio:.1f})')
    elif indicators.get('vol_stacking', False):
        _add_bull(0.08, '堆量→主力介入')

    if '底背离' in obv_div:
        _add_bull(0.10, 'OBV底背离→下跌量能衰竭')
    elif '顶背离' in obv_div:
        _add_bear(0.10, 'OBV顶背离→上涨量能衰竭')

    # ── 法则8: 市场阶段（方向性指标，非时间信号）──
    mp_phase = str(indicators.get('mp_phase', ''))
    mp_conf = _safe(indicators.get('mp_confidence'), 0)

    if mp_phase == '拉升' and mp_conf > 0.5:
        _add_bull(0.06, '拉升阶段→顺势做多')
    elif mp_phase == '下跌' and mp_conf > 0.5:
        _add_bear(0.10, '下跌阶段→顺势做空')
    elif mp_phase == '筑底' and mp_conf > 0.5:
        _add_bull(0.04, '筑底阶段→关注时间窗口')
    elif mp_phase == '盘头' and mp_conf > 0.5:
        _add_bear(0.06, '盘头阶段→谨慎')

    # 动量超买超卖（方向性）
    momentum_state_s = str(indicators.get('momentum_state_short', ''))
    momentum_state_l = str(indicators.get('momentum_state_long', ''))
    if '超卖' in momentum_state_s and '超卖' in momentum_state_l:
        _add_bull(0.04, '短+长周期超卖→变盘窗口临近')
    if '超买' in momentum_state_s and '超买' in momentum_state_l:
        _add_bear(0.04, '短+长周期超买→变盘窗口临近')

    # ── 法则5: 动量共振 ──
    if indicators.get('momentum_resonance', False):
        mom_dir = str(indicators.get('momentum_direction', ''))
        if 'bull' in mom_dir:
            _add_bull(0.06, '多周期动量共振→看多')
        elif 'bear' in mom_dir:
            _add_bear(0.06, '多周期动量共振→看空')

    # ── 法则9: 创新高/新低 ──
    if cp > 0 and bb_u > 0 and cp >= bb_u * 1.02:
        _add_bull(0.06, '突破布林上轨→创新高信号')
    elif cp > 0 and bb_l > 0 and cp <= bb_l * 0.98:
        _add_bear(0.06, '跌破布林下轨→创新低信号')

    # ── 波动法则: 多维度共振 ──
    resonance_count = 0
    if pat_bot_n >= 2: resonance_count += 1
    if rsi < 35: resonance_count += 1
    if '底背离' in obv_div: resonance_count += 1
    if mp_phase == '筑底' and mp_conf > 0.5: resonance_count += 1

    if resonance_count >= 3:
        _add_bull(0.12, f'江恩共振({resonance_count}维)→强信号')
    elif resonance_count >= 2:
        _add_bull(0.05, f'江恩弱共振({resonance_count}维)')

    # ── BEGE ──
    bebe_reg = str(indicators.get('bebe_regime', ''))
    if bebe_reg == 'good_environment':
        _add_bull(0.04, 'BEGE有利环境')
    elif bebe_reg == 'bad_environment':
        _add_bear(0.06, 'BEGE不利环境')

    # ── ADX趋势强度确认 ──
    adx = _safe(indicators.get('dmi_adx'), 20)
    di_dir = str(indicators.get('dmi_di_direction', ''))
    if adx > 25:
        if '多' in di_dir:
            _add_bull(0.06, f'ADX={adx:.1f}>25趋势确认偏多')
        elif '空' in di_dir:
            _add_bear(0.06, f'ADX={adx:.1f}>25趋势确认偏空')

    # ════════════════════════════════════════════════════════════════
    # SECTION B — Gann advanced features (price-directional only)
    # ════════════════════════════════════════════════════════════════
    has_time_signal = False

    try:
        from gann_advanced_features import compute_gann_signal_score_split
        sb, sbe, rb, rbe, is_time = compute_gann_signal_score_split(indicators)
        score_bull += sb; score_bear += sbe
        reasons_bull.extend(rb); reasons_bear.extend(rbe)
        if is_time: has_time_signal = True
    except Exception:
        pass

    # ════════════════════════════════════════════════════════════════
    # SECTION C — Gann final features (3d_swing + gann_fan = price;
    #             seasonal_windows = time-only)
    # ════════════════════════════════════════════════════════════════
    try:
        from gann_final_features import compute_gann_final_score_split
        sb2, sbe2, rb2, rbe2, hp, ht = compute_gann_final_score_split(indicators)
        score_bull += sb2; score_bear += sbe2
        reasons_bull.extend(rb2); reasons_bear.extend(rbe2)
        if ht: has_time_signal = True
    except Exception:
        pass

    # ════════════════════════════════════════════════════════════════
    # SECTION D — Nonlinear squashing (both polarities)
    # ════════════════════════════════════════════════════════════════
    score_bull = 1.0 - np.exp(-score_bull * 1.5)
    score_bear = 1.0 - np.exp(-score_bear * 1.5)

    # ════════════════════════════════════════════════════════════════
    # SECTION E — Confidence modifier stack
    #   (volatility / ADX / discipline — multiplicative, applied LAST)
    # ════════════════════════════════════════════════════════════════
    conf_mod = 1.0
    conf_reasons = []

    # 法则12: 快速市场
    if atr_pct > 0.05:
        conf_mod *= 0.75; conf_reasons.append(f'高波动(ATR={atr_pct:.1%})→趋势不持久')
    elif atr_pct > 0.035:
        conf_mod *= 0.85

    # 波动率环境
    rv_lvl = str(indicators.get('rv_level', ''))
    rv_pct = _safe(indicators.get('rv_percentile'), 0.5)
    if '极高' in rv_lvl or rv_pct > 0.9:
        conf_mod *= 0.65; conf_reasons.append(f'极端波动(分位{rv_pct:.0%})→减仓观望')
    elif '高' in rv_lvl:
        conf_mod *= 0.80

    # ADX震荡
    if adx < 18:
        conf_mod *= 0.75; conf_reasons.append(f'ADX={adx:.1f}<18→无趋势震荡，江恩建议观望')
    elif adx < 22:
        conf_mod *= 0.90

    # 纪律1: 永不逆市
    if score_bull > score_bear and ma5 < ma20 < ma60 and cp > 0:
        conf_mod *= 0.60; conf_reasons.append('[纪律]逆大势做多→风险')
    if score_bear > score_bull and ma5 > ma20 > ma60 and cp > 0:
        conf_mod *= 0.60; conf_reasons.append('[纪律]逆大势做空→风险')

    # 纪律16: 不过度交易
    if adx < 20 and abs(score_bull - score_bear) < 0.10:
        conf_mod *= 0.65; conf_reasons.append('[纪律]趋势不明→减少交易')

    # ════════════════════════════════════════════════════════════════
    # SECTION F — Conflict detection (MUST precede double confirmation)
    # ════════════════════════════════════════════════════════════════
    if score_bull > 0.40 and score_bear > 0.40:
        final_score = 0.0; confidence = 0.10; direction = 'neutral'
        reasons_final = ['多空共振→观望']
    else:
        # Net score from compressed polarities
        raw_final = np.clip(score_bull - score_bear, -1.0, 1.0)

        # ── Derive has_price_signal dynamically ──
        #   Time signals are guaranteed to contribute ZERO to score_bull/score_bear
        #   (enforced in gann_advanced_features.py and gann_final_features.py).
        #   Therefore: any positive polarity score is, by construction, a price signal.
        has_price_signal = (score_bull > 0) or (score_bear > 0)

        # ── Double Confirmation multiplier ──
        multiplier = 1.50 if (has_price_signal and has_time_signal) else \
                     (0.70 if has_price_signal else 0.0)

        if has_price_signal and has_time_signal:
            conf_reasons.append('江恩双重确认(价+时)→高置信')
        elif has_price_signal and not has_time_signal:
            conf_reasons.append('仅有价格信号→降置信')
        elif not has_price_signal and has_time_signal:
            conf_reasons.append('仅有时间信号→无交易')

        final_score = raw_final * multiplier * conf_mod
        confidence = min(abs(final_score) * 1.20, 0.95)
        direction = 'bullish' if final_score > 0.06 else \
                    ('bearish' if final_score < -0.06 else 'neutral')
        reasons_final = (reasons_bull if final_score > 0 else
                         (reasons_bear if final_score < 0 else
                          reasons_bull + reasons_bear))[:6]
        reasons_final.extend(conf_reasons[:3])

    return {
        'direction': direction,
        'score': round(float(final_score), 3),
        'confidence': round(float(confidence), 3),
        'reasons': reasons_final,
    }


def _compute_school_wyckoff(indicators: Dict) -> Dict:
    """威科夫量价学派 — 极性分组 + 非线性压缩 + 多空共振拦截 + Coil放大器

    Architecture (aligned with Busch / Gann topology):
      1. Polarity-grouped accumulation: ALL original Wyckoff signals
         (OBV, Smart Money, Effort/Result, Phase, Spring, UTAD, SOS, SOW,
          LPS, VSA composites, VR, ADX, BEGE) → score_bull / score_bear
      2. Advanced Wyckoff features via compute_wyckoff_advanced_score_split
         → score_bull/bear += sb/sbe, is_coil_active extracted
      3. Nonlinear squashing: 1.0 - exp(-x * 1.5) on both polarities
      4. Resonance interception: if bull > 0.40 AND bear > 0.40 → final_score = 0
      5. Net score: raw_final = clip(score_bull - score_bear, -1.0, 1.0)
      6. Coil amplifier: multiplier = 1.30 if is_coil_active else 1.0
      7. final_score = raw_final * multiplier
    """
    import numpy as np
    score_bull = 0.0; score_bear = 0.0
    reasons_bull = []; reasons_bear = []
    def _add_bull(w, label):
        nonlocal score_bull; score_bull += w; reasons_bull.append(label)
    def _add_bear(w, label):
        nonlocal score_bear; score_bear += w; reasons_bear.append(label)

    cp = _safe(indicators.get('current_price'), 0)
    atr_pct = _safe(indicators.get('vol_atr_pct'), 0.02)
    vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)

    # ════════════════════════════════════════════════════════════════
    # SECTION A — Law 1: Supply & Demand (OBV + Smart Money)
    # ════════════════════════════════════════════════════════════════
    obv_div = str(indicators.get('obv_divergence', ''))
    obv_trend = str(indicators.get('obv_trend', ''))
    sm_sig = str(indicators.get('sm_smart_signal', ''))

    if '底背离' in obv_div:
        _add_bull(0.12, 'OBV底背离→需求大于供应')
    elif '顶背离' in obv_div:
        _add_bear(0.12, 'OBV顶背离→供应大于需求')

    if obv_trend == 'rising':
        _add_bull(0.06, 'OBV上升→买方主导')
    elif obv_trend == 'falling':
        _add_bear(0.06, 'OBV下降→卖方主导')

    if sm_sig == 'strong_accumulation':
        _add_bull(0.14, '主力强烈收集→需求持续入场')
    elif sm_sig == 'strong_distribution':
        _add_bear(0.14, '主力强烈派发→供应持续入场')
    elif sm_sig == 'accumulation':
        _add_bull(0.07, '主力吸筹')
    elif sm_sig == 'distribution':
        _add_bear(0.07, '主力派发')

    # ════════════════════════════════════════════════════════════════
    # SECTION B — Law 2: Cause & Effect (trading range → trend size)
    # ════════════════════════════════════════════════════════════════
    price_range_5d = _safe(indicators.get('price_range_5d_pct'), 0.03)
    mp_ret_20d = _safe(indicators.get('mp_ret_20d'), 0)

    if price_range_5d < 0.03 and vol_ratio > 1.5:
        _add_bull(0.08, '窄幅+放量→因果蓄势突破')
    elif price_range_5d < 0.02 and abs(mp_ret_20d) < 0.03:
        _add_bull(0.05, '极致窄幅盘整→大因酝酿大果')

    # ════════════════════════════════════════════════════════════════
    # SECTION C — Law 3: Effort vs Result (the heart of Wyckoff/VSA)
    # ════════════════════════════════════════════════════════════════
    vol_p_res = indicators.get('vol_price_resonance', False)
    vol_stack = indicators.get('vol_stacking', False)
    high_vol_stag = indicators.get('high_vol_stagnation', False)
    low_vol_pull = indicators.get('low_vol_pullback', False)

    if high_vol_stag:
        _add_bear(0.18, '放量滞涨→努力大结果小→严重派发')
    if low_vol_pull:
        _add_bull(0.14, '缩量回踩→努力小结果小→供应枯竭洗盘')
    if vol_stack:
        _add_bull(0.12, '堆量→努力持续→主力吸筹')
    if vol_p_res and vol_ratio > 1.5:
        _add_bull(0.10, f'放量共振(量比{vol_ratio:.1f})→努力=结果→健康')
    elif vol_p_res and vol_ratio < 0.7:
        _add_bull(0.06, '缩量共振→供应/需求衰减→变盘前兆')

    # ════════════════════════════════════════════════════════════════
    # SECTION D — Market Phase (Accumulation / Markup / Distribution / Markdown)
    # ════════════════════════════════════════════════════════════════
    mp_phase = str(indicators.get('mp_phase', ''))
    mp_conf = _safe(indicators.get('mp_confidence'), 0)

    if mp_phase == '拉升' and mp_conf > 0.5:
        _add_bull(0.10, '拉升阶段(Markup)→顺势做多')
    elif mp_phase == '筑底' and mp_conf > 0.5:
        _add_bull(0.08, '筑底阶段(Accumulation)→关注Spring/SOS')
    elif mp_phase == '下跌' and mp_conf > 0.5:
        _add_bear(0.12, '下跌阶段(Markdown)→顺势做空')
    elif mp_phase == '盘头' and mp_conf > 0.5:
        _add_bear(0.10, '盘头阶段(Distribution)→关注UTAD/SOW')

    # ════════════════════════════════════════════════════════════════
    # SECTION E — Spring (弹簧 / bear-trap washout) → BULLISH
    # ════════════════════════════════════════════════════════════════
    pat_bot_n = _safe(indicators.get('pattern_bottom_fractal_n'), 0)
    pat_top_n = _safe(indicators.get('pattern_top_fractal_n'), 0)
    rsi = _safe(indicators.get('rsi'), 50)
    cci_s = _safe(indicators.get('cci_short'), 0)

    spring_score = 0
    if cp > 0 and _safe(indicators.get('bb_lower'), 0) > 0:
        bb_l = _safe(indicators.get('bb_lower'))
        bb_u2 = _safe(indicators.get('bb_upper'))
        bb_range = bb_u2 - bb_l
        bb_pos = (cp - bb_l) / bb_range if bb_range > 0 else 0.5
        if bb_pos < 0.2:
            spring_score += 1
    if pat_bot_n >= 1:
        spring_score += 1
    if vol_ratio > 1.3:
        spring_score += 1
    if rsi < 40:
        spring_score += 1
    if cci_s < -100:
        spring_score += 1

    if spring_score >= 4:
        _add_bull(0.20, 'Spring(弹簧)信号→假跌破震仓→最佳买点')
    elif spring_score >= 3:
        _add_bull(0.10, 'Spring雏形→关注次阳确认(#T+1)')
    elif spring_score == 2:
        _add_bull(0.04, 'Spring弱信号')

    # ════════════════════════════════════════════════════════════════
    # SECTION F — UTAD (Upthrust After Distribution) → BEARISH
    # ════════════════════════════════════════════════════════════════
    utad_score = 0
    if cp > 0 and _safe(indicators.get('bb_upper'), 0) > 0:
        bb_u3 = _safe(indicators.get('bb_upper'))
        bb_l3 = _safe(indicators.get('bb_lower'))
        bb_range3 = bb_u3 - bb_l3
        bb_pos3 = (cp - bb_l3) / bb_range3 if bb_range3 > 0 else 0.5
        if bb_pos3 > 0.8:
            utad_score += 1
    if pat_top_n >= 1:
        utad_score += 1
    if vol_ratio > 1.3:
        utad_score += 1
    if rsi > 60:
        utad_score += 1
    if indicators.get('pattern_evening_star_n', 0) > 0:
        utad_score += 1

    if utad_score >= 4:
        _add_bear(0.20, 'UTAD(上冲失败)→假突破派发→最佳卖点')
    elif utad_score >= 3:
        _add_bear(0.10, 'UTAD雏形→警惕高位派发')

    # ════════════════════════════════════════════════════════════════
    # SECTION G — SOS / SOW (Sign of Strength / Weakness)
    # ════════════════════════════════════════════════════════════════
    ma20 = _safe(indicators.get('ma20'), 0)
    # SOS: volume-backed breakout above TR
    if mp_phase == '拉升' and vol_ratio > 1.5 and cp > 0 and ma20 > 0:
        if cp > ma20 * 1.02:
            _add_bull(0.12, 'SOS(强势信号)→放量突破确认')
        elif cp > ma20:
            _add_bull(0.06, 'SOS雏形→价格站上MA20')
    # SOW: volume-backed breakdown below TR
    if mp_phase == '下跌' and vol_ratio > 1.3 and cp > 0 and ma20 > 0:
        if cp < ma20 * 0.98:
            _add_bear(0.12, 'SOW(弱势信号)→放量跌破确认')

    # ════════════════════════════════════════════════════════════════
    # SECTION H — LPS / LPSY (Last Point of Support / Supply)
    # ════════════════════════════════════════════════════════════════
    ver_sup = indicators.get('verified_support', [])
    ver_res = indicators.get('verified_resistance', [])
    if ver_sup and mp_phase == '拉升':
        _add_bull(0.06, f'LPS确认→回踩支撑不破:{ver_sup[:1]}')
    if ver_res and mp_phase == '下跌':
        _add_bear(0.06, f'LPSY确认→反弹阻力有效:{ver_res[:1]}')

    # ════════════════════════════════════════════════════════════════
    # SECTION I — VSA composite patterns
    # ════════════════════════════════════════════════════════════════
    mom_state_s = str(indicators.get('momentum_state_short', ''))

    if vol_stack and '超卖' in mom_state_s:
        _add_bull(0.10, '低位堆量→经典吸筹特征')
    if high_vol_stag and '超买' in mom_state_s:
        _add_bear(0.10, '高位放量滞涨→经典派发特征')

    # ════════════════════════════════════════════════════════════════
    # SECTION J — ROC momentum confirmation
    # ════════════════════════════════════════════════════════════════
    roc_s = _safe(indicators.get('roc_short'), 0)
    roc_l = _safe(indicators.get('roc_long'), 0)
    if roc_s > 0 and roc_l > 0:
        _add_bull(0.03, 'ROC双周期正向')
    elif roc_s < 0 and roc_l < 0:
        _add_bear(0.03, 'ROC双周期负向')

    # ════════════════════════════════════════════════════════════════
    # SECTION K — VR (Volume Ratio) validation
    # ════════════════════════════════════════════════════════════════
    vr_val = _safe(indicators.get('vr_value'), 100)
    vr_sig = str(indicators.get('vr_signal', ''))
    if vr_sig == 'bearish' and vr_val > 300:
        _add_bear(0.08, f'VR超买({vr_val:.0f})→派发区量能异常')
    elif vr_sig == 'contrarian_bullish':
        _add_bull(0.08, f'VR底部({vr_val:.0f})→吸筹区量能枯竭')

    # ════════════════════════════════════════════════════════════════
    # SECTION L — ADX trend strength confirmation
    # ════════════════════════════════════════════════════════════════
    adx = _safe(indicators.get('dmi_adx'), 20)
    di_dir = str(indicators.get('dmi_di_direction', ''))
    if adx > 25:
        if '多' in di_dir:
            _add_bull(0.06, f'ADX={adx:.1f}>25趋势向上→Markup验证')
        elif '空' in di_dir:
            _add_bear(0.06, f'ADX={adx:.1f}>25趋势向下→Markdown验证')

    # ════════════════════════════════════════════════════════════════
    # SECTION M — BEGE macro environment
    # ════════════════════════════════════════════════════════════════
    bebe_reg = str(indicators.get('bebe_regime', ''))
    if bebe_reg == 'good_environment':
        _add_bull(0.04, 'BEGE有利环境')
    elif bebe_reg == 'bad_environment':
        _add_bear(0.06, 'BEGE不利环境')

    # ════════════════════════════════════════════════════════════════
    # SECTION N — 11 Advanced Wyckoff/VSA features
    #   Contract: M3 Shakeout → score_bull (bear-trap)
    #             M9 Coil → is_coil_active (direction-neutral amplifier)
    # ════════════════════════════════════════════════════════════════
    is_coil_active = False
    try:
        from wyckoff_advanced_features import compute_wyckoff_advanced_score_split
        sb, sbe, rb, rbe, coil_flag = compute_wyckoff_advanced_score_split(indicators)
        score_bull += sb; score_bear += sbe
        reasons_bull.extend(rb); reasons_bear.extend(rbe)
        is_coil_active = coil_flag
    except Exception:
        pass

    # ════════════════════════════════════════════════════════════════
    # SECTION O — Nonlinear squashing (both polarities)
    # ════════════════════════════════════════════════════════════════
    score_bull = 1.0 - np.exp(-score_bull * 1.5)
    score_bear = 1.0 - np.exp(-score_bear * 1.5)

    # ════════════════════════════════════════════════════════════════
    # SECTION P — Confidence modifier stack (volatility / ADX / discipline)
    # ════════════════════════════════════════════════════════════════
    conf_mod = 1.0
    conf_reasons = []

    # Volatility regime
    rv_lvl = str(indicators.get('rv_level', ''))
    if '极高' in rv_lvl:
        conf_mod *= 0.55; conf_reasons.append('极高波动→Spring/UTAD易失真→避险')
    elif '高' in rv_lvl:
        conf_mod *= 0.75; conf_reasons.append('高波动→谨慎')

    # ADX no-trend penalty
    if adx < 18:
        conf_mod *= 0.70; conf_reasons.append(f'ADX={adx:.1f}<18→无趋势TR区间→等待方向')

    # Discipline: going against primary trend
    ma5 = _safe(indicators.get('ma5'), 0)
    ma60 = _safe(indicators.get('ma60'), 0)
    if score_bull > score_bear and ma5 < ma60 and cp > 0:
        conf_mod *= 0.65; conf_reasons.append('[纪律]逆大势做多(MA5<MA60)→Spring可能是下跌中继')
    if score_bear > score_bull and ma5 > ma60 and cp > 0:
        conf_mod *= 0.65; conf_reasons.append('[纪律]逆大势做空(MA5>MA60)→UTAD可能是上涨中继')

    # ════════════════════════════════════════════════════════════════
    # SECTION Q — Conflict detection (resonance interception)
    # ════════════════════════════════════════════════════════════════
    if score_bull > 0.40 and score_bear > 0.40:
        final_score = 0.0; confidence = 0.10; direction = 'neutral'
        reasons_final = ['多空共振→观望']
    else:
        raw_final = np.clip(score_bull - score_bear, -1.0, 1.0)

        # ── Coil amplifier ──
        # When the Wyckoff Coil is active, the trading range is compressed
        # to an extreme. Any directional signal emerging from this compression
        # is amplified — the breakout, when it comes, will be explosive.
        coil_multiplier = 1.30 if is_coil_active else 1.0
        if is_coil_active:
            conf_reasons.append('Coil压缩→变盘临界点→信号放大1.30x')

        final_score = raw_final * coil_multiplier * conf_mod
        confidence = min(abs(final_score) * 1.20, 0.95)
        direction = 'bullish' if final_score > 0.06 else \
                    ('bearish' if final_score < -0.06 else 'neutral')
        reasons_final = (reasons_bull if final_score > 0 else
                         (reasons_bear if final_score < 0 else
                          reasons_bull + reasons_bear))[:6]
        reasons_final.extend(conf_reasons[:3])

    return {
        'direction': direction,
        'score': round(float(final_score), 3),
        'confidence': round(float(confidence), 3),
        'reasons': reasons_final,
    }


def _compute_school_dow(indicators: Dict) -> Dict:
    """道氏理论：三重运动+牛熊三阶段+相互验证+成交量确认 → 综合判断"""
    score = 0.0; reasons = []
    cp = _safe(indicators.get('current_price'), 0)
    ma5, ma20, ma60, ma120 = _safe(indicators.get('ma5'),0), _safe(indicators.get('ma20'),0), _safe(indicators.get('ma60'),0), _safe(indicators.get('ma120'),0)

    # 主要趋势判定
    mp_ret_20d = _safe(indicators.get('mp_ret_20d'), 0)
    mp_ret_60d = _safe(indicators.get('mp_ret_60d'), 0)
    if ma20 > ma60 > ma120 and cp > 0:
        score += 0.20; reasons.append('主要上升趋势(年级别)')
    elif ma20 < ma60 < ma120 and cp > 0:
        score -= 0.20; reasons.append('主要下降趋势(年级别)')
    if mp_ret_60d > 0.05: score += 0.06
    elif mp_ret_60d < -0.05: score -= 0.06

    # 次级折返判定 — 33-66%法则
    adx = _safe(indicators.get('dmi_adx'), 20)
    if 0.15 < abs(mp_ret_20d) < 0.30 and abs(mp_ret_60d) > 0.05:
        if mp_ret_60d > 0 and mp_ret_20d < 0: score += 0.08; reasons.append('次级折返(回调33-66%)→买入机会')
        elif mp_ret_60d < 0 and mp_ret_20d > 0: score -= 0.06; reasons.append('次级反弹→卖出机会')

    # 成交量确认 — 道氏经典规则
    vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)
    if indicators.get('vol_price_resonance', False) and vol_ratio > 1.3:
        if score > 0: score += 0.08; reasons.append('放量确认上涨→趋势健康')
        else: score -= 0.06; reasons.append('放量确认下跌→趋势延续')

    # 牛熊三阶段匹配
    mp_phase = str(indicators.get('mp_phase', '')); mp_conf = _safe(indicators.get('mp_confidence'), 0)
    rsi = _safe(indicators.get('rsi'), 50)
    if mp_phase == '拉升' and mp_conf > 0.5: score += 0.08
    elif mp_phase == '筑底' and mp_conf > 0.5: score += 0.06; reasons.append('筑底→道氏怀疑阶段')
    elif mp_phase == '下跌' and mp_conf > 0.5: score -= 0.10
    elif mp_phase == '盘头' and mp_conf > 0.5: score -= 0.08; reasons.append('盘头→道氏狂热阶段')

    # RSI辅助: 30-70对应三阶段位置
    if rsi < 30: score += 0.05; reasons.append('RSI<30→道氏绝望阶段')
    elif rsi > 70: score -= 0.04; reasons.append('RSI>70→道氏狂热阶段')

    # 震荡市惩罚
    if adx < 18: score *= 0.65; reasons.append(f'ADX={adx:.1f}<18→趋势不明→道氏建议等待')
    bebe_reg = str(indicators.get('bebe_regime', ''))
    if bebe_reg == 'good_environment': score += 0.03
    elif bebe_reg == 'bad_environment': score -= 0.05

    score = _clip_score(score)
    return {'direction': _direction_from_score(score, 0.08), 'score': round(score, 3), 'confidence': round(min(abs(score)*1.1, 0.95), 3), 'reasons': reasons[:5]}


def _compute_school_canslim(indicators: Dict) -> Dict:
    """欧奈尔CANSLIM：RS强度+量价供需+杯柄形态+大盘方向 → 成长股信号"""
    score = 0.0; reasons = []
    cp = _safe(indicators.get('current_price'), 0)
    rsi = _safe(indicators.get('rsi'), 50)
    roc_s = _safe(indicators.get('roc_short'), 0); roc_l = _safe(indicators.get('roc_long'), 0)
    atr_pct = _safe(indicators.get('vol_atr_pct'), 0.02)

    # L: RS相对强度 (用ROC/动量模拟)
    if roc_l > 0.05: score += 0.12; reasons.append('RS强度高(年度涨幅>5%)→领导股')
    elif roc_l > 0.02: score += 0.06
    elif roc_l < -0.05: score -= 0.08; reasons.append('RS强度低→落后股→避开')

    # S: 供需 — 量价共振
    vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)
    if indicators.get('vol_stacking', False): score += 0.10; reasons.append('堆量→需求旺盛')
    if indicators.get('vol_price_resonance', False) and vol_ratio > 1.5: score += 0.08
    if indicators.get('high_vol_stagnation', False): score -= 0.10; reasons.append('放量滞涨→供应过剩')

    # N: 新产品/新高 — 近250日新高模拟
    bb_u = _safe(indicators.get('bb_upper'), 0); bb_l = _safe(indicators.get('bb_lower'), 0)
    if cp > 0 and bb_u > 0 and cp >= bb_u * 0.98: score += 0.08; reasons.append('接近新高→N信号')
    mp_price_pos = _safe(indicators.get('mp_price_position'), 0.5)
    if mp_price_pos > 0.80: score += 0.06

    # M: 大盘方向 — 最重要的过滤器
    mp_phase = str(indicators.get('mp_phase', '')); mp_conf = _safe(indicators.get('mp_confidence'), 0)
    if mp_phase == '拉升' and mp_conf > 0.5: score += 0.10
    elif mp_phase == '下跌' and mp_conf > 0.5: score -= 0.15; reasons.append('大盘下跌→CANSLIM禁止买入')
    elif mp_phase == '盘头' and mp_conf > 0.5: score *= 0.60

    # I: 机构持仓 — 用主力资金检测代替
    sm_sig = str(indicators.get('sm_smart_signal', ''))
    if 'accumulation' in sm_sig: score += 0.08; reasons.append('主力收集→机构增持')

    # C+A: 动量确认 (模拟EPS增速)
    if roc_s > 0.03 and roc_l > 0.05: score += 0.10; reasons.append('短+长期动量加速→盈利增长特征')
    elif roc_s < -0.02 and roc_l < -0.03: score -= 0.08

    # 杯柄形态简化检测 — 近期缩量回踩MA20
    ma20 = _safe(indicators.get('ma20'), 0)
    if cp > 0 and ma20 > 0 and abs(cp/ma20 - 1) < 0.04 and vol_ratio < 0.8: score += 0.06; reasons.append('缩量近MA20→杯柄特征')

    if atr_pct > 0.05: score *= 0.75
    score = _clip_score(score)
    return {'direction': _direction_from_score(score, 0.08), 'score': round(score, 3), 'confidence': round(min(abs(score)*1.1, 0.95), 3), 'reasons': reasons[:5]}


def _compute_school_turtle(indicators: Dict) -> Dict:
    """海龟交易法：唐奇安通道+ATR仓位+突破入市+2N止损 → 机械趋势跟随"""
    score = 0.0; reasons = []
    cp = _safe(indicators.get('current_price'), 0)
    atr14 = _safe(indicators.get('vol_atr14'), 0); atr_pct = _safe(indicators.get('vol_atr_pct'), 0.02)
    ma5 = _safe(indicators.get('ma5'), 0); ma20 = _safe(indicators.get('ma20'), 0); ma60 = _safe(indicators.get('ma60'), 0)
    vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)

    # 唐奇安通道模拟 — 用布林带+MA判断趋势突破
    bb_u = _safe(indicators.get('bb_upper'), 0); bb_l = _safe(indicators.get('bb_lower'), 0)
    if cp > 0 and bb_u > 0:
        # System1: 20日突破模拟
        if cp >= bb_u * 0.99 and vol_ratio > 1.3: score += 0.15; reasons.append('System1: 20日通道突破+放量→入场')
        elif cp <= bb_l * 1.01 and vol_ratio > 1.3: score -= 0.10; reasons.append('System1: 跌破20日通道→空头')
        # System2: 55日突破模拟（用MA60）
        if cp > 0 and ma60 > 0 and cp > ma60 * 1.05 and vol_ratio > 1.2: score += 0.12; reasons.append('System2: 55日突破→长线做多')

    # ATR仓位管理 — 波动率仓位
    if atr_pct > 0.05: score *= 0.60; reasons.append(f'高ATR({atr_pct:.1%})→单笔风险大→减仓')
    elif atr_pct > 0.035: score *= 0.80

    # 趋势跟随核心 — 均线排列
    if ma5 > ma20 > ma60 and cp > 0: score += 0.12; reasons.append('均线多头→海龟顺势做多')
    elif ma5 < ma20 < ma60 and cp > 0: score -= 0.12

    # 退出信号 — 反向唐奇安通道
    if cp > 0 and ma20 > 0 and cp < ma20 * 0.97: score -= 0.08; reasons.append('跌破MA20→System2退出信号')

    # ADX趋势强度
    adx = _safe(indicators.get('dmi_adx'), 20)
    if adx > 25: score *= 1.1
    elif adx < 18: score *= 0.55; reasons.append(f'ADX={adx:.1f}<18→震荡→海龟不交易')

    score = _clip_score(score)
    return {'direction': _direction_from_score(score, 0.08), 'score': round(score, 3), 'confidence': round(min(abs(score)*1.05, 0.95), 3), 'reasons': reasons[:5]}


def _compute_school_harmonic(indicators: Dict) -> Dict:
    """谐波形态学派 V2.0 — 极性分组 + 非线性压缩 + 共振拦截 + PRZ微观确认 + BAMM动量

    Integrated feature pipeline:
      Part 1 (core):  Gartley / Bat / Butterfly / Crab
      Part 2 (adv):   Deep Crab / Cypher / Shark / PRZ Validation / RSI BAMM

    Topology (strictly aligned with Wyckoff / Busch / Gann):
      1. Polarity-grouped accumulation from both core + advanced split interfaces
      2. Baseline macro filters (ADX, BEGE) → conf_mod stack
      3. Nonlinear squashing: 1.0 - exp(-x * 1.5)
      4. Resonance interception: bull > 0.40 AND bear > 0.40 → final_score = 0
      5. PRZ Micro-Validation multiplier + BAMM multiplier + conf_mod
      6. final_score = raw_final * prz_mult * bamm_mult * conf_mod
    """
    import numpy as np
    score_bull = 0.0; score_bear = 0.0
    reasons_bull = []; reasons_bear = []
    def _add_bull(w, label):
        nonlocal score_bull; score_bull += w; reasons_bull.append(label)
    def _add_bear(w, label):
        nonlocal score_bear; score_bear += w; reasons_bear.append(label)

    cp = _safe(indicators.get('current_price'), 0)

    # ════════════════════════════════════════════════════════════════
    # SECTION A — Core harmonic patterns (Part 1: Gartley/Bat/Butterfly/Crab)
    # ════════════════════════════════════════════════════════════════
    try:
        from harmonic_core_features import compute_harmonic_core_score_split
        sb1, sbe1, rb1, rbe1 = compute_harmonic_core_score_split(indicators)
        score_bull += sb1; score_bear += sbe1
        reasons_bull.extend(rb1); reasons_bear.extend(rbe1)
    except Exception:
        pass

    # ════════════════════════════════════════════════════════════════
    # SECTION B — Advanced harmonic patterns (Part 2: DeepCrab/Cypher/Shark)
    #             + PRZ Micro-Validation + RSI BAMM
    # ════════════════════════════════════════════════════════════════
    prz_conf = 0.0
    bamm_mult = 1.0
    try:
        from harmonic_advanced_features import compute_harmonic_advanced_score_split
        sb2, sbe2, rb2, rbe2, prz_conf, bamm_mult = \
            compute_harmonic_advanced_score_split(indicators)
        score_bull += sb2; score_bear += sbe2
        reasons_bull.extend(rb2); reasons_bear.extend(rbe2)
    except Exception:
        pass

    # ════════════════════════════════════════════════════════════════
    # SECTION C — Baseline macro filters (ADX / BEGE / volatility)
    #             → conf_mod stack (applied LAST in the multiplier chain)
    # ════════════════════════════════════════════════════════════════
    conf_mod = 1.0
    conf_reasons = []

    adx = _safe(indicators.get('dmi_adx'), 20)
    if adx < 18:
        conf_mod *= 0.70
        conf_reasons.append(f'ADX={adx:.1f}<18→低趋势环境，谐波信号降权')

    bebe_reg = str(indicators.get('bebe_regime', ''))
    if bebe_reg == 'bad_environment':
        conf_mod *= 0.75
        conf_reasons.append('BEGE不利环境→谐波降权')

    rv_lvl = str(indicators.get('rv_level', ''))
    if '极高' in rv_lvl:
        conf_mod *= 0.55
        conf_reasons.append('极高波动→谐波D点易穿透→避险')
    elif '高' in rv_lvl:
        conf_mod *= 0.75
        conf_reasons.append('高波动→谐波信号谨慎')

    # ADX trending boost — harmonic works BETTER in ranging markets
    # (already penalized in regime weights; here we slightly favor low ADX)
    if 18 <= adx <= 25:
        conf_mod *= 1.05  # mild boost in the harmonic sweet spot

    # ════════════════════════════════════════════════════════════════
    # SECTION D — Nonlinear squashing (both polarities)
    # ════════════════════════════════════════════════════════════════
    score_bull = 1.0 - np.exp(-score_bull * 1.5)
    score_bear = 1.0 - np.exp(-score_bear * 1.5)

    # ════════════════════════════════════════════════════════════════
    # SECTION E — Resonance interception (core defense)
    # ════════════════════════════════════════════════════════════════
    if score_bull > 0.40 and score_bear > 0.40:
        final_score = 0.0; confidence = 0.10; direction = 'neutral'
        reasons_final = ['多空形态共振→观望']
    else:
        raw_final = np.clip(score_bull - score_bear, -1.0, 1.0)

        # ════════════════════════════════════════════════════════════
        # SECTION F — PRZ Micro-Validation multiplier
        # ════════════════════════════════════════════════════════════
        has_pattern = (score_bull > 0) or (score_bear > 0)

        if has_pattern and prz_conf >= 0.20:
            # K-line confirmation at D point → amplify signal
            prz_multiplier = 1.0 + (prz_conf * 0.40)
            if prz_conf > 0.50:
                conf_reasons.append(f'PRZ微观确认({prz_conf:.0%})→K线反转验证通过')
            else:
                conf_reasons.append(f'PRZ弱确认({prz_conf:.0%})→小幅加成')
        elif has_pattern and prz_conf < 0.20:
            # Price blew through PRZ without stopping → danger signal
            prz_multiplier = 0.50
            conf_reasons.append(f'PRZ穿透({prz_conf:.0%})→D点无反转K线→大幅降权')
        else:
            prz_multiplier = 1.0

        # ════════════════════════════════════════════════════════════
        # SECTION G — BAMM multiplier (from advanced split interface)
        # ════════════════════════════════════════════════════════════
        if bamm_mult > 1.0:
            conf_reasons.append(f'RSI BAMM激活→动量背离确认({bamm_mult:.2f}x)')

        # ════════════════════════════════════════════════════════════
        # SECTION H — Final score assembly
        # ════════════════════════════════════════════════════════════
        final_score = raw_final * prz_multiplier * bamm_mult * conf_mod
        confidence = min(abs(final_score) * 1.20, 0.95)
        direction = 'bullish' if final_score > 0.06 else \
                    ('bearish' if final_score < -0.06 else 'neutral')
        reasons_final = (reasons_bull if final_score > 0 else
                         (reasons_bear if final_score < 0 else
                          reasons_bull + reasons_bear))[:6]
        reasons_final.extend(conf_reasons[:3])

    return {
        'direction': direction,
        'score': round(float(final_score), 3),
        'confidence': round(float(confidence), 3),
        'reasons': reasons_final,
    }


def _compute_school_marketprofile(indicators: Dict) -> Dict:
    """市场轮廓：价值区(VAH/VAL/POC)+成交量分布+拍卖失衡 → 价值区交易"""
    score = 0.0; reasons = []
    cp = _safe(indicators.get('current_price'), 0)
    bb_u = _safe(indicators.get('bb_upper'), 0); bb_m = _safe(indicators.get('bb_mid'), 0); bb_l = _safe(indicators.get('bb_lower'), 0)
    mp_phase = str(indicators.get('mp_phase', '')); mp_conf = _safe(indicators.get('mp_confidence'), 0)
    mp_price_pos = _safe(indicators.get('mp_price_position'), 0.5)

    # 价值区模拟 — BB中轨=POC, 上下轨=VAH/VAL
    if cp > 0 and bb_u > 0:
        bb_range = bb_u - bb_l
        if bb_range > 0:
            bb_pos = (cp - bb_l) / bb_range
            if bb_pos < 0.15: score += 0.10; reasons.append('价格近价值区下沿(VAL)→买入区')
            elif bb_pos > 0.85: score -= 0.08; reasons.append('价格近价值区上沿(VAH)→卖出区')
            elif 0.40 < bb_pos < 0.60: reasons.append('价格在POC附近→公允价值')

    # POC/价值区移动方向
    mp_ret_20d = _safe(indicators.get('mp_ret_20d'), 0)
    if mp_ret_20d > 0.03: score += 0.06; reasons.append('POC上移→价值区上移→趋势偏多')
    elif mp_ret_20d < -0.03: score -= 0.06; reasons.append('POC下移→价值区下移→趋势偏空')

    # 成交量分布 — HVN/LVN（简化为堆量和缩量区域）
    if indicators.get('vol_stacking', False): score += 0.08; reasons.append('HVN形成→高成交量节点支撑')
    if indicators.get('low_vol_pullback', False): score += 0.06; reasons.append('LVN回踩→快速穿越后反弹')

    # 拍卖失衡 — 价格在VA外持续
    adx = _safe(indicators.get('dmi_adx'), 20)
    if cp > 0 and bb_u > 0 and cp > bb_u * 1.03 and adx > 25: score += 0.08; reasons.append('VA上方失衡→方向性机动(趋势日)')
    elif cp > 0 and bb_l > 0 and cp < bb_l * 0.97 and adx > 25: score -= 0.08

    # 市场阶段
    if mp_phase == '拉升' and mp_conf > 0.5: score += 0.05
    elif mp_phase == '下跌' and mp_conf > 0.5: score -= 0.08
    if adx < 18: score *= 0.65; reasons.append(f'ADX={adx:.1f}<18→价值区盘整→区间交易')

    score = _clip_score(score)
    return {'direction': _direction_from_score(score, 0.07), 'score': round(score, 3), 'confidence': round(min(abs(score)*1.0, 0.95), 3), 'reasons': reasons[:5]}


def _compute_school_chaos(indicators: Dict) -> Dict:
    """混沌交易法：分形突破+鳄鱼线+AO动量+AC加速+MFI → 五维共振"""
    score = 0.0; reasons = []
    cp = _safe(indicators.get('current_price'), 0)
    ma5, ma20, ma60 = _safe(indicators.get('ma5'),0), _safe(indicators.get('ma20'),0), _safe(indicators.get('ma60'),0)
    pat_bot_n = _safe(indicators.get('pattern_bottom_fractal_n'), 0)
    pat_top_n = _safe(indicators.get('pattern_top_fractal_n'), 0)
    rsi, rsi_val = _safe(indicators.get('rsi'), 50), _safe(indicators.get('rsi'), 50)
    macd_dif, macd_dea, macd_hist = _safe(indicators.get('macd_dif'),0), _safe(indicators.get('macd_dea'),0), _safe(indicators.get('macd_hist'),0)
    vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)

    resonance = 0  # 五维共振得分

    # 维度1: 分形突破
    if pat_bot_n >= 2: resonance += 1; reasons.append('底分形×2→分形突破做多')
    if pat_top_n >= 2: resonance -= 1; reasons.append('顶分形×2→分形突破做空')
    if cp > 0 and ma5 > ma20 > ma60 and ma20 > 0: resonance += 1  # 分形突破向上

    # 维度2: 鳄鱼线(模拟为MA排列)
    if ma5 > ma20 > ma60 and cp > 0: resonance += 1; reasons.append('鳄鱼张嘴(多)→趋势启动')
    elif ma5 < ma20 < ma60 and cp > 0: resonance -= 1; reasons.append('鳄鱼张嘴(空)→空头趋势')
    else: reasons.append('鳄鱼睡觉→三线缠绕→观望')

    # 维度3: AO动量 (MACD模拟)
    if macd_dif > macd_dea and macd_hist > 0: resonance += 1
    elif macd_dif < macd_dea and macd_hist < 0: resonance -= 1

    # 维度4: AC加速 (ROC二次导数)
    roc_s = _safe(indicators.get('roc_short'), 0)
    if roc_s > 0.02 and macd_hist > 0: resonance += 1
    elif roc_s < -0.02 and macd_hist < 0: resonance -= 1

    # 维度5: MFI市场促进 (量比+ATR)
    atr_pct = _safe(indicators.get('vol_atr_pct'), 0.02)
    atr14 = _safe(indicators.get('vol_atr14'), 0)
    if atr_pct > 0.03 and vol_ratio > 1.3: resonance += 1; reasons.append('绿灯(Squat/Green)→市场促进')
    elif vol_ratio < 0.6: resonance += 0; reasons.append('假波(Fake)→无意义波动')

    # 综合共振
    if resonance >= 4: score += 0.20; reasons.append(f'五维共振({resonance}/5)→超级信号')
    elif resonance >= 3: score += 0.12; reasons.append(f'强共振({resonance}/5)')
    elif resonance >= 2: score += 0.06; reasons.append(f'弱共振({resonance}/5)')
    elif resonance <= -3: score -= 0.15
    elif resonance <= -4: score -= 0.20

    # AC+AO方向确认
    if macd_dif > macd_dea and rsi > 50: score += 0.05
    elif macd_dif < macd_dea and rsi < 50: score -= 0.05

    # ADX
    adx = _safe(indicators.get('dmi_adx'), 20)
    if adx < 18: score *= 0.60; reasons.append(f'ADX={adx:.1f}<18→震荡→鳄鱼睡觉')

    score = _clip_score(score)
    return {'direction': _direction_from_score(score, 0.08), 'score': round(score, 3), 'confidence': round(min(abs(score)*1.1, 0.95), 3), 'reasons': reasons[:5]}


def _compute_school_orderflow(indicators: Dict) -> Dict:
    """Order Flow + VWAP + RSI composite — scoring system using existing indicators.

    5-condition scoring (≥3/5 → signal):
      1. RSI oversold/overbought
      2. Price vs MA20 (VWAP proxy)
      3. Volume surge (vol_ratio > 1.5)
      4. OBV trend (order flow proxy)
      5. Momentum resonance
    """
    score = 0.0
    reasons = []
    conditions_met = 0

    rsi = _safe(indicators.get('rsi'), 50)
    price = _safe(indicators.get('current_price'), 0)
    ma20 = _safe(indicators.get('ma20'), price)
    vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)
    obv_trend = str(indicators.get('obv_trend', ''))
    obv_div = str(indicators.get('obv_divergence', ''))
    mom_short = str(indicators.get('momentum_state_short', ''))
    mom_long = str(indicators.get('momentum_state_long', ''))
    vol_type = str(indicators.get('vol_type', ''))
    vol_stacking = _safe(indicators.get('vol_stacking'), 0)
    atr_pct = _safe(indicators.get('vol_atr_pct'), 0.03)

    # 1. RSI condition
    if rsi < 35:
        conditions_met += 1
        reasons.append(f'RSI={rsi:.0f}(超卖)')
        score += 0.20
    elif rsi > 70:
        conditions_met += 1
        reasons.append(f'RSI={rsi:.0f}(超买)')
        score -= 0.20

    # 2. Price vs MA20 (VWAP proxy — dominant trend filter)
    if price > ma20 * 1.02:
        conditions_met += 1
        reasons.append(f'价格>MA20({price/ma20-1:+.1%})')
        score += 0.18
    elif price < ma20 * 0.98:
        conditions_met += 1
        reasons.append(f'价格<MA20({price/ma20-1:+.1%})')
        score -= 0.18

    # 3. Volume surge (relative volume > 1.5x normal)
    if vol_ratio > 1.5:
        conditions_met += 1
        reasons.append(f'放量{vol_ratio:.1f}x')
        score += 0.12 if score >= 0 else -0.12
    if 'high_vol' in vol_type or vol_stacking > 2:
        reasons.append(f'量能堆积({vol_stacking}天)')
        score += 0.05 if score >= 0 else -0.05

    # 4. OBV trend (order flow proxy — CVD equivalent)
    if 'rising' in obv_trend:
        conditions_met += 1
        reasons.append('OBV上升(资金流入)')
        score += 0.20
    elif 'falling' in obv_trend:
        conditions_met += 1
        reasons.append('OBV下降(资金流出)')
        score -= 0.20
    if 'bullish' in obv_div:
        reasons.append('OBV底背离')
        score += 0.15
    elif 'bearish' in obv_div:
        reasons.append('OBV顶背离')
        score -= 0.15

    # 5. Momentum resonance
    if 'bullish' in mom_short and 'bullish' in mom_long:
        conditions_met += 1
        reasons.append('多周期动量共振多')
        score += 0.18
    elif 'bearish' in mom_short and 'bearish' in mom_long:
        conditions_met += 1
        reasons.append('多周期动量共振空')
        score -= 0.18

    # Signal threshold: ≥3/5 conditions for a directional signal
    if conditions_met < 3:
        direction = 'neutral'
        score = 0.0
    elif score > 0.05:
        direction = 'bullish'
    elif score < -0.05:
        direction = 'bearish'
    else:
        direction = 'neutral'
        score = 0.0

    # Confidence: conditions_met / 5
    confidence = min(0.95, conditions_met / 5.0 + abs(score) * 0.3)

    return {
        'direction': direction,
        'score': round(score, 3),
        'confidence': round(confidence, 3),
        'reasons': reasons[:6],
    }


def _compute_school_roc_breakout(indicators: Dict) -> Dict:
    """ROC Momentum Breakout school: ROC+RSI+ATR+RVOL composite.

    Uses existing indicators (roc_short, roc_long, rsi, vol_ratio, vol_atr_pct, ma20).
    5-condition scoring, >=3/5 -> signal.
    """
    score = 0.0
    reasons = []
    conditions_met = 0

    roc_short = _safe(indicators.get('roc_short'), 0)
    roc_long = _safe(indicators.get('roc_long'), 0)
    rsi = _safe(indicators.get('rsi'), 50)
    vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)
    atr_pct = _safe(indicators.get('vol_atr_pct'), 0.03)
    price = _safe(indicators.get('current_price'), 0)
    ma20 = _safe(indicators.get('ma20'), price)
    mom_short = str(indicators.get('momentum_state_short', ''))

    # 1. ROC direction
    if roc_long > 2:
        conditions_met += 1; reasons.append(f'ROC12={roc_long:.1f}%↑'); score += 0.22
    elif roc_long < -2:
        conditions_met += 1; reasons.append(f'ROC12={roc_long:.1f}%↓'); score -= 0.22

    # 2. RSI context
    if rsi < 35:
        conditions_met += 1; reasons.append(f'RSI={rsi:.0f}(超卖)'); score += 0.18
    elif rsi > 65:
        conditions_met += 1; reasons.append(f'RSI={rsi:.0f}(超买)'); score -= 0.18

    # 3. Volume confirmation
    if vol_ratio > 1.3:
        conditions_met += 1; reasons.append(f'放量{vol_ratio:.1f}x')
        score += 0.15 if score > 0 else -0.15

    # 4. ATR expansion (breakout validity)
    if atr_pct > 0.025:
        conditions_met += 1; reasons.append(f'ATR={atr_pct:.1%}(高波动)')
        score += 0.12 if score > 0 else -0.12

    # 5. Trend filter
    if ma20 > 0 and price > ma20 * 1.01:
        conditions_met += 1; reasons.append(f'>MA20(+{price/ma20-1:.1%})'); score += 0.18
    elif ma20 > 0 and price < ma20 * 0.99:
        conditions_met += 1; reasons.append(f'<MA20({price/ma20-1:.1%})'); score -= 0.18

    # ROC acceleration bonus
    if roc_short > roc_long and roc_short > 0:
        reasons.append('ROC加速↑'); score += 0.08

    if conditions_met < 3:
        return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': reasons[:5]}

    direction = 'bullish' if score > 0.03 else ('bearish' if score < -0.03 else 'neutral')
    confidence = min(0.95, conditions_met / 5.0 + abs(score) * 0.3)

    return {
        'direction': direction,
        'score': round(score, 3),
        'confidence': round(confidence, 3),
        'reasons': reasons[:6],
    }


def _compute_school_volume_profile(indicators: Dict) -> Dict:
    """Volume Profile school: POC+VAH+VAL+HVN/LVN based signals.

    Uses existing indicators (current_price, vol_ratio, ma20, rsi, momentum).
    Simulates volume profile logic via price-position scoring.
    """
    score = 0.0; reasons = []; met = 0
    price = _safe(indicators.get('current_price'), 0)
    ma20 = _safe(indicators.get('ma20'), price)
    vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)
    rsi = _safe(indicators.get('rsi'), 50)
    mom_short = str(indicators.get('momentum_state_short', ''))
    atr_pct = _safe(indicators.get('vol_atr_pct'), 0.02)

    # 1. Price vs MA20 (trend = VA position proxy)
    if price > ma20 * 1.02:
        met += 1; reasons.append(f'>MA20(VA突破)'); score += 0.20
    elif price < ma20 * 0.98:
        met += 1; reasons.append(f'<MA20(VA跌破)'); score -= 0.20

    # 2. Volume confirmation (VA breakout needs volume)
    if vol_ratio > 1.2:
        met += 1; reasons.append(f'量{vol_ratio:.1f}x'); score += 0.15 if score>0 else -0.15

    # 3. RSI context
    if rsi < 40: met += 1; reasons.append(f'RSI={rsi:.0f}(VAL反弹)'); score += 0.15
    elif rsi > 60: met += 1; reasons.append(f'RSI={rsi:.0f}(VAH遇阻)'); score -= 0.15

    # 4. Momentum
    if 'bullish' in mom_short: met += 1; reasons.append('动量↑'); score += 0.17
    elif 'bearish' in mom_short: met += 1; reasons.append('动量↓'); score -= 0.17

    # 5. ATR context (high vol = wider VA = stronger signal)
    if atr_pct > 0.025: met += 1; reasons.append(f'ATR={atr_pct:.1%}'); score += 0.10 if score>0 else -0.10

    if met < 3: return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': reasons[:5]}
    direction = 'bullish' if score > 0.03 else ('bearish' if score < -0.03 else 'neutral')
    return {'direction': direction, 'score': round(score,3), 'confidence': round(min(0.95, met/5+abs(score)*0.3),3), 'reasons': reasons[:6]}


def _compute_school_fusion(indicators: Dict) -> Dict:
    """蒸馏融合策略: ADX+FRVP+缠论分型+Delta订单流+ROC动量。使用现有指标近似。"""
    score = 0.0; reasons = []; met = 0
    rsi = _safe(indicators.get('rsi'), 50)
    adx = _safe(indicators.get('dmi_adx'), 20)
    pdi = _safe(indicators.get('dmi_pdi'), 15)
    mdi = _safe(indicators.get('dmi_mdi'), 15)
    price = _safe(indicators.get('current_price'), 0)
    ma20 = _safe(indicators.get('ma20'), price)
    vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)
    obv_trend = str(indicators.get('obv_trend', ''))
    roc = _safe(indicators.get('roc_short'), 0)
    macd_hist = _safe(indicators.get('macd_hist'), 0)
    bb_upper = _safe(indicators.get('bb_upper'), price * 1.1)
    mom_short = str(indicators.get('momentum_state_short', ''))
    pattern_bottom = _safe(indicators.get('pattern_bottom_fractal_n'), 0)
    regime = str(indicators.get('mp_phase', ''))

    # 1. Trend: price > MA20 or ADX uptrend (pdi > mdi)
    if price > ma20 or (adx > 22 and pdi > mdi * 1.1):
        met += 1; reasons.append('趋势向上'); score += 0.18

    # 2. Chanlun structure: bottom fractal or bullish momentum
    if pattern_bottom > 0 or mom_short == 'bullish':
        met += 1; reasons.append('缠论支撑'); score += 0.16

    # 3. Volume: RVOL > 1.3 or OBV rising
    if vol_ratio > 1.3 or 'rising' in obv_trend:
        met += 1; reasons.append(f'量能确认({vol_ratio:.1f}x)'); score += 0.15

    # 4. Momentum: RSI 30-65 AND ROC > -2
    if 30 < rsi < 65 and roc > -2:
        met += 1; reasons.append(f'RSI={rsi:.0f}/ROC↑'); score += 0.14

    # 5. MACD: histogram positive
    if macd_hist > 0:
        met += 1; reasons.append('MACD>0'); score += 0.15

    # 6. Regime filter: no downtrend
    if regime not in ('downtrend', 'markdown') and 'bearish' not in mom_short:
        met += 1; reasons.append('非下跌势'); score += 0.12

    if met < 4 or score < 0:
        return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': reasons[:5]}

    direction = 'bullish' if score > 0.05 else 'neutral'
    conf = min(0.95, met / 6.0 + abs(score) * 0.3)
    return {'direction': direction, 'score': round(score, 3), 'confidence': round(conf, 3), 'reasons': reasons[:6]}


def _compute_school_mean_reversion(indicators: Dict) -> Dict:
    """均值回归(震荡市): 布林下轨+RSI超卖+RIVOL放量 → 反弹至中轨止盈。专攻ADX<22震荡市。"""
    score = 0.0; reasons = []; met = 0
    price = _safe(indicators.get('current_price'), 0)
    bb_lower = _safe(indicators.get('bb_lower'), price * 0.90)
    bb_mid = _safe(indicators.get('bb_mid'), price)
    bb_upper = _safe(indicators.get('bb_upper'), price * 1.10)
    rsi = _safe(indicators.get('rsi'), 50)
    adx = _safe(indicators.get('dmi_adx'), 20)
    vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)
    cci = _safe(indicators.get('cci_short'), 0)
    atr_pct = _safe(indicators.get('vol_atr_pct'), 0.02)
    support = _safe(indicators.get('dynamic_support'), bb_lower)

    # 1. Oscillating regime: ADX < 22 (ranging market)
    is_ranging = adx < 22
    if is_ranging:
        met += 1; reasons.append(f'震荡市(ADX={adx:.0f})'); score += 0.22

    # 2. Near Bollinger lower band (oversold)
    near_lower = price <= bb_lower * 1.03
    near_bb_support = price <= support * 1.02
    if near_lower or near_bb_support:
        met += 1
        reasons.append(f'超卖区(距下轨{price/bb_lower-1:.1%})' if near_lower else '触及支撑')
        score += 0.20

    # 3. RSI oversold or recovering
    if rsi < 35:
        met += 1; reasons.append(f'RSI超卖({rsi:.0f})'); score += 0.18
    elif rsi < 45 and cci > -100:
        met += 1; reasons.append(f'RSI低位回升({rsi:.0f})'); score += 0.12

    # 4. Volume confirmation (capitulation or accumulation)
    if vol_ratio > 1.3:
        met += 1; reasons.append(f'放量({vol_ratio:.1f}x)'); score += 0.15

    # 5. Bottom fractal or candle pattern
    pattern_bottom = _safe(indicators.get('pattern_bottom_fractal_n'), 0)
    morning_star = _safe(indicators.get('pattern_morning_star_n'), 0)
    if pattern_bottom > 0 or morning_star > 0:
        met += 1; reasons.append('底分型/晨星'); score += 0.15

    # 6. Not in strong downtrend (price > MA60 or ADX not in bear trend)
    ma60 = _safe(indicators.get('ma60'), price)
    pdi = _safe(indicators.get('dmi_pdi'), 15)
    mdi = _safe(indicators.get('dmi_mdi'), 15)
    if price > ma60 or pdi >= mdi * 0.8:
        met += 1; reasons.append('非强下跌'); score += 0.10

    if met < 3 or score < 0:
        return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': reasons[:5]}

    direction = 'bullish' if score > 0.05 else 'neutral'
    conf = min(0.85, met / 6.0 + abs(score) * 0.3)
    return {'direction': direction, 'score': round(score, 3), 'confidence': round(conf, 3), 'reasons': reasons[:6]}


def _compute_school_capital_flow(indicators: Dict) -> Dict:
    """资金流向学派: 5维打分 — 北向+主力+大单+融资+共振。回测中无历史资金数据时返回中性。"""
    # In backtest, we don't have historical capital flow data — return neutral
    # In live trading, get_capital_flow_signal() provides real data
    try:
        from capital_flow import get_capital_flow_signal
        symbol = str(indicators.get('symbol', ''))
        if symbol:
            cf = get_capital_flow_signal(symbol)
            if cf['direction'] == 'bullish':
                return {'direction': 'bullish', 'score': cf['score'],
                        'confidence': cf['confidence'], 'reasons': cf['reasons']}
            elif cf['direction'] == 'bearish':
                return {'direction': 'bearish', 'score': -cf['score'],
                        'confidence': cf['confidence'], 'reasons': cf['reasons']}
    except Exception:
        pass
    # Fallback: neutral (no historical data in backtest)
    return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
            'reasons': ['资金数据不可用(回测模式)']}


def _compute_school_pattern_features(indicators: Dict) -> Dict:
    """经典形态特征因子库: 连续形态因子 → 单K+双K+三K+图表+合成 5维评分"""
    score = 0.0
    reasons = []

    cp = _safe(indicators.get('current_price'), 0)
    o = _safe(indicators.get('open'), cp)
    h = _safe(indicators.get('high'), cp)
    l = _safe(indicators.get('low'), cp)
    vol = _safe(indicators.get('volume'), 0)

    # ── 维度1: 单K几何形态 ──
    body = abs(cp - o)
    candle_range = h - l
    body_ratio = body / (candle_range + 1e-10)
    lower_shadow = (min(o, cp) - l) / (candle_range + 1e-10)
    upper_shadow = (h - max(o, cp)) / (candle_range + 1e-10)

    # 锤子线: 长下影+小实体+下跌趋势
    ma20 = _safe(indicators.get('ma20'), cp)
    in_downtrend = cp < ma20
    hammer_score = lower_shadow * (1 - body_ratio) * (1 - upper_shadow)
    if in_downtrend and hammer_score > 0.4:
        score += 0.12; reasons.append(f'锤子线(下影{lower_shadow:.1%})→底部反转')
    elif hammer_score > 0.3:
        score += 0.05

    # 射击之星: 长上影+小实体+上涨趋势
    star_score = upper_shadow * (1 - body_ratio) * (1 - lower_shadow)
    if not in_downtrend and star_score > 0.4:
        score -= 0.12; reasons.append(f'射击之星(上影{upper_shadow:.1%})→顶部反转')
    elif star_score > 0.3:
        score -= 0.05

    # 十字星: body < 10% range
    if body_ratio < 0.10:
        reasons.append('十字星→变盘前兆')
        rsi = _safe(indicators.get('rsi'), 50)
        if rsi < 35:
            score += 0.06; reasons.append('低位十字星→底部信号')
        elif rsi > 65:
            score -= 0.06; reasons.append('高位十字星→顶部信号')

    # ── 维度2: 量价确认 ──
    vol_ratio = _safe(indicators.get('vol_ratio'), 1.0)
    if indicators.get('vol_price_resonance', False) and vol_ratio > 1.5:
        if score > 0:
            score += 0.06; reasons.append('放量确认形态→信号增强')
        elif score < 0:
            score -= 0.04
    if indicators.get('high_vol_stagnation', False):
        score -= 0.10; reasons.append('放量滞涨→形态信号减弱')
    if indicators.get('low_vol_pullback', False):
        score += 0.06; reasons.append('缩量回踩→供应枯竭配合形态')

    # ── 维度3: 分形支撑/阻力 ──
    pat_bot = _safe(indicators.get('pattern_bottom_fractal_n'), 0)
    pat_top = _safe(indicators.get('pattern_top_fractal_n'), 0)
    if pat_bot >= 2:
        score += 0.10; reasons.append(f'多底分形({int(pat_bot)}次)→强支撑确认')
    elif pat_bot == 1:
        score += 0.04
    if pat_top >= 2:
        score -= 0.10; reasons.append(f'多顶分形({int(pat_top)}次)→强阻力确认')
    elif pat_top == 1:
        score -= 0.04

    # ── 维度4: 趋势环境 ──
    ma5 = _safe(indicators.get('ma5'), cp)
    ma60 = _safe(indicators.get('ma60'), cp)
    adx = _safe(indicators.get('dmi_adx'), 20)
    atr_pct = _safe(indicators.get('vol_atr_pct'), 0.02)

    if ma5 > ma20 > ma60 and cp > 0:
        score += 0.06; reasons.append('均线多头→形态顺势')
    elif ma5 < ma20 < ma60 and cp > 0:
        score -= 0.06; reasons.append('均线空头→形态逆势')

    if adx < 18:
        score *= 0.70; reasons.append(f'ADX={adx:.1f}<18→震荡，形态可靠性降')
    if atr_pct > 0.05:
        score *= 0.75; reasons.append(f'高波动(ATR={atr_pct:.1%})→形态易失真')

    # ── 维度5: 反转信号检测 ──
    pat_mor = _safe(indicators.get('pattern_morning_star_n'), 0)
    pat_eve = _safe(indicators.get('pattern_evening_star_n'), 0)
    obv_div = str(indicators.get('obv_divergence', ''))

    if pat_mor > 0:
        score += 0.14; reasons.append('启明星→强反转')
    if pat_eve > 0:
        score -= 0.14; reasons.append('黄昏之星→强反转')
    if '底背离' in obv_div:
        score += 0.10; reasons.append('OBV底背离→形态+量价共振')
    elif '顶背离' in obv_div:
        score -= 0.10; reasons.append('OBV顶背离→形态+量价共振')

    score = _clip_score(score)
    return {
        'direction': _direction_from_score(score, 0.07),
        'score': round(score, 3),
        'confidence': round(min(abs(score) * 1.15, 0.92), 3),
        'reasons': reasons[:6],
    }


# Map school IDs to compute functions
def _compute_school_mean_reversion(indicators: Dict) -> Dict:
    """均值回归学派：BB+RSI+BIAS+KC 多因子 → 连续信号 (实盘全量DataFrame)"""
    try:
        from mean_reversion_school import compute_mean_reversion_signal
        df = indicators.get('_df')
        if df is None or len(df) < 20:
            return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
                    'reasons': ['实盘全量数据不足']}
        result = compute_mean_reversion_signal(df)
        if result:
            return {'direction': result['signal'],
                    'score': result['metadata'].get('score', 0),
                    'confidence': result['confidence'],
                    'reasons': result['metadata'].get('reasons', [])}
    except Exception:
        pass
    return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': []}


def _compute_school_nd(indicators: Dict) -> Dict:
    """正态分布学派：Z-Score超卖企稳/动量突破 → 连续信号 (实盘全量DataFrame)"""
    try:
        from normal_distribution_school import compute_nd_signal
        df = indicators.get('_df')
        if df is None or len(df) < 20:
            return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
                    'reasons': ['实盘全量数据不足']}
        result = compute_nd_signal(df)
        if result:
            return {'direction': result['signal'],
                    'score': result['metadata'].get('score', 0),
                    'confidence': result['confidence'],
                    'reasons': result['metadata'].get('reasons', [])}
    except Exception:
        pass
    return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': []}


def _compute_school_vp(indicators: Dict) -> Dict:
    """成交量分布学派：POC突破/回踩/受阻 → 连续信号 (实盘全量DataFrame)"""
    try:
        from volume_profile_school import compute_vp_signal
        df = indicators.get('_df')
        if df is None or len(df) < 20:
            return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
                    'reasons': ['实盘全量数据不足']}
        result = compute_vp_signal(df)
        if result:
            return {'direction': result['signal'],
                    'score': result['metadata'].get('score', 0),
                    'confidence': result['confidence'],
                    'reasons': result['metadata'].get('reasons', [])}
    except Exception:
        pass
    return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': []}


def _compute_school_ewm_beta(indicators: Dict) -> Dict:
    """EWM Beta学派：指数衰减加权β → 连续信号 (实盘全量DataFrame)"""
    try:
        from ewm_beta_factor import compute_ewm_beta_signal
        df = indicators.get('_df')
        if df is None or len(df) < 20:
            return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
                    'reasons': ['实盘全量数据不足']}
        result = compute_ewm_beta_signal(df)
        if result:
            return {'direction': result['signal'],
                    'score': result['metadata'].get('score', 0),
                    'confidence': result['confidence'],
                    'reasons': result['metadata'].get('reasons', [])}
    except Exception:
        pass
    return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': []}


def _compute_school_beta(indicators: Dict) -> Dict:
    """历史Beta学派：高Beta牛/熊市判定 → 连续信号 (实盘全量DataFrame)"""
    try:
        from beta_factor import compute_beta_signal
        df = indicators.get('_df')
        if df is None or len(df) < 20:
            return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
                    'reasons': ['实盘全量数据不足']}
        result = compute_beta_signal(df)
        if result:
            return {'direction': result['signal'],
                    'score': result['metadata'].get('score', 0),
                    'confidence': result['confidence'],
                    'reasons': result['metadata'].get('reasons', [])}
    except Exception:
        pass
    return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': []}


def _compute_school_roc_factor(indicators: Dict) -> Dict:
    """ROC动量因子学派：AX/BX ROC变体 → 连续信号 (实盘全量DataFrame)"""
    try:
        from roc_factor import compute_roc_signal
        df = indicators.get('_df')
        if df is None or len(df) < 20:
            return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
                    'reasons': ['实盘全量数据不足']}
        result = compute_roc_signal(df)
        if result:
            return {'direction': result['signal'],
                    'score': result['metadata'].get('score', 0),
                    'confidence': result['confidence'],
                    'reasons': result['metadata'].get('reasons', [])}
    except Exception:
        pass
    return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': []}


def _compute_school_brooks_pa(indicators: Dict) -> Dict:
    """Brooks价格行为学派 V1.0 — 八段工业级拓扑

    Topology:
      Step 1: Polarity accumulation (macro + micro split interfaces)
      Step 2: Baseline macro filters (ADX/BEGE/RV → conf_mod)
      Step 3: Nonlinear squashing (1.0 - exp(-x * 1.5))
      Step 4: Resonance interception (bull>0.40 & bear>0.40 → 0)
      Step 5: Net score: raw = clip(bull - bear, -1, 1)
      Step 6: Trap asymmetric amplifier (trap_mult 1.45x)
      Step 7: EMA magnet attenuator (magnet_mult 0.65x penalty)
      Step 8: Final assembly: raw × trap × magnet × conf_mod
    """
    import numpy as np
    score_bull = 0.0; score_bear = 0.0
    reasons_bull = []; reasons_bear = []

    # ════════════════════════════════════════════════════════════════
    # STEP 1 — Polarity accumulation (macro + micro split interfaces)
    # ════════════════════════════════════════════════════════════════
    ema_magnet = 0.0; ema_direction = 0.0; always_in_score = 0.0
    bull_trap_active = 0.0; bear_trap_active = 0.0

    try:
        from brooks_macro_features import compute_brooks_macro_score_split
        sb1, sbe1, rb1, rbe1, ema_magnet, ema_direction, always_in_score = \
            compute_brooks_macro_score_split(indicators)
        score_bull += sb1; score_bear += sbe1
        reasons_bull.extend(rb1); reasons_bear.extend(rbe1)
    except Exception:
        pass

    try:
        from brooks_micro_features import compute_brooks_micro_score_split
        sb2, sbe2, rb2, rbe2, bull_trap_active, bear_trap_active = \
            compute_brooks_micro_score_split(indicators)
        score_bull += sb2; score_bear += sbe2
        reasons_bull.extend(rb2); reasons_bear.extend(rbe2)
    except Exception:
        pass

    # ════════════════════════════════════════════════════════════════
    # STEP 2 — Baseline macro filters (conf_mod stack)
    # ════════════════════════════════════════════════════════════════
    conf_mod = 1.0
    conf_reasons = []

    adx = _safe(indicators.get('dmi_adx'), 20)
    if adx < 18:
        conf_mod *= 0.70
        conf_reasons.append(f'ADX={adx:.1f}<18→低趋势，PA信号降权')

    bebe_reg = str(indicators.get('bebe_regime', ''))
    if bebe_reg == 'bad_environment':
        conf_mod *= 0.75
        conf_reasons.append('BEGE不利→PA降权')

    rv_lvl = str(indicators.get('rv_level', ''))
    if '极高' in rv_lvl:
        conf_mod *= 0.55
        conf_reasons.append('极高波动→PA陷阱易失真')
    elif '高' in rv_lvl:
        conf_mod *= 0.75

    # ADX mild boost: PA works best in trending markets (ADX > 22)
    if adx > 25:
        conf_mod *= 1.05
        conf_reasons.append(f'ADX={adx:.1f}>25→趋势环境，PA信号增强')

    # Always In Trend Discipline (Brooks 核心纪律：绝不逆强势)
    # always_in_score ∈ [-1.0, 1.0]. |ai| > 0.30 → confirmed trend
    if score_bull > score_bear and always_in_score < -0.30:
        conf_mod *= 0.65
        conf_reasons.append(f'逆 Always In 空头方向(ai={always_in_score:.2f})→违背顺势纪律降权')
    elif score_bear > score_bull and always_in_score > 0.30:
        conf_mod *= 0.65
        conf_reasons.append(f'逆 Always In 多头方向(ai={always_in_score:.2f})→违背顺势纪律降权')

    # ════════════════════════════════════════════════════════════════
    # STEP 3 — Nonlinear squashing
    # ════════════════════════════════════════════════════════════════
    score_bull = 1.0 - np.exp(-score_bull * 1.5)
    score_bear = 1.0 - np.exp(-score_bear * 1.5)

    # ════════════════════════════════════════════════════════════════
    # STEP 4 — Resonance interception
    # ════════════════════════════════════════════════════════════════
    if score_bull > 0.40 and score_bear > 0.40:
        final_score = 0.0; confidence = 0.10; direction = 'neutral'
        reasons_final = ['PA多空信号剧烈冲突→宽幅震荡观望']
    else:
        # ════════════════════════════════════════════════════════════
        # STEP 5 — Net score
        # ════════════════════════════════════════════════════════════
        raw_final = np.clip(score_bull - score_bear, -1.0, 1.0)

        # ════════════════════════════════════════════════════════════
        # STEP 6 — Trap asymmetric amplifier
        # ════════════════════════════════════════════════════════════
        if raw_final > 0 and bear_trap_active > 0:
            trap_mult = 1.45
            reasons_bull.append(f'Bear Trap激活→诱空反转溢价(×{trap_mult:.2f})')
        elif raw_final < 0 and bull_trap_active > 0:
            trap_mult = 1.45
            reasons_bear.append(f'Bull Trap激活→诱多反转溢价(×{trap_mult:.2f})')
        else:
            trap_mult = 1.0

        # ════════════════════════════════════════════════════════════
        # STEP 7 — EMA magnet attenuator (防追高惩罚)
        # ════════════════════════════════════════════════════════════
        if ema_magnet > 0.60:
            if raw_final > 0 and ema_direction > 0:
                magnet_mult = 0.65
                conf_reasons.append(f'EMA上方极度偏离(mag={ema_magnet:.2f})→追高风险折价')
            elif raw_final < 0 and ema_direction < 0:
                magnet_mult = 0.65
                conf_reasons.append(f'EMA下方极度偏离(mag={ema_magnet:.2f})→追空风险折价')
            else:
                magnet_mult = 1.0
        else:
            magnet_mult = 1.0

        # ════════════════════════════════════════════════════════════
        # STEP 8 — Final assembly
        # ════════════════════════════════════════════════════════════
        final_score = raw_final * trap_mult * magnet_mult * conf_mod
        final_score = np.clip(final_score, -1.0, 1.0)

        confidence = min(abs(final_score) * 1.20, 0.95)
        direction = 'bullish' if final_score > 0.06 else \
                    ('bearish' if final_score < -0.06 else 'neutral')
        reasons_final = (reasons_bull if final_score > 0 else
                         (reasons_bear if final_score < 0 else
                          reasons_bull + reasons_bear))[:6]
        reasons_final.extend(conf_reasons[:3])

    return {
        'direction': direction,
        'score': round(float(final_score), 3),
        'confidence': round(float(confidence), 3),
        'reasons': reasons_final,
    }


SCHOOL_COMPUTE = {
    'school_chanlun': _compute_school_chanlun,
    'school_tang': _compute_school_tang,
    'school_livermore': _compute_school_livermore,
    'school_busch': _compute_school_busch,
    'school_classical': _compute_school_classical,
    'school_risk': _compute_school_risk,
    'school_gann': _compute_school_gann,
    'school_wyckoff': _compute_school_wyckoff,
    'school_harmonic': _compute_school_harmonic,
    'school_roc_breakout': _compute_school_roc_breakout,
    'school_volume_profile': _compute_school_volume_profile,
    'school_fusion': _compute_school_fusion,
    'school_mean_reversion': _compute_school_mean_reversion,
    'school_capital_flow': _compute_school_capital_flow,
    'school_pattern_features': _compute_school_pattern_features,
    'school_nd': _compute_school_nd,
    'school_vp': _compute_school_vp,
    'school_roc_factor': _compute_school_roc_factor,
    'school_beta': _compute_school_beta,
    'school_ewm_beta': _compute_school_ewm_beta,
    'school_brooks_pa': _compute_school_brooks_pa,
}

# ── ML Inference: cached model loading (read once from disk, never again) ──
_ML_MODEL = None
_ML_MODEL_LOADED = False

def _load_ml_model():
    """Load XGBoost model from disk ONCE. Returns None if unavailable."""
    global _ML_MODEL, _ML_MODEL_LOADED
    if _ML_MODEL_LOADED:
        return _ML_MODEL
    _ML_MODEL_LOADED = True
    try:
        import xgboost as xgb
        from pathlib import Path
        model_path = Path(__file__).parent / 'models' / 'xgb_master.json'
        if model_path.exists():
            _ML_MODEL = xgb.XGBClassifier()
            _ML_MODEL.load_model(str(model_path))
            return _ML_MODEL
    except Exception:
        pass
    return None


_ML_SCHOOL_KEYS = [
    'school_chanlun', 'school_tang', 'school_livermore', 'school_busch',
    'school_classical', 'school_risk', 'school_gann', 'school_wyckoff',
    'school_harmonic', 'school_roc_breakout', 'school_volume_profile',
    'school_fusion', 'school_mean_reversion', 'school_capital_flow',
    'school_pattern_features', 'school_brooks_pa',
]


def _compute_school_ml(indicators: Dict) -> Dict:
    """
    ML school (16th school) — XGBoost prediction from 15 school votes.

    Feature vector: for each of the 15 schools, direction_num × confidence.
    Falls back to neutral if model is not trained yet.
    """
    model = _load_ml_model()
    if model is None:
        return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
                'reasons': ['ML model not trained yet']}

    import numpy as np
    # indicators dict doesn't have school_signals — we need them from the caller.
    # _compute_school_ml is called from compute_expert_ensemble AFTER
    # school_signals are computed but BEFORE the ML school is called.
    # Solution: the indicators dict must carry a '_precomputed_school_signals' key
    # OR we accept a second parameter. For compatibility with SCHOOL_COMPUTE
    # which is called with (indicators), we use a sentinel.
    school_signals = indicators.get('_precomputed_school_signals', {})
    if not school_signals:
        return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
                'reasons': ['No school signals available for ML inference']}

    # Build 15-dim feature vector
    feats = np.zeros(len(_ML_SCHOOL_KEYS), dtype=np.float64)
    for j, key in enumerate(_ML_SCHOOL_KEYS):
        sig = school_signals.get(key, {})
        direction = str(sig.get('direction', 'neutral')).lower()
        confidence = float(sig.get('confidence', 0))
        if direction == 'bullish':
            d_num = 1.0
        elif direction == 'bearish':
            d_num = -1.0
        else:
            d_num = 0.0
        feats[j] = d_num * confidence

    # Predict
    try:
        proba = model.predict_proba(feats.reshape(1, -1))[0]
        # proba[0] = P(class 0 = bearish/short), proba[1] = P(class 1 = bullish/long)
        bull_prob = float(proba[1])
        if bull_prob > 0.55:
            direction = 'bullish'
            confidence = bull_prob
            score = (bull_prob - 0.5) * 2.0
        elif bull_prob < 0.45:
            direction = 'bearish'
            confidence = 1.0 - bull_prob
            score = -(0.5 - bull_prob) * 2.0
        else:
            direction = 'neutral'
            confidence = 0.5
            score = 0.0

        return {
            'direction': direction,
            'score': round(score, 3),
            'confidence': round(confidence, 3),
            'reasons': [f'XGBoost prob={bull_prob:.3f}'],
        }
    except Exception as e:
        return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
                'reasons': [f'ML inference error: {e}']}


# Register ML school (must be after _compute_school_ml definition)
SCHOOL_COMPUTE['school_mean_reversion'] = _compute_school_mean_reversion
SCHOOL_COMPUTE['school_ml'] = _compute_school_ml

# ============================================================
# Nüwa Router — performance-based adaptive weighting
# ============================================================

def get_nuwa_school_weights(indicators: Dict = None, verbose=None) -> Dict[str, float]:
    if verbose is None:
        try:
            from config import NUWA_VERBOSE
            verbose = NUWA_VERBOSE
        except ImportError:
            verbose = False
    """
    Three-source adaptive school weight blending:

        w_i = (1-α-β) × regime_weight_i + α × learned_weight_i + β × trust_score_i

    regime_weight_i  — how appropriate school i is for the current regime
    learned_weight_i — data-driven weight from backtest performance (SchoolWeightLearner)
    trust_score_i    — historical accuracy from live signal feedback

    α grows with data: α = min(0.60, n_trades / 200)
    β = 0.25 (fixed trust-score share)

    Falls back to the legacy formula when no learned weights are available.
    """
    weights = {name: 1.0 for name in SCHOOL_NAMES}

    # Determine regime for regime-appropriate weighting (shared logic)
    regime = 'transitional'
    regime_weights = None
    if indicators is not None:
        regime = _detect_market_regime(indicators)
        regime_weights = _REGIME_WEIGHTS[regime]

    # ---- Attempt three-source blending ----
    try:
        from backtest_feedback import get_school_trust_scores
        trust_scores = get_school_trust_scores() or {}

        # Try to load learned weights from evolution DB
        learned_weights = None
        n_trades_attributed = 0
        try:
            from trade_logger import load_regime_weights_learned
            all_learned = load_regime_weights_learned()
            if all_learned and regime in all_learned:
                learned_weights = all_learned[regime]
                # Estimate n_trades from DB
                from trade_logger import list_backtest_runs
                runs = list_backtest_runs(limit=10)
                n_trades_attributed = sum(r.get('total_trades', 0) for r in runs[:5])
        except Exception:
            pass

        # Simplified blend: 50% learned + 50% regime
        alpha = min(0.70, n_trades_attributed / 30.0)

        if learned_weights and n_trades_attributed >= 5:
            for name in SCHOOL_NAMES:
                rw = regime_weights.get(name, 0.5) if regime_weights else 0.5
                lw = learned_weights.get(name, 0.5)
                weights[name] = round((1.0 - alpha) * rw + alpha * lw, 3)
            if verbose:
                top3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
                labels = [f"{SCHOOLS[n]['short_label']}({w:.2f})" for n, w in top3]
                print(f"  [Nuwa] 学派权重Top3 (alpha={alpha:.2f} learned+regime): {', '.join(labels)}")
        elif trust_scores:
            # Legacy: regime × trust blend (no learned weights yet)
            for name in SCHOOL_NAMES:
                school_score = trust_scores.get(name, 0.5)
                rw = regime_weights.get(name, 0.6) if regime_weights else 0.6
                weights[name] = round(rw * (0.40 + 0.60 * school_score), 3)
            if verbose:
                top3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
                labels = [f"{SCHOOLS[n]['short_label']}({w:.2f})" for n, w in top3]
                print(f"  [Nüwa] 学派权重Top3 (legacy regime×trust): {', '.join(labels)}")
        elif regime_weights:
            weights = {name: round(v, 3) for name, v in regime_weights.items()}
            if verbose:
                top3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
                labels = [f"{SCHOOLS[n]['short_label']}({w:.2f})" for n, w in top3]
                print(f"  [Nüwa] 学派权重Top3 (regime-only): {', '.join(labels)}")
        else:
            if verbose:
                print(f"  [Nüwa] 学派权重: equal (no data available)")
    except Exception as e:
        if regime_weights:
            weights = {name: round(v, 3) for name, v in regime_weights.items()}
        if verbose:
            top3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
            labels = [f"{SCHOOLS[n]['short_label']}({w:.2f})" for n, w in top3]
            source = "regime" if regime_weights else "equal"
            print(f"  [Nüwa] 学派权重Top3 ({source}): {', '.join(labels)}")

    # ADX<22: dampen trend-following schools, boost mean-reversion
    if indicators is not None:
        adx = _safe(indicators.get('dmi_adx'), 20)
        if adx < 22:
            trend_schools = {'school_chanlun', 'school_livermore', 'school_classical',
                           'school_gann', 'school_busch', 'school_dow', 'school_turtle',
                           'school_roc_breakout'}
            for name in trend_schools:
                if name in weights:
                    weights[name] = round(weights[name] * 0.60, 3)
            # Boost mean reversion in ranging
            if 'school_mean_reversion' in weights:
                weights['school_mean_reversion'] = min(1.0, weights['school_mean_reversion'] * 1.5)
            if 'school_volume_profile' in weights:
                weights['school_volume_profile'] = min(1.0, weights['school_volume_profile'] * 1.2)
            if 'school_harmonic' in weights:
                weights['school_harmonic'] = min(1.0, weights['school_harmonic'] * 1.2)

    # ---- Collinearity penalty: penalize highly correlated schools ----
    try:
        from factor_pipeline import CollinearityPenalty
        if not hasattr(get_nuwa_school_weights, '_collinearity'):
            get_nuwa_school_weights._collinearity = CollinearityPenalty()
        cp = get_nuwa_school_weights._collinearity
        school_scores = {name: weights.get(name, 0.5) for name in SCHOOL_NAMES}
        weights = cp.apply_penalties(weights, school_scores)
    except ImportError:
        pass

    # ---- Multi-timeframe half-life decay ----
    try:
        from advanced_ensemble import HalfLifeDecay
        hld = HalfLifeDecay()
        # Schools with hourly/intraday indicators get faster decay
        hourly_schools = {'school_orderflow'}  # Omitted from list
        daily_schools = {n for n in SCHOOL_NAMES if n not in hourly_schools}
        for name in daily_schools:
            if name in weights:
                weights[name] = round(weights[name] * hld.weight('daily', 0.0), 3)
    except ImportError:
        pass

    return weights


# ============================================================
# Synergy Matrix — pairwise school synergy measurement
# ============================================================

def compute_synergy_matrix(school_signals: Dict[str, Dict]) -> Dict:
    """Measure pairwise synergy between schools."""
    schools = list(school_signals.keys())
    n = len(schools)
    synergy = {}

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = schools[i], schools[j]
            s1, s2 = school_signals[si], school_signals[sj]
            same_dir = s1['direction'] == s2['direction']
            both_dir = s1['direction'] != 'neutral' and s2['direction'] != 'neutral'

            if both_dir:
                if same_dir:
                    shared = len(set(s1.get('reasons', [])) & set(s2.get('reasons', [])))
                    total = len(set(s1.get('reasons', [])) | set(s2.get('reasons', [])))
                    overlap = shared / max(total, 1)
                    diversity_bonus = 1.0 - overlap
                    base_syn = 0.6 + 0.4 * diversity_bonus
                else:
                    base_syn = -0.5
            else:
                base_syn = 0.1

            score_prod = abs(s1['score']) * abs(s2['score'])
            synergy[f'{si}_{sj}'] = {
                'synergy_score': round(base_syn * (0.5 + score_prod), 3),
                'same_direction': same_dir,
                'pair_label': f'{SCHOOLS[si]["short_label"]}×{SCHOOLS[sj]["short_label"]}',
            }

    return synergy


# ============================================================
# Enhanced Diversity
# ============================================================

def compute_enhanced_diversity(school_signals: Dict) -> Dict:
    """Multi-metric diversity across schools."""
    names = list(school_signals.keys())
    n = len(names)
    confidences = np.array([school_signals[nm]['confidence'] for nm in names])
    directions = [school_signals[nm]['direction'] for nm in names]

    # Pairwise direction disagreement
    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            d1, d2 = directions[i], directions[j]
            if d1 == d2:
                pairwise.append(0.0)
            elif d1 == 'neutral' or d2 == 'neutral':
                pairwise.append(0.5)
            else:
                pairwise.append(1.0)
    diversity_pd = float(np.mean(pairwise)) if pairwise else 0.0

    # Reason overlap
    all_tokens = []
    for nm in names:
        reasons = ' '.join(school_signals[nm].get('reasons', []))
        tokens = set(reasons.replace('/', ' ').replace(',', ' ').replace('，', ' ').split())
        all_tokens.append(tokens)
    jaccards = []
    for i in range(n):
        for j in range(i + 1, n):
            if all_tokens[i] and all_tokens[j]:
                jac = len(all_tokens[i] & all_tokens[j]) / max(len(all_tokens[i] | all_tokens[j]), 1)
                jaccards.append(jac)
    mean_jac = float(np.mean(jaccards)) if jaccards else 0.5
    diversity_reason = 1.0 - mean_jac

    # Effective N (Herfindahl inverse)
    total_conf = np.sum(confidences)
    if total_conf > 0:
        w = confidences / total_conf
        n_eff = 1.0 / max(np.sum(w ** 2), 0.01)
    else:
        n_eff = float(n)

    # Entropy diversity
    if total_conf > 0:
        eps = 1e-10
        w_ent = confidences / total_conf
        entropy = -np.sum(w_ent * np.log(w_ent + eps))
        max_entropy = np.log(float(n))
        diversity_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.5
    else:
        diversity_entropy = 0.5

    diversity_composite = (0.25 * diversity_pd + 0.25 * diversity_reason +
                           0.25 * min(np.log(n_eff) / np.log(n), 1.0) + 0.25 * diversity_entropy)

    return {
        'diversity_pd': round(diversity_pd, 3),
        'diversity_reason': round(diversity_reason, 3),
        'diversity_n_eff': round(float(n_eff), 2),
        'diversity_entropy': round(diversity_entropy, 3),
        'diversity_composite': round(diversity_composite, 3),
        'reason_overlap_high': mean_jac > 0.6,
    }


# ============================================================
# Expert Soup — softmax-weighted score blending
# ============================================================

def compute_expert_soup(school_signals: Dict, temperature: float = 1.0,
                        nuwa_weights: Dict[str, float] = None) -> Dict:
    """Weight-averaging fusion with optional Nüwa weights."""
    scores = np.array([s['score'] for s in school_signals.values()])
    confidences = np.array([s['confidence'] for s in school_signals.values()])
    names = list(school_signals.keys())

    # Apply Nüwa weights
    if nuwa_weights:
        for i, name in enumerate(names):
            confidences[i] *= nuwa_weights.get(name, 1.0)

    logits = confidences / max(temperature, 0.1)
    exp_logits = np.exp(logits - np.max(logits))
    softmax_w = exp_logits / np.sum(exp_logits)

    soup_score = float(np.sum(softmax_w * scores))
    soup_confidence = float(np.sum(softmax_w * confidences) / max(np.sum(softmax_w), 0.01))

    if soup_score > 0.06:
        soup_dir = 'bullish'
    elif soup_score < -0.06:
        soup_dir = 'bearish'
    else:
        soup_dir = 'neutral'

    return {
        'soup_score': round(soup_score, 4),
        'soup_confidence': round(min(soup_confidence, 0.95), 3),
        'soup_direction': soup_dir,
        'soup_weights': {names[i]: round(float(softmax_w[i]), 3) for i in range(len(names))},
    }


# ============================================================
# Shared regime detection + weight tables (single source of truth)
# ============================================================

_REGIME_WEIGHTS = {
    'trending': {
        'school_chanlun': 0.3, 'school_tang': 0.4,
        'school_livermore': 0.7, 'school_busch': 0.5, 'school_classical': 0.8,
        'school_risk': 0.5, 'school_gann': 0.8, 'school_wyckoff': 0.8,
        'school_harmonic': 0.4,

        'school_roc_breakout': 0.9,
        'school_volume_profile': 0.6,
        'school_fusion': 0.3,
        'school_mean_reversion': 0.2,
        'school_capital_flow': 0.7,
    },
    'ranging': {
        'school_chanlun': 0.3, 'school_tang': 1.0,
        'school_livermore': 0.4, 'school_busch': 0.7, 'school_classical': 0.5,
        'school_risk': 0.6, 'school_gann': 0.6, 'school_wyckoff': 1.0,
        'school_harmonic': 0.7,        'school_roc_breakout': 0.4,
        'school_volume_profile': 1.0,
        'school_fusion': 0.3,
        'school_mean_reversion': 0.5,
        'school_capital_flow': 0.6,
    },
    'volatile': {
        'school_chanlun': 0.3, 'school_tang': 0.4,
        'school_livermore': 0.4, 'school_busch': 0.5, 'school_classical': 0.3,
        'school_risk': 1.0, 'school_gann': 0.5, 'school_wyckoff': 0.4,
        'school_harmonic': 0.3,        'school_roc_breakout': 0.8,
        'school_volume_profile': 0.5,
        'school_fusion': 0.3,
        'school_mean_reversion': 0.3,
        'school_capital_flow': 0.7,
    },
    'transitional': {
        'school_chanlun': 0.3, 'school_tang': 0.6,
        'school_livermore': 0.5, 'school_busch': 0.6, 'school_classical': 0.6,
        'school_risk': 0.6, 'school_gann': 0.7, 'school_wyckoff': 0.8,
        'school_harmonic': 0.6,        'school_roc_breakout': 0.7,
        'school_volume_profile': 0.8,
        'school_fusion': 0.3,
        'school_mean_reversion': 0.4,
        'school_capital_flow': 0.7,
    },
}


def _detect_market_regime(indicators: Dict) -> str:
    """Classify current market regime from indicators. Single source of truth."""
    adx = _safe(indicators.get('dmi_adx'), 15)
    rv_pct = _safe(indicators.get('rv_percentile'), 0.5)
    if adx > 25:
        return 'trending'
    elif adx < 18:
        return 'ranging'
    elif rv_pct > 0.85:
        return 'volatile'
    return 'transitional'


# ============================================================
# Dynamic Router — market-regime-aware gating (14 schools)
# ============================================================

def compute_dynamic_router(indicators: Dict, school_signals: Dict) -> Dict:
    """Market-condition-aware gating for 15 schools."""
    regime = _detect_market_regime(indicators)
    weights = _REGIME_WEIGHTS[regime]

    gated_score = 0.0
    gated_conf = 0.0
    aligned_conf = 0.0
    total_conf = 0.0
    active_schools = []
    for name, sig in school_signals.items():
        w = weights.get(name, 0.5)
        if w >= 0.5:
            active_schools.append(name)
        d = 1.0 if sig['direction'] == 'bullish' else (-1.0 if sig['direction'] == 'bearish' else 0.0)
        gated_score += w * d * sig['confidence']
        gated_conf += w * sig['confidence']
        total_conf += sig['confidence']

    if gated_conf > 0:
        gated_score /= gated_conf

    # Proper confidence: fraction of weighted agreement vs total, cross-checked with regime weights
    agreed_weight = sum(
        weights.get(n, 0.5) * s['confidence']
        for n, s in school_signals.items()
        if s['direction'] == _direction_from_score(gated_score, 0.08)
    )
    proper_confidence = agreed_weight / max(gated_conf, 0.001)
    # Blend with regime signal clarity: how decisive is the regime's preference?
    active_agree = sum(1 for n in active_schools
                       if school_signals[n]['direction'] == _direction_from_score(gated_score, 0.08))
    active_clarity = active_agree / max(len(active_schools), 1)
    proper_confidence = 0.7 * proper_confidence + 0.3 * active_clarity

    return {
        'regime': regime,
        'gate_weights': weights,
        'active_schools': active_schools,
        'gated_score': round(gated_score, 4),
        'gated_confidence': round(min(proper_confidence, 0.95), 3),
        'gated_direction': _direction_from_score(gated_score, 0.08),
    }


# ============================================================
# Confidence recalibration — decouple confidence from raw score
# ============================================================

def _recalibrate_school_confidences(school_signals: Dict, indicators: Dict) -> Dict:
    """
    Replace score-derived confidence with evidence-quality-based confidence.

    Confidence = f(evidence_count, evidence_consistency, trend_clarity)

    Rationale: confidence = abs(score)*1.25 is circular — it makes confidence
    proportional to directional conviction, not to evidence quality.
    Two schools with score=0.8 should have DIFFERENT confidence if one has
    2 weak reasons and the other has 5 converging reasons.
    """
    adx = _safe(indicators.get('dmi_adx'), 20)
    trend_clarity = min(1.0, max(0.3, adx / 30.0))  # 0.3 (choppy) to 1.0 (strong trend)

    for name, sig in school_signals.items():
        reasons = sig.get('reasons', [])
        n_reasons = len(reasons)

        # Evidence count score: more independent reasons → higher confidence
        evidence_count_score = min(1.0, n_reasons / 5.0)  # max at 5 reasons

        # Evidence consistency: detect genuine internal contradictions
        # A contradiction requires BOTH bullish AND bearish reasons from the SAME domain
        # (e.g. "MA多头排列" + "死叉卖出" = real conflict; "支撑有效" + "接近阻力" = normal)
        contradiction_keywords = {
            'bull_action': ['做多', '买入', '入场', '加仓', '突破买入', '抄底'],
            'bear_action': ['做空', '卖出', '离场', '减仓', '突破卖出', '逃顶'],
            'bull_pattern': ['金叉', '底背离', '底分型', '早晨之星', '看涨吞没', '双底', '头肩底'],
            'bear_pattern': ['死叉', '顶背离', '顶分型', '黄昏之星', '看跌吞没', '双顶', '头肩顶'],
        }
        all_reasons_text = ' '.join(reasons)

        has_bull_action = any(kw in all_reasons_text for kw in contradiction_keywords['bull_action'])
        has_bear_action = any(kw in all_reasons_text for kw in contradiction_keywords['bear_action'])
        has_bull_pattern = any(kw in all_reasons_text for kw in contradiction_keywords['bull_pattern'])
        has_bear_pattern = any(kw in all_reasons_text for kw in contradiction_keywords['bear_pattern'])

        # Genuine contradiction: opposing ACTION signals OR opposing PATTERN signals
        action_contradiction = has_bull_action and has_bear_action
        pattern_contradiction = has_bull_pattern and has_bear_pattern

        if action_contradiction or pattern_contradiction:
            consistency = 0.45  # genuine internal contradiction
        elif (has_bull_action or has_bull_pattern) and not (has_bear_action or has_bear_pattern):
            consistency = 0.85  # consistently bullish
        elif (has_bear_action or has_bear_pattern) and not (has_bull_action or has_bull_pattern):
            consistency = 0.85  # consistently bearish
        elif (has_bull_action or has_bull_pattern) or (has_bear_action or has_bear_pattern):
            consistency = 0.65  # mixed but action+pattern from same direction is not contradictory
        else:
            consistency = 0.55  # neutral/ambiguous

        # Trend clarity bonus: in clear trends, all signals are more reliable
        clarity_bonus = 0.7 + 0.3 * trend_clarity

        # Combined confidence
        raw_conf = (0.35 * evidence_count_score + 0.40 * consistency + 0.25 * clarity_bonus)
        raw_conf = min(raw_conf, 0.92)

        # Original score magnitude still matters as a floor
        score_magnitude = abs(sig['score'])
        floor_conf = min(score_magnitude * 0.4, 0.35)

        sig['confidence'] = round(max(raw_conf, floor_conf), 3)

    return school_signals


# ============================================================
# School Ensemble — primary entry point
# ============================================================

def compute_market_regime_prior(indicators: Dict) -> Dict:
    """
    Bayesian market regime PRIOR — shifts log-odds, NEVER gates.

    Instead of hard-blocking signals, this returns a log-odds SHIFT:
      - Bearish market: prior_shift < 0 (shifts toward bearish)
      - Bullish market: prior_shift > 0 (shifts toward bullish)
      - Neutral: prior_shift ≈ 0

    The shift is applied additively in log-odds space BEFORE sigmoid,
    so even in a bear market, exceptionally strong bullish multi-source
    signals can overcome the negative prior.
    """
    mp_phase = str(indicators.get('mp_phase', ''))
    mp_conf = _safe(indicators.get('mp_confidence'), 0)
    top_grade = str(indicators.get('top_escape_grade', ''))
    rv_level = str(indicators.get('rv_level', ''))
    bebe_regime = str(indicators.get('bebe_regime', ''))

    prior_shift = 0.0     # log-odds shift: + = bullish prior, - = bearish prior
    confidence_floor = 0.05  # minimum confidence after sigmoid
    regime_label = 'neutral'
    reasons = []

    # Bear phase → negative prior shift
    if mp_phase == '下跌' and mp_conf > 0.6:
        shift_strength = -1.5 * mp_conf        # max -1.5 log-odds
        prior_shift += shift_strength
        regime_label = 'bearish'
        reasons.append(f'大盘下跌(conf={mp_conf:.0%})→轻微降温{shift_strength:.1f}')

    elif mp_phase == '盘头' and mp_conf > 0.55:
        shift_strength = -0.8 * mp_conf        # max -0.8 log-odds
        prior_shift += shift_strength
        regime_label = 'bearish'
        reasons.append(f'大盘盘头(conf={mp_conf:.0%})→轻微降温{shift_strength:.1f}')

    # Bull phase → positive prior shift
    if mp_phase == '上涨' and mp_conf > 0.6:
        shift_strength = 1.2 * mp_conf
        prior_shift += shift_strength
        regime_label = 'bullish'
        reasons.append(f'大盘上涨(conf={mp_conf:.0%})→轻微降温+{shift_strength:.1f}')

    # Critical top → stronger negative prior
    if top_grade == 'critical':
        prior_shift += -2.0
        confidence_floor = 0.03
        reasons.append('逃顶预警critical→强力看空先验')

    # Extreme volatility → wider uncertainty (lower effective prior magnitude)
    if rv_level in ('极高波动', '高波动'):
        prior_shift *= 0.7
        reasons.append('高波动→先验强度衰减30%')

    return {
        'prior_shift': round(prior_shift, 3),
        'confidence_floor': confidence_floor,
        'regime_label': regime_label,
        'reasons': reasons,
    }


def compute_expert_ensemble(indicators: Dict) -> Dict:
    """
    Run all 15 school experts, measure diversity, and fuse signals.

    Two-stage fusion:
      1. Each school internally resolves contradictions → {direction, confidence}
      2. Schools vote with Nüwa performance-weighted blending
    """
    # Stage 1: Run 15 rule-based schools (ML computed separately, not in Log-Odds)
    school_signals = {}
    for name, compute_fn in SCHOOL_COMPUTE.items():
        if name == 'school_ml':
            continue  # computed independently — post-filter only, never in fusion
        try:
            school_signals[name] = compute_fn(indicators)
        except Exception:
            school_signals[name] = {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': []}

    # Stage 1.5: Recalibrate confidences (decouple from raw score)
    school_signals = _recalibrate_school_confidences(school_signals, indicators)

    # ── Mean Reversion VIP: one-vote veto in ranging markets ──
    mr_vip_triggered = False
    mr_sig = school_signals.get('school_mean_reversion', {})
    mp_phase_mr = str(indicators.get('mp_phase', ''))
    if mp_phase_mr in ('震荡', '盘整', '筑底') and mr_sig.get('direction') == 'bullish':
        mr_vip_triggered = True

    # Nüwa adaptive weights (from regime + historical backtest learning)
    nuwa_weights = get_nuwa_school_weights(indicators=indicators)

    # Pre-compute dynamic router (used as tiebreaker + result output)
    dynamic_router = compute_dynamic_router(indicators, school_signals)

    # Stage 2: Log-Odds (Naive Bayes) fusion
    total_log_odds = 0.0
    raw_votes = {'bullish': 0, 'bearish': 0, 'neutral': 0}
    weighted_votes = {'bullish': 0.0, 'bearish': 0.0, 'neutral': 0.0}

    for name, sig in school_signals.items():
        if name == 'school_ml':
            continue

        direction = sig['direction']
        nw = nuwa_weights.get(name, 1.0)
        conf = max(0.05, min(0.95, sig['confidence']))
        raw_votes[direction] += 1
        weighted_votes[direction] += conf * nw

        if direction == 'bullish':
            total_log_odds += np.log(conf / (1.0 - conf)) * nw
        elif direction == 'bearish':
            total_log_odds -= np.log(conf / (1.0 - conf)) * nw

    # Bayesian prior shift from market regime
    market_prior = compute_market_regime_prior(indicators)
    total_log_odds += market_prior['prior_shift']

    # Sigmoid: log-odds → probability
    ensemble_conf = 1.0 / (1.0 + np.exp(-abs(total_log_odds)))
    ensemble_conf = max(market_prior['confidence_floor'], min(0.95, ensemble_conf))

    # Direction from sign of log-odds
    if total_log_odds > 0.15:
        ensemble_dir = 'bullish'
        base_confidence = ensemble_conf
    elif total_log_odds < -0.15:
        ensemble_dir = 'bearish'
        base_confidence = ensemble_conf
    else:
        ensemble_dir = 'neutral'
        base_confidence = max(0.20, ensemble_conf * 0.6)

    # MR VIP override
    if mr_vip_triggered and ensemble_dir != 'bullish':
        ensemble_dir = 'bullish'
        base_confidence = min(0.85, mr_sig.get('confidence', 0.7))

    n_bullish = raw_votes['bullish']  # for downstream compatibility

    # Synergy & diversity
    synergy = compute_synergy_matrix(school_signals)
    enhanced_div = compute_enhanced_diversity(school_signals)
    diversity = enhanced_div['diversity_pd']

    # Diversity bonus
    if diversity > 0.3 and ensemble_dir != 'neutral':
        base_confidence *= min(1.0 + diversity * 0.25, 1.25)

    # Synergy bonus/penalty
    syn_pairs = [k for k, v in synergy.items() if v['synergy_score'] > 0.5]
    conflict_pairs = [k for k, v in synergy.items() if v['synergy_score'] < -0.3]
    if syn_pairs:
        base_confidence *= min(1.0 + len(syn_pairs) * 0.04, 1.15)
    if conflict_pairs:
        base_confidence *= max(0.55, 1.0 - len(conflict_pairs) * 0.12)

    # Pseudo-consensus penalty
    if enhanced_div['reason_overlap_high'] and ensemble_dir != 'neutral':
        base_confidence *= 0.85
    if enhanced_div['diversity_n_eff'] < 3.0 and ensemble_dir != 'neutral':
        base_confidence *= 0.90

    base_confidence = np.clip(base_confidence, 0.20, 0.95)

    # Collect reasons
    supporting = []
    opposing = []
    for name, sig in school_signals.items():
        if sig['direction'] == ensemble_dir:
            supporting.extend(sig.get('reasons', []))
        elif sig['direction'] != 'neutral':
            opposing.extend(sig.get('reasons', []))

    # Expert soup (market prior already embedded via log-odds, no gate here)
    expert_soup = compute_expert_soup(school_signals, nuwa_weights=nuwa_weights)
    # Append market regime prior reasons to supporting context
    if market_prior.get('reasons'):
        supporting.extend(market_prior['reasons'])

    result = {
        'ensemble_signal': ensemble_dir,
        'ensemble_confidence': round(float(base_confidence), 3),
        'ensemble_score': round(float(total_log_odds), 3),
        'votes': raw_votes,
        'weighted_votes': {k: round(v, 3) for k, v in weighted_votes.items()},
        'diversity': round(diversity, 3),
        'synergy_pairs': syn_pairs,
        'conflict_pairs': conflict_pairs,
        'consensus': {
            'bullish_voters': raw_votes['bullish'],
            'mr_vip_triggered': mr_vip_triggered,
        },
        'supporting_reasons': supporting[:8],
        'opposing_reasons': opposing[:5],
        'school_signals': {
            name: {
                'label': SCHOOLS[name]['label'],
                'short_label': SCHOOLS[name]['short_label'],
                'direction': sig['direction'],
                'confidence': sig['confidence'],
                'score': sig['score'],
                'reasons': sig['reasons'],
                'nuwa_weight': nuwa_weights.get(name, 1.0),
            }
            for name, sig in school_signals.items()
        },
        'dynamic_router': dynamic_router,
        'expert_soup': expert_soup,
        'enhanced_diversity': enhanced_div,
        'nuwa_weights': nuwa_weights,
        'market_prior': market_prior,
        'bayesian_refinement': _bayesian_ensemble_refinement(school_signals),
    }

    # ---- XGBoost Nonlinear Ensemble (override when trained) ----
    xgb_prediction = None
    try:
        from walk_forward_xgb import WalkForwardXGB
        # Lazy-load or create global Walk-Forward XGBoost
        if not hasattr(compute_expert_ensemble, '_wf_xgb'):
            compute_expert_ensemble._wf_xgb = WalkForwardXGB()
            # Auto-train from backtest trades
            _auto_train_wf_xgb(compute_expert_ensemble._wf_xgb)
        wf = compute_expert_ensemble._wf_xgb
        if wf._models:
            # 特征标准化: 每窗口独立 RobustScaler (已在 fit 中实现)
            features = wf.build_features(school_signals)
            # 使用最近3个模型的集成预测
            direction, prob, meta = wf.predict(school_signals)
            xgb_prediction = {
                'direction': direction,
                'probability': round(prob, 3),
                'model_count': meta['model_count'] if meta else 0,
                'method': 'walk_forward_xgb',
            }
        else:
            # Fallback to simple XGBoost
            from advanced_ensemble import NonLinearEnsemble
            if not hasattr(compute_expert_ensemble, '_xgb_model'):
                compute_expert_ensemble._xgb_model = NonLinearEnsemble()
                _auto_train_xgb(compute_expert_ensemble._xgb_model)
            xgb = compute_expert_ensemble._xgb_model
            if xgb._model is not None:
                xgb_dir, xgb_prob = xgb.predict(school_signals)
                xgb_prediction = {'direction': xgb_dir, 'probability': round(xgb_prob, 3)}
    except ImportError:
        pass
    result['xgb_prediction'] = xgb_prediction

    # ---- XGBRanker + SHAP Explainability ----
    try:
        from ranker_ensemble import explain_prediction
        ranker_result = explain_prediction(school_signals)
        result['ranker_score'] = ranker_result.get('rank_score', 3.0)
        shap_explanation = ranker_result.get('shap_explanation', '')
        if shap_explanation:
            # Inject into supporting_reasons
            result['supporting_reasons'] = (
                [shap_explanation] + result.get('supporting_reasons', [])[:7]
            )
    except ImportError:
        pass

    # ── Shadow Trading: strict post-filter (ML excluded from Log-Odds) ──
    from config import USE_ML_AS_MASTER

    # Extract ML signal from xgb_prediction (computed above, outside Log-Odds)
    ml_dir = 'neutral'
    ml_conf = 0.0
    if xgb_prediction and xgb_prediction.get('direction'):
        ml_dir = xgb_prediction['direction']
        ml_conf = xgb_prediction.get('probability', 0.0)

    # Rule-based signal (15-school Log-Odds fusion only)
    rule_dir = result['ensemble_signal']
    rule_conf = result['ensemble_confidence']

    # Always record shadow fields for DuckDB logging
    result['shadow_rule_dir'] = rule_dir
    result['shadow_rule_conf'] = rule_conf
    result['shadow_ml_dir'] = ml_dir
    result['shadow_ml_conf'] = ml_conf

    if USE_ML_AS_MASTER:
        # ML as strict post-filter: rules say buy AND ML confirms → buy
        if rule_dir == 'bullish':
            if ml_dir != 'bullish' or ml_conf < 0.55:
                result['ensemble_signal'] = 'neutral'
                result['ensemble_confidence'] = 0.0
                result['master_source'] = 'ml_filtered'
            else:
                result['master_source'] = 'ml_confirmed'
        else:
            result['master_source'] = 'rule'
    else:
        # Shadow mode: rules execute, ML prediction is logged-only (no effect)
        result['ensemble_signal'] = rule_dir
        result['ensemble_confidence'] = rule_conf
        result['master_source'] = 'rule'

    result = apply_confidence_calibration(result)
    return result


def _auto_train_wf_xgb(wf_model) -> None:
    """
    Auto-train Walk-Forward XGBoost from CLEAN ML panel dataset.

    CRITICAL: Uses ml_features.parquet (full-universe panel data),
    NOT trades_df (selection-biased execution data).
    """
    try:
        # Cache gate: skip if global model already loaded
        from ml_feature_pipeline import _GLOBAL_XGB_MODEL
        if _GLOBAL_XGB_MODEL is not None and _GLOBAL_XGB_MODEL._models:
            trained = _GLOBAL_XGB_MODEL
        else:
            import os
            parquet_path = os.path.join(os.path.dirname(__file__), 'data', 'ml_features.parquet')
            if not os.path.exists(parquet_path):
                return
            from ml_feature_pipeline import load_from_parquet, train_xgb_from_parquet
            df_panel = load_from_parquet(parquet_path)
            if len(df_panel) < 100:
                return
            trained = train_xgb_from_parquet(parquet_path, window_size=252, step_size=21)

        if trained._models:
            wf_model._models = trained._models
            wf_model._model_dates = trained._model_dates
            wf_model._last_train_date = trained._last_train_date
            wf_model.feature_names = trained.feature_names
    except Exception:
        pass


def _auto_train_xgb(model) -> None:
    """Auto-train XGBoost from backtest trades in the database."""
    try:
        import duckdb, json
        from config import DB_PATH
        conn = duckdb.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""SELECT net_pnl_pct, school_votes_json FROM backtest_trades
                     WHERE school_votes_json IS NOT NULL AND net_pnl_pct IS NOT NULL
                     ORDER BY entry_date DESC LIMIT 2000""")
        rows = c.fetchall()
        conn.close()
        if len(rows) < 50:
            return  # Not enough data

        X_list, y_list = [], []
        for pnl_pct, votes_json in rows:
            if isinstance(votes_json, str):
                try:
                    votes = json.loads(votes_json)
                except json.JSONDecodeError:
                    continue
            else:
                votes = votes_json
            features = model.build_features(votes).flatten()
            X_list.append(features)
            y_list.append(1 if (pnl_pct or 0) > 0 else 0)

        if len(X_list) >= 50:
            X = np.array(X_list)
            y = np.array(y_list)
            model.fit(X, y)
    except Exception:
        pass


# ============================================================
# Bayesian refinement of ensemble
# ============================================================

def _bayesian_ensemble_refinement(school_signals):
    """Bayesian refinement: HDI + ROPE + shrinkage for school-level scores."""
    if not _BAYES_REFINEMENT_AVAILABLE:
        return {
            'hdi_low': 0.0, 'hdi_high': 0.0, 'hdi_width': 0.0,
            'rope_result': 'unavailable', 'rope_decision_type': 'unavailable',
            'calibration': {}, 'bayes_direction': 'neutral', 'bayes_confidence': 0.0,
            'shrinkage_factors': {}, 'most_shrunk': 'N/A',
        }

    scores = np.array([s['score'] for s in school_signals.values()])
    confidences = np.array([s['confidence'] for s in school_signals.values()])
    names = list(school_signals.keys())

    rng = np.random.default_rng(42)
    n_schools = len(scores)
    samples = np.zeros(1000)
    for i in range(1000):
        w = rng.dirichlet(np.ones(n_schools) * 0.5)
        samples[i] = np.sum(w * scores)

    hdi_low, hdi_high = compute_hdi(samples, 0.95)
    hdi_width = hdi_high - hdi_low

    rope_result, _, _ = rope_decision(samples, rope=(-0.05, 0.05), cred_mass=0.95)

    std_errors = np.maximum(0.05, 1.0 - np.array(confidences))
    shrunk_scores, shrink_factors = james_stein_shrinkage(scores, std_errors)
    shrunk_mean = float(np.mean(shrunk_scores))

    if rope_result == 'reject_null' and shrunk_mean > 0:
        bayes_dir = 'bullish'
    elif rope_result == 'reject_null' and shrunk_mean < 0:
        bayes_dir = 'bearish'
    elif rope_result == 'accept_null':
        bayes_dir = 'neutral'
    else:
        bayes_dir = 'bullish' if shrunk_mean > 0 else ('bearish' if shrunk_mean < 0 else 'neutral')

    bayes_conf = min(0.95, 1.0 - hdi_width / max(abs(shrunk_mean) + 0.01, 0.01) * 0.3)

    return {
        'bayes_direction': bayes_dir,
        'bayes_confidence': round(bayes_conf, 3),
        'hdi_low': round(hdi_low, 4),
        'hdi_high': round(hdi_high, 4),
        'hdi_width': round(hdi_width, 4),
        'rope_decision': rope_result,
        'shrunk_ensemble_score': round(shrunk_mean, 4),
        'shrinkage_factors': {name: round(float(f), 3) for name, f in zip(names, shrink_factors)},
        'ensemble_samples_mean': round(float(np.mean(samples)), 4),
    }


def compute_expert_ensemble_bayesian(indicators: Dict) -> Dict:
    """Bayesian-enhanced ensemble using school-level experts."""
    result = compute_expert_ensemble(indicators)
    bayes = result['bayesian_refinement']
    new_dir = bayes['bayes_direction']
    old_dir = result['ensemble_signal']

    if new_dir != old_dir and new_dir != 'neutral':
        # Swap reasons: old supporting become opposing, rebuild supporting from bayes-aligned schools
        old_supporting = result.get('supporting_reasons', [])
        old_opposing = result.get('opposing_reasons', [])
        result['supporting_reasons'] = old_opposing[:8] if old_opposing else ['贝叶斯优化后修正']
        result['opposing_reasons'] = old_supporting[:5] if old_supporting else ['原集成方向推翻']

    result['ensemble_signal'] = new_dir
    result['ensemble_confidence'] = bayes['bayes_confidence']
    return result


# ============================================================
# Confidence Calibration
# ============================================================

def apply_confidence_calibration(ensemble_result: Dict) -> Dict:
    """Shrink ensemble confidence using historical calibration factors."""
    try:
        from backtest_feedback import get_calibration_factors
        cal = get_calibration_factors()
    except Exception:
        return ensemble_result

    if not cal.get('calibrated'):
        return ensemble_result

    cf = cal['calibration_factor']
    if cf >= 0.99:
        return ensemble_result

    old_conf = ensemble_result.get('ensemble_confidence', 0.5)
    new_conf = round(old_conf * cf, 3)
    ensemble_result['ensemble_confidence'] = max(0.15, new_conf)
    ensemble_result['calibration_factor'] = cf
    ensemble_result['calibration_note'] = cal.get('note', '')
    # Note: per-school confidences are NOT uniformly scaled — each school's
    # confidence reflects its own evidence quality, not a global multiplier.

    return ensemble_result


# ============================================================
# Multi-Timeframe Coordination
# ============================================================

def compute_timeframe_coordination(
    daily_indicators: Dict,
    hourly_indicators: Optional[Dict] = None,
    weekly_trend: str = '震荡',
    daily_ensemble: Optional[Dict] = None,
) -> Dict:
    """MARL-inspired multi-timeframe coordination with 15-school ensemble.

    Args:
        daily_ensemble: pre-computed daily ensemble result. If None, computed here.
    """
    rounds = []

    if daily_ensemble is None:
        daily_ensemble = compute_expert_ensemble(daily_indicators)
    rounds.append({
        'round': 1,
        'timeframe': '日线',
        'intention': daily_ensemble['ensemble_signal'],
        'confidence': daily_ensemble['ensemble_confidence'],
    })

    hourly_adj = 1.0
    hourly_detail = '无小时线数据'
    if hourly_indicators is not None:
        hourly_ensemble = compute_expert_ensemble(hourly_indicators)
        rounds.append({
            'round': 2,
            'timeframe': '60分钟',
            'intention': hourly_ensemble['ensemble_signal'],
            'confidence': hourly_ensemble['ensemble_confidence'],
        })

        if hourly_ensemble['ensemble_signal'] == daily_ensemble['ensemble_signal']:
            hourly_detail = '60分钟确认日线方向'
            hourly_adj = 1.05
        elif hourly_ensemble['ensemble_signal'] == 'neutral':
            hourly_detail = '60分钟无明确方向，等待'
            hourly_adj = 0.85
        else:
            hourly_detail = '60分钟与日线冲突，等待'
            hourly_adj = 0.65

    # Weekly validation
    weekly_adj = 1.0
    weekly_detail = ''
    daily_dir = daily_ensemble['ensemble_signal']
    if '多' in weekly_trend or 'bull' in weekly_trend.lower():
        if daily_dir == 'bullish':
            weekly_detail = '周线顺势(多头)，可适度放大仓位'
            weekly_adj = 1.15
        elif daily_dir == 'bearish':
            weekly_detail = '周线逆势(多头)，做空为逆大势操作'
            weekly_adj = 0.60
        else:
            weekly_detail = '周线多头背景'
            weekly_adj = 1.0
    elif '空' in weekly_trend or 'bear' in weekly_trend.lower():
        if daily_dir == 'bearish':
            weekly_detail = '周线顺势(空头)，可适度放大仓位'
            weekly_adj = 1.15
        elif daily_dir == 'bullish':
            weekly_detail = '周线逆势(空头)，做多为逆大势操作'
            weekly_adj = 0.60
        else:
            weekly_detail = '周线空头背景'
            weekly_adj = 1.0
    else:
        weekly_detail = '周线震荡，以日线信号为主'
        weekly_adj = 0.75

    rounds.append({
        'round': 3,
        'timeframe': '周线',
        'intention': daily_dir if weekly_adj >= 1.0 else ('neutral' if weekly_adj > 0.7 else _opposite_dir(daily_dir)),
        'confidence': round(0.5 * weekly_adj, 3),
        'trend': weekly_trend,
        'adjustment': round(weekly_adj, 2),
    })

    final_mult = hourly_adj * weekly_adj
    adj_conf = daily_ensemble['ensemble_confidence'] * final_mult
    adj_conf = np.clip(adj_conf, 0.20, 0.95)

    if final_mult >= 1.1:
        coord_quality = 'strong'
    elif final_mult >= 0.8:
        coord_quality = 'moderate'
    elif final_mult >= 0.6:
        coord_quality = 'weak'
    else:
        coord_quality = 'conflict'

    return {
        'coordination_quality': coord_quality,
        'adjusted_confidence': round(float(adj_conf), 3),
        'confidence_multiplier': round(final_mult, 3),
        'rounds': rounds,
        'hourly_detail': hourly_detail,
        'weekly_detail': weekly_detail,
        'final_signal': daily_ensemble['ensemble_signal'],
        'expert_votes': daily_ensemble['votes'],
        'diversity': daily_ensemble['diversity'],
    }


# ============================================================
# Backward compatibility aliases (for local_trade_plan.py etc.)
# ============================================================

# Old expert domain labels mapped to new schools for backward compat
OLD_EXPERT_LABELS = {
    'expert_a': '趋势结构',  # now in school_chanlun + school_classical + school_busch
    'expert_b': '动量震荡',  # now in school_classical
    'expert_c': '量价资金',  # now in school_classical + school_busch
    'expert_d': '形态价格',  # now in school_classical
    'expert_e': '风险波动',  # now in school_risk
}

# The old EXPERT_DOMAINS and EXPERT_COMPUTE_FUNCTIONS are no longer used.
# For backward compatibility, map old expert names to school equivalents:
OLD_TO_SCHOOL_MAP = {
    'expert_a': 'school_chanlun',   # closest match: trend/structure
    'expert_b': 'school_classical',  # closest match: momentum
    'expert_c': 'school_busch',      # closest match: volume/capital
    'expert_d': 'school_classical',  # closest match: pattern
    'expert_e': 'school_risk',       # closest match: risk
}
