# indicators_v2.py
import talib
import numpy as np
from advanced_indicators import (
    chanlun_analysis,
    td_sequential,
    dynamic_support_resistance,
    KMeansPattern,
    momentum_indicators,
    volume_price_analysis,
    dmi_analysis,
    candlestick_patterns,    # K线形态识别
    dma_analysis,            # DMA指标
    obv_analysis,            # 增强OBV
    volume_pattern_analysis, # 量价形态增强
    ew_enhanced_analysis,    # 艾略特波浪（含增强12项）
    compute_realized_volatility,  # 已实现波动率
    har_model,                    # HAR波动率预测
    bebe_volatility_decomposition,  # BEGE好坏环境波动率分解
    volatility_trading_signals,   # 波动率交易信号
    bayesian_confidence_fusion,   # 贝叶斯置信度融合
    multi_signal_bayesian_fusion, # 多信号贝叶斯融合
    vr_indicator,                 # VR容量比率
    volume_price_algebra,         # Busch量价代数
    busch_2560_strategy,          # 2560战法
    market_phase_detection,       # 市场四阶段
    busch_dmi_analysis,           # Busch DMI增强
    comprehensive_volume_analysis, # 综合量价分析
    volume_pattern_classify,      # 量能形态分类
    volume_9_relations,           # 九大量价关系(完整版)
    smart_money_detection,        # 主力资金检测
    granville_8_complete,         # 格兰维尔八准则(完整版)
    velez_nrb_detection,          # NRB检测
    tang_jiato_jiaya,             # 唐能通价托/价压
    tang_sanjincha_sansicha,      # 唐能通三金叉/三死叉
    tang_paodao_houdu,            # 唐能通跑道厚度
    tang_2T_1L_target,            # 唐能通2T-1L目标位
    tang_33_filter,               # 唐能通三三过滤制
    tang_laoyatou,                # 唐能通老鸭头形态
    livermore_pivotal_points,     # 利弗莫尔关键点
    livermore_danger_signal,      # 利弗莫尔危险信号
    chanlun_lidubi,               # 解缠论力度比
    chanlun_zhongshu_cg_drift,    # 解缠论中枢重心偏移
    chanlun_turn_score,           # 解缠论5条件拐点评分
    chanlun_intra_stroke_divergence,  # 解缠论笔内背离
    top_escape_signals,           # 逃顶十二招
    distribution_days_count,      # CAN SLIM分配日计数
)


def _verify_sr_levels(sr_levels, close, volume, vol_ma, sr_type):
    """成交量验重支撑/阻力位：近期价格触及该位时量能>均值则通过验证"""
    verified = []
    if not sr_levels or len(close) < 20:
        return verified
    for level, strength in sr_levels:
        if level is None or level <= 0 or not np.isfinite(level):
            continue
        for i in range(max(0, len(close) - 20), len(close)):
            if not np.isfinite(close[i]):
                continue
            if abs(close[i] - level) / level < 0.015:
                if np.isfinite(volume[i]) and volume[i] > vol_ma and vol_ma > 0:
                    verified.append({'level': level, 'strength': min(strength + 15, 100),
                                     'vol_ratio': round(float(volume[i] / vol_ma), 2)})
                    break
    return verified[:3]


def compute_all_indicators_v2(df_daily, df_hourly, df_weekly=None, symbol=None):
    if df_daily is None or df_daily.empty:
        return None

    # Validate required columns exist
    required_cols = ['open', 'close', 'high', 'low', 'volume']
    missing = [c for c in required_cols if c not in df_daily.columns]
    if missing:
        raise KeyError(f"日线数据缺少必要列: {missing}，现有列: {list(df_daily.columns)}")

    close_d = df_daily['close'].astype(float).values
    high_d = df_daily['high'].astype(float).values
    low_d = df_daily['low'].astype(float).values
    vol_d = df_daily['volume'].astype(float).values

    n_bars = len(close_d)
    if n_bars < 30:
        # Insufficient bars for meaningful computation beyond basic indicators
        pass  # Let TA-Lib handle NaN; downstream code is NaN-aware

    # ---- 经典指标 (NaN-guarded) ----
    def _last_valid(arr, default=0.0):
        """Return last element of TA-Lib result, replacing NaN with default."""
        if arr is None or len(arr) == 0:
            return default
        v = arr[-1]
        return default if (isinstance(v, float) and np.isnan(v)) or not np.isfinite(v) else float(v)

    ma5 = _last_valid(talib.SMA(close_d, 5))
    ma20 = _last_valid(talib.SMA(close_d, 20))
    ma60 = _last_valid(talib.SMA(close_d, 60))
    upper, middle, lower = talib.BBANDS(close_d, 20, 2, 2)
    bb_upper = _last_valid(upper)
    bb_mid = _last_valid(middle)
    bb_lower = _last_valid(lower)
    macd, macd_signal, macd_hist = talib.MACD(close_d)
    macd_val = _last_valid(macd)
    macd_sig_val = _last_valid(macd_signal)
    macd_hist_val = _last_valid(macd_hist)
    rsi = _last_valid(talib.RSI(close_d, 14))

    # ---- 高级指标 ----
    # Per-call KMeans instantiation for thread safety
    km_model = KMeansPattern(n_clusters=6)
    td = td_sequential(df_daily)
    chan = chanlun_analysis(df_daily, df_60min=df_hourly)
    dyn_sup, dyn_res = dynamic_support_resistance(df_daily)
    cluster = km_model.predict(df_daily) if km_model.fitted else -1
    mom = momentum_indicators(df_daily)
    vol = volume_price_analysis(df_daily)      # 量价
    dmi = dmi_analysis(df_daily)               # DMI
    pat = candlestick_patterns(df_daily)       # K线形态
    dma = dma_analysis(df_daily)               # DMA
    obv2 = obv_analysis(df_daily)              # 增强OBV
    vol_pat = volume_pattern_analysis(df_daily)  # 量价形态增强
    rv = compute_realized_volatility(df_daily)    # 已实现波动率
    har = har_model(df_daily)                     # HAR波动率预测
    bebe = bebe_volatility_decomposition(df_daily)  # BEGE好坏环境波动率分解
    vol_sig = volatility_trading_signals(df_daily, rv_result=rv, har_result=har, bebe_result=bebe)  # 波动率交易信号
    ew_enh = ew_enhanced_analysis(df_daily)        # 增强艾略特波浪(12项)
    vr = vr_indicator(df_daily)                    # VR容量比率
    vpa = volume_price_algebra(df_daily)           # Busch量价代数
    b2560 = busch_2560_strategy(df_daily)          # 2560战法
    mphase = market_phase_detection(df_daily)      # 市场四阶段
    bdmi = busch_dmi_analysis(df_daily)            # Busch DMI增强
    cva = comprehensive_volume_analysis(df_daily)  # 综合量价分析
    vc = volume_pattern_classify(df_daily)         # 量能形态分类
    nrb = velez_nrb_detection(df_daily)            # NRB检测
    v9r = volume_9_relations(df_daily)             # 九大量价关系
    smm = smart_money_detection(df_daily)          # 主力资金检测
    g8 = granville_8_complete(df_daily)            # 格兰维尔八准则
    # 唐能通系统
    tang_jj = tang_jiato_jiaya(high_d, low_d, close_d)               # 价托/价压
    tang_sc = tang_sanjincha_sansicha(close_d, vol_d)                 # 三金叉/三死叉
    tang_ph = tang_paodao_houdu(close_d)                              # 跑道厚度
    tang_2t = tang_2T_1L_target(close_d[-1], max(high_d[-60:]), min(low_d[-60:]))  # 2T-1L目标
    tang_33 = tang_33_filter(close_d, high_d, low_d, max(high_d[-60:]))  # 三三过滤
    tang_lyt = tang_laoyatou(high_d, low_d, close_d)                  # 老鸭头
    # 利弗莫尔战法
    atr_arr = talib.ATR(high_d, low_d, close_d, 14)
    liv_pp = livermore_pivotal_points(high_d, low_d, close_d, atr_arr)  # 关键点
    liv_ds = livermore_danger_signal(df_daily['open'].values, high_d, low_d, close_d, atr_arr)  # 危险信号
    # 解缠论增强
    cl_lb = chanlun_lidubi(high_d, low_d, close_d)                    # 力度比
    cl_cg = chanlun_zhongshu_cg_drift(close_d)                        # 中枢重心偏移
    cl_ts = chanlun_turn_score(high_d, low_d, close_d, vol_d)         # 拐点评分
    cl_isd = chanlun_intra_stroke_divergence(high_d, low_d, close_d)  # 笔内背离
    # 逃顶检测
    top_esc = top_escape_signals(high_d, low_d, close_d, df_daily['open'].values, vol_d, atr_arr)  # 逃顶信号
    dist_days = distribution_days_count(close_d, vol_d)               # 分配日计数

    # ---- 周线趋势 ----
    weekly_trend = "震荡"
    if df_weekly is not None and len(df_weekly) >= 20:
        close_w = df_weekly['close'].astype(float).values
        ma20_w = talib.SMA(close_w, 20)[-1]
        ma60_w = talib.SMA(close_w, 60)[-1]
        if not np.isnan(ma20_w) and not np.isnan(ma60_w):
            if ma20_w > ma60_w:
                weekly_trend = "多头"
            elif ma20_w < ma60_w:
                weekly_trend = "空头"

    # ---- 多指标验证关键位（简例） ----
    vol_ma_verify = np.mean(vol_d[-21:])
    verified_support = _verify_sr_levels(dyn_sup, close_d, vol_d, vol_ma_verify, 'support')
    verified_resistance = _verify_sr_levels(dyn_res, close_d, vol_d, vol_ma_verify, 'resistance')

    # ---- 贝叶斯置信度融合（在所有指标计算完成后调用） ----
    # 先构建部分指标快照供贝叶斯分析使用（使用.get()防御缺失键）
    _partial_indicators = {
        'ma5': ma5, 'ma20': ma20, 'ma60': ma60,
        'momentum_resonance': mom.get('momentum_resonance', 0.0) if mom else 0.0,
        'momentum_direction': mom.get('momentum_direction', 'neutral') if mom else 'neutral',
        'vol_price_resonance': vol.get('vol_price_resonance', 0.0) if vol else 0.0,
        'granville_signal': vol.get('granville_signal') if vol else None,
        'vol_stacking': vol_pat.get('vol_stacking', False) if vol_pat else False,
        'high_vol_stagnation': vol_pat.get('high_vol_stagnation', False) if vol_pat else False,
        'low_vol_pullback': vol_pat.get('low_vol_pullback', False) if vol_pat else False,
        'dmi_adx': dmi.get('adx', 15) if dmi else 15,
        'dmi_di_direction': dmi.get('di_direction', 'neutral') if dmi else 'neutral',
        'pattern_recent': pat.get('recent_signals', '') if pat else '',
        'har_direction': har.get('har_direction', 'stable') if har else 'stable',
        'rv_level': rv.get('rv_level', 'normal') if rv else 'normal',
        'dma_signal': dma.get('dma_signal', 'neutral') if dma else 'neutral',
        'obv_divergence': obv2.get('obv_divergence', 'none') if obv2 else 'none',
        'obv_signal': obv2.get('obv_signal', 'neutral') if obv2 else 'neutral',
    }
    bayes = bayesian_confidence_fusion(_partial_indicators, df_daily)

    # ---- 返回所有指标 ----
    return {
        '_symbol': symbol,
        'ma5': ma5, 'ma20': ma20, 'ma60': ma60,
        'bb_upper': bb_upper, 'bb_mid': bb_mid, 'bb_lower': bb_lower,
        'macd_dif': macd_val, 'macd_dea': macd_sig_val, 'macd_hist': macd_hist_val,
        'rsi': rsi,
        'td_count': td['buy_setup_count'] if td['direction'] == 'buy' else td['sell_setup_count'],
        'td_direction': td['direction'],
        'td_completed': td['completed'],
        'chanlun_status': chan['status'],
        'chanlun_buy': chan['buy_points'],
        'chanlun_sell': chan['sell_points'],
        # 缠论（完整版：笔状态机/背驰/防狼术/中枢监视/多级别联立）
        'chanlun_stroke_state': chan['stroke_state']['state'],
        'chanlun_stroke_label': chan['stroke_state']['state_label'],
        'chanlun_stroke_direction': chan['stroke_state']['direction'],
        'chanlun_divergence_type': chan['divergence']['type'],
        'chanlun_divergence_strength': chan['divergence']['strength'],
        'chanlun_divergence_detail': chan['divergence']['details'],
        'chanlun_wolf_signal': chan['wolf_defense']['signal'],
        'chanlun_wolf_position': chan['wolf_defense']['macd_position'],
        'chanlun_wolf_days': chan['wolf_defense']['days_below_zero'],
        'chanlun_trend_type': chan['trend_type']['type'],
        'chanlun_trend_direction': chan['trend_type']['direction'],
        'chanlun_zn_value': chan['zn_monitor']['zn_value'],
        'chanlun_zn_position': chan['zn_monitor']['position'],
        'chanlun_zn_pattern': chan['zn_monitor']['pattern'],
        'chanlun_fractal_type': chan['fractal_recent']['type'] if chan['fractal_recent'] else '',
        'chanlun_fractal_price': chan['fractal_recent']['price'] if chan['fractal_recent'] else 0,
        'chanlun_segment_count': chan['segment_count'],
        'chanlun_multi_tf_alignment': chan['multi_tf']['alignment'],
        'chanlun_multi_tf_recommendation': chan['multi_tf']['recommendation'],
        'chanlun_bi_count': chan['bi_count'],
        # 线段破坏
        'chanlun_segment_destructions': [s.get('destruction', '') for s in chan.get('_segments', [])],
        'chanlun_last_segment_destruction': chan['_segments'][-1].get('destruction', '') if chan.get('_segments') else '',
        # 中枢扩展
        'chanlun_zs_expanded': chan['zhongshu_expansion']['expanded'],
        'chanlun_zs_upgraded': chan['zhongshu_expansion']['upgraded_zhongshu'] is not None,
        'chanlun_zs_upgraded_level': chan['zhongshu_expansion']['upgraded_zhongshu']['level'] if chan['zhongshu_expansion']['upgraded_zhongshu'] else '',
        # 小转大
        'chanlun_xzd_active': chan['xiao_zhuan_da']['xzd_active'],
        'chanlun_xzd_direction': chan['xiao_zhuan_da']['xzd_direction'] or '',
        'chanlun_xzd_confidence': chan['xiao_zhuan_da']['xzd_confidence'],
        'chanlun_xzd_reasons': '; '.join(chan['xiao_zhuan_da']['xzd_reasons']) if chan['xiao_zhuan_da']['xzd_reasons'] else '',
        'dynamic_support': dyn_sup[:3],
        'dynamic_resistance': dyn_res[:3],
        'verified_support': verified_support,
        'verified_resistance': verified_resistance,
        'kmeans_cluster': cluster,
        'weekly_trend': weekly_trend,
        'current_price': close_d[-1],
        # 动量
        'roc_short': mom['roc_short'],
        'roc_long': mom['roc_long'],
        'cci_short': mom['cci_short'],
        'cci_long': mom['cci_long'],
        'momentum_state_short': mom['momentum_state_short'],
        'momentum_state_long': mom['momentum_state_long'],
        'momentum_resonance': mom['momentum_resonance'],
        'momentum_direction': mom['momentum_direction'],
        # 量价
        'vol_ratio': vol['vol_ratio'],
        'vol_price_resonance': vol['vol_price_resonance'],
        'obv_trend': vol['obv_trend'],
        'vol_verified_support': vol['vol_verified_support'],
        'vol_verified_resistance': vol['vol_verified_resistance'],
        'vol_type': vol['vol_type'],
        'vol_drought': vol['vol_drought'],
        'vol_spike': vol['vol_spike'],
        'vol_drought_ratio': vol['vol_drought_ratio'],
        'vol_spike_ratio': vol['vol_spike_ratio'],
        'granville_signal': vol.get('granville_signal'),
        # 量价形态增强
        'vol_stacking': vol_pat['vol_stacking'],
        'vol_stacking_days': vol_pat['vol_stacking_days'],
        'high_vol_stagnation': vol_pat['high_vol_stagnation'],
        'low_vol_pullback': vol_pat['low_vol_pullback'],
        'volume_breakout': vol_pat['volume_breakout'],
        'vol_divergence_detail_10d': vol_pat['vol_divergence_detail'],
        'vol_trend': vol_pat['vol_trend'],
        'price_range_5d_pct': vol_pat['price_range_5d_pct'],
        # DMI
        'dmi_adx': dmi['adx'],
        'dmi_adx_trend': dmi['adx_trend'],
        'dmi_pdi': dmi['pdi'],
        'dmi_mdi': dmi['mdi'],
        'dmi_di_direction': dmi['di_direction'],
        # K线形态
        'pattern_dominant': pat['dominant_pattern'],
        'pattern_recent': pat['recent_signals'],
        'pattern_latest_fractal': pat['latest_fractal'],
        'pattern_latest_reversal': pat['latest_reversal'],
        'pattern_top_fractal_n': pat['counts'].get('top_fractal', 0),
        'pattern_bottom_fractal_n': pat['counts'].get('bottom_fractal', 0),
        'pattern_evening_star_n': pat['counts'].get('evening_star', 0),
        'pattern_morning_star_n': pat['counts'].get('morning_star', 0),
        # DMA
        'dma': dma['dma'],
        'dma_ama': dma['ama'],
        'dma_diff': dma['dma_diff'],
        'dma_signal': dma['dma_signal'],
        'dma_trend': dma['dma_trend'],
        # 增强OBV
        'obv_divergence': obv2.get('obv_divergence', 'none'),
        'obv_signal': obv2.get('obv_signal', 'neutral'),
        'obv_price_direction': obv2.get('obv_price_direction', 'neutral'),
        'obv_direction': obv2.get('obv_direction', 'flat'),
        # 艾略特波浪
        'ew_pattern': ew_enh['wave_pattern'],
        'ew_current_wave': ew_enh['current_wave'],
        'ew_structure': ew_enh['wave_structure'],
        'ew_rules_valid': ew_enh['rules_valid'],
        'ew_rule_violations': ew_enh['rule_violations'],
        'ew_fib_ratios': ew_enh['fib_ratios'],
        'ew_next_target': ew_enh['next_target'],
        'ew_next_support': ew_enh['next_support'],
        'ew_confidence': ew_enh['confidence'],
        'ew_trade_signal': ew_enh['trade_setup']['signal'],
        'ew_trade_setup': ew_enh['trade_setup']['setup_type'],
        'ew_trade_entry': ew_enh['trade_setup']['entry_zone'],
        'ew_trade_stop': ew_enh['trade_setup']['stop_loss'],
        'ew_trade_target': ew_enh['trade_setup']['target'],
        'ew_trade_rationale': ew_enh['trade_setup']['rationale'],
        'ew_pivot_count': ew_enh['pivot_count'],
        'ew_trend': ew_enh['trend_direction'],
        # 已实现波动率
        'rv_parkinson': rv['rv_parkinson'],
        'rv_garman_klass': rv['rv_garman_klass'],
        'rv_yang_zhang': rv['rv_yang_zhang'],
        'rv_composite': rv['rv_composite'],
        'rv_trend': rv['rv_trend'],
        'rv_percentile': rv['rv_percentile'],
        'rv_level': rv['rv_level'],
        # HAR波动率预测
        'har_rv_daily': har['har_rv_daily'],
        'har_rv_weekly': har['har_rv_weekly'],
        'har_rv_monthly': har['har_rv_monthly'],
        'har_forecast': har['har_forecast'],
        'har_direction': har['har_direction'],
        'har_r_squared': har['har_r_squared'],
        'har_decay': har['har_decay'],
        'har_beta_d': har['har_beta_d'],
        'har_beta_w': har['har_beta_w'],
        'har_beta_m': har['har_beta_m'],
        # 波动率交易信号
        'vol_signal': vol_sig['vol_signal'],
        'vol_advice': vol_sig['vol_advice'],
        'vol_position_mult': vol_sig['vol_position_mult'],
        'vol_position_advice': vol_sig['vol_position_advice'],
        'vol_adaptive_stop_pct': vol_sig['vol_adaptive_stop_pct'],
        'vol_adaptive_stop_price': vol_sig['vol_adaptive_stop_price'],
        'vol_entry_quality': vol_sig['vol_entry_quality'],
        'vol_atr14': vol_sig['vol_atr14'],
        'vol_atr_pct': vol_sig['vol_atr_pct'],
        'vol_bebe_regime': vol_sig['vol_bebe_regime'],
        'vol_bebe_vrp': vol_sig['vol_bebe_vrp'],
        'vol_bebe_asymmetry': vol_sig['vol_bebe_asymmetry'],
        # BEGE好坏环境波动率分解
        'bebe_good_vol': bebe['bebe_good_vol'],
        'bebe_bad_vol': bebe['bebe_bad_vol'],
        'bebe_total_vol': bebe['bebe_total_vol'],
        'bebe_good_bad_ratio': bebe['bebe_good_bad_ratio'],
        'bebe_regime': bebe['bebe_regime'],
        'bebe_skew': bebe['bebe_skew'],
        'bebe_vrp_proxy': bebe['bebe_vrp_proxy'],
        'bebe_vrp_signal': bebe['bebe_vrp_signal'],
        'bebe_good_vol_trend': bebe['bebe_good_vol_trend'],
        'bebe_bad_vol_trend': bebe['bebe_bad_vol_trend'],
        'bebe_vol_asymmetry': bebe['bebe_vol_asymmetry'],
        # σ-LSTM波动率特征（Rodikov & Antulov-Fantulin 2022）
        'sigma_lstm_cell': bebe.get('bebe_sigma_lstm', {}).get('sigma_lstm_cell', 0),
        'sigma_lstm_hidden': bebe.get('bebe_sigma_lstm', {}).get('sigma_lstm_hidden', 0),
        'sigma_lstm_forget': bebe.get('bebe_sigma_lstm', {}).get('sigma_lstm_forget', 0.5),
        'sigma_lstm_regime': bebe.get('bebe_sigma_lstm', {}).get('sigma_lstm_regime', 'no_data'),
        'sigma_lstm_h_ratio': bebe.get('bebe_sigma_lstm', {}).get('sigma_lstm_h_ratio', 1.0),
        # 贝叶斯置信度融合
        'bayes_fused_posterior': bayes['bayes_fused_posterior'],
        'bayes_fused_signal': bayes['bayes_fused_signal'],
        'bayes_fused_entropy': bayes['bayes_fused_entropy'],
        'bayes_multi_posterior': bayes['bayes_multi_signal_posterior'],
        'bayes_beta_posterior': bayes['bayes_beta_trend_posterior'],
        'bayes_dimensions_active': bayes['bayes_dimensions_active'],
        'bayes_dimension_contributions': bayes['bayes_dimension_contributions'],
        'bayes_entropy_high': bayes['bayes_entropy_high'],
        # 增强艾略特波浪
        'ew_enhanced': ew_enh.get('ew_enhanced', {}),
        'ew_alternation_score': ew_enh.get('ew_enhanced', {}).get('alternation', {}).get('alternation_score', 0),
        'ew_alternation_valid': ew_enh.get('ew_enhanced', {}).get('alternation', {}).get('alternation_valid', False),
        'ew_extension_wave': ew_enh.get('ew_enhanced', {}).get('extension', {}).get('extended_wave'),
        'ew_extension_magnitude': ew_enh.get('ew_enhanced', {}).get('extension', {}).get('extension_magnitude', 1.0),
        'ew_corrective_type': ew_enh.get('ew_enhanced', {}).get('corrective_pattern', {}).get('corrective_type', 'unknown'),
        'ew_is_diagonal': ew_enh.get('ew_enhanced', {}).get('diagonal', {}).get('is_diagonal', False),
        'ew_diagonal_type': ew_enh.get('ew_enhanced', {}).get('diagonal', {}).get('type'),
        'ew_personality_score': ew_enh.get('ew_enhanced', {}).get('personality', {}).get('personality_score', 0.5),
        'ew_channel_upper': ew_enh.get('ew_enhanced', {}).get('channel', {}).get('channel_upper'),
        'ew_channel_lower': ew_enh.get('ew_enhanced', {}).get('channel', {}).get('channel_lower'),
        'ew_invalidation_levels': ew_enh.get('ew_enhanced', {}).get('invalidation_levels', {}),
        'ew_time_projection': ew_enh.get('ew_enhanced', {}).get('time_ratios', {}).get('time_projection'),
        # VR容量比率
        'vr_value': vr['vr_value'],
        'vr_zone': vr['vr_zone'],
        'vr_signal': vr['vr_signal'],
        'vr_divergence': vr['vr_divergence'],
        # Busch量价代数
        'vpa_formula': vpa['vpa_formula'],
        'vpa_signal': vpa['vpa_signal'],
        'vpa_interpretation': vpa['vpa_interpretation'],
        'vpa_action': vpa['vpa_action'],
        'vpa_is_extreme_vol': vpa['vpa_is_extreme_vol'],
        # 2560战法
        'b2560_signal': b2560['b2560_signal'],
        'b2560_description': b2560['b2560_description'],
        'b2560_ma25_direction': b2560['b2560_ma25_direction'],
        'b2560_ma_golden_cross': b2560['b2560_ma_golden_cross'],
        'b2560_ma_dead_cross': b2560['b2560_ma_dead_cross'],
        'b2560_vol_golden_cross': b2560['b2560_vol_golden_cross'],
        'b2560_weekly_signal': b2560['b2560_weekly_signal'],
        # 市场四阶段
        'mp_phase': mphase['mp_phase'],
        'mp_confidence': mphase['mp_confidence'],
        'mp_price_position': mphase['mp_price_position'],
        'mp_ret_20d': mphase['mp_ret_20d'],
        'mp_ret_60d': mphase['mp_ret_60d'],
        'mp_vol_ratio': mphase['mp_vol_ratio'],
        'mp_big_trend': mphase['mp_big_trend'],
        'mp_mid_trend': mphase['mp_mid_trend'],
        'mp_small_trend': mphase['mp_small_trend'],
        'mp_trend_credibility': mphase['mp_trend_credibility'],
        # Busch DMI增强
        'bdmi_signal': bdmi['bdmi_signal'],
        'bdmi_description': bdmi['bdmi_description'],
        'bdmi_pdi': bdmi['bdmi_pdi'],
        'bdmi_mdi': bdmi['bdmi_mdi'],
        'bdmi_adx': bdmi['bdmi_adx'],
        'bdmi_mdi_peaking': bdmi['bdmi_mdi_peaking'],
        'bdmi_pdi_rising': bdmi['bdmi_pdi_rising'],
        'bdmi_big_momentum': bdmi['bdmi_big_momentum'],
        # 综合量价分析
        'cva_composite_signal': cva.get('cva_composite_signal', '中性/分歧'),
        'cva_bull_count': cva.get('cva_bull_count', 0),
        'cva_bear_count': cva.get('cva_bear_count', 0),
        # 量能形态分类
        'vc_pattern': vc['vc_pattern'],
        'vc_description': vc['vc_description'],
        'vc_signal': vc['vc_signal'],
        'vc_cv': vc['vc_cv'],
        'vc_vol_ratio': vc['vc_vol_ratio'],
        # NRB检测
        'nrb_signal': nrb['nrb_signal'],
        'nrb_type': nrb['nrb_type'],
        'nrb_description': nrb['nrb_description'],
        # 九大量价关系
        'v9r_id': v9r['relation_id'],
        'v9r_name': v9r['relation_name'],
        'v9r_price_state': v9r['price_state'],
        'v9r_volume_state': v9r['volume_state'],
        'v9r_interpretation': v9r['interpretation'],
        'v9r_action': v9r['action'],
        'v9r_strength': v9r['strength'],
        'v9r_consistency': v9r['consistency'],
        # 主力资金检测
        'sm_mfi': smm['sm_mfi'],
        'sm_mfi_signal': smm['sm_mfi_signal'],
        'sm_vwap_deviation': smm['sm_vwap_deviation'],
        'sm_vwap_signal': smm['sm_vwap_signal'],
        'sm_ad_trend': smm['sm_ad_trend'],
        'sm_ad_divergence': smm['sm_ad_divergence'],
        'sm_smart_signal': smm['sm_smart_signal'],
        'sm_bull_score': smm['sm_bull_score'],
        'sm_bear_score': smm['sm_bear_score'],
        # 格兰维尔八准则
        'g8_active_rule': g8['g8_active_rule'],
        'g8_rule_category': g8['g8_rule_category'] or 'none',
        'g8_description': g8['g8_description'],
        'g8_ma60_rising': g8['g8_ma60_rising'],
        'g8_ma60_slope': g8['g8_ma60_slope'],
        'g8_price_vs_ma60': g8['g8_price_vs_ma60'],
        'g8_deviation_sigma': g8['g8_deviation_sigma'],
        'g8_vol_confirmed': g8['g8_vol_confirmed'],
        'g8_long_term': g8['g8_long_term'],
        'g8_land_volume_zone': g8['g8_land_volume_zone'],
        'g8_land_volume_days': g8['g8_land_volume_days'],
        # 唐能通系统
        'tang_jiato': tang_jj[0], 'tang_jiaya': tang_jj[1],
        'tang_jiato_strength': tang_jj[2], 'tang_jiaya_strength': tang_jj[3],
        'tang_jiato_phase': tang_jj[4], 'tang_jiaya_phase': tang_jj[5],
        'tang_triangle_area': tang_jj[6],
        'tang_golden_cross': tang_sc[0], 'tang_death_cross': tang_sc[1],
        'tang_golden_score': tang_sc[2], 'tang_death_score': tang_sc[3],
        'tang_cross_days': tang_sc[4], 'tang_triple_status': tang_sc[5],
        'tang_runway': tang_ph[0], 'tang_runway_thickness': tang_ph[1],
        'tang_runway_grade': tang_ph[2], 'tang_runway_slope': tang_ph[3],
        'tang_target_2t1l': tang_2t[0], 'tang_target_conservative': tang_2t[1],
        'tang_target_pct': tang_2t[2], 'tang_stop_2t': tang_2t[3],
        'tang_risk_reward': tang_2t[4],
        'tang_33_valid': tang_33[0], 'tang_33_score': tang_33[1],
        'tang_33_days_above': tang_33[2], 'tang_33_pct_above': tang_33[3],
        'tang_laoyatou': tang_lyt[0], 'tang_laoyatou_phase': tang_lyt[1],
        'tang_laoyatou_score': tang_lyt[2], 'tang_laoyatou_neck': tang_lyt[3],
        'tang_laoyatou_head_h': tang_lyt[4], 'tang_laoyatou_mouth_a': tang_lyt[5],
        # 利弗莫尔战法
        'livermore_pivots': liv_pp[0], 'livermore_nearest_pivot': liv_pp[1],
        'livermore_breakout': liv_pp[2], 'livermore_breakout_str': liv_pp[3],
        'livermore_signal': liv_pp[4],
        'livermore_danger': liv_ds[0], 'livermore_danger_signals': liv_ds[1],
        'livermore_danger_level': liv_ds[2], 'livermore_action': liv_ds[3],
        # 解缠论增强
        'cl_power_ratio': cl_lb[0], 'cl_divergence_type': cl_lb[1],
        'cl_divergence_warning': cl_lb[2], 'cl_current_power': cl_lb[3],
        'cl_prior_power': cl_lb[4], 'cl_ratio_level': cl_lb[5],
        'cl_cg_current': cl_cg[0], 'cl_cg_prev': cl_cg[1],
        'cl_cg_drift': cl_cg[2], 'cl_cg_direction': cl_cg[3],
        'cl_cg_trend_str': cl_cg[4], 'cl_cg_consecutive': cl_cg[5],
        'cl_turn_score': cl_ts[0], 'cl_turn_direction': cl_ts[1],
        'cl_turn_conditions': cl_ts[2], 'cl_turn_confidence': cl_ts[3],
        'cl_turn_signal': cl_ts[4],
        'cl_intra_divergence': cl_isd[0], 'cl_intra_div_type': cl_isd[1],
        'cl_intra_div_severity': cl_isd[2],
        # 逃顶检测
        'top_escape_count': top_esc[0], 'top_escape_signals': top_esc[1],
        'top_escape_prob': top_esc[2], 'top_escape_grade': top_esc[3],
        'top_escape_action': top_esc[4],
        'dist_days_count': dist_days[0], 'dist_days_warning': dist_days[2],
        'dist_days_pct': dist_days[3], 'dist_days_stale': dist_days[4],
    }