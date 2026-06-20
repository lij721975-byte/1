# local_trade_plan.py — 从本地量化指标中提取结构化交易计划
import numpy as np
from localization import t
from config import ACCOUNT_EQUITY
from expert_ensemble import compute_expert_ensemble, compute_timeframe_coordination
from advanced_indicators import (
    compute_hdi,
    rope_decision,
    james_stein_shrinkage,
    empirical_bayes_shrinkage,
    sigma_event_check,
    bayesian_bandit_update,
    asymmetric_trading_loss,
    daily_loss_limit_check,
    portfolio_risk_assessment,
    gap_risk_assessment,
)


def is_sellable_today(buy_date):
    """T+1结算约束：今日买入的股票今日不可卖出（A股规则）。

    Args:
        buy_date: 买入日期字符串，格式 'YYYY-MM-DD'，或 None（无持仓）

    Returns:
        bool: True 表示可以卖出，False 表示T+1锁定
    """
    if buy_date is None:
        return True  # 无持仓，无约束
    from datetime import date
    today_str = date.today().isoformat()
    return buy_date != today_str


def _safe_float(v, default=0.0):
    """安全转float，处理字符串/np值"""
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _extract_float_price(s):
    """从类似 '10.50附近(回调)' 的字符串中提取首个数字"""
    if isinstance(s, (int, float)):
        return float(s)
    if not s:
        return None
    import re
    m = re.search(r'[\d]+\.?[\d]*', str(s))
    return float(m.group()) if m else None


def apply_adaptive_fusion_to_local_plan(local_plan, ai_signal_dict, trust_scores=None):
    """
    轻量级：将AI信号与本地计划做自适应融合，原地更新local_plan。

    复用extract_local_trade_plan中的Phase 2逻辑，但不重新提取整个计划。
    仅更新: signal, confidence, fusion_type, alignment, adaptive_fusion
    """
    if local_plan is None:
        return local_plan

    ai_signal = ai_signal_dict.get('signal', None) if ai_signal_dict else None
    if ai_signal not in ('bullish', 'bearish', 'neutral'):
        return local_plan

    ai_confidence = ai_signal_dict.get('confidence', 0.5)
    bayes_signal = local_plan.get('bayes_signal', 'neutral')
    bayes_posterior = local_plan.get('bayes_posterior', 0.5)
    ensemble_info = local_plan.get('ensemble', {})
    ensemble_signal = ensemble_info.get('signal', 'neutral')
    ensemble_confidence = ensemble_info.get('confidence', 0.5)

    # 重新计算本地内部融合（与extract_local_trade_plan的Phase 1一致）
    if bayes_signal == ensemble_signal:
        fused_signal = bayes_signal
        fused_confidence = max(bayes_posterior, ensemble_confidence) * 1.1
    elif bayes_signal == 'neutral' or ensemble_signal == 'neutral':
        if bayes_signal != 'neutral':
            fused_signal = bayes_signal
            fused_confidence = bayes_posterior * 0.85
        elif ensemble_signal != 'neutral':
            fused_signal = ensemble_signal
            fused_confidence = ensemble_confidence * 0.85
        else:
            fused_signal = 'neutral'
            fused_confidence = 0.35
    else:
        fused_signal = bayes_signal
        fused_confidence = bayes_posterior * 0.6
    fused_confidence = min(fused_confidence, 0.95)

    # 获取信任分
    if trust_scores and isinstance(trust_scores, dict):
        w_local = trust_scores.get('fusion_weight_local', 0.5)
        ai_trust = trust_scores.get('ai_trust', 0.5)
        local_trust = trust_scores.get('local_trust', 0.5)
        trust_ratio = trust_scores.get('trust_ratio', 1.0)
    else:
        w_local = 0.5
        ai_trust = 0.5
        local_trust = 0.5
        trust_ratio = 1.0

    w_ai = 1.0 - w_local

    # 对齐状态
    if fused_signal == 'neutral' or ai_signal == 'neutral':
        alignment = '偏离'
        if fused_signal == 'neutral' and ai_signal != 'neutral':
            fused_signal = ai_signal
            fused_confidence = (ai_confidence or 0.5) * 0.75 * (ai_trust / 0.5)
        elif ai_signal == 'neutral' and fused_signal != 'neutral':
            fused_confidence *= 0.85
    elif fused_signal == ai_signal:
        alignment = '一致'
        ai_conf_val = ai_confidence or 0.5
        boost = 1.0 + 0.15 * (ai_trust + local_trust) / 2.0
        fused_confidence = (w_local * fused_confidence + w_ai * ai_conf_val) * boost
    else:
        alignment = '冲突'
        ai_conf_val = ai_confidence or 0.5
        conflict_w_local = 1.0 / (1.0 + np.exp(-3.0 * (trust_ratio - 1.0)))
        if conflict_w_local >= 0.5:
            fused_signal = bayes_signal if bayes_signal != 'neutral' else ensemble_signal
            fused_confidence *= conflict_w_local * 0.8
        else:
            fused_signal = ai_signal
            fused_confidence = ai_conf_val * (1 - conflict_w_local) * 0.8

    fused_confidence = min(fused_confidence, 0.95)

    # 原地更新
    local_plan['signal'] = fused_signal
    local_plan['confidence'] = round(fused_confidence, 4)
    local_plan['signal_cn'] = t(fused_signal)
    local_plan['alignment'] = alignment
    local_plan['adaptive_fusion'] = {
        'alignment': alignment,
        'w_local': round(w_local, 3),
        'w_ai': round(w_ai, 3),
        'ai_trust': round(ai_trust, 3),
        'local_trust': round(local_trust, 3),
        'recommendation': trust_scores.get('recommendation', '') if trust_scores else '',
    }
    # 更新fusion_type以反映AI融合
    local_plan['fusion_type'] = (
        f'{local_plan.get("fusion_type", "?")}+AI{"一致" if alignment == "一致" else "冲突" if alignment == "冲突" else "偏离"}'
    )

    return local_plan


def extract_local_trade_plan(indicators, symbol=None, daily_df=None, trade_results_today=None,
                             ai_signal_dict=None, trust_scores=None,
                             precomputed_ensemble=None, buy_date=None):
    """
    从已计算的123字段指标字典中提取本地量化模型的交易计划

    Args:
        indicators: 指标字典
        daily_df: 日线DataFrame（可选，用于缺口风险评估）
        trade_results_today: 当日交易结果列表（可选，用于日内亏损限额检测）
        ai_signal_dict: AI(DeepSeek)信号字典（可选，用于自适应融合）
        trust_scores: 引擎信任分字典（可选，来自get_engine_trust_scores）
        precomputed_ensemble: 预计算的ensemble结果（避免重复计算15学派）
        buy_date: 买入日期字符串 'YYYY-MM-DD'（可选，用于T+1约束检测）

    返回结构化字典：
    - signal, confidence (来自贝叶斯融合+AI自适应融合)
    - entry_zone, stop_loss, stop_loss_rule (结构化价位)
    - targets (list of {price, ratio, reason})
    - principles (list of str: 各指标触发的判断规则)
    - key_values (dict: 关键指标快照)
    """
    if not indicators:
        return None

    p = indicators  # shorthand
    cp = _safe_float(p.get('current_price', 0))
    atr = _safe_float(p.get('vol_atr14', cp * 0.02))
    if atr <= 0:
        atr = cp * 0.02

    # === 信号与置信度 ===
    bayes_signal = p.get('bayes_fused_signal', 'neutral')
    bayes_posterior = _safe_float(p.get('bayes_fused_posterior', 0.5))
    bayes_entropy = _safe_float(p.get('bayes_fused_entropy', 0.5))

    # === Nüwa 15学派集成：15学派独立判断 + 加权投票 ===
    ensemble = precomputed_ensemble if precomputed_ensemble is not None else compute_expert_ensemble(p)
    ensemble_signal = ensemble['ensemble_signal']
    ensemble_confidence = ensemble['ensemble_confidence']

    # ===== Phase 1: 本地引擎内部融合（贝叶斯后验 + 专家集成） =====
    if bayes_signal == ensemble_signal:
        fused_signal = bayes_signal
        fused_confidence = max(bayes_posterior, ensemble_confidence) * 1.1
    elif bayes_signal == 'neutral' or ensemble_signal == 'neutral':
        if bayes_signal != 'neutral':
            fused_signal = bayes_signal
            fused_confidence = bayes_posterior * 0.85
        elif ensemble_signal != 'neutral':
            fused_signal = ensemble_signal
            fused_confidence = ensemble_confidence * 0.85
        else:
            fused_signal = 'neutral'
            fused_confidence = 0.35
    else:
        # 本地内部冲突：贝叶斯优先，但权重由后续AI融合阶段调整
        fused_signal = bayes_signal
        fused_confidence = bayes_posterior * 0.6
    fused_confidence = min(fused_confidence, 0.95)

    # ===== Phase 2: 本地 vs AI 自适应融合（基于历史信任分） =====
    ai_signal = None
    ai_confidence = None
    adaptive_fusion = None

    if ai_signal_dict and isinstance(ai_signal_dict, dict):
        ai_signal = ai_signal_dict.get('signal', None)
        ai_confidence = ai_signal_dict.get('confidence', None)

        if ai_signal and ai_signal in ('bullish', 'bearish', 'neutral'):
            # 获取信任分（默认等权）
            if trust_scores and isinstance(trust_scores, dict):
                w_local = trust_scores.get('fusion_weight_local', 0.5)
                ai_trust = trust_scores.get('ai_trust', 0.5)
                local_trust = trust_scores.get('local_trust', 0.5)
            else:
                w_local = 0.5
                ai_trust = 0.5
                local_trust = 0.5

            w_ai = 1.0 - w_local

            # 确定本地vs AI对齐状态
            if fused_signal == 'neutral' or ai_signal == 'neutral':
                alignment = '偏离'
                if fused_signal == 'neutral' and ai_signal != 'neutral':
                    # 本地观望，AI有方向 → 以AI为准（降置信度）
                    fused_signal = ai_signal
                    fused_confidence = (ai_confidence or 0.5) * 0.75 * (ai_trust / 0.5)
                elif ai_signal == 'neutral' and fused_signal != 'neutral':
                    # AI观望，本地有方向 → 保持本地
                    fused_confidence *= 0.85
                # else: both neutral, do nothing

            elif fused_signal == ai_signal:
                # 一致 → 置信度增强，按信任分加权
                alignment = '一致'
                ai_conf_val = ai_confidence or 0.5
                # 软融合：信任分越高权重越大
                boost = 1.0 + 0.15 * (ai_trust + local_trust) / 2.0  # max 1.15x
                fused_confidence = (
                    w_local * fused_confidence + w_ai * ai_conf_val
                ) * boost

            else:
                # 冲突 → 自适应加权，信任分高者获胜
                alignment = '冲突'
                ai_conf_val = ai_confidence or 0.5

                # 冲突时本地权重需要更大的置信区间
                # 当本地信任分显著高于AI时，w_local趋近1
                # 当AI信任分显著高于本地时，w_local趋近0
                conflict_w_local = 1.0 / (1.0 + np.exp(-3.0 * (trust_scores.get('trust_ratio', 1.0) - 1.0))) \
                    if trust_scores else 0.5

                if conflict_w_local >= 0.5:
                    # 信任本地更多 → 以本地信号为准
                    fused_signal = bayes_signal if bayes_signal != 'neutral' else ensemble_signal
                    fused_confidence *= conflict_w_local * 0.8
                else:
                    # 信任AI更多 → AI信号为准
                    fused_signal = ai_signal
                    fused_confidence = ai_conf_val * (1 - conflict_w_local) * 0.8

            fused_confidence = min(fused_confidence, 0.95)

            adaptive_fusion = {
                'alignment': alignment,
                'w_local': round(w_local, 3),
                'w_ai': round(w_ai, 3),
                'ai_trust': round(ai_trust, 3),
                'local_trust': round(local_trust, 3),
                'recommendation': trust_scores.get('recommendation', '') if trust_scores else '',
            }

    fused_confidence = min(fused_confidence, 0.95)

    # === Additive risk scoring (replaces multiplicative penalty chain) ===
    # Each risk factor contributes 0-3 points. Final adjustment via bounded sigmoid.
    # risk_score: 0=no risk, 4+ = significant risk, 8+ = extreme risk
    _extra_principles = []
    risk_score = 0.0

    # BEGE risk
    bebe_regime = str(p.get('bebe_regime', ''))
    bebe_vrp = str(p.get('bebe_vrp_signal', 'neutral'))
    bebe_ratio = _safe_float(p.get('bebe_good_bad_ratio'), 1.0)
    bebe_bad_trend = str(p.get('bebe_bad_vol_trend', 'stable'))
    bebe_good_trend = str(p.get('bebe_good_vol_trend', 'stable'))

    if bebe_regime == 'bad_environment':
        risk_score += 2.5
        _extra_principles.append(f'BEGE坏环境主导(好坏比{bebe_ratio:.2f})')
    if bebe_vrp == 'high_premium':
        risk_score += 1.5
        _extra_principles.append('VRP高溢价')
    elif bebe_vrp == 'negative_premium':
        risk_score += 2.0
        _extra_principles.append('VRP负溢价异常')
    if bebe_bad_trend == 'rising' and bebe_good_trend == 'falling':
        risk_score += 3.0
        _extra_principles.append('分布偏移预警(坏波升+好波降)')

    # Top escape risk
    top_esc_prob = _safe_float(p.get('top_escape_prob'), 0)
    top_esc_grade = str(p.get('top_escape_grade', ''))
    if top_esc_grade == 'critical':
        risk_score += 4.0
        _extra_principles.append(f'逃顶临界(概率{top_esc_prob:.0f}%)→强制大幅降置信度')
    elif top_esc_grade == 'high_risk':
        risk_score += 2.5
        _extra_principles.append(f'逃顶高风险(概率{top_esc_prob:.0f}%)→显著降置信度')
    elif top_esc_grade == 'elevated_risk':
        risk_score += 1.0
        _extra_principles.append('逃顶风险上升→适度降置信度')

    # Livermore danger signals
    liv_danger = p.get('livermore_danger', False)
    liv_danger_level = _safe_float(p.get('livermore_danger_level'), 0)
    if liv_danger and liv_danger_level >= 70:
        risk_score += 3.0
        _extra_principles.append('利弗莫尔危险信号严重→强制降置信度')
    elif liv_danger and liv_danger_level >= 40:
        risk_score += 1.5
        _extra_principles.append('利弗莫尔危险信号→降置信度')

    # Distribution day warning
    if p.get('dist_days_warning', False):
        dist_count = _safe_float(p.get('dist_days_count'), 0)
        risk_score += 1.5
        _extra_principles.append(f'分配日{int(dist_count)}个→市场顶部预警')

    # Turn score confirmation/conflict (bidirectional)
    cl_ts = _safe_float(p.get('cl_turn_score'), 0)
    cl_ts_sig = str(p.get('cl_turn_signal', ''))
    if cl_ts >= 80:
        if cl_ts_sig == 'strong_buy':
            if fused_signal == 'bearish':
                risk_score += 3.0
                _extra_principles.append(f'高概率底部拐点与做空冲突(评分{cl_ts:.0f})→置信度大幅下调')
            else:
                risk_score -= 0.5  # confirming signal reduces risk
                _extra_principles.append(f'高概率底部拐点确认做多(评分{cl_ts:.0f})→置信度上调')
        elif cl_ts_sig == 'strong_sell':
            if fused_signal == 'bullish':
                risk_score += 3.0
                _extra_principles.append(f'高概率顶部拐点与做多冲突(评分{cl_ts:.0f})→置信度大幅下调')
            else:
                risk_score -= 0.5
                _extra_principles.append(f'高概率顶部拐点确认做空(评分{cl_ts:.0f})→置信度上调')
    elif cl_ts >= 60:
        if cl_ts_sig == 'buy_watch':
            if fused_signal == 'bearish':
                risk_score += 1.0
            else:
                risk_score -= 0.3
        elif cl_ts_sig == 'sell_watch':
            if fused_signal == 'bullish':
                risk_score += 1.0
            else:
                risk_score -= 0.3

    # Dynamic router & Expert Soup conflicts
    dynamic_router = ensemble.get('dynamic_router', {})
    expert_soup = ensemble.get('expert_soup', {})
    enhanced_div = ensemble.get('enhanced_diversity', {})

    gated_dir = dynamic_router.get('gated_direction', 'neutral')
    if gated_dir != 'neutral' and gated_dir != fused_signal:
        risk_score += 1.0
        _extra_principles.append(f'动态路由器方向({gated_dir})与主信号冲突')
    elif gated_dir == fused_signal and fused_signal != 'neutral':
        risk_score -= 0.3

    soup_dir = expert_soup.get('soup_direction', 'neutral')
    if soup_dir != 'neutral' and soup_dir != fused_signal:
        risk_score += 1.0
        _extra_principles.append('Expert Soup软融合与投票方向冲突')

    # Effective N and reason overlap
    n_eff = enhanced_div.get('diversity_n_eff', 5.0)
    if n_eff < 2.0 and fused_signal != 'neutral':
        risk_score += 1.0
        _extra_principles.append(f'专家N_eff={n_eff:.1f}<2→群体盲点风险')
    elif n_eff > 3.5:
        risk_score -= 0.3

    if enhanced_div.get('reason_overlap_high', False) and fused_signal != 'neutral':
        risk_score += 0.5
        _extra_principles.append('专家理由重叠→伪共振风险')

    # Apply risk adjustment via bounded sigmoid (preserves differentiation)
    risk_score = max(risk_score, -1.0)  # floor on bonus
    confidence_multiplier = 1.0 / (1.0 + 0.22 * max(risk_score, 0))
    fused_confidence *= confidence_multiplier
    fused_confidence = min(fused_confidence, 0.95)
    fused_confidence = max(fused_confidence, 0.05)

    # 信号方向
    if fused_signal == 'bullish':
        direction = 1
    elif fused_signal == 'bearish':
        direction = -1
    else:
        direction = 0

    # === T+1 结算约束（A股：今日买入不可今日卖出） ===
    t1_blocked = False
    if buy_date is not None and direction == -1:
        if not is_sellable_today(buy_date):
            t1_blocked = True
            direction = 0
            fused_signal = 'neutral'
            fused_confidence = 0.05
            _extra_principles.insert(0, f'T+1锁定: 今日买入({buy_date})不可今日卖出，卖出信号已阻断')

    # === 入场区间 ===
    # 做多：当前价下方ATR/2到当前价；做空：当前价到上方ATR/2
    half_atr = atr / 2
    if direction > 0:
        # 偏向支撑位方向
        supports = p.get('dynamic_support', [])
        nearest_sup = None
        if supports:
            for s in supports[:3]:
                sp = _safe_float(s[0]) if isinstance(s, (list, tuple)) else _safe_float(s)
                if sp and sp < cp:
                    nearest_sup = sp
                    break
        lower = max(nearest_sup if nearest_sup else cp - half_atr, cp - half_atr)
        upper = cp
        entry_zone = f'{lower:.2f}-{upper:.2f}'
        # 精确入场价：支撑位上方ATR/3处（有支撑确认时的安全入场点）
        if nearest_sup and nearest_sup > cp - atr:
            entry_price = nearest_sup + atr / 3
            entry_rationale = f'支撑{nearest_sup:.2f}上方ATR/3处入场，止损置于支撑下方'
        else:
            entry_price = cp - half_atr + atr / 4
            entry_rationale = f'回调至{entry_price:.2f}(半ATR区间内+利弗莫尔关键点确认)入场'
    elif direction < 0:
        resistances = p.get('dynamic_resistance', [])
        nearest_res = None
        if resistances:
            for r in resistances[:3]:
                rp = _safe_float(r[0]) if isinstance(r, (list, tuple)) else _safe_float(r)
                if rp and rp > cp:
                    nearest_res = rp
                    break
        lower = cp
        upper = min(nearest_res if nearest_res else cp + half_atr, cp + half_atr)
        entry_zone = f'{lower:.2f}-{upper:.2f}'
        if nearest_res and nearest_res < cp + atr:
            entry_price = nearest_res - atr / 3
            entry_rationale = f'阻力{nearest_res:.2f}下方ATR/3处入场，止损置于阻力上方'
        else:
            entry_price = cp + half_atr - atr / 4
            entry_rationale = f'反弹至{entry_price:.2f}(半ATR区间内)入场，无明确阻力参考'
    else:
        entry_zone = f'{cp - half_atr:.2f}-{cp + half_atr:.2f}'
        entry_price = cp
        entry_rationale = '震荡市无明确方向，以当前价附近为参考入场点'

    # === 止损价 ===
    # 优先使用波动率自适应止损，其次是EW止损
    vol_stop = _safe_float(p.get('vol_adaptive_stop_price', 0))
    ew_stop_str = p.get('ew_trade_stop', '')
    ew_stop = _extract_float_price(ew_stop_str) if ew_stop_str else None

    # 2倍ATR止损作为底线
    atr_stop_long = cp - 2 * atr
    atr_stop_short = cp + 2 * atr

    if direction > 0:
        candidates = [s for s in [vol_stop, ew_stop, atr_stop_long] if s and s < cp]
        stop_loss = max(candidates) if candidates else atr_stop_long  # 最近的一个
        stop_rule = f'自适应止损(ATR×2.0={atr*2.0:.2f})' if vol_stop <= 0 else '波动率自适应止损'
    elif direction < 0:
        candidates = [s for s in [vol_stop, ew_stop, atr_stop_short] if s and s > cp]
        stop_loss = min(candidates) if candidates else atr_stop_short
        stop_rule = f'自适应止损(ATR×2.0={atr*2.0:.2f})' if vol_stop <= 0 else '波动率自适应止损'
    else:
        stop_loss = atr_stop_long
        stop_rule = '震荡市中建议轻仓观望'

    # === 目标价 ===
    targets = _derive_targets(cp, direction, atr, p)

    # === 一次性止盈点（从分批目标中合成） ===
    if targets and direction != 0:
        # 取第一目标（最近的目标，通常是最高概率止盈位）
        one_shot = targets[0]
        one_shot_target = {
            'price': one_shot['price'],
            'ratio': '100%',
            'reason': f'(一次性止盈) {one_shot["reason"]}',
        }
        # 如果是做多，取移动平均阻力作为合成一次性止盈
        all_tgt_prices = [_safe_float(t['price']) for t in targets]
        valid_prices = [p for p in all_tgt_prices if p and ((direction > 0 and p > cp) or (direction < 0 and p < cp))]
        if len(valid_prices) >= 2:
            if direction > 0:
                composite = min(valid_prices)  # 最近目标作为一次性止盈
            else:
                composite = max(valid_prices)
            one_shot_target['price'] = f'{composite:.2f}'
            one_shot_target['reason'] = f'(一次性止盈) 最近有效目标'
        elif len(valid_prices) == 1:
            one_shot_target['price'] = f'{valid_prices[0]:.2f}'
    else:
        one_shot_target = {'price': '-', 'ratio': '-', 'reason': '震荡市无明确止盈目标'}

    # === 判断原则 ===
    principles = _derive_principles(p, cp)
    if _extra_principles:
        principles = _extra_principles + principles

    # === 关键指标值 ===
    key_values = {
        'price': round(cp, 2),
        'atr': round(atr, 2),
        'atr_pct': round(p.get('vol_atr_pct', 0), 1),
        'adx': round(_safe_float(p.get('dmi_adx', 0)), 1),
        'rsi': round(_safe_float(p.get('rsi', 50)), 1),
        'ma5': round(_safe_float(p.get('ma5', 0)), 2),
        'ma20': round(_safe_float(p.get('ma20', 0)), 2),
        'ma60': round(_safe_float(p.get('ma60', 0)), 2),
        'vol_ratio': round(_safe_float(p.get('vol_ratio', 1)), 2),
        'rv_composite': round(_safe_float(p.get('rv_composite', 0)), 4),
        'rv_level': t(p.get('rv_level', '?')),
        'bebe_regime': t(bebe_regime),
        'bebe_good_bad_ratio': round(bebe_ratio, 3),
        'bebe_vrp_signal': t(bebe_vrp),
        'bayes_posterior': round(bayes_posterior, 4),
        'bayes_entropy': round(bayes_entropy, 4),
        'bayes_signal': t(bayes_signal),
        'mom_direction': t(p.get('momentum_direction', '?')),
        'weekly_trend': t(p.get('weekly_trend', '?')),
        'dmi_direction': t(p.get('dmi_di_direction', '?')),
        'chan_stroke': t(p.get('chanlun_stroke_state', '?')),
        'chan_divergence': t(p.get('chanlun_divergence_type', '无')),
        'chan_wolf': t(p.get('chanlun_wolf_signal', '?')),
        'chan_trend': t(p.get('chanlun_trend_type', '?')),
        'chan_tf_align': t(p.get('chanlun_multi_tf_alignment', '?')),
        'chan_seg_destr': t(p.get('chanlun_last_segment_destruction', '')),
        'chan_zs_expanded': p.get('chanlun_zs_expanded', False),
        'chan_xzd_active': p.get('chanlun_xzd_active', False),
        'chan_xzd_dir': t(p.get('chanlun_xzd_direction', '')),
        'chan_xzd_conf': _safe_float(p.get('chanlun_xzd_confidence', 0)),
        'chan_xzd_reasons': t(p.get('chanlun_xzd_reasons', '')),
        'b2560_signal': t(p.get('b2560_signal', '?')),
        'b2560_ma25_dir': t(p.get('b2560_ma25_direction', '?')),
        'mp_phase': t(p.get('mp_phase', '?')),
        'mp_trend_cred': _safe_float(p.get('mp_trend_credibility', 0.5)),
        'vr_zone': t(p.get('vr_zone', '?')),
        'vr_signal': t(p.get('vr_signal', '?')),
        'vpa_signal': t(p.get('vpa_signal', '?')),
        'bdmi_signal': t(p.get('bdmi_signal', '?')),
        'vc_pattern': t(p.get('vc_pattern', '?')),
        'ew_alt_valid': p.get('ew_alternation_valid', False),
        'ew_extension': p.get('ew_extension_wave', None),
        'nrb_signal': t(p.get('nrb_signal', '无')),
        'v9r_name': t(p.get('v9r_name', '?')),
        'v9r_strength': t(p.get('v9r_strength', '?')),
        'v9r_action': t(p.get('v9r_action', '?')),
        'sm_mfi': _safe_float(p.get('sm_mfi', 50)),
        'sm_vwap_dev': _safe_float(p.get('sm_vwap_deviation', 0)),
        'sm_ad_trend': t(p.get('sm_ad_trend', '?')),
        'sm_ad_divergence': t(p.get('sm_ad_divergence', '')),
        'sm_signal': t(p.get('sm_smart_signal', '?')),
        'g8_rule': p.get('g8_active_rule'),
        'g8_category': t(p.get('g8_rule_category', 'none')),
        'g8_land_zone': t(p.get('g8_land_volume_zone', '')),
        # 唐能通
        'tang_jiato': p.get('tang_jiato', False),
        'tang_jiaya': p.get('tang_jiaya', False),
        'tang_runway': t(p.get('tang_runway_grade', '?')),
        'tang_laoyatou': p.get('tang_laoyatou', False),
        'tang_laoyatou_phase': t(p.get('tang_laoyatou_phase', '?')),
        'tang_33_valid': p.get('tang_33_valid', False),
        'tang_triple': t(p.get('tang_triple_status', '无')),
        # 利弗莫尔
        'livermore_signal': t(p.get('livermore_signal', '?')),
        'livermore_danger': p.get('livermore_danger', False),
        'livermore_danger_level': _safe_float(p.get('livermore_danger_level'), 0),
        'livermore_action': t(p.get('livermore_action', '?')),
        # 解缠论增强
        'cl_power_ratio': _safe_float(p.get('cl_power_ratio'), 1),
        'cl_divergence_type': t(p.get('cl_divergence_type', '无')),
        'cl_cg_direction': t(p.get('cl_cg_direction', '?')),
        'cl_turn_score': _safe_float(p.get('cl_turn_score'), 0),
        'cl_turn_signal': t(p.get('cl_turn_signal', '?')),
        'cl_intra_div': p.get('cl_intra_divergence', False),
        # 逃顶
        'top_escape_count': _safe_float(p.get('top_escape_count'), 0),
        'top_escape_prob': top_esc_prob,
        'top_escape_grade': t(top_esc_grade),
        'dist_days_count': _safe_float(p.get('dist_days_count'), 0),
        'dist_days_warn': p.get('dist_days_warning', False),
    }

    # === 仓位计算 ===
    position_sizing = _compute_position_sizing(
        entry_price, float(stop_loss), float(atr), fused_confidence,
        cp, fused_signal, one_shot_target
    )

    # === 缺口风险评估（如有日线数据） ===
    gap_risk = None
    if daily_df is not None and len(daily_df) >= 30:
        gap_risk = gap_risk_assessment(daily_df, current_position={
            'direction': fused_signal,
            'entry_price': float(entry_price),
            'shares': position_sizing.get('shares', 0),
        } if fused_signal != 'neutral' else None)
        reduce_factor = gap_risk.get('position_reduce_factor', 1.0)
        if reduce_factor < 1.0:
            position_sizing['shares'] = int(position_sizing['shares'] * reduce_factor / 100) * 100
            position_sizing['position_value'] = round(position_sizing['shares'] * float(entry_price), 2)
            position_sizing['position_pct'] = round(position_sizing['position_value'] / ACCOUNT_EQUITY * 100, 2)
            position_sizing['gap_reduce_factor'] = reduce_factor
            position_sizing['note'] = position_sizing.get('note', '') + f' | 缺口风险调整{reduce_factor:.0%}'

    # === TWAP slicing for large orders (>2% ADV) ===
    twap_plan = None
    try:
        from market_microstructure import TWAPSlicer, AlmgrenChrissImpact
        slicer = TWAPSlicer()
        shares = position_sizing.get('shares', 0)
        if shares > 0:
         adv_est = _estimate_adv(symbol) if symbol is not None else 10000000
         if adv_est > 0 and slicer.should_slice(shares, adv_est):
                twap_slices = slicer.compute_slices(shares, adv_est)
                impact_savings = slicer.estimated_impact_savings(
                    shares, adv_est, AlmgrenChrissImpact())
                twap_plan = {
                    'n_slices': len(twap_slices),
                    'slices': [{'id': s.slice_id, 'qty': s.quantity,
                                'start': str(s.start_time), 'end': str(s.end_time)}
                               for s in twap_slices],
                    'impact_savings_bps': impact_savings.get('savings_bps', 0),
                }
                _extra_principles.append(
                    f'TWAP拆单: {shares}股→{len(twap_slices)}片, '
                    f'节省冲击{impact_savings.get("savings_bps",0)}bp')
    except ImportError:
        pass

    # === Capacity check: warn if position exceeds liquidity limit ===
    try:
        from risk_management import CapacityCalculator
        cc = CapacityCalculator()
        adv_est = _estimate_adv(symbol) if symbol is not None else 10000000
        current_price = indicators.get('close', 10.0)  # <--- 新增这行，从指标库里获取最新收盘价
        max_pos = cc.max_position_value(adv_est, current_price)
        pos_value = position_sizing.get('position_value', 0)
        if pos_value > max_pos * 0.8:
            capped_pct = max_pos * 0.8 / pos_value
            position_sizing['shares'] = int(position_sizing['shares'] * capped_pct / 100) * 100
            position_sizing['position_value'] = position_sizing['shares'] * current_price
            position_sizing['capacity_capped'] = True
            _extra_principles.append(f'流动性限制: 仓位从{pos_value:.0f}→{position_sizing["position_value"]:.0f}')
    except ImportError:
        pass

    # === T+0 grid for existing positions (high vol days) ===
    t0_grid_plan = None
    try:
        from market_microstructure import IntradayT0Algo
        t0 = IntradayT0Algo()
        atr_pct = position_sizing.get('atr_pct', _safe_float(
            indicators.get('vol_atr_pct', 0.02), 0.02))
        has_existing_position = False
        if t0.should_activate(atr_pct) and has_existing_position:
            pos_info = existing_position_info or {}
            avail = pos_info.get('available_today', 0)
            total = pos_info.get('total_holding', 0)
            if avail > 0 and total > 0:
                atr_val = _safe_float(indicators.get('vol_atr14', 0), 0)
                grid = t0.generate_grid(symbol, current_price, max(atr_val, 0.01),
                                       avail, total)
                if grid:
                    t0_grid_plan = {
                        'levels': [{'price': g.price, 'action': g.action,
                                    'qty': g.quantity} for g in grid],
                        'atr_pct': round(atr_pct, 3),
                    }
                    _extra_principles.append(
                        f'T+0网格: {len(grid)}层, ATR={atr_pct:.1%}')
    except ImportError:
        pass

    # === Portfolio optimization (CVXPY + Barra) ===
    portf_weights = None
    try:
        from portfolio_optimizer import optimize_portfolio
        # Single-stock context: just this candidate
        candidates = [{'symbol': symbol, 'ranker_score': fused_confidence}]
        prev_w = {symbol: position_sizing.get('position_pct', 0) / 100}
        result = optimize_portfolio(candidates, prev_w)
        if result.get('weights'):
            portf_weights = result['weights']
            diag = result.get('diagnostics', {})
            if diag.get('turnover', 0) > 0.5:
                _extra_principles.append(
                    f'组合优化: 换手率{diag["turnover"]:.0%}, '
                    f'有效持仓{diag.get("effective_n",0)}只')
    except ImportError:
        pass

    # === 日内亏损限额检查 ===
    daily_loss = daily_loss_limit_check(
        trade_results_today if trade_results_today else [],
        max_daily_loss_pct=0.03,
    )
    if daily_loss.get('halt_trading'):
        _extra_principles.insert(0, f'⛔ 日内亏损限额熔断(强制平仓/不开新仓): {daily_loss["status"]}')
        fused_signal = 'neutral'
        direction = 0
        fused_confidence = 0.05

    # === 涨跌停板检测 ===
    symbol = str(indicators.get('_symbol', ''))
    limit_check = _check_price_limit(daily_df, symbol)
    if limit_check['is_limit_up'] and fused_signal == 'bullish':
        fused_confidence *= 0.30
        _extra_principles.insert(0, f'⚠️ 涨停板({limit_check["change_pct"]:.1f}%)→无法买入，做多信号失效')
    elif limit_check['is_limit_down'] and fused_signal == 'bearish':
        fused_confidence *= 0.30
        _extra_principles.insert(0, f'⚠️ 跌停板({limit_check["change_pct"]:.1f}%)→无法卖出，做空信号失效')
    elif limit_check['is_limit_up']:
        fused_confidence *= 0.50
        _extra_principles.insert(0, f'⚠️ 涨停板→无法买入，信号仅作参考')
    elif limit_check['is_limit_down']:
        fused_confidence *= 0.50
        _extra_principles.insert(0, f'⚠️ 跌停板→流动性枯竭，信号不可靠')

    return {
        'signal': fused_signal,
        'confidence': round(fused_confidence, 4),
        'signal_cn': t(fused_signal),
        'entry_zone': entry_zone,
        'entry_price': f'{entry_price:.2f}',
        'entry_rationale': entry_rationale,
        'stop_loss': f'{stop_loss:.2f}',
        'stop_loss_rule': stop_rule,
        'targets': targets,
        'one_shot_target': one_shot_target,
        'principles': principles,
        'key_values': key_values,
        # Nüwa 7学派集成详情
        'ensemble': {
            'signal': ensemble_signal,
            'confidence': ensemble_confidence,
            'votes': ensemble['votes'],
            'diversity': ensemble['diversity'],
            'synergy_pairs': ensemble['synergy_pairs'],
            'conflict_pairs': ensemble['conflict_pairs'],
            'supporting_reasons': ensemble['supporting_reasons'],
            'opposing_reasons': ensemble['opposing_reasons'],
            'schools': ensemble['school_signals'],
            'enhanced_diversity': enhanced_div,
            'dynamic_router': dynamic_router,
            'expert_soup': expert_soup,
        },
        # 风险快照（基于指标的轻量风控估计）
        'risk_snapshot': _derive_risk_snapshot(p),
        # 贝叶斯增强风险评估
        'integrated_risk': _integrated_risk_assessment(p),
        'sigma_events': _detect_sigma_events(p),
        'posterior_confidence_adj': _compute_posterior_confidence_adjustment(p),
        'strategy_bandit': _bayesian_strategy_bandit(p),
        # 贝叶斯序贯更新示例（从先验50%出发，基于当前贝叶斯后验计算证据强度）
        'sequential_update': _bayesian_sequential_update(0.5, max(bayes_posterior / (1 - bayes_posterior + 1e-10), 0.01), 1),
        'bayes_signal': bayes_signal,
        'bayes_posterior': bayes_posterior,
        'fusion_type': _build_fusion_type(bayes_signal, ensemble_signal, adaptive_fusion),
        # AI vs 本地自适应融合
        'alignment': adaptive_fusion.get('alignment', '?') if adaptive_fusion else '?',
        'adaptive_fusion': adaptive_fusion,
        # 仓位管理
        'position_sizing': position_sizing,
        # 缺口风险评估（如有日线数据）
        'gap_risk': gap_risk,
        # 日内亏损限额框架状态
        'daily_loss_limit': daily_loss,
        'daily_loss_blocked': daily_loss.get('halt_trading', False),
        # T+1约束状态
        't1_blocked': t1_blocked,
        'buy_date': buy_date,
    }


def _build_fusion_type(bayes_signal, ensemble_signal, adaptive_fusion):
    """构建融合类型标签，包含本地融合和AI融合两个阶段的状态。

    Args:
        bayes_signal: 贝叶斯后验信号
        ensemble_signal: 专家集成信号
        adaptive_fusion: AI自适应融合结果字典（含alignment字段），或None

    Returns:
        融合类型字符串，如 "一致增强+AI一致" 或 "贝叶斯优先(冲突)+AI冲突"
    """
    # Phase 1: 本地融合类型
    if bayes_signal == ensemble_signal:
        base = '一致增强'
    elif bayes_signal != 'neutral' and ensemble_signal != 'neutral' and bayes_signal != ensemble_signal:
        base = '贝叶斯优先(冲突)'
    else:
        base = '单引擎有效'

    # Phase 2: AI融合追加
    if adaptive_fusion:
        alignment = adaptive_fusion.get('alignment', '?')
        ai_suffix = '一致' if alignment == '一致' else '冲突' if alignment == '冲突' else '偏离'
        return f'{base}+AI{ai_suffix}'

    return base


def _estimate_adv(symbol: str) -> int:
    """Estimate average daily volume for a stock from recent data."""
    try:
        from data_loader import get_daily_kline
        df = get_daily_kline(symbol, days=30)
        if df is not None and not df.empty:
            return int(df['volume'].tail(20).mean()) * 100  # 手→股
    except Exception:
        pass
    return 1_000_000  # Default: 1M shares


def _get_regime_win_rate(direction: str = 'bullish') -> float:
    """
    Get historical win rate from backtest data for the current regime.
    This is the CALIBRATED probability — NOT LLM semantic output.

    Falls back to 0.45 (slightly below 50%) if no backtest data available.
    """
    try:
        from backtest_feedback import get_historical_stats
        stats = get_historical_stats()
        if stats and stats.get('overall', {}).get('total', 0) > 10:
            overall = stats['overall']
            wr = overall['wins'] / overall['total']
            return round(wr, 3)
    except Exception:
        pass

    # Fallback: try DB directly
    try:
        import duckdb
        from config import DB_PATH
        conn = duckdb.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""SELECT
            SUM(CASE WHEN net_pnl_pct > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
            FROM backtest_trades WHERE direction='long'""")
        row = c.fetchone()
        conn.close()
        if row and row[0]:
            return round(float(row[0]), 3)
    except Exception:
        pass

    return 0.45  # Conservative default


def _compute_position_sizing(entry_price, stop_loss, atr, confidence,
                              current_price, direction, nearest_target):
    """
    基于风险的仓位计算。

    Args:
        entry_price: 入场价
        stop_loss: 止损价
        atr: 平均真实波幅
        confidence: 融合置信度 (0-1)
        current_price: 当前价格
        direction: 'bullish' | 'bearish' | 'neutral'
        nearest_target: 最近目标价 (用于计算R:R)

    Returns:
        dict with position sizing details
    """
    if direction == 'neutral' or confidence <= 0 or entry_price <= 0 or stop_loss <= 0:
        return {
            'risk_pct': 0.0, 'position_pct': 0.0, 'position_value': 0.0,
            'shares': 0, 'risk_amount': 0.0, 'reward_risk_ratio': 0.0,
            'expected_value_pct': 0.0, 'position_sizing_method': 'none',
            'note': '无法计算仓位（信号中性或参数无效）',
        }

    # 风险参数
    equity = ACCOUNT_EQUITY            # 来自config.py
    MAX_RISK_PER_TRADE = 0.01         # 单笔最大风险 = 1% 账户
    MAX_POSITION_PCT = 0.25           # 单笔最大仓位 = 25%
    CONFIDENCE_FLOOR = 0.30           # 低于此置信度不开仓

    if confidence < CONFIDENCE_FLOOR:
        return {
            'risk_pct': 0.0, 'position_pct': 0.0, 'position_value': 0.0,
            'shares': 0, 'risk_amount': 0.0, 'reward_risk_ratio': 0.0,
            'expected_value_pct': 0.0, 'position_sizing_method': 'confidence_floor',
            'note': f'置信度{confidence:.2f}<阈值{CONFIDENCE_FLOOR}，不开仓',
        }

    # 风险距离（百分比）
    risk_distance = abs(entry_price - stop_loss) / entry_price
    if risk_distance < 0.002:
        risk_distance = 0.002  # 最小0.2%防止除零

    # ATR归一化的风险距离（用于波动率调整）
    atr_pct = atr / current_price if current_price > 0 else 0.02
    vol_adj = max(0.5, min(1.5, 0.02 / atr_pct))  # 低波动扩大仓位，高波动缩小

    # ---- Safe Kelly: use BACKTEST win_rate, NOT LLM confidence ----
    # LLM confidence is uncalibrated semantic output → only as risk multiplier
    # p = historical win rate in current regime (from backtest_feedback)
    p_historical = _get_regime_win_rate(direction)  # Backtest-validated
    p_kelly = min(0.60, max(0.30, p_historical))    # Clamp to sane range

    # LLM confidence multiplier: 0.5 (bearish) to 1.0 (bullish)
    # LLM is used to scale DOWN positions, never to inflate them
    llm_multiplier = 0.5 + 0.5 * confidence           # Maps [0,1] → [0.5, 1.0]

    # Edge = p - q (win_rate - loss_rate for even-odds approximation)
    edge = max(0.01, 2 * p_kelly - 1.0)               # From historical data only
    edge *= llm_multiplier                             # LLM can only REDUCE

    reward_distance = 0.0
    nearest_target_price = _safe_float(nearest_target.get('price', nearest_target)) if isinstance(nearest_target, dict) else nearest_target
    if nearest_target_price and nearest_target_price > 0:
        if direction == 'bullish':
            reward_distance = (nearest_target_price - entry_price) / entry_price
        else:
            reward_distance = (entry_price - nearest_target_price) / entry_price
    if reward_distance < risk_distance:
        reward_distance = risk_distance * 2.0  # 默认2:1盈亏比
    odds_ratio = risk_distance / max(reward_distance, 0.001)
    kelly_f = max(0.0, (edge - odds_ratio * (1 - edge)) / max(odds_ratio, 0.01))
    kelly_f = min(kelly_f, 0.25)  # 凯利上限25%

    # 半凯利（更保守实用）
    half_kelly = kelly_f * 0.5

    # 风险金额
    risk_amount = equity * MAX_RISK_PER_TRADE * confidence * vol_adj

    # 仓位比例 = 风险金额 / (每股风险 × 股价)
    # 每股风险 = |entry - stop|
    risk_per_share = abs(entry_price - stop_loss)
    shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
    shares = int(shares / 100) * 100  # A股百股整数倍
    position_value = shares * entry_price
    position_pct = position_value / ACCOUNT_EQUITY

    # 仓位上限约束
    max_position_value = equity * MAX_POSITION_PCT
    if position_value > max_position_value:
        position_value = max_position_value
        position_pct = MAX_POSITION_PCT
        shares = int(position_value / entry_price / 100) * 100

    # 凯利上限约束
    kelly_max_value = equity * half_kelly
    if position_value > kelly_max_value:
        position_value = min(position_value, kelly_max_value)
        position_pct = position_value / ACCOUNT_EQUITY
        shares = int(position_value / entry_price / 100) * 100

    # A股最小交易单位约束：不足1手（100股）则不开仓
    if shares < 100:
        return {
            'risk_pct': 0.0, 'position_pct': 0.0, 'position_value': 0.0,
            'shares': 0, 'risk_amount': 0.0, 'reward_risk_ratio': 0.0,
            'expected_value_pct': 0.0, 'position_sizing_method': 'lot_size_floor',
            'note': f'计算股数{shares}股不足1手(100股)，仓位太小无法执行',
        }

    # R:R比
    reward_risk = reward_distance / risk_distance if risk_distance > 0 else 0.0

    # 期望收益（%账户）
    win_rate_estimate = confidence
    expected_value_pct = (win_rate_estimate * reward_distance - (1 - win_rate_estimate) * risk_distance) * position_pct * 100

    return {
        'risk_pct': round(MAX_RISK_PER_TRADE * 100, 2),
        'position_pct': round(position_pct * 100, 2),
        'position_value': round(position_value, 2),
        'shares': shares,
        'risk_amount': round(risk_amount, 2),
        'risk_distance_pct': round(risk_distance * 100, 2),
        'reward_distance_pct': round(reward_distance * 100, 2),
        'reward_risk_ratio': round(reward_risk, 2),
        'expected_value_pct': round(expected_value_pct, 2),
        'kelly_half': round(half_kelly * 100, 2),
        'kelly_full': round(kelly_f * 100, 2),
        'vol_adjustment': round(vol_adj, 2),
        'position_sizing_method': 'risk_based_half_kelly',
        'note': f'半凯利{half_kelly*100:.1f}% | {MAX_RISK_PER_TRADE*100:.0f}%风险预算 | 波动率调整{vol_adj:.2f}x',
    }


def _check_price_limit(daily_df, symbol):
    """
    检测A股涨跌停板状态。

    A股涨跌幅限制：
    - 主板(60/00开头): ±10%
    - 创业板(30开头): ±20%
    - 科创板(688开头): ±20%
    - ST股票: ±5%（简化处理：默认10%）

    返回: {is_limit_up, is_limit_down, change_pct, limit_pct}
    """
    if daily_df is None or len(daily_df) < 2:
        return {'is_limit_up': False, 'is_limit_down': False, 'change_pct': 0, 'limit_pct': 10}

    last = daily_df.iloc[-1]
    prev_close = float(daily_df['close'].iloc[-2])
    current = float(last['close'])
    if prev_close <= 0:
        return {'is_limit_up': False, 'is_limit_down': False, 'change_pct': 0, 'limit_pct': 10}

    change_pct = (current - prev_close) / prev_close * 100

    # 确定涨跌停幅度
    code = str(symbol).replace('.SH', '').replace('.SZ', '')
    if code.startswith('30') or code.startswith('688'):
        limit_pct = 20
    else:
        limit_pct = 10

    near_limit = limit_pct * 0.95  # within 95% of limit → treated as limit hit
    is_limit_up = change_pct >= near_limit
    is_limit_down = change_pct <= -near_limit

    return {
        'is_limit_up': is_limit_up,
        'is_limit_down': is_limit_down,
        'change_pct': round(change_pct, 1),
        'limit_pct': limit_pct,
    }


def _derive_targets(cp, direction, atr, p):
    """从多个来源综合推导目标价"""
    targets = []
    seen_prices = set()

    def add_target(price, weight, reason):
        if price is None or price <= 0:
            return
        pkey = round(price, 2)
        if pkey in seen_prices:
            return
        seen_prices.add(pkey)
        targets.append({
            'price': f'{price:.2f}',
            'ratio': weight,
            'reason': reason,
        })

    # 来源1: 艾略特波浪目标
    ew_target = _extract_float_price(p.get('ew_trade_target', ''))
    if ew_target and ((direction > 0 and ew_target > cp) or (direction < 0 and ew_target < cp)):
        add_target(ew_target, '40%', f'艾略特波浪目标: {p.get("ew_trade_setup", "")}')

    ew_next = _safe_float(p.get('ew_next_target', 0))
    if ew_next and ew_next > 0 and ((direction > 0 and ew_next > cp) or (direction < 0 and ew_next < cp)):
        add_target(ew_next, '35%', f'斐波那契目标: {ew_next:.2f}')

    # 来源2: 动态支撑/阻力
    if direction > 0:
        resistances = p.get('dynamic_resistance', [])
        for r in resistances[:2]:
            rp = _safe_float(r[0]) if isinstance(r, (list, tuple)) else _safe_float(r)
            if rp and rp > cp:
                add_target(rp, '30%', f'动态阻力位')

        vol_res = p.get('vol_verified_resistance', [])
        for vr in vol_res[:1]:
            vp = _safe_float(vr[0]) if isinstance(vr, (list, tuple)) else _safe_float(vr)
            if vp and vp > cp:
                add_target(vp, '25%', f'量能验证阻力')
    else:
        supports = p.get('dynamic_support', [])
        for s in supports[:2]:
            sp = _safe_float(s[0]) if isinstance(s, (list, tuple)) else _safe_float(s)
            if sp and sp < cp:
                add_target(sp, '30%', f'动态支撑位')

        vol_sup = p.get('vol_verified_support', [])
        for vs in vol_sup[:1]:
            vp = _safe_float(vs[0]) if isinstance(vs, (list, tuple)) else _safe_float(vs)
            if vp and vp < cp:
                add_target(vp, '25%', f'量能验证支撑')

    # 来源3: 布林带
    bb_upper = _safe_float(p.get('bb_upper', 0))
    bb_lower = _safe_float(p.get('bb_lower', 0))
    if direction > 0 and bb_upper > cp:
        add_target(bb_upper, '25%', '布林带上轨')
    elif direction < 0 and bb_lower < cp:
        add_target(bb_lower, '25%', '布林带下轨')

    # 来源4: ATR倍数目标（兜底）
    if len(targets) == 0:
        if direction > 0:
            t1 = cp + atr * 1.5
            t2 = cp + atr * 3.0
            add_target(t1, '60%', f'1.5×ATR({atr:.2f})')
            add_target(t2, '40%', f'3.0×ATR({atr:.2f})')
        elif direction < 0:
            t1 = cp - atr * 1.5
            t2 = cp - atr * 3.0
            add_target(t1, '60%', f'1.5×ATR({atr:.2f})')
            add_target(t2, '40%', f'3.0×ATR({atr:.2f})')

    # 按距离排序
    if direction > 0:
        targets.sort(key=lambda x: _safe_float(x['price']))
    else:
        targets.sort(key=lambda x: _safe_float(x['price']), reverse=True)

    return targets[:3]


def _derive_principles(p, cp):
    """从指标状态自动生成判断原则列表（中文）"""
    rules = []
    c = cp  # current price

    # 1. MA排列
    ma5 = _safe_float(p.get('ma5', 0))
    ma20 = _safe_float(p.get('ma20', 0))
    ma60 = _safe_float(p.get('ma60', 0))
    if ma5 > ma20 > ma60:
        rules.append(f'均线多头排列(MA5={ma5:.2f}>MA20={ma20:.2f}>MA60={ma60:.2f})，中期上升趋势')
    elif ma5 < ma20 < ma60:
        rules.append(f'均线空头排列(MA5={ma5:.2f}<MA20={ma20:.2f}<MA60={ma60:.2f})，中期下降趋势')
    else:
        if ma5 > ma20:
            rules.append(f'短均金叉(MA5={ma5:.2f}>MA20={ma20:.2f})，短期偏多但MA60={ma60:.2f}压制')
        elif ma5 < ma20:
            rules.append(f'短均死叉(MA5={ma5:.2f}<MA20={ma20:.2f})，短期偏空')
        else:
            rules.append(f'均线缠绕(MA5≈MA20={ma20:.2f})，无明确方向')

    # 2. 价格vs均线
    if c > ma20:
        rules.append(f'价格{c:.2f}在MA20({ma20:.2f})上方，短线强势')
    else:
        rules.append(f'价格{c:.2f}在MA20({ma20:.2f})下方，短线弱势')

    # 3. ADX + DMI
    adx = _safe_float(p.get('dmi_adx', 0))
    di_dir = p.get('dmi_di_direction', '')
    adx_trend = p.get('dmi_adx_trend', '')
    if adx > 25:
        rules.append(f'ADX={adx:.1f}>25，趋势明确（{t(adx_trend)}），趋势策略有效')
    elif adx > 20:
        rules.append(f'ADX={adx:.1f}(20-25)，趋势正在形成，{t(di_dir)}方向占优')
    else:
        rules.append(f'ADX={adx:.1f}<20，震荡市特征，追涨杀跌风险大')

    # 4. RSI
    rsi = _safe_float(p.get('rsi', 50))
    if rsi > 70:
        rules.append(f'RSI={rsi:.1f}>70，超买区域，警惕回调')
    elif rsi > 60:
        rules.append(f'RSI={rsi:.1f}(60-70)，偏强但未超买')
    elif rsi < 30:
        rules.append(f'RSI={rsi:.1f}<30，超卖区域，关注反弹')
    elif rsi < 40:
        rules.append(f'RSI={rsi:.1f}(30-40)，偏弱但未超卖')
    else:
        rules.append(f'RSI={rsi:.1f}，中性区间')

    # 5. MACD
    macd_dif = _safe_float(p.get('macd_dif', 0))
    macd_dea = _safe_float(p.get('macd_dea', 0))
    macd_hist = _safe_float(p.get('macd_hist', 0))
    if macd_dif > macd_dea and macd_hist > 0:
        rules.append(f'MACD金叉(DIF={macd_dif:.2f}>DEA={macd_dea:.2f})，柱状线正值')
    elif macd_dif < macd_dea and macd_hist < 0:
        rules.append(f'MACD死叉(DIF={macd_dif:.2f}<DEA={macd_dea:.2f})，柱状线负值')
    else:
        rules.append(f'MACD方向不明(DIF={macd_dif:.2f}, DEA={macd_dea:.2f})')

    # 6. 量价关系
    vol_ratio = _safe_float(p.get('vol_ratio', 1))
    vol_resonance = p.get('vol_price_resonance', False)
    if vol_resonance:
        rules.append(f'量价配合(量比={vol_ratio:.2f})，放量方向可信')
    else:
        rules.append(f'量价背离(量比={vol_ratio:.2f})，当前方向需警惕')

    # 7. 成交量特征
    vol_type = p.get('vol_type', '')
    vol_trend = p.get('vol_trend', '')
    if vol_type:
        rules.append(f'成交量状态: {t(vol_type)}，趋势: {t(vol_trend)}')
    if p.get('vol_stacking', False):
        rules.append(f'堆量形态({p.get("vol_stacking_days", 0)}天)，主力资金介入迹象')
    if p.get('high_vol_stagnation', False):
        rules.append('⚠️ 放量滞涨，警惕主力出货')
    if p.get('low_vol_pullback', False):
        rules.append('缩量回踩，洗盘特征，关注支撑有效性')

    # 8. K线形态
    pat_rec = str(p.get('pattern_recent', ''))
    pat_dom = str(p.get('pattern_dominant', ''))
    if pat_dom and pat_dom != '无':
        rules.append(f'主导K线形态: {t(pat_dom)}')
    if pat_rec and pat_rec != '[]':
        # 提取关键形态
        for kw, kw_cn in [('底分型', '底分型(看涨)'), ('顶分型', '顶分型(看跌)'),
                           ('晨星', '晨星(强看涨反转)'), ('暮星', '暮星(强看跌反转)'),
                           ('吞没看涨', '看涨吞没'), ('吞没看跌', '看跌吞没'),
                           ('刺透', '刺透形态(看涨)'), ('乌云盖顶', '乌云盖顶(看跌)')]:
            if kw in pat_rec:
                rules.append(f'K线形态: {kw_cn}')

    # 9. DMA
    dma_sig = str(p.get('dma_signal', ''))
    dma_val = _safe_float(p.get('dma', 0))
    dma_ama = _safe_float(p.get('dma_ama', 0))
    if '金叉' in dma_sig:
        rules.append(f'DMA金叉(DMA={dma_val:.2f}>AMA={dma_ama:.2f})，中期买入信号')
    elif '死叉' in dma_sig:
        rules.append(f'DMA死叉(DMA={dma_val:.2f}<AMA={dma_ama:.2f})，中期卖出信号')
    elif dma_sig:
        rules.append(f'DMA: {dma_sig}(DMA={dma_val:.2f}, AMA={dma_ama:.2f})')

    # 10. OBV
    obv_div = str(p.get('obv_divergence', ''))
    obv_sig = str(p.get('obv_signal', ''))
    if '底背离' in obv_div:
        rules.append('OBV底背离: 价跌量升，资金暗中吸筹')
    elif '顶背离' in obv_div:
        rules.append('OBV顶背离: 价涨量降，资金高位出逃')
    if obv_sig and obv_sig != '无':
        rules.append(f'OBV信号: {t(obv_sig)}')

    # 11. 艾略特波浪
    ew_pattern = str(p.get('ew_pattern', ''))
    ew_wave = str(p.get('ew_current_wave', ''))
    ew_sig = str(p.get('ew_trade_signal', ''))
    ew_conf = _safe_float(p.get('ew_confidence', 0))
    if ew_pattern and ew_pattern != 'unknown':
        rules.append(f'波浪结构: {t(ew_pattern)}，当前{t(ew_wave)}，交易信号={t(ew_sig)}')
    if ew_conf >= 0.5:
        rules.append(f'波浪置信度={ew_conf:.2f}，结构识别可靠')
    elif ew_conf >= 0.3:
        rules.append(f'波浪置信度={ew_conf:.2f}，结构仅供参考')

    # 12. 波动率
    rv_level = p.get('rv_level', '')
    rv_pct = _safe_float(p.get('rv_percentile', 0.5))
    har_dir = p.get('har_direction', '')
    if rv_level:
        rules.append(f'波动率: {t(rv_level)}(分位={rv_pct:.0%})，HAR预测{t(har_dir)}')
    if rv_level == '极高波动':
        rules.append('⚠️ 极高波动环境，建议减仓避险')
    elif rv_level == '极低波动' and har_dir == '扩张':
        rules.append('低波动+波动率扩张，可能酝酿趋势突破')

    # 12B. BEGE好坏环境分解
    bebe_regime_princ = str(p.get('bebe_regime', ''))
    bebe_ratio_princ = _safe_float(p.get('bebe_good_bad_ratio'), 1.0)
    bebe_vrp_princ = str(p.get('bebe_vrp_signal', 'neutral'))
    bebe_asym_princ = _safe_float(p.get('bebe_vol_asymmetry', 0.5))
    bebe_bad_t_princ = str(p.get('bebe_bad_vol_trend', 'stable'))
    bebe_good_t_princ = str(p.get('bebe_good_vol_trend', 'stable'))

    if bebe_regime_princ == 'bad_environment':
        rules.append(f'⚠️ BEGE坏环境主导(不对称{bebe_asym_princ:.0%})→下行风险高，仓位应显著降低')
    elif bebe_regime_princ == 'good_environment':
        rules.append(f'BEGE好环境主导(好坏比{bebe_ratio_princ:.2f})→风险偏好强，可适度进取')
    else:
        rules.append(f'BEGE环境平衡(好坏比{bebe_ratio_princ:.2f})')

    if bebe_vrp_princ == 'high_premium':
        rules.append('⚠️ VRP高溢价→市场定价恐慌，风控收紧')
    elif bebe_vrp_princ == 'negative_premium':
        rules.append('⚠️ VRP负溢价→异常信号，建议观望或轻仓')

    if bebe_bad_t_princ == 'rising' and bebe_good_t_princ == 'falling':
        rules.append('⚠️ BEGE分布偏移→坏波动率上升+好波动率下降，市场结构可能切换')

    # 12C. 缠论分析（缠中说禅108课）
    # 防狼术
    wolf_sig = str(p.get('chanlun_wolf_signal', ''))
    wolf_days = p.get('chanlun_wolf_days', 0) or 0
    wolf_pos = str(p.get('chanlun_wolf_position', ''))
    if wolf_sig == 'danger' and wolf_days > 10:
        rules.append(f'⚠️ 缠论防狼术：MACD双线0轴下{wolf_days}天→空头主导，不入场做多')
    elif wolf_sig == 'danger':
        rules.append(f'缠论防狼术：{wolf_pos}，谨慎做多')

    # 笔状态机
    stroke_state = str(p.get('chanlun_stroke_state', ''))
    stroke_label = str(p.get('chanlun_stroke_label', ''))
    if stroke_state:
        rules.append(f'缠论笔状态: {stroke_state}({stroke_label})')

    # 背驰
    div_type = str(p.get('chanlun_divergence_type', ''))
    div_strength = _safe_float(p.get('chanlun_divergence_strength'), 0)
    div_detail = str(p.get('chanlun_divergence_detail', ''))
    if div_type and div_type != 'None':
        if div_strength > 0.5:
            rules.append(f'⚠️ 缠论{div_type}(强度{div_strength:.1%})→{div_detail}→高概率反转信号')
        else:
            rules.append(f'缠论{div_type}(强度{div_strength:.1%})→{div_detail}')

    # 走势类型
    trend_t = str(p.get('chanlun_trend_type', ''))
    if '下跌趋势' in trend_t:
        rules.append(f'缠论走势: {trend_t}→顺势做空或观望')
    elif '上涨趋势' in trend_t:
        rules.append(f'缠论走势: {trend_t}→顺势做多')
    elif '盘整' in trend_t:
        rules.append(f'缠论走势: {trend_t}→中枢震荡策略(高抛低吸)')
    elif trend_t:
        rules.append(f'缠论走势: {trend_t}')

    # 中枢Zn
    zn_pos = str(p.get('chanlun_zn_position', ''))
    zn_pattern = str(p.get('chanlun_zn_pattern', ''))
    if '超卖' in zn_pos:
        rules.append(f'缠论Zn中枢{zn_pos}→超卖反弹机会')
    elif '超买' in zn_pos:
        rules.append(f'缠论Zn中枢{zn_pos}→超买回调风险')
    if '窄幅收敛' in zn_pattern:
        rules.append('缠论Zn窄幅收敛→即将变盘，注意突破方向')
    elif '宽幅震荡' in zn_pattern:
        rules.append('缠论Zn宽幅震荡→中枢扩展，方向不明确')

    # 多级别联立
    tf_align = str(p.get('chanlun_multi_tf_alignment', ''))
    tf_rec = str(p.get('chanlun_multi_tf_recommendation', ''))
    if '共振看多' in tf_align:
        rules.append(f'缠论多级别: {tf_align}→{tf_rec}')
    elif '共振看空' in tf_align:
        rules.append(f'缠论多级别: {tf_align}→{tf_rec}')
    elif tf_align and '单级别' not in tf_align:
        rules.append(f'缠论多级别: {tf_align}')
        if tf_rec:
            rules.append(f'  建议: {tf_rec}')

    # 线段破坏
    seg_destr = str(p.get('chanlun_last_segment_destruction', ''))
    if '标准破坏' in seg_destr:
        rules.append(f'缠论线段: {seg_destr}→趋势确认，跟随破坏方向')
    elif '背驰完成' in seg_destr:
        rules.append(f'缠论线段: {seg_destr}→背驰型终结，反转概率提升')

    # 中枢扩展
    zs_expanded = p.get('chanlun_zs_expanded', False)
    zs_upgraded_level = str(p.get('chanlun_zs_upgraded_level', ''))
    if zs_expanded:
        rules.append(f'缠论中枢扩展({zs_upgraded_level})→级别升级，中枢震荡加剧')

    # 小转大
    xzd_active = p.get('chanlun_xzd_active', False)
    xzd_dir = str(p.get('chanlun_xzd_direction', ''))
    xzd_conf = _safe_float(p.get('chanlun_xzd_confidence'), 0)
    xzd_reasons = str(p.get('chanlun_xzd_reasons', ''))
    if xzd_active and xzd_conf > 0.6:
        direction_cn = '看多反转' if xzd_dir == 'bullish' else '看空反转'
        rules.append(f'⚠️ 缠论小转大({direction_cn}, 置信{xzd_conf:.0%}): {xzd_reasons}')
    elif xzd_active:
        rules.append(f'缠论小转大预警(置信{xzd_conf:.0%}): {xzd_reasons}')

    # 12D. Busch 2560战法（安德烈布殊）
    b2560 = str(p.get('b2560_signal', ''))
    b2560_desc = str(p.get('b2560_description', ''))
    if b2560 == 'strong_buy':
        rules.append(f'Busch 2560买入: {b2560_desc}')
    elif b2560 == 'sell':
        rules.append(f'⚠️ Busch 2560卖出: {b2560_desc}')
    elif b2560 == 'hold_long':
        rules.append(f'Busch 2560持仓: {b2560_desc}')
    elif b2560 == 'hold_short':
        rules.append(f'Busch 2560观望: {b2560_desc}')

    b2560_weekly = str(p.get('b2560_weekly_signal', ''))
    if b2560_weekly == 'weekly_golden_cross':
        rules.append('Busch周线金叉(MA20/MA55)→中长期看多确认')
    elif b2560_weekly == 'weekly_dead_cross':
        rules.append('Busch周线死叉(MA20/MA55)→中长期看空确认')

    # 12E. 市场四阶段（彭冬初）
    mphase = str(p.get('mp_phase', ''))
    mp_conf = _safe_float(p.get('mp_confidence'), 0)
    mp_cred = _safe_float(p.get('mp_trend_credibility'), 0.5)
    if mphase and mphase != 'unknown':
        rules.append(f'市场阶段: {mphase}(置信{mp_conf:.0%}) 三势可信度={mp_cred:.0%}')
    if mphase == '拉升' and mp_conf > 0.6:
        rules.append('→拉升阶段顺势做多，仓位可适当放大')
    elif mphase == '盘头':
        rules.append('→盘头阶段注意减仓，警惕反转')
    elif mphase == '下跌':
        rules.append('→下跌阶段以空仓或做空为主')
    elif mphase == '筑底':
        rules.append('→筑底阶段轻仓试探，等待突破确认')

    # 12F. VR容量比率 + Busch量价代数
    vr_zone = str(p.get('vr_zone', ''))
    vr_sig = str(p.get('vr_signal', ''))
    vr_div = str(p.get('vr_divergence', ''))
    if vr_zone:
        rules.append(f'VR容量比率: {vr_zone}({vr_sig})')
    if '顶背离' in vr_div:
        rules.append('⚠️ VR顶背离→资金离场信号，注意减仓')
    elif '底背离' in vr_div:
        rules.append('VR底背离→资金进场信号，关注反弹')

    vpa_formula = str(p.get('vpa_formula', ''))
    vpa_action = str(p.get('vpa_action', ''))
    if vpa_formula:
        rules.append(f'Busch量价代数: {vpa_formula} → {vpa_action}')
    if p.get('vpa_is_extreme_vol', False):
        rules.append('⚠️ 天量警告: 成交量创60日新高→趋势衰竭预警')

    # 12G. 综合量价信号
    cva_sig = str(p.get('cva_composite_signal', ''))
    if cva_sig and '中性' not in cva_sig:
        rules.append(f'综合量价信号: {cva_sig}')
    vc_pat = str(p.get('vc_pattern', ''))
    vc_desc = str(p.get('vc_description', ''))
    if vc_pat and vc_pat != '未知':
        rules.append(f'量能形态: {vc_pat}({vc_desc})')

    # 12G-2. 九大量价关系
    v9r_name = str(p.get('v9r_name', ''))
    v9r_strength = str(p.get('v9r_strength', ''))
    v9r_action = str(p.get('v9r_action', ''))
    v9r_cons = p.get('v9r_consistency', {})
    if v9r_name and v9r_name != '量价不明':
        rules.append(f'量价关系: {v9r_name}({v9r_strength}) → {v9r_action}')
        if v9r_cons:
            cons_str = ', '.join([f'{k}={v}' for k, v in v9r_cons.items()])
            rules.append(f'  量价一致性: {cons_str}')

    # 12G-3. 主力资金检测 (MFI/VWAP/A/D)
    sm_sig = str(p.get('sm_smart_signal', ''))
    sm_mfi = _safe_float(p.get('sm_mfi', 50))
    sm_ad_div = str(p.get('sm_ad_divergence', ''))
    if sm_sig in ['strong_accumulation', 'accumulation']:
        rules.append(f'主力资金: 收集(累积) MFI={sm_mfi:.0f}')
    elif sm_sig in ['strong_distribution', 'distribution']:
        rules.append(f'⚠️ 主力资金: 派发(出货) MFI={sm_mfi:.0f}')
    if sm_ad_div:
        rules.append(f'主力A/D线背离: {sm_ad_div}')

    # 12G-4. 格兰维尔八准则
    g8_rule = p.get('g8_active_rule')
    g8_cat = str(p.get('g8_rule_category', 'none'))
    g8_desc = str(p.get('g8_description', ''))
    g8_land = str(p.get('g8_land_volume_zone', ''))
    if g8_rule and g8_cat != 'none':
        prefix = '格兰维尔买入' if 'buy' in g8_cat else '格兰维尔卖出' if 'sell' in g8_cat else '格兰维尔'
        rules.append(f'{prefix}: {g8_desc}')
    if g8_land:
        rules.append(f'格兰维尔地量分析: {g8_land}')

    # 12H. Busch DMI转折信号
    bdmi_sig = str(p.get('bdmi_signal', ''))
    bdmi_desc = str(p.get('bdmi_description', ''))
    if bdmi_sig == 'bullish_reversal':
        rules.append(f'Busch DMI转折买入: {bdmi_desc}')
    elif bdmi_sig == 'bearish_trend':
        rules.append(f'Busch DMI: {bdmi_desc}')
    if p.get('bdmi_big_momentum', False):
        rules.append('Busch大动量信号→大行情可能启动')

    # 12I. 增强艾略特波浪
    if p.get('ew_alternation_valid', False):
        rules.append('EW交替规则验证通过→波浪结构可信度提升')
    ew_ext = p.get('ew_extension_wave')
    if ew_ext:
        rules.append(f'EW延长浪检测: 浪{ew_ext}延长')
    if p.get('ew_is_diagonal', False):
        rules.append('⚠️ EW倾斜三角形→楔形反转预警，注意急速反向')
    ew_corr = str(p.get('ew_corrective_type', ''))
    if ew_corr and ew_corr != 'unknown':
        rules.append(f'EW修正浪形态: {ew_corr}')

    # 12J. NRB短线信号（Velez）
    nrb_sig = str(p.get('nrb_signal', ''))
    nrb_desc = str(p.get('nrb_description', ''))
    if nrb_sig != 'none' and nrb_sig:
        rules.append(f'NRB信号: {nrb_desc}')

    # 13. 贝叶斯总结
    bayes_post = _safe_float(p.get('bayes_fused_posterior', 0.5))
    bayes_ent = _safe_float(p.get('bayes_fused_entropy', 0.5))
    dims = p.get('bayes_dimensions_active', 0)
    dim_contrib = p.get('bayes_dimension_contributions', [])
    rules.append(f'贝叶斯后验={bayes_post:.4f}({t(p.get("bayes_fused_signal","neutral"))})，熵={bayes_ent:.4f}，{dims}个维度共振')
    if dim_contrib:
        rules.append(f'维度贡献: {dim_contrib}')
    if p.get('bayes_entropy_high', False):
        rules.append('⚠️ 贝叶斯高熵警告: 信号不确定性大，建议减仓或观望')

    # 14. 动量
    mom_res = p.get('momentum_resonance', False)
    mom_dir = p.get('momentum_direction', '')
    roc_short = _safe_float(p.get('roc_short', 0))
    if mom_res:
        rules.append(f'动量共振({t(mom_dir)})，ROC10={roc_short:.1f}')

    # 15. 格兰维尔
    gs = str(p.get('granville_signal', ''))
    if gs and gs != '无' and gs != 'None':
        rules.append(f'格兰维尔信号: {gs}')

    return rules


def _derive_risk_snapshot(p):
    """从指标派生轻量风险快照——用于本地风控参考。
    完整的 behavioral_guardrails 和 busch_position_sizing 需配合实盘账户数据调用。
    - behavioral_guardrails(trade_history, current_drawdown_pct, consecutive_losses)
    - busch_position_sizing(account_capital, accumulated_profit, base_risk_pct)
    """
    risk_factors = []
    risk_score = 0

    # 波动率风险
    rv_level = str(p.get('rv_level', ''))
    if rv_level in ['high', 'extreme']:
        risk_factors.append(f'高波动率({rv_level})')
        risk_score += 2

    # 趋势冲突
    chan_dir = str(p.get('chanlun_trend_direction', ''))
    mom_dir = str(p.get('momentum_direction', ''))
    if chan_dir == 'down' and mom_dir == 'up':
        risk_factors.append('缠论空+动量多→趋势冲突')
        risk_score += 1
    elif chan_dir == 'up' and mom_dir == 'down':
        risk_factors.append('缠论多+动量空→趋势冲突')
        risk_score += 1

    # 背离风险
    div_type = str(p.get('chanlun_divergence_type', ''))
    if '顶背驰' in div_type:
        risk_score += 1
        risk_factors.append('顶背驰→反转风险')

    # 主力派发
    sm_sig = str(p.get('sm_smart_signal', ''))
    if 'distribution' in sm_sig:
        risk_factors.append('主力资金派发')
        risk_score += 1

    # 中枢扩展风险
    if p.get('chanlun_zs_expanded', False):
        risk_factors.append('中枢扩展→方向不确定')
        risk_score += 1

    # 市场阶段风险
    mphase = str(p.get('mp_phase', ''))
    if mphase == '盘头':
        risk_factors.append('盘头阶段')
        risk_score += 1
    elif mphase == '下跌':
        risk_factors.append('下跌阶段')
        risk_score += 2

    if risk_score >= 4:
        risk_level = 'high'
    elif risk_score >= 2:
        risk_level = 'medium'
    else:
        risk_level = 'low'

    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'risk_factors': risk_factors,
        'note': '完整风控(行为熔断+Busch仓位)请在账户层调用advanced_indicators.behavioral_guardrails和busch_position_sizing',
    }


def _compute_posterior_confidence_adjustment(p):
    """基于后验分布宽度的置信度调整。
    窄HDI→高置信度（模型精确）；宽HDI→低置信度（模型不确定）。
    返回: {adjustment_factor, posterior_width_category, entropy_penalty}
    """
    bayes_entropy = _safe_float(p.get('bayes_fused_entropy', 0.5))
    posterior = _safe_float(p.get('bayes_fused_posterior', 0.5))
    dims_active = p.get('bayes_dimensions_active', 5) or 5

    # 熵惩罚: 高熵→后验分散, 置信度应下降
    if bayes_entropy > 0.9:
        entropy_penalty = 0.5
        width_cat = 'very_wide'
    elif bayes_entropy > 0.75:
        entropy_penalty = 0.7
        width_cat = 'wide'
    elif bayes_entropy > 0.5:
        entropy_penalty = 0.85
        width_cat = 'moderate'
    elif bayes_entropy > 0.3:
        entropy_penalty = 0.95
        width_cat = 'narrow'
    else:
        entropy_penalty = 1.0
        width_cat = 'very_narrow'

    # 维度不足惩罚: 共振维度少→后验可能不稳健
    dim_density = min(dims_active / 15.0, 1.0)
    if dim_density < 0.3:
        dim_penalty = 0.7
    elif dim_density < 0.5:
        dim_penalty = 0.85
    else:
        dim_penalty = 1.0

    adjustment = entropy_penalty * dim_penalty

    # 后验极端性: 后验接近0或1时，窄HDI意味着强信号
    posterior_extremity = abs(posterior - 0.5) * 2  # 0到1，0=50/50，1=100%确定
    if width_cat in ('narrow', 'very_narrow') and posterior_extremity > 0.7:
        # 窄后验+极端概率 = 高可信度，不做额外惩罚
        pass
    elif width_cat == 'very_wide' and posterior_extremity > 0.5:
        # 宽后验但声称极端 = 矛盾，额外惩罚
        adjustment *= 0.7

    return {
        'adjustment_factor': round(adjustment, 4),
        'posterior_width_category': width_cat,
        'entropy_penalty': round(entropy_penalty, 4),
        'dimension_penalty': round(dim_penalty, 4),
        'note': f'后验宽度={width_cat}, 熵={bayes_entropy:.3f}',
    }


def _detect_sigma_events(p):
    """检测各指标相对于历史分布是否处于极端区域(>3σ)。
    使用本地指标中的sigma_event_check函数。

    返回: {has_sigma_event, sigma_flags, aggregate_penalty}
    """
    sigma_flags = []

    # 定义检查项: (指标名, 当前值, 均值, 标准差)
    checks = []

    # RSI极端检查
    rsi = _safe_float(p.get('rsi', 50))
    checks.append(('RSI', rsi, 50, 15))
    if rsi > 90:
        sigma_flags.append({'indicator': 'RSI', 'value': rsi, 'sigma': (rsi - 50) / 15,
                            'direction': 'extreme_overbought', 'action': '该维度降权至0'})
    elif rsi < 10:
        sigma_flags.append({'indicator': 'RSI', 'value': rsi, 'sigma': (50 - rsi) / 15,
                            'direction': 'extreme_oversold', 'action': '该维度降权至0'})

    # 量比极端
    vol_ratio = _safe_float(p.get('vol_ratio', 1))
    checks.append(('vol_ratio', vol_ratio, 1, 0.5))
    if vol_ratio > 3.5:
        sigma_flags.append({'indicator': '量比', 'value': vol_ratio, 'sigma': (vol_ratio - 1) / 0.5,
                            'direction': 'extreme_volume', 'action': '天量异常,量价信号降权'})

    # VR极端
    vr_value = _safe_float(p.get('vr_value', 100))
    checks.append(('VR', vr_value, 150, 80))
    if vr_value > 500:
        sigma_flags.append({'indicator': 'VR容量比率', 'value': vr_value, 'sigma': (vr_value - 150) / 80,
                            'direction': 'extreme_overbought', 'action': 'VR极端超买,反转风险极高'})
    elif vr_value < 40:
        sigma_flags.append({'indicator': 'VR容量比率', 'value': vr_value, 'sigma': (150 - vr_value) / 80,
                            'direction': 'extreme_oversold', 'action': 'VR极端超卖,关注反转'})

    # 波动率极端
    rv_composite = _safe_float(p.get('rv_composite', 0.02))
    rv_mean = 0.025  # 典型A股日波动率约2.5%
    rv_std = 0.015
    rv_sigma = (rv_composite - rv_mean) / rv_std if rv_std > 0 else 0
    checks.append(('RV', rv_composite, rv_mean, rv_std))
    if abs(rv_sigma) > 3:
        direction = 'extreme_high_vol' if rv_sigma > 0 else 'extreme_low_vol'
        sigma_flags.append({'indicator': '已实现波动率', 'value': rv_composite, 'sigma': rv_sigma,
                            'direction': direction, 'action': '极端波动,仓位降至正常30%'})

    # ADX极端
    adx = _safe_float(p.get('dmi_adx', 20))
    if adx > 60:
        sigma_flags.append({'indicator': 'ADX', 'value': adx, 'sigma': (adx - 25) / 12,
                            'direction': 'extreme_trend', 'action': '极强趋势但可能尾声'})

    # BEGE极端
    bebe_ratio = _safe_float(p.get('bebe_good_bad_ratio'), 1.0)
    if bebe_ratio < 0.3:
        sigma_flags.append({'indicator': 'BEGE好坏比', 'value': bebe_ratio, 'sigma': (1.0 - bebe_ratio) / 0.25,
                            'direction': 'extreme_bad_env', 'action': '极端坏环境,强制观望'})
    elif bebe_ratio > 3.0:
        sigma_flags.append({'indicator': 'BEGE好坏比', 'value': bebe_ratio, 'sigma': (bebe_ratio - 1.0) / 0.25,
                            'direction': 'extreme_good_env', 'action': '极端好环境但警惕均值回归'})

    has_sigma = len(sigma_flags) > 0
    # 每个sigma事件降低置信度15%, 但最多降50%
    aggregate_penalty = max(0.5, 1.0 - len(sigma_flags) * 0.15) if has_sigma else 1.0

    return {
        'has_sigma_event': has_sigma,
        'sigma_flags': sigma_flags,
        'aggregate_penalty': aggregate_penalty,
        'note': f'检测到{len(sigma_flags)}个sigma极端事件' if has_sigma else '无sigma极端事件',
    }


def _bayesian_strategy_bandit(p, strategy_variants=None):
    """Beta-Binomial Bandit策略比较框架。
    使用Thompson Sampling比较多个策略变体（如不同止损倍数、不同持仓周期）。
    在当前指标环境下模拟各策略变体的期望收益。

    strategy_variants: list of {name, wins, trials, rate_prior}
    返回: {best_strategy, regret_bound, variant_scores}
    """
    if strategy_variants is None:
        # 默认比较三种止损策略
        atr = _safe_float(p.get('vol_atr14', 0))
        if atr <= 0:
            atr = _safe_float(p.get('current_price', 10)) * 0.02

        strategy_variants = [
            {'name': 'tight_stop_1.5ATR', 'wins': 12, 'trials': 30, 'rate_prior': 0.4},
            {'name': 'standard_stop_2ATR', 'wins': 18, 'trials': 30, 'rate_prior': 0.6},
            {'name': 'wide_stop_3ATR', 'wins': 22, 'trials': 30, 'rate_prior': 0.7},
        ]

    variant_scores = []
    for sv in strategy_variants:
        # 当前环境调整: 高波动→宽止损更优; 低波动→紧止损更优
        rv_level = str(p.get('rv_level', 'normal'))
        env_bonus = 0
        if 'high' in rv_level or 'extreme' in rv_level:
            if 'wide' in sv['name']:
                env_bonus = 0.1  # 高波动环境中宽止损有额外优势
            elif 'tight' in sv['name']:
                env_bonus = -0.05
        elif 'low' in rv_level:
            if 'tight' in sv['name']:
                env_bonus = 0.05

        # Beta-Binomial后验: Beta(prior_a + wins, prior_b + trials - wins + 1)
        prior_a = sv['rate_prior'] * 5  # 先验强度=5次虚拟试验
        prior_b = (1 - sv['rate_prior']) * 5
        posterior_alpha = prior_a + sv['wins']
        posterior_beta = prior_b + sv['trials'] - sv['wins']
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

        # 用后验均值+环境调整作为得分
        score = posterior_mean + env_bonus
        variant_scores.append({
            'name': sv['name'],
            'posterior_mean': round(posterior_mean, 4),
            'env_adjustment': env_bonus,
            'score': round(score, 4),
            'alpha': round(posterior_alpha, 2),
            'beta': round(posterior_beta, 2),
        })

    variant_scores.sort(key=lambda x: x['score'], reverse=True)
    best = variant_scores[0]
    second_best = variant_scores[1] if len(variant_scores) > 1 else variant_scores[0]

    # Regret bound: 最优与次优的得分差
    regret_bound = best['score'] - second_best['score']

    return {
        'best_strategy': best['name'],
        'regret_bound': round(regret_bound, 4),
        'recommendation': f'当前环境建议{best["name"]}' if regret_bound > 0.05 else '策略间差异不显著,选任意即可',
        'variant_scores': variant_scores,
    }


def _bayesian_sequential_update(prior_prob, likelihood_ratio, n_observations=1):
    """贝叶斯序贯更新框架。
    posterior_odds = prior_odds × likelihood_ratio^n

    prior_prob: 先验概率 (0到1)
    likelihood_ratio: P(E|H₁)/P(E|H₀), >1支持H₁, <1支持H₀
    n_observations: 独立观测次数

    返回: {prior_odds, posterior_odds, posterior_prob, evidence_strength}
    """
    prior_prob = max(0.01, min(0.99, prior_prob))
    prior_odds = prior_prob / (1 - prior_prob)

    posterior_odds = prior_odds * (likelihood_ratio ** n_observations)
    posterior_prob = posterior_odds / (1 + posterior_odds)

    # 证据强度分类 (参考Kass & Raftery, 1995)
    log_bayes_factor = n_observations * np.log(max(likelihood_ratio, 1e-10))
    if abs(log_bayes_factor) < 1:
        evidence_strength = 'barely_worth_mentioning'
    elif abs(log_bayes_factor) < 3:
        evidence_strength = 'substantial'
    elif abs(log_bayes_factor) < 5:
        evidence_strength = 'strong'
    elif abs(log_bayes_factor) < 10:
        evidence_strength = 'very_strong'
    else:
        evidence_strength = 'decisive'

    return {
        'prior_prob': round(prior_prob, 4),
        'prior_odds': round(prior_odds, 4),
        'likelihood_ratio': round(likelihood_ratio, 4),
        'n_observations': n_observations,
        'posterior_odds': round(posterior_odds, 4),
        'posterior_prob': round(posterior_prob, 4),
        'evidence_strength': evidence_strength,
        'log_bayes_factor': round(log_bayes_factor, 4),
    }


def _calibration_framework(predicted_probs, actual_outcomes, n_bins=5):
    """概率校准评估框架。
    比较预测概率与实际频率,检测过度自信或信心不足。

    predicted_probs: 预测概率列表
    actual_outcomes: 实际二元结果列表 (1=正确, 0=错误)
    n_bins: 分箱数

    返回: {calibration_error, overconfidence_score, reliability_diagram_bins}
    """
    if len(predicted_probs) < 10:
        return {'calibration_error': None, 'note': '数据不足(需≥10次预测)', 'overconfidence_score': None}

    predicted = np.array(predicted_probs)
    actual = np.array(actual_outcomes)

    # 分箱校准
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if mask.sum() > 0:
            avg_pred = predicted[mask].mean()
            actual_freq = actual[mask].mean()
            bins.append({
                'bin_range': f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}',
                'count': int(mask.sum()),
                'avg_predicted': round(avg_pred, 4),
                'actual_frequency': round(actual_freq, 4),
                'bias': round(avg_pred - actual_freq, 4),
            })

    # 校准误差 (ECE - Expected Calibration Error)
    if bins:
        ece = sum(b['count'] * abs(b['bias']) for b in bins) / sum(b['count'] for b in bins)
    else:
        ece = None

    # 过度自信检测: 高置信度(>0.7)区间的实际频率是否显著低于预测
    high_conf_mask = predicted > 0.7
    if high_conf_mask.sum() > 0:
        high_conf_pred = predicted[high_conf_mask].mean()
        high_conf_actual = actual[high_conf_mask].mean()
        overconfidence = high_conf_pred - high_conf_actual
    else:
        overconfidence = None

    return {
        'calibration_error': round(ece, 4) if ece is not None else None,
        'overconfidence_score': round(overconfidence, 4) if overconfidence is not None else None,
        'reliability_bins': bins,
        'interpretation': (
            '校准良好(ECE<0.05)' if ece and ece < 0.05
            else '轻微过度自信(ECE=0.05-0.10)' if ece and ece < 0.10
            else '显著过度自信(ECE>0.10)—建议系统性降置信度' if ece
            else '数据不足'
        ),
        'note': '历史校准评估; 持续更新以检测策略退化',
    }


def _integrated_risk_assessment(p):
    """综合贝叶斯风险评估：整合后验宽度、sigma事件、BEGE、市场阶段。
    这是对原有 _derive_risk_snapshot 的增强版, 用于更精细的风险分解。
    """
    base_risk = _derive_risk_snapshot(p)

    # 后验宽度评估
    posterior_adj = _compute_posterior_confidence_adjustment(p)

    # Sigma事件检测
    sigma_events = _detect_sigma_events(p)

    # BEGE环境综合
    bebe_regime = str(p.get('bebe_regime', 'neutral'))
    bebe_ratio = _safe_float(p.get('bebe_good_bad_ratio'), 1.0)
    bebe_bad_trend = str(p.get('bebe_bad_vol_trend', 'stable'))
    bebe_good_trend = str(p.get('bebe_good_vol_trend', 'stable'))

    # 综合后验风险分
    composite_risk = base_risk['risk_score']
    if posterior_adj['adjustment_factor'] < 0.7:
        composite_risk += 2  # 后验高度不确定
    if sigma_events['has_sigma_event']:
        composite_risk += len(sigma_events['sigma_flags'])
    if bebe_bad_trend == 'rising' and bebe_good_trend == 'falling':
        composite_risk += 3  # 分布偏移

    # 最终置信度调整因子
    final_adj = posterior_adj['adjustment_factor'] * sigma_events['aggregate_penalty']
    if bebe_regime == 'bad_environment':
        final_adj *= 0.65
    elif bebe_regime == 'good_environment' and bebe_ratio > 1.5:
        final_adj *= 1.05  # 好环境下轻微提升

    return {
        'composite_risk_score': composite_risk,
        'risk_level': 'high' if composite_risk >= 6 else 'medium' if composite_risk >= 3 else 'low',
        'confidence_adjustment': round(final_adj, 4),
        'posterior_width': posterior_adj,
        'sigma_events': sigma_events,
        'bebe_risk': {
            'regime': bebe_regime,
            'good_bad_ratio': round(bebe_ratio, 3),
            'bad_vol_trend': bebe_bad_trend,
            'good_vol_trend': bebe_good_trend,
        },
        'base_risk': base_risk,
    }
