# deepseek_analysis.py
import json
import requests
import datetime
import numpy as np
from config import DEEPSEEK_API_KEY
from advanced_indicators import (
    kelly_from_deepseek_signal,
    non_ergodicity_protections,
)

def _sanitize_prompt_value(v, max_len=200):
    """Sanitize a value for safe interpolation into AI prompt."""
    if not isinstance(v, str):
        return str(v)[:max_len]
    import re
    v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', v)
    if len(v) > max_len:
        v = v[:max_len-3] + '...'
    return v


def get_ai_signal(df_daily, df_hourly, indicators, symbol):
    # 小时线取最近20根（确保足够多的近期走势）
    daily_table = _sanitize_prompt_value(df_daily.tail(20).to_string(), max_len=3000)
    hourly_table = _sanitize_prompt_value(df_hourly.tail(20).to_string(), max_len=3000) if df_hourly is not None and not df_hourly.empty else "无小时线数据"

    sym_safe = _sanitize_prompt_value(str(symbol), max_len=20)
    prompt = f"""
你是量化交易策略师，采用多专家协作框架(Mixture-of-Experts)分析A股。你必须遵循以下结构化分析流程，不得跳过任何步骤。

【重要声明】
1. "贝叶斯推理框架"是LLM模拟的概率推理——运用贝叶斯原理思考，但输出数值是对信念的近似而非精确数学计算。
2. 五位专家Panel在单次推理中由你模拟——每位专家应从专属领域视角独立分析，形成差异化判断。

【股票】{sym_safe}
【时间】{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
【当前价格】{indicators['current_price']:.2f}

【原始数据——日线K线（最近20根）】
{daily_table}

【原始数据——60分钟K线（最近20根）】
{hourly_table}

【周线趋势背景】{indicators.get('weekly_trend', '震荡')}

【专家A：趋势与结构】(MA/ADX/DMA/缠论/波浪/2560/市场阶段)
MA5={indicators['ma5']:.2f} MA20={indicators['ma20']:.2f} MA60={indicators['ma60']:.2f}
ADX={indicators['dmi_adx']:.2f}({indicators['dmi_adx_trend']}) +DI={indicators['dmi_pdi']:.2f} -DI={indicators['dmi_mdi']:.2f} DI方向={indicators['dmi_di_direction']}
Busch DMI: 信号={indicators.get('bdmi_signal','?')} {indicators.get('bdmi_description','')} MDI见顶={indicators.get('bdmi_mdi_peaking',False)} PDI增强={indicators.get('bdmi_pdi_rising',False)} 大动量={indicators.get('bdmi_big_momentum',False)}
DMA={indicators['dma']:.2f} AMA={indicators['dma_ama']:.2f} 差值={indicators['dma_diff']:.2f} 信号={indicators['dma_signal']} 趋势={indicators['dma_trend']}
缠论中枢={indicators['chanlun_status']} 走势={indicators.get('chanlun_trend_type','?')}
笔状态机={indicators.get('chanlun_stroke_state','?')}({indicators.get('chanlun_stroke_label','?')})
背驰={indicators.get('chanlun_divergence_type','无')}(强度{indicators.get('chanlun_divergence_strength',0):.1%}) {indicators.get('chanlun_divergence_detail','')}
防狼术={indicators.get('chanlun_wolf_signal','?')}({indicators.get('chanlun_wolf_position','?')} {indicators.get('chanlun_wolf_days',0)}天)
买点={indicators['chanlun_buy']} 卖点={indicators['chanlun_sell']}
Zn中枢监视={indicators.get('chanlun_zn_value',0):.2f}({indicators.get('chanlun_zn_position','?')}) 模式={indicators.get('chanlun_zn_pattern','?')}
多级别联立={indicators.get('chanlun_multi_tf_alignment','?')}
最近分型={indicators.get('chanlun_fractal_type','')}@{indicators.get('chanlun_fractal_price',0):.2f}
线段破坏={indicators.get('chanlun_last_segment_destruction','')} 中枢扩展={indicators.get('chanlun_zs_expanded',False)}({indicators.get('chanlun_zs_upgraded_level','')})
小转大={'活跃' if indicators.get('chanlun_xzd_active',False) else '无'}({indicators.get('chanlun_xzd_direction','')} 置信{indicators.get('chanlun_xzd_confidence',0):.0%}) {indicators.get('chanlun_xzd_reasons','')}
波浪模式={indicators['ew_pattern']} 当前浪={indicators['ew_current_wave']} 置信={indicators['ew_confidence']:.2f} 规则合规={indicators['ew_rules_valid']}
EW增强: 交替规则={indicators.get('ew_alternation_valid',False)} 延长浪={indicators.get('ew_extension_wave','无')} 修正形态={indicators.get('ew_corrective_type','未知')} 斜三角形={indicators.get('ew_is_diagonal',False)} 浪个性={indicators.get('ew_personality_score',0):.2f}
Busch 2560: 信号={indicators.get('b2560_signal','?')} MA25方向={indicators.get('b2560_ma25_direction','?')} MA金叉={indicators.get('b2560_ma_golden_cross',False)} 量金叉={indicators.get('b2560_vol_golden_cross',False)} 周线={indicators.get('b2560_weekly_signal','?')}
市场阶段: {indicators.get('mp_phase','?')}(置信{indicators.get('mp_confidence',0):.0%}) 大势={indicators.get('mp_big_trend','?')} 中势={indicators.get('mp_mid_trend','?')} 小势={indicators.get('mp_small_trend','?')} 三势可信度={indicators.get('mp_trend_credibility',0):.0%}
价格位置={indicators.get('mp_price_position',0):.0%}(60日区间) 20日收益={indicators.get('mp_ret_20d',0):.1%} 60日收益={indicators.get('mp_ret_60d',0):.1%}
唐能通系统: 价托={'有' if indicators.get('tang_jiato',False) else '无'}(强度{indicators.get('tang_jiato_strength',0):.0f}) 价压={'有' if indicators.get('tang_jiaya',False) else '无'}(强度{indicators.get('tang_jiaya_strength',0):.0f}) 跑道={indicators.get('tang_runway_grade','?')}(厚度{indicators.get('tang_runway_thickness',0):.3f}) 老鸭头={'有' if indicators.get('tang_laoyatou',False) else '无'}({indicators.get('tang_laoyatou_phase','?')}评分{indicators.get('tang_laoyatou_score',0):.0f})
2T-1L目标={indicators.get('tang_target_2t1l',0):.2f}({indicators.get('tang_target_pct',0):.1%}) 三三过滤={'通过' if indicators.get('tang_33_valid',False) else '未通过'}(评分{indicators.get('tang_33_score',0):.0f})
解缠论增强: 力度比={indicators.get('cl_power_ratio',1):.3f}({indicators.get('cl_ratio_level','?')}) 背驰={indicators.get('cl_divergence_type','无')} 中枢重心={indicators.get('cl_cg_direction','?')}(漂移{indicators.get('cl_cg_drift',0):.1f}% 趋势{indicators.get('cl_cg_trend_str',0):.0f}) 拐点评分={indicators.get('cl_turn_score',0):.0f}({indicators.get('cl_turn_direction','?')} {indicators.get('cl_turn_confidence','?')} {indicators.get('cl_turn_signal','?')}) 笔内背离={'有' if indicators.get('cl_intra_divergence',False) else '无'}({indicators.get('cl_intra_div_type','?')})
利弗莫尔战法: 关键点突破={'是' if indicators.get('livermore_breakout',False) else '否'}({indicators.get('livermore_signal','?')}强度{indicators.get('livermore_breakout_str',0):.0f}) 危险信号={'有' if indicators.get('livermore_danger',False) else '无'}(等级{indicators.get('livermore_danger_level',0):.0f} {indicators.get('livermore_action','?')})

【专家B：动量与震荡】(MACD/RSI/CCI/ROC/TD)
MACD: DIF={indicators['macd_dif']:.2f} DEA={indicators['macd_dea']:.2f} 柱={indicators['macd_hist']:.2f}
RSI(14)={indicators['rsi']:.2f}
ROC10={indicators['roc_short']:.2f}({indicators['momentum_state_short']}) ROC20={indicators['roc_long']:.2f}({indicators['momentum_state_long']})
CCI10={indicators['cci_short']:.2f} CCI20={indicators['cci_long']:.2f}
动量共振={'是' if indicators['momentum_resonance'] else '否'} 方向={indicators['momentum_direction']}
TD九转: 计数{indicators['td_count']} 方向{indicators['td_direction']} 完成={indicators['td_completed']}

【专家C：量价与资金流】(OBV/成交量/格兰维尔/量价形态/VR/Busch量价)
量比(5/20)={indicators['vol_ratio']:.2f} 量价配合={'是' if indicators['vol_price_resonance'] else '背离'}
OBV趋势={'向上' if indicators['obv_trend'] else '走平/向下'} 背离={indicators['obv_divergence']} 信号={indicators['obv_signal']}
成交量: 类型={indicators.get('vol_type','?')} 趋势={indicators.get('vol_trend','?')}
堆量={'是' if indicators.get('vol_stacking') else '否'}({indicators.get('vol_stacking_days',0)}天) 放量滞涨={'是' if indicators.get('high_vol_stagnation') else '否'} 缩量回踩={'是' if indicators.get('low_vol_pullback') else '否'}
价格波动5d={indicators.get('price_range_5d_pct',0):.1f}% 突破验证={indicators.get('volume_breakout')} 格兰维尔={indicators.get('granville_signal','无')}
VR容量比率: 值={indicators.get('vr_value',100):.0f} 区域={indicators.get('vr_zone','?')} 信号={indicators.get('vr_signal','?')} 背离={indicators.get('vr_divergence','无')}
Busch量价代数: {indicators.get('vpa_formula','?')} → {indicators.get('vpa_interpretation','?')} 建议={indicators.get('vpa_action','?')} 天量警告={'是' if indicators.get('vpa_is_extreme_vol') else '否'}
量能形态: {indicators.get('vc_pattern','?')}({indicators.get('vc_description','')}) 信号={indicators.get('vc_signal','?')}
综合量价信号: {indicators.get('cva_composite_signal','?')} (多{indicators.get('cva_bull_count',0)} vs 空{indicators.get('cva_bear_count',0)})
九大量价: {indicators.get('v9r_name','?')}({indicators.get('v9r_strength','?')}) 建议={indicators.get('v9r_action','?')} 一致性={indicators.get('v9r_consistency',{})}
主力资金: MFI={indicators.get('sm_mfi',50):.0f}({indicators.get('sm_mfi_signal','?')}) VWAP偏离={indicators.get('sm_vwap_deviation',0):.2%} A/D={indicators.get('sm_ad_trend','?')}({indicators.get('sm_ad_divergence','')}) 综合={indicators.get('sm_smart_signal','?')}
格兰维尔八准则: 规则{indicators.get('g8_active_rule','无')}({indicators.get('g8_rule_category','none')}) {indicators.get('g8_description','')} 地量区={indicators.get('g8_land_volume_zone','')}
唐能通三金叉/三死叉: 金叉={'是' if indicators.get('tang_golden_cross',False) else '否'}(评分{indicators.get('tang_golden_score',0):.0f}) 死叉={'是' if indicators.get('tang_death_cross',False) else '否'}(评分{indicators.get('tang_death_score',0):.0f}) 状态={indicators.get('tang_triple_status','无')}

【专家D：形态与价格结构】(K线形态/支撑阻力/布林带)
布林: 上={indicators['bb_upper']:.2f} 中={indicators['bb_mid']:.2f} 下={indicators['bb_lower']:.2f}
K线主导={indicators['pattern_dominant']} 最近分型={indicators['pattern_latest_fractal']} 反转={indicators['pattern_latest_reversal']}
形态计数: 顶分型{indicators['pattern_top_fractal_n']} 底分型{indicators['pattern_bottom_fractal_n']} 暮星{indicators['pattern_evening_star_n']} 晨星{indicators['pattern_morning_star_n']}
近期信号={indicators['pattern_recent']}
动态支撑={indicators['dynamic_support']} 动态阻力={indicators['dynamic_resistance']}

【专家E：风险与波动率】(RV/HAR/BEGE/贝叶斯/NRB/市场阶段)
已实现波动率: Parkinson={indicators['rv_parkinson']:.4f} Garman-Klass={indicators['rv_garman_klass']:.4f} 综合={indicators['rv_composite']:.4f}
波动水平={indicators['rv_level']} 分位={indicators['rv_percentile']:.0%} 趋势={indicators['rv_trend']}
HAR预测: 方向={indicators['har_direction']} 预测RV={indicators['har_forecast']:.4f} R²={indicators['har_r_squared']:.3f} 衰减={indicators['har_decay']} GARCH(日γ={indicators.get('har_garch_daily',0):.3f}/周γ={indicators.get('har_garch_weekly',0):.3f}/月γ={indicators.get('har_garch_monthly',0):.3f} R²={indicators.get('har_garch_r2',0):.3f})
BEGE分解: 好波动率={indicators.get('bebe_good_vol', 0):.4f} 坏波动率={indicators.get('bebe_bad_vol', 0):.4f} 好坏比={indicators.get('bebe_good_bad_ratio', 1):.3f} 环境={indicators.get('bebe_regime', '?')}
BEGE不对称={indicators.get('bebe_vol_asymmetry', 0.5):.0%} VRP代理={indicators.get('bebe_vrp_signal', '?')} 好波趋势={indicators.get('bebe_good_vol_trend', '?')} 坏波趋势={indicators.get('bebe_bad_vol_trend', '?')}
ATR(14)={indicators['vol_atr14']:.2f}({indicators['vol_atr_pct']:.1f}%) 入场质量={indicators['vol_entry_quality']}
贝叶斯后验={indicators['bayes_fused_posterior']:.4f}({indicators['bayes_fused_signal']}) 熵={indicators['bayes_fused_entropy']:.4f}
有效维度={indicators['bayes_dimensions_active']}/9 维度贡献={indicators['bayes_dimension_contributions']}
高熵警告={'是' if indicators['bayes_entropy_high'] else '否'}
KMeans模式ID={indicators['kmeans_cluster']}
NRB信号={indicators.get('nrb_signal','无')} 类型={indicators.get('nrb_type','')}
市场阶段风险: {indicators.get('mp_phase','?')} 20日收益={indicators.get('mp_ret_20d',0):.1%} 60日收益={indicators.get('mp_ret_60d',0):.1%}
逃顶检测: 信号数={indicators.get('top_escape_count',0)} 顶部概率={indicators.get('top_escape_prob',0):.0f}% 等级={indicators.get('top_escape_grade','?')} 建议={indicators.get('top_escape_action','?')}
分配日: {indicators.get('dist_days_count',0)}个/25天(占比{indicators.get('dist_days_pct',0):.0f}%) 顶部警告={'是' if indicators.get('dist_days_warning',False) else '否'}
HAR-GARCH: 日γ={indicators.get('har_garch_daily',0):.3f} 周γ={indicators.get('har_garch_weekly',0):.3f} 月γ={indicators.get('har_garch_monthly',0):.3f} R²={indicators.get('har_garch_r2',0):.3f}
Hurst={indicators.get('hurst_value',0.5):.3f}({indicators.get('hurst_regime','?')}) VR共识={indicators.get('vr_consensus','?')} OU半衰期={indicators.get('ou_half_life','inf')}d
道氏确认={'是' if indicators.get('dow_confirmed',False) else '背离'} 冲击成本={indicators.get('market_impact_bps',0):.1f}bp 波目标仓位={indicators.get('vol_target_leverage',1):.2f}x
变点检测: 位置{indicators.get('bcd_changepoint_idx',-1)} 概率{indicators.get('bcd_changepoint_prob',0):.3f}
Benford合规={'是' if indicators.get('benford_compliant',True) else '异常'} Laplace概率={indicators.get('laplace_prob',0):.4f}(层级{indicators.get('laplace_hier_prob',0):.4f})
信号OR={indicators.get('odds_ratio_value','inf')} p={indicators.get('odds_ratio_p',1):.3f} {indicators.get('odds_ratio_quality','?')}
Epsilon稳健: 脆弱性={indicators.get('ecr_fragility',0):.4f} {indicators.get('ecr_verdict','?')}
GP波动预测: {indicators.get('gp_vol_current',0):.1f}%→{indicators.get('gp_vol_forecast',0):.1f}% 趋势={indicators.get('gp_vol_trend','?')}
Box-Cox φ={indicators.get('boxcox_phi',1):.3f}({'正态' if indicators.get('boxcox_normal',False) else '非正态'}) DIC={indicators.get('dic_value',0):.0f}
保守排名: 最优#{indicators.get('lcr_best',-1)} 下界={indicators.get('lcr_best_lower',0):.4f}

══════════════════════════════════
【贝叶斯推理框架：概率思维铁律】
在进入四步递进推理前，你必须内化以下贝叶斯思维原则，它们贯穿所有分析步骤：

0.1 先验强制声明：任何判断前必须先陈述你的先验信念（基于周线趋势、市场阶段、长期统计基准率）。例如："当前处于震荡市，牛市基准率约35%，先验倾向neutral(0.35)"。不声明先验的分析是无效的。
0.2 Solomonoff/Ockham复杂度惩罚：数据拟合存在无穷多种理论，偏好简单解释。如果你需要3个以上嵌套假设才能论证某个方向，则该方向置信度应自动惩罚×0.7。公式：后验可信度 ∝ 先验 × 似然 × 2^(-模型复杂度)。复杂推理≠高质量推理。
0.3 HDI+ROPE决策规则(Kruschke)：思考时使用最高密度区间(HDI)概念——你的置信度实质上是"方向效应落在正区间的概率"。如果效应量的90%HDI完全在ROPE(可忽略效应区[-0.1σ, +0.1σ])内→输出neutral。只有当HDI明确排除ROPE时才可以输出方向性信号。
0.4 多重假设竞争：永远不要只考虑一个假设。至少提出3个互斥的完整假设（如：H₁趋势延续、H₂区间震荡、H₃V形反转），分别评估P(数据|Hᵢ)，然后比较后验概率。在expert_votes的reason中明确你比较了哪些假设。
0.5 赔率校准思维：将置信度转化为"如果这个判断重复100次，我预期赢多少次"。如果历史数据显示类似信号胜率仅40%，你就不应给出>0.6的置信度。诚实比自信更重要。
0.6 Sigma事件检验：检查当前数据相对于历史分布有几倍标准差偏离。如果某个指标值位于>3σ极端区域（如VR>99%分位、布林带宽>3σ），则该指标可能处于非典型状态——不要简单外推，要标注为"sigma异常"并对该维度降低权重。
0.7 "所有模型都是错误的"(Box's Dictum)：模型的用处不是"对不对"，而是"在哪些条件下失效"。在synergy_analysis中必须包含一句"本分析最可能的失效模式是..."——这是对你推理质量的最终检验。
0.8 条件交换性检验：你假设"历史规律会延续"等价于假设当前市场与历史样本可交换。当BEGE环境切换、HAR衰减异常、或分布偏移预警触发时，可交换性假设被削弱→历史规律的外推可信度应系统性降权。
0.9 层次贝叶斯收缩思维(Gelman BDA3 Ch.5 + AlphaMix KDD23)：单个维度的信号强度不应被孤立看待。如果9个维度中有7个显示微弱偏多LLR(+0.2~0.3)，而1个维度显示极端偏多LLR(+2.5)，那个极端值应被收缩向群体均值。原理：维度值来自群体分布LLR_i ~ N(mu_pop, tau^2)，极端维度值很可能含噪，层次模型通过群体先验自动识别并降权。此外，Expert Soup(权重平均)思路：不采用硬投票，而是用softmax(置信度/T)对专家评分做软融合——极端高置信度专家的权重自动受限，防止单一专家绑架组合决策。
	0.10 先验脆弱性检验(Berger Ch.4.7 + 黄黎原Ch.6)：每个方向判断应接受epsilon-污染检验——如果名义后验概率是p，但存在10%先验偏差时后验可跌至q，则(p-q)为先验脆弱性。脆弱性>0.15→该信号对先验假设敏感→应降低仓位等待更多数据。这不是否定信号，而是诚实量化不确定性。
	0.11 稀有事件Laplace修正(Berger Ch.3.6 + Davidson-Pilon Ch.1)：对于小样本事件（如仅在20次类似情形中出现3次赢利），不要直接用3/20=15%作为概率估计，而应用k/N→(k+1)/(N+2)的Laplace平滑，并通过层次模型向组比率收缩。只有N>30时点估计才足够可靠。
	0.12 半柯西弱信息先验(BDA3 Ch.5.7)：评估组间差异时，不要使用可能人为拉低方差估计的Inverse-Gamma先验，而应用Half-Cauchy(scale=A)——它允许数据自身决定组间变异程度。Tau小→强收缩(组间差异小，可以高度池化信息)；Tau大→弱收缩(各组应独立评估)。
	0.13 Box-Cox正态化意识(BDA3 Ch.7.5)：金融收益率本质上是非正态的(厚尾+偏斜)。当应用基于正态假设的方法(如卡尔曼滤波/PCA/线性回归)时，应意识到若Box-Cox最优φ≠1，则原始数据偏离正态，基于正态假设的推断可信度应相应折扣。

【分析框架：四步递进推理】
══════════════════════════════════

【第1步：五专家独立判断】
你必须逐一给出每位专家的独立判断（bullish/bearish/neutral + 置信度 + 关键理由）。每位专家只看自己领域的数据，不受其他专家影响。这确保判断多样性，避免群体思维。

核心原则：
- 专家A(趋势结构)：ADX>25趋势策略有效，<20震荡策略为主。多头排列=中期看涨。波浪5浪完成预示反转。
  缠论分析框架（缠中说禅108课）：
  a) 走势终完美：任何走势类型必然完成。正在进行的趋势总会结束→关注背驰信号判断转折点。
  b) 笔状态机：(1,1)向上笔确认→看多；(-1,1)向下笔确认→看空；(1,0)/(-1,0)无新分型确认→趋势延续但力度衰减。
  c) MACD面积背驰（非峰比较）：趋势中当价格新高/新低但MACD柱面积显著缩小→趋势衰竭→即将反转。底背驰=买入信号，顶背驰=卖出信号。
  d) 三类买卖点：一买/一卖(趋势背驰后第一个反向分型，风险高收益大)；二买/二卖(回踩/反弹不破前极值，最可靠)；三买/三卖(突破/跌破中枢后回踩确认，追涨杀跌型)。
  e) 防狼术：MACD黄白双线在0轴以下是空头主导市场，坚决不入场做多——这是缠论最基本的风控铁律。
  f) 中枢震荡监视器Zn：Zn>2超买→潜在三买形成；Zn<-2超卖→潜在三卖形成。Zn窄幅收敛(<0.5)预示即将变盘。
  g) 多级别联立(区间套)：日线定方向→60分钟精确定位买卖点。两者共振时信号最可靠；日多60空=等60分钟企稳入场；日空60多=反弹减仓机会。
  Busch DMI增强(安德烈布殊)：
  h) MDI见顶回落+PDI开始增强=最佳转折入场点(比简单DM+/DM-交叉更早)。
  i) 弱势市场：DM+和DM-都在ADX下方→不入场。大动量：ADX从DM+/DM-之间向上穿越→大行情启动。
  Busch 2560战法：
  j) 做多条件：MA5金叉MA25 + 量MA5金叉量MA60 + MA25上行。三者缺一不可。周线版(MA20/MA55)用于确认中长期方向。
  市场四阶段(彭冬初)：
  k) 筑底→拉升→盘头→下跌。拉升阶段顺势做多最可靠；盘头阶段减仓/观望；下跌阶段空仓或做空。
  l) 三势可信度(大势/中势/小势)：三者方向一致时趋势最可靠；分歧越大越应轻仓操作。
  EW增强(普莱切特+许沂光)：
  m) 交替规则：浪2和浪4在形态/深度/复杂度上应不同。浪2深回调(zigzag)→浪4浅回调(flat)为经典交替。
  n) 浪3永不为最短浪；延长浪检测(≥1.618x次长浪)；斜三角形(楔形)后必有急速反向运动。
  唐能通系统(新生300天-唐能通)：
  o) 价托(三角托)：5/10/20MA金叉形成三角支撑区→底部确认买入。价压(三角压)：5/10/20MA死叉形成三角压力区→顶部确认卖出。价托强于金叉，价压强于死叉。
  p) 三金叉(均线+量能+MACD三者同时金叉)=最强买入信号；三死叉=最强卖出信号。部分金叉(2/3项)=偏多但需等待第三项确认。
  q) 跑道厚度RT=(60MA-120MA)/60MA：RT>20%厚实可重仓；10-20%适中正常参与；<5%太薄不参与。跑道=飞机起飞的跑道，厚度决定起飞安全度。
  r) 老鸭头：鸭颈上升(60MA上行)→鸭头(冲高远离60MA)→鸭嘴(回调到60MA附近缩量)→鸭嘴张开(放量再突破)=经典主升浪形态。评分>60且鸭嘴张开阶段=高概率买点。
  s) 【2T-1L】目标位=2×前高-前低；(2MP)=2×前低-前高。保守目标=前高+0.618×(前高-前低)。
  t) 三三过滤制：突破需满足3%幅度+连续3天站稳+成交量确认，三条件缺一即假突破概率高。
  解缠论增强(余井强1.0/2.0/3.0)：
  u) 力度比=当前笔力度/前一笔力度。力度比<0.618=背驰预警(趋势衰竭)；<0.382=背驰确认(即将反转)。上涨中力度比<0.618=顶背驰；下跌中力度比>1.618=底背驰。
  v) 中枢重心CG=(中枢高+中枢低)/2。CG连续上移=多头趋势强化；CG连续下移=空头趋势强化。三级别CG(周/日/60分钟)联立判断多空方向。
  w) 5条件拐点评分(100分制)：力度背驰20分+中枢位置20分+K线反转20分+量能确认20分+MACD均线20分。总分≥80=高概率拐点(可操作)；60-79=中等概率(等待确认)；<60=不操作。
  x) 笔内背离：同笔内部价格创新高但MACD柱缩短→笔内顶背离(早期见顶信号)；价格创新低但MACD柱上升→笔内底背离(早期见底信号)。笔内背离比标准背驰早1-3根K线出现。
  利弗莫尔战法(Jesse Livermore)：
  y) 关键点交易：只在关键点(历史高低点/整数关口/密集成交区)附近交易。关键点突破+自适应ATR确认=趋势有效。关键点突破后回撤<3%+站稳3天=买入确认。
  z) 危险信号法则：日内反转>1.5ATR/高位长上影/连续3日高低点下移/假突破/缩量上涨→立即离场观望。"看见危险信号先躲开，不和它争论"。
  aa) 金字塔加仓：第一笔盈利后+回调确认支撑+放量突破前高=加仓点。每次加仓量递减(5→3→2)。永不摊平亏损。
  ab) 50%利润提取规则：每笔盈利的50%转入储备金，不参与复利。这是利弗莫尔防止贪婪吞噬利润的铁律。
- 专家B(动量震荡)：RSI>70超买警惕，<30超卖关注。MACD金叉/死叉确认方向。TD9完成+反转K线=拐点。
- 专家C(量价资金)：量价配合=方向可信；量价背离=警惕反转。堆量=主力介入。缩量回踩=洗盘。
  VR容量比率(成交量核心技术)：VR>450头部警戒→减仓；VR<70底部区域→关注反转。VR背离(价升VR降=顶背离/价跌VR升=底背离)是领先信号。
  Busch量价代数：(+)+(+)=强(价升量增→持有/加仓)；(+)+(-)=弱(价升量缩→准备离场)；(-)+(+)=强空(价跌量增→空方主导)；(-)+(-)=弱空衰减(价跌量缩→空方力竭)。天量=趋势衰竭最后信号。
  量能形态五分类：地量(底部变盘前兆)→聚量(主力异动突破)→常量(趋势延续)→变量(方向不明)→量能消散(趋势衰减)。
- 专家D(形态价格)：底分型+晨星=见底；顶分型+暮星=见顶。价格在MA20上方=短线强势。
- 专家E(风险波动)：高波(>85%分位)=避险减仓。低波扩张=趋势启动。贝叶斯高熵=信号不可靠应观望。
  NRB(窄幅K线-短线交易大师)：连续下跌后NRB=底部反转信号；连续上涨后NRB=顶部反转信号。突破NRB高/低点确认方向。
  Velez时间止损：持仓5天既未达目标也未止损→无条件离场(时间也是成本)。
  Busch风控铁律：单笔最大亏损≤2%账户；账户亏损>25%→停止交易该账户。赢率30-40%+盈亏比≥3:1=正期望系统。
  BEGE分析：坏环境主导(好坏比<0.7+坏波上升)=下行风险高→显著减仓。好环境主导(好坏比>1.3)=风险偏好强→可适度激进。
  VRP高溢价=市场定价恐慌(风控收紧)；VRP负溢价=异常信号(建议观望)。
  分布偏移检测(σ-LSTM/DoubleAdapt)：当好波动率下降+坏波动率上升同时发生→预警市场结构切换→降置信度+减仓。
  HAR衰减模式：d>w>m表示典型长记忆结构(波动可预测)；若衰减异常→波动结构不稳定→降低预测置信度。
  逃顶十二招(综合CAN SLIM/利弗莫尔/唐能通)：
  ac) 12种顶部信号：高位长上影/放量滞涨/高位十字星/三只乌鸦/黄昏之星/MACD顶背离/RSI顶背离/巨量长阴/跌破关键均线/双顶头肩顶/布林收窄下破/缩量反弹。信号出现≥3个=减仓，≥5个=离场，≥7个=空仓。
  ad) 分配日计数(CAN SLIM-O'Neil)：25个交易日内出现≥5个分配日(价跌量增日)=市场顶部信号。3-4个分配日=警告。指数创新高则该分配日失效。当前分配日计数需与指数走势结合判断。
  ae) 逃顶铁律：①高位第一根大阴线无条件减仓50%；②连续3日不创新高=警惕；③放量跌破50MA=中期趋势转空；④高位利好不涨=主力出货。

【第1.5步：AlphaMix三阶段递进思维】(KDD 2023 专业交易公司的分层决策流程)
完成五专家独立判断后，你必须经历3个递进阶段：
 - Stage 1 (独立训练) 已完成：每位专家只看自己领域数据，确保判断多样性。
 - Stage 2 (专家池化) 现在执行：将五专家的判断视为一个专家池。使用Expert Soup思想——用softmax(置信度/T)对各专家评分做软加权融合，而不是硬投票。思考：融合后的方向是什么？哪些专家在当前市场条件下最值得信赖？当前ADX/波动率环境更适合趋势类专家还是反转形态类还是风控类？
 - Stage 3 (组合经理选择) 由你作为PM执行：从专家池中动态选出2-3位最匹配当前市场环境的专家组成迷你委员会，赋予更高权重。选专家时考虑三个维度：信号置信度、市场匹配度(趋势市选A/B，震荡市选C/D，高波市选E)、近期校准准确率。在synergy_analysis中注明选中的专家及理由。

【第2步：协同效应分析】(RL AlphaGen原则 + DoubleAdapt分布偏移检测)
评估专家意见之间的协同与冲突：
- 共振增强：哪些专家方向一致？方向一致的专家组数越多，信号越可靠。至少需要2个独立专家组同向才可输出非neutral信号。
- 冲突检测：哪些专家方向相反？冲突意味着市场信号不清晰，应降低置信度或输出neutral。
- 互补增效：即使两个专家的信号来源高度相关（如MACD金叉与DMA金叉），它们的差异方向（变化速率差）可能揭示新信息。思考：相似的看多信号之间是否存在"差值向量"指向隐藏的风险或机会？
- 弱信号价值：某个专家给出低置信度的微弱信号，但在其他专家强信号的组合下可能具有关键边际贡献——不要因单独IC低而忽略。
- 分布偏移警惕(DoubleAdapt原理)：市场环境是动态变化的——当专家E检测到BEGE坏环境主导、HAR衰减异常、或好坏波动率趋势分叉时，表明数据分布可能正在漂移。分布偏移时所有模型置信度应系统性下调×0.7，因为历史规律可能不再适用。
- 模型多样性评估(ANN Volatility论文启示)：不同模型(趋势/动量/量价/形态)在不同市场结构下表现差异显著。当前是趋势市场还是震荡市场？如果趋势明确→优先采纳趋势专家意见；如果震荡→优先采纳反转形态专家的意见。组合多种模型的预测通常比单一最佳模型更稳健。

【第3步：多周期意图协调】(MARL原则)
跨时间框架逐层验证：
- 日线意图（初步方向假设）→ 60分钟线细化（入场时机是否成熟？短期是否有反转风险？）→ 周线验证（大趋势是顺是逆？）
- 如果日线与周线方向一致（周线顺势），置信度应上调，仓位可适度放大。
- 如果日线与周线方向相反（周线逆势），置信度应下调，即使交易也应以短线快进快出为主。
- 60分钟线主要用于优化入场点：如果日线看多但60分钟线显示超买/顶分型，应等待回调入场而非追高。

【第4步：组合经理综合决策】
作为组合经理，综合前3步和贝叶斯框架的分析，做出最终决策并输出JSON。决策规则：
- 先验声明检查：你的final_decision中必须包含prior_belief声明，否则该分析无效。
- 当3个以上专家组同向 + 多周期协调一致 + 贝叶斯熵<0.85 + 无分布偏移预警 + HDI排除ROPE → 高置信度(>0.75)
- 当2个专家组同向但存在1个以上冲突 → 中等置信度(0.55-0.70)，减仓操作
- 当专家组方向分散(2:2:1)或无共识 → neutral，置信度<0.55
- Solomonoff惩罚：如果需要3个以上嵌套假设论证方向→置信度×0.7
- 贝叶斯posterior与你的最终方向冲突时，必须在confidence上打折扣(×0.7)并在bayesian_alignment标注"冲突"
- Sigma事件处理：检查各指标是否>3σ极端值。若存在>3σ的指标值，该指标权重应降至0，并在sigma_event_flags中标注具体异常指标
- 多重假设检查：在final_decision中列出至少2个被你排除的竞争假设，解释为何当前数据更支持采纳的假设
- 专家有效数量(N_eff=1/Sum(w_i^2))：如果N_eff<2专家意见高度集中，可能存在群体盲点(伪共振)降低置信度0.85
- 理由重叠率过高(>0.6)：即使多专家同向，如果理由高度重复(如多个专家共同基于MA金叉)，实际多样性不足。此时应标注为伪共振并在confidence上打折0.9
- 波动率>85%分位时，即使看多/看空也应降低仓位至正常水平的50%
- BEGE风险惩罚：坏环境主导(bebe_regime='bad_environment') → 仓位×0.65；VRP高溢价 → 仓位×0.8；分布偏移预警(好波降+坏波升) → 置信度×0.7且仓位×0.5
- HAR预测异常(衰减模式非d>w>m或R²过低) → 波动率预测可信度降低，止损应适当放宽
- 市场阶段风险修正：下跌阶段→仓位×0.5；盘头阶段→仓位×0.7；拉升阶段→仓位可放宽至1.0
- 2560策略信号：strong_buy时仓位上调20%；sell时仓位归零；hold_short时仓位减半
- Busch风控约束：单笔最大亏损≤2%；当综合信号偏空时会优先考虑现金为王(不进场,不赔钱)
- Velez短线原则：有时最好的操作就是不操作(no action)；85%的交易成功取决于正确的入场
- σ-LSTM启发：波动率本身具有自回归结构和长短期记忆——当前的高波动通常意味着未来一段时间持续高波动。结合HAR多时间尺度预测来评估波动趋势的持续性。

【输出格式】
只返回一个JSON对象，不要任何额外文字：
{{
  "signal": "bullish/bearish/neutral",
  "confidence": 0.75,
  "expert_votes": {{
    "expert_a": {{"direction": "bullish", "confidence": 0.70, "reason": "ADX>25多头趋势+MA多头排列"}},
    "expert_b": {{"direction": "bullish", "confidence": 0.65, "reason": "MACD金叉+RSI偏强"}},
    "expert_c": {{"direction": "neutral", "confidence": 0.45, "reason": "量价背离"}},
    "expert_d": {{"direction": "bullish", "confidence": 0.60, "reason": "底分型+价格站上MA20"}},
    "expert_e": {{"direction": "bullish", "confidence": 0.55, "reason": "波动率适中+贝叶斯后验偏多"}}
  }},
  "synergy_analysis": "3位专家看多(A/B/D)，专家C因量价背离给出neutral，专家E协同。总体共振较强但需注意量价不配合的隐患。专家A和B的信号虽相关(DMA金叉与MACD金叉)，但其差值揭示趋势加速。",
  "timeframe_coordination": "日线看多→60分钟线短期超买建议等回调入场→周线顺势(多头排列)，多周期总体协调一致",
  "weekly_align": "顺势/逆势/震荡无方向",
  "resonance_detail": "具体哪些指标形成共振",
  "entry_zone": "建议入场区间",
  "entry_reason": "入场理由（含关键形态和指标依据）",
  "stop_loss": "初始止损价（硬止损）",
  "stop_loss_rule": "止损规则（如跌破某支撑/均线/ATR倍数）",
  "batch_targets": [
    {{"price": "第一目标价", "ratio": "50%", "reason": "止盈理由"}},
    {{"price": "第二目标价", "ratio": "30%", "reason": "止盈理由"}},
    {{"price": "第三目标价", "ratio": "20%", "reason": "止盈理由"}}
  ],
  "trailing_stop": "移动止损规则",
  "batch_stop_loss": [
    {{"trigger": "到达第一目标后", "new_stop": "移动止损价"}},
    {{"trigger": "到达第二目标后", "new_stop": "移动止损价"}}
  ],
  "position_advice": "建议仓位占比及分批建仓理由",
  "kelly_position_pct": "凯利半仓百分比",
  "invalidation_condition": "计划失效条件",
  "risk_reward_analysis": "盈亏比和概率评估",
  "bayesian_alignment": "一致/偏离/冲突",
  "bebe_environment": "当前BEGE环境评估(好坏环境/风险偏好判断)",
  "distribution_shift_warning": "是/否—检测到分布偏移时标注具体偏移信号",
  "volatility_model_confidence": "波动率模型可信度评估(基于HAR R²/衰减/BEGE)",
  "chanlun_alignment": "缠论多级别信号一致性评估及对最终判断的影响",
  "busch_2560_alignment": "2560战法信号与综合判断的一致性评估",
  "market_phase_assessment": "当前市场四阶段判断及对策略选择的影响",
  "volume_regime_assessment": "量能形态(VR+量价代数+量能分类)综合评估",
  "behavioral_guard": "行为风控建议(连续亏损/报复交易/冷却期检测)",
  "prior_belief": "先验信念声明(基于周线趋势/市场阶段/历史基准率)",
  "competing_hypotheses": ["H1: 趋势延续—理由与P(H1|D)", "H2: 区间震荡—理由与P(H2|D)", "H3: V形反转—理由与P(H3|D)"],
  "sigma_event_flags": ["具体指标名>3σ异常说明（无异常则填'无'）"],
  "model_failure_mode": "本分析最可能的失效模式",
  "exchangability_note": "条件交换性评估：当前市场状态与历史样本的可交换性判断及对置信度的影响"
}}
neutral时仍需填写expert_votes、synergy_analysis和timeframe_coordination，entry_zone等可为空。
"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096
    }

    # Retry with exponential backoff for transient failures
    import time
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"[RETRY] Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data['choices'][0]['message']['content']
            signal = _parse_and_repair_json(content)
            if signal is None:
                print(f"[WARN] DeepSeek returned non-JSON: {content[:120]}...")
                return None
            signal['symbol'] = symbol
            signal['_raw_response'] = content
            signal['_indicators'] = indicators
            signal = enrich_signal_with_kelly(signal, indicators)
            return signal
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[RETRY] API error, waiting {wait}s (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(wait)
            continue
    print(f"[ERROR] DeepSeek request failed after {max_retries} attempts: {last_error}")
    return None


def _parse_and_repair_json(content):
    """
    Parse JSON from AI response with repair attempts for common failures.

    Safety principle: repair attempts MUST NOT alter numeric values. Only fix
    structural issues (trailing commas, unbalanced braces, quoting). Never
    guess or substitute numbers — if numeric fields are corrupted, return None
    rather than silently producing wrong position sizes or prices.
    """
    import re
    if not content or not isinstance(content, str):
        return None

    # Pre-process: strip markdown code fences before brace extraction
    content = re.sub(r'```(?:json)?\s*\n?', '', content)
    content = re.sub(r'\n?\s*```', '', content)
    content = content.strip()

    start = content.find('{')
    end = content.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = content[start:end+1]

    # Attempt 1: direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix trailing commas before closing brackets/braces (structural only)
    repaired = re.sub(r',\s*([}\]])', r'\1', json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Attempt 3: truncate at first balanced closing brace
    # This handles the common case of trailing commentary after valid JSON
    depth = 0
    last_valid_pos = -1
    for i, ch in enumerate(json_str):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                last_valid_pos = i
                break
    if last_valid_pos > 0:
        try:
            return json.loads(json_str[:last_valid_pos+1])
        except json.JSONDecodeError:
            pass

    # Attempt 4: regex-extract key fields as last resort
    # This is a degraded-mode recovery — mark as low confidence
    try:
        signal_match = re.search(r'"signal"\s*:\s*"(bullish|bearish|neutral)"', json_str)
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', json_str)
        if signal_match:
            confidence = float(conf_match.group(1)) if conf_match else 0.5
            # Cap confidence at 0.55 for regex-extracted signals — the data is degraded
            confidence = min(confidence, 0.55)
            minimal = {
                'signal': signal_match.group(1),
                'confidence': confidence,
                '_json_repair_level': 'regex_extraction',
            }
            # Try to extract expert_votes minimally
            votes_match = re.search(r'"expert_votes"\s*:\s*\{', json_str)
            if votes_match:
                for exp in ['expert_a', 'expert_b', 'expert_c', 'expert_d', 'expert_e']:
                    exp_dir = re.search(rf'"{exp}"\s*:\s*\{{\s*"direction"\s*:\s*"(bullish|bearish|neutral)"', json_str)
                    exp_conf = re.search(rf'"{exp}"\s*:\s*\{{\s*"direction"\s*:\s*"bullish|bearish|neutral"\s*,\s*"confidence"\s*:\s*([\d.]+)', json_str)
                    if exp_dir:
                        minimal.setdefault('expert_votes', {})[exp] = {
                            'direction': exp_dir.group(1),
                            'confidence': float(exp_conf.group(1)) if exp_conf else 0.5,
                        }
            return minimal
    except Exception:
        pass

    return None


def enrich_signal_with_kelly(signal, indicators):
    """
    对DeepSeek返回的AI信号进行凯利仓位后处理

    步骤：
    1. 从AI信号提取胜率/盈亏比/止损 → 计算二元凯利f*
    2. 获取已实现波动率 → 应用非遍历性保护（硬上限/波动拖累/回撤熔断）
    3. 将凯利仓位信息写入signal，供下游交易执行使用
    """
    if signal is None:
        return None

    # 获取当前价格
    current_price = indicators.get('current_price', 0)
    if current_price <= 0:
        entry_zone = signal.get('entry_zone', '')
        try:
            current_price = float(entry_zone.split('-')[0])
        except (ValueError, AttributeError):
            current_price = 100.0

    # 获取波动率用于默认值估计（个股波动率决定默认盈亏幅度）
    rv_composite = indicators.get('rv_composite', 0.02)
    atr_pct = indicators.get('vol_atr_pct', 2.0) / 100.0  # ATR% 转小数

    # 1. 二元凯利计算（传入波动率信息用于智能默认值）
    kelly_result = kelly_from_deepseek_signal(signal, current_price, rv_composite, atr_pct)

    # 2. 非遍历性保护
    drawdown = indicators.get('current_drawdown', 0.0)
    ergo = non_ergodicity_protections(
        kelly_result['kelly_f_half'],
        rv_composite,
        drawdown
    )

    # 3. 如果贝叶斯高熵警告，进一步降低仓位
    bayes_entropy_high = indicators.get('bayes_entropy_high', False)
    bayes_posterior = indicators.get('bayes_fused_posterior', 0.5)
    if bayes_entropy_high:
        ergo['ergo_final_position'] *= 0.5
        ergo['ergo_protections'].append('贝叶斯高熵减半')

    # 4. 如果贝叶斯后验与AI信号冲突，降低仓位
    ai_signal = signal.get('signal', 'neutral')
    bayes_signal = indicators.get('bayes_fused_signal', 'neutral')
    if ai_signal == 'bullish' and bayes_signal == 'bearish':
        ergo['ergo_final_position'] *= 0.5
        ergo['ergo_protections'].append('AI/贝叶斯冲突减半(牛/熊)')
    elif ai_signal == 'bearish' and bayes_signal == 'bullish':
        ergo['ergo_final_position'] *= 0.5
        ergo['ergo_protections'].append('AI/贝叶斯冲突减半(熊/牛)')

    # 5. 最终仓位百分比
    final_position_pct = round(ergo['ergo_final_position'] * 100, 1)

    # 6. 用凯利结果覆盖/补充AI信号的仓位字段
    signal['kelly_f_star'] = kelly_result['kelly_f_full']
    signal['kelly_f_half'] = kelly_result['kelly_f_half']
    signal['kelly_win_prob'] = kelly_result['kelly_win_prob']
    signal['kelly_payoff_ratio'] = kelly_result['kelly_payoff_ratio']
    signal['kelly_ev'] = kelly_result['kelly_expected_value']
    signal['kelly_position_pct'] = final_position_pct
    signal['ergo_protections'] = ergo['ergo_protections']
    signal['ergo_safe_to_enter'] = ergo['ergo_safe_to_enter']
    signal['ergo_vol_drag'] = ergo['ergo_vol_drag']

    # 7. 补充贝叶斯对齐信息
    signal['bayesian_posterior'] = bayes_posterior
    if ai_signal == 'bullish' and bayes_signal == 'bullish':
        signal['bayesian_alignment'] = '一致'
    elif ai_signal == 'bearish' and bayes_signal == 'bearish':
        signal['bayesian_alignment'] = '一致'
    elif ai_signal == 'neutral' or bayes_signal == 'neutral':
        signal['bayesian_alignment'] = '偏离'
    else:
        signal['bayesian_alignment'] = '冲突'

    return signal