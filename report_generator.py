# report_generator.py
import datetime
import os
import webbrowser
from data_loader import get_stock_name
from localization import t

COLORS = {
    'bullish': {'bg': '#e8f5e9', 'border': '#4caf50', 'badge': '#2e7d32', 'text': '#1b5e20'},
    'bearish': {'bg': '#fbe9e7', 'border': '#f44336', 'badge': '#c62828', 'text': '#b71c1c'},
    'neutral': {'bg': '#fff8e1', 'border': '#ff9800', 'badge': '#e65100', 'text': '#e65100'},
}


def _html_escape(s):
    """Escape HTML special characters to prevent XSS/injection."""
    if s is None:
        return ''
    s = str(s)
    s = s.replace('&', '&amp;')
    s = s.replace('<', '&lt;')
    s = s.replace('>', '&gt;')
    s = s.replace('"', '&quot;')
    s = s.replace("'", '&#39;')
    return s


def _fmt_pnl(val):
    """Format avg_pnl for display, returning 'N/A' when None."""
    if val is None:
        return 'N/A'
    return f'{val}%'


def generate_html_report(signals, report_type="分析报告", eval_stats=None, hist_stats=None):
    if not signals:
        print("无信号数据，跳过报告生成")
        return None

    now = datetime.datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    filename = f"logs/report_{ts}.html"
    os.makedirs("logs", exist_ok=True)

    bullish_n = sum(1 for s in signals if s.get('signal') == 'bullish')
    bearish_n = sum(1 for s in signals if s.get('signal') == 'bearish')
    neutral_n = sum(1 for s in signals if s.get('signal') == 'neutral')
    total = len(signals)

    # --- 构建评估统计区 ---
    eval_html = ""
    if eval_stats and eval_stats.get('total', 0) > 0:
        wr = eval_stats.get('win_rate', 0)
        wr_color = '#2e7d32' if wr >= 50 else '#c62828'
        eval_html = f"""
        <div class="eval-section">
            <h2>📈 历史信号回顾</h2>
            <div class="eval-grid">
                <div class="eval-stat"><div class="enum">{eval_stats['total']}</div><div class="elabel">已评估信号</div></div>
                <div class="eval-stat"><div class="enum" style="color:{wr_color}">{wr}%</div><div class="elabel">胜率</div></div>
                <div class="eval-stat"><div class="enum">{_fmt_pnl(eval_stats['avg_pnl'])}</div><div class="elabel">平均盈亏</div></div>
                <div class="eval-stat"><div class="enum">{eval_stats['hit_targets']}</div><div class="elabel">触及止盈</div></div>
                <div class="eval-stat"><div class="enum">{eval_stats['stopped_out']}</div><div class="elabel">触发止损</div></div>
            </div>
        </div>"""

    hist_html = ""
    if hist_stats and hist_stats.get('overall', {}).get('total', 0) > 0:
        overall = hist_stats['overall']
        h_wr = round(overall['wins'] / overall['total'] * 100, 1) if overall['total'] > 0 else 0
        by_type_rows = ""
        for bt in hist_stats.get('by_type', []):
            bt_wr = round(bt['wins'] / bt['cnt'] * 100, 1) if bt['cnt'] > 0 else 0
            by_type_rows += f"<tr><td>{bt['signal']}</td><td>{bt['cnt']}</td><td>{bt_wr}%</td><td>{_fmt_pnl(bt['avg_pnl'])}</td><td>{bt['hit_targets']}</td><td>{bt['stopped']}</td></tr>"
        hist_html = f"""
        <div class="hist-section">
            <h2>📋 全部历史统计（自建库以来）</h2>
            <div class="hist-summary">
                总信号 <b>{overall['total']}</b> · 胜率 <b>{h_wr}%</b> · 平均盈亏 <b>{_fmt_pnl(overall['avg_pnl'])}</b>
            </div>
            <table class="hist-table">
                <tr><th>方向</th><th>数量</th><th>胜率</th><th>均盈亏</th><th>止盈</th><th>止损</th></tr>
                {by_type_rows}
            </table>
        </div>"""

    # sort: highest confidence first
    signals_sorted = sorted(signals, key=lambda s: s.get('confidence', 0), reverse=True)

    rows_html = ""
    for s in signals_sorted:
        sig = s.get('signal', 'neutral')
        c = COLORS.get(sig, COLORS['neutral'])
        sym = s.get('symbol', '?')
        name = _html_escape(get_stock_name(sym))
        conf = s.get('confidence', 0)
        conf_pct = f"{conf:.0%}"
        weekly = _html_escape(s.get('weekly_align', ''))
        resonance = _html_escape(s.get('resonance_detail', s.get('detail', '')))
        entry = _html_escape(s.get('entry_zone', ''))
        stop = _html_escape(s.get('stop_loss', ''))
        trailing = _html_escape(s.get('trailing_stop', ''))

        # 分批止盈目标
        batch_targets = s.get('batch_targets', [])
        if batch_targets:
            targets_str = "<br>".join(
                f"TP{t+1}: {bt.get('price','?')} ({bt.get('ratio','?')}) — {_html_escape(bt.get('reason',''))}"
                for t, bt in enumerate(batch_targets[:3])
            )
        else:
            # 兼容旧格式
            targets = s.get('targets', [])
            if isinstance(targets, list):
                targets_str = "<br>".join(targets[:3])
            else:
                targets_str = str(targets)

        # 分批止损
        batch_sl = s.get('batch_stop_loss', [])
        if batch_sl:
            sl_str = "<br>".join(
                f"{_html_escape(bs.get('trigger',''))}: {_html_escape(bs.get('new_stop',''))}"
                for bs in batch_sl[:2]
            )
        else:
            sl_str = stop

        position = _html_escape(s.get('position_advice', ''))
        source = _html_escape(s.get('source', 'DeepSeek'))
        entry_reason = _html_escape(s.get('entry_reason', ''))
        invalidation = _html_escape(s.get('invalidation_condition', ''))
        risk = _html_escape(s.get('risk_reward_analysis', ''))

        rows_html += f"""
        <div class="card" style="border-left: 5px solid {c['border']}; background: {c['bg']};">
            <div class="card-header">
                <span class="symbol">{name}(<b>{_html_escape(sym)}</b>)</span>
                <span class="badge" style="background:{c['badge']}; color:white; padding:4px 12px; border-radius:12px; font-weight:bold;">
                    {_html_escape(t(sig))} · {conf_pct}
                </span>
                <span class="source-tag">[{source}]</span>
            </div>
            <div class="card-body">
                <div class="meta">
                    <span>周线: <b>{weekly}</b></span>
                </div>
                <div class="resonance"><b>共振:</b> {resonance}</div>
                <div class="resonance"><b>入场理由:</b> {entry_reason}</div>
                <div class="prices">
                    <div class="price-box entry"><b>入场:</b> {entry}</div>
                    <div class="price-box stop"><b>止损:</b> {sl_str}</div>
                </div>
                <div class="batch-section">
                    <div class="batch-title">分批止盈目标</div>
                    <div class="batch-content">{targets_str}</div>
                </div>
                <div class="meta" style="margin-top:8px;">
                    <span><b>移动止损:</b> {trailing if trailing else '无'}</span>
                </div>
                <div class="position"><b>仓位:</b> {position}</div>
                <div class="meta" style="margin-top:4px;">
                    <span style="color:#888;"><b>失效:</b> {invalidation}</span>
                </div>
                <div class="meta">
                    <span style="color:#888;"><b>盈亏比:</b> {risk}</span>
                </div>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化{report_type} - {ts}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif; background:#f5f5f5; color:#333; }}
.header {{ background: linear-gradient(135deg, #1a237e, #283593); color:white; padding:30px 40px; }}
.header h1 {{ font-size:26px; margin-bottom:6px; }}
.header .time {{ opacity:0.75; font-size:14px; }}
.summary {{ display:flex; gap:20px; padding:24px 40px; background:white; box-shadow:0 1px 3px rgba(0,0,0,0.1); }}
.stat {{ flex:1; text-align:center; padding:16px; border-radius:8px; }}
.stat .num {{ font-size:36px; font-weight:bold; }}
.stat .label {{ font-size:13px; color:#666; margin-top:4px; }}
.stat.bullish {{ background:#e8f5e9; }}
.stat.bullish .num {{ color:#2e7d32; }}
.stat.bearish {{ background:#fbe9e7; }}
.stat.bearish .num {{ color:#c62828; }}
.stat.neutral {{ background:#fff8e1; }}
.stat.neutral .num {{ color:#e65100; }}
.eval-section {{ margin:20px 40px; padding:20px; background:white; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1); }}
.eval-section h2 {{ font-size:18px; margin-bottom:14px; }}
.eval-grid {{ display:flex; gap:16px; }}
.eval-stat {{ flex:1; text-align:center; padding:12px; background:#f5f5f5; border-radius:6px; }}
.eval-stat .enum {{ font-size:28px; font-weight:bold; color:#333; }}
.eval-stat .elabel {{ font-size:12px; color:#888; margin-top:4px; }}
.hist-section {{ margin:20px 40px; padding:20px; background:white; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1); }}
.hist-section h2 {{ font-size:18px; margin-bottom:14px; }}
.hist-summary {{ font-size:14px; color:#555; margin-bottom:12px; }}
.hist-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.hist-table th, .hist-table td {{ padding:8px 12px; border-bottom:1px solid #eee; text-align:center; }}
.hist-table th {{ background:#f5f5f5; font-weight:bold; }}
.container {{ max-width:1100px; margin:0 auto; padding:20px 40px; }}
.card {{ margin:16px 0; padding:20px; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.08); }}
.card-header {{ display:flex; align-items:center; gap:12px; margin-bottom:12px; }}
.symbol {{ font-size:20px; font-weight:bold; }}
.source-tag {{ font-size:11px; color:#888; margin-left:auto; }}
.card-body {{ font-size:14px; line-height:1.8; }}
.meta {{ margin-bottom:4px; }}
.resonance {{ color:#555; margin-bottom:8px; }}
.prices {{ display:flex; gap:12px; margin:8px 0; flex-wrap:wrap; }}
.price-box {{ padding:6px 14px; border-radius:6px; font-size:13px; }}
.price-box.entry {{ background:#e3f2fd; }}
.price-box.stop {{ background:#ffebee; }}
.price-box.target {{ background:#f3e5f5; }}
.position {{ color:#666; }}
.batch-section {{ margin-top:8px; padding:8px 12px; background:rgba(0,0,0,0.03); border-radius:6px; }}
.batch-title {{ font-weight:bold; font-size:12px; color:#555; margin-bottom:4px; }}
.batch-content {{ font-size:13px; line-height:1.6; }}
.footer {{ text-align:center; padding:30px; color:#999; font-size:12px; }}
</style>
</head>
<body>
<div class="header">
    <h1>量化{report_type}</h1>
    <div class="time">生成时间: {now.strftime('%Y-%m-%d %H:%M:%S')} · 共 {total} 只股票</div>
</div>
<div class="summary">
    <div class="stat bullish"><div class="num">{bullish_n}</div><div class="label">看多</div></div>
    <div class="stat bearish"><div class="num">{bearish_n}</div><div class="label">看空</div></div>
    <div class="stat neutral"><div class="num">{neutral_n}</div><div class="label">观望</div></div>
</div>
{eval_html}
{hist_html}
<div class="container">
    {rows_html}
</div>
<div class="footer">
    本报告由量化系统自动生成 · 仅供参考，不构成投资建议
</div>
</body>
</html>"""

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)

    abs_path = os.path.abspath(filename)
    print(f"\n[报告] 已生成: {abs_path}")

    try:
        webbrowser.open(abs_path)
    except Exception:
        pass

    return abs_path


# ===================== V2: 三面板综合报告 =====================

def generate_html_report_v2(signals, report_type="分析报告", eval_stats=None, hist_stats=None):
    """
    增强版HTML报告：每只股票三个面板
    - 面板1: 本地量化模型（指标驱动的交易计划+判断原则）
    - 面板2: AI深度分析（DeepSeek分析结果）
    - 面板3: 综合结论（本地+AI合并结论+凯利仓位）
    """
    if not signals:
        print("无信号数据，跳过报告生成")
        return None

    now = datetime.datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    filename = f"logs/report_{ts}.html"
    os.makedirs("logs", exist_ok=True)

    bullish_n = sum(1 for s in signals if s.get('signal') == 'bullish')
    bearish_n = sum(1 for s in signals if s.get('signal') == 'bearish')
    neutral_n = sum(1 for s in signals if s.get('signal') == 'neutral')
    total = len(signals)

    # 本地模型统计
    local_bullish = sum(1 for s in signals if s.get('_local_plan', {}).get('signal') == 'bullish')
    local_bearish = sum(1 for s in signals if s.get('_local_plan', {}).get('signal') == 'bearish')
    local_neutral = sum(1 for s in signals if s.get('_local_plan', {}).get('signal') == 'neutral')

    # --- 评估统计 ---
    eval_html = ""
    if eval_stats and eval_stats.get('total', 0) > 0:
        wr = eval_stats.get('win_rate', 0)
        wr_color = '#2e7d32' if wr >= 50 else '#c62828'
        eval_html = f"""
        <div class="eval-section">
            <h2>📈 历史信号回顾</h2>
            <div class="eval-grid">
                <div class="eval-stat"><div class="enum">{eval_stats['total']}</div><div class="elabel">已评估信号</div></div>
                <div class="eval-stat"><div class="enum" style="color:{wr_color}">{wr}%</div><div class="elabel">胜率</div></div>
                <div class="eval-stat"><div class="enum">{_fmt_pnl(eval_stats['avg_pnl'])}</div><div class="elabel">平均盈亏</div></div>
                <div class="eval-stat"><div class="enum">{eval_stats['hit_targets']}</div><div class="elabel">触及止盈</div></div>
                <div class="eval-stat"><div class="enum">{eval_stats['stopped_out']}</div><div class="elabel">触发止损</div></div>
            </div>
        </div>"""

    hist_html = ""
    if hist_stats and hist_stats.get('overall', {}).get('total', 0) > 0:
        overall = hist_stats['overall']
        h_wr = round(overall['wins'] / overall['total'] * 100, 1) if overall['total'] > 0 else 0
        by_type_rows = ""
        for bt in hist_stats.get('by_type', []):
            bt_wr = round(bt['wins'] / bt['cnt'] * 100, 1) if bt['cnt'] > 0 else 0
            by_type_rows += f"<tr><td>{t(bt['signal'])}</td><td>{bt['cnt']}</td><td>{bt_wr}%</td><td>{_fmt_pnl(bt['avg_pnl'])}</td><td>{bt['hit_targets']}</td><td>{bt['stopped']}</td></tr>"
        hist_html = f"""
        <div class="hist-section">
            <h2>📋 全部历史统计（自建库以来）</h2>
            <div class="hist-summary">
                总信号 <b>{overall['total']}</b> · 胜率 <b>{h_wr}%</b> · 平均盈亏 <b>{_fmt_pnl(overall['avg_pnl'])}</b>
            </div>
            <table class="hist-table">
                <tr><th>方向</th><th>数量</th><th>胜率</th><th>均盈亏</th><th>止盈</th><th>止损</th></tr>
                {by_type_rows}
            </table>
        </div>"""

    signals_sorted = sorted(signals, key=lambda s: s.get('confidence', 0), reverse=True)

    # Consensus: both AI and local agree bullish
    consensus_bullish = sum(1 for s in signals_sorted
                           if s.get('signal') == 'bullish'
                           and s.get('_local_plan', {}).get('signal') == 'bullish')

    rows_html = ""
    for idx, s in enumerate(signals_sorted):
        sig = s.get('signal', 'neutral')
        sym = s.get('symbol', '?')
        name = _html_escape(get_stock_name(sym))
        conf = s.get('confidence', 0)
        weekly = _html_escape(s.get('weekly_align', ''))
        source = _html_escape(s.get('source', 'DeepSeek'))
        local_plan = s.get('_local_plan', {})
        local_sig = local_plan.get('signal', 'neutral') if local_plan else 'neutral'
        local_conf = local_plan.get('confidence', 0) if local_plan else 0

        c = COLORS.get(sig, COLORS['neutral'])
        local_c = COLORS.get(local_sig, COLORS['neutral'])
        sig_cn = _html_escape(t(sig))
        local_sig_cn = _html_escape(t(local_sig))

        # ===== Panel 1: 本地量化模型 =====
        panel1 = _build_panel1_local_coequal(local_plan, c, sym)

        # ===== Panel 2: AI深度分析 =====
        panel2 = _build_panel2_ai_coequal(s, c)

        # ===== Panel 3: 综合结论 =====
        panel3 = _build_panel3_merged(s, local_plan, c)

        # 对齐状态
        alignment = s.get("bayesian_alignment", "?")
        if alignment == "一致":
            align_badge = "<span class='align-badge align-ok'>一致</span>"
        elif alignment == "冲突":
            align_badge = "<span class='align-badge align-conflict'>冲突</span>"
        else:
            align_badge = "<span class='align-badge align-partial'>部分对齐</span>"

        rows_html += f"""
        <details class="card" id="stock_{idx}" style="border-left: 5px solid {c['border']}; background: white;" open>
            <summary class="card-header" style="cursor:pointer; list-style:none;">
                <span class="symbol">{name}(<b>{_html_escape(sym)}</b>)</span>
                <span class="badge" style="background:{c['badge']}; color:white; padding:4px 12px; border-radius:12px; font-weight:bold;">
                    AI: {sig_cn} · {conf:.0%}
                </span>
                <span class="badge" style="background:{local_c['badge']}; color:white; padding:4px 12px; border-radius:12px; font-weight:bold;">
                    本地: {local_sig_cn} · {local_conf:.0%}
                </span>
                {align_badge}
                <span class="source-tag">周线: {weekly}</span>
                <span style="margin-left:auto;color:#999;font-size:11px;">点击展开/收起</span>
            </summary>
            <div class="card-columns">
                <div class="card-col card-col-local">
                    <div class="col-header" style="color:{local_c['badge']};">本地量化模型</div>
                    {panel1}
                </div>
                <div class="card-col card-col-ai">
                    <div class="col-header" style="color:{c['badge']};">AI 深度分析</div>
                    {panel2}
                </div>
            </div>
            <div class="card-merged">
                {panel3}
            </div>
        </details>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化{report_type} - {ts}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif; background:#f0f2f5; color:#333; }}
.header {{ background: linear-gradient(135deg, #1a237e, #283593); color:white; padding:30px 40px; }}
.header h1 {{ font-size:26px; margin-bottom:6px; }}
.header .time {{ opacity:0.75; font-size:14px; }}
.summary {{ display:flex; flex-direction:column; gap:16px; padding:24px 40px; background:white; box-shadow:0 1px 3px rgba(0,0,0,0.1); }}
.summary-row {{ display:flex; gap:20px; align-items:center; }}
.summary-label {{ font-size:13px; font-weight:bold; color:#888; min-width:70px; }}
.stat {{ flex:1; text-align:center; padding:16px; border-radius:8px; }}
.stat .num {{ font-size:36px; font-weight:bold; }}
.stat .label {{ font-size:13px; color:#666; margin-top:4px; }}
.stat.bullish {{ background:#e8f5e9; }}
.stat.bullish .num {{ color:#2e7d32; }}
.stat.bearish {{ background:#fbe9e7; }}
.stat.bearish .num {{ color:#c62828; }}
.stat.neutral {{ background:#fff8e1; }}
.stat.neutral .num {{ color:#e65100; }}
.eval-section {{ margin:20px 40px; padding:20px; background:white; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1); }}
.eval-section h2 {{ font-size:18px; margin-bottom:14px; }}
.eval-grid {{ display:flex; gap:16px; }}
.eval-stat {{ flex:1; text-align:center; padding:12px; background:#f5f5f5; border-radius:6px; }}
.eval-stat .enum {{ font-size:28px; font-weight:bold; color:#333; }}
.eval-stat .elabel {{ font-size:12px; color:#888; margin-top:4px; }}
.hist-section {{ margin:20px 40px; padding:20px; background:white; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1); }}
.hist-section h2 {{ font-size:18px; margin-bottom:14px; }}
.hist-summary {{ font-size:14px; color:#555; margin-bottom:12px; }}
.hist-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.hist-table th, .hist-table td {{ padding:8px 12px; border-bottom:1px solid #eee; text-align:center; }}
.hist-table th {{ background:#f5f5f5; font-weight:bold; }}
.container {{ max-width:1200px; margin:0 auto; padding:20px 40px; }}
.card {{ margin:16px 0; padding:0; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.1); overflow:hidden; }}
.card-header {{ display:flex; align-items:center; gap:10px; padding:16px 20px; background:#fafafa; border-bottom:1px solid #eee; flex-wrap:wrap; }}
summary.card-header::-webkit-details-marker {{ display:none; }}
summary.card-header::marker {{ display:none; content:""; }}
details.card[open] summary.card-header {{ border-bottom:1px solid #eee; }}
details.card:not([open]) summary.card-header {{ border-bottom:none; }}
.align-badge {{ font-size:11px; padding:2px 8px; border-radius:10px; font-weight:bold; white-space:nowrap; }}
.align-badge.align-ok {{ background:#e8f5e9; color:#2e7d32; }}
.align-badge.align-conflict {{ background:#ffebee; color:#c62828; }}
.align-badge.align-partial {{ background:#fff3e0; color:#e65100; }}
.card-columns {{ display:grid; grid-template-columns:1fr 1fr; gap:0; border-bottom:1px solid #f0f0f0; }}
.card-col {{ padding:16px 20px; font-size:14px; line-height:1.7; }}
.card-col-local {{ border-right:1px solid #e0e0e0; }}
.col-header {{ font-size:15px; font-weight:bold; margin-bottom:10px; padding-bottom:6px; border-bottom:2px solid currentColor; }}
.card-merged {{ padding:16px 20px; }}
.symbol {{ font-size:18px; font-weight:bold; }}
.source-tag {{ font-size:11px; color:#888; margin-left:auto; }}
.panel {{ padding:16px 20px; border-bottom:1px solid #f0f0f0; }}
.panel:last-child {{ border-bottom:none; }}
.panel summary {{ font-size:15px; font-weight:bold; cursor:pointer; padding:4px 0; color:#444; }}
.panel summary:hover {{ color:#1a237e; }}
.panel-content {{ margin-top:12px; font-size:14px; line-height:1.8; }}
.prices {{ display:flex; gap:12px; margin:8px 0; flex-wrap:wrap; }}
.price-box {{ padding:8px 16px; border-radius:8px; font-size:13px; font-weight:bold; }}
.price-box.entry {{ background:#e3f2fd; border:1px solid #90caf9; }}
.price-box.stop {{ background:#ffebee; border:1px solid #ef9a9a; }}
.price-box.target {{ background:#f3e5f5; border:1px solid #ce93d8; }}
.principles-list {{ list-style:decimal; padding-left:20px; }}
.principles-list li {{ margin:3px 0; font-size:13px; color:#555; }}
.kv-table {{ width:100%; border-collapse:collapse; font-size:12px; margin:8px 0; }}
.kv-table td {{ padding:4px 10px; border-bottom:1px solid #f0f0f0; }}
.kv-table td:first-child {{ font-weight:bold; color:#666; width:120px; }}
.merge-section {{ background:#f9f9f9; border-radius:6px; padding:12px; margin:8px 0; }}
.merge-title {{ font-weight:bold; color:#333; margin-bottom:6px; }}
.protect-tag {{ display:inline-block; background:#fff3e0; color:#e65100; padding:2px 8px; border-radius:4px; font-size:11px; margin:2px; }}
.meta-row {{ display:flex; gap:24px; flex-wrap:wrap; margin:4px 0; font-size:13px; }}
.meta-item {{ }}
.meta-item b {{ color:#555; }}
.footer {{ text-align:center; padding:30px; color:#999; font-size:12px; }}
</style>
</head>
<body>
<div class="header">
    <h1>📊 量化{report_type}</h1>
    <div class="time">生成时间: {now.strftime('%Y-%m-%d %H:%M:%S')} · 共 {total} 只股票</div>
</div>
<div class="summary">
    <div class="summary-row">
        <div class="summary-label">AI 分析</div>
        <div class="stat bullish"><div class="num">{bullish_n}</div><div class="label">AI看多</div></div>
        <div class="stat bearish"><div class="num">{bearish_n}</div><div class="label">AI看空</div></div>
        <div class="stat neutral"><div class="num">{neutral_n}</div><div class="label">AI观望</div></div>
    </div>
    <div class="summary-row">
        <div class="summary-label">本地模型</div>
        <div class="stat bullish"><div class="num">{local_bullish}</div><div class="label">本地看多</div></div>
        <div class="stat bearish"><div class="num">{local_bearish}</div><div class="label">本地看空</div></div>
        <div class="stat neutral"><div class="num">{local_neutral}</div><div class="label">本地观望</div></div>
    </div>
    <div class="summary-row">
        <div class="summary-label">共同确认</div>
        <div class="stat" style="background:#c8e6c9; flex:0.5"><div class="num" style="color:#1b5e20;">{consensus_bullish}</div><div class="label">AI+本地共同看多</div></div>
    </div>
</div>
{eval_html}
{hist_html}
<div class="quick-nav" style="margin:20px 40px; padding:16px; background:white; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1); display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
    <b>快速跳转:</b>
    <a href="#bullish_sec" style="background:#e8f5e9; color:#2e7d32; padding:8px 16px; border-radius:6px; text-decoration:none; font-weight:bold;">看多 ({bullish_n})</a>
    <a href="#bearish_sec" style="background:#fbe9e7; color:#c62828; padding:8px 16px; border-radius:6px; text-decoration:none; font-weight:bold;">看空 ({bearish_n})</a>
    <a href="#neutral_sec" style="background:#fff8e1; color:#e65100; padding:8px 16px; border-radius:6px; text-decoration:none; font-weight:bold;">观望 ({neutral_n})</a>
    <span style="color:#999; font-size:12px; margin-left:auto;">按置信度排序 · 点击标签跳转 · 点击股票名展开/收起</span>
</div>
<div class="container">
    <div id="bullish_sec"><h3 style="color:#2e7d32; margin:0 0 10px; padding-top:10px;">看多</h3></div>
    {rows_html}
</div>
<div class="footer">
    本报告由量化系统自动生成 · 本地模型（贝叶斯+EW+DMI+波动率+量价形态） + AI深度分析（DeepSeek） · 仅供参考，不构成投资建议
</div>
</body>
</html>"""

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)

    abs_path = os.path.abspath(filename)
    print(f"\n[报告V2] 已生成: {abs_path}")

    try:
        webbrowser.open(abs_path)
    except Exception:
        pass

    return abs_path


def _build_bebe_panel(lp):
    """构建BEGE好坏环境波动率分解面板 (NBER w27108)"""
    kv = lp.get('key_values', {})
    regime = str(kv.get('bebe_regime', ''))
    ratio = float(kv.get('bebe_good_bad_ratio', 1.0))
    vrp = str(kv.get('bebe_vrp_signal', 'neutral'))

    if regime == 'unknown' or regime == '':
        return ''

    # 环境颜色和标签
    if regime == '坏环境主导' or regime == 'bad_environment':
        env_color = '#c62828'
        env_bg = '#ffebee'
        env_label = '坏环境主导'
        env_desc = '下行波动率显著超过上行→高风险厌恶'
    elif regime == '好环境主导' or regime == 'good_environment':
        env_color = '#2e7d32'
        env_bg = '#e8f5e9'
        env_label = '好环境主导'
        env_desc = '上行波动率显著超过下行→风险偏好强劲'
    else:
        env_color = '#e65100'
        env_bg = '#fff3e0'
        env_label = '环境平衡'
        env_desc = '上下行波动率基本对等→中性风险态度'

    # VRP颜色
    if 'high' in vrp.lower():
        vrp_color = '#c62828'
        vrp_bg = '#ffebee'
        vrp_desc = '市场定价恐慌，方差风险补偿高'
    elif 'negative' in vrp.lower():
        vrp_color = '#6a1b9a'
        vrp_bg = '#f3e5f5'
        vrp_desc = '异常信号，需警惕市场结构切换'
    elif 'low' in vrp.lower():
        vrp_color = '#e65100'
        vrp_bg = '#fff3e0'
        vrp_desc = '市场自满，方差风险补偿低'
    else:
        vrp_color = '#666'
        vrp_bg = '#f5f5f5'
        vrp_desc = '方差风险溢价适中'

    # 好坏比进度条（0到2，1为平衡）
    bar_pct = min(ratio / 2.0, 1.0) * 100
    bar_color = '#4caf50' if ratio > 1.0 else '#f44336'

    return f"""
    <div style="margin-top:12px;padding:10px;background:#f9f9fb;border-radius:6px;border:1px solid #e0e0e0;">
        <b>[BEGE波动率环境]</b> (NBER w27108 好坏环境分解)
        <div style="display:flex;gap:12px;margin-top:8px;flex-wrap:wrap;">
            <div style="flex:1;min-width:180px;padding:8px;background:{env_bg};border-radius:4px;border-left:3px solid {env_color};">
                <div style="font-size:11px;color:#888;">环境判定</div>
                <div style="font-weight:700;color:{env_color};">{env_label}</div>
                <div style="font-size:11px;color:#666;">{env_desc}</div>
            </div>
            <div style="flex:1;min-width:180px;padding:8px;background:{vrp_bg};border-radius:4px;border-left:3px solid {vrp_color};">
                <div style="font-size:11px;color:#888;">方差风险溢价(VRP)</div>
                <div style="font-weight:700;color:{vrp_color};">{vrp}</div>
                <div style="font-size:11px;color:#666;">{vrp_desc}</div>
            </div>
        </div>
        <div style="margin-top:8px;font-size:11px;color:#666;">
            好坏波动率比: {ratio:.3f}
            <span style="display:inline-block;width:120px;height:10px;background:#eee;border-radius:5px;vertical-align:middle;margin-left:6px;">
                <span style="display:inline-block;width:{bar_pct:.0f}%;height:10px;background:{bar_color};border-radius:5px;vertical-align:top;"></span>
            </span>
            <span style="margin-left:4px;">{"←坏环境 (下行风险高)" if ratio < 0.7 else "→好环境 (上行空间大)" if ratio > 1.3 else "→平衡"}</span>
        </div>
    </div>"""


def _build_chanlun_panel(lp):
    """构建缠论分析面板（缠中说禅108课）"""
    kv = lp.get('key_values', {})
    stroke_state = str(kv.get('chan_stroke', ''))
    stroke_label = ''  # will derive from state
    div_type = str(kv.get('chan_divergence', ''))
    wolf_sig = str(kv.get('chan_wolf', ''))
    trend_type = str(kv.get('chan_trend', ''))
    tf_align = str(kv.get('chan_tf_align', ''))

    # 笔状态机颜色
    if '(1,1)' in stroke_state:
        stroke_color = '#2e7d32'
        stroke_bg = '#e8f5e9'
        stroke_label = '向上笔确认'
    elif '(1,0)' in stroke_state:
        stroke_color = '#66bb6a'
        stroke_bg = '#f1f8e9'
        stroke_label = '向上笔延伸'
    elif '(-1,1)' in stroke_state:
        stroke_color = '#c62828'
        stroke_bg = '#ffebee'
        stroke_label = '向下笔确认'
    elif '(-1,0)' in stroke_state:
        stroke_color = '#ef5350'
        stroke_bg = '#fce4ec'
        stroke_label = '向下笔延伸'
    else:
        stroke_color = '#888'
        stroke_bg = '#f5f5f5'
        stroke_label = '无明确方向'

    # 背驰信号
    if div_type and div_type != 'None' and div_type != '无':
        if '底' in div_type:
            div_color = '#2e7d32'
            div_bg = '#e8f5e9'
            div_icon = '▲ 底部反转'
        else:
            div_color = '#c62828'
            div_bg = '#ffebee'
            div_icon = '▼ 顶部反转'
    else:
        div_color = '#888'
        div_bg = '#f5f5f5'
        div_icon = '无背驰'

    # 防狼术
    if 'danger' in wolf_sig:
        wolf_color = '#c62828'
        wolf_bg = '#ffebee'
        wolf_text = '空头主导·不入场'
    elif 'safe' in wolf_sig:
        wolf_color = '#2e7d32'
        wolf_bg = '#e8f5e9'
        wolf_text = '安全区·可操作'
    else:
        wolf_color = '#888'
        wolf_bg = '#f5f5f5'
        wolf_text = '未知'

    # 多级别联立
    if '共振看多' in tf_align:
        tf_color = '#2e7d32'
    elif '共振看空' in tf_align:
        tf_color = '#c62828'
    elif '回调' in tf_align or '反弹' in tf_align:
        tf_color = '#e65100'
    else:
        tf_color = '#888'

    return f"""
    <div style="margin-top:12px;padding:10px;background:#fafafa;border-radius:6px;border:1px solid #e0e0e0;">
        <b>[缠论分析]</b> (缠中说禅108课·多维度结构分析)
        <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;">
            <div style="flex:1;min-width:130px;padding:6px 8px;background:{stroke_bg};border-radius:4px;border-left:3px solid {stroke_color};">
                <div style="font-size:10px;color:#888;">笔状态机</div>
                <div style="font-weight:700;color:{stroke_color};font-size:13px;">{stroke_state}</div>
                <div style="font-size:10px;color:#666;">{stroke_label}</div>
            </div>
            <div style="flex:1;min-width:130px;padding:6px 8px;background:{div_bg};border-radius:4px;border-left:3px solid {div_color};">
                <div style="font-size:10px;color:#888;">MACD面积背驰</div>
                <div style="font-weight:700;color:{div_color};font-size:13px;">{div_icon}</div>
                <div style="font-size:10px;color:#666;">{div_type if div_type and div_type != 'None' else '无信号'}</div>
            </div>
            <div style="flex:1;min-width:130px;padding:6px 8px;background:{wolf_bg};border-radius:4px;border-left:3px solid {wolf_color};">
                <div style="font-size:10px;color:#888;">防狼术</div>
                <div style="font-weight:700;color:{wolf_color};font-size:13px;">{wolf_text}</div>
                <div style="font-size:10px;color:#666;">MACD 0轴判定</div>
            </div>
        </div>
        <div style="display:flex;gap:8px;margin-top:6px;flex-wrap:wrap;">
            <div style="flex:1;min-width:160px;padding:4px 8px;background:#f5f5f5;border-radius:3px;">
                <span style="font-size:10px;color:#888;">走势类型:</span>
                <span style="font-weight:600;font-size:12px;">{trend_type if trend_type else '?'}</span>
            </div>
            <div style="flex:1;min-width:220px;padding:4px 8px;background:#f5f5f5;border-radius:3px;">
                <span style="font-size:10px;color:#888;">多级别联立:</span>
                <span style="font-weight:600;font-size:12px;color:{tf_color};">{tf_align if tf_align else '?'}</span>
            </div>
        </div>
    </div>"""


def _build_busch_market_panel(lp):
    """构建Busch 2560 + 市场阶段 + 量价综合面板"""
    kv = lp.get('key_values', {})
    b2560_sig = str(kv.get('b2560_signal', ''))
    b2560_dir = str(kv.get('b2560_ma25_dir', ''))
    mphase = str(kv.get('mp_phase', ''))
    mp_cred = kv.get('mp_trend_cred', 0.5)
    vr_zone = str(kv.get('vr_zone', ''))
    vr_sig = str(kv.get('vr_signal', ''))
    vpa_sig = str(kv.get('vpa_signal', ''))
    bdmi_sig = str(kv.get('bdmi_signal', ''))
    vc_pat = str(kv.get('vc_pattern', ''))
    nrb_sig = str(kv.get('nrb_signal', ''))

    # 2560信号颜色
    if b2560_sig == 'strong_buy':
        b2560_color, b2560_bg, b2560_label = '#2e7d32', '#e8f5e9', '买入信号'
    elif b2560_sig == 'sell':
        b2560_color, b2560_bg, b2560_label = '#c62828', '#ffebee', '卖出信号'
    elif b2560_sig == 'hold_long':
        b2560_color, b2560_bg, b2560_label = '#66bb6a', '#f1f8e9', '持仓看多'
    elif b2560_sig == 'hold_short':
        b2560_color, b2560_bg, b2560_label = '#ef5350', '#fce4ec', '观望/空仓'
    else:
        b2560_color, b2560_bg, b2560_label = '#888', '#f5f5f5', '中性'

    # 市场阶段颜色
    if mphase == '拉升':
        mp_color, mp_bg = '#2e7d32', '#e8f5e9'
    elif mphase in ('拉升初期', '筑底'):
        mp_color, mp_bg = '#66bb6a', '#f1f8e9'
    elif mphase in ('盘头', '下跌'):
        mp_color, mp_bg = '#c62828', '#ffebee'
    else:
        mp_color, mp_bg = '#888', '#f5f5f5'

    # VR颜色
    if '底部' in vr_zone or '偏弱' in vr_zone:
        vr_color = '#1565c0'
    elif '头部' in vr_zone or '超买' in vr_zone:
        vr_color = '#c62828'
    elif '强势' in vr_zone or '偏强' in vr_zone:
        vr_color = '#2e7d32'
    else:
        vr_color = '#888'

    # DMI信号
    if bdmi_sig == 'bullish_reversal':
        bdmi_text, bdmi_color = '转折买入', '#2e7d32'
    elif bdmi_sig == 'bullish_trend':
        bdmi_text, bdmi_color = '趋势做多', '#66bb6a'
    elif bdmi_sig == 'bearish_trend':
        bdmi_text, bdmi_color = '趋势做空', '#c62828'
    elif bdmi_sig == 'weak_market':
        bdmi_text, bdmi_color = '弱势市场', '#f57c00'
    else:
        bdmi_text, bdmi_color = '中性', '#888'

    # NRB信号
    if 'bullish' in nrb_sig:
        nrb_color = '#2e7d32'
    elif 'bearish' in nrb_sig:
        nrb_color = '#c62828'
    else:
        nrb_color = '#888'

    return f"""
    <div style="margin-top:12px;padding:10px;background:#fafafa;border-radius:6px;border:1px solid #e0e0e0;">
        <b>[Busch 2560 · 市场阶段 · 量价综合]</b>
        <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;">
            <div style="flex:1;min-width:120px;padding:6px 8px;background:{b2560_bg};border-radius:4px;border-left:3px solid {b2560_color};">
                <div style="font-size:10px;color:#888;">Busch 2560战法</div>
                <div style="font-weight:700;color:{b2560_color};font-size:13px;">{b2560_label}</div>
                <div style="font-size:10px;color:#666;">MA25 {b2560_dir}</div>
            </div>
            <div style="flex:1;min-width:120px;padding:6px 8px;background:{mp_bg};border-radius:4px;border-left:3px solid {mp_color};">
                <div style="font-size:10px;color:#888;">市场阶段</div>
                <div style="font-weight:700;color:{mp_color};font-size:13px;">{mphase if mphase != 'unknown' else '?'}</div>
                <div style="font-size:10px;color:#666;">三势可信度{mp_cred:.0%}</div>
            </div>
            <div style="flex:1;min-width:120px;padding:6px 8px;background:#e8eaf6;border-radius:4px;border-left:3px solid {vr_color};">
                <div style="font-size:10px;color:#888;">VR容量比率</div>
                <div style="font-weight:700;color:{vr_color};font-size:13px;">{vr_zone}</div>
                <div style="font-size:10px;color:#666;">{vr_sig}</div>
            </div>
        </div>
        <div style="display:flex;gap:8px;margin-top:6px;flex-wrap:wrap;">
            <div style="flex:1;min-width:120px;padding:4px 8px;background:#f5f5f5;border-radius:3px;">
                <span style="font-size:10px;color:#888;">Busch DMI:</span>
                <span style="font-weight:600;font-size:12px;color:{bdmi_color};">{bdmi_text}</span>
            </div>
            <div style="flex:1;min-width:120px;padding:4px 8px;background:#f5f5f5;border-radius:3px;">
                <span style="font-size:10px;color:#888;">量价代数:</span>
                <span style="font-weight:600;font-size:12px;">{vpa_sig}</span>
            </div>
            <div style="flex:1;min-width:120px;padding:4px 8px;background:#f5f5f5;border-radius:3px;">
                <span style="font-size:10px;color:#888;">量能形态:</span>
                <span style="font-weight:600;font-size:12px;">{vc_pat}</span>
            </div>
            <div style="flex:1;min-width:100px;padding:4px 8px;background:#f5f5f5;border-radius:3px;">
                <span style="font-size:10px;color:#888;">NRB:</span>
                <span style="font-weight:600;font-size:12px;color:{nrb_color};">{nrb_sig}</span>
            </div>
        </div>
    </div>"""


def _build_panel1_local(local_plan, c, sym):
    """构建面板1: 本地量化模型"""
    if not local_plan:
        return '<div class="panel"><summary style="color:#999;">[本地量化模型] -- 无数据</summary></div>'

    lp = local_plan
    principles = lp.get('principles', [])
    kv = lp.get('key_values', {})
    targets = lp.get('targets', [])

    # 关键指标表
    kv_rows = ""
    for k, v in kv.items():
        if isinstance(v, float):
            v_str = f'{v:.4f}' if abs(v) < 0.01 else f'{v:.2f}'
        else:
            v_str = str(v)
        kv_rows += f"<tr><td>{_html_escape(k)}</td><td>{_html_escape(v_str)}</td></tr>"

    # 目标价
    tgt_html = ""
    if targets:
        for i, bt in enumerate(targets):
            tgt_html += f'<span class="price-box target">TP{i+1}: {bt.get("price","?")} ({bt.get("ratio","?")}) -- {_html_escape(bt.get("reason",""))}</span>'
    else:
        tgt_html = '<span style="color:#999;">无明确目标（震荡市或信号不明确）</span>'

    # 一次性止盈点
    one_shot = lp.get('one_shot_target', {})
    one_shot_html = ""
    if one_shot and one_shot.get('price', '-') != '-':
        one_shot_html = f'<span class="price-box target" style="background:#c8e6c9;border:2px solid #4caf50;">🎯 一次性止盈: {_html_escape(str(one_shot["price"]))} — {_html_escape(one_shot.get("reason",""))}</span>'

    # 判断原则
    pr_html = ""
    if principles:
        pr_html = "<ol class='principles-list'>" + "".join(f"<li>{_html_escape(p)}</li>" for p in principles) + "</ol>"

    # ===== Nüwa 7学派集成面板 =====
    ensemble_html = ""
    ensemble = lp.get('ensemble', {})
    if ensemble:
        votes = ensemble.get('votes', {})
        schools = ensemble.get('schools', {})
        diversity = ensemble.get('diversity', 0)
        synergy = ensemble.get('synergy_pairs', [])
        conflicts = ensemble.get('conflict_pairs', [])
        supporting = ensemble.get('supporting_reasons', [])
        opposing = ensemble.get('opposing_reasons', [])

        # 学派投票条
        def _vote_bar(label, count, total, color):
            pct = count / total * 100 if total > 0 else 0
            return f'<span style="display:inline-block;width:60px;">{_html_escape(label)}</span><span style="display:inline-block;background:{color};height:16px;width:{pct}%;min-width:20px;border-radius:3px;text-align:center;color:#fff;font-size:11px;line-height:16px;">{count}</span>'

        total_votes = sum(votes.values())
        vote_html = (
            _vote_bar('看多', votes.get('bullish', 0), total_votes, '#4caf50') +
            _vote_bar('看空', votes.get('bearish', 0), total_votes, '#f44336') +
            _vote_bar('观望', votes.get('neutral', 0), total_votes, '#ff9800')
        )

        # 7学派详情
        school_rows = ""
        for sname, sdata in schools.items():
            dir_cn = _html_escape(t(sdata['direction']))
            scolor = {'bullish': '#4caf50', 'bearish': '#f44336', 'neutral': '#ff9800'}.get(sdata['direction'], '#999')
            sreasons = _html_escape('; '.join(sdata.get('reasons', [])[:3]))
            school_rows += f"""
            <tr>
                <td style="font-weight:600;">{_html_escape(sdata['label'])}</td>
                <td style="color:{scolor};font-weight:600;">{dir_cn}</td>
                <td>{sdata['confidence']:.0%}</td>
                <td style="font-size:11px;color:#666;">{sreasons}</td>
            </tr>"""

        # 多样性指标
        div_color = '#4caf50' if diversity >= 0.3 else '#ff9800' if diversity >= 0.15 else '#f44336'
        div_label = '高度独立' if diversity >= 0.3 else '中等独立' if diversity >= 0.15 else '低度独立(可能同质化)'

        ensemble_html = f"""
        <div style="margin-top:14px;padding:10px;background:#f5f5f5;border-radius:6px;border-left:3px solid #1976d2;">
            <b>[Nüwa 7学派集成]</b> (7学派MoE框架)
            <div style="margin-top:6px;">
                <b>投票分布:</b> {vote_html}
                <span style="margin-left:12px;font-size:12px;color:#666;">多样性:{diversity:.2f}({div_label})</span>
            </div>
            <table style="width:100%;margin-top:8px;border-collapse:collapse;font-size:12px;">
                <tr style="background:#e0e0e0;"><th>学派</th><th>判断</th><th>置信度</th><th>关键理由</th></tr>
                {school_rows}
            </table>
            <div style="margin-top:6px;font-size:12px;">
                <b>协同对:</b> {len(synergy)}组 |
                <b>冲突对:</b> {len(conflicts)}组 |
                <b>融合类型:</b> {_html_escape(str(lp.get('fusion_type','?')))} |
                <b>贝叶斯:</b> {lp.get('bayes_posterior',0):.3f}({_html_escape(str(lp.get('bayes_signal','?')))})
            </div>"""

        if supporting:
            ensemble_html += f'<div style="margin-top:4px;font-size:12px;color:#2e7d32;"><b>支持理由:</b> {_html_escape("; ".join(supporting[:6]))}</div>'
        if opposing:
            ensemble_html += f'<div style="margin-top:2px;font-size:12px;color:#c62828;"><b>反对理由:</b> {_html_escape("; ".join(opposing[:4]))}</div>'

        ensemble_html += '</div>'

    # 精确入场价
    entry_price = lp.get('entry_price', '')
    entry_rationale = _html_escape(lp.get('entry_rationale', ''))
    entry_price_html = ""
    if entry_price and entry_price != '?':
        entry_price_html = f'<span class="price-box entry" style="background:#bbdefb;border:2px solid #1976d2;">📍 精确入场: {_html_escape(str(entry_price))}</span>'

    return f"""
    <details class="panel">
        <summary>[本地量化模型] -- <b style="color:{c['badge']};">{_html_escape(str(lp.get('signal_cn','?')))}</b> · 后验概率 {lp.get('confidence',0):.2%}</summary>
        <div class="panel-content">
            <div class="prices">
                <span class="price-box entry">入场区间: {_html_escape(str(lp.get('entry_zone','?')))}</span>
                {entry_price_html}
                <span class="price-box stop">止损: {_html_escape(str(lp.get('stop_loss','?')))} ({_html_escape(str(lp.get('stop_loss_rule','')))})</span>
                {one_shot_html}
                {tgt_html}
            </div>
            <div style="font-size:12px;color:#666;margin:4px 0;"><b>入场理由:</b> {entry_rationale}</div>
            {ensemble_html}
            {_build_bebe_panel(lp)}
            {_build_chanlun_panel(lp)}
            {_build_busch_market_panel(lp)}
            {_build_tang_livermore_panel(lp)}
            {_build_top_escape_panel(lp)}
            <div style="margin-top:12px;"><b>判断原则（{len(principles)}条）:</b></div>
            {pr_html}
            <div style="margin-top:12px;"><b>关键指标快照:</b></div>
            <table class="kv-table">{kv_rows}</table>
        </div>
    </details>"""


def _build_tang_livermore_panel(lp):
    """构建唐能通 + 利弗莫尔战法 + 解缠论增强面板"""
    kv = lp.get('key_values', {})
    parts = []

    # 唐能通区
    jiato = kv.get('tang_jiato', False)
    jiaya = kv.get('tang_jiaya', False)
    runway = str(kv.get('tang_runway', '?'))
    lyt = kv.get('tang_laoyatou', False)
    lyt_phase = str(kv.get('tang_laoyatou_phase', '?'))
    triple = str(kv.get('tang_triple', '无'))
    t33 = kv.get('tang_33_valid', False)

    tang_signals = []
    if jiato: tang_signals.append('<span style="color:#2e7d32;font-weight:bold;">价托(三角支撑)</span>')
    if jiaya: tang_signals.append('<span style="color:#c62828;font-weight:bold;">价压(三角压力)</span>')
    if lyt: tang_signals.append(f'<span style="color:#2e7d32;">🦆老鸭头({lyt_phase})</span>')
    if 'triple_golden' in triple: tang_signals.append('<span style="color:#2e7d32;">三金叉共振</span>')
    if 'triple_death' in triple: tang_signals.append('<span style="color:#c62828;">三死叉共振</span>')
    if t33: tang_signals.append('<span style="color:#1565c0;">三三过滤通过</span>')
    tang_display = ' | '.join(tang_signals) if tang_signals else '<span style="color:#888;">无唐能通信号</span>'

    # 跑道颜色
    if runway == 'thick': rw_color, rw_bg = '#2e7d32', '#e8f5e9'
    elif runway == 'moderate': rw_color, rw_bg = '#66bb6a', '#f1f8e9'
    elif runway in ('thin', 'too_thin'): rw_color, rw_bg = '#ef5350', '#fce4ec'
    else: rw_color, rw_bg = '#888', '#f5f5f5'

    # 利弗莫尔区
    liv_sig = str(kv.get('livermore_signal', '?'))
    liv_danger = kv.get('livermore_danger', False)
    liv_danger_level = kv.get('livermore_danger_level', 0)
    liv_action = str(kv.get('livermore_action', '?'))

    if liv_sig in ('strong_buy', 'buy'): liv_color = '#2e7d32'
    elif liv_sig in ('strong_sell', 'sell'): liv_color = '#c62828'
    else: liv_color = '#888'

    if liv_danger:
        danger_color = '#c62828' if liv_danger_level >= 50 else '#ef6c00'
        danger_text = f'⚠危险Lv.{liv_danger_level:.0f}→{liv_action}'
    else:
        danger_color = '#888'
        danger_text = '无危险信号'

    # 解缠论增强区
    cl_power = kv.get('cl_power_ratio', 1)
    cl_div = str(kv.get('cl_divergence_type', '无'))
    cl_cg = str(kv.get('cl_cg_direction', '?'))
    cl_ts = kv.get('cl_turn_score', 0)
    cl_ts_sig = str(kv.get('cl_turn_signal', '?'))
    cl_intra = kv.get('cl_intra_div', False)

    if cl_ts >= 80: ts_color = '#2e7d32' if 'buy' in cl_ts_sig else '#c62828'
    elif cl_ts >= 60: ts_color = '#66bb6a' if 'buy' in cl_ts_sig else '#ef6c00'
    else: ts_color = '#888'

    parts.append(f'''<div style="margin-top:10px;padding:8px 12px;border-radius:6px;background:#fafafa;border:1px solid #e0e0e0;">
    <div style="font-weight:bold;margin-bottom:6px;color:#333;">📊 唐能通 + 利弗莫尔 + 解缠论增强</div>
    <table style="width:100%%;font-size:12px;border-collapse:collapse;">
    <tr><td style="width:25%%;color:#666;">唐能通信号:</td><td>{tang_display}</td></tr>
    <tr><td style="color:#666;">跑道厚度:</td><td><span style="display:inline-block;padding:1px 8px;border-radius:10px;background:{rw_bg};color:{rw_color};font-weight:bold;">{runway}</span></td></tr>
    <tr><td style="color:#666;">三金叉/死叉:</td><td>{triple}</td></tr>
    <tr><td style="color:#666;">利弗莫尔方向:</td><td><span style="color:{liv_color};font-weight:bold;">{liv_sig}</span></td></tr>
    <tr><td style="color:#666;">危险信号:</td><td><span style="color:{danger_color};font-weight:bold;">{danger_text}</span></td></tr>
    <tr><td style="color:#666;">力度比:</td><td>{cl_power:.3f} ({cl_div})</td></tr>
    <tr><td style="color:#666;">中枢重心:</td><td>{cl_cg}</td></tr>
    <tr><td style="color:#666;">拐点评分/信号:</td><td><span style="color:{ts_color};font-weight:bold;">{cl_ts:.0f}分</span> {cl_ts_sig}</td></tr>
    <tr><td style="color:#666;">笔内背离:</td><td>{'有' if cl_intra else '无'}</td></tr>
    </table></div>''')

    return '\n'.join(parts)


def _build_top_escape_panel(lp):
    """构建逃顶检测 + 分配日计数面板"""
    kv = lp.get('key_values', {})
    top_count = kv.get('top_escape_count', 0)
    top_prob = kv.get('top_escape_prob', 0)
    top_grade = str(kv.get('top_escape_grade', '?'))
    dist_count = kv.get('dist_days_count', 0)
    dist_warn = kv.get('dist_days_warn', False)

    # 逃顶等级颜色
    if top_grade == 'critical':
        grade_color, grade_bg, grade_label = '#b71c1c', '#ffebee', 'CRITICAL — 立即离场'
    elif top_grade == 'high_risk':
        grade_color, grade_bg, grade_label = '#c62828', '#ffebee', '高风险 — 减仓至30%'
    elif top_grade == 'elevated_risk':
        grade_color, grade_bg, grade_label = '#ef6c00', '#fff3e0', '风险上升 — 减仓至50%'
    elif top_grade == 'early_warning':
        grade_color, grade_bg, grade_label = '#f9a825', '#fffde7', '早期预警 — 保持警惕'
    elif top_grade == 'safe':
        grade_color, grade_bg, grade_label = '#2e7d32', '#e8f5e9', '安全 — 无顶部信号'
    else:
        grade_color, grade_bg, grade_label = '#888', '#f5f5f5', top_grade

    # 分配日颜色
    if dist_warn and dist_count >= 5:
        dist_color = '#c62828'
        dist_text = f'⚠ {int(dist_count)}个分配日/25天→市场顶部'
    elif dist_warn:
        dist_color = '#ef6c00'
        dist_text = f'⚠ {int(dist_count)}个分配日/25天→警告'
    else:
        dist_color = '#888'
        dist_text = f'{int(dist_count)}个分配日/25天→正常'

    return f'''<div style="margin-top:10px;padding:8px 12px;border-radius:6px;background:#fafafa;border:1px solid #e0e0e0;">
    <div style="font-weight:bold;margin-bottom:6px;color:#333;">🚨 逃顶检测 + 分配日计数</div>
    <div style="margin-bottom:6px;">
        <span style="display:inline-block;padding:3px 12px;border-radius:12px;background:{grade_bg};color:{grade_color};font-weight:bold;font-size:14px;">{grade_label}</span>
        <span style="font-size:12px;color:#666;margin-left:8px;">逃顶概率 {top_prob:.0f}% | 信号数 {int(top_count)}</span>
    </div>
    <div style="font-size:12px;">
        <span style="color:{dist_color};font-weight:bold;">{dist_text}</span>
    </div>
    </div>'''

def _build_panel1_local_coequal(local_plan, c, sym):
    """构建面板1: 本地量化模型（无折叠，用于并排展示）"""
    if not local_plan:
        return '<div style="color:#999;padding:20px;text-align:center;">无本地模型数据</div>'

    lp = local_plan
    principles = lp.get('principles', [])
    kv = lp.get('key_values', {})
    targets = lp.get('targets', [])

    # 关键指标表
    kv_rows = ""
    for k, v in kv.items():
        if isinstance(v, float):
            v_str = f'{v:.4f}' if abs(v) < 0.01 else f'{v:.2f}'
        else:
            v_str = str(v)
        kv_rows += f"<tr><td>{_html_escape(k)}</td><td>{_html_escape(v_str)}</td></tr>"

    # 目标价
    tgt_html = ""
    if targets:
        for i, bt in enumerate(targets):
            tgt_html += f'<span class="price-box target">TP{i+1}: {bt.get("price","?")} ({bt.get("ratio","?")}) -- {_html_escape(bt.get("reason",""))}</span>'
    else:
        tgt_html = '<span style="color:#999;">无明确目标（震荡市或信号不明确）</span>'

    # 一次性止盈点
    one_shot = lp.get('one_shot_target', {})
    one_shot_html = ""
    if one_shot and one_shot.get('price', '-') != '-':
        one_shot_html = f'<span class="price-box target" style="background:#c8e6c9;border:2px solid #4caf50;">一次性止盈: {_html_escape(str(one_shot["price"]))} -- {_html_escape(one_shot.get("reason",""))}</span>'

    # 判断原则
    pr_html = ""
    if principles:
        pr_html = "<ol class='principles-list'>" + "".join(f"<li>{_html_escape(p)}</li>" for p in principles) + "</ol>"

    # Nüwa 7学派集成面板
    ensemble_html = ""
    ensemble = lp.get('ensemble', {})
    if ensemble:
        votes = ensemble.get('votes', {})
        schools = ensemble.get('schools', {})
        diversity = ensemble.get('diversity', 0)
        synergy = ensemble.get('synergy_pairs', [])
        conflicts = ensemble.get('conflict_pairs', [])
        supporting = ensemble.get('supporting_reasons', [])
        opposing = ensemble.get('opposing_reasons', [])

        def _vote_bar(label, count, total, color):
            pct = count / total * 100 if total > 0 else 0
            return f'<span style="display:inline-block;width:50px;">{_html_escape(label)}</span><span style="display:inline-block;background:{color};height:14px;width:{pct}%;min-width:16px;border-radius:3px;text-align:center;color:#fff;font-size:10px;line-height:14px;">{count}</span>'

        total_votes = sum(votes.values())
        vote_html = (
            _vote_bar('多', votes.get('bullish', 0), total_votes, '#4caf50') +
            _vote_bar('空', votes.get('bearish', 0), total_votes, '#f44336') +
            _vote_bar('观', votes.get('neutral', 0), total_votes, '#ff9800')
        )

        school_rows = ""
        for sname, sdata in schools.items():
            dir_cn = _html_escape(t(sdata['direction']))
            scolor = {'bullish': '#4caf50', 'bearish': '#f44336', 'neutral': '#ff9800'}.get(sdata['direction'], '#999')
            sreasons = _html_escape('; '.join(sdata.get('reasons', [])[:2]))
            school_rows += f"""
            <tr>
                <td style="font-weight:600;font-size:11px;">{_html_escape(sdata['label'])}</td>
                <td style="color:{scolor};font-weight:600;font-size:11px;">{dir_cn}</td>
                <td style="font-size:11px;">{sdata['confidence']:.0%}</td>
                <td style="font-size:10px;color:#666;">{sreasons}</td>
            </tr>"""

        div_color = '#4caf50' if diversity >= 0.3 else '#ff9800' if diversity >= 0.15 else '#f44336'
        div_label = '高度独立' if diversity >= 0.3 else '中等独立' if diversity >= 0.15 else '低度独立'

        ensemble_html = f"""
        <div style="margin-top:10px;padding:8px;background:#f5f5f5;border-radius:6px;border-left:3px solid #1976d2;">
            <b style="font-size:12px;">[Nüwa 7学派集成]</b>
            <div style="margin-top:4px;font-size:11px;">
                <b>投票:</b> {vote_html}
                <span style="margin-left:8px;color:#666;">多样性:{diversity:.2f}({div_label})</span>
            </div>
            <table style="width:100%;margin-top:6px;border-collapse:collapse;font-size:11px;">
                <tr style="background:#e0e0e0;"><th>学派</th><th>判断</th><th>置信</th><th>理由</th></tr>
                {school_rows}
            </table>
            <div style="margin-top:4px;font-size:10px;color:#666;">
                协同对:{len(synergy)} | 冲突对:{len(conflicts)} | 融合:{_html_escape(str(lp.get('fusion_type','?')))} | 贝叶斯:{lp.get('bayes_posterior',0):.3f}
            </div>"""

        if supporting:
            ensemble_html += f'<div style="margin-top:2px;font-size:10px;color:#2e7d32;"><b>支持:</b> {_html_escape("; ".join(supporting[:4]))}</div>'
        if opposing:
            ensemble_html += f'<div style="margin-top:1px;font-size:10px;color:#c62828;"><b>反对:</b> {_html_escape("; ".join(opposing[:3]))}</div>'

        ensemble_html += '</div>'

    # 精确入场价
    entry_price = lp.get('entry_price', '')
    entry_rationale = _html_escape(lp.get('entry_rationale', ''))
    entry_price_html = ""
    if entry_price and entry_price != '?':
        entry_price_html = f'<span class="price-box entry" style="background:#bbdefb;border:2px solid #1976d2;">精确入场: {_html_escape(str(entry_price))}</span>'

    return f"""
    <div class="prices" style="font-size:12px;">
        <span class="price-box entry">入场: {_html_escape(str(lp.get('entry_zone','?')))}</span>
        {entry_price_html}
        <span class="price-box stop">止损: {_html_escape(str(lp.get('stop_loss','?')))} ({_html_escape(str(lp.get('stop_loss_rule','')))})</span>
        {one_shot_html}
        {tgt_html}
    </div>
    <div style="font-size:11px;color:#666;margin:4px 0;"><b>入场理由:</b> {entry_rationale}</div>
    {ensemble_html}
    {_build_bebe_panel(lp)}
    {_build_chanlun_panel(lp)}
    {_build_busch_market_panel(lp)}
    {_build_tang_livermore_panel(lp)}
    {_build_top_escape_panel(lp)}
    <div style="margin-top:10px;"><b style="font-size:12px;">判断原则（{len(principles)}条）:</b></div>
    {pr_html}
    <div style="margin-top:10px;"><b style="font-size:12px;">关键指标快照:</b></div>
    <table class="kv-table">{kv_rows}</table>"""

def _build_panel2_ai(s, c):
    """构建面板2: AI深度分析"""
    sig = s.get('signal', 'neutral')
    entry = _html_escape(s.get('entry_zone', ''))
    stop = _html_escape(s.get('stop_loss', ''))
    stop_rule = _html_escape(s.get('stop_loss_rule', ''))
    trailing = _html_escape(s.get('trailing_stop', ''))
    resonance = _html_escape(s.get('resonance_detail', ''))
    entry_reason = _html_escape(s.get('entry_reason', ''))
    batch_targets = s.get('batch_targets', [])
    batch_sl = s.get('batch_stop_loss', [])
    position = _html_escape(s.get('position_advice', ''))
    risk = _html_escape(s.get('risk_reward_analysis', ''))
    invalidation = _html_escape(s.get('invalidation_condition', ''))

    tgt_html = ""
    if batch_targets:
        tgt_html = "".join(
            f'<span class="price-box target">TP{t+1}: {bt.get("price","?")} ({bt.get("ratio","?")}) — {_html_escape(bt.get("reason",""))}</span>'
            for t, bt in enumerate(batch_targets[:3])
        )

    sl_str = str(stop) if stop else ''
    if batch_sl:
        sl_str += "<br>" + "<br>".join(
            f"{_html_escape(bs.get('trigger',''))} → {_html_escape(bs.get('new_stop',''))}"
            for bs in batch_sl[:2]
        )

    return f"""
    <details class="panel">
        <summary>🤖 AI深度分析 — <b style="color:{c['badge']};">{_html_escape(t(sig))}</b> · 置信度 {s.get('confidence',0):.0%} · 周线: {_html_escape(t(s.get('weekly_align','')))}</summary>
        <div class="panel-content">
            <div><b>共振:</b> {resonance}</div>
            <div><b>入场理由:</b> {entry_reason}</div>
            <div class="prices">
                <span class="price-box entry">🎯 入场: {entry}</span>
                <span class="price-box stop">🛑 止损: {sl_str}</span>
                {tgt_html}
            </div>
            <div class="meta-row">
                <span class="meta-item"><b>止损规则:</b> {stop_rule}</span>
            </div>
            <div class="meta-row">
                <span class="meta-item"><b>移动止损:</b> {trailing if trailing else '无'}</span>
            </div>
            <div class="meta-row">
                <span class="meta-item"><b>仓位建议:</b> {position}</span>
            </div>
            <div class="meta-row">
                <span class="meta-item" style="color:#888;"><b>失效条件:</b> {invalidation}</span>
            </div>
            <div class="meta-row">
                <span class="meta-item" style="color:#888;"><b>盈亏比:</b> {risk}</span>
            </div>
        </div>
    </details>"""

def _build_panel2_ai_coequal(s, c):
    """构建面板2: AI深度分析（无折叠，用于并排展示）"""
    sig = s.get('signal', 'neutral')
    entry = _html_escape(s.get('entry_zone', ''))
    stop = _html_escape(s.get('stop_loss', ''))
    stop_rule = _html_escape(s.get('stop_loss_rule', ''))
    trailing = _html_escape(s.get('trailing_stop', ''))
    resonance = _html_escape(s.get('resonance_detail', ''))
    entry_reason = _html_escape(s.get('entry_reason', ''))
    batch_targets = s.get('batch_targets', [])
    batch_sl = s.get('batch_stop_loss', [])
    position = _html_escape(s.get('position_advice', ''))
    risk = _html_escape(s.get('risk_reward_analysis', ''))
    invalidation = _html_escape(s.get('invalidation_condition', ''))

    tgt_html = ""
    if batch_targets:
        tgt_html = "".join(
            f'<span class="price-box target">TP{t+1}: {bt.get("price","?")} ({bt.get("ratio","?")}) -- {_html_escape(bt.get("reason",""))}</span>'
            for t, bt in enumerate(batch_targets[:3])
        )

    sl_str = str(stop) if stop else ''
    if batch_sl:
        sl_str += "<br>" + "<br>".join(
            f"{_html_escape(bs.get('trigger',''))} -> {_html_escape(bs.get('new_stop',''))}"
            for bs in batch_sl[:2]
        )

    return f"""
    <div style="font-size:13px;"><b>共振:</b> {resonance}</div>
    <div style="font-size:13px;"><b>入场理由:</b> {entry_reason}</div>
    <div class="prices" style="font-size:12px;">
        <span class="price-box entry">入场: {entry}</span>
        <span class="price-box stop">止损: {sl_str}</span>
        {tgt_html}
    </div>
    <div class="meta-row" style="font-size:12px;">
        <span class="meta-item"><b>止损规则:</b> {stop_rule}</span>
    </div>
    <div class="meta-row" style="font-size:12px;">
        <span class="meta-item"><b>移动止损:</b> {trailing if trailing else '无'}</span>
    </div>
    <div class="meta-row" style="font-size:12px;">
        <span class="meta-item"><b>仓位建议:</b> {position}</span>
    </div>
    <div class="meta-row" style="font-size:12px;color:#888;">
        <span class="meta-item"><b>失效条件:</b> {invalidation}</span>
    </div>
    <div class="meta-row" style="font-size:12px;color:#888;">
        <span class="meta-item"><b>盈亏比:</b> {risk}</span>
    </div>"""

def _build_panel3_merged(s, local_plan, c):
    """构建面板3: 综合结论"""
    sig = s.get('signal', 'neutral')
    alignment = s.get('bayesian_alignment', '?')
    align_cn = _html_escape(t(alignment))

    kelly_pct = float(s.get('kelly_position_pct', 0) or 0)
    kelly_fstar = float(s.get('kelly_f_star', 0) or 0)
    kelly_fhalf = float(s.get('kelly_f_half', 0) or 0)
    kelly_win = float(s.get('kelly_win_prob', 0) or 0)
    kelly_payoff = float(s.get('kelly_payoff_ratio', 0) or 0)
    protections = s.get('ergo_protections', [])
    safe = s.get('ergo_safe_to_enter', False)

    bayes_post = s.get('bayesian_posterior', 0)
    local_signal_cn = _html_escape(str(local_plan.get('signal_cn', '?'))) if local_plan else '?'

    # 对齐状态颜色
    if alignment == '一致':
        align_color = '#2e7d32'
        align_icon = '[OK]'
    elif alignment == '冲突':
        align_color = '#c62828'
        align_icon = '[!!]'
    else:
        align_color = '#e65100'
        align_icon = '[~]'

    # === MARL多周期意图协调 ===
    tf_coord = s.get('_tf_coordination', {})
    tf_html = ""
    if tf_coord:
        tf_quality = tf_coord.get('coordination_quality', 'moderate')
        tf_adj_conf = tf_coord.get('adjusted_confidence', 0)
        tf_mult = tf_coord.get('confidence_multiplier', 1.0)
        tf_quality_cn = _html_escape({'strong': '强协调(多周期一致)', 'moderate': '中等协调', 'weak': '弱协调(存在分歧)', 'conflict': '冲突(建议观望)'}.get(tf_quality, '?'))
        tf_qcolor = {'strong': '#2e7d32', 'moderate': '#ff9800', 'weak': '#e65100', 'conflict': '#c62828'}.get(tf_quality, '#999')
        tf_weekly_detail = _html_escape(str(tf_coord.get('weekly_detail', '')))
        tf_hourly_detail = _html_escape(str(tf_coord.get('hourly_detail', '')))

        tf_html = f"""
        <div class="merge-section">
            <div class="merge-title">[多周期协调] MARL意图协同</div>
            <div class="meta-row">
                <span class="meta-item">协调质量: <b style="color:{tf_qcolor};">{tf_quality_cn}</b></span>
                <span class="meta-item">调整后置信度: <b>{tf_adj_conf:.3f}</b></span>
                <span class="meta-item">乘数: <b>{tf_mult:.2f}x</b></span>
            </div>
            <div class="meta-row" style="font-size:12px;color:#666;">
                <span class="meta-item">周线:{tf_weekly_detail}</span>
            </div>
            <div class="meta-row" style="font-size:12px;color:#666;">
                <span class="meta-item">60分钟:{tf_hourly_detail}</span>
            </div>"""

        # Add round details
        rounds = tf_coord.get('rounds', [])
        if rounds:
            tf_html += '<div class="meta-row" style="margin-top:4px;">'
            for rd in rounds:
                rn = _html_escape(str(rd.get('round', '?')))
                rtf = _html_escape(str(rd.get('timeframe', '?')))
                intent = _html_escape(str(rd.get('intention', rd.get('trend', '?'))))
                conf = rd.get('confidence', rd.get('adjustment', 0))
                tf_html += f'<span class="meta-item" style="font-size:11px;">R{rn}-{rtf}:{intent}({conf:.2f})</span>'
            tf_html += '</div>'

        tf_html += '</div>'

    # 合并价位: AI为主，本地补充
    ai_entry = _html_escape(str(s.get('entry_zone', ''))) if s.get('entry_zone') else ''
    ai_stop = _html_escape(str(s.get('stop_loss', ''))) if s.get('stop_loss') else ''
    local_entry = _html_escape(str(local_plan.get('entry_zone', ''))) if local_plan else ''
    local_stop = _html_escape(str(local_plan.get('stop_loss', ''))) if local_plan else ''

    prot_tags = "".join(f'<span class="protect-tag">{_html_escape(t(p))}</span>' for p in protections) if protections else '<span style="color:#888;">无</span>'

    return f"""
    <details class="panel" open>
        <summary>[综合结论] -- <b style="color:{align_color};">{align_icon} AI/贝叶斯: {align_cn}</b></summary>
        <div class="panel-content">
            <div class="merge-section">
                <div class="merge-title">信号对齐状态</div>
                <div class="meta-row">
                    <span class="meta-item">AI信号: <b style="color:{c['badge']};">{_html_escape(t(sig))}</b> (置信 {s.get('confidence',0):.0%})</span>
                    <span class="meta-item">贝叶斯后验: <b>{(bayes_post or 0):.4f}</b> ({local_signal_cn})</span>
                    <span class="meta-item">对齐: <b style="color:{align_color};">{align_cn}</b></span>
                </div>
            </div>
            {tf_html}
                <div class="meta-row">
                    <span class="meta-item">全凯利f*: <b>{kelly_fstar:.2%}</b></span>
                    <span class="meta-item">半凯利f: <b>{kelly_fhalf:.2%}</b></span>
                    <span class="meta-item">最终仓位: <b style="color:{c['badge']};">{kelly_pct:.1f}%</b></span>
                    <span class="meta-item">收缩胜率: <b>{kelly_win:.2%}</b></span>
                    <span class="meta-item">盈亏比: <b>{kelly_payoff:.2f}</b></span>
                </div>
                <div class="meta-row" style="margin-top:4px;">
                    <span class="meta-item"><b>保护措施:</b> {prot_tags}</span>
                </div>
                <div class="meta-row">
                    <span class="meta-item">入场安全: <b>{'✅ 是' if safe else '❌ 否'}</b></span>
                </div>
            </div>
            <div class="merge-section">
                <div class="merge-title">📋 综合价位参考</div>
                <div class="prices">
                    <span class="price-box entry">AI入场: {ai_entry}</span>
                    <span class="price-box entry">本地入场: {local_entry}</span>
                    <span class="price-box stop">AI止损: {ai_stop}</span>
                    <span class="price-box stop">本地止损: {local_stop}</span>
                </div>
            </div>
            <div style="margin-top:8px; font-size:12px; color:#888;">
                💡 综合建议：AI与本地模型{ '方向一致，信号可靠性较高' if alignment == '一致' else '存在分歧，建议降低仓位或观望' if alignment == '冲突' else '部分偏离，需谨慎判断' }
                { '，凯利建议仓位 {:.1f}%'.format(kelly_pct) if kelly_pct > 0 else '，当前不建议入场' }
            </div>
        </div>
    </details>"""
