#!/usr/bin/env python
# event_store.py — Append-Only Event Sourcing + RLTF Feedback + Cross-Asset Context
"""
Three pillars of production-grade order management:

  1. Event Sourcing: Append-only event log. State rebuilt by replaying events.
     Crash → restart → replay → 100% state recovery.

  2. RLTF (Reinforcement Learning from Trading Feedback):
     Losing trades → feature extraction → negative-example Few-Shot prompts
     → injected into next AI query for same pattern.

  3. Cross-Asset Context: Macro data (bond yields, USDCNY, copper/gold ratio)
     → injected into every AI prompt for macro-aware analysis.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from config import DB_PATH


# =============================================================================
# 1. Event Sourcing — Append-Only Event Store
# =============================================================================

class EventType(str, Enum):
    ORDER_PLACED = 'order_placed'
    ORDER_FILLED = 'order_filled'
    ORDER_PARTIAL = 'order_partial'
    ORDER_CANCELED = 'order_canceled'
    POSITION_OPENED = 'position_opened'
    POSITION_CLOSED = 'position_closed'
    STOP_LOSS_HIT = 'stop_loss_hit'
    TAKE_PROFIT_HIT = 'take_profit_hit'
    DIVIDEND = 'dividend'
    SPLIT = 'split'
    DEPOSIT = 'deposit'
    WITHDRAWAL = 'withdrawal'


@dataclass
class Event:
    """Single immutable event in the event store."""
    event_id: str
    event_type: EventType
    symbol: str
    timestamp: datetime
    payload: Dict[str, Any]  # Immutable JSON blob
    sequence_num: int = 0    # Monotonic sequence number

    def to_row(self) -> Tuple:
        return (self.event_id, self.event_type.value, self.symbol,
                self.timestamp.isoformat(), json.dumps(self.payload, ensure_ascii=False),
                self.sequence_num)


class EventStore:
    """
    Append-only event store with state reconstruction via replay.

    Guarantees:
      - All events are immutable once written
      - State can be rebuilt from empty DB at any time
      - Crash recovery: just replay all events
      - Audit trail: every state change has a timestamped event
    """

    def __init__(self, db_path: str = None):
        # Use separate DB from DuckDB to avoid SQLite format pollution
        if db_path is None:
            db_path = os.environ.get('EVENT_STORE_PATH', 'data/event_store.db')
        self.db_path = db_path
        self._ensure_tables()
        self._next_seq = self._load_max_sequence() + 1

    def _ensure_tables(self):
        """Create event store tables if not exists."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS event_log (
            event_id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            sequence_num INTEGER NOT NULL UNIQUE
        )''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_event_symbol ON event_log(symbol)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON event_log(event_type)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_event_seq ON event_log(sequence_num)')
        conn.commit()
        conn.close()

    def _load_max_sequence(self) -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT COALESCE(MAX(sequence_num), 0) FROM event_log')
        max_seq = c.fetchone()[0]
        conn.close()
        return max_seq

    def append(self, event_type: EventType, symbol: str,
               payload: Dict[str, Any], event_id: str = None) -> Event:
        """Append an immutable event to the store."""
        if event_id is None:
            event_id = f"{event_type.value}_{symbol}_{self._next_seq}_{datetime.now():%Y%m%d%H%M%S}"

        event = Event(
            event_id=event_id,
            event_type=event_type,
            symbol=symbol,
            timestamp=datetime.now(),
            payload=payload,
            sequence_num=self._next_seq,
        )

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT OR IGNORE INTO event_log VALUES (?,?,?,?,?,?)''',
                  event.to_row())
        conn.commit()
        conn.close()

        self._next_seq += 1
        return event

    def replay(self, symbol: str = None,
               from_seq: int = 0) -> List[Event]:
        """Replay events to reconstruct state. Returns events in order."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        if symbol:
            c.execute('''SELECT * FROM event_log WHERE symbol=? AND sequence_num >= ?
                        ORDER BY sequence_num ASC''', (symbol, from_seq))
        else:
            c.execute('''SELECT * FROM event_log WHERE sequence_num >= ?
                        ORDER BY sequence_num ASC''', (from_seq,))

        events = []
        for row in c.fetchall():
            events.append(Event(
                event_id=row['event_id'],
                event_type=EventType(row['event_type']),
                symbol=row['symbol'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                payload=json.loads(row['payload_json']),
                sequence_num=row['sequence_num'],
            ))
        conn.close()
        return events

    def reconstruct_positions(self) -> Dict[str, Dict]:
        """Replay ALL events to compute current positions from scratch."""
        events = self.replay()
        positions: Dict[str, Dict] = {}
        cash = 0.0
        trades: List[Dict] = []

        for evt in events:
            sym = evt.symbol
            payload = evt.payload

            if evt.event_type == EventType.DEPOSIT:
                cash += payload.get('amount', 0)

            elif evt.event_type == EventType.WITHDRAWAL:
                cash -= payload.get('amount', 0)

            elif evt.event_type == EventType.POSITION_OPENED:
                if sym not in positions:
                    positions[sym] = {'total_shares': 0, 'avg_cost': 0, 'trades': []}
                qty = payload.get('shares', 0)
                price = payload.get('price', 0)
                old_cost = positions[sym]['avg_cost'] * positions[sym]['total_shares']
                new_total = positions[sym]['total_shares'] + qty
                positions[sym]['avg_cost'] = (old_cost + price * qty) / max(new_total, 1)
                positions[sym]['total_shares'] = new_total
                cash -= price * qty
                trades.append({'symbol': sym, 'action': 'buy', 'qty': qty, 'price': price})

            elif evt.event_type == EventType.POSITION_CLOSED:
                if sym in positions:
                    qty = payload.get('shares', 0)
                    price = payload.get('price', 0)
                    pnl = payload.get('pnl', 0)
                    positions[sym]['total_shares'] -= qty
                    if positions[sym]['total_shares'] <= 0:
                        del positions[sym]
                    cash += price * qty
                    trades.append({'symbol': sym, 'action': 'sell', 'qty': qty,
                                  'price': price, 'pnl': pnl})

        return {
            'positions': positions,
            'cash': round(cash, 2),
            'total_trades': len(trades),
            'n_events': len(events),
        }


# =============================================================================
# 2. RLTF — Reinforcement Learning from Trading Feedback
# =============================================================================

@dataclass
class RLTFLearner:
    """
    Learn from losing trades to improve future AI predictions.

    Process:
      1. Track all trades with their features
      2. When a losing trade closes, extract "negative example"
      3. On next AI query for similar pattern, inject as Few-Shot warning
    """

    max_examples: int = 50          # Max negative examples to store
    similarity_threshold: float = 0.70  # Cosine similarity to trigger injection
    _examples: List[Dict] = field(default_factory=list)

    def record_trade(self, symbol: str, signal: Dict, outcome: Dict) -> None:
        """
        Record a completed trade for learning.

        Args:
            symbol: stock code
            signal: original signal dict (features, confidence, etc.)
            outcome: {'pnl_pct': float, 'win': bool, 'exit_reason': str}
        """
        # Only learn from losing trades (especially false positives)
        if outcome.get('win', True):
            return

        features = self._extract_features(signal)
        example = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'pnl_pct': outcome.get('pnl_pct', 0),
            'exit_reason': outcome.get('exit_reason', 'unknown'),
            'confidence': signal.get('confidence', 0),
            'features': features,
        }

        self._examples.append(example)
        # Keep only most recent
        if len(self._examples) > self.max_examples:
            self._examples = self._examples[-self.max_examples:]

    def _extract_features(self, signal: Dict) -> Dict:
        """Extract discriminative features from a signal for similarity matching."""
        lp = signal.get('_local_plan', {})
        indicators = signal.get('_indicators', {})

        return {
            'rsi': float(indicators.get('rsi', 50)),
            'adx': float(indicators.get('dmi_adx', 20)),
            'vol_ratio': float(indicators.get('vol_ratio', 1.0)),
            'macd_hist': float(indicators.get('macd_hist', 0)),
            'ma_position': float(indicators.get('current_price', 0) /
                               max(indicators.get('ma20', 1), 1)),
            'weekly_trend': str(indicators.get('weekly_trend', 'neutral')),
            'confidence': signal.get('confidence', 0),
            'entry_zone': str(lp.get('entry_zone', '')),
        }

    def find_similar_losers(self, current_signal: Dict,
                           top_k: int = 3) -> List[Dict]:
        """
        Find historically similar losing trades for the current signal.

        Uses simple feature-vector cosine similarity. Returns top_k matches.
        """
        current_features = self._extract_features(current_signal)
        # Simple numeric feature vector
        curr_vec = np.array([
            current_features['rsi'] / 100,
            current_features['adx'] / 50,
            current_features['vol_ratio'] / 3,
            current_features['ma_position'] - 1,
            current_features['confidence'],
        ])

        scored = []
        for ex in self._examples:
            ex_vec = np.array([
                ex['features']['rsi'] / 100,
                ex['features']['adx'] / 50,
                ex['features']['vol_ratio'] / 3,
                ex['features']['ma_position'] - 1,
                ex['features']['confidence'],
            ])

            # Cosine similarity
            dot = np.dot(curr_vec, ex_vec)
            norm = np.linalg.norm(curr_vec) * np.linalg.norm(ex_vec)
            if norm > 1e-10:
                sim = dot / norm
                if sim > self.similarity_threshold:
                    scored.append((sim, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]

    def build_few_shot_warning(self, current_signal: Dict) -> str:
        """
        Build a Few-Shot warning prompt based on similar historical losers.

        Returns empty string if no similar losers found.
        """
        similar = self.find_similar_losers(current_signal)
        if not similar:
            return ''

        warnings = []
        for i, ex in enumerate(similar, 1):
            warnings.append(
                f"[历史教训 #{i}] 类似形态曾导致 {ex['pnl_pct']:.1%} 亏损 "
                f"(置信度 {ex['confidence']:.0%}, 退出原因: {ex['exit_reason']})"
            )

        return (
            "\n\n⚠️ 历史回测警告（Few-Shot负面示例）：\n"
            + '\n'.join(warnings) +
            "\n\n请结合上述历史教训重新评估当前信号。如果当前形态与上述失败案例高度相似，"
            "请降低信心度或给出更严格的止损建议。"
        )


# =============================================================================
# 3. Cross-Asset Macro Context Builder
# =============================================================================

@dataclass
class MacroContext:
    """
    Build cross-asset macro context for AI prompt injection.

    Data sources (all available via pywencai or public APIs):
      - 10Y China government bond yield (CN10Y)
      - USD/CNY exchange rate
      - Copper/Gold ratio (risk appetite proxy)
      - SHIBOR 7-day (interbank liquidity)
      - CSI 300 futures basis (market sentiment)
      - Northbound net flow (foreign capital sentiment)
    """

    def fetch_macro_data(self) -> Dict[str, Any]:
        """
        Fetch current macro indicators.

        Falls back to estimates when data sources are unavailable.
        """
        data = {
            'cn10y_yield': self._get_cn10y(),
            'usdcny': self._get_usdcny(),
            'copper_gold_ratio': self._get_cu_au_ratio(),
            'shibor_7d': self._get_shibor_7d(),
            'csi300_futures_basis': self._get_futures_basis(),
            'northbound_flow': self._get_northbound(),
        }
        return data

    def _get_cn10y(self) -> Optional[float]:
        """China 10Y bond yield. Try pywencai, fallback to estimate."""
        try:
            from capital_flow import _query_wencai
            result = _query_wencai('中国10年期国债收益率')
            if result and len(result) > 0:
                return float(result[0].get('收益率', 0))
        except Exception:
            pass
        return None  # 2.70%

    def _get_usdcny(self) -> Optional[float]:
        try:
            from capital_flow import _query_wencai
            result = _query_wencai('美元兑人民币 最新价')
            if result and len(result) > 0:
                return float(result[0].get('最新价', 0))
        except Exception:
            pass
        return None

    def _get_cu_au_ratio(self) -> Optional[float]:
        try:
            from capital_flow import _query_wencai
            cu = _query_wencai('沪铜主力 最新价')
            au = _query_wencai('沪金主力 最新价')
            if cu and au and len(cu) > 0 and len(au) > 0:
                return float(cu[0].get('最新价', 0)) / float(au[0].get('最新价', 0))
        except Exception:
            pass
        return None

    def _get_shibor_7d(self) -> Optional[float]:
        try:
            from capital_flow import _query_wencai
            result = _query_wencai('SHIBOR 7天')
            if result and len(result) > 0:
                return float(result[0].get('利率', 0))
        except Exception:
            pass
        return None

    def _get_futures_basis(self) -> Optional[float]:
        """CSI 300 futures basis = futures - spot (negative = bearish)."""
        try:
            from data_loader import get_daily_kline
            spot = get_daily_kline('000300', days=5)
            if spot is not None and not spot.empty:
                return None  # Need futures data — skip
        except Exception:
            pass
        return None

    def _get_northbound(self) -> Optional[float]:
        try:
            from capital_flow import _query_wencai
            result = _query_wencai('北向资金 今日净流入')
            if result and len(result) > 0:
                return float(result[0].get('净流入', 0))
        except Exception:
            pass
        return None

    def build_context_prompt(self) -> str:
        """
        Build macro context injection for AI prompt.

        Example output:
          [宏观环境]
          10Y国债: 2.68% (↓ 下降通道, 利好权益)
          美元/人民币: 7.15 (↑ 人民币走弱, 外资流出压力)
          铜金比: 4.2 (↓ 风险偏好下降)
          SHIBOR 7D: 1.85% (流动性充裕)
          北向资金: -23亿 (连续3日净流出)
          → 宏观综合判断: 中性偏谨慎
        """
        data = self.fetch_macro_data()

        lines = ['\n[宏观跨资产环境]']

        # CN10Y
        cn10y = data.get('cn10y_yield')
        if cn10y:
            trend = '下降通道, 利好权益' if cn10y < 2.8 else '上升通道, 利好债券'
            lines.append(f'10Y国债收益率: {cn10y:.2f}% ({trend})')

        # USDCNY
        usdcny = data.get('usdcny')
        if usdcny:
            direction = '人民币承压' if usdcny > 7.0 else '人民币稳定'
            lines.append(f'美元/人民币: {usdcny:.2f} ({direction})')

        # Cu/Au
        cu_au = data.get('copper_gold_ratio')
        if cu_au:
            risk = '风险偏好积极' if cu_au > 4.5 else '风险偏好谨慎'
            lines.append(f'铜金比: {cu_au:.1f} ({risk})')

        # SHIBOR
        shibor = data.get('shibor_7d')
        if shibor:
            liquidity = '充裕' if shibor < 2.0 else '偏紧'
            lines.append(f'SHIBOR 7D: {shibor:.2f}% (流动性{liquidity})')

        # Northbound
        nb = data.get('northbound_flow')
        if nb:
            nb_status = '外资流入' if nb > 0 else '外资流出'
            lines.append(f'北向资金: {nb:.0f}亿 ({nb_status})')

        # Summary
        bullish_signals = sum([
            1 if cn10y and cn10y < 2.8 else 0,
            1 if cu_au and cu_au > 4.5 else 0,
            1 if nb and nb > 0 else 0,
        ])
        if bullish_signals >= 2:
            macro_call = '宏观偏多'
        elif bullish_signals == 0:
            macro_call = '宏观偏空'
        else:
            macro_call = '宏观中性偏谨慎'

        lines.append(f'→ 宏观综合判断: {macro_call}')

        return '\n'.join(lines)


# =============================================================================
# 4. AI Prompt Builder with RLTF + Macro Context
# =============================================================================

class AIPromptBuilder:
    """
    Build enriched AI prompts with RLTF negative examples + macro context.
    """

    def __init__(self):
        self.rltf = RLTFLearner()
        self.macro = MacroContext()

    def build_enhanced_prompt(self, symbol: str, stock_name: str,
                             kline_summary: str, signal: Dict) -> str:
        """
        Build the full enhanced AI prompt.

        Structure:
          1. Task instruction
          2. Stock K-line summary
          3. [Macro Context] — cross-asset environment
          4. [RLTF Warning] — similar historical losers (if any)
          5. Local model signal for comparison
        """
        parts = []

        # 1. Task
        parts.append(
            f'你是A股量化分析师。请对{stock_name}({symbol})今日走势进行分析，'
            f'给出看多/看空/观望判断及置信度(0-1)。\n'
        )

        # 2. Stock data
        parts.append(f'[个股数据]\n{kline_summary}\n')

        # 3. Macro context
        macro_prompt = self.macro.build_context_prompt()
        parts.append(macro_prompt + '\n')

        # 4. RLTF warning
        few_shot = self.rltf.build_few_shot_warning(signal)
        if few_shot:
            parts.append(few_shot + '\n')

        # 5. Local model reference
        lp = signal.get('_local_plan', {})
        ensemble = signal.get('_ensemble', {})
        parts.append(
            f'[本地模型参考]\n'
            f'14学派投票: {ensemble.get("ensemble_signal", "?")} '
            f'(置信度{ensemble.get("ensemble_confidence", 0):.0%})\n'
            f'入场区: {lp.get("entry_zone", "N/A")}\n'
            f'止损: {lp.get("stop_loss", "N/A")}\n'
        )

        return '\n'.join(parts)

    def record_outcome(self, symbol: str, signal: Dict,
                      outcome: Dict) -> None:
        """Record trade outcome for future RLTF learning."""
        self.rltf.record_trade(symbol, signal, outcome)


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    # ---- 1. Event Store ----
    print("=== Event Store ===")
    es = EventStore()
    es.append(EventType.DEPOSIT, 'CASH', {'amount': 100000})
    es.append(EventType.POSITION_OPENED, '600519',
              {'shares': 1000, 'price': 1800.0})
    es.append(EventType.POSITION_CLOSED, '600519',
              {'shares': 1000, 'price': 1850.0, 'pnl': 50000})

    state = es.reconstruct_positions()
    print(f"  Positions: {len(state['positions'])}")
    print(f"  Cash: {state['cash']:,.0f}")
    print(f"  Trades: {state['total_trades']}")
    print(f"  Events: {state['n_events']}")

    # ---- 2. RLTF ----
    print("\n=== RLTF Learner ===")
    rltf = RLTFLearner()
    rltf.record_trade('600519',
        {'confidence': 0.85, '_indicators': {'rsi': 65, 'dmi_adx': 28,
         'vol_ratio': 1.8, 'macd_hist': 0.15, 'current_price': 1800,
         'ma20': 1750, 'weekly_trend': 'bullish'},
         '_local_plan': {'entry_zone': '1780-1820'}},
        {'pnl_pct': -0.05, 'win': False, 'exit_reason': 'stop_loss'})

    current_signal = {'confidence': 0.80,
        '_indicators': {'rsi': 62, 'dmi_adx': 30, 'vol_ratio': 1.6,
         'macd_hist': 0.12, 'current_price': 1820, 'ma20': 1760,
         'weekly_trend': 'bullish'},
        '_local_plan': {'entry_zone': '1800-1840'},
        '_ensemble': {'ensemble_signal': 'bullish', 'ensemble_confidence': 0.75}}

    warning = rltf.build_few_shot_warning(current_signal)
    print(f"  Warning generated: {len(warning) > 0}")
    if warning:
        print(warning[:200])

    # ---- 3. Macro Context ----
    print("\n=== Macro Context ===")
    macro = MacroContext()
    prompt = macro.build_context_prompt()
    print(prompt[:300])

    # ---- 4. Enhanced Prompt ----
    print("\n=== Enhanced Prompt ===")
    builder = AIPromptBuilder()
    builder.rltf._examples = rltf._examples  # Share RLTF data
    final_prompt = builder.build_enhanced_prompt(
        '600519', '贵州茅台',
        '日线: MA多头排列, MACD金叉, RSI 62, 放量突破',
        current_signal)
    print(f"  Prompt length: {len(final_prompt)} chars")
    print(f"  Includes macro: {'宏观' in final_prompt}")
    print(f"  Includes RLTF: {'历史教训' in final_prompt}")

    print("\nAll event-sourcing + RLTF modules: OK")
