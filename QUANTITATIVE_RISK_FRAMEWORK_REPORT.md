# Quantifiable Risk Framework: Extracted Rules from Trading Classics

## Report Summary

**Sources analyzed (5 books):**
- "逃顶十二招" (12 Techniques for Escaping Tops) -- FULLY EXTRACTED
- "顶级交易的三大技巧" (The Three Skills of Top Trading) by Hank Pruden -- FULLY EXTRACTED
- "投资最重要的事" (The Most Important Thing) by Howard Marks -- FULLY EXTRACTED
- "华尔街操盘手日记" (Wall Street Trader's Diary) by 高山 -- FULLY EXTRACTED
- "投资大师也会犯的错" (Mistakes Even Masters Make) -- **UNUSABLE: file contains only advertising spam, zero book content**

**Priority filter:** Every rule below answers "Can we code this as an if-then rule or a numerical threshold?"

---

## SECTION 1: TOP EXIT TECHNIQUES (逃顶十二招)

All 12 techniques follow a uniform **7-Step Analysis Framework**. The exit rule is consistently:
- **Step 6 (Reduce):** Sell 1/2 to 2/3 of position at trigger
- **Step 7 (Exit):** Sell remaining position when price breaks below the confirmation level (typically the low of the signal candle)

### Technique 1: "顶部背离" (Top Divergence)
- **Core Rule:** Price makes higher high, but MACD makes lower high
- **Codable Trigger:** `price_high[1] > price_high[0] AND macd_high[1] < macd_high[0]`
- **Reduce:** On confirmation of divergence, sell 1/2 to 2/3
- **Exit:** When price breaks below the prior swing low
- **Key detail:** Second MACD peak must be lower than first while price second peak is higher

### Technique 2: "两阴防守" (Two-Yin Defense)
- **Core Rule:** When two consecutive bearish candles appear in an uptrend, place stop at the lowest price of those two candles
- **Codable Trigger:** `count_bearish_candles(2) AND trend == UP`
- **Dynamic update:** Each new "two-yin" pair replaces the prior stop level
- **Expiry:** A new stop level is valid for only 5 trading days
- **Additional filter:** Price breaking below 5-day MA strengthens the signal
- **Reduce:** On break of two-yin low, sell 1/2 to 2/3
- **Exit:** Full exit at close if below two-yin low

### Technique 3: "巨量大阴" (Giant Volume Big Yin)
- **Core Rule:** A large bearish candle with abnormally high volume at relative highs
- **Codable Trigger:** `candle_body > avg_body * 1.5 AND volume > avg_volume * 2.0 AND price_at_high_percentile(80)`
- **Additional filter:** When the big-yin appears near the upper Bollinger Band (13-period), probability increases
- **Volume context:** A volume peak ("天量") confirms
- **Reduce:** On close of the big-yin candle, sell 1/2 to 2/3
- **Exit:** Full exit when price breaks below big-yin's low

### Technique 4: "巨量流星" (Giant Volume Shooting Star)
- **Core Rule:** A shooting star candle (small body, long upper shadow) with abnormally high volume at highs
- **Codable Trigger:** `upper_shadow > body * 2 AND lower_shadow < body * 0.5 AND volume > avg_volume * 2.0`
- **Reduce:** Sell 1/2 to 2/3 on confirmation
- **Exit:** Full exit on break below signal candle low

### Technique 5: "闪电霹雳" (Lightning Strike)
- **Core Rule:** Price pattern: decline -> rebound -> decline below prior low (a failed bounce)
- **Codable Trigger:** `recent_low < prior_swing_low AND prior_move == "decline-rebound-decline"`
- **Volume pattern:** Volume shrinks on decline, expands on rebound (distribution), shrinks again on new decline
- **MACD:** Bearish crossover at high levels
- **KD:** Bearish crossover near overbought zone
- **Reduce:** When close < "lightning" lowest price, sell 1/2 to 2/3
- **Exit:** Full exit on continuation below signal low

### Technique 6: "量峰跟进" (Volume Peak Trailing Stop)
- **Core Rule:** After entry, when a day's volume is lower than previous day's volume, set stop at that day's low. Trail this stop upward as volume patterns evolve.
- **Codable Trigger:** `volume[today] < volume[yesterday] THEN stop = low[today]`
- **Trend filter:** Price losing contact with 8-day MA (trend weakening)
- **Reduce:** When price breaks below the trailing volume-peak stop
- **Exit:** Full exit

### Technique 7: "跌势吞没" (Bearish Engulfing)
- **Core Rule:** A large bearish candle fully engulfs the prior bullish candle (open above prior close, close below prior open)
- **Codable Trigger:** `close[today] < open[yesterday] AND open[today] > close[yesterday] AND candle_color == BEARISH`
- **Volume:** Stage-level high volume ("天量") at the engulfing candle
- **Reduce:** On confirmation, sell 1/2 to 2/3
- **Exit:** Full exit when price breaks below engulfing candle low

### Technique 8: "黄昏之星" (Evening Star)
- **Core Rule:** Three-candle pattern: (1) large bullish candle, (2) small-body star (doji or spinning top), (3) large bearish candle. Forms a "品" shape at highs.
- **Codable Trigger:** `three_candle_pattern == "大阳+小星+大阴" AND at_highs`
- **Volume:** Massive volume ("天量") at the star/third candle
- **Bollinger Band:** Pattern appears near upper band (13-period)
- **Reduce:** On third candle confirmation, sell 1/2 to 2/3
- **Exit:** Full exit on break below third candle low

### Technique 9: "垂暮双顶" (Aged Double Top)
- **Core Rule:** M-shaped double top at highs. Differs from standard double top: requires volume + indicator confirmation.
- **Codable Trigger:** `double_top_pattern AND volume_at_both_peaks_elevated`
- **MACD:** Bearish divergence (MACD lower at second peak while price same or higher)
- **KD:** Second peak's KD below 50 midline (first peak's KD above 50) -- strong bearish signal
- **Volume:** Both A-top and B-top show volume spikes
- **Reduce:** When price breaks neckline, sell 1/2 to 2/3
- **Exit:** Full exit

### Technique 10: "半步回转" (Half-Step Reversal / 0.5 Retracement)
- **Core Rule:** After a decline, price retraces to the midpoint (0.5 Fibonacci) of the decline and then reverses
- **Codable Trigger:** `price_retracement == 0.5 * (high - low) + low AND reversal_candle == BEARISH_ENGULFING`
- **120-day MA filter:** If price is below 120-day MA, the 0.5 retracement is more likely a rebound, not a new uptrend
- **KD:** Near overbought with bearish crossover tendency
- **Volume:** Local volume expansion at 0.5 level
- **Reduce:** At 0.5 level with bearish engulfing candle, sell 1/2 to 2/3 at next open
- **Exit:** When close (2 min before market close) < engulfing candle low, sell all

### Technique 11: "断头铡刀" (Decapitation Blade)
- **Core Rule:** A bearish candle whose body cuts through BOTH the 5-day MA and 15-day MA simultaneously, when 5MA is below 15MA
- **Codable Trigger:** `candle_low < MA5 AND candle_low < MA15 AND candle_open > MA5 AND candle_open > MA15 AND MA5 < MA15 AND candle_color == BEARISH`
- **Volume prerequisite:** Elevated volume MUST have appeared BEFORE the signal candle (NOT after)
- **KD:** Around 50 (market transitioning from bullish to neutral)
- **Reduce:** On the next candle open after signal, sell 1/2 to 2/3 (not less than 1/2)
- **Exit:** When close (2 min before close) < decapitation candle low, sell all

### Technique 12: "动量死叉" (Momentum Death Cross)
- **Core Rule:** MACD death cross occurs TWICE within a short period while above the zero line
- **Codable Trigger:** `macd_death_cross_count(recent_period, above_zero=True) >= 2`
- **Volume:** Stage high volume between the two death crosses
- **21-day MA:** Price breaks below 21MA twice in short succession
- **KD:** KD death cross confirms at second MACD death cross
- **Reduce:** On second death cross confirmation, sell 1/2 to 2/3
- **Exit:** Full exit

### Universal Exit Rules (Codable):
```python
# ALL 12 techniques share this exit framework:
REDUCE_RATIO = (0.5, 0.667)  # Sell 1/2 to 2/3 on first signal
EXIT_TIMING = "2_minutes_before_close"  # Final exit timing
EXIT_CONDITION = "close < signal_candle_low"  # Common exit trigger
```

---

## SECTION 2: RISK CONTROL RULES (Howard Marks, 投资最重要的事)

### Foundational Definition
- **Risk = Probability of Permanent Loss** (NOT volatility, NOT beta, NOT temporary drawdown)
- Risk is: subjective (different for each investor), hidden (invisible before it materializes), unquantifiable (cannot be reduced to a single number)
- This is a **philosophical principle**, not a formula -- but it informs all formula design

### The "Perversity of Risk" (风险的反常性)
- Risk is HIGHEST when investors perceive it as lowest (at market tops)
- Risk is LOWEST when investors perceive it as highest (at market bottoms)
- **Codable insight:** Investor sentiment is an inverse indicator of actual risk

### Key Quantifiable Heuristics from Marks:

| Rule | Codable Form |
|------|-------------|
| Price too high = biggest risk source | `price > intrinsic_value_estimate * 1.2 --> RISK_HIGH` |
| Second-level thinking required | Need to model "what is already priced in" |
| Risk cannot be measured, only judged | Use Bayesian posterior-width as confidence interval |
| Bubbles are marked by: no fear, low credit standards, wide leverage | `fear_index < threshold AND margin_debt_high --> BUBBLE_WARNING` |
| Fallen angels (distressed debt): buy when fear is maximum | `credit_spread > 2_std_above_mean AND fear_index > 90th_percentile --> OPPORTUNITY_SCAN` |
| Risk control is different from risk avoidance | Risk control = take calculated risks with defined max loss |

### Marks' Risk Checklist (Codable as a Scoring System):
```python
MARKS_RISK_SCORE = sum([
    1 if market_narrative == "this_time_is_different" else 0,
    1 if fear_index < 20th_percentile else 0,
    1 if margin_debt > historical_90th else 0,
    1 if ipo_volume_spiking else 0,
    1 if junk_bond_yields_near_historic_lows else 0,
    1 if new_investors_surging else 0,
    # Score >= 4 --> HIGH RISK REGIME
])
```

### Position Sizing Philosophy from Marks:
- When others are fearful (risk premium high), INCREASE exposure
- When others are greedy (risk premium low), DECREASE exposure
- The superior investor's hallmark: asymmetric risk-return (participate in upside, protect against downside)
- **Codable:** Scale position size inversely to sentiment extremes

---

## SECTION 3: THREE SKILLS OF TOP TRADING (Hank Pruden, 顶级交易的三大技巧)

### The Three Skills:
1. **Behavioral System Development** (行为系统开发) - Building trading systems grounded in behavioral finance
2. **Pattern Recognition / Wyckoff Method** (模式识别) - Reading supply/demand via price-volume analysis
3. **Mental State Management** (心态管理) - The 10 Tasks model

### The 10 Tasks of Top Trading (Complete):

| Task | Phase | Action | Codable/Checklist? |
|------|-------|--------|-------------------|
| 1 | Pre-trade | Self-assessment: physical/mental state, energy level | Checklist (score 1-5) |
| 2 | Pre-trade | Mental rehearsal: visualize the trade process | Checklist |
| 3 | Pre-trade | Develop low-risk idea / plan | Systematic |
| 4 | Entry | Execute entry per plan | Rule-based |
| 5 | Monitoring | Monitor position, adjust stops | Systematic |
| 6 | Monitoring | Add to position per plan | Rule-based |
| 7 | Monitoring | Move stop to breakeven/protective | Systematic |
| 8 | Monitoring | Trail stop per methodology | Rule-based |
| 9 | Exit | Exit per plan (profit or stop) | Rule-based |
| 10 | Post-trade | Self-review: what went right/wrong | Checklist |

### The 10 Rules for Making a Living (谋生的10个规则):

1. **Be Your Own Master (自己做主):** Create a "密室" (private room/chamber). Isolate from news, tips, social media. Trade YOUR plan, not others' opinions.
2. **Surpass the Competition (超越对手):** Master 21st-century concepts: behavioral finance, right-brain pattern recognition, whole-brain trading
3. **Learn the Wyckoff Method (学习威科夫方法):** Study supply/demand via price-volume on bar charts and point-and-figure charts
4. **Become the Composite Operator (成为综合人):** Think like the market manipulator -- accumulate in fear, distribute in greed
5. **Maintain Discipline Forever (永远保持自律):** Execute the 10 Tasks on every trade
6. **Build Your System on Behavioral Finance (基于行为金融学开发系统):** Use Mass Behavior Lifecycle Model (S-curve + bell-curve of price, volume, sentiment, time)
7. **Use Decision Support Systems (使用决策支持系统):** Lifecycle model as decision aid -- compare price, volume, time, sentiment
8. **Test Your System via Action Sequence (用行动顺序测试系统):** Walk-forward simulation, uncovering 5-10 bars at a time
9. **Plan and Control the Process (计划和控制过程):** Define targets -> measure performance -> identify gaps -> adjust. Control loop: Analysis -> Plan -> Application -> Control
10. **Fuse All Three Skills (融合三大技巧):** Behavioral systems + Wyckoff patterns + mental state management as one unified practice

### Codable Elements from the 10 Rules:

```python
# Rule 5 (Discipline): Track 10-Task completion rate
task_completion_rate = completed_tasks / 10  # Target: 100%

# Rule 8 (Testing): Action sequence walk-forward
# Simulate by revealing N bars at a time (5-10 for daily, 1 bar for weekly)
TEST_BAR_COUNT = 5  # for daily charts

# Rule 9 (Control): Performance benchmark check every 2 months
PERFORMANCE_REVIEW_INTERVAL = "60_days"
MIN_RISK_REWARD_RATIO = 3.0

# Rule 7 (Lifecycle model): Classify market phase
# Phase detection via S-curve fit on price, volume, sentiment
```

### Wyckoff Method -- 5 Steps (Codable):
1. **Determine market position (trend):** 200-day MA for long-term trend
2. **Determine relative strength:** Is stock stronger/weaker than market?
3. **Find cause for entry:** Point-and-figure count for price target
4. **Determine readiness to move:** Volume + price action (springs, upthrusts, tests)
5. **Time the entry:** Execute at specific Wyckoff test points

### Wyckoff 9 Buying/Selling Test Points:
- BUY: Secondary test (higher low), spring, shakeout, backup to creek, breakout pullback
- SELL: Secondary test (lower high), upthrust, rally to ice, breakdown rally
- **Codable:** Each test point has a specific price-volume signature

### Key Numerics from Pruden:
- 200-day MA = primary trend indicator
- 34-period MA = intermediate trend divergence
- 5% price moves = minimum for "intermediate swing" capture
- Risk management = 80% of trading success
- Risk per trade max: 5% of total capital (from Canada's case study)
- Stop-loss distance: 100-200 basis points for forex
- Price targets via point-and-figure horizontal counts (cause = effect principle)

---

## SECTION 4: COMMON MISTAKES (投资大师也会犯的错)

**STATUS: BOOK UNUSABLE.** The extracted file at `.book_extracts/投资大师也会犯的错 教你在投资的过程中少犯错误(高清).txt` contains only 889 lines of advertising spam (URLs, phone numbers). Zero book content was recovered.

### Behavioral Biases Documented in the Diary (as proxy for "mistakes"):

| Bias | Description | Codable Mitigation |
|------|------------|-------------------|
| Overconfidence (过度自信) | Traders overestimate their accuracy | Track calibration: `actual_win_rate / self_assessed_win_rate` |
| Anchoring (锚定效应) | Fixating on entry price, not current market | Force stop-loss based on technical level, not entry price |
| Loss Aversion (损失回避) | Holding losers too long, selling winners too early | Enforce mechanical trailing stops; reward/risk >= 3 before entry |
| Confirmation Bias (证实偏差) | Seeking evidence that confirms existing position | Require disconfirming evidence checklist before each trade |
| Availability Bias (易得性偏差) | Over-weighting recent/easily-recalled information | Use systematic multi-factor checklist, not "gut feeling" |
| Hindsight Bias (后见之明) | "I knew it all along" after the fact | Keep pre-trade journal with explicit prediction; compare to outcome |
| Disposition Effect (处置效应) | Selling winners to lock in gains, holding losers hoping for recovery | Mechanical rule: cut losers at -2R, let winners run to +3R minimum |
| Self-Attribution Bias (自我归因) | Attributing wins to skill, losses to bad luck | Force post-trade review attributing cause: system/skill/luck for BOTH wins and losses |

---

## SECTION 5: PRACTICAL TRADER PSYCHOLOGY (华尔街操盘手日记)

### SpeedTrade Risk Control System (Directly Codable):

| Parameter | New Trainee | Level 2 | Level 3 | Level 4 | Senior |
|-----------|------------|---------|---------|---------|--------|
| Capital | $10,000 | $20,000 | $100,000 | $200,000 | $500,000+ |
| Max shares | 100 | 200 | 500-1000 | 2000-3000 | 5000+ |
| Max daily loss | $30 | $50 | $150 | Variable | Variable |
| Overnight positions | PROHIBITED | PROHIBITED | PROHIBITED | PROHIBITED | PROHIBITED |
| Training threshold | N/A | Stable profit required | Month profit $2000 | Sustained profit | Sustained profit |

### Position Sizing Progression Rule:
```python
def speedtrade_position_scale(daily_pnl_avg: float, win_rate: float, days_consistent: int):
    """
    Scale position size based on proven profitability, not account size.
    Key: You don't get more size until you PROVE you can handle current size.
    """
    if days_consistent < 60: return 100  # shares
    if daily_pnl_avg < 0: return 100  # no upgrade while losing
    if win_rate < 0.5: return 100  # must be above 50% win rate

    if daily_pnl_avg > 30 and days_consistent > 90: return 200
    if daily_pnl_avg > 50 and days_consistent > 120: return 500
    if daily_pnl_avg > 100 and days_consistent > 150: return 1000
    if daily_pnl_avg > 200 and days_consistent > 180: return 2000
    return 100  # default: stay small
```

### Single-Day Max Loss Formula:
```
MAX_DAILY_LOSS = AVERAGE_DAILY_PROFIT
```
Example: If a trader averages $1000/day profit -> max daily loss allowed = $1000.
If hit, account is FORCE-LIQUIDATED and LOCKED for the rest of the day.

**Codable:**
```python
MAX_DAILY_LOSS = max(rolling_average_daily_pnl(20), BASE_MAX_LOSS)
if daily_pnl <= -MAX_DAILY_LOSS:
    liquidate_all()
    lock_account_for_day()
```

### The Two-Exit System (分仓进出):
From both 逃顶十二招 and the diary:
```python
# First exit: Reduce risk, take partial profit / cut partial loss
if exit_signal_1:
    close_position(ratio=0.5)  # Always sell at least 1/2

# Second exit: Clean break
if exit_signal_2:
    close_position(ratio=1.0)  # Sell remainder
```

### Psychological Capital (心理资本) Framework:
4 components, all measurable:
1. **Confidence (自信):** `confidence_score = self_rated_ability_to_execute_plan(1-10)`
2. **Hope (希望):** `hope_score = expectation_of_achieving_long_term_goal(1-10)`
3. **Optimism (乐观):** `optimism_score = positive_outlook_on_future_trades(1-10)`
4. **Resilience (韧性):** `resilience_score = ability_to_recover_from_losses(1-10)`

```python
PSYCHOLOGICAL_CAPITAL = (confidence + hope + optimism + resilience) / 4

# Critical rule: Psychological capital amplifies trading results
if PSYCHOLOGICAL_CAPITAL < 4.0:
    REDUCE_POSITION_SIZE_BY = 0.5  # Half normal size
if PSYCHOLOGICAL_CAPITAL < 2.0:
    STOP_TRADING  # Take mandatory break
```

**Key insight from the diary:** Psychological capital can go NEGATIVE. When negative, the trader will inevitably lose money regardless of strategy quality.

### Stop-Loss Rules (Codable):
1. **Absolute rule:** No position EVER beyond stop-loss -- this is the #1 survival rule
2. **Stop-loss violation escalation:**
   - 1st offense: Written review + 1-day trading suspension
   - 2nd offense: 2-day suspension
   - 3rd offense: 3-day suspension (FINAL WARNING)
   - 4th offense: TERMINATED
3. **Mindset reframe:** "Being able to stop-loss" = PLEASURE; "Not being able to stop-loss" = PAIN
4. **Never add to a losing position to "average down"** (摊低成本 is forbidden)

### The "Don't Double Down to Recover" Rule:
```python
if current_drawdown > 2 * average_daily_range:
    POSITION_SIZE = min(POSITION_SIZE, BASE_SIZE * 0.5)
    # Never increase size to "make back" losses
```

### Focus of Attention (注意的焦点) Principle:
The single most important mental technique. Ask these questions before each trade:
1. "Where am I focusing my attention right now?"
2. "Should I be focusing it elsewhere?"
3. "What is the highest-quality question I can ask about this market right now?"

---

## SECTION 6: POSITION SIZING FORMULAS

### Formula 1: SpeedTrade Tiers (from Section 5)
Proportional to PROVEN profit, not account equity.

### Formula 2: Pruden/Wyckoff Maximum Risk
```
MAX_RISK_PER_TRADE = 5% of total capital
STOP_LOSS_DISTANCE = 100-200 basis points (forex) or ATR-based
POSITION_SIZE = (CAPITAL * 0.05) / STOP_LOSS_DISTANCE
```

### Formula 3: The "Middle Way" (中庸之道) Position Sizing
From the diary: Balance between aggression and protection.
```python
def middle_way_position(account_equity: float, atr: float, confidence: float):
    """
    Neither over-aggressive (gambling) nor over-conservative (fearful).
    Confidence ranges from 0-10.
    """
    base_risk = 0.02  # 2% of equity per trade (middle ground)
    adjustment = (confidence - 5) * 0.002  # +/- 1% adjustment
    risk_pct = clamp(base_risk + adjustment, 0.01, 0.04)  # Cap at 1%-4%
    return (account_equity * risk_pct) / atr
```

### Formula 4: Two-Part Exit Sizing
```python
FIRST_EXIT_RATIO = random.uniform(0.5, 0.667)  # Context-dependent
REMAINING_POSITION = 1.0 - FIRST_EXIT_RATIO
# Second exit triggers at confirmation level
```

### Formula 5: Stress-Adaptive Sizing
From diary entries about losing streaks:
```python
if consecutive_losses >= 3:
    position_multiplier = 0.5  # Cut size in half
elif consecutive_losses >= 5:
    position_multiplier = 0.25  # Quarter size
    require_mandatory_break()  # Step away
elif consecutive_wins >= 5:
    position_multiplier = min(position_multiplier * 1.2, 1.5)  # Gradual scale-up
```

### Formula 6: Wyckoff Cause-Effect Price Targeting
```python
# Point-and-figure horizontal count
price_target = breakout_level + (horizontal_count * box_size * reversal)
# Only enter if:
if potential_reward / potential_risk >= 3.0:
    EXECUTE_TRADE
```

---

## SECTION 7: DRAWDOWN MANAGEMENT

### Rule 1: SpeedTrade Hard Daily Limit
```python
MAX_DAILY_LOSS = rolling_avg_daily_profit(30)
if intraday_drawdown >= MAX_DAILY_LOSS:
    force_liquidate_all_positions()
    lock_account_for_remainder_of_day()
```

### Rule 2: Losing Streak Protocol (from diary, multiple entries)
```python
if daily_losses_consecutive >= 3:
    # Next day: reduce position size by 50%
    position_size *= 0.5

if daily_losses_consecutive >= 5:
    # Mandatory 1-day break, then return at 25% size
    mandatory_break(1)
    position_size = base_size * 0.25
```

### Rule 3: Psychological Capital Drawdown (from diary)
When psychological capital drops, the trader MUST reduce size or stop:
```python
# Assess at start of each day
psych_capital_indicators = {
    "slept_well": bool,
    "not_stressed": bool,
    "confident_in_method": bool,
    "no_recent_trauma": bool,  # Major loss in past 3 days
    "can_accept_loss": bool,
}
psych_score = sum(psych_capital_indicators.values()) / len(psych_capital_indicators)

if psych_score < 0.5:
    MAX_SIZE_TODAY = 0  # Do not trade
elif psych_score < 0.75:
    MAX_SIZE_TODAY = base_size * 0.5
```

### Rule 4: "Start Fresh Every Day" (from SpeedTrade)
```python
# No overnight positions means:
# Every day is a clean slate
# Yesterday's losses don't reduce today's fighting capital
# Reset emotional state at market close
at_market_close:
    liquidate_all()
    reset_daily_pnl()
    # Tomorrow starts at zero
```

### Rule 5: The "Stable Environment" Requirement
From diary -- 4 requirements before any trading:
1. Clarity (清晰的判断力)
2. Stop-loss discipline (坚决止损)
3. Appropriate position size (合适的仓位)
4. Stable emotional environment (稳定的交易环境)

```python
def can_trade_today():
    checklist = [
        assess_mental_clarity() > 6/10,
        verify_recent_stop_discipline(10) > 0.95,  # 95% stop compliance in last 10 trades
        position_size_within_limits(),
        stress_level < 5/10,
    ]
    return all(checklist)
```

### Rule 6: Reduce, Not Eliminate, on Adversity
```python
# Key principle: "用小的仓位去适应市场的变化，然后再恢复正常仓位"
# "Use small positions to adapt to market changes, then resume normal size"
if struggling:
    position_size = base_size * 0.25
    # Trade small until you feel in sync with market again
    # Then gradually scale back up
    # NEVER: use large positions to "make it back quickly"
```

---

## SECTION 8: MARKET CYCLE RISK

### 8.1: Mass Behavior Lifecycle Model (Pruden)

The market moves in an S-curve and sentiment follows a bell curve:
```
Phase 1: Accumulation (Smart Money buying, pessimism/fear)
    -> Risk: LOW (price near value), Sentiment: FEAR
Phase 2: Markup (Trend followers enter, optimism builds)
    -> Risk: MODERATE, Sentiment: RISING OPTIMISM
Phase 3: Distribution (Smart Money selling, euphoria)
    -> Risk: HIGH (price far above value), Sentiment: GREED
Phase 4: Markdown (Panic selling, despair)
    -> Risk: DECLINING (price approaching value), Sentiment: PANIC
```

**Codable Phase Detection:**
```python
def detect_cycle_phase(price, volume, sentiment, time):
    """
    Use S-curve fitting on normalized price.
    Use bell-curve on sentiment indicators.
    """
    # Price position in 2-year range
    price_percentile = percentile(price, lookback=504)  # ~2 years trading days

    # Volume trend
    volume_trend = slope(volume_ma(50), periods=20)

    # Sentiment indicators
    fear_greed = fear_greed_index()
    put_call = put_call_ratio()

    if price_percentile < 30 and volume_trend < 0 and fear_greed < 30:
        return "ACCUMULATION"  # Low risk phase -- MAXIMUM position size
    elif price_percentile < 60 and volume_trend > 0 and 30 < fear_greed < 70:
        return "MARKUP"  # Moderate risk -- FULL position size
    elif price_percentile > 70 and volume_trend > 0 and fear_greed > 70:
        return "DISTRIBUTION"  # High risk -- REDUCED position size
    elif price_percentile > 50 and volume_trend < 0 and fear_greed < 50:
        return "MARKDOWN"  # Declining risk -- CASH / SHORT only
```

### 8.2: Howard Marks' Cycle Risk Adjustment

Core principle: **Risk is highest when perceived risk is lowest.**

```python
# Marks' cycle risk adjustment:
PERCEIVED_RISK = fear_index  # VIX or similar
ACTUAL_RISK = 100 - PERCEIVED_RISK  # Inverse relationship

POSITION_MULTIPLIER = ACTUAL_RISK / 50  # Normalized:
# When fear = 20 (everyone calm): actual_risk = 80, multiplier = 1.6 (AGGRESSIVE)
# When fear = 80 (everyone panicked): actual_risk = 20, multiplier = 0.4 (DEFENSIVE)
# Wait -- this is inverted. Let me restate:

# CORRECT interpretation:
# When perceived risk is LOW (complacency) -> ACTUAL risk is HIGH -> REDUCE size
# When perceived risk is HIGH (fear) -> ACTUAL risk is LOW -> INCREASE size

position_scale = 1.0
if fear_index < 20:  # Extreme complacency
    position_scale = 0.5
    tighten_stops = True
elif fear_index > 60:  # Elevated fear
    position_scale = 1.5  # Scale up carefully
    widen_stops_slightly = True  # Allow for volatility
```

### 8.3: News/Sentiment as Contrarian Signal (Pruden + Marks)

```python
# When CNBC/财经新闻 starts talking about a trend -> trend is mature
# "傻钱" (dumb money) enters late in the cycle

def media_contrarian_signal(topic_frequency: int, sentiment_score: float):
    """
    If a market narrative appears everywhere, it's likely priced in.
    The more media coverage, the closer to the end of the cycle.
    """
    if topic_frequency > 2_std_above_historical:
        # Extreme media attention -> distribution phase
        return "CONTRARIAN_SELL"
    elif topic_frequency < 2_std_below_historical:
        # No one talking about it -> accumulation phase
        return "CONTRARIAN_BUY"
```

### 8.4: Cycle-Aware Position Sizing Matrix

```python
CYCLE_POSITION_MATRIX = {
    ("ACCUMULATION", "LONG"):  1.50,  # Max long, smart money is buying
    ("ACCUMULATION", "SHORT"): 0.00,  # Don't short at bottom

    ("MARKUP", "LONG"):        1.00,  # Full position, trend following
    ("MARKUP", "SHORT"):       0.25,  # Counter-trend only with tight stops

    ("DISTRIBUTION", "LONG"):  0.50,  # Reduced long, take profits
    ("DISTRIBUTION", "SHORT"): 0.75,  # Start scaling short

    ("MARKDOWN", "LONG"):      0.00,  # No long positions in markdown
    ("MARKDOWN", "SHORT"):     1.00,  # Full short, trend following

    ("UNKNOWN", "ANY"):        0.50,  # Unclear phase -> half size
}
```

### 8.5: Composite Risk Score (Integrating All Sources)

```python
def calculate_composite_risk_score():
    """
    0-10 scale. Higher = riskier = smaller positions.
    Integrates: cycle phase, sentiment, technicals, fundamentals.
    """
    score = 0

    # 1. Cycle phase (Pruden/Marks) -- 0 to 3 points
    phase = detect_cycle_phase()
    if phase == "DISTRIBUTION": score += 3
    elif phase == "MARKDOWN": score += 1  # Risk declining
    elif phase == "MARKUP": score += 0

    # 2. Sentiment extremes (Marks) -- 0 to 2 points
    if fear_greed_index > 80: score += 2  # Extreme greed
    elif fear_greed_index < 20: score -= 1  # Extreme fear = opportunity

    # 3. Technical warning signals (逃顶十二招) -- 0 to 3 points
    active_top_signals = count_active_exit_signals()  # How many of 12 active
    if active_top_signals >= 2: score += 2
    if active_top_signals >= 4: score += 1  # Additional point

    # 4. Psychological capital (日记) -- 0 to 2 points
    psych = get_psychological_capital_score()
    if psych < 4: score += 2
    elif psych < 6: score += 1

    # Result:
    # score 0-2: LOW risk -> use 100% position size
    # score 3-5: MODERATE risk -> use 75% position size
    # score 6-7: HIGH risk -> use 50% position size
    # score 8-10: EXTREME risk -> use 25% or less

    return score

def position_size_from_risk_score(risk_score, base_size):
    if risk_score <= 2: return base_size * 1.0
    elif risk_score <= 5: return base_size * 0.75
    elif risk_score <= 7: return base_size * 0.5
    else: return base_size * 0.25
```

---

## APPENDIX: Complete Codable Rule Inventory

### Priority 1 -- Hard Stops (Must Implement):
1. **Max daily loss = 1x average daily profit** (SpeedTrade)
2. **Single trade risk = max 5% of capital** (Pruden/Wyckoff)
3. **12 exit patterns** with reduce/exit logic (逃顶十二招)
4. **No doubling down on losers** (Diary)
5. **Stop-loss compliance tracking with escalation** (Diary)

### Priority 2 -- Position Sizing:
6. **Tiered capital allocation** based on proven performance (SpeedTrade)
7. **Cycle-adjusted position sizing** (Section 8.4)
8. **Stress-adaptive reduction** on 3+ consecutive losses (Diary)
9. **Psychological capital gating** (Diary)

### Priority 3 -- Risk Regime Detection:
10. **Composite risk score** (Section 8.5)
11. **Cycle phase classification** via S-curve + sentiment (Pruden)
12. **Sentiment contrarian indicator** (Marks + Pruden)
13. **Media attention tracking** for cycle positioning (Pruden/Canada)

### Priority 4 -- Process Control:
14. **10-Task checklist completion tracking** (Pruden)
15. **Bi-monthly performance audit with risk-reward >= 3 check** (Pruden)
16. **Pre-trade psych capital assessment** (Diary)
17. **Post-trade attribution journal** (skill vs. luck, Diary)

---

## Sources:
- "逃顶十二招" -- 亚柏专业理财机构
- "顶级交易的三大技巧" -- Hank Pruden (汉克·普鲁登)
- "投资最重要的事" -- Howard Marks (霍华德·马克斯)
- "华尔街操盘手日记(第2版)" -- 高山
- "投资大师也会犯的错" -- UNUSABLE (file corrupted/advertising only)
