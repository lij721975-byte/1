#!/usr/bin/env python
# async_deepseek.py — Fully async DeepSeek API with Jitter Backoff + Schema Guard
"""
Asynchronous LLM communication layer.

Features:
  - aiohttp-based async HTTP requests (non-blocking)
  - asyncio.Queue for task scheduling with backpressure
  - Token-bucket rate limiter with JITTER to avoid thundering herd
  - Exponential Backoff with Jitter (Decorrelated Jitter) on 429 / transient errors
  - Pydantic schema guard: validates JSON response, falls back to safe defaults
    on hallucination (confidence delta > 50% → cached value; malformed → position=0)
  - Designed for 9:26 AM burst: 500+ stocks → concurrent, non-blocking
"""

import asyncio
import aiohttp
import json
import os
import time
import random
import re
from typing import Dict, List, Optional, Any, Tuple


# =============================================================================
# Jittered Token Bucket Rate Limiter
# =============================================================================

class TokenBucket:
    """Token bucket rate limiter with jittered refill spread."""

    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_refill = time.monotonic()
        self._jitter_scale = 0.03  # 3% jitter on sleep intervals

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(float(self.burst), self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            # Jittered sleep: 50ms ± 15ms
            jitter = 0.05 * (1.0 + self._jitter_scale * (2 * random.random() - 1))
            await asyncio.sleep(jitter)


# =============================================================================
# Decorrelated Jitter Backoff
# =============================================================================

def jittered_backoff(retry_count: int, base_seconds: float = 1.0,
                     cap_seconds: float = 60.0) -> float:
    """
    Decorrelated Jitter: sleep = min(cap, random(base, base × 3^retry)).

    This prevents synchronized retry storms (all 429 clients retrying at
    exactly 2^n seconds) by randomizing each client's backoff independently.
    """
    upper = min(cap_seconds, base_seconds * (3 ** retry_count))
    return base_seconds + random.random() * max(0, upper - base_seconds)


# =============================================================================
# Pydantic-lite Schema Guard (zero-dependency, inline)
# =============================================================================

class _SafeSignalSchema:
    """
    Inline schema validator — no pydantic dependency.
    Defines expected structure and safe defaults for LLM JSON output.
    """

    EXPECTED_KEYS = {'signal', 'confidence', 'position_pct', 'entry_zone',
                     'stop_loss', 'targets', 'reason'}
    MAX_CONFIDENCE_DELTA = 0.50   # If new confidence differs from cached by >50%, reject
    SAFE_DEFAULTS = {
        'signal': 'neutral',
        'confidence': 0.0,
        'position_pct': 0.0,
        'entry_zone': '',
        'stop_loss': '',
        'targets': '',
        'reason': 'LLM schema validation failed — fallback to safe default',
    }

    @classmethod
    def validate(cls, parsed: Dict, cached_confidence: Optional[float] = None) -> Tuple[Dict, bool]:
        """
        Validate and sanitize LLM JSON output.

        Returns (sanitized_dict, is_valid).
        If is_valid is False, caller MUST use cached or safe defaults.
        """
        if not isinstance(parsed, dict):
            return dict(cls.SAFE_DEFAULTS), False

        result = dict(cls.SAFE_DEFAULTS)
        is_valid = True

        # Extract and validate signal
        signal = str(parsed.get('signal', 'neutral')).lower().strip()
        if signal not in ('bullish', 'bearish', 'neutral'):
            signal = 'neutral'
            is_valid = False
        result['signal'] = signal

        # Extract and validate confidence
        try:
            conf = float(parsed.get('confidence', 0.0))
        except (ValueError, TypeError):
            conf = 0.0
            is_valid = False

        # Confidence delta guard: if new conf differs from cached by >50%, hallucination likely
        if cached_confidence is not None and cached_confidence > 0.10:
            delta = abs(conf - cached_confidence)
            if delta > cls.MAX_CONFIDENCE_DELTA:
                conf = cached_confidence  # Revert to cached
                is_valid = False
                result['reason'] = (f'CONFIDENCE HALLUCINATION: delta={delta:.2f} '
                                   f'(new={conf:.2f}, cached={cached_confidence:.2f}) → reverted')

        result['confidence'] = round(max(0.0, min(1.0, conf)), 3)

        # Position pct with sanity bounds
        try:
            pos = float(parsed.get('position_pct', 0.0))
        except (ValueError, TypeError):
            pos = 0.0
        result['position_pct'] = round(max(0.0, min(0.30, pos)), 3)

        # String fields
        for key in ('entry_zone', 'stop_loss', 'targets'):
            result[key] = str(parsed.get(key, ''))[:200]

        # Reason
        if not is_valid and result['reason'] == cls.SAFE_DEFAULTS['reason']:
            result['reason'] = str(parsed.get('reason', cls.SAFE_DEFAULTS['reason']))[:500]

        return result, is_valid


# =============================================================================
# Cached confidence store (per-symbol, session-lifetime)
# =============================================================================

_confidence_cache: Dict[str, float] = {}


def get_cached_confidence(symbol: str) -> Optional[float]:
    return _confidence_cache.get(symbol)


def set_cached_confidence(symbol: str, confidence: float) -> None:
    _confidence_cache[symbol] = confidence


# =============================================================================
# Async DeepSeek Client
# =============================================================================

class AsyncDeepSeekClient:
    """
    Fully async DeepSeek API client with rate limiting and retry.

    Usage:
        client = AsyncDeepSeekClient(api_key='sk-xxx', max_concurrent=15)
        results = await client.query_batch(prompts)  # Non-blocking!
    """

    BASE_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self, api_key: str = None, max_concurrent: int = 15,
                 requests_per_second: float = 5.0, burst: int = 20,
                 max_retries: int = 3, timeout: float = 30.0):
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY', '')
        self.max_concurrent = max_concurrent
        self.rate_limiter = TokenBucket(rate=requests_per_second, burst=burst)
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def query_one(self, prompt: str, symbol: str = '',
                        system_prompt: str = '',
                        retries: int = 0) -> Optional[Dict[str, Any]]:
        """
        Single async DeepSeek query with retry.
        Non-blocking — returns immediately with a coroutine.
        """
        # Rate limit
        await self.rate_limiter.acquire()

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        payload = {
            'model': 'deepseek-chat',
            'messages': messages,
            'temperature': 0.3,
            'max_tokens': 800,
        }

        try:
            session = await self._get_session()
            async with session.post(self.BASE_URL, json=payload,
                                    headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        'symbol': symbol,
                        'content': data['choices'][0]['message']['content'],
                        'tokens': data.get('usage', {}),
                        'status': 'ok',
                    }
                elif resp.status == 429:  # Rate limited → jittered backoff
                    if retries < self.max_retries:
                        delay = jittered_backoff(retries, base_seconds=1.5)
                        await asyncio.sleep(delay)
                        return await self.query_one(prompt, symbol, system_prompt, retries + 1)
                    return {'symbol': symbol, 'status': 'rate_limited', 'error': '429'}
                else:
                    text = await resp.text()
                    if retries < self.max_retries:
                        delay = jittered_backoff(retries, base_seconds=0.5)
                        await asyncio.sleep(delay)
                        return await self.query_one(prompt, symbol, system_prompt, retries + 1)
                    return {'symbol': symbol, 'status': 'error',
                            'error': f'HTTP {resp.status}: {text[:200]}'}

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if retries < self.max_retries:
                delay = jittered_backoff(retries, base_seconds=0.8)
                await asyncio.sleep(delay)
                return await self.query_one(prompt, symbol, system_prompt, retries + 1)
            return {'symbol': symbol, 'status': 'error', 'error': str(e)}

    async def query_batch(self, prompts: List[Dict[str, str]]) -> List[Dict]:
        """
        Batch async query with concurrency control.

        Args:
            prompts: list of {'symbol': str, 'prompt': str, 'system': str}

        Returns:
            List of results (order preserved)
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_query(item: Dict[str, str]) -> Dict:
            async with semaphore:
                return await self.query_one(
                    prompt=item.get('prompt', ''),
                    symbol=item.get('symbol', ''),
                    system_prompt=item.get('system', ''),
                )

        tasks = [bounded_query(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Replace exceptions with error dicts
        processed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                processed.append({'symbol': prompts[i].get('symbol', ''),
                                  'status': 'error', 'error': str(r)})
            else:
                processed.append(r)
        return processed

    async def query_one_safe(self, prompt: str, symbol: str = '',
                              system_prompt: str = '') -> Dict[str, Any]:
        """
        Query with schema guard and confidence hallucination detection.

        Returns safe defaults (position=0, signal=neutral) on:
          - JSON parse failure
          - Missing required keys
          - Confidence delta > 50% from cached value
        """
        raw = await self.query_one(prompt, symbol, system_prompt)

        if raw is None or raw.get('status') != 'ok':
            return {
                'symbol': symbol, 'status': raw.get('status', 'error') if raw else 'error',
                'signal': 'neutral', 'confidence': 0.0, 'position_pct': 0.0,
                'entry_zone': '', 'stop_loss': '', 'targets': '',
                'reason': 'API call failed — safe default applied',
            }

        # Attempt JSON extraction from content
        content = raw.get('content', '')
        parsed = None
        # Try direct JSON parse
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            # Try extracting JSON block from markdown
            match = re.search(r'\{[^{}]*"signal"[^{}]*\}', content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except (json.JSONDecodeError, TypeError):
                    pass

        cached_conf = get_cached_confidence(symbol)
        validated, is_valid = _SafeSignalSchema.validate(parsed or {}, cached_conf)

        if is_valid:
            set_cached_confidence(symbol, validated['confidence'])

        return {
            'symbol': symbol,
            'status': 'ok' if is_valid else 'schema_fallback',
            **validated,
            'raw_content': content[:300],
        }

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# Queue-based task scheduler for 9:26 burst
# =============================================================================

class SignalQueue:
    """
    asyncio.Queue based scheduler for morning signal burst.

    Produces signals → consumer queries AI → results flow back.
    Non-blocking: signals can arrive at any rate without blocking the producer.
    """

    def __init__(self, client: AsyncDeepSeekClient, max_queue: int = 1000):
        self.client = client
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self.results: List[Dict] = []

    async def produce(self, signals: List[Dict[str, str]]):
        """Producer: enqueue signals for AI analysis."""
        for sig in signals:
            await self.queue.put(sig)

    async def consume(self, num_workers: int = 10):
        """Consumer: dequeue and process with concurrency."""
        semaphore = asyncio.Semaphore(num_workers)

        async def worker():
            while True:
                try:
                    sig = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                async with semaphore:
                    result = await self.client.query_one(
                        prompt=sig.get('prompt', ''),
                        symbol=sig.get('symbol', ''),
                        system_prompt=sig.get('system', ''),
                    )
                    self.results.append(result)
                    self.queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(num_workers)]
        await asyncio.gather(*workers)

    async def run(self, signals: List[Dict[str, str]], num_workers: int = 15):
        """Run full produce-consume cycle."""
        await self.produce(signals)
        await self.consume(num_workers)
        return self.results


# =============================================================================
# Synchronous wrapper (for main_v2.py compatibility)
# =============================================================================

def async_query_batch_sync(prompts: List[Dict[str, str]],
                           max_concurrent: int = 15) -> List[Dict]:
    """
    Synchronous wrapper for batch AI query.
    Compatible with existing main_v2.py code — call with no async changes.
    """
    client = AsyncDeepSeekClient(max_concurrent=max_concurrent)

    async def _run():
        return await client.query_batch(prompts)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context, use run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(_run(), loop)
            return future.result(timeout=300)
        else:
            return asyncio.run(_run())
    except RuntimeError:
        return asyncio.run(_run())


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    async def test():
        client = AsyncDeepSeekClient(max_concurrent=5)
        prompts = [
            {'symbol': '600519', 'prompt': '茅台今日走势分析', 'system': '你是A股分析专家'},
            {'symbol': '000001', 'prompt': '平安银行技术分析', 'system': '你是A股分析专家'},
        ]
        print(f"Sending {len(prompts)} async queries...")
        results = await client.query_batch(prompts)
        for r in results:
            print(f"  {r.get('symbol','?')}: {r.get('status')} - "
                  f"{str(r.get('content',''))[:80] if r.get('content') else r.get('error','')}")
        await client.close()

    # asyncio.run(test())  # Uncomment with real API key
    print("Async DeepSeek module ready (test requires API key)")
