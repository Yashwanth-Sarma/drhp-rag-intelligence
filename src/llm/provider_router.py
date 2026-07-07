"""
src/llm/provider_router.py

Smart multi-provider LLM router for FinSight.

Key design decisions:
- Two failure states: rate_limited (temporary, auto-recovers) and quota_exhausted (daily, permanent)
- rate_limited clears automatically after RATE_LIMIT_COOLDOWN_SECONDS
- quota_exhausted never clears within the same run
- OpenAI SDK auto-retries disabled (max_retries=0) — router owns all retry logic
- Provider-specific retry delays extracted from error messages
- Gemini base_url has no trailing slash (prevents double-slash in some SDK versions)

Provider free tier limits:
    Cerebras gemma3-27b:  ~6 RPM, 1M tokens/day
    Groq llama-3.3-70b:   30 RPM, 100K tokens/day
    Gemini Flash:         10 RPM, 1M context, 1500 req/day
    Gemini Pro:            2 RPM, 25 req/day
"""

import os
import re
import time
import logging
from enum import Enum
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RATE_LIMIT_COOLDOWN_SECONDS = 60


class TaskType(Enum):
    ROUTING = "routing"
    ENRICHMENT = "enrichment"
    ENTITY_EXTRACTION = "entity_extraction"
    GAP_DETECTION = "gap_detection"
    COMPRESSION = "compression"
    REASONING = "reasoning"
    VERIFICATION = "verification"
    FALLBACK = "fallback"


class ProviderStatus:
    """Tracks status and cooldown timestamp for one provider."""

    def __init__(self) -> None:
        self.state = "ok"
        self.rate_limited_at: Optional[float] = None

    def mark_rate_limited(self) -> None:
        self.state = "rate_limited"
        self.rate_limited_at = time.time()

    def mark_quota_exhausted(self) -> None:
        """Daily quota hit — do not retry for remainder of run."""
        self.state = "quota_exhausted"
        self.rate_limited_at = None

    def mark_ok(self) -> None:
        self.state = "ok"
        self.rate_limited_at = None

    def is_available(self) -> bool:
        if self.state == "ok":
            return True
        if self.state in ("unavailable", "quota_exhausted"):
            return False
        if self.state == "rate_limited":
            if self.rate_limited_at is None:
                return True
            elapsed = time.time() - self.rate_limited_at
            if elapsed >= RATE_LIMIT_COOLDOWN_SECONDS:
                self.mark_ok()
                return True
            return False
        return False

    def seconds_until_available(self) -> float:
        if self.state != "rate_limited" or self.rate_limited_at is None:
            return 0.0
        elapsed = time.time() - self.rate_limited_at
        return max(0.0, RATE_LIMIT_COOLDOWN_SECONDS - elapsed)

    def __repr__(self) -> str:
        if self.state == "rate_limited" and self.rate_limited_at:
            return f"rate_limited ({self.seconds_until_available():.0f}s cooldown)"
        return self.state


PROVIDERS = {
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "model": "gemma-4-31b",
        "description": "Cerebras — ~6 RPM, 1M tokens/day free",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "description": "Groq — 30 RPM, 100K tokens/day free",
    },
    "gemini_flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
        "description": "Gemini 2.5 Flash — stable alias, 10 RPM, 1M context free",
    },
    "gemini_pro": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-pro",
        "description": "Gemini 2.5 Pro — stable alias, 2 RPM, 25 req/day free",
    },
}

TASK_PROVIDER_MAP: dict[TaskType, list[str]] = {
    TaskType.ENRICHMENT:         ["cerebras", "groq", "gemini_flash"],
    TaskType.ENTITY_EXTRACTION:  ["cerebras", "groq", "gemini_flash"],
    TaskType.GAP_DETECTION:      ["cerebras", "groq"],
    TaskType.COMPRESSION:        ["cerebras", "groq"],
    TaskType.VERIFICATION:       ["cerebras", "groq"],
    TaskType.ROUTING:            ["groq", "cerebras"],
    TaskType.REASONING:          ["gemini_flash", "cerebras", "groq"],
    TaskType.FALLBACK:           ["groq", "cerebras"],
}

HARD_REASONING_PROVIDERS: list[str] = [
    "gemini_pro", "gemini_flash", "cerebras", "groq"
]


def _extract_retry_after(error_str: str) -> float:
    """Extract retry delay from error message. Returns default if not found."""
    match = re.search(
        r'(?:retry after|retry in|wait)\s*(\d+(?:\.\d+)?)\s*s',
        error_str, re.IGNORECASE
    )
    if match:
        return float(match.group(1)) + 2

    match = re.search(r'(\d+)\s*requests?\s*per\s*minute', error_str, re.IGNORECASE)
    if match:
        rpm = int(match.group(1))
        return max(10.0, 60.0 / rpm + 2)

    return RATE_LIMIT_COOLDOWN_SECONDS


def _is_quota_exhausted(error_str: str) -> bool:
    """Detect daily quota exhaustion vs temporary rate limiting."""
    patterns = [
        "quota exceeded",
        "generaterequestsperdayperprojectpermodel",
        "daily quota",
        "requests per day",
        "per day",
        "exceeded your current quota",
    ]
    lower = error_str.lower()
    return any(p in lower for p in patterns)


class ProviderRouter:
    """
    Routes LLM calls to the optimal free API provider with smart fallback.
    Distinguishes temporary rate limits from daily quota exhaustion.
    All retry logic lives here — OpenAI SDK retries are disabled.
    """

    def __init__(self) -> None:
        self._clients: dict = {}
        self._status: dict[str, ProviderStatus] = {
            name: ProviderStatus() for name in PROVIDERS
        }
        self._init_clients()

    def _init_clients(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        available = []
        print("\nProviderRouter initialization:")

        for name, config in PROVIDERS.items():
            api_key = os.getenv(config["api_key_env"])

            if not api_key:
                print(f"  SKIP {name}: {config['api_key_env']} not in .env")
                self._status[name].state = "unavailable"
                continue

            api_key = api_key.strip()

            try:
                if name == "gemini_pro" and "gemini_flash" in self._clients:
                    self._clients[name] = self._clients["gemini_flash"]
                    available.append(name)
                    print(f"  OK   {name}: shares Gemini client")
                    continue

                self._clients[name] = OpenAI(
                    api_key=api_key,
                    base_url=config["base_url"],
                    max_retries=0,
                )
                available.append(name)
                print(f"  OK   {name}: {config['description']}")

            except Exception as e:
                self._status[name].state = "unavailable"
                print(f"  FAIL {name}: {str(e)[:80]}")

        if not available:
            print("\nDIAGNOSTIC — .env variable check:")
            for name, config in PROVIDERS.items():
                val = os.getenv(config["api_key_env"])
                print(f"  {config['api_key_env']}: {'SET (' + str(len(val)) + ' chars)' if val else 'NOT SET'}")
            print("\nFix: no spaces around = sign")
            print("  Correct: CEREBRAS_API_KEY=csk-abc123")
            print("  Wrong:   CEREBRAS_API_KEY = csk-abc123")
            raise RuntimeError(
                "No LLM providers available. "
                "Add GROQ_API_KEY or CEREBRAS_API_KEY to .env"
            )

        print(f"  Ready: {available}\n")

    def generate(
        self,
        task: TaskType,
        prompt: str,
        system: str = (
            "You are a senior financial analyst specializing in Indian capital markets. "
            "Be precise, cite sources, and never invent financial data."
        ),
        temperature: float = 0.1,
        max_tokens: int = 1000,
        hard_query: bool = False,
    ) -> str:
        """
        Generate text using the best available provider for this task.

        On rate limit: records timestamp, switches to next provider.
        On quota exhausted: permanently removes provider from rotation.
        On all providers exhausted: waits for shortest cooldown, retries.
        """
        if task == TaskType.REASONING and hard_query:
            provider_order = [p for p in HARD_REASONING_PROVIDERS if p in self._clients]
        else:
            provider_order = [
                p for p in TASK_PROVIDER_MAP.get(task, ["groq", "cerebras"])
                if p in self._clients
            ]

        if not provider_order:
            raise RuntimeError(
                f"No providers configured for '{task.value}'. "
                f"Status: {self.status_report()}"
            )

        last_error: Optional[Exception] = None

        # First pass: try available providers
        for provider_name in provider_order:
            status = self._status[provider_name]
            if not status.is_available():
                continue

            try:
                result = self._call_provider(
                    provider_name, prompt, system, temperature, max_tokens
                )
                if status.state != "ok":
                    status.mark_ok()
                    logger.info(f"[Router] {provider_name} recovered")
                return result

            except Exception as e:
                error_str = str(e)
                last_error = e
                is_rate_limit = any(
                    kw in error_str.lower()
                    for kw in ["429", "rate_limit", "rate limit", "too many requests"]
                )
                is_quota = _is_quota_exhausted(error_str)

                if is_quota:
                    status.mark_quota_exhausted()
                    print(f"[Router] {provider_name} daily quota exhausted — removed from rotation")
                    logger.warning(f"[Router] {provider_name} quota_exhausted: {error_str[:100]}")

                elif is_rate_limit:
                    status.mark_rate_limited()
                    remaining = [
                        p for p in provider_order
                        if p != provider_name and self._status[p].is_available()
                    ]
                    next_p = remaining[0] if remaining else "none"
                    if next_p == "none":
                        msg = f"[Router] {provider_name} rate limited. All providers unavailable. Entering retry phase."
                    else:
                        msg = f"[Router] {provider_name} rate limited → switching to {next_p}"
                    print(msg)
                    logger.warning(msg)

                else:
                    logger.error(f"[Router] {provider_name} error: {error_str[:120]}")

        # Second pass: wait for rate-limited providers to recover
        cooling_down = [
            p for p in provider_order
            if self._status[p].state == "rate_limited" and p in self._clients
        ]

        if cooling_down:
            wait_times = [self._status[p].seconds_until_available() for p in cooling_down]
            wait_s = max(1.0, min(wait_times) + 2)
            print(f"[Router] Waiting {wait_s:.0f}s for providers to recover: {cooling_down}")
            time.sleep(wait_s)

            for provider_name in cooling_down:
                if not self._status[provider_name].is_available():
                    continue
                try:
                    result = self._call_provider(
                        provider_name, prompt, system, temperature, max_tokens
                    )
                    self._status[provider_name].mark_ok()
                    print(f"[Router] {provider_name} recovered after cooldown")
                    return result
                except Exception as e:
                    last_error = e
                    if _is_quota_exhausted(str(e)):
                        self._status[provider_name].mark_quota_exhausted()
                    else:
                        self._status[provider_name].mark_rate_limited()

        raise RuntimeError(
            f"All providers failed for '{task.value}'. "
            f"Last error: {last_error}. "
            f"Status: {self.status_report()}"
        )

    def _call_provider(
        self,
        provider_name: str,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = self._clients[provider_name]
        model = PROVIDERS[provider_name]["model"]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def available_providers(self) -> list[str]:
        return [
            name for name, status in self._status.items()
            if status.state not in ("unavailable", "quota_exhausted")
        ]

    def status_report(self) -> dict:
        return {name: str(status) for name, status in self._status.items()}

    def reset_rate_limits(self) -> None:
        """Reset rate_limited flags. Does NOT reset quota_exhausted."""
        for name, status in self._status.items():
            if status.state == "rate_limited":
                status.mark_ok()
        logger.info("Rate limit flags reset (quota_exhausted preserved)")


_router_instance: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = ProviderRouter()
    return _router_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    print("=" * 55)
    print("FinSight Provider Router — Connection Test")
    print("=" * 55)

    router = ProviderRouter()
    print(f"Status: {router.status_report()}")

    tests = [
        (TaskType.ROUTING,    "One word: financial", 5),
        (TaskType.ENRICHMENT, "One sentence: what is a DRHP?", 50),
        (TaskType.REASONING,  "One sentence: what is an IPO?", 60),
    ]

    print("\nTest calls:")
    for task, prompt, max_tok in tests:
        print(f"\n  {task.value}:")
        try:
            result = router.generate(task=task, prompt=prompt, max_tokens=max_tok)
            print(f"  → {result[:80]}")
        except Exception as e:
            print(f"  → FAILED: {e}")

    print(f"\nFinal status: {router.status_report()}")