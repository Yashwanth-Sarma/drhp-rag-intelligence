"""
src/llm/provider_router.py

Smart multi-provider LLM router for FinSight.

Every task in the pipeline is routed to the optimal free API provider.
If the primary provider fails (rate limit, timeout, error), it automatically
falls back to the next provider. The pipeline never crashes due to one API failing.

Provider assignments (based on free tier limits + model quality):
    ENRICHMENT / ENTITY_EXTRACTION  → Cerebras  (1M tokens/day, very fast)
    ROUTING / CLASSIFICATION        → Groq      (fast, good at short tasks)
    GAP_DETECTION / COMPRESSION     → Cerebras  (bulk work, speed matters)
    REASONING (easy queries)        → Gemini 2.5 Flash (free, 500 req/day)
    REASONING (hard queries)        → Gemini 2.5 Pro   (free, 25 req/day)
    VERIFICATION (semantic only)    → Cerebras  (fast, cheap)

Before this file:
    Groq 100K tokens/day was used for EVERYTHING → constant blocking
After this file:
    Cerebras handles bulk work (1M/day) → Groq used sparingly → no more blocking

Fallback chain per task:
    ENRICHMENT:  Cerebras → Groq → error
    REASONING:   Gemini Flash/Pro → Cerebras → Groq → error
    ROUTING:     Groq → Cerebras → error

Usage:
    from src.llm.provider_router import get_router, TaskType
    router = get_router()
    text = router.generate(TaskType.ENRICHMENT, prompt="Generate context...")
    text = router.generate(TaskType.REASONING, prompt="Answer this...", hard_query=True)
"""

import os
import time
import logging
from dotenv import load_dotenv
load_dotenv()
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """
    Every LLM task in the pipeline maps to one of these.
    The router picks the right provider automatically.
    """
    ROUTING = "routing"               # Query intent classification
    ENRICHMENT = "enrichment"         # Contextual chunk enrichment (Stage 2)
    ENTITY_EXTRACTION = "entity_extraction"  # GraphRAG entity extraction
    GAP_DETECTION = "gap_detection"   # Evidence gap detection
    COMPRESSION = "compression"       # Evidence compression
    REASONING = "reasoning"           # Final answer generation (Flash/Pro adaptive)
    VERIFICATION = "verification"     # Semantic claim verification
    FALLBACK = "fallback"             # Generic fallback task


# ── Provider definitions ────────────────────────────────────────────────────
# Every provider uses OpenAI-compatible format. To swap a provider,
# change one entry here — nothing else in the codebase needs to change.

PROVIDERS = {
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "model": "gemma-4-31b",
        "daily_token_limit": 1_000_000,
        "rpm_limit": 30,
        "description": "Cerebras WSE — 1M tokens/day free, very fast",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "daily_token_limit": 100_000,
        "rpm_limit": 30,
        "description": "Groq LPU — 100K tokens/day free, fastest latency",
    },
    "gemini_flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
        "daily_token_limit": 1_000_000,
        "rpm_limit": 10,
        "description": "Gemini 2.5 Flash — free, 1M context, best for standard reasoning",
    },
    "gemini_pro": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-pro",
        "daily_token_limit": 50_000,
        "rpm_limit": 2,
        "description": "Gemini 2.5 Pro — free (25 req/day), best quality for hard queries",
    },
}

# Task → ordered list of providers to try (primary first, fallbacks after)
TASK_PROVIDER_MAP: dict[TaskType, list[str]] = {
    TaskType.ENRICHMENT:         ["cerebras", "groq"],
    TaskType.ENTITY_EXTRACTION:  ["cerebras", "groq"],
    TaskType.GAP_DETECTION:      ["cerebras", "groq"],
    TaskType.COMPRESSION:        ["cerebras", "groq"],
    TaskType.VERIFICATION:       ["cerebras", "groq"],
    TaskType.ROUTING:            ["groq", "cerebras"],
    TaskType.REASONING:          ["gemini_flash", "cerebras", "groq"],
    TaskType.FALLBACK:           ["groq", "cerebras"],
}

# Hard reasoning queries use Gemini Pro instead of Flash
HARD_REASONING_PROVIDERS: list[str] = ["gemini_pro", "gemini_flash", "cerebras", "groq"]


class ProviderRouter:
    """
    Routes every LLM call to the optimal free API provider.
    Handles fallback automatically when a provider fails or rate-limits.

    Reports clearly when switching providers:
        "Cerebras rate limited → switching to Groq"
        "Groq unavailable → switching to Cerebras"
    """

    def __init__(self) -> None:
        self._clients: dict = {}
        self._provider_status: dict[str, str] = {}  # "ok" | "rate_limited" | "unavailable"
        self._init_clients()

    def _init_clients(self) -> None:
        """
        Try to initialize a client for every configured provider.
        Providers without an API key in .env are skipped (not an error).
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Run: pip install openai"
            )

        available = []
        missing_keys = []

        for name, config in PROVIDERS.items():
            api_key = os.getenv(config["api_key_env"])
            if not api_key:
                missing_keys.append(f"{name} ({config['api_key_env']})")
                self._provider_status[name] = "unavailable"
                continue

            try:
                self._clients[name] = OpenAI(
                    api_key=api_key,
                    base_url=config["base_url"],
                )
                self._provider_status[name] = "ok"
                available.append(name)
                logger.info(f"Provider '{name}' ready — {config['description']}")
            except Exception as e:
                self._provider_status[name] = "unavailable"
                logger.warning(f"Provider '{name}' failed to initialize: {e}")

        if not available:
            raise RuntimeError(
                "No LLM providers available. "
                "At minimum, add GROQ_API_KEY or CEREBRAS_API_KEY to your .env file."
            )

        if missing_keys:
            logger.info(
                f"Providers not configured (missing API keys): {missing_keys}. "
                "This is fine — the router will use available providers."
            )

        logger.info(f"ProviderRouter ready. Available providers: {available}")

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
        Generate a response using the optimal provider for the given task.

        Args:
            task:        Which task this is — determines provider selection.
            prompt:      The user/instruction message.
            system:      System prompt (default is financial analyst context).
            temperature: Sampling temperature. Use 0.0 for deterministic tasks.
            max_tokens:  Maximum tokens to generate.
            hard_query:  For REASONING tasks only. True = use Gemini Pro instead of Flash.
                         Hard queries: cross-document comparison, contradiction analysis,
                         multi-hop financial reasoning.

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If all providers fail for this task.

        Examples:
            # Stage 2 contextual enrichment (goes to Cerebras)
            context = router.generate(
                task=TaskType.ENRICHMENT,
                prompt="Generate context for: [chunk text]",
                max_tokens=120,
            )

            # Easy factual query (goes to Gemini Flash)
            answer = router.generate(
                task=TaskType.REASONING,
                prompt="What are the risk factors? Evidence: [evidence]",
            )

            # Hard cross-document query (goes to Gemini Pro)
            answer = router.generate(
                task=TaskType.REASONING,
                prompt="Compare Zomato and Paytm's litigation... Evidence: [evidence]",
                hard_query=True,
            )
        """
        # Select provider order based on task and difficulty
        if task == TaskType.REASONING and hard_query:
            provider_order = HARD_REASONING_PROVIDERS
        else:
            provider_order = TASK_PROVIDER_MAP.get(task, ["groq", "cerebras"])

        last_error: Optional[Exception] = None

        for provider_name in provider_order:
            # Skip providers that aren't configured
            if provider_name not in self._clients:
                continue

            # Skip providers we know are currently rate-limited (try them after others)
            if self._provider_status.get(provider_name) == "rate_limited":
                logger.debug(f"Skipping '{provider_name}' (rate limited, trying next)")
                continue

            try:
                result = self._call_provider(
                    provider_name=provider_name,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Restore status if it was previously rate-limited and now works
                if self._provider_status.get(provider_name) != "ok":
                    logger.info(f"Provider '{provider_name}' recovered — marking as ok")
                    self._provider_status[provider_name] = "ok"

                logger.debug(
                    f"Task '{task.value}' completed via '{provider_name}' "
                    f"({len(result)} chars output)"
                )
                return result

            except Exception as e:
                error_str = str(e)
                last_error = e

                if "429" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
                    self._provider_status[provider_name] = "rate_limited"
                    # Find next available provider for the message
                    remaining = [
                        p for p in provider_order
                        if p != provider_name and p in self._clients
                    ]
                    next_p = remaining[0] if remaining else "none"
                    logger.warning(
                        f"PROVIDER SWITCH: '{provider_name}' rate limited → "
                        f"switching to '{next_p}' for task '{task.value}'"
                    )
                    time.sleep(1)
                    continue

                elif "timeout" in error_str.lower() or "connection" in error_str.lower():
                    logger.warning(
                        f"PROVIDER SWITCH: '{provider_name}' connection issue → "
                        f"trying next provider. Error: {error_str[:80]}"
                    )
                    time.sleep(2)
                    continue

                else:
                    logger.error(
                        f"Provider '{provider_name}' failed with unexpected error: "
                        f"{error_str[:120]}"
                    )
                    continue

        # All providers exhausted — try rate-limited providers one more time with backoff
        rate_limited_providers = [
            p for p in provider_order
            if p in self._clients and self._provider_status.get(p) == "rate_limited"
        ]

        if rate_limited_providers:
            logger.info(
                f"All fresh providers exhausted for task '{task.value}'. "
                f"Waiting 15s then retrying rate-limited providers: {rate_limited_providers}"
            )
            time.sleep(15)
            for provider_name in rate_limited_providers:
                try:
                    result = self._call_provider(
                        provider_name=provider_name,
                        prompt=prompt,
                        system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    self._provider_status[provider_name] = "ok"
                    logger.info(f"Rate-limited provider '{provider_name}' succeeded after wait")
                    return result
                except Exception as e:
                    logger.warning(f"Retry of '{provider_name}' failed: {str(e)[:80]}")
                    continue

        raise RuntimeError(
            f"All providers failed for task '{task.value}'. "
            f"Last error: {last_error}. "
            f"Provider status: {self._provider_status}"
        )

    def _call_provider(
        self,
        provider_name: str,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Make the actual API call to one specific provider."""
        client = self._clients[provider_name]
        config = PROVIDERS[provider_name]

        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def status_report(self) -> dict:
        """
        Returns current status of all providers.
        Useful for debugging when things are slow.

        Example output:
            {
                "cerebras": "ok",
                "groq": "rate_limited",
                "gemini_flash": "ok",
                "gemini_pro": "unavailable"
            }
        """
        return {
            name: self._provider_status.get(name, "unknown")
            for name in PROVIDERS
        }

    def reset_rate_limits(self) -> None:
        """
        Resets all provider statuses to 'ok'.
        Call this at the start of a new session after daily limits reset.
        """
        for name in self._clients:
            self._provider_status[name] = "ok"
        logger.info("All provider rate limit flags reset.")


# ── Global singleton ─────────────────────────────────────────────────────────
# Use get_router() everywhere instead of creating ProviderRouter() directly.
# This ensures one shared instance across all modules.

_router_instance: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    """
    Get or create the global ProviderRouter singleton.
    Thread-safe for single-process use (Streamlit, FastAPI single-worker).
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = ProviderRouter()
    return _router_instance


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    print("Initializing ProviderRouter...")
    router = ProviderRouter()

    print(f"\nProvider status: {router.status_report()}")

    # Test each available provider
    test_cases = [
        (TaskType.ROUTING, "Is 'What are the risk factors?' a risk or financial question? Reply: risk/financial/other", 10),
        (TaskType.ENRICHMENT, "In one sentence, what is a DRHP?", 50),
        (TaskType.REASONING, "What is the main purpose of a DRHP filing? One paragraph.", 150),
    ]

    for task, prompt, max_tok in test_cases:
        print(f"\nTask: {task.value}")
        try:
            result = router.generate(task=task, prompt=prompt, max_tokens=max_tok)
            print(f"Result: {result[:100]}{'...' if len(result) > 100 else ''}")
        except Exception as e:
            print(f"Failed: {e}")

    print(f"\nFinal provider status: {router.status_report()}")