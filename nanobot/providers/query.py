"""Provider query utilities — list models, check balance/usage.

All three providers (Moonshot, Volcengine/Doubao, DashScope/Qwen)
expose an OpenAI-compatible ``GET /models`` endpoint.  Balance/usage
endpoints are provider-specific and may not exist for all providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

_TIMEOUT = 15


@dataclass
class ModelInfo:
    id: str
    owned_by: str = ""
    created: int = 0


@dataclass
class BalanceInfo:
    available: str = ""
    currency: str = ""
    raw: dict[str, Any] | None = None


# ------------------------------------------------------------------
# Generic OpenAI-compatible /models
# ------------------------------------------------------------------

def list_models(api_base: str, api_key: str) -> list[ModelInfo]:
    """Query ``GET {api_base}/models`` and return a sorted model list."""
    url = f"{api_base.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = httpx.get(url, headers=headers, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    models = [
        ModelInfo(
            id=m.get("id", "?"),
            owned_by=m.get("owned_by", ""),
            created=m.get("created", 0),
        )
        for m in data
    ]
    models.sort(key=lambda m: m.id)
    return models


# ------------------------------------------------------------------
# Provider-specific balance / usage
# ------------------------------------------------------------------

def query_moonshot_balance(api_base: str, api_key: str) -> BalanceInfo | None:
    """Moonshot ``GET /v1/users/me/balance``."""
    url = f"{api_base.rstrip('/')}/users/me/balance"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = httpx.get(url, headers=headers, timeout=_TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data", body)
        return BalanceInfo(
            available=str(data.get("available_balance", data.get("balance", "?"))),
            currency=str(data.get("currency", "")),
            raw=data,
        )
    except Exception:
        return None


def query_dashscope_usage(api_base: str, api_key: str) -> BalanceInfo | None:
    """DashScope does not have a public balance API; return None."""
    return None


def query_doubao_usage(api_base: str, api_key: str) -> BalanceInfo | None:
    """Volcengine Ark does not have a public balance API; return None."""
    return None


# ------------------------------------------------------------------
# Dispatcher — pick the right function by provider name
# ------------------------------------------------------------------

_BALANCE_FN = {
    "moonshot": query_moonshot_balance,
    "dashscope": query_dashscope_usage,
    "doubao": query_doubao_usage,
}


def query_balance(provider_name: str, api_base: str, api_key: str) -> BalanceInfo | None:
    fn = _BALANCE_FN.get(provider_name)
    if fn is None:
        return None
    return fn(api_base, api_key)
