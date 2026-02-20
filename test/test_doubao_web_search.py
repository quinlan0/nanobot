"""
Test script for Doubao (Volcengine Ark) web search integration.

Doubao's web search is a top-level request body parameter (NOT a tool
in the tools array).  Format:
    "web_search": {"enable": true, "search_mode": "auto"}

Usage:
    python test/test_doubao_web_search.py

Set VOLCENGINE_API_KEY env var or edit API_KEY below before running.
"""

import json
import os

import httpx

API_KEY = "1a6bbc97-0185-4557-a9dc-800981d602f1"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-seed-2-0-pro-260215"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}


def test_web_search_only():
    """Test A: web_search as top-level param, no function tools."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个有用的助手，可以联网搜索最新信息。"},
            {"role": "user", "content": "今天最新的新闻信息"},
        ],
        "max_tokens": 4096,
        "temperature": 0.7,
        "web_search": {
            "enable": True,
            "search_mode": "auto",
        },
    }
    _send("Test A: web_search only (no function tools)", payload)


def test_web_search_with_tools():
    """Test B: web_search as top-level param + function tools in tools array."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个有用的助手，可以联网搜索最新信息。"},
            {"role": "user", "content": "最近交易日A股大盘行情怎么样？给出关键指数的情况"},
        ],
        "tools": [FUNCTION_TOOL],
        "tool_choice": "auto",
        "max_tokens": 4096,
        "temperature": 0.7,
        "web_search": {
            "enable": True,
            "search_mode": "auto",
        },
    }
    _send("Test B: web_search + function tools", payload)


def _send(label: str, payload: dict):
    print("=" * 60)
    print(label)
    print("=" * 60)
    print("REQUEST (excluding messages):")
    log = {k: v for k, v in payload.items() if k != "messages"}
    print(json.dumps(log, indent=2, ensure_ascii=False))
    print()

    url = f"{BASE_URL}/chat/completions"
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, headers=HEADERS, json=payload)

    print(f"HTTP Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
        print()
        return

    data = resp.json()
    choice = data["choices"][0]
    message = choice["message"]
    finish_reason = choice.get("finish_reason")

    print(f"finish_reason: {finish_reason}")
    content = message.get("content", "(empty)")
    print(f"content: {content[:600]}")

    if message.get("tool_calls"):
        print(f"\ntool_calls ({len(message['tool_calls'])}):")
        for tc in message["tool_calls"]:
            print(f"  - id={tc['id']}, type={tc.get('type')}, "
                  f"name={tc['function']['name']}, "
                  f"args={tc['function']['arguments'][:200]}")

    if "web_search" in data:
        print(f"\nweb_search in response: {json.dumps(data['web_search'], indent=2, ensure_ascii=False)[:500]}")

    usage = data.get("usage", {})
    print(f"\nusage: prompt={usage.get('prompt_tokens')}, "
          f"completion={usage.get('completion_tokens')}, "
          f"total={usage.get('total_tokens')}")
    print()


if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        print("请先设置 VOLCENGINE_API_KEY 环境变量或编辑脚本中的 API_KEY")
        print("  export VOLCENGINE_API_KEY=your-key-here")
        exit(1)

    test_web_search_only()
    test_web_search_with_tools()
