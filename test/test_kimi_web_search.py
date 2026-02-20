"""Direct Moonshot API test — verify $web_search with thinking mode handling."""

import json
import httpx
import asyncio

API_KEY = "sk-uJ8fwU6cMctqNatKB9HxdohSfR9hmwVFg8rDpDBzeBIg5MdV"
API_BASE = "https://api.moonshot.cn/v1"
TOOLS = [{"type": "builtin_function", "function": {"name": "$web_search"}}]


async def call_api(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    return {"status": resp.status_code, "data": resp.json()}


async def test_variant(label: str, extra_params: dict, assistant_transform=None):
    """Test a specific variant of the web search flow."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": "今天北京天气怎么样"}]
    payload = {
        "model": "kimi-k2.5",
        "messages": messages,
        "tools": TOOLS,
        "temperature": 1.0,
        "max_tokens": 2048,
        **extra_params,
    }

    r1 = await call_api(payload)
    if r1["status"] != 200:
        print(f"Round 1 FAILED: {r1['data']}")
        return

    choice = r1["data"]["choices"][0]
    assistant_msg = choice["message"]
    print(f"Round 1: finish={choice['finish_reason']}, keys={list(assistant_msg.keys())}")

    if choice["finish_reason"] != "tool_calls":
        print(f"Direct response: {assistant_msg.get('content', '')[:200]}")
        return

    tc = assistant_msg["tool_calls"][0]
    print(f"  tool_call: {tc['function']['name']}, id={tc['id']}")

    # Build round 2
    echo = dict(assistant_msg)
    if assistant_transform:
        echo = assistant_transform(echo)

    messages.append(echo)
    messages.append({
        "role": "tool",
        "tool_call_id": tc["id"],
        "name": "$web_search",
        "content": tc["function"]["arguments"],
    })

    payload2 = {
        "model": "kimi-k2.5",
        "messages": messages,
        "tools": TOOLS,
        "temperature": 1.0,
        "max_tokens": 2048,
        **extra_params,
    }

    r2 = await call_api(payload2)
    if r2["status"] != 200:
        print(f"Round 2 FAILED: {json.dumps(r2['data'], ensure_ascii=False)}")
        return

    content = r2["data"]["choices"][0]["message"].get("content", "")
    print(f"Round 2 SUCCESS: {content[:300]}")


async def main():
    # Test 1: Disable thinking explicitly
    await test_variant(
        "thinking disabled",
        extra_params={"thinking": {"type": "disabled"}},
    )

    # Test 2: Enable thinking, pass reasoning_content=None
    await test_variant(
        "thinking enabled + reasoning_content=None",
        extra_params={},
        assistant_transform=lambda m: {**m, "reasoning_content": None},
    )

    # Test 3: Enable thinking, pass reasoning_content=" " (space)
    await test_variant(
        "thinking enabled + reasoning_content=' '",
        extra_params={},
        assistant_transform=lambda m: {**m, "reasoning_content": " "},
    )


if __name__ == "__main__":
    asyncio.run(main())
