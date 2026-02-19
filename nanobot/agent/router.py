"""Multi-agent router: LLM-based message routing to specialist agents."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop


_KEYWORD_PATTERNS: dict[str, list[re.Pattern]] = {}


def _compile_keywords(keywords: dict[str, list[str]]) -> dict[str, list[re.Pattern]]:
    """Pre-compile keyword patterns for rule-based fallback routing."""
    return {
        name: [re.compile(kw, re.IGNORECASE) for kw in kws]
        for name, kws in keywords.items()
    }


class RouterLoop:
    """Routes incoming messages to specialist AgentLoop instances.

    Routing strategy:
    1. Try LLM classification (fast, low-token call).
    2. On LLM failure, fall back to keyword matching.
    3. If nothing matches, use the default agent.
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        agents: dict[str, AgentLoop],
        model: str | None = None,
        default_agent: str | None = None,
        keyword_map: dict[str, list[str]] | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.agents = agents
        self.model = model
        self.default_agent = default_agent or next(iter(agents))
        self._running = False

        self._routing_prompt = self._build_routing_prompt()
        self._keyword_patterns = _compile_keywords(keyword_map or {})

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_routing_prompt(self) -> str:
        agents_desc = "\n".join(
            f"- {name}: {agent.specialist_description}"
            for name, agent in self.agents.items()
        )
        return (
            "You are a message router. Based on the user's message, "
            "decide which specialist agent should handle it.\n\n"
            f"Available agents:\n{agents_desc}\n\n"
            "Reply with ONLY the agent name (one word). Nothing else."
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def _route(self, content: str) -> str:
        """Determine which agent handles *content*."""
        # 1) LLM routing
        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": self._routing_prompt},
                    {"role": "user", "content": content},
                ],
                model=self.model,
                max_tokens=32,
                temperature=0.0,
            )
            raw = (response.content or "").strip().lower()
            if raw in self.agents:
                return raw
            for name in self.agents:
                if name in raw:
                    return name
        except Exception as e:
            logger.warning(f"Router LLM failed, using keyword fallback: {e}")

        # 2) Keyword fallback
        return self._route_by_keywords(content)

    def _route_by_keywords(self, content: str) -> str:
        for name, patterns in self._keyword_patterns.items():
            if name not in self.agents:
                continue
            for pat in patterns:
                if pat.search(content):
                    return name
        return self.default_agent

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Consume inbound messages and dispatch to specialist agents.

        Each message is handled in its own asyncio task so that multiple
        users (or multiple messages) can be processed concurrently by
        different specialist agents.
        """
        self._running = True
        self._pending_tasks: set[asyncio.Task] = set()
        agent_names = list(self.agents.keys())
        logger.info(f"Router started — agents: {agent_names}, default: {self.default_agent}")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                task = asyncio.create_task(self._safe_dispatch(msg))
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)
            except asyncio.TimeoutError:
                continue

    async def _safe_dispatch(self, msg: InboundMessage) -> None:
        """Dispatch a single message with error handling (runs as a task)."""
        try:
            response = await self._dispatch(msg)
            if response:
                await self.bus.publish_outbound(response)
        except Exception as e:
            logger.error(f"Router dispatch error: {e}")
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Sorry, I encountered an error: {str(e)}",
            ))

    def stop(self) -> None:
        self._running = False
        for task in getattr(self, "_pending_tasks", set()):
            task.cancel()
        logger.info("Router stopping")

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, msg: InboundMessage) -> OutboundMessage | None:
        # System messages always go to the default agent
        if msg.channel == "system":
            return await self.agents[self.default_agent]._process_message(msg)

        cmd = msg.content.strip().lower()

        # /new — clear sessions for ALL agents
        if cmd == "/new":
            for name, agent in self.agents.items():
                session_key = f"{name}:{msg.session_key}"
                session = agent.sessions.get_or_create(session_key)
                session.clear()
                agent.sessions.save(session)
                agent.sessions.invalidate(session.key)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="All agent sessions cleared. Starting fresh.",
            )

        # /help
        if cmd == "/help":
            lines = [f"  • {n}: {a.specialist_description}" for n, a in self.agents.items()]
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    "🐈 nanobot (multi-agent mode)\n\n"
                    "Commands:\n  /new — Clear all sessions\n  /help — This help\n\n"
                    f"Specialist agents:\n" + "\n".join(lines)
                ),
            )

        # Route the message
        agent_name = await self._route(msg.content)
        logger.info(f"Routed [{msg.channel}:{msg.sender_id}] → [{agent_name}]: {msg.content[:60]}...")
        agent = self.agents[agent_name]
        session_key = f"{agent_name}:{msg.session_key}"
        return await agent._process_message(msg, session_key=session_key)
