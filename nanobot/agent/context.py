"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]

    # Per-file character limit when embedding file content into the system prompt.
    # Files exceeding this limit will be truncated to avoid blowing up the prompt.
    MAX_PROMPT_FILE_CHARS = 100_000  # ~100 KB
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)

    def _safe_read_file(self, path: Path, label: str = "") -> str | None:
        """Read a text file with size guard for prompt embedding.

        Returns ``None`` when the file does not exist.  If the file
        exceeds ``MAX_PROMPT_FILE_CHARS`` the content is truncated and a
        warning is logged so the operator can investigate.
        """
        if not path.exists():
            return None
        tag = label or path.name
        file_size = path.stat().st_size
        if file_size > self.MAX_PROMPT_FILE_CHARS * 4:
            # Extremely large file — skip reading entirely
            logger.warning(f"Prompt file {tag} skipped: {file_size} bytes on disk")
            return f"[{tag}: skipped, file too large ({file_size} bytes)]"
        content = path.read_text(encoding="utf-8")
        if len(content) > self.MAX_PROMPT_FILE_CHARS:
            logger.warning(
                f"Prompt file {tag} truncated: {len(content)} chars > "
                f"{self.MAX_PROMPT_FILE_CHARS} limit"
            )
            content = (
                content[: self.MAX_PROMPT_FILE_CHARS]
                + f"\n\n[... {tag} truncated, {len(content)} chars total, "
                f"limit {self.MAX_PROMPT_FILE_CHARS} ...]"
            )
        return content
    
    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.
        
        Args:
            skill_names: Optional list of skills to include.
        
        Returns:
            Complete system prompt.
        """
        parts = []
        
        # Core identity
        parts.append(self._get_identity())
        
        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        
        # Memory context (with size guard)
        memory = self.memory.get_memory_context()
        if memory:
            if len(memory) > self.MAX_PROMPT_FILE_CHARS:
                logger.warning(
                    f"Memory context truncated: {len(memory)} chars > "
                    f"{self.MAX_PROMPT_FILE_CHARS} limit"
                )
                memory = (
                    memory[: self.MAX_PROMPT_FILE_CHARS]
                    + f"\n\n[... memory truncated, {len(memory)} chars total, "
                    f"limit {self.MAX_PROMPT_FILE_CHARS} ...]"
                )
            parts.append(f"# Memory\n\n{memory}")
        
        # Skills - progressive loading
        # 1. Always-loaded skills: include full content (with size guard)
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                if len(always_content) > self.MAX_PROMPT_FILE_CHARS:
                    logger.warning(
                        f"Active skills truncated: {len(always_content)} chars > "
                        f"{self.MAX_PROMPT_FILE_CHARS} limit"
                    )
                    always_content = (
                        always_content[: self.MAX_PROMPT_FILE_CHARS]
                        + f"\n\n[... skills truncated, {len(always_content)} chars total, "
                        f"limit {self.MAX_PROMPT_FILE_CHARS} ...]"
                    )
                parts.append(f"# Active Skills\n\n{always_content}")
        
        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")
        
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """Get the core identity section."""
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        
        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant. You have access to tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search the web and fetch web pages
- Send messages to users on chat channels
- Spawn subagents for complex background tasks

## Current Time
{now} ({tz})

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable)
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

IMPORTANT: When responding to direct questions or conversations, reply directly with your text response.
Only use the 'message' tool when you need to send a message to a specific chat channel (like Telegram).
For normal conversation, just respond with text - do not call the message tool.

Always be helpful, accurate, and concise. When using tools, think step by step: what you know, what you need, and why you chose this tool.
When remembering something important, write to {workspace_path}/memory/MEMORY.md
To recall past events, grep {workspace_path}/memory/HISTORY.md"""
    
    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace (with size guard)."""
        parts = []
        
        for filename in self.BOOTSTRAP_FILES:
            content = self._safe_read_file(self.workspace / filename, label=filename)
            if content is not None:
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""
    
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.
            model: Model identifier (used to decide encoding strategy).

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt
        system_prompt = self.build_system_prompt(skill_names)
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media, model)
        messages.append({"role": "user", "content": user_content})

        return messages

    # Models with strict request-size limits that cannot accept base64 images
    _TEXT_ONLY_MODEL_KEYWORDS = ("moonshot", "kimi")

    def _build_user_content(
        self, text: str, media: list[str] | None, model: str | None = None,
    ) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images.

        For models that have strict payload size limits (e.g. Moonshot 4 MB),
        images are referenced by path instead of being base64-encoded.
        """
        if not media:
            return text

        model_lower = (model or "").lower()
        use_text_only = any(kw in model_lower for kw in self._TEXT_ONLY_MODEL_KEYWORDS)

        if use_text_only:
            refs = []
            for path in media:
                p = Path(path)
                if p.is_file():
                    size_kb = p.stat().st_size / 1024
                    refs.append(f"[Image: {path} ({size_kb:.0f}KB)]")
            if refs:
                return text + "\n" + "\n".join(refs)
            return text

        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        
        if not images:
            return text
        return images + [{"type": "text", "text": text}]
    
    # Hard cap for any single tool result embedded into the conversation.
    MAX_TOOL_RESULT_CHARS = 3 * 1024 * 1024  # 3 MB (aligned with read_file limit)

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list (with size guard).
        
        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
        
        Returns:
            Updated message list.
        """
        if len(result) > self.MAX_TOOL_RESULT_CHARS:
            logger.warning(
                f"Tool result from '{tool_name}' truncated: "
                f"{len(result)} chars > {self.MAX_TOOL_RESULT_CHARS} limit"
            )
            result = (
                result[: self.MAX_TOOL_RESULT_CHARS]
                + f"\n\n[... output truncated, {len(result)} chars total, limit 3MB ...]"
            )
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        return messages
    
    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        
        if tool_calls:
            msg["tool_calls"] = tool_calls
        
        # Thinking models reject history without this
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        
        messages.append(msg)
        return messages
