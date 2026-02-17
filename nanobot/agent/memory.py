"""Memory system for persistent agent memory."""

from pathlib import Path

from loguru import logger

from nanobot.utils.helpers import ensure_dir

# Max characters when reading memory files for prompt embedding.
_MAX_MEMORY_CHARS = 100_000


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        if not self.memory_file.exists():
            return ""
        file_size = self.memory_file.stat().st_size
        if file_size > _MAX_MEMORY_CHARS * 4:
            logger.warning(f"MEMORY.md too large ({file_size} bytes), skipped")
            return f"[MEMORY.md skipped: file too large ({file_size} bytes)]"
        content = self.memory_file.read_text(encoding="utf-8")
        if len(content) > _MAX_MEMORY_CHARS:
            logger.warning(f"MEMORY.md truncated: {len(content)} chars > {_MAX_MEMORY_CHARS}")
            content = (
                content[:_MAX_MEMORY_CHARS]
                + f"\n\n[... MEMORY.md truncated, {len(content)} chars total ...]"
            )
        return content

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""
