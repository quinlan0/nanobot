"""Image tools for reading and sending images."""

import base64
from pathlib import Path
from typing import Any, Callable, Awaitable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class SendImageTool(Tool):
    """Tool to send images to users."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = ""
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    @property
    def name(self) -> str:
        return "send_image"

    @property
    def description(self) -> str:
        return "Send an image file to the user. The image will be displayed in the chat."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file to send"
                },
                "caption": {
                    "type": "string",
                    "description": "Optional caption for the image"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                }
            },
            "required": ["image_path"]
        }

    async def execute(
        self,
        image_path: str,
        caption: str = "",
        channel: str | None = None,
        chat_id: str | None = None,
        **kwargs: Any
    ) -> str:
        # Validate image file first
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            return f"Error: Image file not found: {image_path}"

        if not path.is_file():
            return f"Error: Not a file: {image_path}"

        # Check file extension for common image types
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        if path.suffix.lower() not in valid_extensions:
            return f"Error: Unsupported image format: {path.suffix}. Supported: {', '.join(valid_extensions)}"

        # Check file size (some platforms have limits)
        max_size = 10 * 1024 * 1024  # 10MB
        if path.stat().st_size > max_size:
            return f"Error: Image too large ({path.stat().st_size} bytes). Maximum size: {max_size} bytes"

        channel = channel or self._default_channel
        chat_id = chat_id or self._default_chat_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        content = caption or f"[Image: {path.name}]"

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=[str(path)]
        )

        try:
            await self._send_callback(msg)
            return f"Image sent to {channel}:{chat_id} - {path.name}"
        except Exception as e:
            return f"Error sending image: {str(e)}"


class ReadImageTool(Tool):
    """Tool to read and describe images."""

    @property
    def name(self) -> str:
        return "read_image"

    @property
    def description(self) -> str:
        return "Read an image file and return information about it (format, size, etc.)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file to read"
                }
            },
            "required": ["image_path"]
        }

    async def execute(self, image_path: str, **kwargs: Any) -> str:
        try:
            path = Path(image_path).expanduser().resolve()
            if not path.exists():
                return f"Error: Image file not found: {image_path}"

            if not path.is_file():
                return f"Error: Not a file: {image_path}"

            stat = path.stat()
            size_bytes = stat.st_size
            size_mb = size_bytes / (1024 * 1024)

            # Get basic image info
            ext = path.suffix.lower()
            format_name = {
                '.jpg': 'JPEG',
                '.jpeg': 'JPEG',
                '.png': 'PNG',
                '.gif': 'GIF',
                '.bmp': 'BMP',
                '.webp': 'WebP'
            }.get(ext, f'Unknown ({ext})')

            return f"""Image Information:
File: {path.name}
Path: {path}
Format: {format_name}
Size: {size_bytes:,} bytes ({size_mb:.2f} MB)
Modified: {stat.st_mtime}"""

        except Exception as e:
            return f"Error reading image: {str(e)}"