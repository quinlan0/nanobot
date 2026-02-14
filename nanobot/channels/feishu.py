"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import json
import re
import threading
from collections import OrderedDict
from typing import Any
from pathlib import Path

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest,
        CreateMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        CreateImageRequest,
        CreateImageRequestBody,
        Emoji,
        P2ImMessageReceiveV1,
    )
    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None
    Emoji = None
    CreateImageRequest = None
    CreateImageRequestBody = None

# Message type display mapping
MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}


class FeishuChannel(BaseChannel):
    """
    Feishu/Lark channel using WebSocket long connection.
    
    Uses WebSocket to receive events - no public IP or webhook required.
    
    Requires:
    - App ID and App Secret from Feishu Open Platform
    - Bot capability enabled
    - Event subscription enabled (im.message.receive_v1)
    """
    
    name = "feishu"
    
    def __init__(self, config: FeishuConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()  # Ordered dedup cache
        self._loop: asyncio.AbstractEventLoop | None = None
    
    async def start(self) -> None:
        """Start the Feishu bot with WebSocket long connection."""
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return
        
        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return
        
        self._running = True
        self._loop = asyncio.get_running_loop()
        
        # Create Lark client for sending messages
        self._client = lark.Client.builder() \
            .app_id(self.config.app_id) \
            .app_secret(self.config.app_secret) \
            .log_level(lark.LogLevel.DEBUG) \
            .build()
        
        # Create event handler (only register message receive, ignore other events)
        event_handler = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        ).register_p2_im_message_receive_v1(
            self._on_message_sync
        ).build()
        
        # Create WebSocket client for long connection
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.DEBUG
        )
        
        # Start WebSocket client in a separate thread with reconnect loop
        def run_ws():
            while self._running:
                try:
                    self._ws_client.start()
                except Exception as e:
                    logger.warning(f"Feishu WebSocket error: {e}")
                if self._running:
                    import time; time.sleep(5)
        
        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()
        
        logger.info("Feishu bot started with WebSocket long connection")
        logger.info("No public IP required - using WebSocket to receive events")
        
        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)
    
    async def stop(self) -> None:
        """Stop the Feishu bot."""
        self._running = False
        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception as e:
                logger.warning(f"Error stopping WebSocket client: {e}")
        logger.info("Feishu bot stopped")
    
    def _add_reaction_sync(self, message_id: str, emoji_type: str) -> None:
        """Sync helper for adding reaction (runs in thread pool)."""
        try:
            request = CreateMessageReactionRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                ).build()
            
            response = self._client.im.v1.message_reaction.create(request)
            
            if not response.success():
                logger.warning(f"Failed to add reaction: code={response.code}, msg={response.msg}")
            else:
                logger.debug(f"Added {emoji_type} reaction to message {message_id}")
        except Exception as e:
            logger.warning(f"Error adding reaction: {e}")

    async def _add_reaction(self, message_id: str, emoji_type: str = "THUMBSUP") -> None:
        """
        Add a reaction emoji to a message (non-blocking).
        
        Common emoji types: THUMBSUP, OK, EYES, DONE, OnIt, HEART
        """
        if not self._client or not Emoji:
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._add_reaction_sync, message_id, emoji_type)
    
    # Regex to match markdown tables (header + separator + data rows)
    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    _CODE_BLOCK_RE = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        """Parse a markdown table into a Feishu table element."""
        lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            return None
        split = lambda l: [c.strip() for c in l.strip("|").split("|")]
        headers = split(lines[0])
        rows = [split(l) for l in lines[2:]]
        columns = [{"tag": "column", "name": f"c{i}", "display_name": h, "width": "auto"}
                   for i, h in enumerate(headers)]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": columns,
            "rows": [{f"c{i}": r[i] if i < len(r) else "" for i in range(len(headers))} for r in rows],
        }

    def _build_card_elements(self, content: str) -> list[dict]:
        """Split content into div/markdown + table elements for Feishu card."""
        elements, last_end = [], 0
        for m in self._TABLE_RE.finditer(content):
            before = content[last_end:m.start()]
            if before.strip():
                elements.extend(self._split_headings(before))
            elements.append(self._parse_md_table(m.group(1)) or {"tag": "markdown", "content": m.group(1)})
            last_end = m.end()
        remaining = content[last_end:]
        if remaining.strip():
            elements.extend(self._split_headings(remaining))
        return elements or [{"tag": "markdown", "content": content}]

    def _split_headings(self, content: str) -> list[dict]:
        """Split content by headings, converting headings to div elements."""
        protected = content
        code_blocks = []
        for m in self._CODE_BLOCK_RE.finditer(content):
            code_blocks.append(m.group(1))
            protected = protected.replace(m.group(1), f"\x00CODE{len(code_blocks)-1}\x00", 1)

        elements = []
        last_end = 0
        for m in self._HEADING_RE.finditer(protected):
            before = protected[last_end:m.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            level = len(m.group(1))
            text = m.group(2).strip()
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**{text}**",
                },
            })
            last_end = m.end()
        remaining = protected[last_end:].strip()
        if remaining:
            elements.append({"tag": "markdown", "content": remaining})

        for i, cb in enumerate(code_blocks):
            for el in elements:
                if el.get("tag") == "markdown":
                    el["content"] = el["content"].replace(f"\x00CODE{i}\x00", cb)

        return elements or [{"tag": "markdown", "content": content}]

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return

        try:
            # Determine receive_id_type based on chat_id format
            # open_id starts with "ou_", chat_id starts with "oc_"
            if msg.chat_id.startswith("oc_"):
                receive_id_type = "chat_id"
            else:
                receive_id_type = "open_id"

            # Send media files first (if any)
            if msg.media:
                await self._send_media_files(msg.chat_id, receive_id_type, msg.media)

            # Send text content if present
            if msg.content.strip():
                await self._send_text_message(msg.chat_id, receive_id_type, msg.content)

        except Exception as e:
            logger.error(f"Error sending Feishu message: {e}")

    async def _send_media_files(self, chat_id: str, receive_id_type: str, media_paths: list[str]) -> None:
        """Send media files through Feishu."""
        from pathlib import Path

        logger.info(f"Feishu: Starting to send {len(media_paths)} media files to {chat_id}")

        if not CreateImageRequest or not CreateImageRequestBody:
            logger.warning("Feishu image API not available - lark-oapi SDK not properly imported")
            return

        for media_path in media_paths:
            logger.info(f"Feishu: Processing media file: {media_path}")
            try:
                path = Path(media_path)
                if not path.exists():
                    logger.error(f"Feishu: Media file not found: {media_path}")
                    continue

                file_size = path.stat().st_size
                logger.info(f"Feishu: File exists, size: {file_size} bytes, extension: {path.suffix}")

                # Determine image type from file extension
                image_type = self._get_image_type(path.suffix.lower())
                logger.info(f"Feishu: Determined image type: {image_type}")

                # Upload image first to get image_key
                logger.info("Feishu: Starting image upload...")

                # Read file into memory first (飞书API可能需要BytesIO而不是文件对象)
                with open(path, 'rb') as f:
                    image_data = f.read()

                logger.info(f"Feishu: Read {len(image_data)} bytes of image data")

                # Use BytesIO for better compatibility
                from io import BytesIO
                image_stream = BytesIO(image_data)

                # Try a different approach: use only files field without request_body
                # This is because the SDK serializes BytesIO to {} in JSON
                upload_request = CreateImageRequest.builder().build()

                # Set files field for multipart/form-data upload (飞书API需要)
                upload_request.files = {
                    'image': (path.name, image_stream, f'image/{image_type}')
                }

                logger.debug(f"Feishu: Upload request created for {path.name} ({len(image_data)} bytes, type: {image_type})")
                logger.debug(f"Feishu: Request body: {upload_request.request_body}, files: {upload_request.files is not None}")

                # Try direct HTTP request first (SDK seems to have issues)
                logger.info("Feishu: Trying direct HTTP upload first...")
                image_key = await self._upload_image_direct(path, image_data, image_type)

                if image_key:
                    logger.info(f"Feishu: Direct upload successful, image_key: {image_key}")
                else:
                    # Direct HTTP failed, try SDK as fallback
                    logger.warning("Feishu: Direct HTTP upload failed, trying SDK...")
                    upload_response = await self._client.im.v1.image.acreate(upload_request)

                    logger.info(f"Feishu: SDK upload response received, success: {upload_response.success()}")

                    if upload_response.success():
                        image_key = upload_response.data.image_key
                        logger.info(f"Feishu: Image uploaded successfully via SDK, image_key: {image_key}")
                    else:
                        logger.error(f"Feishu: SDK upload also failed: code={upload_response.code}, msg={upload_response.msg}")
                        # As a last resort, send base64 encoded image in a text message
                        logger.warning("Feishu: All upload methods failed, trying base64 encoding...")
                        image_key = await self._send_image_as_base64(path, image_data, image_type, chat_id, receive_id_type)
                        if image_key:
                            logger.info("Feishu: Base64 image sent as text message")
                            return  # Successfully sent, exit the loop
                        else:
                            logger.error("Feishu: All image sending methods failed")
                            continue

                # Send image message using the image_key
                logger.info("Feishu: Sending image message...")
                await self._send_image_message(chat_id, receive_id_type, image_key)

            except Exception as e:
                logger.error(f"Feishu: Error sending media file {media_path}: {e}")
                logger.error(f"Feishu: Exception type: {type(e)}")
                import traceback
                logger.error(f"Feishu: Full traceback: {traceback.format_exc()}")

    async def _upload_image_direct(self, image_path: Path, image_data: bytes, image_type: str) -> str | None:
        """Upload image using direct HTTP request as fallback."""
        try:
            # Use synchronous requests in thread pool for simplicity
            import requests
            from concurrent.futures import ThreadPoolExecutor
            import asyncio

            def upload_sync():
                try:
                    # Get tenant access token using synchronous call
                    try:
                        token_req = self._client.build_tenant_access_token_req()
                        logger.debug("Feishu: Built tenant access token request")

                        token_resp = self._client.request(token_req)  # Synchronous call
                        logger.debug("Feishu: Token request completed")

                        logger.info("Feishu: Token request sent, checking response...")
                        logger.info(f"Feishu: Token response success: {token_resp.success()}")

                        if not token_resp.success():
                            logger.error(f"Feishu: Failed to get tenant access token: code={token_resp.code}, msg={token_resp.msg}")
                            logger.error(f"Feishu: Token response data: {token_resp}")
                            return None

                        token = getattr(token_resp.data, 'tenant_access_token', None)
                        logger.info(f"Feishu: Got tenant access token (length: {len(token) if token else 0})")

                        if not token:
                            logger.error("Feishu: Tenant access token is empty")
                            logger.error(f"Feishu: Token response data attributes: {dir(token_resp.data) if token_resp.data else 'None'}")
                            return None

                    except Exception as token_error:
                        logger.error(f"Feishu: Exception getting tenant token: {token_error}")
                        import traceback
                        logger.error(f"Feishu: Token traceback: {traceback.format_exc()}")
                        return None

                    # Prepare multipart form data using requests
                    # Try different field names that Feishu might expect
                    files = {
                        'image': (image_path.name, image_data, f'image/{image_type}')
                    }

                    # Alternative: try without content_type, let requests auto-detect
                    # files = {
                    #     'image': (image_path.name, image_data)
                    # }

                    logger.debug(f"Feishu: Using multipart files: {list(files.keys())} with content_type: image/{image_type}")

                    # Send request
                    url = "https://open.feishu.cn/open-apis/im/v1/images"
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "User-Agent": "nanobot/1.0"
                    }

                    logger.info(f"Feishu: Sending direct HTTP POST to {url} with file {image_path.name}")
                    logger.debug(f"Feishu: Request headers: {headers}")
                    logger.debug(f"Feishu: Files: {list(files.keys())}")

                    response = requests.post(url, files=files, headers=headers, timeout=30)
                    logger.info(f"Feishu: Direct upload response status: {response.status_code}")
                    logger.debug(f"Feishu: Response headers: {dict(response.headers)}")

                    # Log response content for debugging
                    response_text = response.text
                    logger.debug(f"Feishu: Raw response: {response_text[:500]}...")

                    if response.status_code == 200:
                        try:
                            result = response.json()
                            logger.debug(f"Feishu: Parsed response: {result}")

                            if result.get('code') == 0:
                                image_key = result['data']['image_key']
                                logger.info(f"Feishu: Direct upload successful, image_key: {image_key}")
                                return image_key
                            else:
                                logger.error(f"Feishu: Direct upload API error: code={result.get('code')}, msg={result.get('msg')}")
                        except Exception as json_error:
                            logger.error(f"Feishu: Failed to parse JSON response: {json_error}")
                            logger.error(f"Feishu: Raw response was: {response_text}")
                    else:
                        logger.error(f"Feishu: Direct upload HTTP error {response.status_code}: {response_text}")

                    return None

                except Exception as e:
                    logger.error(f"Feishu: Direct upload sync failed: {e}")
                    import traceback
                    logger.error(f"Feishu: Direct upload sync traceback: {traceback.format_exc()}")
                    return None

            # Run synchronous upload in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, upload_sync)

            return result

        except Exception as e:
            logger.error(f"Feishu: Direct upload failed: {e}")
            import traceback
            logger.error(f"Feishu: Direct upload traceback: {traceback.format_exc()}")
            return None

    async def _send_image_as_base64(self, image_path: Path, image_data: bytes, image_type: str, chat_id: str, receive_id_type: str) -> bool:
        """Send image as base64 encoded text message as last resort."""
        try:
            import base64

            # Encode image data to base64
            base64_data = base64.b64encode(image_data).decode('utf-8')
            logger.info(f"Feishu: Encoded image to base64 (length: {len(base64_data)})")

            # Create a data URL
            data_url = f"data:image/{image_type};base64,{base64_data}"

            # Send as rich text message with image
            post_content = {
                "zh_cn": {
                    "title": f"Image: {image_path.name}",
                    "content": [
                        [{
                            "tag": "img",
                            "image_url": data_url,
                            "alt": {
                                "tag": "plain_text",
                                "content": f"Image: {image_path.name}"
                            }
                        }],
                        [{
                            "tag": "text",
                            "text": f"Image file: {image_path.name} ({len(image_data)} bytes)"
                        }]
                    ]
                }
            }

            content = json.dumps(post_content, ensure_ascii=False)

            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("post")
                    .content(content)
                    .build()
                ).build()

            response = await self._client.im.v1.message.acreate(request)

            if response.success():
                logger.info("Feishu: Base64 image sent successfully as post message")
                return True
            else:
                logger.error(f"Feishu: Failed to send base64 image: code={response.code}, msg={response.msg}")
                return False

        except Exception as e:
            logger.error(f"Feishu: Base64 image sending failed: {e}")
            import traceback
            logger.error(f"Feishu: Base64 traceback: {traceback.format_exc()}")
            return False

    async def _send_image_message(self, chat_id: str, receive_id_type: str, image_key: str) -> None:
        """Send an image message using image_key."""
        logger.info(f"Feishu: Creating image message for chat_id={chat_id}, receive_id_type={receive_id_type}")

        # Try multiple approaches for sending images

        # Approach 1: Using "image" message type with image_key
        logger.info("Feishu: Attempting approach 1 - image message type")
        try:
            image_content = {
                "image_key": image_key
            }
            content = json.dumps(image_content, ensure_ascii=False)
            logger.debug(f"Feishu: Image content JSON: {content}")

            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("image")
                    .content(content)
                    .build()
                ).build()

            response = await self._client.im.v1.message.acreate(request)

            if response.success():
                logger.info("Feishu: Image sent successfully using approach 1")
                return
            else:
                logger.warning(f"Feishu: Approach 1 failed: code={response.code}, msg={response.msg}")

        except Exception as e:
            logger.warning(f"Feishu: Approach 1 exception: {e}")

        # Approach 2: Using "post" message type with image element
        logger.info("Feishu: Attempting approach 2 - post message type with image")
        try:
            post_content = {
                "zh_cn": {
                    "title": "Image",
                    "content": [
                        [{
                            "tag": "img",
                            "image_key": image_key,
                            "width": 400,
                            "height": 300
                        }]
                    ]
                }
            }
            content = json.dumps(post_content, ensure_ascii=False)
            logger.debug(f"Feishu: Post content JSON: {content}")

            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("post")
                    .content(content)
                    .build()
                ).build()

            response = await self._client.im.v1.message.acreate(request)

            if response.success():
                logger.info("Feishu: Image sent successfully using approach 2")
                return
            else:
                logger.warning(f"Feishu: Approach 2 failed: code={response.code}, msg={response.msg}")

        except Exception as e:
            logger.warning(f"Feishu: Approach 2 exception: {e}")

        # Approach 3: Using interactive card with image
        logger.info("Feishu: Attempting approach 3 - interactive card with image")
        try:
            card_content = {
                "config": {"wide_screen_mode": True},
                "elements": [{
                    "tag": "img",
                    "img_key": image_key,
                    "alt": {
                        "tag": "plain_text",
                        "content": "Image"
                    }
                }]
            }
            content = json.dumps(card_content, ensure_ascii=False)
            logger.debug(f"Feishu: Card content JSON: {content}")

            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("interactive")
                    .content(content)
                    .build()
                ).build()

            response = await self._client.im.v1.message.acreate(request)

            if response.success():
                logger.info("Feishu: Image sent successfully using approach 3")
                return
            else:
                logger.warning(f"Feishu: Approach 3 failed: code={response.code}, msg={response.msg}")

        except Exception as e:
            logger.warning(f"Feishu: Approach 3 exception: {e}")

        logger.error("Feishu: All image sending approaches failed")

    async def _send_text_message(self, chat_id: str, receive_id_type: str, content: str) -> None:
        """Send a text message through Feishu."""
        # Build card with markdown + table support
        elements = self._build_card_elements(content)
        card = {
            "config": {"wide_screen_mode": True},
            "elements": elements,
        }
        card_content = json.dumps(card, ensure_ascii=False)

        request = CreateMessageRequest.builder() \
            .receive_id_type(receive_id_type) \
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type("interactive")
                .content(card_content)
                .build()
            ).build()

        response = await self._client.im.v1.message.acreate(request)

        if not response.success():
            logger.error(
                f"Failed to send Feishu text message: code={response.code}, "
                f"msg={response.msg}, log_id={response.get_log_id()}"
            )
        else:
            logger.debug(f"Feishu text message sent to {chat_id}")

    def _get_image_type(self, extension: str) -> str:
        """Get Feishu image type from file extension."""
        type_map = {
            '.jpg': 'jpeg',
            '.jpeg': 'jpeg',
            '.png': 'png',
            '.gif': 'gif',
            '.bmp': 'bmp',
            '.webp': 'webp'
        }
        return type_map.get(extension, 'jpeg')  # Default to jpeg
    
    def _on_message_sync(self, data: "P2ImMessageReceiveV1") -> None:
        """
        Sync handler for incoming messages (called from WebSocket thread).
        Schedules async handling in the main event loop.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)
    
    async def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        """Handle incoming message from Feishu."""
        try:
            event = data.event
            message = event.message
            sender = event.sender
            
            # Deduplication check
            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            
            # Trim cache: keep most recent 500 when exceeds 1000
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)
            
            # Skip bot messages
            sender_type = sender.sender_type
            if sender_type == "bot":
                return
            
            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type  # "p2p" or "group"
            msg_type = message.message_type
            
            # Add reaction to indicate "seen"
            await self._add_reaction(message_id, "THUMBSUP")
            
            # Parse message content
            if msg_type == "text":
                try:
                    content = json.loads(message.content).get("text", "")
                except json.JSONDecodeError:
                    content = message.content or ""
            else:
                content = MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")
            
            if not content:
                return
            
            # Forward to message bus
            reply_to = chat_id if chat_type == "group" else sender_id
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}")
